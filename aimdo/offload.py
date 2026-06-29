#==================================================================================================
#
#   Copyright (C) 2026-2026 turbo-aimdo authors. <https://omega.gg/turbo-aimdo>
#
#   Author: Benjamin Arnaud. <https://bunjee.me> <bunjee@omega.gg>
#
#   This file is part of turbo-aimdo.
#
#   - GNU General Public License Usage:
#   This file may be used under the terms of the GNU General Public License version 3 as published
#   by the Free Software Foundation and appearing in the LICENSE.md file included in the packaging
#   of this file. Please review the following information to ensure the GNU General Public License
#   requirements will be met: https://www.gnu.org/licenses/gpl.html.
#
#==================================================================================================

# =================================================================================================
#  VBAR-residency + fast-DMA file streamer = ComfyUI's DynamicVRAM weight-streaming path, ported to
#  diffusers on top of comfy_aimdo. Each nn.Linear weight is streamed to the GPU per forward
#  instead of kept resident, so a model larger than VRAM (even > VRAM+RAM) runs; spill-safe (a full
#  VBAR fault returns OOM and falls back to a temp buffer, so aimdo never overcommits).
#
#  Design, rationale, and the ComfyUI/comfy-aimdo [CU ...] / [AI ...] line references live in
#  aimdo.md (kept in sync with the pinned upstream commits noted there).
# =================================================================================================
import os, json, struct, torch
from collections import namedtuple

import comfy_aimdo.control as _ctl     # device init / CUDA alloc hooks
import comfy_aimdo.torch as _at        # raw-pointer <-> torch.Tensor bridge
import comfy_aimdo.host_buffer as _hb  # read_file_to_device (fast-DMA)
import comfy_aimdo.vram_buffer as _vb  # reserved (VBAR) GPU cast buffer
from comfy_aimdo.model_vbar import (   # VBAR residency allocator
    ModelVBAR, vbar_fault, vbar_unpin, vbar_signature_compare,
    vbars_reset_watermark_limits,
)


#--------------------------------------------------------------------------------------------------
# Per-generation reclaim
#--------------------------------------------------------------------------------------------------

def reclaim_between_runs(device="cuda:0"):
    # Per-generation housekeeping: return torch's retained allocator pool to the driver and drop
    # the VBARs' watermark floors. The VBAR has its own CUDA VMM separate from torch's pool; if
    # torch's cached blocks stay charged against the VBAR budget it can't stay resident and
    # re-streams every layer (~16 s/step vs ~2). Call once per generation. See aimdo.md.
    torch.cuda.synchronize(torch.device(device))
    torch.cuda.empty_cache()
    try:
        vbars_reset_watermark_limits()
    except Exception:
        import traceback
        print("[aimdo] reclaim_between_runs: vbars_reset_watermark_limits failed:\n"
              + traceback.format_exc(), flush=True)


#--------------------------------------------------------------------------------------------------
# Checkpoint offsets
#--------------------------------------------------------------------------------------------------

_DT = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32, "F64": torch.float64,
       "I64": torch.int64, "I32": torch.int32, "I16": torch.int16, "I8": torch.int8,
       "U8": torch.uint8, "BOOL": torch.bool, "F8_E4M3": torch.float8_e4m3fn}


def _align(n, a=512):
    return (n + a - 1) & ~(a - 1)


# Where a streamed weight lives on disk -- the diffusers analog of ComfyUI's per-tensor file slice.
# See aimdo.md.
_Slice = namedtuple("_Slice", "file file_offset num_bytes dtype shape")


def _offsets(tdir):
    # Build {checkpoint key -> _Slice} from the transformer's .safetensors shard headers.
    idx = os.path.join(tdir, "diffusion_pytorch_model.safetensors.index.json")
    shards = set(json.load(open(idx))["weight_map"].values()) if os.path.exists(idx) \
        else [f for f in os.listdir(tdir) if f.endswith(".safetensors")]
    offsets = {}
    for shard in shards:
        p = os.path.join(tdir, shard)
        with open(p, "rb") as f:
            header_len = struct.unpack("<Q", f.read(8))[0]; hdr = json.loads(f.read(header_len))
        data_start = 8 + header_len
        for k, info in hdr.items():
            if k == "__metadata__":
                continue
            start, end = info["data_offsets"]
            offsets[k] = _Slice(p, data_start + start, end - start,
                                _DT[info["dtype"]], tuple(info["shape"]))
    return offsets


#--------------------------------------------------------------------------------------------------
# Reserved cast buffers
#--------------------------------------------------------------------------------------------------

# Persistent per-device cast buffers for the OFFLOADED path (a weight that did NOT fault into its
# VBAR slot): one copy stream + a reserved aimdo VRAMBuffer carved into two ping-pong views,
# created once per device and reused across builds. Reserved (not torch.empty) so the aimdo
# allocator accounts for it and it never fights cudaMallocAsync for activation VRAM. See aimdo.md.
STREAM_AIMDO_CAST_BUFFERS = {}
# 16 GiB virtual reservation (VBAR address space, committed lazily by .get()).
DEFAULT_AIMDO_CAST_BUFFER_RESERVATION_SIZE = 16 * 1024 ** 3


def get_aimdo_cast_buffer(device_index, buffer_size):
    entry = STREAM_AIMDO_CAST_BUFFERS.get(device_index)
    if entry is None:
        entry = {"stream": torch.cuda.Stream(torch.device("cuda", device_index)),
                 "vram": _vb.VRAMBuffer(DEFAULT_AIMDO_CAST_BUFFER_RESERVATION_SIZE, device_index),
                 "bufs": None, "bsz": -1}
        STREAM_AIMDO_CAST_BUFFERS[device_index] = entry
    if buffer_size > entry["bsz"]:
        # Carve two buffer_size ping-pong regions (offsets 0 and buffer_size); commits up to 2x.
        vram = entry["vram"]; device_str = "cuda:%d" % device_index
        entry["bufs"] = [_at.aimdo_to_tensor(vram.get(buffer_size, 0), device_str),
                         _at.aimdo_to_tensor(vram.get(buffer_size, buffer_size), device_str)]
        entry["bsz"] = buffer_size
    return entry["stream"], entry["bufs"]


#--------------------------------------------------------------------------------------------------
# Offloader
#--------------------------------------------------------------------------------------------------

# Per-Linear streaming state: where the weight lives (file / file_offset / num_bytes / shape /
# dtype), its VBAR slot, residency bookkeeping (signature / gpu), the between-forwards placeholder
# (ph), the pinned host view (host), the on-cast LoRA delta (lora), prefetch fields (ready/ev/dst).
class _StreamedWeight:
    __slots__ = ("file", "file_offset", "num_bytes", "shape", "dtype", "slot", "signature", "gpu",
                 "ph", "lora", "host", "ready", "ev", "dst")


# ComfyUI-style manager for coexisting dynamic (VBAR) models sharing one GPU (e.g. text encoder +
# transformer): keep both loaded, reclaim the inactive one's GPU pages at the active one's load
# boundary (its root forward). Host RAM stays bounded because big weights stream from disk.
# aimdo.md. device_index -> [Offloader], most-recently-active first (mirrors
# current_loaded_models).
_LOADED = {}
_ACTIVE = {}    # device_index -> the Offloader whose pages are currently prioritized + resident


class Offloader:
    #----------------------------------------------------------------------------------------------
    # Initialize
    #----------------------------------------------------------------------------------------------

    def __init__(self, root, tdir=None, device="cuda:0", compute_dtype=torch.bfloat16,
                 lora_files=None, from_module=False, pin_budget=0, manage=False):
        self.device = torch.device(device); self.device_index = self.device.index or 0
        self.compute_dtype = compute_dtype
        self._freed = False

        # tdir: the model's checkpoint dir. REQUIRED -- big weights always stream from disk so an
        # inactive model's host RAM is just OS page cache. See aimdo.md.
        self.tdir = tdir

        # manage: opt into the coexisting-dynamic-models manager so a managed model (e.g. the TE)
        # stays loaded across requests while its GPU footprint is reclaimed during denoise.
        # aimdo.md.
        self.manage = manage

        # Manager state (a LoadedModel-like entry). root = the module whose forward is the load
        # boundary; _staged = streamed modules; _released/_resident_backup track the CPU<->GPU
        # move.
        self.root = root; self._staged = []
        self._released = False; self._resident_backup = None; self._act_handle = None

        # Pinning tier (file path): weights pinned into a HostBuffer up to pin_budget bytes get
        # truly-async H2D; the rest stream file->GPU. Budget measured by the caller (placement).
        self.pin_budget = int(pin_budget)
        self.hb = None; self._registered = []; self.pins = {}

        # Double-buffered prefetch overlap (file path only): MEASURED -- ON when the whole set is
        # pinned (RAM-bound), OFF when some weights stream from disk (disk-bound). Env override
        # SKY_AIMDO_VBAR_PREFETCH=0/1. See aimdo.md.
        self._prefetch_env = os.environ.get("SKY_AIMDO_VBAR_PREFETCH")
        self.prefetch = False  # set after pinning (file path)
        self.order = []; self.pos = {}
        self.offload_stream = None; self.cast_buffers = None
        if not _ctl.devctxs:
            _ctl.init_device(self.device_index)

        # LoRA as on-cast weight patches (not PEFT adapters): keep the small up/down factors
        # resident and add (up@down)*scale to the base weight right after it streams in (see _pre).
        # aimdo.md.
        self._lora = self._load_lora(lora_files) if lora_files else {}

        # from_module: root is a LIVE module (e.g. a diffusers/transformers text encoder that
        # rewrote the checkpoint key names). Stream its big Linears from disk by matching live
        # names to disk keys (_match_disk_keys); keep only small params resident. See aimdo.md.
        if from_module:
            self._init_from_module(root)
            return

        offsets = _offsets(tdir)

        # one open handle per shard, kept for the per-forward fast-DMA reads (closed in free()).
        self.files = {p: open(p, "rb") for p, *_ in {(v.file,) for v in offsets.values()}}

        # Collect the nn.Linear weights to stream.
        linears = {}; skipped = 0; tiny = 0
        for name, m in root.named_modules():
            if isinstance(m, torch.nn.Linear):
                # PEFT wraps a target Linear as `<name>.base_layer`; strip it so the streamed BASE
                # weight matches the checkpoint. Adapter Linears are absent -> skipped (kept
                # resident; the adapter adds its delta). See aimdo.md.
                base_name = name[:-len(".base_layer")] if name.endswith(".base_layer") else name
                weight_key = base_name + ".weight"
                if weight_key not in offsets:
                    skipped += 1; continue  # lora adapter / tied / unsaved

                # Tiny weights (<=16 KiB) stay resident -- mixing tiny + giant streamed weights
                # stalls stream-buffer rotations (ComfyUI force-loads them). See aimdo.md.
                if offsets[weight_key].num_bytes <= 16 * 1024:
                    tiny += 1; continue
                linears[m] = weight_key

        if skipped:
            print("[aimdo] skipped %d Linear(s) absent from checkpoint" % skipped, flush=True)
        if tiny:
            print("[aimdo] %d tiny Linear(s) <=16KiB kept resident (not streamed)"
                  % tiny, flush=True)

        # dtype guard (DIFFERENCE #3): we stream + use weights in their STORED dtype (== compute,
        # bf16 here). Warn loudly if a model mixes dtypes. See aimdo.md.
        bad = {offsets[weight_key].dtype for weight_key in linears.values()
               if offsets[weight_key].dtype != self.compute_dtype}
        if bad:
            print("[aimdo] WARNING: streamed weight dtype(s) %s != compute_dtype %s; NOT cast "
                  "(DIFFERENCE #3; ComfyUI casts in [CU model_management.py cast_to L1453])"
                  % (sorted(map(str, bad)), self.compute_dtype), flush=True)

        # Materialise everything that is NOT a streamed weight (norms, embeddings, biases) resident
        # on GPU; only big weights stream. assign=True installs without cloning.
        from safetensors import safe_open
        streamed = set(linears.values())
        state_dict = {}
        for p in self.files:
            with safe_open(p, framework="pt") as sf:
                for k in sf.keys():
                    if k not in streamed:
                        state_dict[k] = sf.get_tensor(k).to(self.device)
        root.load_state_dict(state_dict, strict=False, assign=True)

        # One VBAR backs every streamed weight; pages committed only on fault().
        total = sum(_align(offsets[weight_key].num_bytes) for weight_key in linears.values())
        self.vbar = ModelVBAR(int(total) + (64 << 20), device=self.device_index)

        # Mark this VBAR's pages high-priority so aimdo keeps as many of our weights resident as
        # fit.
        self.vbar.prioritize()

        # Pin the streamed set into the HostBuffer only if the WHOLE set fits the RAM budget;
        # partial-pinning a >RAM model exhausts host-registration and OOMs, so >RAM streams
        # pageable from the page cache. See aimdo.md.
        if 0 < total <= self.pin_budget:
            self._pin_memory(linears, offsets)

        # Prefetch ON iff every streamed weight is pinned (RAM-bound), else OFF. Env forces it.
        all_pinned = len(linears) > 0 and len(self.pins) == len(linears)
        self.prefetch = ((self._prefetch_env == "1") if self._prefetch_env is not None
                         else all_pinned)
        print("[aimdo] prefetch=%s (all_pinned=%s, env=%s)"
              % (self.prefetch, all_pinned, self._prefetch_env), flush=True)

        # Cast buffers for the OFFLOADED case (weight didn't fault into its slot): two ping-pong
        # views from the reserved VRAMBuffer. The faulted case reads straight into the VBAR slot.
        largest = max(_align(offsets[weight_key].num_bytes) for weight_key in linears.values())
        buffer_size = _align(largest) + 512

        # offload_stream is used only when prefetching; the sync path uses cast_buffer (= bufs[0]).
        self.offload_stream, self.cast_buffers = get_aimdo_cast_buffer(
            self.device_index, buffer_size)
        self.cast_buffer = self.cast_buffers[0]
        if not self.prefetch:
            self.offload_stream = None

        for m, weight_key in linears.items():
            self._stage(m, weight_key, offsets)

        if self.manage:
            self._register_manager()

    def _load_lora(self, specs):
        # Parse kohya/diffusers LoRA files -> {base_weight_key: [(up, down, scale), ...]}. Stacked
        # LoRAs accumulate; scale = (alpha/rank) * per-file weight. See aimdo.md.
        from safetensors import safe_open
        lora = {}
        sufs = ((".lora_down.weight", ".lora_up.weight"), (".lora_A.weight", ".lora_B.weight"))
        for spec in (specs if isinstance(specs, (list, tuple)) else [specs]):
            path, file_weight = spec if isinstance(spec, (list, tuple)) else (spec, 1.0)
            with safe_open(path, framework="pt") as sf:
                keys = set(sf.keys())
                prefixes = {k[:-len(ds)] for k in keys for ds, _us in sufs if k.endswith(ds)}
                for prefix in prefixes:
                    ds, us = next((d, u) for d, u in sufs if prefix + d in keys)

                    # [rank, in]
                    down = sf.get_tensor(prefix + ds).to(self.device, self.compute_dtype)

                    # [out, rank]
                    up = sf.get_tensor(prefix + us).to(self.device, self.compute_dtype)
                    rank = down.shape[0]
                    scale = ((sf.get_tensor(prefix + ".alpha").item() / rank)
                             if (prefix + ".alpha") in keys else 1.0) * file_weight
                    lora.setdefault(prefix + ".weight", []).append((up, down, float(scale)))

        print("[aimdo] loaded LoRA over %d target(s) from %d file(s)"
              % (len(lora),
                 len(specs if isinstance(specs, (list, tuple)) else [specs])), flush=True)
        return lora

    def _discard_cuda_async_error(self):
        # Drain a sticky async CUDA error (e.g. a failed cudaHostRegister) so it doesn't resurface.
        try:
            a = torch.ones(1, dtype=torch.uint8, device=self.device); _ = a + a
            torch.cuda.synchronize(self.device)
        except RuntimeError:
            pass

    def _pin_memory(self, linears, offsets):
        # Pin the streamed weights into a HostBuffer up to pin_budget bytes for truly-async H2D:
        # read each file slice in, cudaHostRegister it, keep the pinned view as the H2D source.
        # In-order up to the budget. See aimdo.md.
        used = 0; pinned = []
        for m, weight_key in linears.items():
            a = _align(offsets[weight_key].num_bytes)
            if used + a > self.pin_budget:
                continue  # past budget -> this weight streams from file
            pinned.append(weight_key); used += a
        if not pinned:
            return

        # HostBuffer sized to the pinned set (+ headroom).
        self.hb = _hb.HostBuffer(0, 64 * 1024 * 1024, used + (64 << 20))
        layout = {}
        for weight_key in pinned:
            p, file_offset, num_bytes, dtype, shape = offsets[weight_key]
            offset = self.hb.size
            self.hb.extend(_align(num_bytes), register=False)

            # file -> HostBuffer once (host-only)
            self.hb.read_file_slice(self.files[p], file_offset, num_bytes, offset=offset)
            layout[weight_key] = (offset, num_bytes, dtype, shape)

        host = _at.hostbuf_to_tensor(self.hb)  # uint8 view over the staged buffer
        base = host.data_ptr(); cudart = torch.cuda.cudart(); ok = 0
        for weight_key, (offset, num_bytes, dtype, shape) in layout.items():
            # cudaHostRegister the exact region so torch sees a pinned (async-copy) H2D source.
            if int(cudart.cudaHostRegister(base + offset, num_bytes, 0)) == 0:
                self._registered.append(base + offset); ok += num_bytes
            else:
                self._discard_cuda_async_error()
            self.pins[weight_key] = host[offset:offset + num_bytes].view(dtype).view(shape)

        print("[aimdo] pinned %d/%d streamed weight(s) = %.2f GB (budget %.2f GB)"
              % (len(self.pins), len(linears), ok / 1024 ** 3,
                 self.pin_budget / 1024 ** 3), flush=True)

    def _stage(self, m, weight_key, offsets):
        p, file_offset, num_bytes, dtype, shape = offsets[weight_key]
        state = _StreamedWeight()
        state.file = self.files[p]; state.file_offset = file_offset; state.num_bytes = num_bytes
        state.shape = shape; state.dtype = dtype

        # pinned view if pinned, else None -> file-stream
        state.host = self.pins.get(weight_key)

        # VBAR slot for this weight
        state.slot = self.vbar.alloc(_align(num_bytes))
        state.signature = None; state.gpu = None

        # prefetch bookkeeping (unused on the sync path)
        state.ready = False; state.ev = None; state.dst = None
        state.lora = self._lora.get(weight_key)  # (up, down, scale) on-cast patch, or None

        # dtype placeholder between forwards
        state.ph = torch.empty(0, dtype=dtype, device=self.device)

        del m._parameters["weight"]; setattr(m, "weight", state.ph)
        m._aimdo = state; self._staged.append(m)
        m.register_forward_pre_hook(self._pre)
        m.register_forward_hook(self._post)

    def _init_from_module(self, root):
        # root is a LIVE module (not a meta model): materialise its small NON-streamed params on
        # GPU and STREAM its big Linears from disk (live weights dropped -> host RAM = page cache).
        # See aimdo.md.
        if self.tdir is None:
            raise ValueError("Offloader(from_module=True) requires tdir (the model's "
                             "checkpoint dir) so big weights stream from disk instead of being "
                             "held in RAM.")
        linears = []
        for name, m in root.named_modules():
            # Skip lm_head: its weight is tied to the embedding (shared storage) -> follows it to
            # GPU resident; never streamed.
            if isinstance(m, torch.nn.Linear) and "lm_head" not in name \
                    and m.weight is not None and m.weight.device.type == "cpu":
                linears.append((name, m))

        # Map each live Linear -> its disk key (model-agnostic) before materialising residents.
        offsets = _offsets(self.tdir)
        pairs, missing, bad_dt = self._match_disk_keys(linears, offsets)
        if missing:
            raise RuntimeError("[aimdo] from_module: %d Linear(s) did not map to a unique "
                               "checkpoint key (e.g. %r). The model<->checkpoint name mapping "
                               "is ambiguous." % (len(missing), missing[0]))
        if bad_dt:
            print("[aimdo] WARNING: streamed dtype(s) %s != compute_dtype %s; NOT cast "
                  "[CU model_management.py cast_to L1453]"
                  % (sorted(bad_dt), self.compute_dtype), flush=True)

        # Resident: every param/buffer that is NOT a streamed weight -> GPU (incl. the tied
        # embedding). Done before staging so the embedding lands first.
        streamed_ids = {id(m.weight) for m, _ in pairs}
        for p in root.parameters(recurse=True):
            if id(p) not in streamed_ids and p.device.type == "cpu":
                p.data = p.data.to(self.device)
        for b in root.buffers(recurse=True):
            if b.device.type == "cpu":
                b.data = b.data.to(self.device)

        total = sum(_align(offsets[weight_key].num_bytes) for _, weight_key in pairs)
        self.vbar = ModelVBAR(int(total) + (64 << 20), device=self.device_index)
        self.vbar.prioritize()  # high-priority VRAM retention

        # Shared reserved cast buffer for OOM'd layers; no pinning (stream from page cache). Runs
        # once per request, so no prefetch.
        largest = max(_align(offsets[weight_key].num_bytes) for _, weight_key in pairs)
        self.offload_stream, self.cast_buffers = get_aimdo_cast_buffer(
            self.device_index, _align(largest) + 512)
        self.cast_buffer = self.cast_buffers[0]; self.offload_stream = None
        self.pins = {}  # no pinned host weights -> stream from file
        self.files = {p: open(p, "rb")
                      for p, *_ in {(offsets[weight_key].file,) for _, weight_key in pairs}}

        for m, weight_key in pairs:
            # frees the live CPU weight (del m._parameters["weight"])
            self._stage(m, weight_key, offsets)

        import gc as _gc; _gc.collect()              # reclaim the freed live Linear weights now
        print("[aimdo] from_module: streaming %d Linear(s) from disk = %.2f GB "
              "(host RAM = page cache only)"
              % (len(pairs), total / 1024 ** 3), flush=True)

        if self.manage:
            self._register_manager()

    def _match_disk_keys(self, linears, offsets):
        # Map each live streamed Linear -> its disk key WITHOUT per-model hardcoding: an upstream
        # loader rewrites the NAME but not the tensor, so a live name and its disk key share a
        # dotted SUFFIX and (shape, dtype). Pick the longest common segment-suffix; require it
        # unique. Returns (pairs[(module, disk_key)], missing[names], bad_dtypes). See aimdo.md.
        by_kind = {}
        for k, spec in offsets.items():
            by_kind.setdefault((tuple(spec.shape), spec.dtype), []).append(k.split("."))

        pairs = []; missing = []; bad_dt = set()
        for name, m in linears:
            shape = tuple(m.weight.shape); dtype = m.weight.dtype
            live_key = (name + ".weight").split(".")
            best = None; best_n = -1; tie = False
            for disk_key in by_kind.get((shape, dtype), ()):
                n = 0
                while (n < len(disk_key) and n < len(live_key)
                       and disk_key[-1 - n] == live_key[-1 - n]):
                    n += 1
                if n > best_n:
                    best, best_n, tie = disk_key, n, False
                elif n == best_n:
                    tie = True
            if best is None or best_n == 0 or tie:
                missing.append(name); continue
            weight_key = ".".join(best)
            if dtype != self.compute_dtype:
                bad_dt.add(str(dtype))
            pairs.append((m, weight_key))
        return pairs, missing, bad_dt

    #----------------------------------------------------------------------------------------------
    # Dynamic-model manager (load boundary + release / reload)
    #----------------------------------------------------------------------------------------------

    def _register_manager(self):
        # Join the registry + hook the root forward as the load boundary (the analog of
        # load_models_gpu(model) running before the model executes).
        _LOADED.setdefault(self.device_index, []).append(self)
        self._act_handle = self.root.register_forward_pre_hook(self._activate_hook)

    def _activate_hook(self, m, args):
        # Fires before every root forward; cheap re-entry guard so per-step calls are ~free.
        if _ACTIVE.get(self.device_index) is self:
            return
        self._activate()

    def activate(self):
        # Public load-boundary trigger: call BEFORE the framework reads device placement (diffusers
        # reads _execution_device at pipe() start), so a managed model is GPU-resident in time.
        # No-op if unmanaged. See aimdo.md.
        if self.manage:
            self._activate()

    def _activate(self):
        # Become active: reload if released, prioritize our pages, release the OTHER dynamic models
        # on this device (explicit reclaim at the load boundary). See aimdo.md.
        if self._released:
            self.restore_loaded_backups()
        loaded = _LOADED.setdefault(self.device_index, [])
        if self in loaded:
            loaded.remove(self)
        loaded.insert(0, self)  # most-recently-active first
        _ACTIVE[self.device_index] = self

        self.vbar.prioritize()
        for other in loaded[1:]:  # release the OTHER dynamic models
            other.partially_unload()

    def partially_unload(self):
        # Reclaim this model's GPU footprint, reloadably: decommit its VBAR pages and move its
        # resident (non-streamed) params/buffers back to CPU (recorded for restore_loaded_backups).
        # See aimdo.md.
        if self._released:
            return
        try:
            if getattr(self, "offload_stream", None) is not None:
                self.offload_stream.synchronize()

            # no in-flight reads into the slots we are about to free
            torch.cuda.synchronize(self.device)
        except Exception:
            pass
        free0 = torch.cuda.mem_get_info(self.device)[0]               # free VRAM before reclaim
        vbar_freed = int(self.vbar.free_memory(1 << 62))  # decommit ALL pages; returns bytes freed
        self.vbar.deprioritize()  # release retention priority

        # Force a clean re-read on the next fault (a stale gpu/signature would falsely "match").
        for m in self._staged:
            state = m._aimdo; state.gpu = None; state.signature = None
            state.ready = False; state.dst = None; state.ev = None

        # Resident params/buffers -> CPU. parameters() dedups (tied embed/lm_head moves once); the
        # streamed weights are placeholders, not Parameters, so they are skipped.
        backups = []; res_bytes = 0
        for p in self.root.parameters(recurse=True):
            if p.device.type == "cuda":
                res_bytes += p.numel() * p.element_size(); backups.append(p)
                p.data = p.data.to("cpu")
        for b in self.root.buffers(recurse=True):
            if b.device.type == "cuda":
                res_bytes += b.numel() * b.element_size(); backups.append(b)
                b.data = b.data.to("cpu")
        self._resident_backup = backups

        # The bounce buffer is the shared reserved cast buffer (persistent across swaps) -> leave
        # it; only the streamed VBAR pages + resident params were ours to free.
        self._released = True

        # Return the freed VRAM to the allocator so the newly-active model's VBAR can commit it
        # resident (else it stays reserved and the active model re-streams every layer). See
        # aimdo.md.
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        free1 = torch.cuda.mem_get_info(self.device)[0]               # free VRAM after reclaim
        GB = 1024 ** 3
        print("[aimdo] partially_unload: VBAR freed %.3f GB + %d resident tensor(s) "
              "%.3f GB -> CPU; free VRAM %.2f -> %.2f GB (+%.2f)"
              % (vbar_freed / GB, len(backups), res_bytes / GB, free0 / GB, free1 / GB,
                 (free1 - free0) / GB), flush=True)

    def restore_loaded_backups(self):
        # Inverse of partially_unload: resident params/buffers back to GPU, reprioritize. Streamed
        # weights re-fault on the next forward (their slots were decommitted).
        if not self._released:
            return
        for t in self._resident_backup or []:
            t.data = t.data.to(self.device)
        self._resident_backup = None
        self.vbar.prioritize()
        self._released = False
        print("[aimdo] restore_loaded_backups: resident tensors -> GPU", flush=True)

    #----------------------------------------------------------------------------------------------
    # Per-forward streaming
    #----------------------------------------------------------------------------------------------

    def _pre(self, m, args):
        # Per-forward (synchronous path): fault, then read the weight into its slot (or the reused
        # temp buffer if offloaded), via the shared _do_read. See aimdo.md.
        state = m._aimdo
        if self.prefetch:
            return self._pre_prefetch(m, state)
        signature = vbar_fault(state.slot)
        if (signature is not None and state.gpu is not None
                and vbar_signature_compare(signature, state.signature)):
            m.weight = state.gpu  # resident: reuse, NO read
            return

        # destination: resident VBAR slot if faulted in, else the reused temp buffer (offloaded)
        if signature is not None:
            weight = (_at.aimdo_to_tensor(state.slot, self.device)[:state.num_bytes]
                      .view(state.dtype).view(state.shape))
        else:
            weight = self.cast_buffer[:state.num_bytes].view(state.dtype).view(state.shape)
        self._do_read(state, weight, None)   # H2D (+ on-cast LoRA) on the compute stream

        state.gpu = weight; state.signature = signature
        m.weight = weight

    def _pre_prefetch(self, m, state):
        # Provide this layer's weight (faulting inline if not prefetched), then kick the NEXT
        # layer's read on the copy stream to overlap compute. Order is learned on step 1. See
        # aimdo.md.
        i = self.pos.get(id(m))
        if i is None:
            i = len(self.order); self.pos[id(m)] = i; self.order.append(m)
        if not state.ready:
            self._fetch(state, copy=False, bufidx=i & 1)  # not prefetched -> compute stream
        if state.ev is not None:
            # compute waits for the copy
            torch.cuda.current_stream(self.device).wait_event(state.ev)
        m.weight = state.dst
        if i + 1 < len(self.order):  # prefetch next into the OTHER buf
            nxt = self.order[i + 1]._aimdo
            if not nxt.ready:
                self._fetch(nxt, copy=True, bufidx=(i + 1) & 1)

    def _fetch(self, state, copy, bufidx):
        # Fault (pins the slot) + read this layer into its VBAR slot (faulted/resident) or a
        # ping-pong temp buffer (offloaded). copy=True runs on the copy stream and records
        # state.ev. aimdo.md.
        signature = vbar_fault(state.slot)
        if (signature is not None and state.gpu is not None
                and vbar_signature_compare(signature, state.signature)):
            state.dst = state.gpu; state.signature = signature  # resident: reuse, NO read
            state.ev = None; state.ready = True
            return
        if signature is not None:
            weight = (_at.aimdo_to_tensor(state.slot, self.device)[:state.num_bytes]
                      .view(state.dtype).view(state.shape))
        else:
            weight = (self.cast_buffers[bufidx][:state.num_bytes]
                      .view(state.dtype).view(state.shape))
        offload_stream = self.offload_stream if copy else None
        if offload_stream is not None:
            # don't overwrite a ping-pong buffer until the compute that last read it has finished
            offload_stream.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(offload_stream):
                self._do_read(state, weight, offload_stream)
                state.ev = offload_stream.record_event()  # consumer waits on this
        else:
            self._do_read(state, weight, None)
            state.ev = None

        state.gpu = weight; state.signature = signature; state.dst = weight; state.ready = True

    def _do_read(self, state, weight, offload_stream):
        # H2D for one layer into `weight` on `offload_stream` (None = compute stream). Shared by
        # the prefetch and not-yet-prefetched paths. See aimdo.md.
        strm = int((offload_stream or torch.cuda.current_stream(self.device)).cuda_stream)
        if state.host is not None:
            # pinned HostBuffer view -> async H2D
            weight.copy_(state.host, non_blocking=True)
        else:
            _hb.read_file_to_device(state.file, state.file_offset, state.num_bytes,
                                    strm, weight.data_ptr(), self.device_index)
        if state.lora is not None:
            for up, down, scale in state.lora:
                weight.addmm_(up, down, alpha=scale)

    def _post(self, m, args, output):
        state = m._aimdo
        if state.signature is not None:
            vbar_unpin(state.slot)
        else:
            state.gpu = None  # temp reused next layer
        if self.prefetch:
            # clear this layer's prefetch state so it re-faults next step
            state.ready = False; state.dst = None; state.ev = None
        m.weight = state.ph
        return output

    #----------------------------------------------------------------------------------------------
    # Teardown
    #----------------------------------------------------------------------------------------------

    def free(self):
        if self._freed:
            return
        self._freed = True

        # Leave the manager first so a teardown can't re-trigger activation or be released by a
        # peer.
        try:
            if self._act_handle is not None:
                self._act_handle.remove()
        except Exception:
            pass
        loaded = _LOADED.get(self.device_index)
        if loaded and self in loaded:
            loaded.remove(self)
        if _ACTIVE.get(self.device_index) is self:
            _ACTIVE.pop(self.device_index, None)

        # If manager-released, restore first so teardown frees the VBAR in its normal populated
        # state (freeing a free_memory()'d VBAR then loading the next pipe was observed to
        # segfault).
        if self._released:
            try:
                self.restore_loaded_backups()
            except Exception:
                pass
        try:
            # quiesce the copy stream first (in-flight prefetch reads); freeing under a copy is
            # UAF.
            if self.offload_stream is not None:
                self.offload_stream.synchronize()
            torch.cuda.synchronize(self.device)
        except Exception:
            pass

        # cudaHostUnregister every pinned region BEFORE freeing the HostBuffer, else the next
        # HostBuffer (often the same host addresses) fails with "already mapped".
        cudart = torch.cuda.cudart()
        for ptr in getattr(self, "_registered", []):
            if int(cudart.cudaHostUnregister(ptr)) != 0:
                self._discard_cuda_async_error()
        self._registered = []
        hb = getattr(self, "hb", None)
        if hb is not None:
            try:
                hb.truncate(0, do_unregister=False)   # decommit without re-unregistering base
                hb.__del__()  # blocks on the async decommit drain (rebuild-safe)
            except Exception:
                pass
            self.hb = None
        self.pins = {}
        self.vbar = None        # ModelVBAR.__del__ -> vbar_free

        # Drop refs to the cast buffers/stream but do NOT free them: the reserved VRAMBuffer + copy
        # stream are persistent (STREAM_AIMDO_CAST_BUFFERS) and reused across model swaps.
        self.cast_buffer = None; self.cast_buffers = None; self.offload_stream = None
        self.order = []; self.pos = {}
        for f in getattr(self, "files", {}).values():
            try:
                f.close()
            except Exception:
                pass
        self.files = {}
