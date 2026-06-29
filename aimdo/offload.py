#==================================================================================================
#
#   Copyright (C) 2026-2026 turbo-comfy authors. <https://omega.gg/turbo-comfy>
#
#   Author: Benjamin Arnaud. <https://bunjee.me> <bunjee@omega.gg>
#
#   This file is part of turbo-comfy.
#
#   - GNU General Public License Usage:
#   This file may be used under the terms of the GNU General Public License version 3 as published
#   by the Free Software Foundation and appearing in the LICENSE.md file included in the packaging
#   of this file. Please review the following information to ensure the GNU General Public License
#   requirements will be met: https://www.gnu.org/licenses/gpl.html.
#
#==================================================================================================

# =================================================================================================
#  VBAR-residency + fast-DMA file streamer = ComfyUI's *DynamicVRAM* path, for diffusers.
#
#  Derived from ComfyUI's LOWVRAM path. Two changes turn that streamer into ComfyUI's DynamicVRAM
#  behaviour, which runs the 39 GB bf16 qwen-image transformer at ~25 s/step even when the model is
#  bigger than BOTH VRAM and RAM -- see aimdo.md, PLAN-bf16-vbar.md:
#
#    1. HOST SOURCE = fast-DMA file reader, not a staged HostBuffer. Each weight is read
#       straight from
#       its .safetensors shard into the GPU via comfy_aimdo.host_buffer.read_file_to_device -- the
#       native fast-DMA path ComfyUI uses (its read_tensor_file_slice_into
#       [CU memory_management.py L18] wraps the same primitive). The OS page cache holds the hot
#       set; the `mark_cold` flag is comfy-aimdo's RAM-pressure cache, which manages eviction (the
#       `Using RAM pressure cache` line). No 39 GB copy into RAM, and -- crucially -- NOT torch's
#       pageable `copy_` (that path measured ~128 s/step here, GPU 7 % util, transfer-starved:
#       exactly §9's "unregistered copies are synchronous and slow"). On this box cudaHostRegister
#       fails (§9), so the native reader's fast-DMA is the only way to get a fast H2D without a
#       full in-RAM HostBuffer.
#    2. GPU RESIDENCY via a VBAR. Each weight gets a VBAR slot; per forward vbar_fault() decides:
#       resident (signature unchanged) -> reuse, NO read; faulted in (VRAM free) -> read
#       file->slot; offloaded (VRAM full) -> read file->temp tensor. unpin() after lets aimdo evict
#       under pressure. This is ComfyUI's `_v` branch
#       [CU ops.py L128-L141 fault/resident, L392 unpin] -- the residency loop the plain LOWVRAM
#       streamer omits; the mechanism is the existing aimdo_flux2.py port.
#
#  Spill-safe where the LOWVRAM streamer's resident_gb wasn't (aimdo.md §9): vbar_fault() returns
#  OOM when VRAM is full -> we read into a temp tensor -> aimdo NEVER overcommits -> no WDDM
#  VRAM->RAM spill.
#
#  ------------------------------------------------------------------------------------------------
#  Pinned reference commits for the [CU ...] / [AI ...] tags (re-verify + bump when updating):
#    ComfyUI      C:\dev\test\ComfyUI       @ 5955ddff52a2eda2ba0cf7f3fb0927c93fb2fbb8
#    comfy-aimdo  C:\dev\test\comfy-aimdo   @ ace72abefa1ede12a4b8a4e2c99919804e5f38e0
#
#  SCOPE (gate build): synchronous path (fault -> read -> use -> unpin) on the default stream, no
#  prefetch overlap. The double-buffer overlap is the Phase-2 add. The transformer has NO
#  checkpoint key-remap (only the Qwen2.5-VL TE does); the TE is handled by the runner, not here.
# =================================================================================================
import os, json, struct, torch

import comfy_aimdo.control as _ctl     # [AI control.py] device init / CUDA alloc hooks
import comfy_aimdo.torch as _at        # [AI torch.py] raw-pointer <-> torch.Tensor bridge
import comfy_aimdo.host_buffer as _hb  # [AI host_buffer.py] read_file_to_device (fast-DMA)
import comfy_aimdo.vram_buffer as _vb  # [AI vram_buffer.py] reserved (VBAR) GPU cast buffer
from comfy_aimdo.model_vbar import (   # [AI model_vbar.py] VBAR residency allocator
    ModelVBAR, vbar_fault, vbar_unpin, vbar_signature_compare,
    vbars_reset_watermark_limits,  # [AI model_vbar.py L149] drop per-VBAR resident floors
)


def reclaim_between_runs(device="cuda:0"):
    # Per-generation aimdo housekeeping -- the faithful port of ComfyUI's per-execution `finally`
    # [CU execution.py L543-549], which runs on EVERY node when aimdo_enabled. Call once per
    # generation (server run_job's post-pipe finally), for any aimdo pipe. Two reclaims, in order:
    #
    # 1. Return torch's retained allocator pool to the driver == reset_cast_buffers() ->
    #    soft_empty_cache()
    #    [CU model_management.py L1383 -> L1950-1966]. The VBAR maps physical VRAM through its OWN
    #    CUDA VMM (cuMemCreate/cuMemMap per page) [AI plat.h three_stooges L182-220], SEPARATE from
    #    torch's caching / cudaMallocAsync pool, so the two compete for the same physical VRAM with
    #    nothing to arbitrate [AI README.md L54]. Between generations torch keeps the prior run's
    #    freed activation blocks cached in its pool; aimdo's alloc hook accounts that retained VRAM
    #    against the VBAR budget [AI pyt-cu-plug-alloc-async.c L166], so a persistent VBAR can't
    #    stay resident and re-streams every layer (measured: ~16 s/step vs ~2 once the pool is
    #    returned).
    # 2. Drop every VBAR's protected-resident floor (watermark_limit -> 0) ==
    #    vbars_reset_watermark_limits()
    #    [CU execution.py L549] -> [AI model_vbar.py L149, model-vbar.c L285-291]. A no-op while we
    #    never call set_watermark_limit (we juggle residency via prioritize/deprioritize +
    #    free_memory instead), ported for fidelity so a future watermark-floor use can't leak
    #    protection across generations.
    # == soft_empty_cache's CUDA path [CU model_management.py L1964-1965]: synchronize ->
    # empty_cache. We omit its third call, ipc_collect [CU L1966] -- that only reclaims CUDA memory
    # shared cross-process via IPC handles, of which a single-process inference server has none, so
    # it is a pure no-op here.
    torch.cuda.synchronize(torch.device(device))
    torch.cuda.empty_cache()
    try:
        vbars_reset_watermark_limits()
    except Exception:
        import traceback
        print("[aimdo] reclaim_between_runs: vbars_reset_watermark_limits failed:\n"
              + traceback.format_exc(), flush=True)

_DT = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32, "F64": torch.float64,
       "I64": torch.int64, "I32": torch.int32, "I16": torch.int16, "I8": torch.int8,
       "U8": torch.uint8, "BOOL": torch.bool, "F8_E4M3": torch.float8_e4m3fn}


def _align(n, a=512):
    return (n + a - 1) & ~(a - 1)


def _offsets(tdir):
    # {key: (file_path, abs_byte_offset, byte_len, dtype, shape)} -- the data behind ComfyUI's
    # `_comfy_tensor_file_slice` (file_ref/offset/size) [CU memory_management.py L36-L52].
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
            offsets[k] = (p, data_start + start, end - start,
                          _DT[info["dtype"]], tuple(info["shape"]))
    return offsets


# Persistent per-device cast buffers for the OFFLOADED ping-pong path (a weight that did NOT fault
# into its VBAR slot). One copy stream + a single RESERVED aimdo VRAMBuffer carved into two views,
# created ONCE per device and reused across every model build (the VRAMBuffer grows to fit the
# largest layer seen). This is ComfyUI's STREAM_AIMDO_CAST_BUFFERS
# [CU model_management.py get_aimdo_cast_buffer L1343]; its bounce tensor is
# aimdo_to_tensor(vrambuf.get(size, offset), device) [CU ops.py get_cast_buffer L124]. Using a
# RESERVED buffer (not torch.empty) is the point: the aimdo allocator accounts for it so it never
# fights cudaMallocAsync for activation VRAM -- the per-layer-temp-alloc cost ComfyUI avoids
# (aimdo.md s5).
STREAM_AIMDO_CAST_BUFFERS = {}
# 16 GiB virtual reservation, matching ComfyUI [CU model_management.py L1309]. VBAR address space,
# committed lazily by .get(), so this is cheap even on a low-VRAM card.
DEFAULT_AIMDO_CAST_BUFFER_RESERVATION_SIZE = 16 * 1024 ** 3


def get_aimdo_cast_buffer(device_index, buffer_size):
    entry = STREAM_AIMDO_CAST_BUFFERS.get(device_index)
    if entry is None:
        entry = {"stream": torch.cuda.Stream(torch.device("cuda", device_index)),
                 # [AI vram_buffer.py]
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


# Per-Linear streaming state. file/file_offset/num_bytes = where the weight lives on disk (file
# mode); host = the
# live CPU weight tensor (from_module mode); slot = the VBAR allocation; signature/gpu = residency
# bookkeeping; ph = the dtype placeholder between forwards; lora = (up, down, scale) GPU tensors
# for the on-cast LoRA delta, or None.
class _StreamedWeight:
    __slots__ = ("file", "file_offset", "num_bytes", "shape", "dtype", "slot", "signature", "gpu",
                 "ph", "lora", "host", "ready", "ev", "dst")


# ==== ComfyUI-style dynamic-model manager (mirrors current_loaded_models +
# load_models_gpu/free_memory) ==== Two coexisting dynamic (VBAR) models share one GPU (e.g. a text
# encoder + a diffusion transformer). ComfyUI keeps both loaded and reclaims GPU pages at load
# boundaries: load_models_gpu(model) [CU model_management.py L849] -> free_memory -> the inactive
# model's partially_unload -> vbar.free_memory [CU model_patcher.py L1937-1938] +
# restore_loaded_backups (resident weights -> off-GPU) [CU model_patcher.py L1768, L1941]; host RAM
# is bounded because every weight streams from its .safetensors file (_comfy_tensor_file_slice
# [CU memory_management.py L36-L75]) rather than being held in RAM, and inactive pins are released
# (partially_unload_ram [CU model_patcher.py L1976]). We mirror both halves: the manager does the
# GPU reclaim at each model's root forward (the load boundary), and every offloader streams its big
# weights from disk (file path + from_module-file below), so an inactive model's host cost is only
# OS page cache. GPU-agnostic: all sizes are measured (placement.py / mem_get_info); nothing
# hardcodes VRAM/RAM.
# dev -> [Offloader] most-recently-active first (== current_loaded_models [CU L805/945])
_LOADED = {}
_ACTIVE = {}    # dev -> the Offloader whose pages are currently prioritized + resident


class Offloader:
    def __init__(self, root, tdir=None, device="cuda:0", compute_dtype=torch.bfloat16,
                 lora_files=None, from_module=False, pin_budget=0, manage=False):
        self.device = torch.device(device); self.device_index = self.device.index or 0
        self.compute_dtype = compute_dtype
        self.from_module = from_module
        self._freed = False
        # tdir: the model's checkpoint dir (its .safetensors live here). REQUIRED -- big weights
        # always stream from disk so an inactive model's host RAM is just OS page cache (==
        # ComfyUI's file-slice source [CU memory_management.py L36-L75]), never a held copy.
        # from_module=True means root is a LIVE (already loaded) module whose live Linear names are
        # matched to disk keys model-agnostically (_match_disk_keys) rather than read straight from
        # a meta module by key. NOT holding the ~15 GB of live CPU weights is the fix for the
        # page-cache starvation in PLAN-te-streaming.md "EMPIRICAL".
        self.tdir = tdir
        # manage: opt into the ComfyUI-style coexisting-dynamic-models manager (registry + GPU
        # release/reload at load boundaries
        # [CU model_management.py L849, model_patcher.py L1937-1941]). Lets the TE stay loaded
        # across requests (no per-request rebuild) while its GPU footprint is reclaimed during
        # denoise. Safe now that big weights stream from disk (bounded host RAM); the earlier
        # net-loss was holding the TE's live weights in RAM (PLAN-te-streaming.md "EMPIRICAL"),
        # fixed by from_module-file streaming.
        self.manage = manage
        # Manager state (== a LoadedModel entry [CU model_management.py LoadedModel]). root = the
        # module whose forward is the load boundary; _staged = the streamed modules (for release
        # re-fault); _released/_resident_backup track the CPU<->GPU move of resident params.
        # Registered at end of ctor.
        self.root = root; self._staged = []
        self._released = False; self._resident_backup = None; self._act_handle = None
        # Pinning tier (file path): streamed weights pinned into a HostBuffer up to `pin_budget`
        # bytes get truly-async H2D; weights beyond the budget stream file->GPU. Copies ComfyUI's
        # pin_memory [CU pinned_memory.py L66-L119]; the budget is measured by the caller
        # (placement.pin_budget(), mirroring [CU model_management.py ensure_pin_budget L645]). No
        # hardcoded sizes.
        self.pin_budget = int(pin_budget)
        self.hb = None; self._registered = []; self.pins = {}
        # Double-buffered prefetch overlap (file path only; the TE runs once, not per step).
        # MEASURED by default -- decided after pinning (see below): ON when the model fully fits
        # the RAM budget (RAM-bound -> overlapping the next pinned H2D behind compute is a win),
        # OFF when some weights stream from disk (>RAM -> disk-bound; overlap can't beat the disk
        # and adds sync cost, measured +16% on qwen). SKY_AIMDO_VBAR_PREFETCH=0/1 forces it.
        # Mirrors the LOWVRAM streamer's copy-stream + ping-pong buffers and ComfyUI's
        # offload-stream prefetch [CU model_prefetch.py L34, ops.py cast_modules_with_vbar L91].
        # Safe because an in-use layer is PINNED (vbar_fault pins, _post unpins
        # [CU ops.py L129/L392]) so prefetching the next layer's fault cannot evict it; on a full
        # VBAR the fault returns OOM and we read into a temp buffer instead (no crash, no spill).
        # "0" | "1" | None(=measured)
        self._prefetch_env = os.environ.get("SKY_AIMDO_VBAR_PREFETCH")
        self.prefetch = False  # set after pinning (file path)
        self.order = []; self.pos = {}
        self.offload_stream = None; self.cast_buffer2 = None; self.cast_buffers = None
        if not _ctl.devctxs:
            _ctl.init_device(self.device_index)  # [AI control.py init_device]

        # LoRA as ComfyUI-style on-cast weight patches (NOT PEFT adapters): per target weight keep
        # the small up/down factors GPU-resident and add (up@down)*scale to the base weight right
        # after it streams in (see _pre). == ComfyUI's s.weight_function applied during cast_to
        # [CU ops.py L357-L380]. Keyed by the base weight's checkpoint key.
        self._lora = self._load_lora(lora_files) if lora_files else {}

        # from_module: root is a LIVE (already-loaded) module. Used for a text encoder loaded by an
        # upstream framework (e.g. diffusers/transformers) that rewrites the checkpoint keys
        # (aimdo.md §10) so a plain name->disk-key match would skip every Linear. We still STREAM
        # its big Linears from disk (matching live names to disk keys model-agnostically,
        # _match_disk_keys) -- only its small resident params (token embedding, vision-tower
        # conv3d, norms, biases) go on-GPU so the module runs fully on CUDA (conv3d on CUDA, not
        # CPU -- a CPU encode poisons the VAE's CUDA conv3d).
        if from_module:
            self._init_from_module(root)
            return

        offsets = _offsets(tdir)
        # One open handle per shard, kept for read_file_to_device during forward (closed in
        # free()). ComfyUI keeps the equivalent in `_comfy_tensor_file_slice.file_ref`
        # [CU memory_management.py L44].
        self.files = {p: open(p, "rb") for p, *_ in {(v[0],) for v in offsets.values()}}

        # Collect nn.Linear weights to stream (== ComfyUI's CastWeightBiasOp weights
        # [CU ops.py L445+]).
        linears = {}; skipped = 0; tiny = 0
        for name, m in root.named_modules():
            if isinstance(m, torch.nn.Linear):
                # PEFT (LoRA) wraps a target Linear as `<name>.base_layer`; the checkpoint key has
                # no `.base_layer`. Strip it so the streamed BASE weight matches the checkpoint.
                # The small lora_A/lora_B adapter Linears are absent from the checkpoint -> skipped
                # here and kept GPU-resident by the caller; LoraLayer.forward adds their low-rank
                # delta on top of the streamed base. This is ComfyUI's weight-patch idea
                # [CU ops.py s.weight_function L357-L380], applied as a resident PEFT adapter
                # instead of an on-cast delta.
                base_name = name[:-len(".base_layer")] if name.endswith(".base_layer") else name
                weight_key = base_name + ".weight"
                if weight_key not in offsets:
                    skipped += 1; continue  # lora adapter / tied / unsaved
                # Tiny weights stay GPU-resident instead of streaming. ComfyUI force-loads modules
                # <= 16 KiB [CU model_patcher.py L1870] because mixing tiny + giant streamed
                # weights causes lopsided stream-buffer rotations that stall. Excluding the key
                # from `linears` here leaves it out of `streamed` below, so it loads resident with
                # the other small params.
                if offsets[weight_key][2] <= 16 * 1024:
                    tiny += 1; continue
                linears[m] = weight_key
        if skipped:
            print("[aimdo] skipped %d Linear(s) absent from checkpoint" % skipped, flush=True)
        if tiny:
            print("[aimdo] %d tiny Linear(s) <=16KiB kept resident (not streamed)"
                  % tiny, flush=True)

        # dtype guard (DIFFERENCE #3): we stream + use weights in their STORED dtype. ComfyUI
        # instead casts to the compute dtype during the copy
        # [CU model_management.py cast_to L1453, applied at ops.py L375-L380]. Safe while
        # stored==compute (bf16 here); warn loudly if a model mixes dtypes so it fails visibly
        # rather than silently mis-running.
        bad = {offsets[weight_key][3] for weight_key in linears.values()
               if offsets[weight_key][3] != self.compute_dtype}
        if bad:
            print("[aimdo] WARNING: streamed weight dtype(s) %s != compute_dtype %s; NOT cast "
                  "(DIFFERENCE #3; ComfyUI casts in [CU model_management.py cast_to L1453])"
                  % (sorted(map(str, bad)), self.compute_dtype), flush=True)

        # Materialise everything that is NOT a streamed weight (norms, embeddings, biases) resident
        # on GPU. ComfyUI keeps these small params on-device too; only big weights stream.
        # assign=True installs without cloning.
        from safetensors import safe_open
        streamed = set(linears.values())
        state_dict = {}
        for p in self.files:
            with safe_open(p, framework="pt") as sf:
                for k in sf.keys():
                    if k not in streamed:
                        state_dict[k] = sf.get_tensor(k).to(self.device)
        root.load_state_dict(state_dict, strict=False, assign=True)

        # One VBAR backs every streamed weight; far bigger than VRAM, pages committed only on
        # fault(). [AI model_vbar.py ModelVBAR L49].
        total = sum(_align(offsets[weight_key][2]) for weight_key in linears.values())
        self.vbar = ModelVBAR(int(total) + (64 << 20), device=self.device_index)
        # Mark this model's VBAR pages high-priority for VRAM retention, so aimdo keeps as many of
        # OUR weights resident as fit before evicting them. ComfyUI does this once per dynamic load
        # [CU model_patcher.py L1809] -> [AI model_vbar.py prioritize L60].
        self.vbar.prioritize()

        # Pin streamed weights into the HostBuffer ONLY when the WHOLE set fits the RAM budget
        # (fits-RAM). Partial pinning of a >RAM model on a tight GPU exhausts the host-registration
        # / BAR mapping and OOMs the next GPU alloc (aimdo.md s6: "pinning a large budget ... not
        # worth it on this box"). That case streams pageable from the page cache instead (the
        # original >RAM path). ComfyUI partial-pins via headroom coordination
        # [CU model_management.py ensure_pin_budget L645] we don't replicate, so gate
        # all-or-nothing. (total = aligned streamed bytes, computed for the VBAR above.)
        if 0 < total <= self.pin_budget:
            self._pin_memory(linears, offsets)

        # Measured prefetch decision: ON iff every streamed weight is pinned (model fits the RAM
        # budget -> RAM-bound, overlap is a win); OFF otherwise (some stream from disk ->
        # disk-bound). Env forces it.
        all_pinned = len(linears) > 0 and len(self.pins) == len(linears)
        self.prefetch = ((self._prefetch_env == "1") if self._prefetch_env is not None
                         else all_pinned)
        print("[aimdo] prefetch=%s (all_pinned=%s, env=%s)"
              % (self.prefetch, all_pinned, self._prefetch_env), flush=True)

        # Cast buffers for the OFFLOADED case (weight didn't fault into its VBAR slot). Two
        # ping-pong views carved from the persistent RESERVED VRAMBuffer
        # (get_aimdo_cast_buffer) -- NOT
        # per-build torch.empty -- so they don't fight cudaMallocAsync for activation VRAM. ==
        # ComfyUI's offload stream [CU model_management.py get_offload_stream L1385] + reserved
        # cast buffer
        # [CU model_management.py get_aimdo_cast_buffer L1343, ops.py get_cast_buffer L112-L124].
        # The faulted case reads straight into the resident VBAR slot; only OOM'd layers use these
        # buffers.
        largest = max(_align(offsets[weight_key][2]) for weight_key in linears.values())
        buffer_size = _align(largest) + 512
        # Reserved 2-buffer ping-pong pool (ComfyUI's reserved cast buffer
        # [CU model_management.py get_aimdo_cast_buffer L1343, ops.py get_cast_buffer L112-L124]);
        # the copy stream is used only when prefetching. The sync/no-prefetch path uses just
        # bufs[0].
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
        # Parse kohya/diffusers LoRA files into {base_weight_key:
        # [(up[out,rank], down[rank,in], scale), ...]}. Each target keeps a LIST so multiple
        # stacked LoRAs (e.g. lightning + angles) accumulate -- their deltas add. delta =
        # scale*(up@down), scale = (alpha/rank) * per-file weight. Each spec is a path or (path,
        # weight); the base weight key is the adapter prefix + ".weight" (1:1 with the streamed
        # checkpoint key).
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
        # Drain a sticky async CUDA error (e.g. a failed cudaHostRegister) so it doesn't resurface
        # at an unrelated later call. == ComfyUI discard_cuda_async_error
        # [CU model_management.py L1505].
        try:
            a = torch.ones(1, dtype=torch.uint8, device=self.device); _ = a + a
            torch.cuda.synchronize(self.device)
        except RuntimeError:
            pass

    def _pin_memory(self, linears, offsets):
        # Pin streamed weights into a HostBuffer up to self.pin_budget bytes for truly-async H2D,
        # copying ComfyUI pin_memory [CU pinned_memory.py L66-L119]: extend the HostBuffer, read
        # the file slice into it (host-only), cudaHostRegister the region
        # [CU pinned_memory.py L98], keep the pinned view as the H2D source. Weights past the
        # budget stay file-streamed. ComfyUI selects by a priority balancer
        # [CU pinned_memory.py _add_to_bucket L12]; for a single-model server, in-order up to the
        # budget is the same set when everything fits.
        used = 0; pinned = []
        for m, weight_key in linears.items():
            a = _align(offsets[weight_key][2])
            if used + a > self.pin_budget:
                continue  # past budget -> this weight streams from file
            pinned.append(weight_key); used += a
        if not pinned:
            return
        # HostBuffer sized to the pinned set (+headroom). [AI host_buffer.py HostBuffer L78];
        # ComfyUI sizes its pinned hostbuf via pinned_hostbuf_size [CU model_management.py L1500].
        self.hb = _hb.HostBuffer(0, 64 * 1024 * 1024, used + (64 << 20))
        layout = {}
        for weight_key in pinned:
            p, file_offset, num_bytes, dtype, shape = offsets[weight_key]
            offset = self.hb.size
            self.hb.extend(_align(num_bytes), register=False)  # [AI host_buffer.py extend L94]
            # file -> HostBuffer once (host-only)
            self.hb.read_file_slice(self.files[p], file_offset, num_bytes, offset=offset)
            layout[weight_key] = (offset, num_bytes, dtype, shape)
        host = _at.hostbuf_to_tensor(self.hb)  # uint8 view over the staged buffer
        base = host.data_ptr(); cudart = torch.cuda.cudart(); ok = 0
        for weight_key, (offset, num_bytes, dtype, shape) in layout.items():
            # cudaHostRegister the exact region so torch sees the H2D source as pinned (async
            # copy). == ComfyUI pin_memory's cudaHostRegister [CU pinned_memory.py L98] (flags 0 =
            # Default vs ComfyUI's 1 = Portable; identical for single-device/single-context use).
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
        # pinned HostBuffer view if pinned, else None -> file-stream
        state.host = self.pins.get(weight_key)
        # [AI model_vbar.py alloc L66] -> (vbar, addr, num_bytes)
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
        # root is a LIVE module already loaded by an upstream framework (not a meta model), so the
        # file __init__ (which loads a meta module from disk by key) can't be used directly.
        # Materialise the small NON-streamed params/buffers (token embedding, vision-tower convs,
        # norms, biases) resident on GPU so the module runs fully on CUDA, and STREAM the big
        # Linears from the .safetensors file -- their live CPU weights are dropped, so host RAM
        # stays page-cache only (== ComfyUI's file-slice source [CU memory_management.py L36-L75],
        # the fix for the PLAN "EMPIRICAL" page-cache starvation).
        if self.tdir is None:
            raise ValueError("Offloader(from_module=True) requires tdir (the model's "
                             "checkpoint dir) so big weights stream from disk instead of being "
                             "held in RAM.")
        linears = []
        for name, m in root.named_modules():
            # Skip lm_head: its weight is tied to the token embedding (shared storage), so it must
            # NOT be stripped/streamed -- it follows the embedding to GPU resident. It is unused in
            # encode.
            if isinstance(m, torch.nn.Linear) and "lm_head" not in name \
                    and m.weight is not None and m.weight.device.type == "cpu":
                linears.append((name, m))

        # Map each live Linear -> its disk key, MODEL-AGNOSTICALLY, before materialising residents
        # (so a mapping failure aborts before we move anything).
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

        # Resident: every param/buffer that is NOT a streamed Linear weight -> GPU (incl. the tied
        # embedding, conv weights, norms, biases). Done before staging so the embedding lands
        # first.
        streamed_ids = {id(m.weight) for m, _ in pairs}
        for p in root.parameters(recurse=True):
            if id(p) not in streamed_ids and p.device.type == "cpu":
                p.data = p.data.to(self.device)
        for b in root.buffers(recurse=True):
            if b.device.type == "cpu":
                b.data = b.data.to(self.device)

        total = sum(_align(offsets[weight_key][2]) for _, weight_key in pairs)
        self.vbar = ModelVBAR(int(total) + (64 << 20), device=self.device_index)
        self.vbar.prioritize()  # high-priority VRAM retention [CU model_patcher.py L1809]
        # Shared RESERVED cast buffer (== the transformer file path
        # [CU model_management.py get_aimdo_cast_buffer L1343]) for OOM'd layers; no pinning here
        # (minimise RAM -> stream from the page cache). Runs once per request, so no prefetch.
        largest = max(_align(offsets[weight_key][2]) for _, weight_key in pairs)
        self.offload_stream, self.cast_buffers = get_aimdo_cast_buffer(
            self.device_index, _align(largest) + 512)
        self.cast_buffer = self.cast_buffers[0]; self.offload_stream = None
        self.pins = {}  # no pinned host weights -> _stage sets L.host=None (file)
        self.files = {p: open(p, "rb")
                      for p, *_ in {(offsets[weight_key][0],) for _, weight_key in pairs}}
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
        # Map each live streamed Linear -> its disk key WITHOUT per-model hardcoding. An upstream
        # loader rewrites only the NAME (it wraps/nests submodules), never the tensor, so a live
        # name and its disk key share a dotted SUFFIX and identical (shape, dtype). For each live
        # weight, among the disk keys with matching (shape, dtype), pick the one with the longest
        # common segment-suffix; require it unique. ComfyUI sidesteps this -- each tensor's storage
        # natively carries its file slice [CU memory_management.py L36, utils.py L113] -- but
        # diffusers' loader drops that link, so we rebuild the name->slice map structurally.
        # Returns (pairs[(module, disk_key)], missing[names], bad_dtypes).
        by_kind = {}
        for k, (_, _, _nb, dtype, shape) in offsets.items():
            by_kind.setdefault((tuple(shape), dtype), []).append(k.split("."))
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

    # ---- ComfyUI-style manager: activation (load boundary) + release/reload (partially_unload)
    # ----
    def _register_manager(self):
        # Join the registry + hook the root forward as the load boundary. == a LoadedModel entering
        # current_loaded_models; the hook is the analog of load_models_gpu(model)
        # [CU model_management.py L849] running before the model executes.
        _LOADED.setdefault(self.device_index, []).append(self)
        self._act_handle = self.root.register_forward_pre_hook(self._activate_hook)

    def _activate_hook(self, m, args):
        # Fires before every root forward; cheap re-entry guard so per-step transformer calls are
        # ~free.
        if _ACTIVE.get(self.device_index) is self:
            return
        self._activate()

    def activate(self):
        # Public load-boundary trigger == ComfyUI load_models_gpu(model) called BEFORE the model
        # runs [CU model_management.py L849]. Use it when the framework probes device placement
        # EARLIER than the module's own forward -- e.g. diffusers reads self._execution_device at
        # pipeline __call__ start and passes it into encode_prompt, so a module forward-pre-hook
        # reloads the TE too late (its params are still on CPU when the device is decided -> input
        # on cpu / weight on cuda mismatch). Calling this before pipe() ensures the model is
        # GPU-resident when the device is read. No-op if unmanaged.
        if self.manage:
            self._activate()

    def _activate(self):
        # This model becomes active: reload it if released, prioritize its pages, and release the
        # other (now lower-priority) dynamic models on this device. == load_models_gpu ->
        # free_memory [CU model_management.py L849, L909-914]; the explicit release replaces the
        # on-demand cross-vbar eviction that does not fire here (see module header).
        if self._released:
            self.restore_loaded_backups()
        loaded = _LOADED.setdefault(self.device_index, [])
        if self in loaded:
            loaded.remove(self)
        loaded.insert(0, self)  # most-recently-active first [CU L945 insert(0,...)]
        _ACTIVE[self.device_index] = self
        self.vbar.prioritize()  # [AI model_vbar.py prioritize L60] [CU L1808-1809]
        for other in loaded[1:]:  # release the OTHER dynamic models [CU free_memory L805-834]
            other.partially_unload()

    def partially_unload(self):
        # Reclaim this model's GPU footprint, reloadably: decommit its VBAR pages and move its
        # resident (non-streamed) params/buffers -- the qwen TE's 1 GB token embedding + vision
        # convs/norms -- back to CPU, recording them for restore_loaded_backups.
        # == partially_unload ->
        # vbar.free_memory [CU model_patcher.py L1937-1938] + restore_loaded_backups
        # [CU model_patcher.py L1768, called at L1941].
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
        self.vbar.deprioritize()  # [AI model_vbar.py free_memory L107 / deprioritize L63]
        # Streamed slots are gone -> force a clean re-read on the next fault
        # (stale gpu/signature would
        # falsely "match" the reused signature). Mirrors set_dirty(_v_signature=None)
        # [CU model_patcher.py L1817-1819].
        for m in self._staged:
            state = m._aimdo; state.gpu = None; state.signature = None
            state.ready = False; state.dst = None; state.ev = None
        # Resident params/buffers -> CPU. parameters() dedups, so the tied embed_tokens/lm_head
        # weight moves once and both modules follow. The streamed weights were stripped to
        # placeholders (not Parameters), so they are skipped here.
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
        # The bounce buffer is the shared RESERVED STREAM_AIMDO_CAST_BUFFERS buffer
        # (persistent across model swaps ==
        # ComfyUI's STREAM_AIMDO_CAST_BUFFERS [CU model_management.py L1343]) -> leave it; only the
        # streamed VBAR pages + resident params were ours to free.
        self._released = True
        # Return the freed VRAM (decommitted VBAR pages + the off-loaded resident params) to the
        # allocator so the NEWLY-active model's VBAR can commit it as resident -- otherwise it
        # stays cached/reserved and the active model re-streams every layer every step (no
        # residency -> disk-bound). == ComfyUI calling soft_empty_cache() after an unload
        # [CU model_management.py L840-846].
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
        # weights re-fault on the next forward (their slots were decommitted). ==
        # restore_loaded_backups back onto device + vbar.prioritize
        # [CU model_patcher.py L1768, L1808-1809].
        if not self._released:
            return
        for t in self._resident_backup or []:
            t.data = t.data.to(self.device)
        self._resident_backup = None
        self.vbar.prioritize()
        self._released = False
        print("[aimdo] restore_loaded_backups: resident tensors -> GPU", flush=True)

    def _pre(self, m, args):
        # Fault, then provide the weight. == ComfyUI DynamicVRAM cast [CU ops.py L128-L141], sync
        # path.
        state = m._aimdo
        if self.prefetch:
            return self._pre_prefetch(m, state)
        signature = vbar_fault(state.slot)  # [AI model_vbar.py L133]
        # [AI L142]
        if (signature is not None and state.gpu is not None
                and vbar_signature_compare(signature, state.signature)):
            m.weight = state.gpu  # resident: reuse, NO read
            return  # [CU ops.py L136-L138]
        # Enqueue the fast-DMA on the COMPUTE stream so the F.linear that consumes the weight is
        # ordered after it on the same stream (no event needed in the sync path).
        strm = int(torch.cuda.current_stream(self.device).cuda_stream)
        # Destination view: the resident VBAR slot if faulted in, else the reused temp buffer
        # (offloaded -- aimdo refused to overcommit; no WDDM spill). [CU ops.py L141 / L163-L164].
        if signature is not None:
            # [AI torch.py L24]
            weight = (_at.aimdo_to_tensor(state.slot, self.device)[:state.num_bytes]
                      .view(state.dtype).view(state.shape))
        else:
            weight = self.cast_buffer[:state.num_bytes].view(state.dtype).view(state.shape)
        if state.host is not None:
            # Pinned HostBuffer view (file path, within pin budget -> truly-async H2D). ==
            # ComfyUI's pinned xfer_source [CU ops.py L148-L150, pinned_memory.py L66].
            weight.copy_(state.host, non_blocking=True)
        else:
            # file: fast-DMA read .safetensors slice -> GPU (page cache if hot, disk if cold;
            # mark_cold drives comfy-aimdo's RAM-pressure cache).
            # [AI host_buffer.py read_file_to_device L67].
            _hb.read_file_to_device(state.file, state.file_offset, state.num_bytes,
                                    strm, weight.data_ptr(), self.device_index)
        if state.lora is not None:
            # On-cast LoRA patch: w += sum_i scale_i*(up_i@down_i), in place on the freshly-read
            # base weight (multiple stacked LoRAs accumulate). For a resident slot this stays
            # applied (reused without re-reading); on re-fault it is re-applied after the base
            # re-read. == ComfyUI weight_function during cast [CU ops.py L357-L380].
            for up, down, scale in state.lora:
                weight.addmm_(up, down, alpha=scale)
        state.gpu = weight; state.signature = signature
        m.weight = weight

    def _do_read(self, state, weight, offload_stream):
        # The actual H2D for one layer into `weight`, on `offload_stream` (None = compute stream).
        # Shared by the prefetch and not-yet-prefetched paths. == ComfyUI cast_to /
        # cast_to_gathered + the on-cast LoRA weight_function
        # [CU model_management.py L1453, ops.py L357-L380].
        strm = int((offload_stream or torch.cuda.current_stream(self.device)).cuda_stream)
        if state.host is not None:
            # Pinned HostBuffer view (file path, within pin budget) -> async H2D copy. Pinned
            # source == ComfyUI's xfer_source from get_pin [CU ops.py L148-L150].
            weight.copy_(state.host, non_blocking=True)
        else:
            _hb.read_file_to_device(state.file, state.file_offset, state.num_bytes,
                                    strm, weight.data_ptr(), self.device_index)
        if state.lora is not None:
            for up, down, scale in state.lora:
                weight.addmm_(up, down, alpha=scale)

    def _fetch(self, state, copy, bufidx):
        # Fault (pins the slot) + read this layer into its VBAR slot (faulted/resident) or a
        # ping-pong temp buffer (offloaded). copy=True runs on the copy stream and records L.ev for
        # the consumer to wait on; copy=False runs inline on the compute stream. == ComfyUI
        # cast_modules_with_vbar [CU ops.py L128-L177].
        signature = vbar_fault(state.slot)  # [AI model_vbar.py L133]
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
            # Don't overwrite a ping-pong buffer until the compute that last read it has finished.
            # == ComfyUI get_offload_stream wait_stream
            # [CU model_management.py L1396].
            offload_stream.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(offload_stream):
                self._do_read(state, weight, offload_stream)
                state.ev = offload_stream.record_event()  # consumer waits on this
        else:
            self._do_read(state, weight, None)
            state.ev = None
        state.gpu = weight; state.signature = signature; state.dst = weight; state.ready = True

    def _pre_prefetch(self, m, state):
        # Provide THIS layer's weight (faulting it inline if it was not already prefetched), then
        # kick the NEXT layer's fault+read on the copy stream so it overlaps this layer's compute.
        # Execution order is learned on the first step (order grows as layers run -> no overlap on
        # step 1). == ComfyUI cast + prefetch_queue_pop
        # [CU ops.py L316-L334, model_prefetch.py L34].
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

    def _post(self, m, args, output):
        state = m._aimdo
        if state.signature is not None:
            vbar_unpin(state.slot)  # [AI model_vbar.py L137]
        else:
            state.gpu = None  # temp reused next layer
        if self.prefetch:
            # Clear this layer's prefetch state so it is re-faulted next step. == ComfyUI dropping
            # the per-module prefetch dict after the cast [CU ops.py L333 delattr(s, "_prefetch")].
            state.ready = False; state.dst = None; state.ev = None  # consumed; refetch next step
        m.weight = state.ph
        return output

    def free(self):
        if self._freed:
            return
        self._freed = True
        # Leave the manager first: drop the root activation hook and the registry/active entries so
        # a teardown can't re-trigger activation and a half-freed offloader can't be released by a
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
        # If released by the manager (vbar pages decommitted, resident params on CPU), restore
        # first so the teardown below frees a vbar in its normal populated/prioritized state -- the
        # same state the proven free()+rebuild path tore down from. Freeing a free_memory()'d vbar
        # + the next pipe's load was observed to segfault (PLAN-te-streaming.md "Known remaining
        # item").
        if self._released:
            try:
                self.restore_loaded_backups()
            except Exception:
                pass
        try:
            # Quiesce the copy stream first: it may hold in-flight prefetch reads into the VBAR /
            # temp buffers; freeing those underneath an outstanding copy is a use-after-free.
            if self.offload_stream is not None:
                self.offload_stream.synchronize()
            torch.cuda.synchronize(self.device)
        except Exception:
            pass
        # cudaHostUnregister every pinned region BEFORE freeing the HostBuffer, then drain any
        # sticky error -- else the orphaned registrations make the next HostBuffer (often the same
        # host addresses) fail with "already mapped". == ComfyUI unpin_memory
        # [CU model_management.py L1553].
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
        # stream are persistent in module-level STREAM_AIMDO_CAST_BUFFERS and reused across
        # model swaps (== ComfyUI
        # keeping STREAM_AIMDO_CAST_BUFFERS for the process). Freeing here would defeat that and
        # re-allocate.
        self.cast_buffer = None; self.cast_buffer2 = None
        self.cast_buffers = None; self.offload_stream = None
        self.order = []; self.pos = {}
        for f in getattr(self, "files", {}).values():
            try:
                f.close()
            except Exception:
                pass
        self.files = {}
