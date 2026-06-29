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
        print("[HBvbar] reclaim_between_runs: vbars_reset_watermark_limits failed:\n"
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
    m = {}
    for sh in shards:
        p = os.path.join(tdir, sh)
        with open(p, "rb") as f:
            hl = struct.unpack("<Q", f.read(8))[0]; hdr = json.loads(f.read(hl))
        ds = 8 + hl
        for k, info in hdr.items():
            if k == "__metadata__":
                continue
            s, e = info["data_offsets"]
            m[k] = (p, ds + s, e - s, _DT[info["dtype"]], tuple(info["shape"]))
    return m


# Persistent per-device cast buffers for the OFFLOADED ping-pong path (a weight that did NOT fault
# into its VBAR slot). One copy stream + a single RESERVED aimdo VRAMBuffer carved into two views,
# created ONCE per device and reused across every model build (the VRAMBuffer grows to fit the
# largest layer seen). This is ComfyUI's STREAM_AIMDO_CAST_BUFFERS
# [CU model_management.py get_aimdo_cast_buffer L1343]; its bounce tensor is
# aimdo_to_tensor(vrambuf.get(size, offset), device) [CU ops.py get_cast_buffer L124]. Using a
# RESERVED buffer (not torch.empty) is the point: the aimdo allocator accounts for it so it never
# fights cudaMallocAsync for activation VRAM -- the per-layer-temp-alloc cost ComfyUI avoids
# (aimdo.md s5).
_CAST = {}
# 16 GiB virtual reservation, matching ComfyUI [CU model_management.py L1309]. VBAR address space,
# committed lazily by .get(), so this is cheap even on a low-VRAM card.
_CAST_RESERVATION = 16 * 1024 ** 3


def _get_cast_buffers(dev, bsz):
    e = _CAST.get(dev)
    if e is None:
        e = {"stream": torch.cuda.Stream(torch.device("cuda", dev)),
             # [AI vram_buffer.py]
             "vram": _vb.VRAMBuffer(_CAST_RESERVATION, dev), "bufs": None, "bsz": -1}
        _CAST[dev] = e
    if bsz > e["bsz"]:
        # Carve two bsz-sized ping-pong regions (offsets 0 and bsz); .get commits up to 2*bsz.
        vram = e["vram"]; ds = "cuda:%d" % dev
        e["bufs"] = [_at.aimdo_to_tensor(vram.get(bsz, 0), ds),
                     _at.aimdo_to_tensor(vram.get(bsz, bsz), ds)]
        e["bsz"] = bsz
    return e["stream"], e["bufs"]


# Per-Linear streaming state. file/foff/nb = where the weight lives on disk (file mode); host = the
# live CPU weight tensor (from_module mode); slot = the VBAR allocation; sig/gpu = residency
# bookkeeping; ph = the dtype placeholder between forwards; lora = (up, down, scale) GPU tensors
# for the on-cast LoRA delta, or None.
class _L:
    __slots__ = ("file", "foff", "nb", "shape", "dtype", "slot", "sig", "gpu", "ph", "lora",
                 "host", "ready", "ev", "dst")


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
        self.device = torch.device(device); self.dev = self.device.index or 0
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
        self.hb = None; self._registered = []; self._pinsrc = {}
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
        self.cstream = None; self._tmp2 = None; self._bufs = None
        if not _ctl.devctxs:
            _ctl.init_device(self.dev)  # [AI control.py init_device]

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

        om = _offsets(tdir)
        # One open handle per shard, kept for read_file_to_device during forward (closed in
        # free()). ComfyUI keeps the equivalent in `_comfy_tensor_file_slice.file_ref`
        # [CU memory_management.py L44].
        self.files = {p: open(p, "rb") for p, *_ in {(v[0],) for v in om.values()}}

        # Collect nn.Linear weights to stream (== ComfyUI's CastWeightBiasOp weights
        # [CU ops.py L445+]).
        lin = {}; skipped = 0; tiny = 0
        for name, m in root.named_modules():
            if isinstance(m, torch.nn.Linear):
                # PEFT (LoRA) wraps a target Linear as `<name>.base_layer`; the checkpoint key has
                # no `.base_layer`. Strip it so the streamed BASE weight matches the checkpoint.
                # The small lora_A/lora_B adapter Linears are absent from the checkpoint -> skipped
                # here and kept GPU-resident by the caller; LoraLayer.forward adds their low-rank
                # delta on top of the streamed base. This is ComfyUI's weight-patch idea
                # [CU ops.py s.weight_function L357-L380], applied as a resident PEFT adapter
                # instead of an on-cast delta.
                ck = name[:-len(".base_layer")] if name.endswith(".base_layer") else name
                wk = ck + ".weight"
                if wk not in om:
                    skipped += 1; continue  # lora adapter / tied / unsaved
                # Tiny weights stay GPU-resident instead of streaming. ComfyUI force-loads modules
                # <= 16 KiB [CU model_patcher.py L1870] because mixing tiny + giant streamed
                # weights causes lopsided stream-buffer rotations that stall. Excluding the key
                # from `lin` here leaves it out of `streamed` below, so it is loaded resident with
                # the other small params.
                if om[wk][2] <= 16 * 1024:
                    tiny += 1; continue
                lin[m] = wk
        if skipped:
            print("[HBvbar] skipped %d Linear(s) absent from checkpoint" % skipped, flush=True)
        if tiny:
            print("[HBvbar] %d tiny Linear(s) <=16KiB kept resident (not streamed)"
                  % tiny, flush=True)

        # dtype guard (DIFFERENCE #3): we stream + use weights in their STORED dtype. ComfyUI
        # instead casts to the compute dtype during the copy
        # [CU model_management.py cast_to L1453, applied at ops.py L375-L380]. Safe while
        # stored==compute (bf16 here); warn loudly if a model mixes dtypes so it fails visibly
        # rather than silently mis-running.
        bad = {om[wk][3] for wk in lin.values() if om[wk][3] != self.compute_dtype}
        if bad:
            print("[HBvbar] WARNING: streamed weight dtype(s) %s != compute_dtype %s; NOT cast "
                  "(DIFFERENCE #3; ComfyUI casts in [CU model_management.py cast_to L1453])"
                  % (sorted(map(str, bad)), self.compute_dtype), flush=True)

        # Materialise everything that is NOT a streamed weight (norms, embeddings, biases) resident
        # on GPU. ComfyUI keeps these small params on-device too; only big weights stream.
        # assign=True installs without cloning.
        from safetensors import safe_open
        streamed = set(lin.values())
        sd = {}
        for p in self.files:
            with safe_open(p, framework="pt") as sf:
                for k in sf.keys():
                    if k not in streamed:
                        sd[k] = sf.get_tensor(k).to(self.device)
        root.load_state_dict(sd, strict=False, assign=True)

        # One VBAR backs every streamed weight; far bigger than VRAM, pages committed only on
        # fault(). [AI model_vbar.py ModelVBAR L49].
        total = sum(_align(om[wk][2]) for wk in lin.values())
        self.vbar = ModelVBAR(int(total) + (64 << 20), device=self.dev)
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
            self._pin_weights(lin, om)

        # Measured prefetch decision: ON iff every streamed weight is pinned (model fits the RAM
        # budget -> RAM-bound, overlap is a win); OFF otherwise (some stream from disk ->
        # disk-bound). Env forces it.
        all_pinned = len(lin) > 0 and len(self._pinsrc) == len(lin)
        self.prefetch = ((self._prefetch_env == "1") if self._prefetch_env is not None
                         else all_pinned)
        print("[HBvbar] prefetch=%s (all_pinned=%s, env=%s)"
              % (self.prefetch, all_pinned, self._prefetch_env), flush=True)

        # Cast buffers for the OFFLOADED case (weight didn't fault into its VBAR slot). Two
        # ping-pong views carved from the persistent RESERVED VRAMBuffer (_get_cast_buffers) -- NOT
        # per-build torch.empty -- so they don't fight cudaMallocAsync for activation VRAM. ==
        # ComfyUI's offload stream [CU model_management.py get_offload_stream L1385] + reserved
        # cast buffer
        # [CU model_management.py get_aimdo_cast_buffer L1343, ops.py get_cast_buffer L112-L124].
        # The faulted case reads straight into the resident VBAR slot; only OOM'd layers use these
        # buffers.
        big = max(_align(om[wk][2]) for wk in lin.values())
        bsz = _align(big) + 512
        # Reserved 2-buffer ping-pong pool (ComfyUI's reserved cast buffer
        # [CU model_management.py get_aimdo_cast_buffer L1343, ops.py get_cast_buffer L112-L124]);
        # the copy stream is used only when prefetching. The sync/no-prefetch path uses just
        # bufs[0].
        self.cstream, self._bufs = _get_cast_buffers(self.dev, bsz)
        self._tmp = self._bufs[0]
        if not self.prefetch:
            self.cstream = None

        for m, wk in lin.items():
            self._stage(m, wk, om)

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
            path, wt = spec if isinstance(spec, (list, tuple)) else (spec, 1.0)
            with safe_open(path, framework="pt") as sf:
                ks = set(sf.keys())
                prefixes = {k[:-len(ds)] for k in ks for ds, _us in sufs if k.endswith(ds)}
                for pre in prefixes:
                    ds, us = next((d, u) for d, u in sufs if pre + d in ks)
                    # [rank, in]
                    down = sf.get_tensor(pre + ds).to(self.device, self.compute_dtype)
                    up = sf.get_tensor(pre + us).to(self.device, self.compute_dtype)  # [out, rank]
                    rank = down.shape[0]
                    scale = ((sf.get_tensor(pre + ".alpha").item() / rank)
                             if (pre + ".alpha") in ks else 1.0) * wt
                    lora.setdefault(pre + ".weight", []).append((up, down, float(scale)))
        print("[HBvbar] loaded LoRA over %d target(s) from %d file(s)"
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

    def _pin_weights(self, lin, om):
        # Pin streamed weights into a HostBuffer up to self.pin_budget bytes for truly-async H2D,
        # copying ComfyUI pin_memory [CU pinned_memory.py L66-L119]: extend the HostBuffer, read
        # the file slice into it (host-only), cudaHostRegister the region
        # [CU pinned_memory.py L98], keep the pinned view as the H2D source. Weights past the
        # budget stay file-streamed. ComfyUI selects by a priority balancer
        # [CU pinned_memory.py _add_to_bucket L12]; for a single-model server, in-order up to the
        # budget is the same set when everything fits.
        used = 0; pinned = []
        for m, wk in lin.items():
            a = _align(om[wk][2])
            if used + a > self.pin_budget:
                continue  # past budget -> this weight streams from file
            pinned.append(wk); used += a
        if not pinned:
            return
        # HostBuffer sized to the pinned set (+headroom). [AI host_buffer.py HostBuffer L78];
        # ComfyUI sizes its pinned hostbuf via pinned_hostbuf_size [CU model_management.py L1500].
        self.hb = _hb.HostBuffer(0, 64 * 1024 * 1024, used + (64 << 20))
        layout = {}
        for wk in pinned:
            p, fo, nb, dt, sh = om[wk]
            off = self.hb.size
            self.hb.extend(_align(nb), register=False)  # [AI host_buffer.py extend L94]
            # file -> HostBuffer once (host-only)
            self.hb.read_file_slice(self.files[p], fo, nb, offset=off)
            layout[wk] = (off, nb, dt, sh)
        host = _at.hostbuf_to_tensor(self.hb)  # uint8 view over the staged buffer
        base = host.data_ptr(); cudart = torch.cuda.cudart(); ok = 0
        for wk, (off, nb, dt, sh) in layout.items():
            # cudaHostRegister the exact region so torch sees the H2D source as pinned (async
            # copy). == ComfyUI pin_memory's cudaHostRegister [CU pinned_memory.py L98] (flags 0 =
            # Default vs ComfyUI's 1 = Portable; identical for single-device/single-context use).
            if int(cudart.cudaHostRegister(base + off, nb, 0)) == 0:
                self._registered.append(base + off); ok += nb
            else:
                self._discard_cuda_async_error()
            self._pinsrc[wk] = host[off:off + nb].view(dt).view(sh)
        print("[HBvbar] pinned %d/%d streamed weight(s) = %.2f GB (budget %.2f GB)"
              % (len(self._pinsrc), len(lin), ok / 1024 ** 3,
                 self.pin_budget / 1024 ** 3), flush=True)

    def _stage(self, m, wk, om):
        p, fo, nb, dt, sh = om[wk]
        L = _L()
        L.file = self.files[p]; L.foff = fo; L.nb = nb; L.shape = sh; L.dtype = dt
        L.host = self._pinsrc.get(wk)  # pinned HostBuffer view if pinned, else None -> file-stream
        L.slot = self.vbar.alloc(_align(nb))  # [AI model_vbar.py alloc L66] -> (vbar, addr, nb)
        L.sig = None; L.gpu = None
        # prefetch bookkeeping (unused on the sync path)
        L.ready = False; L.ev = None; L.dst = None
        L.lora = self._lora.get(wk)                 # (up, down, scale) on-cast patch, or None
        L.ph = torch.empty(0, dtype=dt, device=self.device)   # dtype placeholder between forwards
        del m._parameters["weight"]; setattr(m, "weight", L.ph)
        m._hbv = L; self._staged.append(m)
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
        lin = []
        for name, m in root.named_modules():
            # Skip lm_head: its weight is tied to the token embedding (shared storage), so it must
            # NOT be stripped/streamed -- it follows the embedding to GPU resident. It is unused in
            # encode.
            if isinstance(m, torch.nn.Linear) and "lm_head" not in name \
                    and m.weight is not None and m.weight.device.type == "cpu":
                lin.append((name, m))

        # Map each live Linear -> its disk key, MODEL-AGNOSTICALLY, before materialising residents
        # (so a mapping failure aborts before we move anything).
        om = _offsets(self.tdir)
        pairs, missing, bad_dt = self._match_disk_keys(lin, om)
        if missing:
            raise RuntimeError("[HBvbar] from_module: %d Linear(s) did not map to a unique "
                               "checkpoint key (e.g. %r). The model<->checkpoint name mapping "
                               "is ambiguous." % (len(missing), missing[0]))
        if bad_dt:
            print("[HBvbar] WARNING: streamed dtype(s) %s != compute_dtype %s; NOT cast "
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

        total = sum(_align(om[wk][2]) for _, wk in pairs)
        self.vbar = ModelVBAR(int(total) + (64 << 20), device=self.dev)
        self.vbar.prioritize()  # high-priority VRAM retention [CU model_patcher.py L1809]
        # Shared RESERVED cast buffer (== the transformer file path
        # [CU model_management.py get_aimdo_cast_buffer L1343]) for OOM'd layers; no pinning here
        # (minimise RAM -> stream from the page cache). Runs once per request, so no prefetch.
        big = max(_align(om[wk][2]) for _, wk in pairs)
        self.cstream, self._bufs = _get_cast_buffers(self.dev, _align(big) + 512)
        self._tmp = self._bufs[0]; self.cstream = None
        self._pinsrc = {}  # no pinned host weights -> _stage sets L.host=None (file)
        self.files = {p: open(p, "rb") for p, *_ in {(om[wk][0],) for _, wk in pairs}}
        for m, wk in pairs:
            self._stage(m, wk, om)  # frees the live CPU weight (del m._parameters["weight"])
        import gc as _gc; _gc.collect()              # reclaim the freed live Linear weights now
        print("[HBvbar] from_module: streaming %d Linear(s) from disk = %.2f GB "
              "(host RAM = page cache only)"
              % (len(pairs), total / 1024 ** 3), flush=True)
        if self.manage:
            self._register_manager()

    def _match_disk_keys(self, lin, om):
        # Map each live streamed Linear -> its disk key WITHOUT per-model hardcoding. An upstream
        # loader rewrites only the NAME (it wraps/nests submodules), never the tensor, so a live
        # name and its disk key share a dotted SUFFIX and identical (shape, dtype). For each live
        # weight, among the disk keys with matching (shape, dtype), pick the one with the longest
        # common segment-suffix; require it unique. ComfyUI sidesteps this -- each tensor's storage
        # natively carries its file slice [CU memory_management.py L36, utils.py L113] -- but
        # diffusers' loader drops that link, so we rebuild the name->slice map structurally.
        # Returns (pairs[(module, disk_key)], missing[names], bad_dtypes).
        by_kind = {}
        for k, (_, _, _nb, dt, sh) in om.items():
            by_kind.setdefault((tuple(sh), dt), []).append(k.split("."))
        pairs = []; missing = []; bad_dt = set()
        for name, m in lin:
            sh = tuple(m.weight.shape); dt = m.weight.dtype
            lk = (name + ".weight").split(".")
            best = None; best_n = -1; tie = False
            for dk in by_kind.get((sh, dt), ()):
                n = 0
                while n < len(dk) and n < len(lk) and dk[-1 - n] == lk[-1 - n]:
                    n += 1
                if n > best_n:
                    best, best_n, tie = dk, n, False
                elif n == best_n:
                    tie = True
            if best is None or best_n == 0 or tie:
                missing.append(name); continue
            wk = ".".join(best)
            if dt != self.compute_dtype:
                bad_dt.add(str(dt))
            pairs.append((m, wk))
        return pairs, missing, bad_dt

    # ---- ComfyUI-style manager: activation (load boundary) + release/reload (partially_unload)
    # ----
    def _register_manager(self):
        # Join the registry + hook the root forward as the load boundary. == a LoadedModel entering
        # current_loaded_models; the hook is the analog of load_models_gpu(model)
        # [CU model_management.py L849] running before the model executes.
        _LOADED.setdefault(self.dev, []).append(self)
        self._act_handle = self.root.register_forward_pre_hook(self._activate_hook)

    def _activate_hook(self, m, args):
        # Fires before every root forward; cheap re-entry guard so per-step transformer calls are
        # ~free.
        if _ACTIVE.get(self.dev) is self:
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
            self.reload_gpu()
        lst = _LOADED.setdefault(self.dev, [])
        if self in lst:
            lst.remove(self)
        lst.insert(0, self)  # most-recently-active first [CU L945 insert(0,...)]
        _ACTIVE[self.dev] = self
        self.vbar.prioritize()  # [AI model_vbar.py prioritize L60] [CU L1808-1809]
        for o in lst[1:]:  # release the OTHER dynamic models [CU free_memory L805-834]
            o.release_gpu()

    def release_gpu(self):
        # Reclaim this model's GPU footprint, reloadably: decommit its VBAR pages and move its
        # resident (non-streamed) params/buffers -- the qwen TE's 1 GB token embedding + vision
        # convs/norms -- back to CPU, recording them for reload_gpu. == partially_unload ->
        # vbar.free_memory [CU model_patcher.py L1937-1938] + restore_loaded_backups
        # [CU model_patcher.py L1768, called at L1941].
        if self._released:
            return
        try:
            if getattr(self, "cstream", None) is not None:
                self.cstream.synchronize()
            # no in-flight reads into the slots we are about to free
            torch.cuda.synchronize(self.device)
        except Exception:
            pass
        free0 = torch.cuda.mem_get_info(self.device)[0]               # free VRAM before reclaim
        vbar_freed = int(self.vbar.free_memory(1 << 62))  # decommit ALL pages; returns bytes freed
        self.vbar.deprioritize()  # [AI model_vbar.py free_memory L107 / deprioritize L63]
        # Streamed slots are gone -> force a clean re-read on the next fault (stale gpu/sig would
        # falsely "match" the reused signature). Mirrors set_dirty(_v_signature=None)
        # [CU model_patcher.py L1817-1819].
        for m in self._staged:
            L = m._hbv; L.gpu = None; L.sig = None; L.ready = False; L.dst = None; L.ev = None
        # Resident params/buffers -> CPU. parameters() dedups, so the tied embed_tokens/lm_head
        # weight moves once and both modules follow. The streamed weights were stripped to
        # placeholders (not Parameters), so they are skipped here.
        bk = []; res_bytes = 0
        for p in self.root.parameters(recurse=True):
            if p.device.type == "cuda":
                res_bytes += p.numel() * p.element_size(); bk.append(p); p.data = p.data.to("cpu")
        for b in self.root.buffers(recurse=True):
            if b.device.type == "cuda":
                res_bytes += b.numel() * b.element_size(); bk.append(b); b.data = b.data.to("cpu")
        self._resident_backup = bk
        # The bounce buffer is the shared RESERVED _CAST buffer (persistent across model swaps ==
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
        g = 1024 ** 3
        print("[HBvbar] release_gpu: VBAR freed %.3f GB + %d resident tensor(s) %.3f GB -> CPU; "
              "free VRAM %.2f -> %.2f GB (+%.2f)"
              % (vbar_freed / g, len(bk), res_bytes / g, free0 / g, free1 / g,
                 (free1 - free0) / g), flush=True)

    def reload_gpu(self):
        # Inverse of release_gpu: resident params/buffers back to GPU, reprioritize. Streamed
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
        print("[HBvbar] reload_gpu: resident tensors -> GPU", flush=True)

    def _pre(self, m, args):
        # Fault, then provide the weight. == ComfyUI DynamicVRAM cast [CU ops.py L128-L141], sync
        # path.
        L = m._hbv
        if self.prefetch:
            return self._pre_prefetch(m, L)
        sig = vbar_fault(L.slot)  # [AI model_vbar.py L133]
        # [AI L142]
        if sig is not None and L.gpu is not None and vbar_signature_compare(sig, L.sig):
            m.weight = L.gpu  # resident: reuse, NO read
            return  # [CU ops.py L136-L138]
        # Enqueue the fast-DMA on the COMPUTE stream so the F.linear that consumes the weight is
        # ordered after it on the same stream (no event needed in the sync path).
        strm = int(torch.cuda.current_stream(self.device).cuda_stream)
        # Destination view: the resident VBAR slot if faulted in, else the reused temp buffer
        # (offloaded -- aimdo refused to overcommit; no WDDM spill). [CU ops.py L141 / L163-L164].
        if sig is not None:
            # [AI torch.py L24]
            w = _at.aimdo_to_tensor(L.slot, self.device)[:L.nb].view(L.dtype).view(L.shape)
        else:
            w = self._tmp[:L.nb].view(L.dtype).view(L.shape)
        if L.host is not None:
            # Pinned HostBuffer view (file path, within pin budget -> truly-async H2D). ==
            # ComfyUI's pinned xfer_source [CU ops.py L148-L150, pinned_memory.py L66].
            w.copy_(L.host, non_blocking=True)
        else:
            # file: fast-DMA read .safetensors slice -> GPU (page cache if hot, disk if cold;
            # mark_cold drives comfy-aimdo's RAM-pressure cache).
            # [AI host_buffer.py read_file_to_device L67].
            _hb.read_file_to_device(L.file, L.foff, L.nb, strm, w.data_ptr(), self.dev)
        if L.lora is not None:
            # On-cast LoRA patch: w += sum_i scale_i*(up_i@down_i), in place on the freshly-read
            # base weight (multiple stacked LoRAs accumulate). For a resident slot this stays
            # applied (reused without re-reading); on re-fault it is re-applied after the base
            # re-read. == ComfyUI weight_function during cast [CU ops.py L357-L380].
            for up, down, scale in L.lora:
                w.addmm_(up, down, alpha=scale)
        L.gpu = w; L.sig = sig
        m.weight = w

    def _do_read(self, L, w, cs):
        # The actual H2D for one layer into view `w`, on stream `cs` (None == compute stream).
        # Shared by the prefetch and not-yet-prefetched paths. == ComfyUI cast_to /
        # cast_to_gathered + the on-cast LoRA weight_function
        # [CU model_management.py L1453, ops.py L357-L380].
        strm = int((cs or torch.cuda.current_stream(self.device)).cuda_stream)
        if L.host is not None:
            # Pinned HostBuffer view (file path, within pin budget) -> async H2D copy. Pinned
            # source == ComfyUI's xfer_source from get_pin [CU ops.py L148-L150].
            w.copy_(L.host, non_blocking=True)
        else:
            _hb.read_file_to_device(L.file, L.foff, L.nb, strm, w.data_ptr(), self.dev)
        if L.lora is not None:
            for up, down, scale in L.lora:
                w.addmm_(up, down, alpha=scale)

    def _fetch(self, L, copy, bufidx):
        # Fault (pins the slot) + read this layer into its VBAR slot (faulted/resident) or a
        # ping-pong temp buffer (offloaded). copy=True runs on the copy stream and records L.ev for
        # the consumer to wait on; copy=False runs inline on the compute stream. == ComfyUI
        # cast_modules_with_vbar [CU ops.py L128-L177].
        sig = vbar_fault(L.slot)  # [AI model_vbar.py L133]
        if sig is not None and L.gpu is not None and vbar_signature_compare(sig, L.sig):
            L.dst = L.gpu; L.sig = sig; L.ev = None; L.ready = True  # resident: reuse, NO read
            return
        if sig is not None:
            w = _at.aimdo_to_tensor(L.slot, self.device)[:L.nb].view(L.dtype).view(L.shape)
        else:
            w = self._bufs[bufidx][:L.nb].view(L.dtype).view(L.shape)
        cs = self.cstream if copy else None
        if cs is not None:
            # Don't overwrite a ping-pong buffer until the compute that last read it has finished.
            # == ComfyUI get_offload_stream wait_stream
            # [CU model_management.py L1396].
            cs.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(cs):
                self._do_read(L, w, cs)
                L.ev = cs.record_event()  # consumer waits on this
        else:
            self._do_read(L, w, None)
            L.ev = None
        L.gpu = w; L.sig = sig; L.dst = w; L.ready = True

    def _pre_prefetch(self, m, L):
        # Provide THIS layer's weight (faulting it inline if it was not already prefetched), then
        # kick the NEXT layer's fault+read on the copy stream so it overlaps this layer's compute.
        # Execution order is learned on the first step (order grows as layers run -> no overlap on
        # step 1). == ComfyUI cast + prefetch_queue_pop
        # [CU ops.py L316-L334, model_prefetch.py L34].
        i = self.pos.get(id(m))
        if i is None:
            i = len(self.order); self.pos[id(m)] = i; self.order.append(m)
        if not L.ready:
            self._fetch(L, copy=False, bufidx=i & 1)  # not prefetched -> compute stream
        if L.ev is not None:
            torch.cuda.current_stream(self.device).wait_event(L.ev)  # compute waits for the copy
        m.weight = L.dst
        if i + 1 < len(self.order):  # prefetch next into the OTHER buf
            nxt = self.order[i + 1]._hbv
            if not nxt.ready:
                self._fetch(nxt, copy=True, bufidx=(i + 1) & 1)

    def _post(self, m, args, output):
        L = m._hbv
        if L.sig is not None:
            vbar_unpin(L.slot)  # [AI model_vbar.py L137]
        else:
            L.gpu = None  # temp reused next layer
        if self.prefetch:
            # Clear this layer's prefetch state so it is re-faulted next step. == ComfyUI dropping
            # the per-module prefetch dict after the cast [CU ops.py L333 delattr(s, "_prefetch")].
            L.ready = False; L.dst = None; L.ev = None  # consumed; refetch next step
        m.weight = L.ph
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
        lst = _LOADED.get(self.dev)
        if lst and self in lst:
            lst.remove(self)
        if _ACTIVE.get(self.dev) is self:
            _ACTIVE.pop(self.dev, None)
        # If released by the manager (vbar pages decommitted, resident params on CPU), restore
        # first so the teardown below frees a vbar in its normal populated/prioritized state -- the
        # same state the proven free()+rebuild path tore down from. Freeing a free_memory()'d vbar
        # + the next pipe's load was observed to segfault (PLAN-te-streaming.md "Known remaining
        # item").
        if self._released:
            try:
                self.reload_gpu()
            except Exception:
                pass
        try:
            # Quiesce the copy stream first: it may hold in-flight prefetch reads into the VBAR /
            # temp buffers; freeing those underneath an outstanding copy is a use-after-free.
            if self.cstream is not None:
                self.cstream.synchronize()
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
        self._pinsrc = {}
        self.vbar = None        # ModelVBAR.__del__ -> vbar_free
        # Drop refs to the cast buffers/stream but do NOT free them: the reserved VRAMBuffer + copy
        # stream are persistent in module-level _CAST and reused across model swaps (== ComfyUI
        # keeping STREAM_AIMDO_CAST_BUFFERS for the process). Freeing here would defeat that and
        # re-allocate.
        self._tmp = None; self._tmp2 = None; self._bufs = None; self.cstream = None
        self.order = []; self.pos = {}
        for f in getattr(self, "files", {}).values():
            try:
                f.close()
            except Exception:
                pass
        self.files = {}
