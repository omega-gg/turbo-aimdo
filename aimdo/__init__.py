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
#
#  aimdo offload backend (v2) -- the GPL "custom block" the (LGPL) runner discovers as
#  backend/<mode>/ (here backend/aimdo/ for cuda_offload="aimdo") and drives through this seam only:
#
#      pre_torch_init()                       - one-time setup; MUST run before `import torch`
#      available()                            - True once the vendored offloader imports (any device)
#      supports(engine)                       - True if this backend can place `engine`
#      load_pipe(model, dtype, engine, ...)   - build a fully-placed diffusers pipeline
#      prepare/reclaim/release(pipe)          - per-generation / teardown hooks
#
#  v2 delegates all offloading to a byte-for-byte vendored ComfyUI snapshot (aimdo/comfy/) via the
#  thin bridge in aimdo/adapter.py, so the SAME device-agnostic path serves CPU / CUDA / MPS.
#  comfy-aimdo's CUDA-only VBAR is an optional accelerator (aimdo_enabled) that runs ComfyUI's
#  ModelPatcherDynamic (partial GPU residency) instead of the plain patcher -- streaming weights from
#  host RAM, or disk->VRAM for a component too big for RAM.
#
# =================================================================================================

import os
import traceback

# True once pre_torch_init() has run. Unlike v1, the native path needs no comfy-aimdo, so this is
# True on CPU/MPS too.
_available = False


# Per-engine diffusers classes: (PipelineCls, TransformerCls). The transformer class is used to
# meta-load an oversized transformer for the disk-stream path. Add engines as scripts migrate.
def _classes(engine):
    if engine == "flux2":
        from diffusers import Flux2KleinPipeline, Flux2Transformer2DModel
        return Flux2KleinPipeline, Flux2Transformer2DModel

    if engine == "z-image":
        from diffusers import ZImagePipeline, ZImageTransformer2DModel
        return ZImagePipeline, ZImageTransformer2DModel

    if engine == "qwen-image-edit":
        from diffusers import QwenImageEditPlusPipeline, QwenImageTransformer2DModel
        return QwenImageEditPlusPipeline, QwenImageTransformer2DModel

    raise ValueError("aimdo backend: unsupported engine %r" % (engine,))


def supports(engine):
    """True if this backend can place `engine`."""
    return engine in ("flux2", "z-image", "qwen-image-edit")


_vbar_ready = False


def pre_torch_init():
    """One-time setup; MUST run before `import torch` (the runner calls it there). Its ONLY job that
    truly needs to precede torch is installing comfy-aimdo's CUDA allocator hooks -- so this
    initialises comfy-aimdo here (and nowhere else) and imports NOTHING that pulls in torch.

    comfy-aimdo is optional: absent (cpu/mps builds) or a failed init just leaves `_vbar_ready` False
    and the portable native cast path runs. When it succeeds, load_pipe flips aimdo_enabled and the
    transformer streams through the VBAR dynamic path. The vendored `comfy` package (which imports
    torch) is imported lazily in load_pipe, AFTER the runner's own `import torch`."""
    global _available, _vbar_ready

    try:
        import comfy_aimdo.control as ctl  # must not import torch at module load
        _vbar_ready = bool(ctl.init())
        if _vbar_ready:
            ctl.set_log_warning()
    except Exception:
        _vbar_ready = False

    _available = True


def available():
    """True once pre_torch_init() has brought up the vendored offloader (any device)."""
    return _available


def load_pipe(model, dtype, engine, device="cuda:0", lora_files=None):
    """Build a diffusers pipeline whose big models (transformer, text encoder) are offloaded through
    ComfyUI's ModelPatcher and streamed from host RAM to the compute device per forward -- exactly how
    ComfyUI offloads a UNet. Device-agnostic: `device` selects CPU / CUDA / MPS via the adapter. On
    CUDA with comfy-aimdo present it uses ComfyUI's ModelPatcherDynamic (partial GPU residency sized
    from live free VRAM), streaming each component from host RAM or, when it's too big for RAM, from
    disk; otherwise the portable native cast path. lora_files: [(path, strength)]."""
    import torch  # noqa: F401
    from . import adapter

    load_dev = adapter.set_device(device)

    PipelineCls, Transformer = _classes(engine)

    # VBAR: CUDA + comfy-aimdo (initialised in pre_torch_init) -> flip aimdo_enabled so ComfyUI's
    # ModelPatcherDynamic engages (the "dynamic VRAM loading" path). Must run before loading so the
    # vendored device-selection / lazy-load branches see the flag. Off CUDA (or without comfy-aimdo)
    # this is False and the plain-ModelPatcher native cast path runs instead.
    use_vbar = _vbar_ready and adapter.enable_vbar(device)

    patchers = []

    # Weights live in host RAM and stream to VRAM per-forward -- byte-for-byte what ComfyUI does: it
    # loads the checkpoint to CPU (mmap safetensors), wraps each big model in a ModelPatcher, and
    # partial-loads / streams to the GPU (ComfyUI's own log: "prepared for dynamic VRAM loading. NNNN
    # MB Staged"). Residency (how much stays GPU-resident vs streams) is decided GPU-agnostically by
    # ComfyUI's load_models_gpu from live free VRAM -- no card-specific thresholds.
    #
    # A component too big for host RAM instead streams disk->VRAM through file-slices
    # (adapter.fits_in_ram -> load_streamed / assign_streamed_weights) -- the only way to run a model
    # larger than RAM (e.g. qwen-image-edit's ~55GB on a small box). That path needs the dynamic
    # patcher, so it's gated on use_vbar; the fit test is on-disk size vs host RAM, GPU/card-agnostic.
    build = adapter.build_dynamic_patcher if use_vbar else adapter.build_patcher

    # Transformer: RAM-resident (fast) when it fits, else disk-streamed via a meta-loaded module whose
    # weights are file-sliced (never materialised in RAM). load_streamed comfy-izes internally.
    stream_transformer = use_vbar and not adapter.fits_in_ram(os.path.join(model, "transformer"))

    if stream_transformer:
        transformer, missing = adapter.load_streamed(Transformer, os.path.join(model, "transformer"),
                                                     dtype)
        if missing:
            print("aimdo: %d transformer weights had no matching module (skipped)" % len(missing),
                  flush=True)
        p = PipelineCls.from_pretrained(model, transformer=transformer, torch_dtype=dtype,
                                        use_safetensors=True, low_cpu_mem_usage=True,
                                        local_files_only=True)
    else:
        p = PipelineCls.from_pretrained(model, torch_dtype=dtype, use_safetensors=True,
                                        low_cpu_mem_usage=True, local_files_only=True)
        adapter.comfy_ize(p.transformer)

    adapter.keep_uncastable_resident(p.transformer, load_dev)
    if use_vbar:
        adapter.install_prefetch(p.transformer)  # overlap block weight streaming with compute
    adapter.use_comfy_attention(p.transformer)  # ComfyUI's exact SDPA (comfy.ops, cuDNN-first + size gate)
    adapter.use_kitchen_rope(p.transformer)  # ComfyUI's comfy_kitchen fused RoPE
    transformer_patcher = build(p.transformer)

    # On-cast LoRA (e.g. qwen's Lightning 4-step speed LoRA): applied to the transformer while its
    # weights stream in, via ComfyUI's add_patches -> calculate_weight. lora_files: [(path, strength)].
    if lora_files:
        n = adapter.add_lora(transformer_patcher, lora_files)
        print("aimdo: applied %d LoRA patches" % n, flush=True)

    patchers.append(transformer_patcher)

    # Text encoder: same gate -- RAM-resident (fast) when it fits, else its weights are swapped for
    # file-slices and streamed disk->VRAM. Custom norms (e.g. Qwen3RMSNorm) are kept resident by
    # keep_uncastable_resident since they can't be comfy-ized.
    if getattr(p, "text_encoder", None) is not None:
        adapter.comfy_ize(p.text_encoder)
        adapter.use_comfy_attention(p.text_encoder)
        if use_vbar and not adapter.fits_in_ram(os.path.join(model, "text_encoder")):
            te_missing = adapter.assign_streamed_weights(p.text_encoder,
                                                         os.path.join(model, "text_encoder"))
            if te_missing:
                print("aimdo: %d text-encoder weights had no matching param (skipped)"
                      % len(te_missing), flush=True)
        adapter.keep_uncastable_resident(p.text_encoder, load_dev)
        encoder_patcher = build(p.text_encoder)
        patchers.append(encoder_patcher)
        p._aimdo_encoder = encoder_patcher

    # Small resident modules (VAE) go straight to the compute device; the offloader handles the
    # heavy ones. VAE tiling/slicing is left to the caller.
    if getattr(p, "vae", None) is not None:
        p.vae.to(load_dev)

    p._aimdo_patchers = patchers
    p._aimdo_device = load_dev

    # Diffusers reads _execution_device from module placement; pin it to the compute device so inputs
    # land there while offloaded weights stream in.
    try:
        p._execution_device = load_dev
    except Exception:
        pass

    # NOTE: This might improve performances.
    p.safety_checker = lambda images, **kwargs: (images, [False] * len(images))

    # Cache the text-encoder output by prompt (ComfyUI's node cache): a repeated prompt then skips the
    # encoder's forward, so the streamed encoder is never loaded and the transformer keeps the VRAM.
    adapter.install_encode_cache(p)

    return p


def prepare(pipe):
    """Per-generation load boundary: hand the managed models to ComfyUI's load_models_gpu, which
    partial-loads / streams them to the compute device (its dynamic path streams weights per-forward,
    so this marks them loaded without pinning the whole set). When install_encode_cache serves a repeated
    prompt the text encoder's forward never runs, so its weights never stream and the transformer keeps
    the throughput."""
    patchers = getattr(pipe, "_aimdo_patchers", None)
    if not patchers:
        return

    import comfy.model_management as mm
    mm.load_models_gpu(patchers)


def reclaim(pipe):
    """Per-generation housekeeping: run ComfyUI's per-execution aimdo teardown, then free non-resident
    models + return the allocator pool.

    The teardown copies what ComfyUI runs in the `finally` after EVERY node execution when aimdo is on
    (comfy/execution.py:544-549): reset_cast_buffers + cleanup_prefetch_queues +
    vbars_reset_watermark_limits. It resets the VBAR streaming state (the globally-cached cast buffers,
    the block prefetch queues, the VBAR watermarks) between generations. reclaim() is our analog of
    that per-execution boundary, so we do the same thing ComfyUI does. (This is faithful teardown /
    hygiene; it is NOT what fixed the flux2->z-image first-gen nondeterminism -- that was copying
    ComfyUI's exact SDPA size gate in adapter.use_comfy_attention.)"""
    if not getattr(pipe, "_aimdo_patchers", None):
        return

    import comfy.model_management as mm
    import comfy.memory_management as memm

    if getattr(memm, "aimdo_enabled", False):
        try:
            mm.reset_cast_buffers()

            import comfy.model_prefetch as mp
            mp.cleanup_prefetch_queues()

            import comfy_aimdo.model_vbar as mv
            mv.vbars_reset_watermark_limits()
        except Exception:
            print(traceback.format_exc(), flush=True)

    dev = getattr(pipe, "_aimdo_device", None)
    if dev is not None:
        mm.free_memory(mm.minimum_inference_memory(), dev)
    mm.soft_empty_cache()


def release(pipe):
    """Tear down the offloaders before the pipe is dropped (detach unpatches weights and lets the
    current_loaded_models finalizers fire)."""
    for patcher in getattr(pipe, "_aimdo_patchers", []):
        try:
            patcher.detach(unpatch_all=True)
        except Exception:
            print(traceback.format_exc(), flush=True)
