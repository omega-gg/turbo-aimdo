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
#  aimdo offload backend -- the GPL "custom block" pulled in by the (LGPL) run*.sh / server.sh
#  scripts through a generic, license-neutral seam.
#
#  The scripts never reference aimdo/comfy directly: they discover a subpackage backend/<mode>/
#  (here, backend/aimdo/ for cuda_offload="aimdo") and drive its __init__ through this interface
#  only:
#
#      pre_torch_init()                       - one-time setup that MUST run before `import torch`
#      available()                            - True once comfy-aimdo has initialised (CUDA-only)
#      supports(engine)                       - True if this backend can place `engine`
#      load_pipe(model, dtype, engine, ...)   - build a fully-placed diffusers pipeline, ready
#                                               to run
#      prepare(pipe) / reclaim(pipe) / release(pipe)
#                                             - optional per-generation / teardown lifecycle hooks.
#                                               A pipe from load_pipe is already runnable for a
#                                               single generation; these matter for a long-lived
#                                               host that caches + reuses the pipe across requests
#                                               (server.sh). No-ops for a pipe that needs none.
#
#  Everything GPL-derived lives in this backend/aimdo/ package (comfy_aimdo is the external dep;
#  the ComfyUI-style VBAR weight streaming in backend/aimdo/offload.py and the measured
#  placement in backend/aimdo/placement.py sit next to this file), so the calling shell scripts
#  stay GPL-free.
#
#  This package (__init__.py + offload.py + placement.py) is GPL; license text maintained
#  separately.
# =================================================================================================

import os

# Set by pre_torch_init(): True once comfy-aimdo's native library brought up a device. cpu/mps
# builds do not ship comfy-aimdo, so pre_torch_init() raises there and the caller leaves the
# backend out.
_available = False


# Per-engine diffusers classes: (PipelineCls, TransformerCls). Add engines here as their scripts
# migrate.
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


def pre_torch_init():
    """Install the comfy-aimdo CUDA hooks. MUST be called before torch is imported.

    Raises if comfy-aimdo is not installed (cpu/mps builds), so a host that wants to degrade
    gracefully should call this inside try/except and gate on available().
    """
    global _available

    import comfy_aimdo.control as ctl

    _available = bool(ctl.init())
    ctl.set_log_warning()


def available():
    """True once pre_torch_init() has brought up comfy-aimdo (CUDA-only; absent on cpu/mps
    builds)."""
    return _available


def load_pipe(model, dtype, engine, device="cuda:0", lora_files=None):
    """Resource-driven placement of `engine` (ComfyUI-style, all budgets MEASURED -- placement.py):
    full_resident when the transformer fits VRAM (== ComfyUI full_load, a big GPU), else streamed
    through the merged VBAR offloader (offload) -- pinned HostBuffer up to the measured RAM
    budget + reserved cast buffers + measured prefetch + file overflow + VBAR residency. No
    hardcoded VRAM/RAM sizes (mem_get_info / psutil). Returns a fully-placed pipeline, ready to
    run.

    lora_files: list of LoRA specs (path or (path, weight)) applied on-cast by the streamer (qwen's
    lightning(+angles)); their deltas accumulate. None for engines/paths without LoRA.
    """
    from . import placement

    plan = placement.probe(model)
    print(placement.describe(plan), flush=True)

    if plan["mode"] == "full_resident":
        # Whole transformer fits VRAM: load resident, no offloader (== ComfyUI full_load). Reached
        # only by small models on big GPUs -- qwen never fits, so its LoRA path is not skipped
        # here.
        PipelineCls, _ = _classes(engine)

        p = PipelineCls.from_pretrained(model, torch_dtype=dtype, use_safetensors=True,
                                        low_cpu_mem_usage=True, local_files_only=True)
        p.to("cuda")
    else:
        p = _load_streamed(model, dtype, engine, lora_files)   # unified merged VBAR offloader

    # NOTE: This might improve performances.
    p.safety_checker = lambda images, **kwargs: (images, [False] * len(images))

    return p


def _load_streamed(model, dtype, engine, lora_files):
    # Unified aimdo loader for all engines via the merged VBAR offloader (offload):
    # meta-load the transformer (no pageable copy of its weights), then stream its Linear weights
    # to the GPU per forward -- pinned into a HostBuffer up to the measured RAM budget
    # (all-or-nothing by fits-RAM, truly-async H2D), VBAR residency for the hot set, reserved cast
    # buffers + measured prefetch, file overflow beyond RAM. == ComfyUI's DynamicVram cast path
    # [CU ops.py cast_bias_weight L281, model_patcher.py dynamic load L1779]. VAE stays
    # GPU-resident.
    #
    # Uniform across engines (copy ComfyUI: every model is a managed dynamic VBAR -- see below):
    # the transformer AND the text encoder are each a managed Offloader with its own VBAR,
    # coordinated by the manager at load boundaries. Only the class table + qwen's on-cast
    # lightning(+angles) LoRA (passed in as lora_files) differ per engine. The TE is always
    # streamed from disk via from_module (host RAM = page cache only); qwen's Qwen2.5-VL loader
    # rewrites the checkpoint keys and its vision-tower conv3d must stay on CUDA, both handled
    # model-agnostically by the shared from_module path.
    from . import offload
    from . import placement

    from accelerate import init_empty_weights

    PipelineCls, Transformer = _classes(engine)

    tdir = os.path.join(model, "transformer")
    cfg = Transformer.load_config(tdir)

    with init_empty_weights():
        meta_transformer = Transformer.from_config(cfg).to(dtype)

    # Build the pipe FIRST so placement.pin_budget() is measured AFTER the text encoder is in host
    # RAM (conservative -- the >RAM qwen case then correctly pins nothing). from_pretrained accepts
    # the meta transformer as-is (never loads its weights), so the offloader hooks it afterwards.
    p = PipelineCls.from_pretrained(
        model,
        transformer=meta_transformer,
        torch_dtype=dtype,
        use_safetensors=True,
        low_cpu_mem_usage=True,
        local_files_only=True
    )

    p.vae.to("cuda")
    # VAE stays full-frame by default: slicing is a no-op at batch=1, and a full-frame decode fits
    # the VRAM ceiling at <=1024x768 with cudaMallocAsync (validated, qwen 1024x768 aimdo). This is
    # only the default -- the caller can still enable VAE tiling (its slicing="slice" knob) for
    # much higher resolutions, where a full-frame decode would otherwise spill to system RAM
    # (aimdo.md).

    # On-cast LoRA (qwen): always lightning; add angles @0.9 for <sks> prompts. The caller resolves
    # the specs (which file at what weight is app config) and passes them as lora_files; each spec
    # is (path, weight) and their deltas accumulate. None for flux2/z-image.

    # Every model is a managed dynamic VBAR -- copy ComfyUI, where under aimdo the diffusion
    # transformer AND the text encoder are BOTH ModelPatcherDynamic with their own VBAR
    # [CU model_patcher.py L1743 ModelVBAR, L1799 _vbar_get], both registered in
    # current_loaded_models [CU model_management.py L945], coordinated at each load boundary
    # [CU model_management.py L849 load_models_gpu]. There is no per-model "manage" flag in ComfyUI
    # (registration is unconditional); manage=True everywhere mirrors that. The manager then
    # reclaims the inactive model's GPU footprint at the active one's load boundary (==
    # partially_unload -> vbar.free_memory + restore_loaded_backups
    # [CU model_patcher.py L1937-1941]) -- made EXPLICIT here rather than via comfy-aimdo's
    # on-demand cross-vbar eviction, which underperforms on this box (the deliberate
    # dynamic-for-dynamic no-unload is [CU model_management.py L824-828]; see PLAN-te-streaming.md
    # "EMPIRICAL").
    transformer_offloader = offload.Offloader(
        meta_transformer, tdir, "cuda:0", lora_files=lora_files or None,
        pin_budget=placement.pin_budget(), manage=True)

    # Text encoder: a managed dynamic VBAR for EVERY engine (== the TE being a ModelPatcherDynamic
    # under aimdo [CU model_management.py text_encoder_initial_device L1138 -> offload device,
    # streamed via its VBAR]). from_module is the universal streamer: _match_disk_keys maps each
    # live nn.Linear name -> its disk key model-agnostically (longest common segment-suffix +
    # shape/dtype), so it works whether the loader rewrites the checkpoint keys (qwen Qwen2.5-VL)
    # or they already match (flux2/z-image Qwen3). The big Linears stream from
    # text_encoder/model.safetensors -> host RAM stays page-cache only (drops the ~7.5 GB
    # HostBuffer pin that pushed z-image to 96% RAM); the small resident params (token embedding,
    # norms; qwen's vision-tower conv3d) go on CUDA so the encode runs on GPU. The tied embedding
    # stays resident (it is not a streamed Linear) and lm_head is never streamed ("lm_head" not in
    # name), so the flux2 lm_head<->embedding tie is preserved without an explicit tie_weights().
    # manage=True -> the manager releases the TE's GPU footprint at the transformer's denoise
    # boundary and reloads it for the next encode (ComfyUI coexisting dynamic models
    # [CU model_patcher.py L1937-1941]), so host RAM is bounded and the transformer gets denoise
    # VRAM.
    encoder_offloader = offload.Offloader(
        p.text_encoder, tdir=os.path.join(model, "text_encoder"), from_module=True,
        device="cuda:0", manage=True)
    # prepare() reloads the TE to GPU BEFORE the pipeline reads self._execution_device (computed at
    # __call__ start and passed into encode_prompt), else input_ids land on CPU while the reloaded
    # weight is on CUDA == ComfyUI load_models_gpu([te]) before the model runs
    # [CU model_management.py L849]. On a FRESH pipe nothing is released yet (both models
    # resident), so a single-shot run works WITHOUT prepare(); it only matters once a prior
    # generation has left the TE released -- i.e. a host that reuses the cached pipe.
    p._aimdo_encoder = encoder_offloader

    # Keep both offloaders reachable so release() can free() them (VBARs/HostBuffers + file handles
    # + the offloader<->module hook cycle); otherwise they leak via that cycle.
    p._aimdo_offloaders = [transformer_offloader, encoder_offloader]

    return p


def prepare(pipe):
    """Per-generation load boundary: reload the managed text encoder to GPU BEFORE pipe() reads its
    _execution_device, so prompt/input tensors land on CUDA with the reloaded weights (== ComfyUI
    load_models_gpu([te]) before the model runs [CU model_management.py L849]). No-op for a pipe
    with no managed encoder (e.g. a full_resident pipe). Only needed when the pipe is reused across
    generations; a fresh pipe from load_pipe runs once without it."""
    enc = getattr(pipe, "_aimdo_encoder", None)

    if enc is not None:
        enc.activate()


def reclaim(pipe):
    """Per-generation housekeeping -- the port of ComfyUI's per-execution finally
    [CU execution.py L543-549, gated on aimdo_enabled]: return torch's retained allocator pool to
    the driver + drop the VBAR resident floors, so the NEXT generation's activations don't starve
    the persistent VBAR weights (a SEPARATE CUDA VMM [AI plat.h three_stooges L182]) into
    re-streaming every layer (measured ~16 s/ step vs ~2). See PLAN-aimdo-pool-reclaim.md and
    offload.reclaim_between_runs. No-op for a pipe with no offloaders (no VBAR competing
    with the torch pool)."""
    if getattr(pipe, "_aimdo_offloaders", None):
        from . import offload

        offload.reclaim_between_runs()


def release(pipe):
    """Tear down any offloaders before the pipe is dropped: free() cudaHostUnregisters every staged
    weight region and decommits the host buffer. This reclaims the host RAM (a plain del would leak
    it: the offloader<->module reference cycle keeps the buffer's __del__ from running) AND leaves
    the host addresses clean so the NEXT model can build in the same process (without it, a rebuild
    faults with "already mapped"). See offload.Offloader.free. No-op for a pipe with
    no offloaders."""
    import traceback

    for offloader in getattr(pipe, "_aimdo_offloaders", []):
        try:
            offloader.free()
        except Exception:
            print(traceback.format_exc(), flush=True)
