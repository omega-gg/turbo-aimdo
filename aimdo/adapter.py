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
#   The thin middleman between turboCLI's diffusers pipelines and the vendored ComfyUI offloading
#   subsystem (aimdo/comfy/). It contains the ONLY real logic in the package; everything under
#   aimdo/comfy/ is a verbatim ComfyUI snapshot. The bridge is two ideas:
#
#     1. comfy_ize(model): re-class each vanilla torch leaf module (Linear/Conv/Norm/Embedding) to
#        its ComfyUI `disable_weight_init.*` counterpart. Those classes ARE `torch.nn.X` +
#        CastWeightBiasOp, so their forward routes through ComfyUI's device-agnostic cast path
#        (forward_comfy_cast_weights -> cast_bias_weight -> cast_to) exactly when ModelPatcher flags
#        the module for offload -- and computes identically to the original op otherwise.
#
#     2. build_patcher(model): wrap the comfy-ized module in a ComfyUI ModelPatcher. From there the
#        seam drives ComfyUI's own load_models_gpu / free_memory to stream weights CPU<->device per
#        forward, the same way ComfyUI offloads a UNet -- on CPU, CUDA or MPS.
#
#   comfy_aimdo's CUDA-only VBAR path is left dormant (aimdo_enabled False) until Phase D.
#
#==================================================================================================

import aimdo.comfy  # noqa: F401  establishes the top-level `comfy` alias (see aimdo/comfy/__init__.py)

import os

import torch

import comfy.model_management as mm
import comfy.model_patcher as model_patcher
import comfy.ops as ops


# --------------------------------------------------------------------------------------------------
# Op mapping: torch leaf type -> ComfyUI disable_weight_init counterpart (a torch.nn.X subclass that
# also mixes in CastWeightBiasOp). Checked in order; the first isinstance() match wins. Every diffuser
# transformer/text-encoder leaf that carries an offloadable weight is one of these.
# --------------------------------------------------------------------------------------------------
_dwi = ops.disable_weight_init

_OP_MAP = [
    (torch.nn.Linear,    _dwi.Linear),
    (torch.nn.Conv1d,    _dwi.Conv1d),
    (torch.nn.Conv2d,    _dwi.Conv2d),
    (torch.nn.Conv3d,    _dwi.Conv3d),
    (torch.nn.GroupNorm, _dwi.GroupNorm),
    (torch.nn.LayerNorm, _dwi.LayerNorm),
    (torch.nn.Embedding, _dwi.Embedding),
]

# torch.nn.RMSNorm exists only on recent torch; add it when present (diffusers uses it widely).
if hasattr(torch.nn, "RMSNorm") and hasattr(_dwi, "RMSNorm"):
    _OP_MAP.append((torch.nn.RMSNorm, _dwi.RMSNorm))


def _comfy_class_for(module):
    """The disable_weight_init class to re-class `module` into, or None if it isn't an offloadable
    leaf op. Skips modules that already are CastWeightBiasOp (idempotent)."""
    if isinstance(module, ops.CastWeightBiasOp):
        return None
    for torch_cls, dwi_cls in _OP_MAP:
        if isinstance(module, torch_cls):
            return dwi_cls
    # Custom RMSNorm classes (diffusers/transformers write their own, e.g. Qwen3RMSNorm, not
    # torch.nn.RMSNorm) run an eager mul/rsqrt that is ~3.5x slower than ComfyUI's fused path.
    # ComfyUI's own disable_weight_init.RMSNorm calls torch.nn.functional.rms_norm; route these
    # through the SAME vendored class so the kernel (and behavior) matches ComfyUI exactly.
    if hasattr(_dwi, "RMSNorm") and type(module).__name__.endswith("RMSNorm") \
            and getattr(module, "weight", None) is not None:
        return _dwi.RMSNorm
    return None


def _prep_rmsnorm(module):
    """Give a re-classed custom RMSNorm the attributes torch.nn.RMSNorm / F.rms_norm read
    (normalized_shape, eps) that its original class didn't expose under those names."""
    if not hasattr(module, "normalized_shape"):
        module.normalized_shape = tuple(module.weight.shape)
    if not hasattr(module, "eps") or module.eps is None:
        module.eps = (getattr(module, "variance_epsilon", None)
                      or getattr(module, "epsilon", None) or 1e-6)


def comfy_ize(model):
    """Re-class every offloadable leaf of `model` in place so its forward routes through ComfyUI's
    cast path when flagged. Injects exactly the attributes ModelPatcher.load()/cast_bias_weight read
    (comfy_cast_weights, weight_function, bias_function, and bias=None for weight-only ops). Returns
    the number of modules converted.

    Re-classing to ComfyUI's own class (rather than a synthesized mixin) keeps true 1:1 parity: the
    running forward IS disable_weight_init.<Op>.forward. The lazy-init __init__ of those classes is
    bypassed here (we mutate existing instances that already hold their weights)."""
    count = 0
    for _name, m in model.named_modules():
        dwi_cls = _comfy_class_for(m)
        if dwi_cls is None:
            continue

        # Custom RMSNorm needs normalized_shape/eps before re-classing so F.rms_norm can read them.
        if dwi_cls is _dwi.RMSNorm and not isinstance(m, torch.nn.RMSNorm):
            _prep_rmsnorm(m)

        # Re-class the live instance. torch.nn.X subclasses share a compatible object layout, so
        # __class__ assignment is valid; diffusers custom subclasses that add only config (no forward
        # override that matters for weight application) keep working through the base op.
        m.__class__ = dwi_cls

        # cast_bias_weight always reads s.bias; weight-only ops (Embedding/RMSNorm) have none. Mirror
        # what disable_weight_init.__init__ would have set.
        if not hasattr(m, "bias"):
            m.bias = None

        # Instance-level (never the shared CastWeightBiasOp class lists) so per-module offload flags
        # don't alias. ModelPatcher.load() overwrites these when it decides to offload the module.
        m.comfy_cast_weights = False
        m.weight_function = []
        m.bias_function = []
        count += 1
    return count


def make_patchable(model):
    """ComfyUI's ModelPatcher assigns `self.model.device = ...`, but diffusers ModelMixin and HF
    PreTrainedModel expose `device` as a READ-ONLY property. Shadow it with a plain, settable class
    attribute via a one-off subclass so the assignment sticks (its value -- the load device -- is
    what those models would report anyway once loaded). No-op if `device` is already settable."""
    cls = type(model)
    if isinstance(getattr(cls, "device", None), property):
        # Masquerade as the parent for str(cls): transformers 5.x keys its output-capture registry
        # on `str(self.__class__)` (_CAN_RECORD_REGISTRY.get(str(self.__class__))), so a subclass
        # with a new class string silently loses output_hidden_states. Copying __module__/__qualname__
        # makes str(subclass) == str(cls), so the lookup (and hidden-state capture) keeps working.
        ns = {"device": None, "__module__": cls.__module__, "__qualname__": cls.__qualname__}
        model.__class__ = type(cls.__name__, (cls,), ns)
    return model


def keep_uncastable_resident(model, device):
    """Move every parameter/buffer that ComfyUI's offloader will NOT stream onto `device`, so it
    stays resident. The offloader only streams the weight/bias of CastWeightBiasOp leaves (the ones
    comfy_ize converted); anything else -- a custom leaf norm (transformers' Qwen3RMSNorm, forward
    `self.weight * hidden_states`), or a bare parameter/buffer hung directly on a container module
    (diffusers' `x_pad_token`, rotary caches, class tokens) -- would otherwise be left on the
    offload device by ModelPatcher.load() and blow up mid-forward with a cuda-vs-cpu mismatch.

    We walk ALL modules and move only their DIRECT (recurse=False) params/buffers, skipping
    CastWeightBiasOp modules entirely so their big weights keep streaming. These stragglers are
    almost always tiny; returns (count, bytes) for logging so a large one is visible."""
    moved = 0
    nbytes = 0
    for _name, m in model.named_modules():
        if isinstance(m, ops.CastWeightBiasOp):
            continue  # its weight/bias stream via the cast path -- leave on the offload device
        for _n, p in m.named_parameters(recurse=False):
            if p is not None and p.device != device:
                p.data = p.data.to(device)
                moved += 1
                nbytes += p.numel() * p.element_size()
        for bn, b in m.named_buffers(recurse=False):
            if b is not None and b.device != device:
                m._buffers[bn] = b.to(device)
                moved += 1
                nbytes += b.numel() * b.element_size()
    return moved, nbytes


def build_patcher(model, load_device=None, offload_device=None, size=0):
    """Wrap a comfy-ized `model` in a ComfyUI ModelPatcher. Defaults load/offload devices from
    ComfyUI's own device selection so one code path serves CPU/CUDA/MPS. `size` (bytes) seeds
    model_size(); 0 lets ModelPatcher measure it via module_size()."""
    if load_device is None:
        load_device = mm.get_torch_device()
    if offload_device is None:
        offload_device = mm.unet_offload_device()
    make_patchable(model)
    return model_patcher.ModelPatcher(model, load_device=load_device,
                                      offload_device=offload_device, size=size)


# --------------------------------------------------------------------------------------------------
# Optional VBAR acceleration (Phase D, CUDA only). Flipping comfy.memory_management.aimdo_enabled
# turns on ComfyUI's own dynamic path: ModelPatcherDynamic allocates a comfy-aimdo ModelVBAR slot per
# castable module and streams each weight disk->VRAM per forward via TensorFileSlice, so a model far
# larger than VRAM (or VRAM+RAM) still runs. The weights must be mmap + file-sliced (not real CPU
# tensors) for this to pay off; load_streamed() below wires that up using ComfyUI's own
# load_safetensors(). On CPU/MPS this stays off and the native cast_to path runs.
# --------------------------------------------------------------------------------------------------
def enable_vbar(device):
    """Flip aimdo_enabled so the vendored VBAR dynamic path engages. CUDA only. comfy-aimdo's global
    init (allocator hooks) must have run in the seam's pre_torch_init() BEFORE torch was imported;
    here -- after torch, with the device known -- we do the per-device init (init_device, like v1)
    that ModelVBAR needs, then set the flag. Returns True when VBAR is active, False (-> native path)
    off CUDA or if the device init fails."""
    dev = str(device).lower()
    if dev.split(":")[0] != "cuda":
        return False
    index = int(dev.split(":")[1]) if ":" in dev else 0
    try:
        import comfy_aimdo.control as ctl
        if not ctl.devctxs:
            ctl.init_device(index)
    except Exception:
        return False
    import comfy.memory_management as memm
    memm.aimdo_enabled = True
    return True


def _shards(model_dir):
    """The .safetensors shard paths for a diffusers component dir, honouring a .index.json if present
    (mirrors v1's _offsets shard discovery)."""
    import json
    idx = os.path.join(model_dir, "diffusion_pytorch_model.safetensors.index.json")
    if os.path.exists(idx):
        with open(idx) as f:
            names = sorted(set(json.load(f)["weight_map"].values()))
        return [os.path.join(model_dir, n) for n in names]
    return [os.path.join(model_dir, n) for n in sorted(os.listdir(model_dir))
            if n.endswith(".safetensors")]


def assign_streamed_weights(model, model_dir):
    """Replace `model`'s weights with mmap-backed, file-sliced tensors from its .safetensors shard(s)
    (via ComfyUI's own load_safetensors), so the VBAR path can fault each straight from disk. Assumes
    disk keys match model.named_parameters() (true for diffusers transformers and standard
    transformers text encoders alike). Returns the list of disk keys with no matching param."""
    import comfy.utils as cu

    sd = {}
    for shard in _shards(model_dir):
        part, _meta = cu.load_safetensors(shard)
        sd.update(part)

    own = set(n for n, _ in model.named_parameters())
    own |= set(n for n, _ in model.named_buffers())
    missing = []
    for name, tensor in sd.items():
        if name not in own:
            missing.append(name)
            continue
        cu.set_attr_param(model, name, tensor)  # wraps in Parameter; keeps the file-sliced storage
    return missing


def load_streamed(model_cls, model_dir, dtype):
    """Build a diffusers model whose weights stream disk->VRAM through the VBAR path: meta-load the
    module (no weight RAM), comfy-ize it, then assign file-sliced weights. Returns (model, missing)."""
    from accelerate import init_empty_weights

    cfg = model_cls.load_config(model_dir)
    with init_empty_weights():
        model = model_cls.from_config(cfg).to(dtype)

    comfy_ize(model)
    missing = assign_streamed_weights(model, model_dir)
    return model, missing


def prefer_cudnn_attention(model):
    """Give a diffusers/transformers model ComfyUI's attention behavior: prioritize the cuDNN-flash
    SDPA backend (~2x the cutlass efficient kernel torch otherwise picks here).

    ComfyUI applies this inside comfy.ops.scaled_dot_product_attention via sdpa_kernel(). We can't
    just redirect torch's F.scaled_dot_product_attention to that wrapper -- the wrapper calls
    F.scaled_dot_product_attention itself, so the redirect self-recurses. So we apply the SAME
    mechanism (torch.nn.attention.sdpa_kernel with the SAME priority list comfy uses) around the
    model's forward, which every internal F.sdpa call then respects. No-op if torch lacks
    set_priority."""
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
        import inspect
        if "set_priority" not in inspect.signature(sdpa_kernel).parameters:
            return False
    except Exception:
        return False

    # Same list comfy.ops builds (ops.py: [CUDNN, FLASH, EFFICIENT, MATH]).
    priority = [SDPBackend.CUDNN_ATTENTION, SDPBackend.FLASH_ATTENTION,
                SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
    orig_forward = model.forward

    def forward(*args, **kwargs):
        with sdpa_kernel(priority, set_priority=True):
            return orig_forward(*args, **kwargs)

    model.forward = forward
    return True


def install_prefetch(model):
    """Overlap weight streaming with compute by copying ComfyUI's model_prefetch mechanism onto a
    diffusers transformer. ComfyUI's own models call prefetch_queue_pop() between transformer blocks
    so block N+1's weights stream on the offload stream while block N computes (av_model.py:913). A
    diffusers forward doesn't, so every layer stalls waiting for its weights (~30% of a VBAR step).

    We reproduce it with forward hooks: for each ModuleList of repeated blocks, build a prefetch
    queue at the transformer's forward-start and pop it before each block runs; tear the queues down
    at the transformer's forward-end. No-op unless comfy-aimdo/VBAR is active (make_prefetch_queue
    returns None off the dynamic path). Returns the number of block sequences instrumented."""
    import torch.nn as nn
    import comfy.model_prefetch as mp

    # Block sequences = top-level ModuleLists of >=2 identical-typed modules (layers + refiners).
    sequences = []
    for _name, child in model.named_children():
        if isinstance(child, nn.ModuleList) and len(child) >= 2:
            sequences.append(child)
    if not sequences:
        return 0

    state = {}  # id(block_list) -> live queue for this forward

    def _device_of(args, kwargs):
        for t in list(args) + list(kwargs.values()):
            if hasattr(t, "device"):
                return t.device
        return None

    def _start(_m, args, kwargs):
        dev = _device_of(args, kwargs)
        if dev is None:
            return
        opts = {"prefetch_dynamic_vbars": True}
        for seq in sequences:
            state[id(seq)] = mp.make_prefetch_queue(list(seq), dev, opts)

    def _block_pre(seq):
        def hook(block, args, kwargs):
            q = state.get(id(seq))
            if q is not None:
                dev = _device_of(args, kwargs)
                if dev is not None:
                    mp.prefetch_queue_pop(q, dev, block)
        return hook

    def _end(_m, args, kwargs, output):
        dev = _device_of(args, kwargs)
        for seq in sequences:
            q = state.pop(id(seq), None)
            if q is not None and dev is not None:
                mp.prefetch_queue_pop(q, dev, None)  # drain/cleanup the last prefetched block
        return output

    model.register_forward_pre_hook(_start, with_kwargs=True)
    for seq in sequences:
        for block in seq:
            block.register_forward_pre_hook(_block_pre(seq), with_kwargs=True)
    model.register_forward_hook(_end, with_kwargs=True)
    return len(sequences)


def build_dynamic_patcher(model, load_device=None, offload_device=None, size=0):
    """Wrap a streamed model in ModelPatcherDynamic (the VBAR-aware patcher). On a CPU load_device it
    transparently reroutes to a plain ModelPatcher (VBAR is GPU-only)."""
    import comfy.model_patcher as model_patcher
    if load_device is None:
        load_device = mm.get_torch_device()
    if offload_device is None:
        offload_device = mm.unet_offload_device()
    make_patchable(model)
    return model_patcher.ModelPatcherDynamic(model, load_device=load_device,
                                             offload_device=offload_device, size=size)


# --------------------------------------------------------------------------------------------------
# Device state: one knob switches ComfyUI's whole device stack. get_torch_device() / offload-device
# helpers read cpu_state, so setting it from the seam's device= arg routes load/offload to CPU, MPS
# or the GPU without touching any other code.
# --------------------------------------------------------------------------------------------------
def set_device(device):
    """Force ComfyUI's cpu_state from a turboCLI renderer string ('cpu'/'mps'/'cuda[:N]'). Returns
    the resolved torch load device."""
    dev = str(device).split(":")[0].lower()
    if dev == "cpu":
        mm.cpu_state = mm.CPUState.CPU
    elif dev == "mps":
        mm.cpu_state = mm.CPUState.MPS
    else:
        mm.cpu_state = mm.CPUState.GPU
    return mm.get_torch_device()
