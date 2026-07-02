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

from . import comfy  # noqa: F401  establishes the top-level `comfy` alias (see aimdo/comfy/__init__.py)

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
    (normalized_shape, eps) that its original class didn't expose under those names.

    elementwise_affine is set too: it isn't read by F.rms_norm, but torch.nn.RMSNorm.extra_repr()
    reads it, and ModelPatcher.load() reprs each module in a debug-log line -- that .format() runs
    eagerly regardless of log level, so a missing key raises KeyError mid-load (native path). The
    module always carries a weight here, so affine is True."""
    if not hasattr(module, "normalized_shape"):
        module.normalized_shape = tuple(module.weight.shape)
    if not hasattr(module, "eps") or module.eps is None:
        module.eps = (getattr(module, "variance_epsilon", None)
                      or getattr(module, "epsilon", None) or 1e-6)
    if not hasattr(module, "elementwise_affine"):
        module.elementwise_affine = True


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
# VBAR (CUDA only). Flipping comfy.memory_management.aimdo_enabled turns on ComfyUI's own dynamic
# path: ModelPatcherDynamic streams each RAM-resident weight to VRAM per forward and keeps only what
# fits GPU-resident (partial residency, sized by load_models_gpu from live free VRAM), so a model
# larger than VRAM still runs -- exactly as ComfyUI offloads a UNet. On CPU/MPS this stays off and the
# native cast_to path runs.
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


def fits_in_ram(model_dir):
    """True when a component's weights fit in host RAM, so they can stream to the GPU from RAM
    (ComfyUI's fast path) instead of disk->VRAM. GPU/card-agnostic: the size is the on-disk shard
    total, the budget is ComfyUI's own get_total_memory(cpu). Only one big component is hot at a time
    (encode, then sample), so it must fit alone; ~10% is reserved for the OS. A component bigger than
    this streams disk->VRAM via load_streamed / assign_streamed_weights (comfy-aimdo VBAR), which
    never materialises it in RAM -- the only way to run a model larger than host RAM."""
    try:
        size = sum(os.path.getsize(s) for s in _shards(model_dir))
    except OSError:
        return True  # can't size it -> assume RAM path; never force disk on a stat error
    total_ram = mm.get_total_memory(torch.device("cpu"))
    return size <= total_ram * 0.9


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


def _match_by_suffix(live_name, live_tensor, disk_by_shape):
    """Pick the disk key for `live_name` among same-shape/dtype candidates by longest common dotted
    suffix; return it only if that best match is unique (mirrors v1's _match_disk_keys). Handles
    checkpoints whose keys were renamed by a prefix (e.g. transformers' `model.` -> `language_model.`
    for Qwen2.5-VL) where a direct name lookup misses."""
    cands = disk_by_shape.get((tuple(live_tensor.shape), live_tensor.dtype))
    if not cands:
        return None
    live_parts = live_name.split(".")

    def common_suffix(disk_name):
        dp = disk_name.split(".")
        n = 0
        while n < len(dp) and n < len(live_parts) and dp[-1 - n] == live_parts[-1 - n]:
            n += 1
        return n

    scored = sorted(((common_suffix(dk), dk) for dk in cands), reverse=True)
    if scored and (len(scored) == 1 or scored[0][0] > scored[1][0]):
        return scored[0][1]
    return None


def assign_streamed_weights(model, model_dir):
    """Replace `model`'s weights with mmap-backed, file-sliced tensors from its .safetensors shard(s)
    (via ComfyUI's own load_safetensors), so the VBAR path can fault each straight from disk. Matches
    disk keys to model.named_parameters() by name; for any that don't match directly (checkpoints
    with renamed prefixes, e.g. the Qwen2.5-VL text encoder), falls back to structural shape+suffix
    matching. Returns the list of disk keys that still found no home."""
    import comfy.utils as cu

    sd = {}
    for shard in _shards(model_dir):
        part, _meta = cu.load_safetensors(shard)
        sd.update(part)

    own = {n: p for n, p in model.named_parameters()}
    own.update({n: b for n, b in model.named_buffers()})

    used = set()
    # 1) direct name matches.
    for name, tensor in sd.items():
        if name in own:
            cu.set_attr_param(model, name, tensor)  # keeps the file-sliced storage
            used.add(name)

    # 2) structural fallback for the live params still holding their from_pretrained (materialized)
    #    weights: match by shape + longest-common dotted suffix among the leftover disk keys.
    leftover = {k: v for k, v in sd.items() if k not in used}
    if leftover:
        disk_by_shape = {}
        for k, v in leftover.items():
            disk_by_shape.setdefault((tuple(v.shape), v.dtype), []).append(k)
        for name, param in own.items():
            if name in used:
                continue
            dk = _match_by_suffix(name, param, disk_by_shape)
            if dk is not None and dk in leftover:
                cu.set_attr_param(model, name, leftover[dk])
                used.add(dk)
                disk_by_shape[(tuple(param.shape), param.dtype)].remove(dk)

    return [k for k in sd if k not in used]


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


_sdpa_patched = False


def use_comfy_attention(model=None):
    """Route diffusers' attention through a copy of ComfyUI's own `comfy.ops.scaled_dot_product_attention`
    (comfy/ops.py:39-64), so our SDPA behaves EXACTLY like ComfyUI's -- not an approximation.

    ComfyUI, on Windows+CUDA with a recent torch, forces the SDPA backend priority
    [CUDNN, FLASH, EFFICIENT, MATH] -- but ONLY per call and ONLY for large inputs
    (`q.nelement() >= 1024*128`); small attentions fall through to torch's default backend (cuDNN is
    slower there). We reproduce that verbatim. diffusers calls `torch.nn.functional.scaled_dot_product_attention`
    by attribute (diffusers/models/attention_dispatch.py), so we patch that one name, once and
    process-global (idempotent), delegating to the saved original -- no self-recursion. `model` is
    accepted only for signature parity with the other patchers; the patch is global, as ComfyUI's is.
    No-op off Windows/CUDA or on a torch without set_priority (exactly ComfyUI's own guard)."""
    global _sdpa_patched
    if _sdpa_patched:
        return True

    import torch.nn.functional as F
    import comfy.model_management as mm

    try:
        if not (torch.cuda.is_available() and getattr(mm, "WINDOWS", False)):
            return False
        from torch.nn.attention import SDPBackend, sdpa_kernel
        import inspect
        if "set_priority" not in inspect.signature(sdpa_kernel).parameters:
            return False
    except Exception:
        return False

    # comfy/ops.py builds [FLASH, EFFICIENT, MATH] then inserts CUDNN at the front (ops.py:48-54).
    priority = [SDPBackend.CUDNN_ATTENTION, SDPBackend.FLASH_ATTENTION,
                SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
    orig = F.scaled_dot_product_attention

    def scaled_dot_product_attention(*args, **kwargs):
        # comfy/ops.py:56-60 verbatim (the branch + sdpa_kernel), only the arg handling differs:
        # ComfyUI's models call comfy.ops.scaled_dot_product_attention(q, k, v, ...) POSITIONALLY,
        # but diffusers calls torch's F.sdpa by KEYWORD (query=/key=/value=, attention_dispatch.py),
        # so we take the query tensor from args[0] or kwargs["query"] and forward the call untouched.
        q = args[0] if args else kwargs.get("query")
        if q is None or q.nelement() < 1024 * 128:  # comfy: small inputs -> default backend
            return orig(*args, **kwargs)
        with sdpa_kernel(priority, set_priority=True):
            return orig(*args, **kwargs)

    F.scaled_dot_product_attention = scaled_dot_product_attention
    _sdpa_patched = True
    return True


# --------------------------------------------------------------------------------------------------
# comfy-kitchen fused RoPE. ComfyUI runs its flux RoPE through comfy_kitchen's fused kernel
# (comfy/ldm/flux/math.py:apply_rope1 -> comfy.quant_ops.ck.apply_rope1) on the normal bf16 path. We
# use diffusers models, so we route diffusers' rope through the SAME kernel. No kernel is
# reimplemented: we import comfy_kitchen (via the vendored, already-backend-configured
# comfy.quant_ops.ck) and only add a tiny format shim between diffusers' (cos, sin) and ck's
# freqs_cis. Like ComfyUI, we do NOT device-gate -- ck's own registry picks the backend per device
# (CUDA kernel on cu130+, eager on cpu/mps).
# --------------------------------------------------------------------------------------------------

_KITCHEN_ROPE = os.environ.get("AIMDO_KITCHEN_ROPE", "1") != "0"


def _build_freqs_cis(cos, sin):
    """Pack diffusers' (cos, sin) -- each frequency duplicated across its adjacent pair -- into
    comfy_kitchen / ComfyUI's rope() layout [[cos, -sin], [sin, cos]] of shape [..., D/2, 2, 2], in
    float32 (matching diffusers' float accumulation). Reproduces comfy/ldm/flux/math.py:rope() from
    precomputed cos/sin. cos/sin arrive already broadcast to the input rank (via the sequence_dim
    None-insertion), so freqs_cis broadcasts against ck.apply_rope1's reshaped x directly."""
    c = cos[..., 0::2].float()
    s = sin[..., 0::2].float()
    return torch.stack([c, -s, s, c], dim=-1).reshape(*c.shape, 2, 2)


def _make_kitchen_rope(orig):
    """Build the replacement for a diffusers `apply_rotary_emb`. `orig` is the module's own binding
    (the fall-through target). comfy_kitchen is resolved LAZILY on the first *matching* call, so a
    module whose calls never match (or a model we patch but never run) imports nothing."""
    ck_state = {"ck": None, "resolved": False}

    def _ck():
        if not ck_state["resolved"]:
            ck_state["resolved"] = True
            try:
                import comfy.quant_ops as qo  # configures ck backends exactly as ComfyUI does
                if getattr(qo, "_CK_AVAILABLE", False):
                    ck_state["ck"] = qo.ck
                    print("aimdo: kitchen rope engaged; comfy_kitchen backends=%s"
                          % qo.ck.list_backends(), flush=True)
            except Exception:
                ck_state["ck"] = None
        return ck_state["ck"]

    def apply_rotary_emb(x, freqs_cis, use_real=True, use_real_unbind_dim=-1, sequence_dim=2):
        # Only the flux interleaved convention on a (cos, sin) pair, at inference; else native.
        if (not use_real or use_real_unbind_dim != -1 or torch.is_grad_enabled()
                or sequence_dim not in (1, 2) or not isinstance(freqs_cis, (tuple, list))
                or x.dtype not in (torch.bfloat16, torch.float16)):
            return orig(x, freqs_cis, use_real, use_real_unbind_dim, sequence_dim)

        ck = _ck()
        if ck is None:
            return orig(x, freqs_cis, use_real, use_real_unbind_dim, sequence_dim)

        cos, sin = freqs_cis  # [S, D]
        # Same broadcast diffusers applies (embeddings.py), so freqs_cis lands on the right axis.
        if sequence_dim == 2:
            cos, sin = cos[None, None, :, :], sin[None, None, :, :]
        else:
            cos, sin = cos[None, :, None, :], sin[None, :, None, :]

        fc = _build_freqs_cis(cos.to(x.device), sin.to(x.device))
        return ck.apply_rope1(x, fc)

    apply_rotary_emb._aimdo_kitchen = True
    return apply_rotary_emb


def use_kitchen_rope(model):
    """Route a diffusers transformer's `apply_rotary_emb` through comfy_kitchen's fused `apply_rope1`
    -- the same kernel ComfyUI uses for flux RoPE (comfy/ldm/flux/math.py). ComfyUI wires this into
    each model's own code; the diffusers analogue is that each transformer does
    `from ..embeddings import apply_rotary_emb`, binding the name BY VALUE in its own module at import.
    So we patch that name in the TRANSFORMER'S module (`type(model).__module__`) -- patching
    `embeddings` alone would not reach an already-imported binding.

    Fully model-scoped and safe:
      * Only the model's own module is touched -- other engines' modules are never affected, so a
        long-lived server that runs flux2 then z-image is fine.
      * A model whose module has no such symbol (e.g. z-image, whose rope is a complex-valued nested
        function, not a `from ..embeddings import apply_rotary_emb`) is skipped -- nothing is patched,
        numerics byte-for-byte unchanged.
      * Per call, anything outside the flux interleaved convention falls through to the original; ck
        is imported lazily only on a matching call. No device gate (ck's registry picks per device).
    No-op when AIMDO_KITCHEN_ROPE=0. Idempotent (won't double-wrap). Returns True if it patched."""
    if not _KITCHEN_ROPE:
        return False

    import sys

    mod = sys.modules.get(type(model).__module__)
    orig = getattr(mod, "apply_rotary_emb", None)

    if orig is None or getattr(orig, "_aimdo_kitchen", False):
        return False  # no module-level rope symbol here (e.g. z-image), or already patched

    mod.apply_rotary_emb = _make_kitchen_rope(orig)
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


_LORA_SUFFIXES = (".lora_down.weight", ".lora_up.weight", ".lora_A.weight", ".lora_B.weight",
                  ".alpha", ".dora_scale", ".diff", ".diff_b")


def add_lora(patcher, lora_specs):
    """Apply LoRAs on-cast, the ComfyUI way: build the {lora_base_key -> model_weight_key} map, hand
    it to comfy.lora.load_lora (which uses the vendored weight_adapter classes to read lora_up/down
    or lora_A/B + alpha), and register the result via ModelPatcher.add_patches. The patches become
    LowVramPatch weight_functions that comfy.lora.calculate_weight applies while each weight streams
    in -- no fusing, no extra resident copy. lora_specs is [(path, strength), ...]."""
    import comfy.utils as cu
    import comfy.lora as clora

    model_keys = set(n for n, _ in patcher.model.named_parameters())
    applied = 0
    for path, strength in lora_specs:
        sd = cu.load_torch_file(path, safe_load=True)
        bases = set()
        for k in sd:
            for suf in _LORA_SUFFIXES:
                if k.endswith(suf):
                    bases.add(k[:-len(suf)])
                    break
        to_load = {}
        for base in bases:
            mk = base
            for pre in ("transformer.", "diffusion_model.", "lora_unet_"):
                if mk.startswith(pre):
                    mk = mk[len(pre):]
                    break
            mk = mk + ".weight"
            if mk in model_keys:
                to_load[base] = mk
        patch_dict = clora.load_lora(sd, to_load, log_missing=False)
        patcher.add_patches(patch_dict, strength)
        applied += len(patch_dict)
    return applied


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


def install_encode_cache(pipe):
    """Cache the text encoder's output by prompt -- a copy of ComfyUI's default node cache
    (comfy_execution/caching.py::RAMPressureCache): keep every result keyed by the encode call's full
    inputs, HOLD IT ON CPU, and evict under host-RAM pressure. Reuses ComfyUI's own memory reading:
    mm.get_free_memory(cpu) below min(10GB, max(2GB, 10% of RAM)) triggers eviction, worst entry first
    by 1.3**generation_age * cached_bytes (LRU tiebreak) -- RAMPressureCache.ram_release verbatim minus
    the node-graph bookkeeping. ComfyUI holds CONDITIONING on the CPU for exactly this; a diffusers
    pipeline encodes on the compute device, so we stash a CPU copy and move it back on a hit (which
    also frees the VRAM the embeddings would otherwise pin, and lets them count as RAM for eviction).

    diffusers encodes inside __call__ via self.encode_prompt; wrapping that one method means a repeated
    prompt skips the encoder's forward entirely -- and since load_models_gpu is idempotent the un-run
    encoder is never streamed in, leaving the diffusion model resident (why ComfyUI's warm runs spend
    ~0s on a cached encode). The key is the call's FULL (args, kwargs), like ComfyUI keying a node on
    its whole input signature: an argument that isn't cleanly hashable -- a tensor/image, e.g. an
    image-conditioned edit encode -- makes the call uncacheable, so it never gets a false hit. No-op if
    the pipe has no encode_prompt or the cache is already installed."""
    real = getattr(pipe, "encode_prompt", None)

    if real is None or getattr(pipe, "_encode_cache_installed", False):
        return

    import time
    import bisect

    OLD_OOM_MULT = 1.3   # comfy RAM_CACHE_OLD_WORKFLOW_OOM_MULTIPLIER
    BASE_USAGE = 0.05    # comfy RAM_CACHE_DEFAULT_RAM_USAGE (keeps zero-size entries LRU-ordered)
    cpu = torch.device("cpu")
    target = min(10 * 1024 ** 3, max(2 * 1024 ** 3, mm.get_total_memory(cpu) * 0.10))

    uncacheable = object()

    def norm(v):
        if v is None or isinstance(v, (str, int, float, bool)):
            return v
        if isinstance(v, (torch.device, torch.dtype)):
            return str(v)
        if isinstance(v, (list, tuple)):
            parts = tuple(norm(x) for x in v)
            return uncacheable if any(p is uncacheable for p in parts) else parts
        return uncacheable

    def key_of(args, kwargs):
        a = tuple(norm(v) for v in args)
        kw = tuple((k, norm(kwargs[k])) for k in sorted(kwargs))
        if any(x is uncacheable for x in a) or any(v is uncacheable for _, v in kw):
            return None
        return (a, kw)

    def move(obj, device):
        if isinstance(obj, torch.Tensor):
            return obj.to(device)
        if isinstance(obj, (list, tuple)):
            return type(obj)(move(o, device) for o in obj)
        return obj

    def nbytes(obj):
        if isinstance(obj, torch.Tensor):
            return obj.numel() * obj.element_size()
        if isinstance(obj, (list, tuple)):
            return sum(nbytes(o) for o in obj)
        return 0

    cache = {}   # key -> [cpu_value, bytes, timestamp, generation]
    gen = [0]

    def ram_release():
        if mm.get_free_memory(cpu) >= target:
            return
        scored = []
        for k, (_, sz, ts, g) in cache.items():
            if g == gen[0]:                    # comfy: never evict this generation's own entries
                continue
            bisect.insort(scored, ((OLD_OOM_MULT ** (gen[0] - g)) * (sz + BASE_USAGE), ts, k))
        while scored and mm.get_free_memory(cpu) < target:
            cache.pop(scored.pop()[2], None)   # highest oom_score first

    def cached_encode(*args, **kwargs):
        key = key_of(args, kwargs)

        if key is None:                        # tensor/image arg -> not safely cacheable
            return real(*args, **kwargs)

        gen[0] += 1
        hit = cache.get(key)

        if hit is not None:
            cpu_value, sz, _, _ = hit
            cache[key] = [cpu_value, sz, time.time(), gen[0]]
            device = getattr(pipe, "_execution_device", None) or pipe.device
            return move(cpu_value, device)

        out = real(*args, **kwargs)
        cache[key] = [move(out, cpu), nbytes(out), time.time(), gen[0]]
        ram_release()

        return out

    pipe.encode_prompt = cached_encode
    pipe._encode_cache_installed = True


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
