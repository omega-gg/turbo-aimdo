# aimdo (v2)

GPL offload backend for the turboCLI diffusion runner. Instead of cherry-picking lines from
ComfyUI (v1, see `doc/aimdo_v1.md`), v2 **vendors ComfyUI's offloading subsystem verbatim** and adds a
**thin adapter** so turboCLI's diffusers pipelines reuse ComfyUI's own memory manager with 1:1
parity. The same code runs a model larger than VRAM by streaming its weights to VRAM per forward with
partial GPU residency — from host RAM when the model fits (ComfyUI's UNet offload), else disk→VRAM via
comfy-aimdo file-slices for a model larger than RAM.

**Style:** code and comments wrap at 99 columns.

## Layout

```
aimdo/
  __init__.py   backend seam the runner drives (unchanged signatures)
  adapter.py    the thin middleman -- the only real logic in the package
  comfy/        byte-for-byte ComfyUI offloading snapshot (see comfy/resync.md)
```

`aimdo/comfy/` is a pristine mirror of ComfyUI's `model_management.py`, `model_patcher.py`,
`ops.py`, `memory_management.py`, `model_prefetch.py`, `lora.py`, `utils.py`, etc. Aliased to the
top-level `comfy` package via a `sys.modules` entry in `comfy/__init__.py`, so the vendored
`import comfy.X` lines resolve unchanged. The **only** edits over upstream (all documented in
`comfy/resync.md`): three commented-out off-path imports, `try/except` shims for the optional
`comfy_aimdo` / `comfy_kitchen`, and a stubless `cli_args`+`options` (vendored, parses no argv).

## The seam (`__init__.py`)

The runner discovers `backend/<mode>/` and drives it through this interface only:

| fn | v2 behavior |
|---|---|
| `pre_torch_init()` | init comfy-aimdo's global CUDA hooks **before torch import** (its allocator can't hook afterwards); imports nothing that pulls torch. Optional -- absent/failed → native path. |
| `available()` | True once the vendored offloader imports (any device) |
| `supports(engine)` | flux2 / z-image / qwen-image-edit |
| `load_pipe(model, dtype, engine, device, lora_files)` | build a fully-placed diffusers pipeline (below) |
| `prepare(pipe)` | `load_models_gpu(patchers)` -- place managed models on the compute device |
| `reclaim(pipe)` | `free_memory` + `soft_empty_cache` between generations |
| `release(pipe)` | `detach` each patcher |

All GPL-derived code lives in this package; the calling runner stays GPL-free.

## Two paths, one code

- **Native (device-agnostic: CPU / CUDA / MPS).** Weights load onto the offload device (CPU,
  mmap); a `ModelPatcher` streams them to the compute device per forward via ComfyUI's cast path
  (`cast_bias_weight` → `cast_to`). Needs the model to fit RAM. `comfy_aimdo` not required.
- **VBAR (optional, CUDA).** When comfy-aimdo is present, `load_pipe` flips
  `comfy.memory_management.aimdo_enabled` and uses `ModelPatcherDynamic`, which keeps as much of the
  model GPU-resident as fits (partial residency, sized from live free VRAM) and streams the rest per
  forward. Per component (`adapter.fits_in_ram`, on-disk size vs `get_total_memory`): weights stream
  from **host RAM** when the model fits (fast path — `from_pretrained` tensors, same source as native)
  else **disk→VRAM** via comfy-aimdo file-slices (`load_streamed` / `assign_streamed_weights`) for a
  model larger than RAM. This is ComfyUI's "dynamic VRAM loading" path.

The seam picks VBAR automatically on CUDA-with-comfy-aimdo, native otherwise. `set_device()` maps
the seam's `device=` arg onto `comfy.model_management.cpu_state`, the single knob that switches the
whole device stack.

## The adapter bridge (`adapter.py`)

turboCLI's models are vanilla diffusers/transformers `nn.Module`s; ComfyUI's offloader expects its
own `comfy.ops` modules and a `ModelPatcher`. The adapter closes that gap, minimally:

- **`comfy_ize(model)`** — re-class each offloadable leaf (`Linear`/`Conv`/`LayerNorm`/`GroupNorm`/
  `Embedding`/`RMSNorm`, incl. custom `*RMSNorm`) to its `comfy.ops.disable_weight_init.*`
  counterpart, so its forward routes through ComfyUI's cast path with 1:1 parity. Injects the
  `comfy_cast_weights`/`weight_function`/`bias_function` attrs ModelPatcher reads.
- **`build_patcher` / `build_dynamic_patcher`** — wrap in `ModelPatcher` / `ModelPatcherDynamic`.
  `make_patchable` shadows diffusers/HF's read-only `.device` property (which ComfyUI assigns to)
  while keeping `str(cls)` identical so transformers' output-capture registry still fires.
- **`keep_uncastable_resident`** — move every non-castable param/buffer (custom norms, pad tokens,
  rotary caches) to the compute device so nothing is stranded on the offload device mid-forward.
- **`add_lora`** — on-cast LoRA: build the `{lora_key → model_weight_key}` map, hand it to
  `comfy.lora.load_lora` (vendored `weight_adapter`), register via `ModelPatcher.add_patches`. The
  deltas apply while each weight streams in — no fuse, no resident copy.
- **`install_encode_cache`** — wrap the pipeline's `encode_prompt` so a repeated prompt returns cached
  text embeddings and never re-runs (nor re-streams) the encoder — the diffusers analogue of ComfyUI's
  default node cache. Copies `comfy_execution/caching.py::RAMPressureCache`: full-input-signature key,
  embeddings held on CPU, eviction under host-RAM pressure via `comfy.model_management.get_free_memory`
  (below `min(10GB, max(2GB, 10% RAM))`, worst `1.3**age × bytes` first). Validated against ComfyUI's
  own `generate.sh`: identical cold time, encode skipped on a repeat prompt in both.

### Compute parity with ComfyUI

Diffusers runs some ops on slower kernels than ComfyUI. Copied from ComfyUI, not reimplemented:

- **`install_prefetch`** — drives ComfyUI's `comfy.model_prefetch` (`prefetch_queue_pop`) via
  forward hooks on the transformer's block lists, so block N+1's weights stream while block N
  computes (overlap; helps when streaming-bound).
- **`use_comfy_attention`** — a verbatim copy of `comfy.ops.scaled_dot_product_attention`
  (`comfy/ops.py:39-64`): on Windows+CUDA it forces the SDPA priority `[CUDNN, FLASH, EFFICIENT,
  MATH]` **per call**, but only for large inputs (`q.nelement() >= 1024*128`); small attentions use
  torch's default backend. We reproduce it (rather than import it) because it calls `F.sdpa`
  internally, so pointing torch's `F.sdpa` at it self-recurses; instead we patch
  `torch.nn.functional.scaled_dot_product_attention` once (diffusers calls it by attribute) and
  delegate to the saved original. **The per-call size gate matters:** forcing cuDNN on *every*
  attention (incl. small inputs, which comfy skips) makes the first z-image forward after a
  flux2→z-image switch pick a nondeterministic cuDNN plan and diverge; copying comfy's gate verbatim
  is deterministic.
- **`use_kitchen_rope`** — routes the diffusers transformer's RoPE through comfy-kitchen's fused
  `apply_rope1` (the kernel ComfyUI's `comfy/ldm/flux/math.py` uses), via a `(cos,sin)→freqs_cis`
  shim + a module-scoped patch of `diffusers.models.embeddings.apply_rotary_emb`. Lazy: comfy-kitchen
  is imported only on a matching call, so engines with their own rope (z-image) never touch it.
- **fused RMSNorm** — routing custom norms through `disable_weight_init.RMSNorm` gives ComfyUI's
  fused `F.rms_norm` (≈ 3.5× the eager `mul`/`rsqrt` diffusers/transformers use).

## Engines

- **flux2** (`Flux2KleinPipeline`) — text-to-image and image-input (edit); the pipeline accepts
  `image=`, so the seam is unchanged.
- **z-image** (`ZImagePipeline`) — text-to-image, Turbo (few-step).
- **qwen-image-edit** (`QwenImageEditPlusPipeline`) — image-edit; on-cast Lightning 4-step LoRA;
  ~55GB of weights (transformer + Qwen2.5-VL TE). With enough RAM they stream from RAM; on a small box
  each oversized component streams disk→VRAM via file-slices (the fit gate decides per component), so
  it runs where it wouldn't fit in RAM.

## Notes

- **Offloading matches ComfyUI.** Same `ModelPatcherDynamic`, same "N MB Staged / M force-preloaded"
  log, same ~13GB pinned host working set (shown as Windows "shared GPU memory", not a VRAM spill).
  Streaming is ~2s/step from pinned RAM; the residual gap vs ComfyUI on some engines is diffusers'
  unfused-qkv model compute, not the offloader.
- **comfy-kitchen** is a quantization-kernel library, unrelated to offloading — not a dependency.
- **Re-syncing** a newer ComfyUI: re-copy the files and re-apply the short edit set in
  `comfy/resync.md`; bump the commit pins there, in `comfy/__init__.py`, and in `README.md`.
- **License.** ComfyUI is GPLv3; the vendored copies live in this already-GPLv3 package.
