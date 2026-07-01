# aimdo (v2)

GPL offload backend for the turboCLI diffusion runner. Instead of cherry-picking lines from
ComfyUI (v1, see `aimdo_v1.md`), v2 **vendors ComfyUI's offloading subsystem verbatim** and adds a
**thin adapter** so turboCLI's diffusers pipelines reuse ComfyUI's own memory manager with 1:1
parity. The same code runs a model larger than VRAM — even larger than VRAM+RAM — by streaming its
weights disk→VRAM per forward.

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
  `comfy.memory_management.aimdo_enabled` and uses `ModelPatcherDynamic`: each castable module gets
  a comfy-aimdo VBAR slot and streams **disk→VRAM** on fault (`TensorFileSlice` +
  `read_tensor_file_slice_into`). Runs models larger than VRAM+RAM. Weights come from ComfyUI's own
  `load_safetensors` (mmap + file-sliced), so nothing is materialized.

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
- **`load_streamed` / `assign_streamed_weights`** — meta-load a transformer (no weight RAM) and
  assign mmap/file-sliced weights from `load_safetensors`; structural (shape + longest-common
  suffix) key matching handles renamed checkpoints (e.g. the Qwen2.5-VL TE's `model.` →
  `language_model.`).
- **`keep_uncastable_resident`** — move every non-castable param/buffer (custom norms, pad tokens,
  rotary caches) to the compute device so nothing is stranded on the offload device mid-forward.
- **`add_lora`** — on-cast LoRA: build the `{lora_key → model_weight_key}` map, hand it to
  `comfy.lora.load_lora` (vendored `weight_adapter`), register via `ModelPatcher.add_patches`. The
  deltas apply while each weight streams in — no fuse, no resident copy.

### Compute parity with ComfyUI

Diffusers runs some ops on slower kernels than ComfyUI. Copied from ComfyUI, not reimplemented:

- **`install_prefetch`** — drives ComfyUI's `comfy.model_prefetch` (`prefetch_queue_pop`) via
  forward hooks on the transformer's block lists, so block N+1's weights stream while block N
  computes (overlap; helps when streaming-bound).
- **`prefer_cudnn_attention`** — wraps the forward in `sdpa_kernel` with ComfyUI's exact backend
  priority `[CUDNN, FLASH, EFFICIENT, MATH]` (cuDNN flash ≈ 2× the cutlass efficient kernel torch
  otherwise picks). We can't redirect `F.sdpa` to `comfy.ops.scaled_dot_product_attention` — that
  wrapper calls `F.sdpa` itself and self-recurses — so we apply the same mechanism it uses.
- **fused RMSNorm** — routing custom norms through `disable_weight_init.RMSNorm` gives ComfyUI's
  fused `F.rms_norm` (≈ 3.5× the eager `mul`/`rsqrt` diffusers/transformers use).

## Engines

- **flux2** (`Flux2KleinPipeline`) — text-to-image and image-input (edit); the pipeline accepts
  `image=`, so the seam is unchanged.
- **z-image** (`ZImagePipeline`) — text-to-image, Turbo (few-step).
- **qwen-image-edit** (`QwenImageEditPlusPipeline`) — image-edit; on-cast Lightning 4-step LoRA;
  ~55GB of weights (transformer + Qwen2.5-VL TE) stream from disk on a small GPU.

## Notes

- **Offloading matches ComfyUI.** Same `ModelPatcherDynamic`, same "N MB Staged / M force-preloaded"
  log, same ~13GB pinned host working set (shown as Windows "shared GPU memory", not a VRAM spill).
  Streaming is ~2s/step from pinned RAM; the residual gap vs ComfyUI on some engines is diffusers'
  unfused-qkv model compute, not the offloader.
- **comfy-kitchen** is a quantization-kernel library, unrelated to offloading — not a dependency.
- **Re-syncing** a newer ComfyUI: re-copy the files and re-apply the short edit set in
  `comfy/resync.md`; bump the commit pins there, in `comfy/__init__.py`, and in `README.md`.
- **License.** ComfyUI is GPLv3; the vendored copies live in this already-GPLv3 package.
