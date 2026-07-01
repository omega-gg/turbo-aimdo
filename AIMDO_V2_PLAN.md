# turbo-aimdo v2 — vendor ComfyUI offloading + thin adapter

## Context

`turbo-aimdo` is the GPL offload backend turboCLI discovers at `backend/<mode>/` and drives
through a fixed seam (`pre_torch_init` / `available` / `supports` / `load_pipe` / `prepare` /
`reclaim` / `release`). **v1** (`aimdo/offload.py` + `placement.py`, ~1100 lines) *cherry-picked
lines* from ComfyUI and reimplemented per-forward weight streaming on top of comfy-aimdo's
**CUDA-only VBAR** primitives. It works, but every ComfyUI change means re-porting hand-copied
lines, and it only runs on CUDA.

**v2 goal:** stop cherry-picking. Instead **vendor ComfyUI's offloading files verbatim** (1:1
parity), and add a **thin middleman adapter** that bridges turboCLI's **diffusers** pipelines to
ComfyUI's `ModelPatcher` machinery — behind the *same seam*, so turboCLI is unchanged. Use
ComfyUI's **native** partial-offload path (`ModelPatcher` + `load_models_gpu` + `LowVramPatch` +
`cast_bias_weight`/`cast_to`), which is device-agnostic (**CPU / CUDA / MPS**). comfy-aimdo VBAR
becomes an *optional* CUDA-only acceleration added later.

**Maintainability is the whole point:** the vendored files stay a clean, diffable mirror of
upstream. Where a line can't apply to turboCLI, **comment it out in place** (marked
`# [aimdo] disabled for turboCLI:`) or feed it a **dummy placeholder** (stub `args`, optional
`comfy_aimdo`) — never rewrite logic. Re-syncing to a future ComfyUI commit = re-copy the files,
re-apply the small documented comment set.

**Decisions (confirmed with user):** native-first, VBAR later · flux2 as first working engine ·
keep v1 `offload.py`/`placement.py` as reference during migration, delete once v2 covers it.

**Not needed:** `comfy-kitchen` is a quantization-kernel library, not a memory manager — ignore it.

**License:** ComfyUI is GPLv3; vendored copies live inside the already-GPLv3 `aimdo/` package. No
conflict. Keep each ComfyUI file's header and record the source commit.

## Target layout

```
aimdo/
  __init__.py            # backend seam — UNCHANGED signatures; thin, delegates to adapter
  adapter.py             # NEW middleman: diffusers <-> ModelPatcher bridge (the only real logic)
  placement.py           # KEEP, shrink to: full_resident-vs-stream gate + device pick
  offload.py             # KEEP as reference during migration; delete after Phase D
  comfy/                 # NEW vendored ComfyUI snapshot (copied byte-for-byte)
    __init__.py          # sys.modules["comfy"] alias + source-commit provenance note
    resync.md            # the exact edit set applied over upstream (for re-vendoring)
    cli_args.py          # STUB (NOT copied) — provides `args` + `PerformanceFeature`
    model_management.py  model_patcher.py  ops.py  memory_management.py
    utils.py  lora.py  float.py  quant_ops.py  patcher_extension.py  hooks.py
    pinned_memory.py     # optional (CUDA-gated)
```

## The central mechanism: make diffusers modules fire ComfyUI casting

turboCLI's models are **vanilla `torch.nn` modules**; their `forward` does not check
`comfy_cast_weights`. ComfyUI's casting only fires because its models are built from
`comfy.ops.disable_weight_init.*` (subclasses of `CastWeightBiasOp`) whose `forward` routes
through `forward_comfy_cast_weights` → `cast_bias_weight` → `cast_to`
(`comfy/ops.py:281`, `:492-503`; `comfy/model_management.py:1453`). `cast_to` is plain
`tensor.to(device)` — device-agnostic.

**Adapter "comfy-ize" step** (in `adapter.py`), for each offloadable leaf module (`Linear`,
`Conv1d/2d/3d`, `LayerNorm`, `GroupNorm`, `Embedding`, `RMSNorm`):
1. Inject the attributes ModelPatcher reads/writes: `comfy_cast_weights = False`,
   `weight_function = []`, `bias_function = []`.
2. **Re-class the instance**: `m.__class__ = _comfy_class_for(type(m))`, where `_comfy_class_for`
   caches a subclass mixing `CastWeightBiasOp` into the module's torch base type and inheriting the
   matching `disable_weight_init.<Class>` `forward`/`forward_comfy_cast_weights`. Because
   `forward_comfy_cast_weights` computes against the base op (`F.linear(input, weight, bias)`), a
   vanilla module works unchanged. Re-classing (not copying bound methods) keeps true 1:1 parity
   with ComfyUI's own methods.
3. Leave `weight`/`bias` as real `nn.Parameter`s on the offload device (CPU); the native branch of
   `cast_bias_weight` streams them to the compute device per forward via `cast_to`.
4. Call `comfy.model_management.archive_model_dtypes(model)` so the `<param>_comfy_model_dtype`
   attrs `_load_list` reads (`comfy/model_patcher.py:903`) exist.

Skip modules whose weight isn't in the checkpoint, and **never independently offload tied weights**
(TE embedding/lm_head) — mirror v1 (`offload.py:316`, `:509`).

**Driving it** — build a patcher per diffusers submodule (transformer, text_encoder), exact
upstream signature (`comfy/model_patcher.py:293`):
```python
mp = comfy.model_patcher.ModelPatcher(model=comfy_ized, load_device=..., offload_device=...,
                                      size=measured_bytes)
```
LoRA is registered via `mp.add_patches({weight_key: [(strength,(up,down),...)]}, strength)` so
`LowVramPatch.__call__` → `comfy.lora.calculate_weight` applies it on cast (same math as v1's
`_do_read` addmm, on the maintained path).

## Seam mapping (signatures unchanged; turboCLI untouched)

| Seam fn | v2 behavior |
|---|---|
| `pre_torch_init()` | Optional `try: import comfy_aimdo; ctl.init(); memory_management.aimdo_enabled=True`. **Never raises on CPU/MPS.** |
| `available()` | True once `aimdo.comfy` imports cleanly (any device) — no longer requires comfy-aimdo |
| `supports(engine)` | unchanged static set |
| `load_pipe(...)` | build pipe with **CPU-resident** weights (mmap safetensors), `adapter.comfy_ize()` transformer + text_encoder, build `ModelPatcher`(s), register LoRA, stash on `pipe._aimdo_patchers` |
| `prepare(pipe)` | `load_models_gpu(pipe._aimdo_patchers, memory_required=...)` — TE loaded **before** pipe reads `_execution_device` |
| `reclaim(pipe)` | `free_memory(reserve, device)` + `soft_empty_cache()` |
| `release(pipe)` | `mp.detach(unpatch_all=True)` per patcher + drop refs so `current_loaded_models` finalizers fire |

## Device selection (one path for CPU / CUDA / MPS)

- `load_device = comfy.model_management.get_torch_device()` (mps/cpu/cuda per `cpu_state`).
- `offload_device = comfy.model_management.unet_offload_device()` (cpu unless HIGH_VRAM);
  TE via `text_encoder_device()` / `text_encoder_offload_device()`.
- Adapter forces `comfy.model_management.cpu_state` from the seam's `device=` arg
  (`CPUState.CPU`/`.MPS`/`.GPU`) before first use — the single knob that switches all three. Same
  `ModelPatcher` + `cast_to` code streams CPU→CUDA / CPU→MPS / stays on CPU.

## Vendoring & import rewiring (the maintainability core)

- **`sys.modules["comfy"]` alias** in `aimdo/comfy/__init__.py` so the vendored files' own
  `import comfy.X` / `from comfy.cli_args import args` resolve to the vendored package with **zero
  per-file import edits** — maximal verbatim parity, trivial re-sync.
- **`cli_args.py` STUB** (do not copy ComfyUI's argparse): provide an `args` object with the fields
  the vendored files read (`args.cpu`, `args.directml`, `args.reserve_vram`, `args.gpu_only`,
  `args.disable_smart_memory`, `args.high_ram`, …, each defaulting to the no-flag value) plus
  `PerformanceFeature`. Enumerate by grepping vendored files for `args.` / `PerformanceFeature.`.
- **`comfy_aimdo` optional** — wrap its top-level imports (`memory_management.py`,
  `model_patcher.py`, `ops.py`) in `try/except ImportError` and gate every aimdo branch on the
  existing `memory_management.aimdo_enabled` flag. On CPU/MPS it stays False and the native
  `cast_to` path runs. **This is the single most important edit for device-agnosticism.**
- **Comment-out policy per file**, all marked `# [aimdo] disabled for turboCLI:` so upstream diffs
  stay legible: `model_management.py` — comment directml block (`:109-121`) and unwanted npu/mlu
  probes; keep VRAMState/CPUState/get_torch_device/load_models_gpu/free_memory/cast_to/
  soft_empty_cache/module_size. `ops.py` — keep CastWeightBiasOp/disable_weight_init/
  cast_bias_weight/forward_comfy_cast_weights and the `QuantizedTensor` import; comment
  `cublas_ops`, the cudnn conv-bug workaround, unused fp8 ops. `hooks.py` — copy, or stub the
  symbols `model_patcher` imports (`EnumHookMode`/`_HookRef`/`HookGroup`/`apply_hooks`) as no-ops.
- **Document every edit** in `aimdo/comfy/resync.md` (source commit + the three edit categories) so
  re-vendoring a future ComfyUI is: re-copy files → re-apply this short list.

## Weight source (chosen: native, all devices)

Do **not** meta-load in v2's primary path. Load the transformer state dict onto CPU
(`offload_device`) as real Parameters via diffusers `from_pretrained(low_cpu_mem_usage=True)`
(mmap-backed safetensors). `ModelPatcher` then streams CPU→GPU per forward exactly as ComfyUI does
for a CPU-resident UNet — least surprise, works on all three devices. Cost ≈ transformer size in
host RAM (page-cache-backed; fine for flux2/z-image, feasible for qwen via mmap).

VBAR **disk**-streaming (run models > VRAM+RAM, `TensorFileSlice` + `read_tensor_file_slice_into`,
`comfy/memory_management.py`) is deferred to Phase D as an optional CUDA acceleration, gated on
`aimdo_enabled`.

## Phased build order

- **Phase A — vendor + stub + import clean on CPU.** Copy mandatory files, add `sys.modules`
  alias, write `cli_args` stub, wrap `comfy_aimdo` in `try/except`. Build the import closure
  iteratively (ComfyUI pulls a tail: `hooks`→`patcher_extension`, `model_base`, `conds`…); stub
  leaf modules aggressively. **Done when** `import aimdo.comfy.model_patcher` works CPU-only with
  `aimdo_enabled == False`.
- **Phase B — flux2 offloading on pure CPU, end-to-end.** Write `adapter.comfy_ize()` (attr inject
  + re-class) and `adapter.build_patcher()`; wire the seam with CPU-resident weights; force
  `cpu_state = CPU`. **Done when** flux2 generates on CPU with the transformer's casting path firing
  and the seam byte-identical from turboCLI's side.
- **Phase C — CUDA + MPS, same code path; add z-image + qwen.** Device pick from seam `device=` arg
  → set `cpu_state`. No new logic. Shrink `placement.py` to the full_resident gate + device pick
  (let `load_models_gpu` own the resident budget, `comfy/model_patcher.py:931-939`, to avoid
  double-counting).
- **Phase D — optional VBAR acceleration (CUDA).** `pre_torch_init` inits comfy-aimdo, sets
  `aimdo_enabled = True`, un-comments the VBAR branches, switches CUDA to `ModelPatcherDynamic` +
  `TensorFileSlice` disk streaming for >RAM models. Restores v1's run-bigger-than-VRAM as
  acceleration atop the portable native path; CPU/MPS unaffected. Then delete v1
  `offload.py`/`placement.py`.

## Risks / hardest parts

1. **Import-chain closure** (Phase A) — long ComfyUI tail; resolve iteratively via `ImportError`,
   stub leaves.
2. **Re-classing correctness** — diffusers/PEFT subclass `nn.Linear` (`base_layer`); match by torch
   base type, skip weights not in checkpoint, preserve tied weights.
3. **`_execution_device` timing** — `prepare()` must `load_models_gpu` before the pipe reads device
   placement (v1 met this via `enc.activate()`).
4. **Memory-accounting parity** — let `load_models_gpu` own the resident budget; keep
   `placement.py` to the gate only.
5. **Re-sync drift** — confine edits to the three documented categories; keep `resync.md` current.

## Verification

Primary harness: **`sandbox/aimdo/flux2/run.sh`** (single-shot flux2 generator; `aimdo` is already a
valid `cuda_offload` value, usage-checked at `run.sh:135`). It drives the seam directly
(`pre_torch_init()` + `load_pipe(model, dtype, engine="flux2")`) — a light single-shot path that
exercises `load_pipe` without `prepare/reclaim/release`. Usage:
`./run.sh "knight in armor" out.png 512 512 <cpu|cuda|mps> 42 4 aimdo`.

**One test-harness change needed for device-agnostic testing:** `run.sh:304` gates the backend on
`renderer == "cuda"` (`if renderer == "cuda" and offload not in (...)`). To exercise v2 on CPU/MPS,
loosen that to allow `aimdo` for any renderer. Keep this change in the sandbox harness (it is a
benchmark script, not deployed turboCLI); the real turboCLI seam stays untouched.

- **Phase A:** `python -c "import aimdo.comfy.model_patcher, aimdo.comfy.model_management as mm;
  print(mm.aimdo_enabled)"` → imports clean, prints `False` on a CPU box.
- **Phase B:** `./run.sh "knight in armor" out_cpu.png 512 512 cpu 42 4 aimdo` (after loosening the
  gate). Confirm a correct image and that streamed Linears took the cast path (log/assert
  `comfy_cast_weights` fired, weights resident on CPU between forwards). Compare vs the same run with
  `sequential_cpu`.
- **Phase C:** `./run.sh ... cuda ... aimdo` and `./run.sh ... mps ... aimdo` with no code change
  beyond the renderer arg; then z-image + qwen via their own bash scripts. Confirm peak VRAM stays
  within the measured resident budget.
- **Phase D:** on CUDA, load a model larger than VRAM+RAM; confirm it runs with VBAR enabled and
  that disabling comfy-aimdo cleanly falls back to the native path. Confirm CPU/MPS unaffected.
- Throughout: diff `aimdo/comfy/*` against the upstream ComfyUI files — the only differences must be
  the documented `# [aimdo]` comments, the optional-import wrappers, and the `cli_args` stub.

## Step 0 (on approval): commit this plan into the repo

`turbo-aimdo` is its own git repo (`C:\dev\workspace\msvc\turbo-aimdo`). The first action after
approval is to copy this plan into the repo (e.g. `turbo-aimdo/AIMDO_V2_PLAN.md`) and commit it, so
the design is versioned alongside the code before implementation begins. (Git writes aren't possible
while in plan mode; this happens as the first implementation step.)

## Critical files

- `C:\dev\test\ComfyUI\comfy\ops.py` — CastWeightBiasOp / disable_weight_init / cast_bias_weight /
  forward_comfy_cast_weights (casting mechanism to vendor + rebind onto diffusers modules)
- `C:\dev\test\ComfyUI\comfy\model_patcher.py` — ModelPatcher / `_load_list` / `load` / LowVramPatch
  / patch_weight_to_device (offload orchestrator the adapter constructs)
- `C:\dev\test\ComfyUI\comfy\model_management.py` — load_models_gpu / free_memory / get_torch_device
  / cpu_state / cast_to / soft_empty_cache (seam targets + device selection)
- `C:\dev\test\ComfyUI\comfy\memory_management.py` — `aimdo_enabled` flag; TensorFileSlice disk path
  (comfy-aimdo import to make optional; Phase D)
- `C:\dev\workspace\msvc\turbo-aimdo\aimdo\__init__.py` — the backend seam that must stay identical;
  `adapter.py` plugs in behind it
