# Re-syncing the vendored ComfyUI snapshot

`aimdo/comfy/` is a **byte-for-byte snapshot** of ComfyUI's offloading subsystem, driven through
`aimdo/adapter.py` so turboCLI's diffusers pipelines reuse ComfyUI's device-agnostic
(CPU/CUDA/MPS) partial-offload path with 1:1 parity. Keeping the files verbatim is what makes
upgrades cheap: to move to a newer ComfyUI you **re-copy the files below, then re-apply the short
edit list here**. Nothing else should differ from upstream.

## Source snapshots (bump together)

| repo        | commit                                     | tag                    |
|-------------|--------------------------------------------|------------------------|
| ComfyUI     | `dd17debce517f8818ae9910b437cb1ebaa673176` | v0.0.2-3160-gdd17debc  |
| comfy-aimdo | `afa70d91ec9f6e1ab6758089d1b551f0269b6457` | —                      |

## Files copied verbatim from `ComfyUI/comfy/`

Flat modules: `model_management.py`, `model_patcher.py`, `ops.py`, `memory_management.py`,
`utils.py`, `lora.py`, `float.py`, `quant_ops.py`, `patcher_extension.py`, `hooks.py`,
`pinned_memory.py`, `model_prefetch.py`, `cli_args.py`, `options.py`.

`model_prefetch.py` is vendored verbatim and driven from `aimdo/adapter.py:install_prefetch`, which
reproduces ComfyUI's per-block `prefetch_queue_pop` loop (its own models call it from their forward,
e.g. `comfy/ldm/lightricks/av_model.py`) via forward hooks on the diffusers transformer's block
ModuleLists.
Packages (whole dir): `comfy_types/`, `weight_adapter/`.

`cli_args.py` + `options.py` are vendored as-is (not stubbed): `options.args_parsing` is `False`, so
`cli_args` does `parser.parse_args([])` and every flag gets its upstream default. This is more
faithful and lower-maintenance than a hand-written `args` stub (no field list to drift).

## The only edits over upstream — three categories

### (1) `sys.modules` alias + optional-comfy_aimdo shim  — `aimdo/comfy/__init__.py` (NEW, ours)
- Aliases this package as top-level `comfy` so the vendored `import comfy.X` lines resolve here
  with zero per-file edits. Always import via `comfy.*`, never `aimdo.comfy.*`.
- When `comfy_aimdo` (CUDA-only VBAR accelerator) is absent (CPU/MPS builds), registers empty
  stand-in submodules in `sys.modules` so the verbatim `import comfy_aimdo.X` lines still resolve.
  All comfy_aimdo *usage* is gated on `memory_management.aimdo_enabled` (default False), so the
  stand-ins are never dereferenced off-CUDA. **No vendored file is edited for comfy_aimdo.**

### (2) optional `comfy_kitchen` — no edit needed
Upstream `float.py` / `quant_ops.py` already wrap `import comfy_kitchen` in try/except and log a
warning when absent. Left verbatim; import proceeds without it (fp8/fp4 quant paths disabled).

### (3) `# [aimdo] disabled for turboCLI:` comment-outs — 3 one-line edits total
Each is a single commented import for a dependency that is **off the offloader's code path**, so the
offloader is unaffected; only unrelated helper functions would `NameError` if ever called (they are
not). Grep `# [aimdo] disabled for turboCLI:` to find them all.

| file      | line   | commented import                              | why it's off-path |
|-----------|--------|-----------------------------------------------|-------------------|
| `utils.py`| ~33    | `from einops import rearrange`                | only used by an attention-reshape helper (~L1351), not the offloader; drops the `einops` dependency |
| `lora.py` | ~22    | `import comfy.model_base`                     | only used by `model_lora_keys_unet()` (LoRA key-mapping at load time), which the adapter bypasses; would otherwise pull the whole model zoo |
| `hooks.py`| ~17    | `from node_helpers import conditioning_set_values` | only used by the conditioning helpers at the bottom of the file (~L692-781); keeps every hook class/enum `ModelPatcher` needs verbatim |

## Re-sync procedure

1. `cp` the files listed above from the new ComfyUI/comfy-aimdo checkouts over `aimdo/comfy/`.
2. Update the two commits in this file and in `aimdo/comfy/__init__.py`.
3. Re-apply the 3 comment-outs in category (3) (grep the marker in the OLD tree first to relocate
   them if line numbers moved).
4. Smoke test — must print `import OK` and `aimdo_enabled= False` on a CPU box:
   ```
   python -c "import aimdo.comfy; import comfy.model_patcher, comfy.ops, \
   comfy.memory_management as m; print('import OK; aimdo_enabled=', m.aimdo_enabled)"
   ```
5. If new upstream imports pull an off-path top-level module (like `node_helpers`), extend
   category (3) with a documented one-line comment-out rather than vendoring the extra tail.
