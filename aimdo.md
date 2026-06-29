# aimdo

GPL offload backend for the turboCLI diffusion runner: a port of ComfyUI's **DynamicVRAM**
weight-streaming path to diffusers, built on the `comfy_aimdo` VBAR allocator. It streams each
`nn.Linear` weight to the GPU per forward instead of keeping it resident, so a model larger than
VRAM — even larger than VRAM + RAM — still runs. Spill-safe: a full VBAR fault returns OOM and we
fall back to a temp buffer, so aimdo never overcommits and there is no WDDM VRAM↔RAM spill.

Files: `offload.py` (the streamer), `placement.py` (resident-vs-stream decision + budgets),
`__init__.py` (the seam the runner drives: `pre_torch_init` / `available` / `supports` /
`load_pipe` / `prepare` / `reclaim` / `release`).

**Supported engines:** flux2, z-image, qwen-image-edit. **CUDA-only** — selected via turboCLI's
`cuda_offload=aimdo`; cpu/mps builds don't ship `comfy_aimdo`, so the backend is skipped there.

Based on:
- ComfyUI 5955ddff52a2eda2ba0cf7f3fb0927c93fb2fbb8
- comfy-aimdo ace72abefa1ede12a4b8a4e2c99919804e5f38e0

The `[CU <file> Lnn]` references below point into the ComfyUI tree; `[AI <file> Lnn]` into
comfy-aimdo. Re-verify and bump the line numbers when bumping either commit.

## How it works (vs ComfyUI LOWVRAM)

Derived from ComfyUI's LOWVRAM cast path; two changes make it DynamicVRAM:

1. **Host source = fast-DMA file reader**, not a staged HostBuffer copy. Each weight is read
   straight from its `.safetensors` shard into the GPU via `read_file_to_device`
   [AI host_buffer.py L67] — the native fast-DMA primitive behind ComfyUI's
   `read_tensor_file_slice_into` [CU memory_management.py L18]. The OS page cache holds the hot set
   (`mark_cold` drives comfy-aimdo's RAM-pressure cache). No full in-RAM copy, and not torch's
   pageable `copy_` (measured ~128 s/step here, transfer-starved). On this box `cudaHostRegister`
   fails, so fast-DMA is the only fast H2D.
2. **GPU residency via a VBAR.** Each weight gets a VBAR slot; per forward `vbar_fault()`
   [AI model_vbar.py L133] decides: resident (signature unchanged) → reuse, no read; faulted in
   (VRAM free) → read file→slot; offloaded (VRAM full) → read file→temp tensor; `vbar_unpin()`
   [AI model_vbar.py L137] after. This is ComfyUI's `_v` branch [CU ops.py L128-L141, L392].

Scope: the synchronous path (fault → read → use → unpin) on the default stream. The double-buffered
prefetch overlap is an optional add (file path, RAM-bound only).

## Cross-reference (offload.py → upstream)

| offload.py | upstream | role |
|---|---|---|
| imports `_ctl` / `_at` / `_hb` / `_vb` / `model_vbar` | [AI control.py / torch.py / host_buffer.py / vram_buffer.py / model_vbar.py] | device init, raw-ptr↔tensor, fast-DMA reader, reserved VRAM buffer, VBAR allocator |
| `reclaim_between_runs` | per-execution finally [CU execution.py L543-549]; reset_cast_buffers→soft_empty_cache [CU model_management.py L1383, L1950-1966]; vbars_reset_watermark_limits [CU execution.py L549, AI model_vbar.py L149] | per-generation pool/watermark reclaim |
| `get_aimdo_cast_buffer`, `STREAM_AIMDO_CAST_BUFFERS`, `DEFAULT_AIMDO_CAST_BUFFER_RESERVATION_SIZE` | get_aimdo_cast_buffer [CU model_management.py L1343]; 16 GiB reservation [CU L1309]; bounce tensor [CU ops.py get_cast_buffer L112-L124]; offload stream [CU L1385] | reserved VRAMBuffer carved into two ping-pong views |
| `_offsets` | `_comfy_tensor_file_slice` (file_ref/offset/size) [CU memory_management.py L36-L75] | name → (file, offset, num_bytes, dtype, shape) |
| manager (`_LOADED` / `_ACTIVE` / `_activate` / `partially_unload` / `restore_loaded_backups`) | current_loaded_models + load_models_gpu [CU model_management.py L849]; free_memory [CU L805-834]; partially_unload + restore_loaded_backups [CU model_patcher.py L1937-1941, L1768] | coexisting dynamic VBAR models; reclaim at the load boundary |
| `vbar.prioritize` / `deprioritize` | [CU model_patcher.py L1808-1809] / [AI model_vbar.py L60, L63] | VRAM retention priority |
| pinning (`_pin_memory`, `pins`, `pin_budget`) | pin_memory + cudaHostRegister [CU pinned_memory.py L66-L119, L98]; pinned_hostbuf_size [CU L1500]; ensure_pin_budget [CU L645]; priority balancer [CU pinned_memory.py L12] | optional truly-async H2D when the set fits RAM |
| on-cast LoRA (`_load_lora`, applied in `_pre`/`_do_read`) | weight_function during cast_to [CU ops.py L357-L380] | add scale·(up@down) to the freshly-read base weight |
| `_pre` / `_fetch` / `_pre_prefetch` / `_post` | cast_bias_weight + cast_modules_with_vbar [CU ops.py L128-L177]; prefetch [CU model_prefetch.py L34, ops.py L316-L334]; unpin [CU ops.py L392] | per-forward fault/read/use/unpin (+ optional prefetch) |
| dtype guard (DIFFERENCE #3) | cast_to [CU model_management.py L1453] | we keep the stored dtype (== compute, bf16); ComfyUI casts on the copy |
| tiny-weight resident skip | force-load ≤16 KiB [CU model_patcher.py L1870] | avoids stalling stream-buffer rotations |
| stale-signature reset in `partially_unload` | set_dirty(_v_signature=None) [CU model_patcher.py L1817-1819] | force a clean re-read after page decommit |
| `free` teardown | unpin_memory [CU model_management.py L1553] | cudaHostUnregister + decommit, rebuild-safe |
| `__init__.py` seam (`pre_torch_init`/`available`/`supports`/`load_pipe`/`prepare`/`reclaim`/`release`) | — | license-neutral interface the runner discovers as `backend/<mode>/` and drives |
| `_load_streamed` two managed VBARs (transformer + text encoder) | DynamicVram cast path [CU ops.py cast_bias_weight L281, model_patcher.py L1779]; both in current_loaded_models [CU model_management.py L945, load_models_gpu L849]; ModelVBAR / _vbar_get [CU model_patcher.py L1743, L1799]; dynamic-for-dynamic no-unload [CU model_management.py L824-828] | transformer + TE coexist as managed dynamic VBARs |
| text-encoder initial device | text_encoder_initial_device [CU model_management.py L1138] | TE offloaded, streamed via its own VBAR |
| placement budgets (`placement.py`) | ensure_pin_budget [CU model_management.py L645-L656]; MIN_WEIGHT_MEMORY_RATIO [CU L453]; extra reserved [CU L789-L793]; minimum_inference_memory [CU L802]; get_free_memory [CU L1653]; lowvram_model_memory [CU L935] | measured GPU resident budget + RAM pin budget |

## Notes / rationale

- **Dynamic-model manager.** Two coexisting dynamic (VBAR) models (e.g. text encoder + transformer)
  share one GPU. ComfyUI keeps both loaded and reclaims the inactive one's GPU pages at each load
  boundary. We mirror that explicitly — the manager reclaims at the active model's root forward —
  rather than via comfy-aimdo's on-demand cross-VBAR eviction, which underperformed here. Host RAM
  stays bounded because every big weight streams from disk (page cache only).

- **All-or-nothing pinning.** Pin the streamed set into a HostBuffer only when the *whole* set fits
  the measured RAM budget. Partial-pinning a >RAM model on a tight GPU exhausts host-registration /
  BAR mapping and OOMs the next alloc, so the >RAM case streams pageable from the page cache instead.
  ComfyUI partial-pins via a headroom balancer we don't replicate.

- **Measured prefetch.** Double-buffer overlap is ON only when every weight is pinned (RAM-bound);
  OFF when some weights stream from disk (disk-bound — overlap can't beat the disk and adds sync
  cost, measured +16% on qwen). `SKY_AIMDO_VBAR_PREFETCH=0/1` forces it. An in-use layer is pinned
  (`vbar_fault` pins, `_post` unpins), so prefetching the next layer can't evict it.

- **Reserved cast buffer.** The OOM/offloaded path bounces through a reserved aimdo VRAMBuffer (not
  `torch.empty`) so the aimdo allocator accounts for it and it never fights cudaMallocAsync for
  activation VRAM. It is persistent per process (`STREAM_AIMDO_CAST_BUFFERS`) and reused across model
  swaps, so `free()` drops the refs but does not free it.

- **`reclaim_between_runs`.** The VBAR maps physical VRAM through its own CUDA VMM, separate from
  torch's caching / cudaMallocAsync pool, so the two compete for the same VRAM. Between generations
  torch's retained pool is charged against the VBAR budget, so a persistent VBAR can't stay resident
  and re-streams every layer (~16 s/step vs ~2). Returning the pool to the driver fixes it. We omit
  `ipc_collect` (single process, no IPC handles). The watermark-floor reset is a no-op kept for
  fidelity (we juggle residency via prioritize/deprioritize + free_memory instead).

- **`from_module` streaming.** For a text encoder already loaded by diffusers/transformers (which
  rewrites the checkpoint key names), we match each live `nn.Linear` to its disk key *structurally*
  (`_match_disk_keys`: among disk keys of the same shape+dtype, the one with the longest common
  dotted-suffix; must be unique) and stream its big weights from disk; only small resident params
  (token embedding, vision-tower conv3d, norms, biases) go on GPU so the module runs fully on CUDA.
  The tied `lm_head`/embedding weight is kept resident, never streamed. Dropping the live CPU weights
  (~15 GB for a large TE) is what keeps host RAM to page cache.

- **PEFT LoRA targets** are wrapped as `<name>.base_layer`; we strip that suffix so the streamed BASE
  weight matches the checkpoint, and the resident adapter adds its low-rank delta on top.

- **`free()` ordering.** If the manager has released a model, restore it before teardown — freeing a
  `free_memory()`'d VBAR and then loading the next pipe was observed to segfault. And
  `cudaHostUnregister` every pinned region *before* freeing the HostBuffer, else the next HostBuffer
  (often at the same host addresses) fails with "already mapped".

- **DIFFERENCE #3 (dtype).** We stream and use weights in their stored dtype, which must equal the
  compute dtype (bf16 here); a mixed-dtype model warns loudly rather than silently mis-running.
  ComfyUI instead casts during the copy.

- **The seam (`__init__.py`).** The (LGPL) runner discovers `backend/<mode>/` and drives it through
  `pre_torch_init` / `available` / `supports` / `load_pipe` / `prepare` / `reclaim` / `release`
  only, so all GPL-derived code stays in this package. `load_pipe` either loads the transformer
  fully resident (it fits VRAM) or builds two coexisting managed VBARs — the transformer and the
  text encoder (the latter via `from_module`, streamed from disk). `manage=True` everywhere mirrors
  ComfyUI, where registration into `current_loaded_models` is unconditional; `prepare()` reloads the
  TE before the pipeline reads `_execution_device`, which only matters once a reused pipe has
  released it.

- **Placement (`placement.py`).** Reproduces the one MEASURED decision ComfyUI makes — the GPU
  resident budget — plus the RAM pin budget; no hardcoded sizes. full_resident iff the transformer +
  activation reserve fits free VRAM, else stream with a measured resident budget. There is
  deliberately no RAM-vs-disk branch (the page cache / RAM-pressure cache handles that tier).
  Constants map to ComfyUI: ~2 GB pin headroom (ensure_pin_budget), 40% resident-weight ratio
  (MIN_WEIGHT_MEMORY_RATIO), 400/600 MB (+100 MB on 16 GB+) extra reserve, 0.8 GB inference reserve.
