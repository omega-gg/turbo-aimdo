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
#   End-to-end driver for the v2 native offload seam, bypassing the runner (drives the seam directly:
#   pre_torch_init / load_pipe / prepare / <generate> / reclaim / release). Validates the
#   device-agnostic path on a small GPU: the big models (transformer + text encoder) are offloaded
#   through ComfyUI's ModelPatcher and streamed to the compute device per forward, so the pipe runs
#   even when neither model fits VRAM.
#
#   The model directory is read from AIMDO_MODEL (a diffusers layout with transformer/, text_encoder/,
#   vae/ ...), so no machine-specific path is baked in.
#
#   Run:
#       AIMDO_MODEL=/path/to/FLUX.2-klein-4B  python tests/drive.py flux2   cuda 1024 768 4
#       AIMDO_MODEL=/path/to/Z-Image-Turbo    python tests/drive.py z-image cuda 512  512  8
#       args: <engine> <device=cuda> <width=512> <height=512> <steps=8>
#             engine: flux2 | z-image | qwen-image-edit   device: cpu | cuda | mps
#
#==================================================================================================

import os
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
# Mirror turboCLI's run.sh environment so tests match deployment. CUDA: stream-ordered allocator so
# large VAE decodes fit and avoid the WDDM RAM spill. MPS: CPU fallback + disable the memory cap.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ENGINE = sys.argv[1] if len(sys.argv) > 1 else "flux2"
DEVICE = sys.argv[2] if len(sys.argv) > 2 else "cuda"
WIDTH = int(sys.argv[3]) if len(sys.argv) > 3 else 512
HEIGHT = int(sys.argv[4]) if len(sys.argv) > 4 else 512
STEPS = int(sys.argv[5]) if len(sys.argv) > 5 else 8

MODEL = os.environ.get("AIMDO_MODEL")
if not MODEL:
    sys.exit("set AIMDO_MODEL to the engine's diffusers model directory")

# Per-engine generate kwargs (guidance differs: flux2 uses distilled CFG=0, z-image/qwen use ~1).
GEN = {
    "flux2": dict(guidance_scale=0.0),
    "z-image": dict(),
    "qwen-image-edit": dict(true_cfg_scale=1.0),
}.get(ENGINE, {})

import aimdo

aimdo.pre_torch_init()
print("engine:", ENGINE, "| available:", aimdo.available())

import torch

dtype = torch.bfloat16 if DEVICE == "cuda" else (torch.float16 if DEVICE == "mps" else torch.float32)
if DEVICE == "cuda":
    torch.cuda.reset_peak_memory_stats()
    print("GPU:", torch.cuda.get_device_name(0),
          "| VRAM %.1f GB" % (torch.cuda.get_device_properties(0).total_memory / 1e9))

t0 = time.time()
pipe = aimdo.load_pipe(model=MODEL, dtype=dtype, engine=ENGINE, device=DEVICE)
print("load_pipe: %.1fs" % (time.time() - t0))

aimdo.prepare(pipe)
print("prepared; execution_device:", getattr(pipe, "_execution_device", "?"))

# Per-phase timing: text-encode (start -> first denoise step) vs per-step denoise, via a step
# callback. Lets us watch how streaming the text encoder / transformer changes each phase.
_t = {"start": 0.0, "steps": []}


def _step_cb(pipe_, step, ts, kw):
    now = time.time()
    _t["steps"].append(now)
    return kw


# Optional image input (edit / img2img): set AIMDO_IMAGE to a file, or "gradient" for a synthetic
# one. flux2 and qwen-image-edit accept image=; text-to-image engines leave it unset.
call_kwargs = dict(GEN)
_img_src = os.environ.get("AIMDO_IMAGE")
if _img_src:
    from PIL import Image
    if _img_src == "gradient":
        import numpy as np
        xs = np.linspace(0, 255, WIDTH, dtype=np.uint8)
        arr = np.stack([np.tile(xs, (HEIGHT, 1)), np.tile(xs[::-1], (HEIGHT, 1)),
                        np.full((HEIGHT, WIDTH), 128, np.uint8)], axis=-1)
        call_kwargs["image"] = Image.fromarray(arr, "RGB")
    else:
        call_kwargs["image"] = Image.open(_img_src).convert("RGB").resize((WIDTH, HEIGHT))
    print("image input:", _img_src)

gen = torch.Generator(device="cpu").manual_seed(42)
t1 = time.time()
_t["start"] = t1
with torch.inference_mode():
    img = pipe(prompt="a knight in armor", width=WIDTH, height=HEIGHT,
               num_inference_steps=STEPS, generator=gen,
               callback_on_step_end=_step_cb, **call_kwargs).images[0]
total = time.time() - t1
print("generate: %.1fs" % total)
_marks = _t["steps"]
if _marks:
    encode = _marks[0] - _t["start"]
    step_times = [b - a for a, b in zip(_marks[:-1], _marks[1:])]
    avg_step = (sum(step_times) / len(step_times)) if step_times else 0.0
    vae = total - (_marks[-1] - _t["start"])
    print("  text-encode+setup: %.1fs | denoise avg/step: %.2fs (%d steps) | vae+decode: %.1fs"
          % (encode, avg_step, len(_marks), vae))
if DEVICE == "cuda":
    print("peak VRAM: %.2f GB" % (torch.cuda.max_memory_allocated() / 1e9))

out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "out_%s_%s_%dx%d.png" % (ENGINE, DEVICE, WIDTH, HEIGHT))
img.save(out)
print("Saved:", out)

aimdo.reclaim(pipe)
aimdo.release(pipe)
print("done")
