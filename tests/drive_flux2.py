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
#   pre_torch_init / load_pipe / prepare / <generate> / reclaim / release). Handy for validating the
#   device-agnostic path on a small GPU: the flux2 transformer (~7.75GB) + text encoder (~8GB) are
#   offloaded through ComfyUI's ModelPatcher and streamed to the compute device per forward, so the
#   pipe runs even when neither model fits VRAM.
#
#   The model directory is read from AIMDO_FLUX2_MODEL (a diffusers layout with transformer/,
#   text_encoder/, vae/ ...), so no machine-specific path is baked in.
#
#   Run:
#       AIMDO_FLUX2_MODEL=/path/to/FLUX.2-klein-4B python tests/drive_flux2.py cuda 1024 768 4
#       args: <device=cuda> <width=128> <height=128> <steps=1>   (device: cpu|cuda|mps)
#
#==================================================================================================

import os
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEVICE = sys.argv[1] if len(sys.argv) > 1 else "cuda"
WIDTH = int(sys.argv[2]) if len(sys.argv) > 2 else 128
HEIGHT = int(sys.argv[3]) if len(sys.argv) > 3 else 128
STEPS = int(sys.argv[4]) if len(sys.argv) > 4 else 1

MODEL = os.environ.get("AIMDO_FLUX2_MODEL")
if not MODEL:
    sys.exit("set AIMDO_FLUX2_MODEL to a flux2 diffusers model directory")

import aimdo

aimdo.pre_torch_init()
print("available:", aimdo.available())

import torch

dtype = torch.bfloat16 if DEVICE == "cuda" else (torch.float16 if DEVICE == "mps" else torch.float32)
if DEVICE == "cuda":
    print("GPU:", torch.cuda.get_device_name(0),
          "| VRAM %.1f GB" % (torch.cuda.get_device_properties(0).total_memory / 1e9))

t0 = time.time()
pipe = aimdo.load_pipe(model=MODEL, dtype=dtype, engine="flux2", device=DEVICE)
print("load_pipe: %.1fs" % (time.time() - t0))

aimdo.prepare(pipe)
print("prepared; execution_device:", getattr(pipe, "_execution_device", "?"))

gen = torch.Generator(device="cpu").manual_seed(42)
t1 = time.time()
with torch.inference_mode():
    img = pipe(prompt="a knight in armor", width=WIDTH, height=HEIGHT,
               guidance_scale=0.0, num_inference_steps=STEPS, generator=gen).images[0]
print("generate: %.1fs" % (time.time() - t1))
if DEVICE == "cuda":
    print("peak VRAM: %.2f GB" % (torch.cuda.max_memory_allocated() / 1e9))

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out_flux2_%s_%dx%d.png" % (DEVICE, WIDTH, HEIGHT))
img.save(out)
print("Saved:", out)

aimdo.reclaim(pipe)
aimdo.release(pipe)
print("done")
