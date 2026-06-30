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

# =================================================================================================
#  placement.py -- resource-driven offloader placement, mirroring ComfyUI's MEASURED decision: it
#  measures free VRAM, subtracts reserves, and keeps a resident budget on the GPU while streaming
#  the rest (the host RAM<->disk tier self-manages via the page cache). Pure + side-effect free
#  (besides reading file sizes / CUDA mem info). Constants + upstream line references: aimdo.md.
# =================================================================================================
import os
import glob

import torch
import psutil

GB = 1024 ** 3

# Pin streamed weights into a HostBuffer up to available RAM minus a ~2 GB headroom floor.
PIN_HEADROOM = 2 * GB

# Keep up to 40% of free VRAM as resident weights before streaming the rest.
MIN_WEIGHT_MEMORY_RATIO = 0.4

_WINDOWS = os.name == "nt"

# OS page-lock ceiling for pinned host memory: 40% of RAM on Windows (~50% limit), 90% elsewhere.
# [CU model_management.py MAX_PINNED_MEMORY L1488]
MAX_PINNED_MEMORY = psutil.virtual_memory().total * (0.40 if _WINDOWS else 0.90)


def extra_reserved_memory(total_vram_bytes):
    # VRAM held back for everything that is not streamed weights: 400 MB, 600 MB on Windows
    # (shared-VRAM), +100 MB on 16 GB+ cards.
    extra = 400 * 1024 * 1024
    if _WINDOWS:
        extra = 600 * 1024 * 1024
        if total_vram_bytes > (15 * 1024 * 1024 * 1024):
            extra += 100 * 1024 * 1024
    return extra


def minimum_inference_memory(total_vram_bytes):
    # Reserve for activations/inference.
    return int(0.8 * GB + extra_reserved_memory(total_vram_bytes))


def gpu_mem(dev=0):
    # (free, total) driver-reported bytes (torch.cuda.mem_get_info).
    free, total = torch.cuda.mem_get_info(dev)
    return int(free), int(total)


def model_bytes(model):
    # Streamed-weight footprint = total bytes of the transformer .safetensors shards (the component
    # that streams; VAE/TE are placed separately). File size ~= weight bytes.
    tdir = os.path.join(model, "transformer")
    shards = glob.glob(os.path.join(tdir, "*.safetensors"))
    return int(sum(os.path.getsize(p) for p in shards))


def text_encoder_bytes(model):
    tedir = os.path.join(model, "text_encoder")
    shards = glob.glob(os.path.join(tedir, "*.safetensors"))
    return int(sum(os.path.getsize(p) for p in shards))


def resident_budget(free_vram, reserve):
    # ComfyUI's lowvram_model_memory: max(0, free - reserve, min(free*RATIO, free - reserve)).
    return int(max(0,
                   free_vram - reserve,
                   min(free_vram * MIN_WEIGHT_MEMORY_RATIO, free_vram - reserve)))


def pin_budget():
    # RAM budget for pinning streamed weights: available minus PIN_HEADROOM, capped at the OS
    # page-lock ceiling MAX_PINNED_MEMORY. Mirrors ComfyUI's ensure_pin_budget. See aimdo.md.
    return int(max(0, min(psutil.virtual_memory().available - PIN_HEADROOM, MAX_PINNED_MEMORY)))


def placement(model_size, free_vram, reserve):
    # The one measured decision: full_resident iff the whole transformer + activation reserve fits
    # VRAM, else stream with a measured resident budget. No RAM-vs-disk branch (the page cache /
    # RAM-pressure cache handles that tier). See aimdo.md.
    if model_size <= 0:
        # no transformer shards found (e.g. an HF-cache snapshot of blob symlinks, or a
        # non-standard layout): can't prove it fits, so stream rather than risk a full_resident
        # OOM.
        return ("stream", resident_budget(free_vram, reserve))
    if model_size + reserve <= free_vram:
        return ("full_resident", int(model_size))
    return ("stream", resident_budget(free_vram, reserve))


def probe(model, dev=0):
    # Measured numbers + the placement they imply (never raises for missing files). Call AFTER
    # VAE/TE placement so free_vram is the real remaining ceiling for the transformer.
    free, total = gpu_mem(dev)
    reserve = minimum_inference_memory(total)
    msize = model_bytes(model)
    mode, budget = placement(msize, free, reserve)
    return {
        "free_vram": free,
        "total_vram": total,
        "reserve": reserve,
        "model_bytes": msize,
        "te_bytes": text_encoder_bytes(model),
        "mode": mode,
        "budget": budget,
        "pin_budget": pin_budget(),
    }


def describe(p):
    return ("[placement] free_vram=%.2fGB total=%.2fGB reserve=%.2fGB transformer=%.2fGB "
            "te=%.2fGB pin_budget=%.2fGB -> mode=%s budget=%.2fGB"
            % (p["free_vram"] / GB, p["total_vram"] / GB, p["reserve"] / GB,
               p["model_bytes"] / GB, p["te_bytes"] / GB, p.get("pin_budget", 0) / GB,
               p["mode"], p["budget"] / GB))
