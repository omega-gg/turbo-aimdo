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
#   Phase B bridge correctness test (no model download needed, runs on CPU in a second).
#
#   Proves the diffusers<->ComfyUI ModelPatcher bridge: after adapter.comfy_ize() + a ModelPatcher
#   load with a tiny lowvram budget, every leaf routes through ComfyUI's forward_comfy_cast_weights
#   cast path (comfy_cast_weights True, model_lowvram True) AND the output is bit-for-bit identical
#   to the original forward -- i.e. the device-agnostic offload path is exercised and correct.
#
#   Run:  python tests/bridge_parity.py
#
#==================================================================================================

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import aimdo.comfy  # noqa: F401  establishes the `comfy` alias
import torch

import aimdo.adapter as adapter
import comfy.model_management as mm
import comfy.ops as ops


def main():
    adapter.set_device("cpu")
    assert mm.get_torch_device().type == "cpu", mm.get_torch_device()
    print("device:", mm.get_torch_device())

    torch.manual_seed(0)

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.a = torch.nn.Linear(64, 128)
            self.b = torch.nn.LayerNorm(128)
            self.c = torch.nn.Linear(128, 64)
            self.emb = torch.nn.Embedding(10, 64)

        def forward(self, x, idx):
            return self.c(self.b(torch.relu(self.a(x)))) + self.emb(idx)

    net = Net().eval()
    x = torch.randn(4, 64)
    idx = torch.tensor([1, 2, 3, 4])
    with torch.inference_mode():
        ref = net(x, idx).clone()

    n = adapter.comfy_ize(net)
    assert n == 4, n
    print("comfy_ized leaves:", n)

    mp = adapter.build_patcher(net)
    # lowvram_model_memory=1 byte -> every castable module gets flagged for the cast path.
    mp.load(device_to=mm.get_torch_device(), lowvram_model_memory=1)

    flagged = [name for name, m in net.named_modules() if getattr(m, "comfy_cast_weights", False)]
    print("modules routed through cast path:", flagged)
    assert set(flagged) == {"a", "b", "c", "emb"}, flagged
    assert net.model_lowvram is True

    with torch.inference_mode():
        out = net(x, idx)

    parity = torch.allclose(ref, out, atol=1e-6)
    print("output parity after offload:", parity)
    assert parity, "cast path changed the output!"

    # Every converted leaf is now a real ComfyUI CastWeightBiasOp carrying the expected attrs.
    for _name, m in net.named_modules():
        if isinstance(m, ops.CastWeightBiasOp):
            for attr in ("comfy_cast_weights", "weight_function", "bias_function", "bias"):
                assert hasattr(m, attr), (_name, attr)

    print("PASS")


if __name__ == "__main__":
    main()
