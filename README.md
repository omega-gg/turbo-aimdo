# turbo-aimdo

turbo-aimdo is a [turboCLI](https://omega.gg/turboCLI) high performance offloader based on [comfy-aimdo](https://github.com/Comfy-Org/comfy-aimdo).
Paired with a compliant CUDA GPU it improves generate times substantially through optimal memory
allocations when the model does not fit into VRAM, even on a very low VRAM GPU.

## ComfyUI

turbo-aimdo's offloader is a **byte-for-byte port** of ComfyUI's memory-management / model-patcher /
ops subsystem (under `aimdo/comfy`), driven through the thin adapter in `aimdo/adapter.py`. The
port is pinned to exact upstream commits. Bump these together with the vendored files and re-apply
the short edit list in [`aimdo/comfy/resync.md`](aimdo/comfy/resync.md):

| dependency  | commit                                     | version                 |
|-------------|--------------------------------------------|-------------------------|
| ComfyUI     | `bb131be9e83d2f773c90f1d6f1e4b248a498c8c5` | `v0.27.0`               |
| comfy-aimdo | `afa70d91ec9f6e1ab6758089d1b551f0269b6457` | `0.4.10`                |

## Credits

- [ComfyUI](https://github.com/comfy-org/comfyui)
- [comfy-aimdo](https://github.com/Comfy-Org/comfy-aimdo)

## Contribute

PR(s) are welcomed

## Platforms

- Windows 10 and later
- Linux 64 bit

## Requirements

- [turboCLI](https://omega.gg/turboCLI) and a CUDA compliant GPU

## License

Copyright (C) 2026-2026 turbo-aimdo authors | https://omega.gg/turbo-aimdo

### Authors

- Benjamin Arnaud aka [bunjee](https://bunjee.me) | <bunjee@omega.gg>

### GNU General Public License Usage

turbo-aimdo may be used under the terms of the GNU General Public License version 3 as published
by the Free Software Foundation and appearing in the LICENSE.md file included in the packaging
of this file. Please review the following information to ensure the GNU General Public License
requirements will be met: https://www.gnu.org/licenses/gpl.html.
