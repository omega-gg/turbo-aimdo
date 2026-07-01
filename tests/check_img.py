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
#   Sanity-check a generated image is real content, not a black/uniform frame: prints spread stats
#   and exits non-zero if the image looks empty (std too low). Used to validate drive_flux2.py output.
#
#   Run:  python tests/check_img.py path/to/image.png
#
#==================================================================================================

import sys

from PIL import Image
import numpy as np


def main(path):
    im = Image.open(path)
    a = np.asarray(im).astype("float32")
    std = float(a.std())
    colors = len(np.unique(a.reshape(-1, a.shape[-1]), axis=0)) if a.ndim == 3 else int((a > 0).sum())
    print("size:", im.size, "mode:", im.mode)
    print("min/mean/max: %.1f/%.1f/%.1f" % (a.min(), a.mean(), a.max()))
    print("std (>~5 => real content): %.1f" % std)
    print("unique colors:", colors)
    if std < 5.0:
        sys.exit("image looks empty (std < 5)")
    print("PASS")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python tests/check_img.py <image.png>")
    main(sys.argv[1])
