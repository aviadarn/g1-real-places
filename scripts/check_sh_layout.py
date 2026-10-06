"""Which way are the 45 higher-order SH coefficients laid out in a .nurec?

3DGUT stores `features_specular` flat, (N, 45). Read coefficient-major (N,15,3) or
channel-major (N,3,15), the two readings give different colours. The right one stays
close in hue to the view-independent DC render; the wrong one adds rainbow tints.
On Linden: chromatic deviation from DC 0.007 coefficient-major vs 0.022 channel-major.

    python scripts/check_sh_layout.py linden     # writes scratch/sh_layout.png
"""

import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from g1places.scenes import SCENES                    # noqa: E402
from g1places.splat import SplatScene, intrinsics, look_at   # noqa: E402


def main(name: str = "linden"):
    sc = SCENES[name]
    s = SplatScene(sc.dir / "splats.npz")
    W, H = 640, 480
    K = intrinsics(W, H, 60)
    coef_major = s.sh.clone()
    n = coef_major.shape[0]
    flat = coef_major[:, 1:, :].reshape(n, 45)
    chan_major = torch.cat([coef_major[:, :1], flat.reshape(n, 3, 15).transpose(1, 2)], 1)
    views = [look_at((-3, -7, 1.5), (2, 2, 1.0)), look_at((4, 5, 1.5), (-1, -4, 1.0))]
    imgs = []
    for sh, deg in [(coef_major, 0), (coef_major, 3), (chan_major, 3)]:
        s.sh, s.sh_degree = sh, deg
        imgs.append(np.concatenate([s.render(sc.world_to_file_pose(v), K, W, H)[0] for v in views], 0))
    for label, img in zip(["coefficient-major", "channel-major"], imgs[1:]):
        d = img - imgs[0]
        print(f"{label:>18}: chromatic deviation from DC {np.abs(d - d.mean(-1, keepdims=True)).mean():.4f}")
    (ROOT / "scratch").mkdir(exist_ok=True)
    iio.imwrite(ROOT / "scratch" / "sh_layout.png", (np.concatenate(imgs, 1) * 255).astype(np.uint8))


if __name__ == "__main__":
    main(*sys.argv[1:])
