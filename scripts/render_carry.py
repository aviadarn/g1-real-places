"""Render the warehouse pick-and-place (train/run_carry_demo.py output) into the splat.

    python scripts/render_carry.py --range 0 400
    python scripts/render_carry.py --encode
    python scripts/render_carry.py --frames 380 600      # stills

Same three views as the walks; the chase camera sits behind and to the right of the
robot so the carton and the stack stay in view while it faces the shelf.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import imageio.v3 as iio
import mujoco
import numpy as np
from scipy.ndimage import gaussian_filter1d

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from g1places.composite import Compositor        # noqa: E402
from g1places.scenes import SCENES               # noqa: E402
from g1places.splat import SplatScene, look_at   # noqa: E402
from render import DEPTH_MAX, colorize_depth, encode, label, set_chase   # noqa: E402

SRC = ROOT / "data" / "linden" / "carry_demo"


def yaw_of(q):
    w, x, y, z = q
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


HANDLING = {"line up", "reach", "grip", "lift", "step back", "lower", "release", "done"}


def chase_poses(qpos, labels, back=2.3, side_handling=-1.0, height=1.55, sigma=15):
    """Straight behind while walking (the aisle is narrow); over the right shoulder while
    handling the carton, so the stack and the carton stay in view. Blended smoothly."""
    side = gaussian_filter1d(np.array([side_handling if l in HANDLING else 0.0 for l in labels]), 20)
    xy = gaussian_filter1d(qpos[:, :2], sigma, axis=0)
    yaw = gaussian_filter1d(np.unwrap([yaw_of(q[3:7]) for q in qpos]), sigma)
    fwd = np.c_[np.cos(yaw), np.sin(yaw)]
    left = np.c_[-np.sin(yaw), np.cos(yaw)]
    eye = np.c_[xy - back * fwd + side[:, None] * left, np.full(len(xy), height)]
    tgt = np.c_[xy + 0.5 * fwd, np.full(len(xy), 0.8)]
    return [look_at(e, t) for e, t in zip(eye, tgt)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, nargs="*")
    ap.add_argument("--range", type=int, nargs=2)
    ap.add_argument("--encode", action="store_true")
    ap.add_argument("--src", type=Path, default=SRC)
    args = ap.parse_args()
    sc = SCENES["linden"]
    z = np.load(str(args.src) + ".npz", allow_pickle=True)
    qpos, fps = z["qpos"], int(z["fps"])
    frames_dir = ROOT / "data" / "linden" / "carry_frames"
    if args.encode:
        return encode("linden_carry", frames_dir, fps, len(qpos))
    model = mujoco.MjModel.from_binary_path(str(args.src) + ".mjb")
    model.vis.global_.offwidth, model.vis.global_.offheight = 1920, 1080   # 2x supersampled robot pass
    data = mujoco.MjData(model)
    splats = SplatScene(sc.dir / "splats.npz")
    comp = Compositor(model, sc, splats, 480, 360, robot_gain=sc.robot_gain)
    chase = chase_poses(qpos, [str(l) for l in z["labels"]])
    if args.frames:
        idx, out = args.frames, ROOT / "data" / "linden" / "carry_stills"
    else:
        lo, hi = args.range if args.range else (0, len(qpos))
        idx, out = range(lo, min(hi, len(qpos))), frames_dir
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for n, i in enumerate(idx):
        data.qpos[:] = qpos[i]
        set_chase(model, data, chase[i])
        mujoco.mj_forward(model, data)
        c = comp.render(data, "chase")
        h = comp.render(data, "head", want_depth=True)
        panel = np.concatenate([
            label((c["rgb"] * 255).astype(np.uint8), f"third person  |  {z['labels'][i]}"),
            label((h["rgb"] * 255).astype(np.uint8), "head camera RGB"),
            label(colorize_depth(h["depth"]), f"head camera depth (0-{DEPTH_MAX:.0f} m)"),
        ], axis=1)
        iio.imwrite(out / f"{i:05d}.png", panel)
        if n % 50 == 0:
            print(f"  frame {n}/{len(idx)}  {(time.time() - t0) / (n + 1):.2f}s/frame", flush=True)


if __name__ == "__main__":
    main()
