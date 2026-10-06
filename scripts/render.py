"""Render a recorded walk as three synchronised views: chase camera, head RGB, head depth.

    python scripts/render.py leake                       # all frames, then media/leake.mp4
    python scripts/render.py leake --range 0 450         # a chunk of frames (no encode)
    python scripts/render.py leake --encode              # encode existing frames only
    python scripts/render.py leake --frames 0 300        # stills only, for checking

macOS throttles GPU work from background processes hard (~7x here), so long renders
are run in foreground chunks with --range.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import imageio.v3 as iio
import matplotlib
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter1d

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from g1places.composite import CV_FROM_GL, Compositor     # noqa: E402
from g1places.scenes import SCENES                         # noqa: E402
from g1places.sim import build_model, yaw_of               # noqa: E402
from g1places.splat import SplatScene, look_at             # noqa: E402

MEDIA = ROOT / "media"
DEPTH_MAX = 20.0


def chase_poses(qpos: np.ndarray, back: float = 2.3, height: float = 1.4, sigma: float = 12):
    """Camera trailing the robot along its own smoothed track, looking past its shoulder."""
    xy = qpos[:, :2]
    yaw = np.unwrap([yaw_of(q[3:7]) for q in qpos])
    yaw_s = gaussian_filter1d(yaw, sigma)
    xy_s = gaussian_filter1d(xy, sigma, axis=0)
    fwd = np.c_[np.cos(yaw_s), np.sin(yaw_s)]
    eye = np.c_[xy_s - back * fwd, np.full(len(xy), height)]
    tgt = np.c_[xy_s + 1.5 * fwd, np.full(len(xy), 0.75)]
    return [look_at(e, t) for e, t in zip(eye, tgt)]


def set_chase(model, data, c2w_cv):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "chase_rig")
    mid = model.body_mocapid[bid]
    data.mocap_pos[mid] = c2w_cv[:3, 3]
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, (c2w_cv[:3, :3] @ CV_FROM_GL).flatten())
    data.mocap_quat[mid] = q


def colorize_depth(d: np.ndarray) -> np.ndarray:
    # sqrt spreads the near range, where the walls and floor of a corridor sit
    x = np.sqrt(np.clip(np.where(np.isfinite(d), d, DEPTH_MAX) / DEPTH_MAX, 0, 1))
    return (matplotlib.colormaps["turbo"](1 - x)[..., :3] * 255).astype(np.uint8)


def label(img: np.ndarray, text: str) -> np.ndarray:
    im = Image.fromarray(img)
    dr = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", max(12, img.shape[0] // 22))
    except OSError:
        font = ImageFont.load_default()
    x0, y0, x1, y1 = dr.textbbox((10, 8), text, font=font)
    dr.rectangle((x0 - 6, y0 - 4, x1 + 6, y1 + 4), fill=(0, 0, 0))
    dr.text((10, 8), text, fill=(255, 255, 255), font=font)
    return np.asarray(im)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene", choices=sorted(SCENES))
    ap.add_argument("--width", type=int, default=480)
    ap.add_argument("--height", type=int, default=360)
    ap.add_argument("--frames", type=int, nargs="*", help="render only these frame indices as PNGs")
    ap.add_argument("--range", type=int, nargs=2, metavar=("START", "END"), help="render frames [START, END)")
    ap.add_argument("--encode", action="store_true", help="only encode existing frames to mp4")
    ap.add_argument("--robot-gain", type=float, default=None, help="default: per scene")
    ap.add_argument("--chase-fovy", type=float, default=60.0)
    ap.add_argument("--head-fovy", type=float, default=70.0)
    args = ap.parse_args()

    sc = SCENES[args.scene]
    walk = np.load(sc.dir / "walk.npz")
    qpos, fps = walk["qpos"], int(walk["fps"])
    model = build_model([])
    model.cam_fovy[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "chase")] = args.chase_fovy
    model.cam_fovy[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "head")] = args.head_fovy
    data = mujoco.MjData(model)
    frames_dir = sc.dir / "frames"
    if args.encode:
        return encode(sc.name, frames_dir, fps, len(qpos))
    t = time.time()
    splats = SplatScene(sc.dir / "splats.npz")
    print(f"{sc.name}: {len(splats):,} gaussians loaded in {time.time() - t:.1f}s, {len(qpos)} frames")
    comp = Compositor(model, sc, splats, args.width, args.height, robot_gain=args.robot_gain or sc.robot_gain)
    chase = chase_poses(qpos)

    if args.frames:
        idx = args.frames
    elif args.range:
        idx = range(args.range[0], min(args.range[1], len(qpos)))
    else:
        idx = range(len(qpos))
    out_dir = sc.dir / "stills" if args.frames else frames_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for n, i in enumerate(idx):
        data.qpos[:] = qpos[i]
        set_chase(model, data, chase[i])
        mujoco.mj_forward(model, data)
        c = comp.render(data, "chase")
        h = comp.render(data, "head", want_depth=True, depth_scale=1)
        panel = np.concatenate([
            label((c["rgb"] * 255).astype(np.uint8), "third person"),
            label((h["rgb"] * 255).astype(np.uint8), "head camera RGB"),
            label(colorize_depth(h["depth"]), f"head camera depth (0-{DEPTH_MAX:.0f} m)"),
        ], axis=1)
        iio.imwrite(out_dir / f"{i:05d}.png", panel)
        if n % 50 == 0:
            el = time.time() - t0
            print(f"  frame {n}/{len(idx)}  {el / (n + 1):.2f}s/frame", flush=True)

    if not (args.frames or args.range):
        encode(sc.name, frames_dir, fps, len(qpos))


def encode(name: str, frames_dir: Path, fps: int, n: int) -> None:
    missing = [i for i in range(n) if not (frames_dir / f"{i:05d}.png").exists()]
    if missing:
        raise SystemExit(f"{len(missing)} frames missing, first {missing[:5]}")
    MEDIA.mkdir(exist_ok=True)
    mp4 = MEDIA / f"{name}.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i",
                    str(frames_dir / "%05d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", "23", "-movflags", "+faststart", str(mp4)], check=True)
    print("wrote", mp4)


if __name__ == "__main__":
    main()
