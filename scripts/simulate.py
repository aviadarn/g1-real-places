"""Plan a walk through a scene and run the G1 along it. Writes data/<scene>/walk.npz.

    python scripts/simulate.py leake
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from g1places.collision import Grid, smooth_path            # noqa: E402
from g1places.scenes import SCENES                          # noqa: E402
from g1places.sim import Walker, build_model                # noqa: E402

FPS = 25
warnings.filterwarnings("ignore", message=".*torch.jit.load.*")


def plan_walk(grid: Grid, length: float, clearance: float, lead_in: float):
    """Longest corridor through the biggest free region, cut to `length` metres
    around the point closest to the reconstruction origin (where capture started)."""
    blocked = grid.occ | ~grid.known
    free = ndimage.distance_transform_edt(~blocked) * grid.cell > clearance + 0.1
    lab, n = ndimage.label(free)
    big = lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1)
    ij = np.argwhere(big)
    xy = grid.to_xy(ij)
    mu = xy.mean(0)
    axis = np.linalg.svd(xy - mu, full_matrices=False)[2][0]
    t = (xy - mu) @ axis
    a, b = xy[np.argmin(t)], xy[np.argmax(t)]
    full = smooth_path(grid.plan(a, b, clearance=clearance))
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(full, axis=0), axis=1))]
    centre = s[np.argmin(np.linalg.norm(full, axis=1))]
    lo = np.clip(centre - length / 2, 0, max(0, s[-1] - length))
    keep = (s >= lo) & (s <= lo + length + lead_in)
    return full[keep], full


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene", choices=sorted(SCENES))
    ap.add_argument("--length", type=float, default=22.0, help="metres of walking")
    ap.add_argument("--speed", type=float, default=0.6)
    ap.add_argument("--clearance", type=float, default=0.6)
    ap.add_argument("--reverse", action="store_true", help="walk the corridor the other way")
    args = ap.parse_args()
    sc = SCENES[args.scene]
    lead_in = 3.0

    grid = Grid(np.load(sc.dir / "pts_world.npy"))
    path, full = plan_walk(grid, args.length, args.clearance, lead_in)
    if args.reverse:
        path = path[::-1].copy()
    # only obstacles the robot could reach; thousands of far-away boxes just cost compile time
    boxes = [b for b in grid.boxes()
             if np.min(np.linalg.norm(path - b[0][:2], axis=1)) < 3.0 + np.linalg.norm(b[1][:2])]
    model = build_model(boxes)
    w = Walker(model, path, speed=args.speed)
    w.reset(lead_in)
    print(f"{sc.name}: {len(boxes)} obstacle boxes, path {w.s[-1]:.1f} m, start {path[0].round(2)}")

    qpos, cmds, contacts = [], [], 0
    steps_per_frame = 50 // FPS if 50 % FPS == 0 else None
    assert steps_per_frame, "FPS must divide the 50 Hz control rate"
    k = 0
    max_steps = int((w.s[-1] / (args.speed * 0.5) + 10) * 50)
    while not w.done and k < max_steps:
        w.step_control()
        contacts += w.obstacle_contacts() > 0
        if k % steps_per_frame == 0:
            qpos.append(w.data.qpos.copy())
            cmds.append(w.cmd.copy())
        if w.fallen():
            print(f"FELL at t={k / 50:.1f}s, progress {w.progress():.1f} m")
            break
        k += 1
    qpos = np.array(qpos)
    dev = np.min(np.linalg.norm(path[None] - qpos[:, None, :2], axis=2), axis=1)
    print(f"walked {w.progress() - lead_in:.1f} m in {k / 50:.1f} s, fell={w.fallen()}, "
          f"control steps touching obstacles={contacts}, path deviation mean {dev.mean():.2f} max {dev.max():.2f} m")
    np.savez(sc.dir / "walk.npz", qpos=qpos, cmd=np.array(cmds), path=path, full_path=full,
             fps=FPS, lead_in=lead_in, fell=w.fallen(), obstacle_steps=contacts,
             occ=grid.occ, known=grid.known, grid_lo=grid.lo, grid_cell=grid.cell)


if __name__ == "__main__":
    main()
