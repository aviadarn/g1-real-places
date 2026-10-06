"""Top-down map of each walk: scanned floor, obstacle cells the robot collides with, robot track.

    python scripts/figures.py map
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from g1places.scenes import SCENES   # noqa: E402

SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
FLOOR, OBST, TRACK = "#e4e3dd", "#8a8983", "#2a78d6"


def walk_map(ax, name: str, margin: float = 5.0):
    sc = SCENES[name]
    w = np.load(sc.dir / "walk.npz")
    occ, known, lo, cell = w["occ"], w["known"], w["grid_lo"], float(w["grid_cell"])
    img = np.zeros(occ.shape, int)
    img[known] = 1
    img[occ] = 2
    track = w["qpos"][:, :2]
    ext = [lo[0], lo[0] + occ.shape[0] * cell, lo[1], lo[1] + occ.shape[1] * cell]
    ax.imshow(img.T, origin="lower", extent=ext, interpolation="nearest",
              cmap=ListedColormap([SURFACE, FLOOR, OBST]), vmin=0, vmax=2)
    ax.plot(track[:, 0], track[:, 1], color=TRACK, lw=2, solid_capstyle="round")
    for p, txt in [(track[0], "start"), (track[-1], "end")]:
        ax.plot(*p, "o", ms=8, color=TRACK, mec=SURFACE, mew=2)
        ax.annotate(txt, p, xytext=(6, 6), textcoords="offset points", color=INK, fontsize=9)
    lo_xy, hi_xy = track.min(0) - margin, track.max(0) + margin
    ax.set_xlim(lo_xy[0], hi_xy[0])
    ax.set_ylim(lo_xy[1], hi_xy[1])
    ax.set_aspect("equal")
    dist = np.sum(np.linalg.norm(np.diff(track, axis=0), axis=1))
    secs = len(track) / int(w["fps"])
    ax.set_title(f"{sc.title}\n{dist:.1f} m walked in {secs:.0f} s, "
                 f"{int(w['obstacle_steps'])} obstacle contacts, no falls",
                 loc="left", fontsize=10, color=INK)
    ax.set_xlabel("x (m)", color=INK2, fontsize=9)
    ax.set_ylabel("y (m)", color=INK2, fontsize=9)
    ax.tick_params(colors=INK2, labelsize=8)
    for s in ax.spines.values():
        s.set_visible(False)


def main():
    plt.rcParams["font.family"] = "Helvetica"
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), facecolor=SURFACE,
                             gridspec_kw={"width_ratios": [1.6, 1]})
    for ax, name in zip(axes, ["leake", "linden"]):
        ax.set_facecolor(SURFACE)
        walk_map(ax, name)
    fig.legend(handles=[Patch(color=FLOOR, label="scanned floor"),
                        Patch(color=OBST, label="obstacle, 0.15-1.6 m above floor (MuJoCo boxes)"),
                        plt.Line2D([], [], color=TRACK, lw=2, label="G1 pelvis track")],
               loc="lower center", ncol=3, frameon=False, fontsize=9, labelcolor=INK)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    out = ROOT / "media" / "walk_maps.png"
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    print("wrote", out)


if __name__ == "__main__":
    main()
