"""Carry figures: success rate vs payload (three policies) and training curves.

    python scripts/figures_carry.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3dd"
SERIES = {  # fixed categorical order (reference palette slots 1-3)
    "payload10_1B": ("trained with 0-10 kg in the hands", "#2a78d6"),
    "payload0_1B": ("same training, no payload (control)", "#eb6834"),
    "unitree_stock": ("Unitree's stock walking policy", "#1baf7a"),
}
LABEL_SHORT = {"payload10_1B": "payload-trained", "payload0_1B": "control", "unitree_stock": "Unitree stock"}


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(colors=INK2, labelsize=9)


def success_panel(ax, prefix: str, title: str):
    style(ax)
    for key, (name, color) in SERIES.items():
        f = ROOT / "results" / f"{prefix}_{key}.json"
        if not f.exists():
            continue
        r = json.loads(f.read_text())["results"]
        keys = sorted(r, key=float)
        kg = np.array([float(x) for x in keys])
        k = np.array([r[x]["success"] for x in keys])
        n = np.array([r[x]["trials"] for x in keys])
        lo, hi = np.array([wilson(a, b) for a, b in zip(k, n)]).T
        ax.fill_between(kg, lo * 100, hi * 100, color=color, alpha=0.12, lw=0)
        ax.plot(kg, k / n * 100, color=color, lw=2, marker="o", ms=8, mec=SURFACE, mew=2, label=name)
    ax.set_xlabel("payload at the palms (kg)", color=INK2, fontsize=9.5)
    ax.set_ylim(-3, 103)
    ax.set_xlim(-0.5, 10.5)
    ax.set_xticks([0, 2, 4, 6, 8, 10])
    ax.set_title(title, loc="left", fontsize=10, color=INK)


def success_curve():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0), facecolor=SURFACE, sharey=True)
    success_panel(axes[0], "carry", "Walk 15 s at 0.5 m/s, three 0.5 m/s shoves")
    success_panel(axes[1], "stand", "Stand still 10 s (zero command)")
    axes[0].set_ylabel("trials without a fall (%)", color=INK2, fontsize=9.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=9, labelcolor=INK)
    fig.text(0.01, 0.97, "Arms in a carry pose; 50 trials per point; bands are 95% Wilson intervals",
             fontsize=8.5, color=INK2, va="top")
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    fig.savefig(ROOT / "media" / "carry_success.png", dpi=170, facecolor=SURFACE)
    plt.close(fig)


def training_curves():
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), facecolor=SURFACE)
    for ax in axes:
        style(ax)
    for key in ("payload10_1B", "payload0_1B"):
        name, color = SERIES[key]
        base = key.replace("_1B", "")
        rows, offset = [], 0
        for run in (base, base + "_800M", key):
            f = ROOT / "runs" / run / "progress.json"
            p = json.loads(f.read_text())
            if rows:
                p = p[1:]                      # step 0 of a continuation repeats the restore
            rows += [{**r, "step": r["step"] + offset} for r in p]
            offset = rows[-1]["step"]
        s_ = np.array([r["step"] for r in rows]) / 1e6
        axes[0].plot(s_, [r["reward"] for r in rows], color=color, lw=2, label=name)
        axes[1].plot(s_, [r["episode_length"] for r in rows], color=color, lw=2, label=name)
    axes[0].set_title("eval episode reward", loc="left", fontsize=9.5, color=INK)
    axes[1].set_title("eval episode length (of 1000 control steps)", loc="left", fontsize=9.5, color=INK)
    for ax in axes:
        ax.set_xlabel("environment steps (millions)", color=INK2, fontsize=9)
        for x in (200, 800):
            ax.axvline(x, color=INK2, lw=0.8, ls=":")
    axes[0].text(805, axes[0].get_ylim()[0], " stand fine-tune", color=INK2, fontsize=8, va="bottom")
    axes[1].legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="lower center", bbox_to_anchor=(0.45, 0.02))
    fig.tight_layout()
    fig.savefig(ROOT / "media" / "carry_training.png", dpi=170, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    plt.rcParams["font.family"] = "Helvetica"
    success_curve()
    training_curves()
    print("wrote media/carry_success.png, media/carry_training.png")
