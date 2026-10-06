"""Collision mesh -> 2.5D occupancy grid -> MuJoCo boxes, plus a grid planner.

MuJoCo collides against the convex hull of a mesh geom, so a scanned room loaded
as one mesh becomes a solid lump with the robot inside it. Instead we keep the
geometry that can touch a 1.3 m humanoid (a height band above the floor), rasterise
it to a 2D grid, and merge occupied cells into boxes. The floor is a plane at z=0:
the walking policy was trained on flat ground.
"""

from __future__ import annotations

import heapq

import numpy as np
from scipy import ndimage


def sample_surface(v: np.ndarray, f: np.ndarray, spacing: float, rng=None) -> np.ndarray:
    """Points on the triangles, roughly one per spacing^2 of area (at least the vertices)."""
    rng = rng or np.random.default_rng(0)
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    n = np.floor(area / spacing**2 + rng.random(len(f))).astype(int)
    idx = np.repeat(np.arange(len(f)), n)
    r1, r2 = rng.random(len(idx)), rng.random(len(idx))
    flip = r1 + r2 > 1
    r1[flip], r2[flip] = 1 - r1[flip], 1 - r2[flip]
    pts = a[idx] + r1[:, None] * (b[idx] - a[idx]) + r2[:, None] * (c[idx] - a[idx])
    return np.concatenate([v, pts])


class Grid:
    def __init__(self, pts_world: np.ndarray, cell: float = 0.1,
                 band=(0.15, 1.6), floor_band=(-0.15, 0.1), min_pts: int = 3):
        self.cell = cell
        xy = pts_world[:, :2]
        self.lo = np.floor(xy.min(0) / cell) * cell - cell
        shape = tuple((np.ceil((xy.max(0) - self.lo) / cell) + 2).astype(int))
        z = pts_world[:, 2]

        def raster(sel):
            ij = ((xy[sel] - self.lo) / cell).astype(int)
            g = np.zeros(shape, np.int32)
            np.add.at(g, (ij[:, 0], ij[:, 1]), 1)
            return g

        self.occ = raster((z > band[0]) & (z < band[1])) >= min_pts
        floor = raster((z > floor_band[0]) & (z < floor_band[1])) >= 1
        # scanned floor, holes closed: cells we can say anything about
        self.known = ndimage.binary_closing(floor, iterations=3) | self.occ

    def to_ij(self, xy) -> tuple[int, int]:
        i, j = ((np.asarray(xy) - self.lo) / self.cell).astype(int)
        return int(i), int(j)

    def to_xy(self, ij) -> np.ndarray:
        return self.lo + (np.asarray(ij) + 0.5) * self.cell

    def boxes(self, height: float = 2.0) -> list[tuple[np.ndarray, np.ndarray]]:
        """Greedy rectangle cover of occupied cells -> (centre xyz, half-size xyz)."""
        occ = self.occ.copy()
        out = []
        W, H = occ.shape
        for i in range(W):
            j = 0
            while j < H:
                if not occ[i, j]:
                    j += 1
                    continue
                j1 = j
                while j1 < H and occ[i, j1]:
                    j1 += 1
                i1 = i + 1
                while i1 < W and occ[i1, j:j1].all():
                    i1 += 1
                occ[i:i1, j:j1] = False
                lo = self.lo + np.array([i, j]) * self.cell
                hi = self.lo + np.array([i1, j1]) * self.cell
                out.append((np.r_[(lo + hi) / 2, height / 2], np.r_[(hi - lo) / 2, height / 2]))
                j = j1
        return out

    def plan(self, start_xy, goal_xy, clearance: float = 0.45) -> np.ndarray:
        """A* on the grid, keeping `clearance` metres from anything occupied or unscanned."""
        blocked = self.occ | ~self.known
        dist = ndimage.distance_transform_edt(~blocked) * self.cell
        free = dist > clearance
        # soft preference for the middle of corridors
        cost_extra = np.clip(1.0 - dist, 0, None) * 4.0
        s, g = self.to_ij(start_xy), self.to_ij(goal_xy)
        free_ij = np.argwhere(free)

        def snap(ij, what):
            # a robot standing at a shelf is inside the clearance band: start from the
            # nearest free cell, if there is one within a metre
            if free[ij]:
                return ij
            k = np.argmin(np.linalg.norm(free_ij - np.array(ij), axis=1))
            if np.linalg.norm(free_ij[k] - np.array(ij)) * self.cell > 1.0:
                raise ValueError(f"{what} is more than 1 m from free space (clearance {clearance})")
            return tuple(int(x) for x in free_ij[k])

        s, g = snap(s, "start"), snap(g, "goal")
        moves = [(1, 0, 1), (-1, 0, 1), (0, 1, 1), (0, -1, 1),
                 (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414)]
        openq = [(0.0, s)]
        came, cost = {s: None}, {s: 0.0}
        while openq:
            _, cur = heapq.heappop(openq)
            if cur == g:
                break
            for di, dj, w in moves:
                nb = (cur[0] + di, cur[1] + dj)
                if not (0 <= nb[0] < free.shape[0] and 0 <= nb[1] < free.shape[1]) or not free[nb]:
                    continue
                nc = cost[cur] + w + cost_extra[nb]
                if nc < cost.get(nb, np.inf):
                    cost[nb] = nc
                    came[nb] = cur
                    h = np.hypot(g[0] - nb[0], g[1] - nb[1])
                    heapq.heappush(openq, (nc + h, nb))
        if g not in came:
            raise ValueError("no path")
        path, cur = [], g
        while cur is not None:
            path.append(cur)
            cur = came[cur]
        return np.array([self.to_xy(p) for p in path[::-1]])


def smooth_path(xy: np.ndarray, step: float = 0.25, window: int = 9) -> np.ndarray:
    """Resample to even spacing, then moving-average so the heading changes gently."""
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    s = np.r_[0, np.cumsum(seg)]
    t = np.arange(0, s[-1], step)
    out = np.c_[np.interp(t, s, xy[:, 0]), np.interp(t, s, xy[:, 1])]
    if len(out) > window:
        k = np.ones(window) / window
        pad = window // 2
        padded = np.pad(out, ((pad, pad), (0, 0)), mode="edge")
        out = np.c_[np.convolve(padded[:, 0], k, "valid"), np.convolve(padded[:, 1], k, "valid")]
    return out


def fit_floor(pts: np.ndarray, cell: float = 0.5, radius: float = 12.0,
              zwin: float = 0.5) -> tuple[np.ndarray, np.ndarray, dict]:
    """Robust plane fit to the floor near the scene centre; returns (R, t, stats) with
    R rotating the plane normal onto +z and t putting the plane at z=0 under the origin.

    Floor samples are the 10th-percentile z per `cell`, so walls and clutter above the
    floor do not pull the fit; residual outliers (kerbs, debris) are dropped iteratively.
    """
    sel = (np.linalg.norm(pts[:, :2], axis=1) < radius) & (np.abs(pts[:, 2]) < zwin)
    q = pts[sel]
    ij = np.floor(q[:, :2] / cell).astype(np.int64)
    order = np.lexsort((ij[:, 1], ij[:, 0]))
    ij, z = ij[order], q[order, 2]
    brk = np.r_[0, np.flatnonzero(np.any(np.diff(ij, axis=0) != 0, axis=1)) + 1, len(ij)]
    keys = ij[brk[:-1]]
    fz = np.array([np.percentile(z[a:b], 10) for a, b in zip(brk[:-1], brk[1:])])
    A = np.c_[(keys + 0.5) * cell, np.ones(len(fz))]
    keep = np.ones(len(fz), bool)
    for _ in range(6):
        coef, *_ = np.linalg.lstsq(A[keep], fz[keep], rcond=None)
        r = fz - A @ coef
        keep = np.abs(r) < 3 * np.median(np.abs(r[keep])) + 0.01
    a, b, c = coef
    n = np.array([-a, -b, 1.0])
    n /= np.linalg.norm(n)
    v = np.cross(n, [0, 0, 1.0])
    s, cth = np.linalg.norm(v), n[2]
    if s < 1e-12:
        R = np.eye(3)
    else:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + vx + vx @ vx * ((1 - cth) / s**2)
    t = np.array([0.0, 0.0, -(R @ np.array([0, 0, c]))[2]])
    stats = {"tilt_deg": float(np.degrees(np.arccos(cth))), "offset_m": float(c),
             "residual_mad_m": float(np.median(np.abs(r[keep]))), "cells": int(keep.sum())}
    return R, t, stats
