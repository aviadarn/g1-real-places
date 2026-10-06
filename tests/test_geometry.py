"""Geometry checks that need no GPU and no scene data.

    .venv/bin/python -m pytest tests
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from g1places.collision import Grid, fit_floor          # noqa: E402
from g1places.scenes import Y_UP_TO_Z_UP, Scene, rot_z  # noqa: E402
from g1places.splat import look_at                      # noqa: E402


def make_scene(R, scale, tmp_path, level=None):
    sc = Scene(name="t", usdz="", scale=scale, R=R, origin=np.array([0.3, -1.2, 0.5]))
    sc.__dict__["level"] = level or (np.eye(3), np.zeros(3))   # bypass floor.json
    return sc


def test_world_to_file_pose_maps_points_like_to_world(tmp_path):
    level = (rot_z(0) @ np.array([[1, 0, 0], [0, np.cos(0.02), -np.sin(0.02)], [0, np.sin(0.02), np.cos(0.02)]]),
             np.array([0, 0, -0.1]))
    for R, s in [(rot_z(60.59), 5.4), (Y_UP_TO_Z_UP, 1.0)]:
        sc = make_scene(R, s, tmp_path, level)
        c2w = look_at((2.0, -1.0, 1.4), (5.0, 0.5, 0.8))
        f = sc.world_to_file_pose(c2w)
        assert np.allclose(f[:3, :3].T @ f[:3, :3], np.eye(3), atol=1e-9), "file pose must stay rigid"
        # a point 3 m in front of the world camera, seen from the file-frame camera,
        # must sit at the same pixel and at depth 3/scale
        p_world = c2w[:3, 3] + 3.0 * c2w[:3, 2]
        p_file = np.linalg.solve(level[0] @ (s * R), (p_world - level[1])) + sc.origin
        assert np.allclose(sc.to_world(p_file[None])[0], p_world, atol=1e-9)
        cam = np.linalg.inv(f) @ np.r_[p_file, 1]
        assert np.allclose(cam[:3], [0, 0, 3.0 / s], atol=1e-9)


def test_fit_floor_recovers_tilt():
    rng = np.random.default_rng(0)
    xy = rng.uniform(-10, 10, (40000, 2))
    z = 0.01 * xy[:, 0] - 0.005 * xy[:, 1] + 0.1 + rng.normal(0, 0.005, len(xy))
    clutter = np.c_[rng.uniform(-10, 10, (4000, 2)), rng.uniform(0.2, 0.45, 4000)]
    R, t, stats = fit_floor(np.r_[np.c_[xy, z], clutter])
    expected = np.degrees(np.arctan(np.hypot(0.01, 0.005)))
    assert abs(stats["tilt_deg"] - expected) < 0.05
    levelled = np.c_[xy, z] @ R.T + t
    assert np.abs(np.median(levelled[:, 2])) < 0.01
    assert np.std(levelled[:, 2]) < 0.01


def test_boxes_cover_exactly_the_occupied_cells():
    rng = np.random.default_rng(1)
    floor = np.c_[rng.uniform(0, 5, (20000, 2)), np.zeros(20000)]
    wall = np.c_[rng.uniform(1, 4, (5000, 1)), np.full((5000, 1), 2.55), rng.uniform(0.3, 1.2, (5000, 1))]
    g = Grid(np.r_[floor, wall], min_pts=1)
    covered = np.zeros_like(g.occ)
    for c, h in g.boxes():
        lo = np.round((c[:2] - h[:2] - g.lo) / g.cell).astype(int)
        hi = np.round((c[:2] + h[:2] - g.lo) / g.cell).astype(int)
        assert not covered[lo[0]:hi[0], lo[1]:hi[1]].any(), "boxes overlap"
        covered[lo[0]:hi[0], lo[1]:hi[1]] = True
    assert (covered == g.occ).all()
