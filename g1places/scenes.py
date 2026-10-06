"""Per-scene frame fixes: file frame -> world frame (Z-up, metres, floor at z=0).

    world = scale * R @ (file - origin)

The two sample USDZs disagree with each other and with their own metadata:

  * Leake Street: splat is Z-up, mesh is rotated into it by gauss.usda, but the
    whole reconstruction is ~5x too small. Walls measure 1.48 units apart and the
    tunnel is published as 8 m wide -> scale 5.4. Cross-check: the reconstruction
    origin sits 0.33 units (1.8 m at that scale) above the floor, which is where a
    handheld 360 camera would be.
  * Linden: metric already, but splat and mesh are both Y-up inside a stage that
    declares Z-up. The origin sits 1.79 m above the floor, same camera height.

On top of that hand-set transform, `prepare.py` fits the floor plane and levels it
(Leake's floor is tilted 0.69 deg in the file, Linden's 0.19 deg); the fit lives in
data/<scene>/floor.json and is applied by `to_world` / `world_to_file_pose`.

Splats are never rotated: cameras are mapped into the file frame instead
(`world_to_file_pose`), so SH view-dependence and Gaussian shapes stay untouched.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def rot_z(deg: float) -> np.ndarray:
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


Y_UP_TO_Z_UP = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0.0]])   # +y_file -> +z_world


@dataclass
class Scene:
    name: str
    usdz: str
    scale: float
    R: np.ndarray
    origin: np.ndarray                 # file-frame point that becomes world (0,0,0)
    title: str = ""
    robot_gain: float = 1.0          # dims MuJoCo's lighting to match the capture's
    waypoints: list = field(default_factory=list)   # world xy, metres

    @property
    def dir(self) -> Path:
        return DATA / self.name

    @cached_property
    def level(self) -> tuple[np.ndarray, np.ndarray]:
        """(rotation, translation) that levels the fitted floor to z=0; identity if not fitted yet."""
        f = self.dir / "floor.json"
        if not f.exists():
            return np.eye(3), np.zeros(3)
        d = json.loads(f.read_text())
        return np.array(d["R"]), np.array(d["t"])

    def to_world(self, p: np.ndarray, level: bool = True) -> np.ndarray:
        w = self.scale * (np.asarray(p) - self.origin) @ self.R.T
        if not level:
            return w
        Rl, tl = self.level
        return w @ Rl.T + tl

    def world_to_file_pose(self, c2w: np.ndarray) -> np.ndarray:
        """Rigid world camera pose -> rigid file-frame pose (translation unscaled)."""
        Rl, tl = self.level
        Rw = Rl.T @ c2w[:3, :3]                  # undo levelling
        pw = Rl.T @ (c2w[:3, 3] - tl)
        out = np.eye(4)
        out[:3, :3] = self.R.T @ Rw
        out[:3, 3] = self.R.T @ pw / self.scale + self.origin
        return out


SCENES = {
    "leake": Scene(
        name="leake",
        usdz="LeakeStreetArches.usdz",
        title="Leake Street Arches, London",
        robot_gain=0.6,
        scale=5.4,
        # tunnel axis in the file frame is (0.491, -0.871): rotate it onto +x
        R=rot_z(60.59),
        origin=np.array([-0.2, 0.35, -0.33]),
    ),
    "linden": Scene(
        name="linden",
        usdz="Linden_Warehouse_Pallet_Racks.usdz",
        title="Warehouse with pallet racks, Linden NJ",
        robot_gain=0.85,
        scale=1.0,
        R=Y_UP_TO_Z_UP,
        origin=np.array([0.0, -1.79, 0.0]),
    ),
}
