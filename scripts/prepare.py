"""Unpack a Niantic Places Library USDZ and build the collision point cloud.

    python scripts/prepare.py leake  ~/Downloads/LeakeStreetArches.usdz
    python scripts/prepare.py linden ~/Downloads/Linden_Warehouse_Pallet_Racks.usdz

Writes data/<scene>/{splats.npz, mesh.npz, pts_world.npy}.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from g1places.collision import fit_floor, sample_surface    # noqa: E402
from g1places.scene import unpack                # noqa: E402
from g1places.scenes import SCENES               # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene", choices=sorted(SCENES))
    ap.add_argument("usdz", type=Path)
    args = ap.parse_args()
    sc = SCENES[args.scene]
    unpack(args.usdz.expanduser(), sc.dir)

    m = np.load(sc.dir / "mesh.npz")
    v = sc.to_world(m["vertices"].astype(np.float64), level=False)
    f = m["faces"]
    # Linden's mesh (27M triangles) is dense enough that its vertices already cover
    # every 10 cm cell; Leake's (0.5M) needs points sampled across each triangle.
    pts = v if len(f) > 3_000_000 else sample_surface(v, f, spacing=0.05)
    R, t, stats = fit_floor(pts)
    (sc.dir / "floor.json").write_text(json.dumps({"R": R.tolist(), "t": t.tolist(), **stats}, indent=1))
    print(f"  floor fit: tilt {stats['tilt_deg']:.2f} deg, offset {stats['offset_m']:.3f} m, "
          f"residual MAD {stats['residual_mad_m'] * 100:.1f} cm over {stats['cells']} cells")
    pts = pts @ R.T + t
    v = v @ R.T + t
    np.save(sc.dir / "pts_world.npy", pts.astype(np.float32))
    h, e = np.histogram(v[:, 2], bins=np.arange(-1, 3, 0.02))
    print(f"  world bounds {v.min(0).round(1)} .. {v.max(0).round(1)}, floor mode z={e[np.argmax(h)]:.2f} m")


if __name__ == "__main__":
    main()
