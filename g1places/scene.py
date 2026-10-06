"""Unpack a Niantic Spatial / NuRec USDZ into arrays this project can use.

A Places Library USDZ holds four files:

    default.usda   references gauss.usda
    gauss.usda     a NuRec Volume prim pointing at the .nurec, plus the mesh prim
                   and the transform that puts the mesh in the volume's frame
    *.nurec        gzip(msgpack) of a 3DGUT checkpoint: 3D Gaussians, fp16
    mesh.usd       the collision mesh, authored Y-up

The stage is Z-up, metres. Positions here come out in that frame, which is
also MuJoCo's, so nothing downstream needs an axis swap.
"""

from __future__ import annotations

import gzip
import re
import zipfile
from pathlib import Path

import msgpack
import numpy as np

PREFIX = ".gaussians_nodes.gaussians."


def _member(z: zipfile.ZipFile, suffix: str) -> str:
    names = [n for n in z.namelist() if n.endswith(suffix)]
    if len(names) != 1:
        raise ValueError(f"expected one *{suffix} in the usdz, found {names}")
    return names[0]


def load_gaussians(usdz: Path) -> dict[str, np.ndarray]:
    """Decode the .nurec into activated-parameter arrays.

    Returns means (N,3), quats (N,4) wxyz, log_scales (N,3), density_logits (N,),
    sh (N,16,3). Activations follow the checkpoint's own config: sigmoid for
    density, exp for scale, normalize for rotation.
    """
    with zipfile.ZipFile(usdz) as z, z.open(_member(z, ".nurec")) as f:
        blob = msgpack.unpackb(gzip.GzipFile(fileobj=f).read(), raw=False,
                               strict_map_key=False)
    nre = blob["nre_data"]
    layer = nre["config"]["layers"]["gaussians"]
    if layer["precision"] != 16:
        raise ValueError(f"only fp16 checkpoints handled, got precision={layer['precision']}")
    act = (layer["density_activation"], layer["scale_activation"], layer["rotation_activation"])
    if act != ("sigmoid", "exp", "normalize"):
        raise ValueError(f"unexpected activations {act}")
    sd = nre["state_dict"]

    def tensor(name: str) -> np.ndarray:
        shape = tuple(sd[PREFIX + name + ".shape"])
        return np.frombuffer(sd[PREFIX + name], dtype=np.float16).reshape(shape)

    albedo = tensor("features_albedo")                       # (N,3)  SH band 0
    specular = tensor("features_specular")                   # (N,45) SH bands 1..3
    n = albedo.shape[0]
    # 3DGUT keeps the higher bands coefficient-major: [c0_rgb, c1_rgb, ...].
    # Checked by rendering both readings: scripts/check_sh_layout.py.
    sh = np.concatenate([albedo[:, None, :], specular.reshape(n, 15, 3)], axis=1)
    return {
        "means": tensor("positions"),
        "quats": tensor("rotations"),
        "log_scales": tensor("scales"),
        "density_logits": tensor("densities").reshape(-1),
        "sh": sh,
    }


def _mesh_transform(usdz: Path) -> np.ndarray:
    """The 4x4 that gauss.usda authors on the `over "mesh"` prim, identity if none.

    USD stores it row-vector style (p' = p M). Leake Street rotates its Y-up mesh
    into the Z-up stage this way; Linden authors nothing, so its mesh and splat
    both stay Y-up (handled per scene in scenes.py).
    """
    with zipfile.ZipFile(usdz) as z:
        text = z.read(_member(z, "gauss.usda")).decode()
    block = text[text.index('over "mesh"'):]
    m = re.search(r"matrix4d xformOp:transform = \(([^\n]*)\)\s*\n", block)
    if not m:
        return np.eye(4)
    nums = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?(?:e-?\d+)?", m.group(1))]
    return np.array(nums, dtype=np.float64).reshape(4, 4)


def load_mesh(usdz: Path, workdir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Collision mesh as (vertices (V,3) in the Z-up stage frame, triangles (F,3))."""
    from pxr import Usd, UsdGeom

    workdir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(usdz) as z:
        z.extract(_member(z, "mesh.usd"), workdir)
    stage = Usd.Stage.Open(str(workdir / "mesh.usd"))
    to_stage = _mesh_transform(usdz)

    verts, tris = [], []
    offset = 0
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        pts = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
        idx = np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
        local = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()), dtype=np.float64)
        m = local @ to_stage                                  # row vectors: p' = p M
        pts_h = np.c_[pts, np.ones(len(pts))] @ m
        # fan-triangulate polygons
        starts = np.r_[0, np.cumsum(counts)[:-1]]
        for k in np.unique(counts):
            sel = starts[counts == k]
            for j in range(1, k - 1):
                tris.append(np.stack([idx[sel], idx[sel + j], idx[sel + j + 1]], 1) + offset)
        verts.append(pts_h[:, :3])
        offset += len(pts)
    return np.concatenate(verts).astype(np.float32), np.concatenate(tris).astype(np.int32)


def unpack(usdz: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    g = load_gaussians(usdz)
    np.savez(out / "splats.npz", **g)
    v, f = load_mesh(usdz, out / "raw")
    np.savez(out / "mesh.npz", vertices=v, faces=f)
    print(f"{usdz.name}: {len(g['means']):,} gaussians, mesh {len(v):,} verts / {len(f):,} tris")
    lo, hi = v.min(0), v.max(0)
    print(f"  mesh bounds  {np.round(lo, 2)} .. {np.round(hi, 2)}")
    mlo, mhi = np.percentile(g["means"].astype(np.float32), [1, 99], axis=0)
    print(f"  splat 1-99%  {np.round(mlo, 2)} .. {np.round(mhi, 2)}")


if __name__ == "__main__":
    import sys
    unpack(Path(sys.argv[1]), Path(sys.argv[2]))
