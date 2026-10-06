"""Render a decoded NuRec splat on the Mac GPU with metal-gauss.

Camera convention everywhere in this project: OpenCV (x right, y down, z forward),
`cam_to_world` 4x4. metal-gauss wants world->camera, so we invert here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from metal_gauss import render as mg_render

SH_C0 = 0.28209479177387814


class SplatScene:
    def __init__(self, npz: Path, device: str = "mps", sh_degree: int = 3,
                 max_dist: float | None = None):
        d = np.load(npz)
        means = d["means"].astype(np.float32)
        keep = np.ones(len(means), bool)
        if max_dist is not None:
            keep &= np.linalg.norm(means, axis=1) < max_dist
        t = lambda a: torch.from_numpy(np.ascontiguousarray(a[keep].astype(np.float32))).to(device)
        self.means = t(means)
        q = d["quats"].astype(np.float32)
        q /= np.linalg.norm(q, axis=1, keepdims=True).clip(1e-12)
        self.quats = t(q)
        self.scales = t(np.exp(d["log_scales"].astype(np.float32)))
        self.opacities = t(1.0 / (1.0 + np.exp(-d["density_logits"].astype(np.float32))))
        nb = (sh_degree + 1) ** 2
        self.sh = t(d["sh"][:, :nb])
        self.sh_degree = sh_degree
        self.device = device

    def __len__(self) -> int:
        return self.means.shape[0]

    @torch.no_grad()
    def render(self, cam_to_world: np.ndarray, K: np.ndarray, W: int, H: int,
               near: float = 0.05, far: float = 200.0, background=(0.0, 0.0, 0.0)):
        """Returns rgb (H,W,3) float32 in [0,1] and alpha (H,W) as numpy."""
        viewmat = torch.from_numpy(np.linalg.inv(cam_to_world).astype(np.float32))
        Kt = torch.from_numpy(K.astype(np.float32))
        rgb, alpha, _ = mg_render(
            self.means, self.quats, self.scales, self.opacities,
            self.sh[:, :1], Kt, viewmat, W, H, sh_degree=self.sh_degree,
            near=near, far=far, background=background,
            sh_rest=self.sh[:, 1:], antialias=True, backend="metal")
        return (rgb.clamp(0, 1).float().cpu().numpy(),
                alpha.reshape(H, W).float().cpu().numpy())

    @torch.no_grad()
    def render_depth(self, cam_to_world: np.ndarray, K: np.ndarray, W: int, H: int,
                     near: float = 0.05, far: float = 200.0, min_alpha: float = 0.5):
        """Alpha-weighted camera-z depth (file units), inf where the splat is mostly empty.

        Each Gaussian's camera-space z, normalised by `far`, is written into its SH DC
        coefficient so the fused Metal path (colour = C0*dc + 0.5, ReLU, no upper clamp)
        rasterises depth. metal-gauss's explicit-colour path does the same job with a
        torch projection and CPU tile binning, ~5x slower.
        """
        w2c = np.linalg.inv(cam_to_world).astype(np.float32)
        viewmat = torch.from_numpy(w2c)
        Rz = torch.from_numpy(w2c[2, :3]).to(self.device)
        z = (self.means @ Rz + float(w2c[2, 3])) / far
        dc = ((z - 0.5) / SH_C0)[:, None, None].expand(-1, 1, 3).contiguous()
        Kt = torch.from_numpy(K.astype(np.float32))
        acc, alpha, _ = mg_render(
            self.means, self.quats, self.scales, self.opacities, dc, Kt, viewmat, W, H,
            sh_degree=0, near=near, far=far, background=(0.0, 0.0, 0.0),
            antialias=True, backend="metal")
        acc = acc[..., 0].float().cpu().numpy() * far
        alpha = alpha.reshape(H, W).float().cpu().numpy()
        depth = np.where(alpha > min_alpha, acc / np.maximum(alpha, 1e-6), np.inf)
        return depth.astype(np.float32), alpha


def intrinsics(W: int, H: int, fovy_deg: float) -> np.ndarray:
    fy = 0.5 * H / np.tan(np.radians(fovy_deg) / 2)
    return np.array([[fy, 0, W / 2], [0, fy, H / 2], [0, 0, 1]], dtype=np.float64)


def look_at(eye, target, up=(0, 0, 1)) -> np.ndarray:
    """OpenCV cam_to_world looking from eye to target, z-up world."""
    eye, target, up = map(lambda a: np.asarray(a, float), (eye, target, up))
    z = target - eye; z /= np.linalg.norm(z)
    x = np.cross(z, up); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = x, y, z, eye
    return m
