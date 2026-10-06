"""Per-frame compositing: Gaussian-splat scene + MuJoCo-rendered robot.

For each camera:
  1. splat RGB and splat depth from the scene (metal-gauss, file frame)
  2. robot RGB, depth and mask from MuJoCo with the scene geometry hidden,
     rendered at 2x and box-filtered so the robot's edge is anti-aliased
  3. a shadow factor: MuJoCo renders a white floor plane with and without the
     robot under a shadow-casting light; the ratio darkens the splat floor
  4. robot pixels win where the robot is nearer than the splat surface
"""

from __future__ import annotations

import mujoco
import numpy as np
from scipy.ndimage import gaussian_filter

from g1places.scenes import Scene
from g1places.sim import FLOOR_GROUP, OBSTACLE_GROUP
from g1places.splat import SplatScene, intrinsics

CV_FROM_GL = np.diag([1.0, -1.0, -1.0])


def downsample(a: np.ndarray, f: int) -> np.ndarray:
    h, w = a.shape[0] // f, a.shape[1] // f
    return a[: h * f, : w * f].reshape(h, f, w, f, *a.shape[2:]).mean(axis=(1, 3))


class Compositor:
    def __init__(self, model: mujoco.MjModel, scene: Scene, splats: SplatScene,
                 W: int, H: int, ss: int = 2, robot_gain: float = 1.0,
                 shadow_strength: float = 0.75):
        self.m, self.scene, self.splats = model, scene, splats
        self.W, self.H, self.ss = W, H, ss
        self.r = mujoco.Renderer(model, H * ss, W * ss)
        self.robot_gain, self.shadow_strength = robot_gain, shadow_strength

        def opt(groups):
            o = mujoco.MjvOption()
            o.geomgroup[:] = 0
            for g in groups:
                o.geomgroup[g] = 1
            return o

        self.opt_robot = opt([0, 1])
        self.opt_robot_floor = opt([0, 1, FLOOR_GROUP])
        self.opt_floor = opt([FLOOR_GROUP])
        assert OBSTACLE_GROUP not in (0, 1, FLOOR_GROUP)

    def _mj(self, data, cam, option, mode="rgb"):
        r = self.r
        r.disable_depth_rendering()
        r.disable_segmentation_rendering()
        if mode == "depth":
            r.enable_depth_rendering()
        elif mode == "seg":
            r.enable_segmentation_rendering()
        r.update_scene(data, camera=cam, scene_option=option)
        return r.render()

    def camera_pose_cv(self, data, cam: str) -> np.ndarray:
        cid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_CAMERA, cam)
        c2w = np.eye(4)
        c2w[:3, :3] = data.cam_xmat[cid].reshape(3, 3) @ CV_FROM_GL
        c2w[:3, 3] = data.cam_xpos[cid]
        return c2w

    def render(self, data, cam: str, want_depth: bool = False, depth_scale: int = 1):
        W, H, f = self.W, self.H, self.ss
        cid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_CAMERA, cam)
        K = intrinsics(W, H, float(self.m.cam_fovy[cid]))
        c2w = self.camera_pose_cv(data, cam)
        c2w_file = self.scene.world_to_file_pose(c2w)

        rgb, _ = self.splats.render(c2w_file, K, W, H)
        dW, dH = W // depth_scale, H // depth_scale
        sd, _ = self.splats.render_depth(c2w_file, intrinsics(dW, dH, float(self.m.cam_fovy[cid])), dW, dH)
        sd = sd * self.scene.scale
        if depth_scale > 1:
            sd = np.repeat(np.repeat(sd, depth_scale, 0), depth_scale, 1)[:H, :W]

        robot = self._mj(data, cam, self.opt_robot).astype(np.float32) / 255
        rdepth = self._mj(data, cam, self.opt_robot, "depth")
        seg = self._mj(data, cam, self.opt_robot, "seg")[..., 0]
        mask_hi = seg >= 0
        # robot is drawn where it is in front of the splat surface (with a small slack
        # so feet planted on the floor are not swallowed by floor Gaussians)
        sd_hi = np.repeat(np.repeat(sd, f, 0), f, 1)
        vis_hi = mask_hi & (rdepth < sd_hi + 0.10)
        alpha = downsample(vis_hi.astype(np.float32), f)
        rcol = downsample(robot * vis_hi[..., None], f) / np.maximum(alpha[..., None], 1e-6)

        with_r = self._mj(data, cam, self.opt_robot_floor).astype(np.float32).mean(-1)
        without = self._mj(data, cam, self.opt_floor).astype(np.float32).mean(-1)
        ratio = np.where((without > 8) & ~mask_hi, with_r / np.maximum(without, 1), 1.0)
        # soften: MuJoCo's shadow map edge is hard, a real floor shadow is not
        shade = gaussian_filter(downsample(np.clip(ratio, 0, 1), f), 1.5)
        shade = 1 - self.shadow_strength * (1 - shade)

        out = rgb * shade[..., None]
        out = out * (1 - alpha[..., None]) + np.clip(rcol * self.robot_gain, 0, 1) * alpha[..., None]
        result = {"rgb": np.clip(out, 0, 1)}
        if want_depth:
            rd = downsample(np.where(vis_hi, rdepth, np.inf), f)
            result["depth"] = np.where(alpha > 0.5, np.minimum(rd, sd), sd).astype(np.float32)
        return result
