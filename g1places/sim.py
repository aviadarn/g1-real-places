"""Unitree G1 in MuJoCo, walking a planned path with Unitree's pretrained policy.

The policy and its observation layout are Unitree's own (unitree_rl_gym,
deploy/deploy_mujoco): 12 leg joints, 47-dim observation, 50 Hz control on a
500 Hz simulation, PD torques. We only add a path-following command on top:
forward speed plus a yaw rate that steers toward a lookahead point.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
import torch

from g1places.scenes import ROOT

UNITREE = ROOT / "third_party" / "unitree_rl_gym"
G1_XML = UNITREE / "resources/robots/g1_description/g1_12dof.xml"
POLICY = UNITREE / "deploy/pre_train/g1/motion.pt"

# deploy/deploy_mujoco/configs/g1.yaml
KPS = np.array([100, 100, 100, 150, 40, 40, 100, 100, 100, 150, 40, 40], np.float32)
KDS = np.array([2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 2], np.float32)
DEFAULT = np.array([-0.1, 0, 0, 0.3, -0.2, 0, -0.1, 0, 0, 0.3, -0.2, 0], np.float32)
SIM_DT, DECIMATION = 0.002, 10
ANG_VEL_SCALE, DOF_VEL_SCALE, ACTION_SCALE = 0.25, 0.05, 0.25
CMD_SCALE = np.array([2.0, 2.0, 0.25], np.float32)
GAIT_PERIOD = 0.8

# head camera: front of the head mesh (head_link spans x -0.07..0.07, z 0.33..0.53
# in the pelvis frame), where the G1's depth camera sits, pitched down a little
HEAD_CAM_POS = np.array([0.08, 0.0, 0.45])
HEAD_CAM_PITCH_DEG = 25.0
FLOOR_GROUP, OBSTACLE_GROUP = 2, 3


def gravity_orientation(q: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = q
    return np.array([2 * (-qz * qx + qw * qy), -2 * (qz * qy + qw * qx),
                     1 - 2 * (qw * qw + qz * qz)])


def yaw_of(q: np.ndarray) -> float:
    qw, qx, qy, qz = q
    return float(np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz)))


def build_model(boxes, fovy: float = 70.0) -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(str(G1_XML))
    spec.option.timestep = SIM_DT
    spec.visual.quality.shadowsize = 4096
    spec.visual.headlight.ambient = [0.35, 0.35, 0.35]
    spec.visual.headlight.diffuse = [0.45, 0.45, 0.45]
    spec.visual.headlight.specular = [0.1, 0.1, 0.1]
    spec.visual.global_.offwidth = 1920
    spec.visual.global_.offheight = 1080
    wb = spec.worldbody
    wb.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05],
                rgba=[0.5, 0.5, 0.5, 1], group=FLOOR_GROUP)
    for k, (c, h) in enumerate(boxes):
        wb.add_geom(name=f"obst{k}", type=mujoco.mjtGeom.mjGEOM_BOX, pos=c, size=h,
                    rgba=[1, 0.3, 0.3, 1], group=OBSTACLE_GROUP)
    pelvis = spec.body("pelvis")
    # The shadow map of a directional light only covers a few metres around the
    # light, so the light rides along with the robot (world-fixed direction).
    pelvis.add_light(name="sun", pos=[-1.0, -0.6, 5.0], dir=[0.2, 0.12, -1],
                     type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
                     mode=mujoco.mjtCamLight.mjCAMLIGHT_TRACKCOM, castshadow=True,
                     diffuse=[0.6, 0.6, 0.6], ambient=[0.1, 0.1, 0.1])
    # MuJoCo cameras look down their -z with +y up; the robot faces +x.
    pitch = np.radians(HEAD_CAM_PITCH_DEG)
    fwd = np.array([np.cos(pitch), 0, -np.sin(pitch)])
    up = np.array([np.sin(pitch), 0, np.cos(pitch)])
    xmat = np.c_[np.cross(up, -fwd), up, -fwd]        # columns: cam x, cam y, cam z
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, xmat.flatten())
    pelvis.add_camera(name="head", pos=HEAD_CAM_POS, quat=quat, fovy=fovy)
    chase = wb.add_body(name="chase_rig", mocap=True)
    chase.add_camera(name="chase", fovy=fovy)
    return spec.compile()


@dataclass
class Walker:
    model: mujoco.MjModel
    path: np.ndarray            # (P,2) world xy, evenly spaced
    speed: float = 0.6
    lookahead: float = 1.2

    def __post_init__(self):
        self.data = mujoco.MjData(self.model)
        self.policy = torch.jit.load(str(POLICY))
        self.obs = np.zeros(47, np.float32)
        self.action = np.zeros(12, np.float32)
        self.target = DEFAULT.copy()
        self.counter = 0
        self.s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(self.path, axis=0), axis=1))]
        self.cmd = np.zeros(3, np.float32)
        self.done = False

    def reset(self, s0: float) -> None:
        d = self.data
        p = self.point_at(s0)
        heading = self.heading_at(s0)
        d.qpos[:3] = [p[0], p[1], 0.793]
        d.qpos[3:7] = [np.cos(heading / 2), 0, 0, np.sin(heading / 2)]
        d.qpos[7:] = DEFAULT
        d.qvel[:] = 0
        mujoco.mj_forward(self.model, d)

    def point_at(self, s: float) -> np.ndarray:
        s = np.clip(s, 0, self.s[-1])
        return np.c_[np.interp(s, self.s, self.path[:, 0]), np.interp(s, self.s, self.path[:, 1])][0]

    def heading_at(self, s: float) -> float:
        a, b = self.point_at(s - 0.3), self.point_at(s + 0.3)
        return float(np.arctan2(b[1] - a[1], b[0] - a[0]))

    def progress(self) -> float:
        xy = self.data.qpos[:2]
        return float(self.s[np.argmin(np.linalg.norm(self.path - xy, axis=1))])

    def _steer(self) -> None:
        s = self.progress()
        if s > self.s[-1] - 0.5:
            self.cmd[:] = 0
            self.done = True
            return
        tgt = self.point_at(s + self.lookahead)
        xy = self.data.qpos[:2]
        err = np.arctan2(tgt[1] - xy[1], tgt[0] - xy[0]) - yaw_of(self.data.qpos[3:7])
        err = (err + np.pi) % (2 * np.pi) - np.pi
        self.cmd[0] = self.speed * max(0.25, np.cos(err))
        self.cmd[1] = 0.0
        self.cmd[2] = float(np.clip(2.0 * err, -0.8, 0.8))

    def step_control(self) -> None:
        """Advance one 50 Hz control period (10 physics steps)."""
        m, d = self.model, self.data
        for _ in range(DECIMATION):
            d.ctrl[:] = (self.target - d.qpos[7:]) * KPS - d.qvel[6:] * KDS
            mujoco.mj_step(m, d)
            self.counter += 1
        self._steer()
        phase = (self.counter * SIM_DT) % GAIT_PERIOD / GAIT_PERIOD
        o = self.obs
        o[:3] = d.qvel[3:6] * ANG_VEL_SCALE
        o[3:6] = gravity_orientation(d.qpos[3:7])
        o[6:9] = self.cmd * CMD_SCALE
        o[9:21] = d.qpos[7:] - DEFAULT
        o[21:33] = d.qvel[6:] * DOF_VEL_SCALE
        o[33:45] = self.action
        o[45:47] = [np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase)]
        with torch.no_grad():
            self.action = self.policy(torch.from_numpy(o).unsqueeze(0)).numpy().squeeze()
        self.target = self.action * ACTION_SCALE + DEFAULT

    def fallen(self) -> bool:
        return self.data.qpos[2] < 0.5

    def obstacle_contacts(self) -> int:
        n = 0
        for c in self.data.contact[: self.data.ncon]:
            for g in (c.geom1, c.geom2):
                if self.model.geom_group[g] == OBSTACLE_GROUP:
                    n += 1
        return n
