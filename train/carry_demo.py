"""Pick-and-place demo in the Linden warehouse scan, driven by a trained carry policy.

Physics only (no rendering): builds the G1 + scene model, runs a scripted task
(walk -> grasp -> lift -> carry -> place -> release) and saves the model as .mjb plus
a per-frame trajectory, which scripts/render_carry.py composites into the splat.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))
sys.path.insert(0, str(ROOT))
import carry_env  # noqa: E402

BOX_HALF = np.array([0.14, 0.16, 0.18])          # 28 x 32 x 36 cm carton: grasp height a G1 reaches
CARDBOARD = [0.62, 0.47, 0.30, 1.0]
HEAD_CAM = dict(pos=[0.08, 0.0, 0.42], pitch_deg=25.0)   # front of head_link, torso frame
WRIST_KP = 20.0                                   # demo arm controller; force limits unchanged
# Flat pads on the inner face of each rubber hand (the hand mesh's inner face sits at
# |y| = 0.045 in the wrist frame). Playground's hand collider is a small capsule: two
# capsule contacts make a hinge and the carton swings on it (24 deg pitch, 28 deg roll).
PAD_HALF = [0.06, 0.005, 0.045]
PAD_POS = {"left": [0.10, -0.040, 0.01], "right": [0.10, 0.040, 0.01]}
PAD_INSET = 0.045                                 # palm site -> pad face, metres
FLOOR_GROUP, OBSTACLE_GROUP, HIDDEN_GROUP = 2, 3, 5
_CARRY_LEFT = np.array([-0.6, 0.1, 0.0, 0.9, 0.0, 0.0, 0.0])
CARRY_SEED = np.r_[_CARRY_LEFT, _CARRY_LEFT * carry_env.MIRROR]


@dataclass
class Surface:
    centre: np.ndarray      # world xy of the slab
    top: float              # height of its top face (m)
    half: np.ndarray        # half extents along (u, v): u points from the slab to the aisle
    yaw: float = 0.0        # direction of u

    @property
    def u(self) -> np.ndarray:
        return np.array([np.cos(self.yaw), np.sin(self.yaw)])

    def edge_point(self) -> np.ndarray:
        """Middle of the face that looks onto the aisle."""
        return self.centre + self.u * self.half[0]


def _cam_quat(pitch_deg: float) -> np.ndarray:
    p = np.radians(pitch_deg)
    fwd = np.array([np.cos(p), 0, -np.sin(p)])
    up = np.array([np.sin(p), 0, np.cos(p)])
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.c_[np.cross(up, -fwd), up, -fwd].flatten())
    return q


def build(env, surfaces: list[Surface], carton_pos, carton_mass: float, pin: bool = False,
          carton_yaw: float = 0.0):
    spec = carry_env.make_spec(env._xml_path, env._model_assets)
    for k in list(spec.keys):          # keyframes would no longer match nq (carton freejoint)
        spec.delete(k)
    wb = spec.worldbody
    for i, s in enumerate(surfaces):
        wb.add_geom(name=f"surface{i}", type=mujoco.mjtGeom.mjGEOM_BOX,
                    pos=[*s.centre, s.top / 2], size=[*s.half, s.top / 2], rgba=[0.3, 0.6, 0.9, 1],
                    quat=[np.cos(s.yaw / 2), 0, 0, np.sin(s.yaw / 2)], contype=0, conaffinity=0)
    carton = wb.add_body(name="carton", pos=list(carton_pos),
                         quat=[np.cos(carton_yaw / 2), 0, 0, np.sin(carton_yaw / 2)])
    carton.add_freejoint()
    carton.add_geom(name="carton", type=mujoco.mjtGeom.mjGEOM_BOX, size=list(BOX_HALF),
                    mass=carton_mass, rgba=CARDBOARD, contype=0, conaffinity=0)
    for side in ("left", "right"):
        spec.body(f"{side}_wrist_yaw_link").add_geom(
            name=f"{side}_pad", type=mujoco.mjtGeom.mjGEOM_BOX, size=PAD_HALF, pos=PAD_POS[side],
            contype=0, conaffinity=0, group=3, density=0)
    pair = dict(condim=4, friction=[1.2, 1.2, 0.02, 0.001, 0.001])
    for other in ["floor", "left_pad", "right_pad"] + [f"surface{i}" for i in range(len(surfaces))]:
        spec.add_pair(geomname1="carton", geomname2=other, **pair)
    spec.body("torso_link").add_camera(name="head", pos=HEAD_CAM["pos"],
                                       quat=_cam_quat(HEAD_CAM["pitch_deg"]), fovy=70)
    wb.add_body(name="chase_rig", mocap=True).add_camera(name="chase", fovy=60)
    spec.body("pelvis").add_light(name="sun", pos=[-1.0, -0.6, 5.0], dir=[0.2, 0.12, -1],
                                  type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
                                  mode=mujoco.mjtCamLight.mjCAMLIGHT_TRACKCOM, castshadow=True,
                                  diffuse=[0.6, 0.6, 0.6], ambient=[0.1, 0.1, 0.1])
    spec.visual.quality.shadowsize = 4096
    spec.visual.global_.offwidth, spec.visual.global_.offheight = 1920, 1080
    spec.visual.headlight.ambient = [0.35, 0.35, 0.35]
    spec.visual.headlight.diffuse = [0.45, 0.45, 0.45]
    spec.visual.headlight.specular = [0.1, 0.1, 0.1]
    # Grasp assist: welds from each palm to the carton, inactive until the pads touch it.
    # A pure friction grasp with Playground's hands (capsules, then flat pads) slipped or
    # hinged in testing; the welds keep the load path real (carton -> palms -> arms -> robot)
    # and replace only the friction.
    for side in ("left", "right"):
        spec.add_equality(type=mujoco.mjtEq.mjEQ_WELD, name=f"grip_{side}", objtype=mujoco.mjtObj.mjOBJ_BODY,
                          name1=f"{side}_wrist_yaw_link", name2="carton", active=False,
                          solref=[0.01, 1.0])
    if pin:
        spec.add_equality(type=mujoco.mjtEq.mjEQ_WELD, name1="pelvis", objtype=mujoco.mjtObj.mjOBJ_BODY)
    m = spec.compile()
    m.opt.timestep = 0.002
    # the hidden training payload is not part of the demo: the carton is the load
    m.body_mass[[m.body("left_payload").id, m.body("right_payload").id]] = 1e-4
    for side in ("left", "right"):
        for j in ("wrist_roll", "wrist_pitch", "wrist_yaw"):
            a = m.actuator(f"{side}_{j}_joint").id
            m.actuator_gainprm[a, 0] = WRIST_KP
            m.actuator_biasprm[a, 1] = -WRIST_KP
    # rendering groups: robot visuals -> 1, robot collision -> hidden, carton -> 0 (drawn
    # and composited with the robot), floor -> 2 (shadow catcher), surfaces -> 3 (hidden)
    grp = m.geom_group.copy()
    grp[m.geom_group == 2] = 1
    grp[m.geom_group == 3] = HIDDEN_GROUP
    grp[m.geom("floor").id] = FLOOR_GROUP
    grp[m.geom("carton").id] = 0
    for i in range(len(surfaces)):
        grp[m.geom(f"surface{i}").id] = OBSTACLE_GROUP
    m.geom_group[:] = grp
    return m


class ArmIK:
    """Damped least squares for both palm sites over shoulder pitch/roll/yaw + elbow.

    Joint bounds are the policy's training range for arm targets (carry_env.CARRY_LO/HI,
    mirrored for the right arm), so the walking policy never sees an arm pose it was not
    trained with. Inside that range the palms face each other, which the pads need."""

    def __init__(self, m: mujoco.MjModel):
        self.m = m
        self.d = mujoco.MjData(m)
        self.sites = [m.site("left_palm").id, m.site("right_palm").id]
        self.qadr, self.dadr = [], []
        for side in ("left", "right"):
            js = [m.joint(f"{side}_{j}_joint") for j in carry_env.ARM_JOINTS]
            self.qadr.append(np.array([int(j.qposadr[0]) for j in js]))
            self.dadr.append(np.array([int(j.dofadr[0]) for j in js]))
        lo, hi = carry_env.CARRY_LO, carry_env.CARRY_HI
        rlo, rhi = lo * carry_env.MIRROR, hi * carry_env.MIRROR
        self.lo = [lo, np.minimum(rlo, rhi)]
        self.hi = [hi, np.maximum(rlo, rhi)]

    def solve(self, qpos: np.ndarray, targets: np.ndarray, q_arm: np.ndarray, iters: int = 60):
        """targets (2,3) world palm positions; q_arm (14,) start; returns (14,) and residual (m)."""
        m, d = self.m, self.d
        d.qpos[:] = qpos
        q = q_arm.copy()
        jac = np.zeros((3, m.nv))
        for _ in range(iters):
            for k in range(2):
                d.qpos[self.qadr[k]] = q[7 * k:7 * k + 7]
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            for k in range(2):
                err = targets[k] - d.site_xpos[self.sites[k]]
                mujoco.mj_jacSite(m, d, jac, None, self.sites[k])
                J = jac[:, self.dadr[k][:4]]
                dq = J.T @ np.linalg.solve(J @ J.T + 1e-3 * np.eye(3), err)
                sl = slice(7 * k, 7 * k + 4)
                q[sl] = np.clip(q[sl] + 0.8 * dq, self.lo[k][:4], self.hi[k][:4])
        for k in range(2):
            d.qpos[self.qadr[k]] = q[7 * k:7 * k + 7]
        mujoco.mj_kinematics(m, d)
        res = max(np.linalg.norm(targets[k] - d.site_xpos[self.sites[k]]) for k in range(2))
        return q, res


def arm_ctrl(m: mujoco.MjModel, d: mujoco.MjData, target: np.ndarray, arm_act=slice(15, 29),
             arm_dof=slice(21, 35), load_kg: float = 0.0) -> np.ndarray:
    """Position targets for the arm actuators plus gravity feed-forward.

    The actuators are PD position servos (kp 75, wrists 20 here) with no gravity term, so a
    held pose sags ~0.16 rad at the shoulder. Adding qfrc_bias/kp to the target makes the
    servo produce the gravity torque at zero error, the usual fix on a position-controlled arm."""
    kp = m.actuator_gainprm[arm_act, 0]
    tau = d.qfrc_bias[arm_dof].copy()
    if load_kg > 0:
        # the held carton's weight, half at each palm, as joint torques: J^T (0, 0, m g / 2)
        jac = np.zeros((3, m.nv))
        f = np.array([0.0, 0.0, load_kg * 9.81 / 2])
        for site in ("left_palm", "right_palm"):
            mujoco.mj_jacSite(m, d, jac, None, m.site(site).id)
            tau += jac[:, arm_dof].T @ f
    return target + tau / kp


class PalmController:
    """Closed-loop palm placement: each control step, IK from the robot's actual base and
    waist (seeded with the last commanded arm angles, not the contact-blocked measured
    ones) toward target + integral correction. The integral acts on x and z only: y is the
    squeeze direction, where the target is deliberately inside the carton."""

    def __init__(self, m: mujoco.MjModel, ki: float = 0.08, limit: float = 0.04, window: float = 0.05):
        self.m, self.ik, self.ki, self.limit, self.window = m, ArmIK(m), ki, limit, window
        self.sites = [m.site("left_palm").id, m.site("right_palm").id]
        self.bias = np.zeros((2, 3))
        self.cmd = None
        self.load_kg = 0.0          # set while gripping: weight feed-forward

    def palms(self, d) -> np.ndarray:
        return np.array([d.site_xpos[s] for s in self.sites])

    def step(self, d: mujoco.MjData, targets: np.ndarray, integrate: bool = True) -> np.ndarray:
        if self.cmd is None:
            # seed from the carry pose: the standing pose sits on a bound of the IK box
            # (shoulder pitch 0.2) and the right arm got stuck there with warm starts
            self.cmd = CARRY_SEED.copy()
        if integrate:
            # only trim small residual sag: integrating while the target is out of reach
            # (robot still stepping in) wound the bias up and drove the IK into a
            # hanging-arm solution it never left
            err = targets - self.palms(d)
            near = (np.linalg.norm(err, axis=1) < self.window)[:, None]
            self.bias[:, [0, 2]] = np.clip(self.bias[:, [0, 2]] + self.ki * err[:, [0, 2]] * near,
                                           -self.limit, self.limit)
        self.cmd, res = self.ik.solve(d.qpos.copy(), targets + self.bias, self.cmd, iters=10)
        if res > 0.03:
            self.cmd, res = self.ik.solve(d.qpos.copy(), targets + self.bias, CARRY_SEED.copy(), iters=150)
        return arm_ctrl(self.m, d, self.cmd, load_kg=self.load_kg)

    def reset_bias(self):
        self.bias[:] = 0


def grip(m: mujoco.MjModel, d: mujoco.MjData, on: bool) -> None:
    """Activate/deactivate the palm-carton welds at the current relative pose (no jump).

    MuJoCo weld data: [anchor(3), relpos(3), relquat(4), torquescale] with relpose = pose of
    body2 (carton) in the frame of body1 (wrist)."""
    for side in ("left", "right"):
        e = m.equality(f"grip_{side}").id
        if on:
            b1, b2 = m.eq_obj1id[e], m.eq_obj2id[e]
            R1 = d.xmat[b1].reshape(3, 3)
            q1inv = np.zeros(4); mujoco.mju_negQuat(q1inv, d.xquat[b1])
            rq = np.zeros(4); mujoco.mju_mulQuat(rq, q1inv, d.xquat[b2])
            m.eq_data[e, 0:3] = 0
            m.eq_data[e, 3:6] = R1.T @ (d.xpos[b2] - d.xpos[b1])
            m.eq_data[e, 6:10] = rq
            m.eq_data[e, 10] = 1.0
        d.eq_active[e] = int(on)


def pads_touching(m: mujoco.MjModel, d: mujoco.MjData) -> bool:
    pads = {m.geom("left_pad").id, m.geom("right_pad").id}
    carton = m.geom("carton").id
    hit = set()
    for c in d.contact[: d.ncon]:
        g = {c.geom1, c.geom2}
        if carton in g:
            hit |= g & pads
    return hit == pads


def surface_from_mesh(v_world: np.ndarray, f: np.ndarray, centre_guess, aisle_xy: np.ndarray,
                      z_band=(0.45, 1.3), radius: float = 1.0) -> Surface:
    """Fit an oriented slab to the upward-facing triangles of one pallet-stack top.

    u = direction from the stack toward the nearest aisle point; extents from the
    triangle centroids projected on u and on its perpendicular."""
    c = v_world[f].mean(1)
    near = (np.linalg.norm(c[:, :2] - centre_guess, axis=1) < radius) & (c[:, 2] > z_band[0]) & (c[:, 2] < z_band[1])
    tri = f[near]
    n = np.cross(v_world[tri[:, 1]] - v_world[tri[:, 0]], v_world[tri[:, 2]] - v_world[tri[:, 0]])
    area = 0.5 * np.linalg.norm(n, axis=1)
    up = np.abs(n[:, 2]) / np.maximum(2 * area, 1e-12) > 0.95
    pts = c[near][up]
    top = float(np.median(pts[:, 2]))
    pts = pts[np.abs(pts[:, 2] - top) < 0.04]
    ctr = pts[:, :2].mean(0)
    to_aisle = aisle_xy[np.argmin(np.linalg.norm(aisle_xy - ctr, axis=1))] - ctr
    u = to_aisle / np.linalg.norm(to_aisle)
    vperp = np.array([-u[1], u[0]])
    pu, pv = (pts[:, :2] - ctr) @ u, (pts[:, :2] - ctr) @ vperp
    lo_u, hi_u = np.percentile(pu, [2, 98]); lo_v, hi_v = np.percentile(pv, [2, 98])
    centre = ctr + u * (lo_u + hi_u) / 2 + vperp * (lo_v + hi_v) / 2
    return Surface(centre=centre, top=top, half=np.array([(hi_u - lo_u) / 2, (hi_v - lo_v) / 2]),
                   yaw=float(np.arctan2(u[1], u[0])))
