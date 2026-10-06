"""G1 carry-locomotion environment on top of MuJoCo Playground's G1 joystick task.

What changes versus G1JoystickFlatTerrain:
  * the arms are not the policy's: every episode samples a target arm pose
    (25% arms down, 75% a mirrored "carry" pose), and the 14 arm actuators track
    it regardless of what the policy outputs. Legs and waist stay the policy's.
  * a hidden payload: two point-ish masses, one per forearm at the palm, each
    M/2 with M ~ U(0, max_payload) kg per environment (domain randomisation).
    The load goes through elbows and shoulders, as a box squeezed between the
    palms would. The actor never observes M; the critic does (privileged).
  * the pose penalty no longer counts the arms (their pose is not the policy's).

`max_payload=0` gives the ablation: same arms, same everything, no payload.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from etils import epath
from mujoco import mjx
from mujoco_playground._src import mjx_env
from mujoco_playground._src.locomotion.g1 import joystick
from mujoco_playground._src.locomotion.g1 import randomize as g1_randomize

ARM_JOINTS = ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
              "wrist_roll", "wrist_pitch", "wrist_yaw"]
# per-joint sign for the right arm when mirroring the left arm's pose
MIRROR = np.array([1, -1, -1, 1, -1, 1, -1], np.float32)
# carry-pose sampling box for the LEFT arm (radians); see check_carry_pose()
CARRY_LO = np.array([-1.1, -0.15, -0.3, 0.0, -0.3, -0.3, -0.3], np.float32)
CARRY_HI = np.array([0.2, 0.35, 0.3, 1.6, 0.3, 0.3, 0.3], np.float32)
P_ARMS_DOWN = 0.25
# palm site position expressed in the elbow link frame, wrists neutral
PALM_IN_ELBOW = (0.27, 0.0, -0.01)


def default_config(max_payload: float = 10.0, p_stand: float = 0.1, stand_fix: bool = False):
    """p_stand: probability a sampled command is all-zero (Playground's own value is 0.1).
    stand_fix: compute the stand-still penalty over the policy's 15 joints only (see below)."""
    cfg = joystick.default_config()
    cfg.max_payload = max_payload
    cfg.p_stand = p_stand
    cfg.stand_fix = stand_fix
    return cfg


def make_spec(xml_path, assets) -> mujoco.MjSpec:
    """Playground's G1 scene plus the two hidden payload bodies (one per forearm, at the palm)."""
    xmls = {k: v for k, v in assets.items() if k.endswith(".xml")}
    spec = mujoco.MjSpec.from_string(epath.Path(xml_path).read_text(), include=xmls, assets=assets)
    for side in ("left", "right"):
        b = spec.body(f"{side}_elbow_link").add_body(name=f"{side}_payload", pos=list(PALM_IN_ELBOW))
        b.explicitinertial = True
        b.mass = 0.5
        b.inertia = [1e-3, 1e-3, 1e-3]
        b.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.04], contype=0,
                   conaffinity=0, group=3, density=0, rgba=[0.8, 0.5, 0.2, 1])
    return spec


class CarryJoystick(joystick.Joystick):

    def __init__(self, *args, **kwargs):
        # the registry normally fetches the Menagerie G1 meshes; we bypass the registry
        mjx_env.ensure_menagerie_exists()
        super().__init__(*args, **kwargs)

    def _post_init(self) -> None:
        # rebuild the model with the two payload bodies before the parent caches ids
        m = make_spec(self._xml_path, self._model_assets).compile()
        m.opt.timestep = self.sim_dt
        m.vis.global_.offwidth, m.vis.global_.offheight = 3840, 2160
        self._mj_model = m
        self._mjx_model = mjx.put_model(m, impl=self._config.impl)
        if self._config.impl == "warp":
            # mujoco_warp prints a line per env per step when the 5-iteration linesearch
            # (Playground's G1 setting) runs out: 4.5 GB of log in 8 minutes. Silence it;
            # the solver settings themselves are unchanged.
            opt = self._mjx_model.opt
            self._mjx_model = self._mjx_model.replace(
                opt=opt.replace(_impl=opt._impl.replace(warn_overflow=0)))
        super()._post_init()

        assert m.body("torso_link").id == g1_randomize.TORSO_BODY_ID, "payload bodies shifted body ids"
        self._payload_ids = jp.array([m.body("left_payload").id, m.body("right_payload").id])
        arm_qidx = [int(m.joint(f"{s}_{j}_joint").qposadr[0]) - 7
                    for s in ("left", "right") for j in ARM_JOINTS]
        # step() overwrites action[15:]; that is only right if actuators follow this order
        assert arm_qidx == list(range(15, 29)), arm_qidx
        assert [m.actuator_trnid[a, 0] - 1 for a in range(15, 29)] == arm_qidx
        self._weights = self._weights.at[15:].set(0.0)      # arms out of the pose penalty

    # -- arm targets -------------------------------------------------------------

    def _sample_arm_target(self, rng: jax.Array) -> jax.Array:
        k1, k2, k3 = jax.random.split(rng, 3)
        left = jax.random.uniform(k1, (7,), minval=CARRY_LO, maxval=CARRY_HI)
        right = left * MIRROR + jax.random.uniform(k2, (7,), minval=-0.05, maxval=0.05)
        carry = jp.concatenate([left, right])
        down = self._default_pose[15:]
        return jp.where(jax.random.uniform(k3) < P_ARMS_DOWN, down, carry)

    def reset(self, rng: jax.Array) -> mjx_env.State:
        rng, arm_rng = jax.random.split(rng)
        state = super().reset(rng)
        state.info["arm_target"] = self._sample_arm_target(arm_rng)
        state.obs["privileged_state"] = jp.concatenate(
            [state.obs["privileged_state"], self._payload_obs()])
        return state

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        # the arm block of the action is overwritten with the episode's arm target:
        # motor_targets = default + action * scale, so solve for that action
        arm_action = (state.info["arm_target"] - self._default_pose[15:]) / self._config.action_scale
        action = action.at[15:].set(arm_action)
        state = super().step(state, action)
        # resample arm targets when the parent resamples the command (step counter wraps)
        rng, arm_rng = jax.random.split(state.info["rng"])
        state.info["rng"] = rng
        state.info["arm_target"] = jp.where(state.info["step"] == 0,
                                            self._sample_arm_target(arm_rng),
                                            state.info["arm_target"])
        state.obs["privileged_state"] = jp.concatenate(
            [state.obs["privileged_state"], self._payload_obs()])
        return state

    def sample_command(self, rng: jax.Array) -> jax.Array:
        """Playground's sampler with a configurable chance of an all-zero (stand) command."""
        r1, r2, r3, r4 = jax.random.split(rng, 4)
        c = self._config
        cmd = jp.hstack([jax.random.uniform(r1, minval=c.lin_vel_x[0], maxval=c.lin_vel_x[1]),
                         jax.random.uniform(r2, minval=c.lin_vel_y[0], maxval=c.lin_vel_y[1]),
                         jax.random.uniform(r3, minval=c.ang_vel_yaw[0], maxval=c.ang_vel_yaw[1])])
        return jp.where(jax.random.bernoulli(r4, p=c.p_stand), jp.zeros(3), cmd)

    def _cost_stand_still(self, commands: jax.Array, qpos: jax.Array) -> jax.Array:
        """Playground sums |q - default| over all 29 joints when the command is zero. Here the
        arms are not the policy's, so that penalty is ~2.4 rad of arm deviation it can never
        remove: ~-24 per 500-step stand segment against -2 for terminating (rewards are
        scaled by dt). The first-round policies fell when told to stand, 0/20. With
        stand_fix the penalty only counts legs and waist."""
        if not self._config.stand_fix:
            return super()._cost_stand_still(commands, qpos)
        cost = jp.sum(jp.abs(qpos[:15] - self._default_pose[:15]))
        return cost * (jp.linalg.norm(commands) < 0.01)

    def _payload_obs(self) -> jax.Array:
        return jp.atleast_1d(self.mjx_model.body_mass[self._payload_ids].sum())


def domain_randomize(model: mjx.Model, rng: jax.Array, payload_ids, max_payload: float):
    """Playground's G1 randomisation, then the payload mass on top."""
    n = rng.shape[0]
    rng, prng = jax.random.split(rng[0])
    model, in_axes = g1_randomize.domain_randomize(model, jax.random.split(rng, n))
    total = jax.random.uniform(prng, (n,), minval=0.0, maxval=max_payload)
    half = jp.maximum(total / 2, 1e-3)
    body_mass = model.body_mass.at[:, payload_ids[0]].set(half).at[:, payload_ids[1]].set(half)
    model = model.tree_replace({"body_mass": body_mass})
    return model, in_axes


def check_carry_pose(env: CarryJoystick, n: int = 2000, seed: int = 0) -> dict[str, Any]:
    """Where do the palms end up for sampled carry poses (torso frame)? Sanity check."""
    m = env.mj_model
    d = mujoco.MjData(m)
    rng = np.random.default_rng(seed)
    torso = m.body("torso_link").id
    palms = [m.site("left_palm").id, m.site("right_palm").id]
    out = []
    for _ in range(n):
        left = rng.uniform(CARRY_LO, CARRY_HI)
        d.qpos[:] = m.keyframe("knees_bent").qpos
        d.qpos[7 + 15:7 + 22] = left
        d.qpos[7 + 22:7 + 29] = left * MIRROR
        mujoco.mj_kinematics(m, d)
        R, p = d.xmat[torso].reshape(3, 3), d.xpos[torso]
        out.append([(d.site_xpos[s] - p) @ R for s in palms])
    out = np.array(out)
    return {"palm_x": np.percentile(out[:, :, 0], [5, 50, 95]).round(2),
            "palm_z": np.percentile(out[:, :, 2], [5, 50, 95]).round(2),
            "gap": np.percentile(out[:, 0, 1] - out[:, 1, 1], [5, 50, 95]).round(2)}
