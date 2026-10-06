"""Run a trained carry policy on a plain-MuJoCo G1 (benchmark and demo share this).

Observation and update order match the Playground env exactly (checked to 3e-8): the
observation after a step is built before the gait phase advances and before last_act is
replaced. The robot must be the first 36 qpos / 35 qvel entries of the model (extra
bodies such as a carton come after).
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import mujoco
import numpy as np

QS, VS = slice(7, 36), slice(6, 35)          # robot joints in qpos / qvel


class PolicyRunner:
    def __init__(self, run_dir: Path, env):
        import jax
        from brax.training.acme import running_statistics
        from brax.training.agents.ppo import networks as ppo_networks

        meta = json.loads((Path(run_dir) / "meta.json").read_text())
        params = pickle.load(open(Path(run_dir) / "params.pkl", "rb"))
        net = {k: tuple(v) if isinstance(v, list) else v for k, v in meta["network"].items()}
        npriv = params[0].mean["privileged_state"].shape[0]
        nets = ppo_networks.make_ppo_networks(
            {"state": (103,), "privileged_state": (npriv,)}, env.mj_model.nu,
            preprocess_observations_fn=running_statistics.normalize, **net)
        self._infer = jax.jit(ppo_networks.make_inference_fn(nets)((params[0], params[1]), deterministic=True))
        self._key = jax.random.PRNGKey(0)
        self._priv = np.zeros(npriv, np.float32)
        self.env = env
        self.default = np.array(env.mj_model.keyframe("knees_bent").qpos[7:])
        self.scale = env._config.action_scale
        self.imu = env.mj_model.site("imu_in_pelvis").id
        self.n_sub = int(round(env._config.ctrl_dt / env._config.sim_dt))
        self.name = Path(run_dir).name

    def reset(self, gait_freq: float = 1.375) -> None:
        self.last = np.zeros(29, np.float32)
        self.phase = np.array([0.0, np.pi])
        self.phase_dt = 2 * np.pi * self.env._config.ctrl_dt * gait_freq
        self.obs = None

    def _obs(self, d, cmd) -> np.ndarray:
        gyro = self.env.get_gyro(d, "pelvis")
        grav = d.site_xmat[self.imu].reshape(3, 3).T @ np.array([0, 0, -1.0])
        lin = self.env.get_local_linvel(d, "pelvis")
        return np.hstack([lin, gyro, grav, cmd, d.qpos[QS] - self.default, d.qvel[VS], self.last,
                          np.cos(self.phase), np.sin(self.phase)]).astype(np.float32)

    def step(self, m, d, cmd, arm_ctrl: np.ndarray, arm_action: np.ndarray) -> None:
        """One 50 Hz control step. arm_ctrl: what the 14 arm actuators get (may include
        feed-forward); arm_action: the arm block the policy is told it sent (its training
        convention: (arm_target - default) / scale)."""
        cmd = np.asarray(cmd, np.float32)
        if self.obs is None:
            self.obs = self._obs(d, cmd)
        act, _ = self._infer({"state": self.obs, "privileged_state": self._priv}, self._key)
        act = np.array(act)
        act[15:] = arm_action
        d.ctrl[:15] = self.default[:15] + act[:15] * self.scale
        d.ctrl[15:] = arm_ctrl
        for _ in range(self.n_sub):
            mujoco.mj_step(m, d)
        self.obs = self._obs(d, cmd)
        self.phase = np.fmod(self.phase + self.phase_dt + np.pi, 2 * np.pi) - np.pi
        self.last = act.astype(np.float32)
