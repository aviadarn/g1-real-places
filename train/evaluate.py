"""Carry benchmark: one protocol, three policies, plain MuJoCo on the CPU.

Trial: 15 s commanded straight at 0.5 m/s (yaw rate command 0, no heading feedback),
arms in a fixed box-carry pose, a payload
of M kg split between the two palms, three 0.5 m/s velocity kicks in random
directions at random times. A trial fails if the pelvis drops below 0.5 m or the
torso tilts past 60 degrees.

    python train/evaluate.py --policy runs/payload10 --trials 50
    python train/evaluate.py --policy unitree --trials 50
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

MASSES = [0, 2, 4, 6, 8, 10]
SECS, VX, KICK, N_KICKS = 15.0, 0.5, 0.5, 3
# left arm in the fixed carry pose; the right arm mirrors it (carry_env.MIRROR)
CARRY_LEFT = np.array([-0.6, 0.1, 0.0, 0.9, 0.0, 0.0, 0.0])


def yaw_of(q) -> float:
    w, x, y, z = q
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def tilt_deg(xmat: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip(xmat.reshape(3, 3)[2, 2], -1, 1))))


# ----------------------------------------------------------------------------- Playground policies

class PlaygroundPolicy:
    """The trained policy, run on the same G1 model it was trained on (plain MuJoCo)."""

    def __init__(self, run_dir: Path):
        import carry_env
        from policy import PolicyRunner
        cfg = carry_env.default_config(); cfg.impl = "jax"
        self.env = carry_env.CarryJoystick(config=cfg)
        self.m = self.env.mj_model
        self.runner = PolicyRunner(run_dir, self.env)
        self.payload = [self.m.body("left_payload").id, self.m.body("right_payload").id]
        self.torso = self.m.body("torso_link").id
        self.name = Path(run_dir).name

    def setup(self, mass: float, arm_left: np.ndarray):
        import carry_env
        m = self.m
        m.body_mass[self.payload] = max(mass / 2, 1e-3)
        d = mujoco.MjData(m)
        d.qpos[:] = m.keyframe("knees_bent").qpos
        self.arm = np.r_[arm_left, arm_left * carry_env.MIRROR]
        d.qpos[7 + 15:] = self.arm
        d.ctrl[:] = d.qpos[7:]
        mujoco.mj_forward(m, d)
        self.runner.reset()
        self.arm_action = (self.arm - self.runner.default[15:]) / self.runner.scale
        return d

    @property
    def phase_dt(self):
        return self.runner.phase_dt

    @phase_dt.setter
    def phase_dt(self, v):
        self.runner.phase_dt = v

    def control(self, d, cmd, rng) -> None:
        self.runner.step(self.m, d, cmd, self.arm, self.arm_action)

    def fallen(self, d) -> bool:
        return d.qpos[2] < 0.5 or tilt_deg(d.xmat[self.torso]) > 60


# ----------------------------------------------------------------------------- Unitree's stock policy

class UnitreePolicy:
    """unitree_rl_gym motion.pt (12 leg joints) on Unitree's 29-dof G1, arms held by PD.

    The 29-dof file leaves armature/damping/frictionloss at 0; the 12-dof file the
    policy was trained on uses 0.01/0.001/0.1, so those are copied over (without it
    the robot falls in under a second even unloaded)."""

    def __init__(self):
        import torch
        from g1places import sim as S
        self.S, self.torch = S, torch
        self.policy = torch.jit.load(str(S.POLICY))
        spec = mujoco.MjSpec.from_file(str(S.UNITREE / "resources/robots/g1_description/g1_29dof.xml"))
        spec.option.timestep = S.SIM_DT
        spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05])
        for side in ("left", "right"):
            b = spec.body(f"{side}_wrist_yaw_link").add_body(name=f"{side}_payload", pos=[0.08, 0, 0])
            b.explicitinertial = True; b.mass = 0.5; b.inertia = [1e-3] * 3
        m = spec.compile()
        m.dof_armature[6:] = 0.01; m.dof_damping[6:] = 0.001; m.dof_frictionloss[6:] = 0.1
        feet = {m.body("left_ankle_roll_link").id, m.body("right_ankle_roll_link").id}
        for g in range(m.ngeom):                      # arms are separate bodies here: no self-contact
            if m.geom_bodyid[g] not in feet and m.geom_bodyid[g] != 0:
                m.geom_contype[g] = 0; m.geom_conaffinity[g] = 0
        self.m = m
        self.payload = [m.body("left_payload").id, m.body("right_payload").id]
        self.torso = m.body("torso_link").id
        self.kp_up = np.r_[[300] * 3, [80] * 14]; self.kd_up = np.r_[[6] * 3, [2] * 14]
        self.name = "unitree_stock"

    def setup(self, mass, arm_left):
        import carry_env
        S, m = self.S, self.m
        m.body_mass[self.payload] = max(mass / 2, 1e-3)
        d = mujoco.MjData(m)
        d.qpos[2] = 0.793; d.qpos[3] = 1; d.qpos[7:19] = S.DEFAULT
        self.up = np.r_[np.zeros(3), arm_left, arm_left * carry_env.MIRROR]
        d.qpos[19:36] = self.up
        mujoco.mj_forward(m, d)
        self.action = np.zeros(12, np.float32); self.target = S.DEFAULT.copy()
        self.obs = np.zeros(47, np.float32); self.counter = 0
        return d

    def control(self, d, cmd, rng) -> None:
        S, m = self.S, self.m
        for _ in range(S.DECIMATION):
            d.ctrl[:12] = (self.target - d.qpos[7:19]) * S.KPS - d.qvel[6:18] * S.KDS
            d.ctrl[12:] = (self.up - d.qpos[19:36]) * self.kp_up - d.qvel[18:35] * self.kd_up
            mujoco.mj_step(m, d)
            self.counter += 1
        phase = (self.counter * S.SIM_DT) % S.GAIT_PERIOD / S.GAIT_PERIOD
        o = self.obs
        o[:3] = d.qvel[3:6] * S.ANG_VEL_SCALE; o[3:6] = S.gravity_orientation(d.qpos[3:7])
        o[6:9] = np.asarray(cmd) * S.CMD_SCALE; o[9:21] = d.qpos[7:19] - S.DEFAULT
        o[21:33] = d.qvel[6:18] * S.DOF_VEL_SCALE; o[33:45] = self.action
        o[45:47] = [np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase)]
        with self.torch.no_grad():
            self.action = self.policy(self.torch.from_numpy(o)[None]).numpy().squeeze()
        self.target = self.action * S.ACTION_SCALE + S.DEFAULT

    def fallen(self, d) -> bool:
        return d.qpos[2] < 0.5 or tilt_deg(d.xmat[self.torso]) > 60


# ----------------------------------------------------------------------------- protocol

STAND_SECS = 10.0


def stand_trial(pol, mass: float, seed: int, arm_left=CARRY_LEFT) -> dict:
    """10 s with an all-zero command (no shoves): the part of a pick or a place where the
    robot has to stay put. Records how far it wanders."""
    rng = np.random.default_rng(seed)
    d = pol.setup(mass, arm_left)
    pol.phase_dt = 2 * np.pi * 0.02 * rng.uniform(1.25, 1.5)
    yaw = rng.uniform(-np.pi, np.pi)
    d.qpos[3:7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
    mujoco.mj_forward(pol.m, d)
    start = d.qpos[:2].copy()
    for k in range(int(STAND_SECS / 0.02)):
        pol.control(d, np.zeros(3, np.float32), rng)
        if pol.fallen(d):
            return {"ok": False, "t": round((k + 1) * 0.02, 2)}
    return {"ok": True, "t": STAND_SECS, "wander_m": round(float(np.linalg.norm(d.qpos[:2] - start)), 3)}


def trial(pol, mass: float, seed: int, arm_left=CARRY_LEFT) -> dict:
    rng = np.random.default_rng(seed)
    d = pol.setup(mass, arm_left)
    pol.phase_dt = 2 * np.pi * 0.02 * rng.uniform(1.25, 1.5)
    yaw = rng.uniform(-np.pi, np.pi)
    d.qpos[3:7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
    mujoco.mj_forward(pol.m, d)
    n = int(SECS / 0.02)
    kicks = set(rng.choice(np.arange(100, n), N_KICKS, replace=False).tolist())
    cmd = np.array([VX, 0.0, 0.0], np.float32)
    vfwd, yaws = [], []
    for k in range(n):
        if k in kicks:
            th = rng.uniform(0, 2 * np.pi)
            d.qvel[:2] += KICK * np.array([np.cos(th), np.sin(th)])
        pol.control(d, cmd, rng)
        if pol.fallen(d):
            return {"ok": False, "t": round((k + 1) * 0.02, 2)}
        h = yaw_of(d.qpos[3:7])
        yaws.append(h)
        if k >= 100:
            vfwd.append(float(d.qvel[:2] @ np.array([np.cos(h), np.sin(h)])))
    drift = abs(np.unwrap(yaws)[-1] - np.unwrap(yaws)[0]) / (SECS - 0.02)
    # speed in the robot's own frame: the yaw command is 0 and nothing closes the loop on
    # heading here, so distance along the start heading would mostly measure yaw drift
    return {"ok": True, "t": SECS, "speed": round(float(np.mean(vfwd)), 3),
            "yaw_drift_deg_s": round(float(np.degrees(drift)), 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True, help="a runs/<name> dir, or 'unitree'")
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--masses", type=float, nargs="*", default=MASSES)
    ap.add_argument("--protocol", choices=["walk", "stand"], default="walk")
    args = ap.parse_args()
    pol = UnitreePolicy() if args.policy == "unitree" else PlaygroundPolicy(Path(args.policy))
    out = {"policy": pol.name, "protocol": {"secs": SECS, "vx": VX, "kick": KICK, "n_kicks": N_KICKS,
                                            "carry_left": CARRY_LEFT.tolist()}, "results": {}}
    fn = trial if args.protocol == "walk" else stand_trial
    if args.protocol == "stand":
        out["protocol"] = {"secs": STAND_SECS, "command": [0, 0, 0], "carry_left": CARRY_LEFT.tolist()}
    for mass in args.masses:
        t0 = time.time()
        rs = [fn(pol, mass, seed) for seed in range(args.trials)]
        if args.protocol == "stand":
            ok = sum(r["ok"] for r in rs)
            w = [r["wander_m"] for r in rs if r["ok"]]
            out["results"][str(mass)] = {"success": ok, "trials": len(rs),
                                         "mean_wander_m": round(float(np.mean(w)), 3) if w else None,
                                         "fail_times": [r["t"] for r in rs if not r["ok"]]}
            print(f"{pol.name:>14}  {mass:4.1f} kg  stand {ok:3d}/{len(rs)}  wander "
                  f"{out['results'][str(mass)]['mean_wander_m']} m  ({time.time() - t0:.0f}s)", flush=True)
            continue
        ok = sum(r["ok"] for r in rs)
        sp = [r["speed"] for r in rs if r["ok"]]
        dr = [r["yaw_drift_deg_s"] for r in rs if r["ok"]]
        out["results"][str(mass)] = {"success": ok, "trials": len(rs),
                                     "mean_speed": round(float(np.mean(sp)), 3) if sp else None,
                                     "mean_yaw_drift_deg_s": round(float(np.mean(dr)), 2) if dr else None,
                                     "fail_times": [r["t"] for r in rs if not r["ok"]]}
        res = out["results"][str(mass)]
        print(f"{pol.name:>14}  {mass:4.1f} kg  {ok:3d}/{len(rs)}  forward speed {res['mean_speed']}  "
              f"yaw drift {res['mean_yaw_drift_deg_s']} deg/s  ({time.time() - t0:.0f}s)", flush=True)
    dest = ROOT / "results" / f"{'carry' if args.protocol == 'walk' else 'stand'}_{pol.name}.json"
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps(out, indent=1))
    print("wrote", dest)


if __name__ == "__main__":
    main()
