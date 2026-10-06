"""Linden warehouse pick-and-place with a trained carry policy (physics only).

    python train/run_carry_demo.py --policy runs/payload10_800M --mass 5

Writes data/linden/carry_demo.mjb (the compiled model) and carry_demo.npz (25 fps qpos +
phase labels) for scripts/render_carry.py.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import carry_demo as D           # noqa: E402
import carry_env as C            # noqa: E402
from g1places.collision import Grid, smooth_path   # noqa: E402
from g1places.scenes import SCENES                 # noqa: E402
from policy import PolicyRunner                    # noqa: E402

FPS = 25
# Reach envelope inside the policy's arm range (IK residual < 1.2 cm): palms at 0.78-0.82 m
# reach 0.28-0.40 m ahead of the pelvis; at 0.74 m only to 0.36. So the robot stands in the
# middle of that band and grips the carton's sides 6 cm above their centre.
TORSO_TO_CARTON = 0.34
GRASP_DZ = 0.0
# Pick from the higher stack (top 0.66 m) and place on the lower one (0.557 m): carried the
# other way, the carton's underside (~0.62 m) hit the front of the 0.66 m stack and the robot
# could not step in. Gripping at mid-height keeps both palm heights (0.84 at the pick, 0.74
# at the drop) inside the reach envelope above.
PICK_GUESS, PLACE_GUESS = (5.15, 3.91), (1.71, -1.78)


def yaw_of(q):
    w, x, y, z = q
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class Demo:
    def __init__(self, policy_dir: Path, mass: float, carry_speed: float = 0.4):
        self.carry_speed = carry_speed
        sc = SCENES["linden"]
        mesh = np.load(sc.dir / "mesh.npz")
        v = sc.to_world(mesh["vertices"].astype(np.float64))
        aisle = np.load(sc.dir / "walk.npz")["qpos"][:, :2]
        self.pick = D.surface_from_mesh(v, mesh["faces"], np.array(PICK_GUESS), aisle)
        self.place = D.surface_from_mesh(v, mesh["faces"], np.array(PLACE_GUESS), aisle)
        self.grid = Grid(np.load(sc.dir / "pts_world.npy"))
        self.aisle = aisle

        cfg = C.default_config(); cfg.impl = "jax"
        self.env = C.CarryJoystick(config=cfg)
        u = self.pick.u
        c_xy = self.pick.edge_point() - u * (D.BOX_HALF[0] + 0.02)
        self.carton0 = np.r_[c_xy, self.pick.top + D.BOX_HALF[2] + 0.001]
        self.m = D.build(self.env, [self.pick, self.place], self.carton0, mass, carton_yaw=self.pick.yaw)
        self.m.opt.iterations, self.m.opt.ls_iterations = 50, 50     # accurate contacts for the demo
        self.d = mujoco.MjData(self.m)
        self.runner = PolicyRunner(policy_dir, self.env)
        self.pc = D.PalmController(self.m)
        self.default_arm = self.runner.default[15:]
        self.frames, self.labels, self.k = [], [], 0
        self.mass = mass

    # -- low-level ---------------------------------------------------------------------

    def pose(self):
        return self.d.qpos[:2].copy(), yaw_of(self.d.qpos[3:7])

    def tick(self, cmd, label, palm_targets=None):
        """One 50 Hz step. palm_targets None -> arms at the default (training) pose."""
        if palm_targets is None:
            arm_ctrl, arm_target = self.default_arm, self.default_arm
            self.pc.cmd = None
        else:
            arm_ctrl = self.pc.step(self.d, palm_targets)
            arm_target = self.pc.cmd
        self.runner.step(self.m, self.d, cmd, arm_ctrl, (arm_target - self.default_arm) / self.runner.scale)
        if self.k % (50 // FPS) == 0:
            self.frames.append(self.d.qpos.copy()); self.labels.append(label)
        self.k += 1
        if self.d.qpos[2] < 0.5:
            raise RuntimeError(f"fell during '{label}' at t={self.k / 50:.1f}s")

    def station_cmd(self, target_xy, heading, vmax=0.3, dead=0.02, vmin=0.12):
        """P control toward a pose in the body frame, with a minimum speed outside a small
        deadband: the policy barely moves for commands under ~0.1 m/s."""
        p, h = self.pose()
        R = np.array([[np.cos(h), np.sin(h)], [-np.sin(h), np.cos(h)]])
        e = R @ (np.asarray(target_xy) - p)
        def axis(err, gain, lim):
            if abs(err) < dead:
                return 0.0
            return float(np.sign(err) * np.clip(gain * abs(err), vmin, lim))
        return [axis(e[0], 2.5, vmax), axis(e[1], 2.5, 0.25), float(np.clip(3.0 * wrap(heading - h), -0.6, 0.6))]

    def walk_path(self, path, label, palms_fn=None, speed=0.5, lookahead=0.8, stop=0.3):
        s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
        for _ in range(int((s[-1] / speed + 10) * 50)):
            p, h = self.pose()
            i = int(np.argmin(np.linalg.norm(path - p, axis=1)))
            if s[-1] - s[i] < stop:
                return
            tgt = path[min(np.searchsorted(s, s[i] + lookahead), len(path) - 1)]
            err = wrap(np.arctan2(tgt[1] - p[1], tgt[0] - p[0]) - h)
            cmd = [speed * max(0.2, np.cos(err)), 0.0, float(np.clip(2.0 * err, -0.8, 0.8))]
            self.tick(cmd, label, palms_fn() if palms_fn else None)
        raise RuntimeError(f"did not finish '{label}'")

    def hold_station(self, target_xy, heading, secs, label, palms_fn=None):
        for _ in range(int(secs * 50)):
            self.tick(self.station_cmd(target_xy, heading), label, palms_fn() if palms_fn else None)

    def turn_to(self, heading, label, palms_fn=None, tol_deg=6.0, max_secs=12.0, hold_xy=None):
        """Rotate in place (the policy under-tracks small yaw commands, ~0.4-0.7x, so the
        gain is high and the floor on the command keeps it turning)."""
        for _ in range(int(max_secs * 50)):
            p, h = self.pose()
            e = wrap(heading - h)
            if abs(np.degrees(e)) < tol_deg:
                return
            wz = float(np.sign(e) * np.clip(3.0 * abs(e), 0.3, 0.7))
            # hold position while turning: with a load the policy translates a lot when it yaws
            vx, vy = self.station_cmd(hold_xy, heading)[:2] if hold_xy is not None else (0.0, 0.0)
            self.tick([vx, vy, wz], label, palms_fn() if palms_fn else None)
        raise RuntimeError(f"could not turn for '{label}': {np.degrees(wrap(heading - self.pose()[1])):+.1f} deg left")

    def approach(self, line_from, line_to, heading, label, palms_fn=None, lat_tol=0.06):
        """Walk straight in along a line, steering by turning only (this policy barely
        follows small sideways commands), then settle facing `heading`."""
        self.turn_to(heading, label, palms_fn, hold_xy=np.asarray(line_from))
        n = max(2, int(np.linalg.norm(np.asarray(line_to) - line_from) / 0.05))
        line = np.linspace(line_from, line_to, n)
        # extend past the goal so the lookahead point never runs out
        line = np.vstack([line, line[-1] + (line[-1] - line[0]) / np.linalg.norm(line[-1] - line[0]) * np.linspace(0.05, 0.5, 10)[:, None]])
        s_end = np.linalg.norm(np.asarray(line_to) - line_from)
        for _ in range(int(30 * 50)):
            p, h = self.pose()
            along = (p - line_from) @ (np.asarray(line_to) - line_from) / s_end
            if along >= s_end - 0.02:
                break
            i = int(np.argmin(np.linalg.norm(line - p, axis=1)))
            tgt = line[min(i + 6, len(line) - 1)]
            err = wrap(np.arctan2(tgt[1] - p[1], tgt[0] - p[0]) - h)
            v = float(np.clip(1.2 * (s_end - along), 0.15, 0.25))
            lat = float((p - line_from) @ np.array([-(line_to[1] - line_from[1]), line_to[0] - line_from[0]]) / s_end)
            self.tick([v * max(0.2, np.cos(err)), float(np.clip(-1.5 * lat, -0.2, 0.2)),
                       float(np.clip(4.0 * err, -0.8, 0.8))], label, palms_fn() if palms_fn else None)
        perp_u = np.array([-(line_to[1] - line_from[1]), line_to[0] - line_from[0]]) / s_end
        for _ in range(150):                 # settle: along-track, lateral and heading
            p, h = self.pose()
            along = (p - line_from) @ (np.asarray(line_to) - line_from) / s_end
            lat = float((p - line_from) @ perp_u)
            Rb = np.array([[np.cos(h), np.sin(h)], [-np.sin(h), np.cos(h)]])
            v_world = (np.asarray(line_to) - line_from) / s_end * (s_end - along) - perp_u * lat
            vb = Rb @ v_world
            self.tick([float(np.clip(2.0 * vb[0], -0.15, 0.15)), float(np.clip(2.0 * vb[1], -0.2, 0.2)),
                       float(np.clip(4.0 * wrap(heading - h), -0.6, 0.6))], label, palms_fn() if palms_fn else None)
        p, h = self.pose()
        perp = np.array([-(line_to[1] - line_from[1]), line_to[0] - line_from[0]]) / s_end
        lateral = float((p - line_to) @ perp)
        if abs(lateral) > lat_tol or abs(np.degrees(wrap(heading - h))) > 6:
            raise RuntimeError(f"approach '{label}' ended {lateral:+.2f} m off the line, "
                               f"{np.degrees(wrap(heading - h)):+.1f} deg")
        return lateral

    def back_off(self, target_xy, heading, label, palms_fn=None, tol=0.08, max_secs=8.0):
        """Back out toward an aisle point, keeping the heading. With a load the policy walks
        backwards slowly (~2 cm/s at 5 kg), so this is a short step clear of the shelf and
        it moves on after max_secs wherever it got to."""
        for _ in range(int(max_secs * 50)):
            p, _ = self.pose()
            if np.linalg.norm(np.asarray(target_xy) - p) < tol:
                return
            self.tick(self.station_cmd(target_xy, heading), label, palms_fn() if palms_fn else None)

    def follow(self, path, label, palms_fn=None, speed=0.4):
        """Walk a path. Empty-handed it first turns in place toward the path; loaded, it turns
        by arcing (walk_path keeps a little forward speed while it yaws), because spinning on
        the spot with 5 kg in the arms is where it fell."""
        if palms_fn is None:
            d0 = path[min(5, len(path) - 1)] - path[0]
            self.turn_to(float(np.arctan2(d0[1], d0[0])), label, None, tol_deg=12.0, hold_xy=path[0])
        self.walk_path(path, label, palms_fn, speed=speed, stop=0.1)

    def line_up(self, target_xy, heading, label, palms_fn=None, tol=0.06, tol_deg=5.0, max_secs=12.0):
        """Station-keep until within tol (m) and tol_deg for 1 s, or give up. The policy never
        stands still (its gait phase always runs), so a few cm is as tight as it holds."""
        good = 0
        for _ in range(int(max_secs * 50)):
            p, h = self.pose()
            ok = np.linalg.norm(np.asarray(target_xy) - p) < tol and abs(np.degrees(wrap(heading - h))) < tol_deg
            good = good + 1 if ok else 0
            if good >= 50:
                return
            self.tick(self.station_cmd(target_xy, heading), label, palms_fn() if palms_fn else None)
        p, h = self.pose()
        raise RuntimeError(f"could not line up for '{label}': {np.linalg.norm(np.asarray(target_xy) - p):.2f} m, "
                           f"{np.degrees(wrap(heading - h)):.1f} deg off")

    # -- palm targets ------------------------------------------------------------------

    def palms_on_carton(self, centre, yaw, dy, dz=0.0):
        """World targets for both palm sites, dy beyond the pad-contact width."""
        side = np.array([-np.sin(yaw), np.cos(yaw), 0.0])       # carton +y
        w = D.BOX_HALF[1] + D.PAD_INSET + dy
        c = np.asarray(centre) + np.array([0, 0, GRASP_DZ + dz - 0.01])
        # the robot faces -u, so its left is the carton's -y side
        return np.array([c - side * w, c + side * w])

    def palms_in_torso(self, local):
        p, h = self.pose()
        R = np.array([[np.cos(h), -np.sin(h), 0], [np.sin(h), np.cos(h), 0], [0, 0, 1]])
        base = np.r_[p, 0.0]
        return np.array([base + R @ l for l in local])

    # -- the task ----------------------------------------------------------------------

    def plan(self, a, b):
        return smooth_path(self.grid.plan(a, b, clearance=0.45), step=0.1, window=7)

    def run(self):
        d, m = self.d, self.m
        u = self.pick.u
        stand_pick = self.carton0[:2] + u * TORSO_TO_CARTON
        approach_pick = self.pick.edge_point() + u * 0.9
        h_pick = wrap(self.pick.yaw + np.pi)
        # start 4 m back along the aisle from the approach point, facing it
        j = int(np.argmin(np.linalg.norm(self.aisle - approach_pick, axis=1)))
        start = self.aisle[max(0, j - 160)]
        heading0 = np.arctan2(*(approach_pick - start)[::-1])
        d.qpos[:36] = self.env.mj_model.keyframe("knees_bent").qpos
        d.qpos[:2] = start
        d.qpos[3:7] = [np.cos(heading0 / 2), 0, 0, np.sin(heading0 / 2)]
        d.qpos[36:39] = self.carton0
        d.qpos[39:43] = [np.cos(self.pick.yaw / 2), 0, 0, np.sin(self.pick.yaw / 2)]
        mujoco.mj_forward(m, d)
        self.runner.reset()

        self.hold_station(start, heading0, 1.0, "settle")
        self.follow(self.plan(start, approach_pick), "walk to the stack", speed=0.5)
        self.approach(approach_pick, stand_pick, h_pick, "line up")
        cc = self.carton0
        pre = lambda: self.palms_on_carton(cc, self.pick.yaw, 0.07)
        self.hold_station(stand_pick, h_pick, 1.5, "reach", pre)
        for k in range(50):
            a = (k + 1) / 50
            self.tick(self.station_cmd(stand_pick, h_pick), "grip",
                      self.palms_on_carton(cc, self.pick.yaw, 0.07 * (1 - a) - 0.01 * a))
        for k in range(50):                   # keep squeezing until both pads touch
            if D.pads_touching(m, d):
                break
            self.tick(self.station_cmd(stand_pick, h_pick), "grip",
                      self.palms_on_carton(cc, self.pick.yaw, -0.01 - 0.0004 * k))
        touching = D.pads_touching(m, d)
        if not touching:
            raise RuntimeError("pads never touched the carton")
        D.grip(m, d, True)
        self.pc.load_kg = self.mass
        for k in range(75):
            a = (k + 1) / 75
            self.tick(self.station_cmd(stand_pick, h_pick), "lift",
                      self.palms_on_carton(cc, self.pick.yaw, 0.0, 0.10 * a))
        # carry: hold the palms where they are, in the robot's frame
        p, h = self.pose()
        R = np.array([[np.cos(h), np.sin(h), 0], [-np.sin(h), np.cos(h), 0], [0, 0, 1]])
        local = [R @ (x - np.r_[p, 0.0]) for x in self.pc.palms(d)]
        carry = lambda: self.palms_in_torso(local)
        self.back_off(self.carton0[:2] + u * (TORSO_TO_CARTON + 0.25), h_pick, "step back", carry)

        up = self.place.u
        stand_place_c = self.place.edge_point() - up * (D.BOX_HALF[0] + 0.02)
        stand_place = stand_place_c + up * TORSO_TO_CARTON
        approach_place = self.place.edge_point() + up * 0.9
        h_place = wrap(self.place.yaw + np.pi)
        self.follow(self.plan(self.pose()[0], approach_place), "carry", carry, speed=self.carry_speed)
        # placing needs less precision than picking: the carton goes down straight ahead of
        # wherever the robot ends up, as long as that is on the stack top
        self.approach(approach_place, stand_place, h_place, "line up", carry, lat_tol=0.15)
        vperp = np.array([-up[1], up[0]])

        def drop_point():
            p, h = self.pose()
            xy = p + np.array([np.cos(h), np.sin(h)]) * TORSO_TO_CARTON
            rel = xy - self.place.centre
            inside = abs(rel @ vperp) <= self.place.half[1] - D.BOX_HALF[1] and rel @ up <= self.place.half[0] - 0.08
            return xy, inside

        for _ in range(int(10 * 50)):          # step in until the carton would land on the stack
            drop_xy, inside = drop_point()
            if inside:
                break
            self.tick(self.station_cmd(stand_place, h_place), "line up", carry())
        drop_xy, inside = drop_point()
        if not inside:
            raise RuntimeError("drop point is off the stack top")
        stand_place_c = drop_xy
        drop = np.r_[drop_xy, self.place.top + D.BOX_HALF[2] + 0.02]
        p0 = self.pc.palms(d)
        tgt = self.palms_on_carton(drop, self.place.yaw, 0.0)
        for k in range(100):
            a = min(1, (k + 1) / 80)
            self.tick(self.station_cmd(stand_place, h_place), "lower", p0 + (tgt - p0) * a)
        D.grip(m, d, False)
        self.pc.load_kg = 0.0
        for k in range(50):
            a = (k + 1) / 50
            self.tick(self.station_cmd(stand_place, h_place), "release",
                      self.palms_on_carton(drop, self.place.yaw, 0.07 * a))
        back = self.place.edge_point() + up * 1.1
        self.back_off(back, h_place, "step back")
        self.hold_station(back, h_place, 1.0, "done")
        cb = m.body("carton").id
        carton = d.xpos[cb].copy()
        on_target = np.linalg.norm(carton[:2] - stand_place_c)
        R = d.xmat[cb].reshape(3, 3)
        return {"pads_touching_at_grip": bool(touching), "carton_final": carton.round(3).tolist(),
                "carton_xy_error_m": round(float(on_target), 3),
                "carton_height_above_place_top_m": round(float(carton[2] - D.BOX_HALF[2] - self.place.top), 3),
                "carton_tilt_deg": round(float(np.degrees(np.arccos(np.clip(R[2, 2], -1, 1)))), 1),
                "seconds": round(self.k / 50, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", type=Path, required=True)
    ap.add_argument("--mass", type=float, default=5.0)
    ap.add_argument("--carry-speed", type=float, default=0.4)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "linden" / "carry_demo")
    args = ap.parse_args()
    demo = Demo(args.policy, args.mass, args.carry_speed)
    try:
        result = demo.run()
        result["ok"] = True
    except RuntimeError as e:
        result = {"ok": False, "error": str(e), "seconds": round(demo.k / 50, 1)}
    print(result)
    mujoco.mj_saveModel(demo.m, str(args.out) + ".mjb")
    np.savez(str(args.out) + ".npz", qpos=np.array(demo.frames), labels=np.array(demo.labels),
             fps=FPS, mass=args.mass, policy=str(args.policy), **{k: str(v) for k, v in result.items()})


if __name__ == "__main__":
    main()
