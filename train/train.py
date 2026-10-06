"""Train the G1 carry-locomotion policy with Brax PPO (MuJoCo Playground settings).

    python train/train.py --max-payload 10 --out runs/payload10      # the policy
    python train/train.py --max-payload 0  --out runs/payload0       # ablation
    python train/train.py --smoke                                    # CPU wiring check
"""

from __future__ import annotations

import argparse
import functools
import json
import pickle
import sys
import time
from pathlib import Path

import jax
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from mujoco_playground import wrapper
from mujoco_playground.config import locomotion_params

sys.path.insert(0, str(Path(__file__).resolve().parent))
import carry_env  # noqa: E402

BASE = "G1JoystickFlatTerrain"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-payload", type=float, default=10.0)
    ap.add_argument("--out", type=Path, default=Path("runs/smoke"))
    ap.add_argument("--timesteps", type=int, default=None, help="default: Playground's G1 setting")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--impl", default="warp", choices=["warp", "jax"])
    ap.add_argument("--smoke", action="store_true", help="tiny CPU run to check the wiring")
    ap.add_argument("--p-stand", type=float, default=0.1, help="share of all-zero commands")
    ap.add_argument("--stand-fix", action="store_true", help="stand-still penalty over legs+waist only")
    ap.add_argument("--restore", type=Path, default=None,
                    help="params.pkl to continue from (normalizer, policy, value; fresh optimizer)")
    args = ap.parse_args()
    if args.smoke:
        args.impl = "jax"

    env_cfg = carry_env.default_config(args.max_payload, p_stand=args.p_stand, stand_fix=args.stand_fix)
    env_cfg.impl = args.impl
    env = carry_env.CarryJoystick(config=env_cfg)
    eval_env = carry_env.CarryJoystick(config=env_cfg)

    ppo_cfg = locomotion_params.brax_ppo_config(BASE, args.impl)
    params = dict(ppo_cfg)
    net = params.pop("network_factory")
    if args.timesteps:
        params["num_timesteps"] = args.timesteps
    if args.smoke:
        params.update(num_timesteps=4096, num_envs=16, batch_size=16, num_minibatches=2,
                      num_evals=2, num_eval_envs=4, unroll_length=8, episode_length=64,
                      num_resets_per_eval=1)
    payload_ids = env._payload_ids
    randomizer = functools.partial(carry_env.domain_randomize, payload_ids=payload_ids,
                                   max_payload=args.max_payload)

    args.out.mkdir(parents=True, exist_ok=True)
    log, t0 = [], time.time()

    def progress(step, metrics):
        row = {"step": int(step), "minutes": round((time.time() - t0) / 60, 2),
               "reward": float(metrics.get("eval/episode_reward", float("nan"))),
               "episode_length": float(metrics.get("eval/avg_episode_length", float("nan")))}
        log.append(row)
        print(json.dumps(row), flush=True)
        (args.out / "progress.json").write_text(json.dumps(log, indent=1))

    restore = None
    if args.restore:
        with open(args.restore, "rb") as f:
            restore = pickle.load(f)
    make_inference_fn, policy_params, _ = ppo.train(
        restore_params=restore,
        environment=env, eval_env=eval_env, wrap_env_fn=wrapper.wrap_for_brax_training,
        network_factory=functools.partial(ppo_networks.make_ppo_networks, **net),
        randomization_fn=randomizer, progress_fn=progress, seed=args.seed, **params)

    with open(args.out / "params.pkl", "wb") as f:
        pickle.dump(jax.device_get(policy_params), f)
    meta = {"max_payload": args.max_payload, "seed": args.seed, "impl": args.impl,
            "restored_from": str(args.restore) if args.restore else None,
            "p_stand": args.p_stand, "stand_fix": args.stand_fix,
            "ppo": {k: (list(v) if isinstance(v, tuple) else v) for k, v in params.items()},
            "network": {k: list(v) if isinstance(v, tuple) else v for k, v in dict(net).items()},
            "minutes": round((time.time() - t0) / 60, 1), "devices": [str(d) for d in jax.devices()]}
    (args.out / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
    print("saved", args.out)


if __name__ == "__main__":
    main()
