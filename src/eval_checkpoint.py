"""Load a TD-JEPA checkpoint written by train.py / train_mixed.py and run zero-shot eval.

The env, obs/act dims and training params are read from the checkpoint; the
agent config comes from the .params file (see params.py). Any TrainParams flag overrides eval settings.

Usage:
  uv run eval_checkpoint.py --ckpt checkpoints/antmaze-medium-navigate-v0/final.pt
  uv run eval_checkpoint.py --ckpt checkpoints/walker_proto/latest.pt --eval_episodes 20
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import dataclasses

import torch

import params
from agents.td_jepa import TD_JEPA
from train import add_train_param_args, cli_overrides, get_dataset, make_evaluator


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="Path to a checkpoint .pt file")
    p.add_argument("--env_name", default=None,
                   help="Eval env (defaults to the env stored in the checkpoint)")
    add_train_param_args(p)
    args = p.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    env_name = args.env_name or ckpt["env_name"]

    # Start from the run's own params (if stored), then apply CLI overrides.
    known = {f.name for f in dataclasses.fields(params.TrainParams)}
    stored = {k: v for k, v in ckpt.get("train_params", {}).items() if k in known}
    tp = params.train_params(env_name, **{**stored, "resume": "", **cli_overrides(args)})

    config = params.agent_config(env_name, ckpt["obs_dim"], ckpt["act_dim"], tp.bfm_name)
    config.train_device = tp.device
    agent = TD_JEPA(config)  # pyright: ignore[reportArgumentType]
    agent.load_state_dict({k: ckpt[k] for k in ("modules", "z_cov")})
    agent.train_mode()
    print(f"Loaded {args.ckpt} (step {ckpt['step']}, trained on {ckpt['env_name']})")

    buffer = get_dataset(env_name, tp)
    results, summary = make_evaluator(env_name, buffer, tp)(agent, ckpt["step"])
    print(f"\n=== Eval @ step {ckpt['step']} (max_steps={tp.eval_max_steps}) ===")
    for task, r in results.items():
        print(f"  {task}: {r:.3f}")
    print(f"  {summary}")


if __name__ == "__main__":
    main()
