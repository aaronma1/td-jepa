"""Train TD-JEPA v2 on OGBench pointmaze and evaluate zero-shot goal-reaching.

Key changes from v1 train.py
-----------------------------
* Uses TDJEPA2 with d_z=50 (not 256), both encoders L2-normalised.
* Default batch_size=1024 (matches reference).
* Default lr=3e-4, bc_alpha=0.3 (reference uses this for navigation).
* Actor uses raw obs (actor_uses_encoder=False) for OGBench.
* Predictor ensemble of 2.
* Z whitening enabled.
"""

from __future__ import annotations

import argparse
import time
from typing import Dict, Tuple

import numpy as np
import torch

import ogbench

from td_jepa2 import TDJEPA2


def build_dataset_tensors(
    dataset: Dict[str, np.ndarray],
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert an OGBench dataset dict to (obs, action, next_obs) tensors.

    Drops rows where terminals==1 so next_obs is always from the same
    trajectory.
    """
    obs = dataset["observations"]
    act = dataset["actions"]
    term = dataset["terminals"]

    next_obs = np.concatenate([obs[1:], obs[-1:]], axis=0)
    valid = term == 0

    obs, act, next_obs = obs[valid], act[valid], next_obs[valid]
    return (
        torch.tensor(obs, dtype=torch.float32, device=device),
        torch.tensor(act, dtype=torch.float32, device=device),
        torch.tensor(next_obs, dtype=torch.float32, device=device),
    )


def sample_batch(
    obs: torch.Tensor,
    act: torch.Tensor,
    nxt: torch.Tensor,
    batch_size: int,
    rng: torch.Generator,
) -> Dict[str, torch.Tensor]:
    idx = torch.randint(0, obs.size(0), (batch_size,), device=obs.device, generator=rng)
    return {"obs": obs[idx], "action": act[idx], "next_obs": nxt[idx]}


def _goal_to_obs(goal: np.ndarray, obs_dim: int) -> np.ndarray:
    """Pad position-only goals with zeros so they live in the obs space."""
    g = np.asarray(goal, dtype=np.float32)
    if g.shape[-1] == obs_dim:
        return g
    if g.shape[-1] < obs_dim:
        pad = np.zeros(obs_dim - g.shape[-1], dtype=np.float32)
        return np.concatenate([g, pad])
    raise ValueError(f"goal dim {g.shape[-1]} > obs dim {obs_dim}")


@torch.no_grad()
def evaluate_zero_shot(
    agent: TDJEPA2,
    env,
    obs_dim: int,
    num_tasks: int = 5,
    num_episodes: int = 10,
    max_steps: int = 1000,
) -> Dict[int, float]:
    """Run the 5 default OGBench tasks; return per-task success rate."""
    out: Dict[int, float] = {}
    for task_id in range(1, num_tasks + 1):
        ok = 0
        for _ in range(num_episodes):
            obs, info = env.reset(options=dict(task_id=task_id, render_goal=False))
            goal = _goal_to_obs(info["goal"], obs_dim)
            g_t = torch.as_tensor(goal, dtype=torch.float32, device=agent.device).unsqueeze(0)
            z = agent.infer_z_from_goal(g_t)
            for _ in range(max_steps):
                o_t = torch.as_tensor(obs, dtype=torch.float32, device=agent.device).unsqueeze(0)
                a = agent.act(o_t, z, deterministic=True).cpu().numpy().squeeze(0)
                obs, _, term, trunc, info = env.step(a)
                if info.get("success", False):
                    ok += 1
                    break
                if term or trunc:
                    break
        out[task_id] = ok / num_episodes
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--env_name", default="pointmaze-medium-navigate-v0")
    p.add_argument("--total_steps", type=int, default=1_000_000)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--lambda_reg", type=float, default=1.0)
    p.add_argument("--bc_alpha", type=float, default=0.3,
                   help="BC regularisation coefficient (0.3 for navigation, 3.0 for manipulation per ref)")
    p.add_argument("--d_z", type=int, default=50,
                   help="Latent dimension for BOTH phi and psi (reference: 50)")
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--n_hidden_enc", type=int, default=3,
                   help="Encoder hidden layers (reference hidden_layers=4 → 3 hidden+1 output+1 embedding=5 total)")
    p.add_argument("--n_hidden_pred", type=int, default=2,
                   help="Predictor body hidden layers (reference hidden_layers=2)")
    p.add_argument("--n_hidden_actor", type=int, default=1,
                   help="Actor body depth (reference: 1)")
    p.add_argument("--n_ensemble", type=int, default=2,
                   help="Number of predictor networks in ensemble")
    p.add_argument("--eval_every", type=int, default=20_000)
    p.add_argument("--eval_episodes", type=int, default=10)
    p.add_argument("--log_every", type=int, default=1_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save_path", default=None, help="Optional path to save final state dict.")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"[setup] env={args.env_name}  device={args.device}  seed={args.seed}")
    env, train_ds, _ = ogbench.make_env_and_datasets(args.env_name)

    obs_dim = int(env.observation_space.shape[0])
    action_dim = int(env.action_space.shape[0])
    print(f"[setup] obs_dim={obs_dim}  action_dim={action_dim}  d_z={args.d_z}")
    print(f"[setup] dataset (raw): {train_ds['observations'].shape[0]:,} transitions")

    obs, act, nxt = build_dataset_tensors(train_ds, args.device)
    print(f"[setup] dataset (valid): {obs.size(0):,} transitions")

    agent = TDJEPA2(
        obs_dim=obs_dim,
        action_dim=action_dim,
        d_z=args.d_z,
        hidden_dim=args.hidden_dim,
        n_hidden_enc=args.n_hidden_enc,
        n_hidden_pred=args.n_hidden_pred,
        n_hidden_actor=args.n_hidden_actor,
        n_ensemble=args.n_ensemble,
        gamma=args.gamma,
        lr=args.lr,
        tau=args.tau,
        lambda_reg=args.lambda_reg,
        bc_alpha=args.bc_alpha,
        actor_uses_encoder=False,  # OGBench: actor uses raw obs
        p_goal=0.5,
        z_whitening_decay=0.995,
        device=args.device,
    )

    rng = torch.Generator(device=args.device)
    rng.manual_seed(args.seed)

    t0 = time.time()
    keys = ("td_loss", "reg_loss", "actor_loss", "bc_loss", "q_mean")
    log_buf = {k: 0.0 for k in keys}

    for step in range(1, args.total_steps + 1):
        batch = sample_batch(obs, act, nxt, args.batch_size, rng)
        info = agent.update(batch)
        for k in keys:
            log_buf[k] += info[k]

        if step % args.log_every == 0:
            sps = step / (time.time() - t0)
            msg = "  ".join(f"{k}={log_buf[k] / args.log_every:+.4f}" for k in keys)
            print(f"[{step:>7d}/{args.total_steps}] ({sps:>4.0f} sps)  {msg}")
            log_buf = {k: 0.0 for k in keys}

        if step % args.eval_every == 0:
            sr = evaluate_zero_shot(agent, env, obs_dim, num_episodes=args.eval_episodes)
            mean_sr = float(np.mean(list(sr.values())))
            per = ", ".join(f"t{k}={v:.2f}" for k, v in sr.items())
            print(f"            >> zero-shot  mean={mean_sr:.3f}  [{per}]")

    print("\n[done] final eval (50 episodes / task) ...")
    sr = evaluate_zero_shot(agent, env, obs_dim, num_episodes=50)
    print(f"FINAL  mean_success={float(np.mean(list(sr.values()))):.3f}")
    for k, v in sr.items():
        print(f"  task {k}: {v:.3f}")

    if args.save_path:
        torch.save(agent.state_dict(), args.save_path)
        print(f"[done] saved to {args.save_path}")


if __name__ == "__main__":
    main()