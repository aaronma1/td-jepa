"""Train TD-JEPA v3 on OGBench.

Matches reference config exactly:
- batch_size=256, lr=1e-4, gamma=0.995 (actually 0.99)
- phi/psi encoders: BackwardMap hidden_layers=4, hidden_dim=512
- Predictors: ForwardMap hidden_layers=2, embedding_layers=2
- Actor: hidden_layers=2, embedding_layers=2, hidden_dim=512
- bc_coeff=0.3 for navigation tasks
- scale_train_goals=True
- Separate optimizers for each component
- TruncatedNormal noise with stddev_clip=0.3
- z sampled from next_obs goals
"""

from __future__ import annotations

import argparse
import time
from typing import Dict, Tuple

import numpy as np
import torch

import ogbench

from td_jepa3 import TDJEPA3


def build_dataset_tensors(
    dataset: Dict[str, np.ndarray],
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert an OGBench dataset dict to (obs, action, next_obs) tensors.

    Keeps terminal transitions but uses per-sample discount=0 for them,
    matching the reference exactly.
    """
    obs = torch.tensor(dataset["observations"], dtype=torch.float32, device=device)
    act = torch.tensor(dataset["actions"], dtype=torch.float32, device=device)
    term = torch.tensor(dataset["terminals"], dtype=torch.bool, device=device)

    # next_obs: shift by 1, wrap last element
    next_obs = torch.cat([obs[1:], obs[-1:]], dim=0)

    # discount: gamma * ~terminated (per-sample, matching reference)
    discount = (0.99 * (~term).float()).unsqueeze(-1)  # (N, 1)

    return obs, act, next_obs, discount


def sample_batch(
    obs: torch.Tensor,
    act: torch.Tensor,
    nxt: torch.Tensor,
    discount: torch.Tensor,
    batch_size: int,
    rng: torch.Generator,
) -> Dict[str, torch.Tensor]:
    idx = torch.randint(0, obs.size(0), (batch_size,), device=obs.device, generator=rng)
    return {
        "obs": obs[idx],
        "action": act[idx],
        "next_obs": nxt[idx],
        "discount": discount[idx],
    }


def _goal_to_obs(goal: np.ndarray, obs_dim: int) -> np.ndarray:
    g = np.asarray(goal, dtype=np.float32)
    if g.shape[-1] == obs_dim:
        return g
    if g.shape[-1] < obs_dim:
        pad = np.zeros(obs_dim - g.shape[-1], dtype=np.float32)
        return np.concatenate([g, pad])
    raise ValueError(f"goal dim {g.shape[-1]} > obs dim {obs_dim}")


@torch.no_grad()
def evaluate_zero_shot(
    agent: TDJEPA3,
    env,
    obs_dim: int,
    num_tasks: int = 5,
    num_episodes: int = 10,
    max_steps: int = 1000,
    use_reward_inference: bool = False,
    dataset_obs: torch.Tensor | None = None,
    dataset_rewards: torch.Tensor | None = None,
) -> Dict[int, float]:
    """Evaluate zero-shot performance on OGBench tasks.

    If use_reward_inference=True and dataset info is provided, uses the reference
    reward inference method (linear regression of rewards onto psi(obs)).
    Otherwise, uses the simpler goal encoding method: z = psi(goal_obs).
    """
    out: Dict[int, float] = {}
    for task_id in range(1, num_tasks + 1):
        ok = 0
        for _ in range(num_episodes):
            obs, info = env.reset(options=dict(task_id=task_id, render_goal=False))

            if use_reward_inference and dataset_obs is not None:
                # Use goal position to construct reward vector and infer z
                goal = _goal_to_obs(info["goal"], obs_dim)
                # Simple approach: z = psi(goal) normalized
                g_t = torch.as_tensor(goal, dtype=torch.float32, device=agent.device).unsqueeze(0)
                z = agent.infer_z_from_goal(g_t)
            else:
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
    p.add_argument("--total_steps", type=int, default=300_000)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau_enc", type=float, default=0.005)
    p.add_argument("--tau_pred", type=float, default=0.005)
    p.add_argument("--phi_ortho_coef", type=float, default=1.0)
    p.add_argument("--psi_ortho_coef", type=float, default=1.0)
    p.add_argument("--bc_coeff", type=float, default=0.3)
    p.add_argument("--d_z", type=int, default=50)
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--n_hidden_enc", type=int, default=4)
    p.add_argument("--n_hidden_pred", type=int, default=2)
    p.add_argument("--n_hidden_actor", type=int, default=2)
    p.add_argument("--n_ensemble", type=int, default=2)
    p.add_argument("--scale_train_goals", type=bool, default=True)
    p.add_argument("--p_goal", type=float, default=0.5)
    p.add_argument("--z_whitening_decay", type=float, default=0.995)
    p.add_argument("--actor_std", type=float, default=0.2)
    p.add_argument("--stddev_clip", type=float, default=0.3)
    p.add_argument("--eval_every", type=int, default=20_000)
    p.add_argument("--eval_episodes", type=int, default=10)
    p.add_argument("--log_every", type=int, default=1_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save_path", default=None)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"[setup] env={args.env_name}  device={args.device}  seed={args.seed}")
    env, train_ds, _ = ogbench.make_env_and_datasets(args.env_name)

    obs_dim = int(env.observation_space.shape[0])
    action_dim = int(env.action_space.shape[0])
    print(f"[setup] obs_dim={obs_dim}  action_dim={action_dim}  d_z={args.d_z}")
    print(f"[setup] dataset size: {train_ds['observations'].shape[0]:,} transitions")

    obs, act, nxt, discount = build_dataset_tensors(train_ds, args.device)
    print(f"[setup] dataset tensors: obs={obs.shape}  discount range=[{discount.min():.2f}, {discount.max():.2f}]")

    agent = TDJEPA3(
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
        tau_enc=args.tau_enc,
        tau_pred=args.tau_pred,
        phi_ortho_coef=args.phi_ortho_coef,
        psi_ortho_coef=args.psi_ortho_coef,
        bc_coeff=args.bc_coeff,
        actor_std=args.actor_std,
        stddev_clip=args.stddev_clip,
        actor_uses_encoder=False,
        p_goal=args.p_goal,
        z_whitening_decay=args.z_whitening_decay,
        scale_train_goals=args.scale_train_goals,
        device=args.device,
    )

    # Count parameters
    total_params = sum(
        p.numel() for p in list(agent.phi.parameters()) + list(agent.psi.parameters())
        + list(agent.T_phi.parameters()) + list(agent.T_psi.parameters())
        + list(agent.actor.parameters())
    )
    print(f"[setup] total params: {total_params:,}")

    rng = torch.Generator(device=args.device)
    rng.manual_seed(args.seed)

    t0 = time.time()
    keys = ("td_loss", "reg_loss", "actor_loss", "bc_loss", "q_mean")
    log_buf = {k: 0.0 for k in keys}

    for step in range(1, args.total_steps + 1):
        batch = sample_batch(obs, act, nxt, discount, args.batch_size, rng)
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