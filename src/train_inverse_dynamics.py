"""Train TD-JEPA (v2) on OGBench antmaze-medium-navigate-v0 (proprio).

Matches the OGBench-proprio config from the reference repo's
``scripts/train/proprio/launch_td_jepa_ogbench.py`` ``sweep_antmaze`` sweep:
1M steps, batch 256, gamma 0.99, bc_coeff 0.3, ortho coefs 1.0 (psi can go 0.1),
phi_dim = psi_dim = 50, encoder hidden 512 / 4 layers, predictor hidden 512 /
2 layers, actor takes raw obs (actor_use_full_encoder=False).

Reward inference matches the reference's ``ogbench_reward_eval``: sample N
buffer transitions, compute the task's true reward at each (sparse: +1 inside
the goal radius, 0 outside), shift by +1 to match the published numbers, then
least-squares onto ``psi(next_obs)``.

Example
-------
    pip install ogbench torch
    python train_v2.py
    # or for the harder large variant:
    python train_v2.py --env_name antmaze-large-navigate-v0
"""
from __future__ import annotations

import argparse
import time
from typing import Dict, Tuple

import numpy as np
import torch

import ogbench

from td_jepa_inverse_dynamics import TDJEPA


# OGBench's goal-reach tolerance, see ogbench/locomaze/maze.py: `self._goal_tol`.
# 0.5 for ant/humanoid, 1.0 for the simpler point morphology.
GOAL_TOL = {
    "ant": 0.5,
    "point": 1.0,
    "humanoid": 0.5,
}


def _detect_morph(env_name: str) -> str:
    for m in GOAL_TOL:
        if env_name.startswith(m):
            return m
    return "ant"


def build_dataset_tensors(
    dataset: Dict[str, np.ndarray],
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """OGBench dataset -> tensors.  Drop trajectory-boundary transitions."""
    obs = dataset["observations"]
    act = dataset["actions"]
    term = dataset["terminals"]

    next_obs = np.concatenate([obs[1:], obs[-1:]], axis=0)
    valid = term == 0

    obs, act, next_obs = obs[valid], act[valid], next_obs[valid]
    term_valid = term[valid].astype(np.float32)
    return (
        torch.tensor(obs, dtype=torch.float32, device=device),
        torch.tensor(act, dtype=torch.float32, device=device),
        torch.tensor(next_obs, dtype=torch.float32, device=device),
        torch.tensor(term_valid, dtype=torch.float32, device=device),
    )


def sample_batch(
    obs, act, nxt, term, batch_size, gamma, rng,
) -> Dict[str, torch.Tensor]:
    idx = torch.randint(0, obs.size(0), (batch_size,), device=obs.device, generator=rng)
    return {
        "obs": obs[idx],
        "action": act[idx],
        "next_obs": nxt[idx],
        "discount": gamma * (1.0 - term[idx]).unsqueeze(-1),
    }


@torch.no_grad()
def infer_z_for_task(
    agent: TDJEPA,
    next_obs_pool: torch.Tensor,
    goal_xy: np.ndarray,
    tol: float,
    shift_reward: float = 1.0,
    center_reward: bool = False,
    verbose: bool = False,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Reward inference matching ``ogbench_reward_eval``.

    Samples are the pool's ``next_obs``; reward is OGBench's true sparse goal
    reward (+1 if within tolerance, 0 otherwise), shifted by ``shift_reward``.

    If ``center_reward=True``, the reward vector is mean-centered before
    least-squares regression.  Centering removes the constant component of the
    reward (which gets absorbed by ``mean(psi)`` if psi has a nonzero data-mean
    direction) and isolates the task-specific signal.  This is an ablation
    over the un-centered formulation used in the reference's ``reward_inference``.

    Returns (z, diagnostics).  Diagnostics include the number of positive-reward
    samples in the pool (``positives``), the pool xy bounding box, and the
    distance from the goal to the nearest pool point.  When ``positives`` is
    tiny (e.g. <50 out of 10k), the least-squares regression is fitting a
    near-zero target and the inferred z is meaningless.
    """
    goal_t = torch.as_tensor(goal_xy, dtype=torch.float32, device=agent.device)
    xy = next_obs_pool[:, :2]                               # ant: first 2 qpos dims are world (x, y)
    dist = (xy - goal_t).norm(dim=-1)
    rew = (dist <= tol).float() + shift_reward
    n_pos = int(rew.sum().item() - rew.numel() * shift_reward)
    if center_reward:
        rew = rew - rew.mean()
    diag = {
        "positives": n_pos,
        "min_dist": float(dist.min().item()),
        "xy_min_x": float(xy[:, 0].min().item()),
        "xy_max_x": float(xy[:, 0].max().item()),
        "xy_min_y": float(xy[:, 1].min().item()),
        "xy_max_y": float(xy[:, 1].max().item()),
    }
    if verbose:
        centered_tag = " centered" if center_reward else ""
        print(
            f"        [z-inf{centered_tag}] goal=({goal_xy[0]:+.2f}, {goal_xy[1]:+.2f})  "
            f"positives={n_pos}/{rew.numel()}  min_dist={diag['min_dist']:.3f}  "
            f"pool x=[{diag['xy_min_x']:+.2f}, {diag['xy_max_x']:+.2f}]  "
            f"y=[{diag['xy_min_y']:+.2f}, {diag['xy_max_y']:+.2f}]"
        )
    z = agent.infer_z_from_rewards(next_obs_pool, rew)
    return z, diag


@torch.no_grad()
def evaluate_zero_shot(
    agent: TDJEPA,
    env,
    next_obs_pool: torch.Tensor,
    morph: str,
    num_tasks: int = 5,
    num_episodes: int = 10,
    max_steps: int = 1000,
    num_inference_samples: int = 10_000,
    center_reward: bool = False,
    verbose: bool = True,
) -> Dict[int, float]:
    """OGBench-style 5-task zero-shot eval with reward inference + diagnostics.

    Prints, every time it runs:
      * ``[psi-mean]``: ``||mean(psi(pool))||``.  If large (>> 0), psi has a
        nonzero data-mean direction and the un-centered reward inference will
        absorb the constant component of every task's reward into that direction,
        making all task z's collapse onto cos~1.  Healthy: << sqrt(d_psi).
      * For each task: positive-reward count in the pool, min dist from goal to
        pool, and pool xy bounding box.  Catches "empty reward signal".
      * Pairwise cosine similarity matrix between inferred z's.  If z's for
        different tasks point in nearly the same direction (cos > 0.9 across
        the board), the eval distinguishes tasks but ψ may be projecting them
        into one direction -- behavioral collapse.
      * Predictor z-sensitivity: on a fixed (s, a) batch, evaluate
        ``Q = T_phi(phi(s), z, a) . z`` for each task's z and a uniform-sphere
        z.  The "spread" across z's measures whether the predictor actually
        uses z.  If spread << |Q|, the predictor has collapsed to z-independence.

    ``center_reward``: if True, mean-center the reward before lstsq inference.
    Removes the constant-reward component that gets absorbed by ``mean(psi)``.
    """
    tol = GOAL_TOL[morph]
    out: Dict[int, float] = {}

    n = min(num_inference_samples, next_obs_pool.shape[0])
    pool_idx = torch.randperm(next_obs_pool.shape[0], device=agent.device)[:n]
    pool = next_obs_pool[pool_idx]

    if verbose:
        psi_pool = agent.psi(pool)                          # (N, d_psi)
        mean_psi = psi_pool.mean(dim=0)
        mean_psi_norm = float(mean_psi.norm().item())
        sqrt_d = agent.psi_dim ** 0.5
        print(
            f"        [psi-mean] ||mean(psi(pool))|| = {mean_psi_norm:.3f}   "
            f"(sqrt(d_psi)={sqrt_d:.2f}; collapse if mean_norm ~ sqrt_d, "
            f"healthy if << sqrt_d)"
        )

    # Collect z's so we can diagnose z-geometry across tasks
    z_per_task: Dict[int, torch.Tensor] = {}
    diag_per_task: Dict[int, Dict[str, float]] = {}

    for task_id in range(1, num_tasks + 1):
        obs, info = env.reset(options=dict(task_id=task_id, render_goal=False))
        goal_xy = np.asarray(
            info.get("goal", getattr(env.unwrapped, "cur_goal_xy", np.zeros(2)))
        )[:2]

        z, diag = infer_z_for_task(
            agent, pool, goal_xy, tol,
            center_reward=center_reward, verbose=verbose,
        )
        z_per_task[task_id] = z
        diag_per_task[task_id] = diag

        ok = 0
        for _ in range(num_episodes):
            obs, info = env.reset(options=dict(task_id=task_id, render_goal=False))
            for _ in range(max_steps):
                o_t = torch.as_tensor(obs, dtype=torch.float32, device=agent.device).unsqueeze(0)
                a = agent.act(o_t, z, deterministic=True).cpu().numpy().squeeze(0)
                obs, _, term, trunc, info = env.step(a)
                if float(info.get("success", 0.0)) > 0.5:
                    ok += 1
                    break
                if term or trunc:
                    break
        out[task_id] = ok / num_episodes

    if verbose:
        _print_z_geometry(z_per_task)
        _print_predictor_z_sensitivity(agent, next_obs_pool, z_per_task)

    return out


@torch.no_grad()
def _print_z_geometry(z_per_task: Dict[int, torch.Tensor]) -> None:
    """Pairwise cosine similarities between task z's.  cos near 1 => same direction."""
    task_ids = sorted(z_per_task.keys())
    zs = torch.cat([z_per_task[t] for t in task_ids], dim=0)            # (T, d_psi)
    zs_n = zs / zs.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    cos = (zs_n @ zs_n.T).cpu().numpy()
    print("        [z-geom] pairwise cosine similarity between task z's:")
    header = "                " + " ".join(f"  t{t}  " for t in task_ids)
    print(header)
    for i, ti in enumerate(task_ids):
        row = "  ".join(f"{cos[i, j]:+.3f}" for j in range(len(task_ids)))
        print(f"          t{ti}    {row}")
    # Average off-diagonal cosine -- high means z's are crowding into one direction
    off = cos - np.eye(len(task_ids))
    mean_off = off.sum() / (len(task_ids) * (len(task_ids) - 1))
    print(f"        [z-geom] mean off-diagonal cos = {mean_off:+.3f}   "
          f"(low/negative = z's spread out; near 1 = collapsed)")


@torch.no_grad()
def _print_predictor_z_sensitivity(
    agent: TDJEPA,
    next_obs_pool: torch.Tensor,
    z_per_task: Dict[int, torch.Tensor],
    n_states: int = 64,
    n_random_zs: int = 4,
) -> None:
    """Q = T_phi(phi(s), z, a).z across different z's on a fixed batch.

    Spread across z's measures whether the predictor uses z at all.  If
    ``spread / |Q|`` is tiny (<5%), the predictor collapsed -- Q is essentially
    z-independent and the actor cannot extract distinct policies per task.
    """
    idx = torch.randperm(next_obs_pool.shape[0], device=agent.device)[:n_states]
    s = next_obs_pool[idx]
    phi_s = agent.phi(s)

    # Build set of z's: each task's z plus a few uniform-sphere samples
    zs = list(z_per_task.values())
    z_rand = agent._sample_z(n_random_zs, psi_next=None)
    for k in range(n_random_zs):
        zs.append(z_rand[k:k + 1])

    # Use the actor's own action choice so we're measuring "Q at the policy"
    qs = []
    for z in zs:
        z_b = z.expand(n_states, -1)
        a = agent.act(s, z_b, deterministic=True)
        # T_phi returns (K, B, d_psi); average over ensemble heads
        pred = agent.T_phi(phi_s, z_b, a).mean(dim=0)
        q = (pred * z_b).sum(-1).mean()
        qs.append(q.item())

    qs_np = np.asarray(qs)
    spread = qs_np.max() - qs_np.min()
    mean_abs = np.abs(qs_np).mean()
    print(f"        [z-sens] Q across {len(qs)} z's ({len(z_per_task)} task + {n_random_zs} rand):")
    print(f"                 values = " + "  ".join(f"{q:+.2f}" for q in qs_np))
    print(f"                 spread = {spread:.2f}   mean|Q| = {mean_abs:.2f}   "
          f"spread/|Q| = {spread / max(mean_abs, 1e-6):.3f}   "
          f"({'COLLAPSED' if spread / max(mean_abs, 1e-6) < 0.05 else 'ok'})")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--env_name", default="antmaze-medium-navigate-v0",
                   help="OGBench env id.  See sweep_antmaze in the reference repo.")
    p.add_argument("--total_steps", type=int, default=1_000_000)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--phi_ortho_coef", type=float, default=1.0)
    p.add_argument("--psi_ortho_coef", type=float, default=1.0)
    p.add_argument("--bc_coeff", type=float, default=0.3,
                   help="0.3 for antmaze (reference sweep_antmaze), 3.0 for cube/scene.")
    p.add_argument("--phi_dim", type=int, default=50)
    p.add_argument("--psi_dim", type=int, default=50)
    p.add_argument("--enc_hidden_dim", type=int, default=512)
    p.add_argument("--enc_n_hidden", type=int, default=4)
    p.add_argument("--pred_hidden_dim", type=int, default=512)
    p.add_argument("--pred_n_hidden", type=int, default=2)
    p.add_argument("--actor_hidden_dim", type=int, default=512)
    p.add_argument("--actor_n_hidden", type=int, default=2)
    p.add_argument("--actor_use_full_encoder", action="store_true",
                   help="If set, actor takes phi(s) instead of raw obs. Default False (matches OGBench).")
    p.add_argument("--num_parallel", type=int, default=2)
    p.add_argument("--scale_train_goals", action="store_true", default=True)
    p.add_argument("--no_scale_train_goals", dest="scale_train_goals", action="store_false")
    p.add_argument("--eval_every", type=int, default=25_000)
    p.add_argument("--eval_episodes", type=int, default=10)
    p.add_argument("--center_reward", action="store_true",
                   help="Mean-center the inference reward before lstsq.  Ablation: "
                        "removes the constant component that gets absorbed by "
                        "mean(psi) and collapses task z's onto cos~1.")
    p.add_argument("--log_every", type=int, default=1_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save_path", default=None)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    morph = _detect_morph(args.env_name)
    print(f"[setup] env={args.env_name}  morph={morph}  device={args.device}  seed={args.seed}")
    env, train_ds, _ = ogbench.make_env_and_datasets(args.env_name)

    obs_dim = int(env.observation_space.shape[0])
    action_dim = int(env.action_space.shape[0])
    print(f"[setup] obs_dim={obs_dim}  action_dim={action_dim}  goal_tol={GOAL_TOL[morph]}")

    obs, act, nxt, term = build_dataset_tensors(train_ds, args.device)
    print(f"[setup] dataset: {obs.size(0):,} valid transitions")

    agent = TDJEPA(
        obs_dim=obs_dim,
        action_dim=action_dim,
        phi_dim=args.phi_dim,
        psi_dim=args.psi_dim,
        enc_hidden_dim=args.enc_hidden_dim,
        enc_n_hidden=args.enc_n_hidden,
        pred_hidden_dim=args.pred_hidden_dim,
        pred_n_hidden=args.pred_n_hidden,
        actor_hidden_dim=args.actor_hidden_dim,
        actor_n_hidden=args.actor_n_hidden,
        actor_use_full_encoder=args.actor_use_full_encoder,
        num_parallel=args.num_parallel,
        gamma=args.gamma,
        lr=args.lr,
        tau=args.tau,
        phi_ortho_coef=args.phi_ortho_coef,
        psi_ortho_coef=args.psi_ortho_coef,
        scale_train_goals=args.scale_train_goals,
        bc_coeff=args.bc_coeff,
        device=args.device,
    )

    rng = torch.Generator(device=args.device)
    rng.manual_seed(args.seed)

    t0 = time.time()
    log_keys = ("td_loss", "phi_orth", "psi_orth", "phi_norm", "psi_norm",
                "actor_loss", "bc_err", "q_mean", "inverse_dynamics_loss")
    log_buf = {k: 0.0 for k in log_keys}

    for step in range(1, args.total_steps + 1):
        batch = sample_batch(obs, act, nxt, term, args.batch_size, args.gamma, rng)
        info = agent.update(batch)
        for k in log_keys:
            log_buf[k] += info[k]

        if step % args.log_every == 0:
            sps = step / (time.time() - t0)
            msg = "  ".join(f"{k}={log_buf[k] / args.log_every:+.3f}" for k in log_keys)
            print(f"[{step:>7d}/{args.total_steps}] ({sps:>4.0f} sps)  {msg}")
            log_buf = {k: 0.0 for k in log_keys}

        if step % args.eval_every == 0:
            sr = evaluate_zero_shot(
                agent, env, nxt, morph,
                num_episodes=args.eval_episodes,
                center_reward=args.center_reward,
            )
            mean_sr = float(np.mean(list(sr.values())))
            per = ", ".join(f"t{k}={v:.2f}" for k, v in sr.items())
            print(f"            >> zero-shot  mean={mean_sr:.3f}  [{per}]")

    print("\n[done] final eval (50 episodes / task) ...")
    sr = evaluate_zero_shot(
        agent, env, nxt, morph, num_episodes=50,
        center_reward=args.center_reward,
    )
    print(f"FINAL  mean_success={float(np.mean(list(sr.values()))):.3f}")
    for k, v in sr.items():
        print(f"  task {k}: {v:.3f}")

    if args.save_path:
        torch.save(
            {
                "phi": agent.phi.state_dict(),
                "psi": agent.psi.state_dict(),
                "T_phi": agent.T_phi.state_dict(),
                "T_psi": agent.T_psi.state_dict(),
                "actor": agent.actor.state_dict(),
                "z_cov": agent.register_z_cov,
                "args": vars(args),
            },
            args.save_path,
        )
        print(f"[done] saved to {args.save_path}")


if __name__ == "__main__":
    main()