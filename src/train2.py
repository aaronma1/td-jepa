from __future__ import annotations

import argparse
import math
import time
from typing import Dict, Tuple

import numpy as np
import ogbench
import torch
import tqdm

from td_jepa import TD_JEPA, TD_JEPA_CONFIG

GOAL_TOL = {"ant": 0.5, "point": 1.0, "humanoid": 0.5}


def _detect_morph(env_name: str) -> str:
    for m in GOAL_TOL:
        if env_name.startswith(m):
            return m
    return "ant"


def build_dataset_tensors(dataset, device):
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


def sample_batch(obs, act, nxt, term, batch_size, gamma, rng):
    idx = torch.randint(0, obs.size(0), (batch_size,), device=obs.device, generator=rng)
    # td_jepa expects states/actions/next_states/terminals as positional args
    return (
        obs[idx],
        act[idx],
        nxt[idx],
        term[idx].unsqueeze(-1),
    )


@torch.no_grad()
def infer_z_for_task(
    agent: TD_JEPA,
    pool: torch.Tensor,
    goal_xy: np.ndarray,
    tol: float,
    shift_reward: float = 0.0,
    center_reward: bool = False,
    verbose: bool = False,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Reward inference for a single task using OGBench's sparse goal reward."""
    goal_t = torch.as_tensor(goal_xy, dtype=torch.float32, device=agent.device)
    xy = pool[:, :2].to(agent.device)
    dist = (xy - goal_t).norm(dim=-1)
    rew = (dist <= tol).float() + shift_reward  # (N,)
    n_pos = int((rew > shift_reward).sum().item())
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
        tag = " centered" if center_reward else ""
        print(
            f"        [z-inf{tag}] goal=({goal_xy[0]:+.2f},{goal_xy[1]:+.2f})  "
            f"positives={n_pos}/{rew.numel()}  min_dist={diag['min_dist']:.3f}  "
            f"x=[{diag['xy_min_x']:+.2f},{diag['xy_max_x']:+.2f}]  "
            f"y=[{diag['xy_min_y']:+.2f},{diag['xy_max_y']:+.2f}]"
        )
    # agent.fit_reward takes (states, rewards), returns unit-norm z (psi_dim,)
    z = agent.fit_reward(pool, rew)
    return z, diag


# ---------- diagnostics ----------


@torch.no_grad()
def psi_collapse_diagnostics(agent, states, name="psi"):
    enc = agent.psi if name == "psi" else agent.phi
    z = enc(states.to(agent.device))
    B, d = z.shape
    mean = z.mean(0)
    zc = z - mean
    cov = (zc.T @ zc) / max(B - 1, 1)
    diag_ = cov.diagonal()
    offdiag_sq = cov.pow(2).sum() - diag_.pow(2).sum()
    diag_sq = diag_.pow(2).sum()
    eigvals = torch.linalg.eigvalsh(cov).clamp_min(0)
    total = eigvals.sum().clamp_min(1e-12)
    p = eigvals / total
    eff_rank = torch.exp(-(p * (p + 1e-12).log()).sum()).item()
    cond = (eigvals[-1] / eigvals[0].clamp_min(1e-12)).item()
    top1 = (eigvals[-1] / total).item()
    top5 = (eigvals[-min(5, d) :].sum() / total).item()
    return {
        f"{name}/mean_norm": mean.norm().item(),
        f"{name}/feature_norm": z.norm(dim=-1).mean().item(),
        f"{name}/offdiag_ratio": (offdiag_sq / diag_sq.clamp_min(1e-12)).sqrt().item(),
        f"{name}/effective_rank": eff_rank,
        f"{name}/condition_num": cond,
        f"{name}/top1_var_frac": top1,
        f"{name}/top5_var_frac": top5,
        f"{name}/dim": d,
    }, eigvals.cpu()


def print_collapse_report(agent, states, name="psi"):
    stats, eigvals = psi_collapse_diagnostics(agent, states, name)
    d = stats[f"{name}/dim"]
    print(f"\n=== {name} collapse diagnostics (B={states.shape[0]}, d={d}) ===")
    for k, v in stats.items():
        print(f"  {k:32s} {v:10.4f}")
    print(
        f"  eigvals (top→bottom): "
        f"{', '.join(f'{x:.3f}' for x in eigvals.flip(0)[: min(8, d)].tolist())}"
        f"{' ...' if d > 8 else ''}"
    )


@torch.no_grad()
def _print_z_geometry(z_per_task):
    task_ids = sorted(z_per_task.keys())
    zs = torch.stack([z_per_task[t] for t in task_ids], dim=0)
    zs_n = zs / zs.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    cos = (zs_n @ zs_n.T).cpu().numpy()
    print("        [z-geom] pairwise cosine between task z's:")
    for i, ti in enumerate(task_ids):
        row = "  ".join(f"{cos[i, j]:+.3f}" for j in range(len(task_ids)))
        print(f"          t{ti}    {row}")
    off = cos - np.eye(len(task_ids))
    mean_off = off.sum() / (len(task_ids) * (len(task_ids) - 1))
    print(f"        [z-geom] mean off-diagonal cos = {mean_off:+.3f}")


@torch.no_grad()
def _print_predictor_z_sensitivity(agent, pool, z_per_task, n_states=64, n_random_zs=4):
    """Q = T_phi(phi(s), a, z) . z across different z's on a fixed batch."""
    idx = torch.randperm(pool.shape[0], device=pool.device)[:n_states]
    s = pool[idx].to(agent.device)
    phi_s = agent.phi(s)

    # task zs + a few uniform-sphere zs (use the same convention as training)
    zs = list(z_per_task.values())
    z_rand = agent._sample_latent_z(n_random_zs)
    for k in range(n_random_zs):
        zs.append(z_rand[k])

    qs = []
    for z in zs:
        z_b = z.unsqueeze(0).expand(n_states, -1).to(agent.device)
        # use the policy's action for each (s, z)
        if agent.config.actor_use_raw_obs:
            a = agent.actor(s, z_b)
        else:
            a = agent.actor(phi_s, z_b)
        pred = agent.T_phi(phi_s, a, z_b)
        q = (pred * z_b).sum(-1).mean()
        qs.append(q.item())

    qs_np = np.asarray(qs)
    spread = qs_np.max() - qs_np.min()
    mean_abs = np.abs(qs_np).mean()
    print(
        f"        [z-sens] Q across {len(qs)} z's ({len(z_per_task)} task + {n_random_zs} rand):"
    )
    print(f"                 values = " + "  ".join(f"{q:+.2f}" for q in qs_np))
    print(
        f"                 spread = {spread:.3f}  mean|Q| = {mean_abs:.3f}  "
        f"spread/|Q| = {spread / max(mean_abs, 1e-6):.3f}  "
        f"({'COLLAPSED' if spread / max(mean_abs, 1e-6) < 0.05 else 'ok'})"
    )


@torch.no_grad()
def _signal_to_noise(agent, pool, actions_pool, next_pool, n=2048):
    """In-distribution vs shuffled-target bilinear ratio.

    Healthy: |T_phi(phi(s), a, z) . psi(s')|_real >> |.|_shuffled.
    Collapse: ratio near 1 -> psi(s') carries no predictable signal.
    """
    idx = torch.randperm(pool.shape[0], device=pool.device)[:n]
    s = pool[idx].to(agent.device)
    a = actions_pool[idx].to(agent.device)
    sp = next_pool[idx].to(agent.device)
    shuf = sp[torch.randperm(n, device=agent.device)]
    z = agent._sample_latent_z(n)

    phi_s = agent.phi(s)
    f = agent.T_phi(phi_s, a, z)  # (n, psi_dim)
    psi_sp = agent.psi(sp)
    psi_sh = agent.psi(shuf)

    ip_real = (f * psi_sp).sum(-1).abs().mean().item()
    ip_shuf = (f * psi_sh).sum(-1).abs().mean().item()
    print(
        f"        [S/N] real|<F,psi>|={ip_real:.3f}  shuffled={ip_shuf:.3f}  "
        f"ratio={ip_real / max(ip_shuf, 1e-6):.3f}  "
        f"(>3 ok, <1.5 collapsed)"
    )


# ---------- main eval ----------


@torch.no_grad()
def evaluate_zero_shot(
    agent: TD_JEPA,
    env,
    buffer_obs: torch.Tensor,
    buffer_act: torch.Tensor,
    buffer_next: torch.Tensor,
    morph: str,
    num_tasks: int = 5,
    num_episodes: int = 10,
    max_steps: int = 1000,
    num_inference_samples: int = 10_000,
    center_reward: bool = False,
    verbose: bool = True,
) -> Dict[int, float]:
    agent.eval_mode()
    tol = GOAL_TOL[morph]
    out = {}

    n = min(num_inference_samples, buffer_next.shape[0])
    pool_idx = torch.randperm(buffer_next.shape[0], device=buffer_next.device)[:n]
    pool = buffer_next[pool_idx]

    # --- representation health ---
    diag_states = buffer_obs[
        torch.randint(0, buffer_obs.shape[0], (2048,), device=buffer_obs.device)
    ]
    if verbose:
        print_collapse_report(agent, diag_states, name="phi")
        print_collapse_report(agent, diag_states, name="psi")
        _signal_to_noise(agent, buffer_obs, buffer_act, buffer_next, n=2048)

        psi_pool = agent.psi(pool.to(agent.device))
        mean_psi_norm = psi_pool.mean(0).norm().item()
        sqrt_d = math.sqrt(agent.config.psi_dim)
        print(
            f"        [psi-mean] ||mean psi(pool)||={mean_psi_norm:.3f}  "
            f"(sqrt(d)={sqrt_d:.2f}; collapse if mean~sqrt_d)"
        )

    z_per_task = {}
    diag_per_task = {}

    for task_id in range(1, num_tasks + 1):
        obs, info = env.reset(options=dict(task_id=task_id, render_goal=False))
        goal_xy = np.asarray(
            info.get("goal", getattr(env.unwrapped, "cur_goal_xy", np.zeros(2)))
        )[:2]

        z, diag = infer_z_for_task(
            agent,
            pool,
            goal_xy,
            tol,
            center_reward=center_reward,
            verbose=verbose,
        )
        z_per_task[task_id] = z
        diag_per_task[task_id] = diag

        ok = 0
        for _ in tqdm.tqdm(range(num_episodes), leave=False, desc=f"task {task_id}"):
            obs, info = env.reset(options=dict(task_id=task_id, render_goal=False))
            for _ in range(max_steps):
                o_t = torch.as_tensor(
                    obs, dtype=torch.float32, device=agent.device
                ).unsqueeze(0)
                z_b = z.unsqueeze(0)
                a = agent.act(o_t, z_b).cpu().numpy().squeeze(0)
                obs, _, term, trunc, info = env.step(a)
                if float(info.get("success", 0.0)) > 0.5:
                    ok += 1
                    break
                if term or trunc:
                    break
        out[task_id] = ok / num_episodes

    if verbose:
        _print_z_geometry(z_per_task)
        _print_predictor_z_sensitivity(agent, pool, z_per_task)

    agent.train_mode()
    return out


# ---------- training loop ----------


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env_name", default="antmaze-medium-navigate-v0")
    p.add_argument("--total_steps", type=int, default=1_000_000)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--ema_coiff", type=float, default=0.99)
    p.add_argument("--reg_lambda", type=float, default=1.0)
    p.add_argument("--phi_dim", type=int, default=64)
    p.add_argument("--psi_dim", type=int, default=64)
    p.add_argument("--latent_z_batch_size", type=int, default=1)
    p.add_argument("--actor_use_raw_obs", action="store_true")
    p.add_argument("--eval_every", type=int, default=25_000)
    p.add_argument("--eval_episodes", type=int, default=10)
    p.add_argument("--center_reward", action="store_true")
    p.add_argument("--log_every", type=int, default=1_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save_path", default=None)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    morph = _detect_morph(args.env_name)
    print(f"[setup] env={args.env_name} morph={morph} device={args.device}")
    env, train_ds, _ = ogbench.make_env_and_datasets(args.env_name)

    obs_dim = int(env.observation_space.shape[0])
    act_dim = int(env.action_space.shape[0])
    print(f"[setup] obs_dim={obs_dim} act_dim={act_dim} tol={GOAL_TOL[morph]}")

    obs, act, nxt, term = build_dataset_tensors(train_ds, args.device)
    print(f"[setup] dataset: {obs.size(0):,} transitions")

    cfg = TD_JEPA_CONFIG(
        obs_dim=obs_dim,
        act_dim=act_dim,
        phi_dim=args.phi_dim,
        psi_dim=args.psi_dim,
        phi_lr=args.lr,
        psi_lr=args.lr,
        T_phi_lr=args.lr,
        T_psi_lr=args.lr,
        actor_lr=args.lr,
        ema_coiff=args.ema_coiff,
        gamma=args.gamma,
        reg_lambda=args.reg_lambda,
        actor_use_raw_obs=args.actor_use_raw_obs,
        latent_z_batch_size=args.latent_z_batch_size,
        train_device=args.device,
        eval_device=args.device,  # eval on same device as train for speed
    )
    agent = TD_JEPA(cfg)

    rng = torch.Generator(device=args.device)
    rng.manual_seed(args.seed)

    t0 = time.time()
    log_keys = (
        "phi td error",
        "psi td error",
        "phi ortho loss",
        "psi ortho loss",
        "actor loss",
        "mean q actor",
        "mean q data",
    )
    log_buf = {k: 0.0 for k in log_keys}
    log_n = 0

    for step in range(1, args.total_steps + 1):
        batch = sample_batch(obs, act, nxt, term, args.batch_size, args.gamma, rng)
        info = agent.train_step(*batch, log=(step % args.log_every == 0))
        if info is not None:
            for k in log_keys:
                log_buf[k] += info.get(k, 0.0)
            log_n += 1

        if step % args.log_every == 0:
            sps = step / (time.time() - t0)
            msg = "  ".join(f"{k}={log_buf[k] / max(log_n, 1):+.3f}" for k in log_keys)
            print(f"[{step:>7d}/{args.total_steps}] ({sps:>4.0f} sps)  {msg}")
            log_buf = {k: 0.0 for k in log_keys}
            log_n = 0

        if step % args.eval_every == 0:
            sr = evaluate_zero_shot(
                agent,
                env,
                obs,
                act,
                nxt,
                morph,
                num_episodes=args.eval_episodes,
                center_reward=args.center_reward,
            )
            mean_sr = float(np.mean(list(sr.values())))
            per = ", ".join(f"t{k}={v:.2f}" for k, v in sr.items())
            print(f"            >> zero-shot mean={mean_sr:.3f}  [{per}]")

    print("\n[done] final eval (50 ep/task) ...")
    sr = evaluate_zero_shot(
        agent,
        env,
        obs,
        act,
        nxt,
        morph,
        num_episodes=50,
        center_reward=args.center_reward,
    )
    print(f"FINAL mean_success={float(np.mean(list(sr.values()))):.3f}")
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
                "args": vars(args),
            },
            args.save_path,
        )
        print(f"[done] saved to {args.save_path}")


if __name__ == "__main__":
    main()
