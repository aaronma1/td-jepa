"""Evaluation suite for TD-JEPA. Decoupled from training.

Public entry: `evaluate_zero_shot(agent, env, buffer_obs, buffer_act, buffer_next,
                                  morph, ...)`.

Assumes `agent` exposes:
    agent.phi(s)              # (B, obs) -> (B, phi_dim), already normalized
    agent.psi(s)              # (B, obs) -> (B, psi_dim), already normalized
    agent.fit_reward(s, r)    # (N, obs), (N,) or (N,1) -> (psi_dim,), projected
    agent.act(s, z)           # (B, obs), (B, psi_dim) -> (B, act_dim), in [-1, 1]
    agent.q_phi(s, a, z)      # (B, obs), (B, act), (B, psi_dim) -> (B, psi_dim)
    agent.q_psi(s, a, z)      # (B, obs), (B, act), (B, psi_dim) -> (B, phi_dim)
    agent.device              # str
    agent.config.psi_dim      # int
    agent.config.act_dim      # int
    agent.eval_mode() / agent.train_mode()
"""

from __future__ import annotations

import math
import os
from typing import Dict, Tuple

import matplotlib
import numpy as np
import torch
import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# OGBench success tolerances. ant_reward_fn hardcodes 0.5; pointmaze uses 1.0.
GOAL_TOL = {"ant": 0.5, "point": 1.0, "humanoid": 0.5}


def detect_morph(env_name: str) -> str:
    for m in GOAL_TOL:
        if env_name.startswith(m):
            return m
    return "ant"


# ---------- helpers ----------


def _dot_z(pred_vec: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """Q scalar from predictor vector. If q_phi already returns a scalar,
    replace with `return pred_vec.squeeze(-1)`."""
    return (pred_vec * z).sum(-1)


@torch.no_grad()
def _sample_z_uniform(agent, n: int) -> torch.Tensor:
    """Uniform on the sqrt(d)-sphere in psi-space."""
    z = torch.randn(n, agent.config.psi_dim, device=agent.device)
    return (
        math.sqrt(agent.config.psi_dim)
        * z
        / z.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    )


def _goal_xy_from_reset(env, task_id: int) -> Tuple[np.ndarray, np.ndarray]:
    obs, info = env.reset(options=dict(task_id=task_id, render_goal=False))
    goal_xy = np.asarray(
        info.get("goal", getattr(env.unwrapped, "cur_goal_xy", np.zeros(2)))
    )[:2]
    return obs, goal_xy


# ---------- z inference (FB style) ----------


def infer_z_for_task(
    agent,
    pool: torch.Tensor,
    goal_xy: np.ndarray,
    tol: float,
    shift_reward: float = 1.0,
    verbose: bool = False,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Match facebookresearch/td_jepa OGBench eval.

    relabel  = (||s'_xy - goal||_2 <= tol).float() - 1.0     # ant_reward_fn
    rewards  = relabel + shift_reward                        # shift=1.0 in launch
    z        = lstsq(psi(next_obs), rewards).solution.T
    z        = sqrt(d) * F.normalize(z, dim=-1)              # in fit_reward
    """
    goal_t = torch.as_tensor(goal_xy, dtype=torch.float32, device=agent.device)
    xy = pool[:, :2].to(agent.device)
    dist = (xy - goal_t).norm(dim=-1)

    relabel = (dist <= tol).float() - 1.0
    rew = relabel + shift_reward

    n_pos = int((dist <= tol).sum().item())
    diag = {
        "positives": n_pos,
        "tol": tol,
        "shift_reward": shift_reward,
        "min_dist": float(dist.min().item()),
        "rew_mean": float(rew.mean().item()),
    }
    if verbose:
        print(
            f"        [z-inf] goal=({goal_xy[0]:+.2f},{goal_xy[1]:+.2f})  "
            f"positives={n_pos}/{rew.numel()} (tol={tol:.2f})  "
            f"shift={shift_reward:+.1f}  "
            f"rew_mean={diag['rew_mean']:+.3f}  "
            f"min_dist={diag['min_dist']:.3f}"
        )

    z = agent.fit_reward(pool, rew)
    return z, diag


# ---------- representation diagnostics ----------


@torch.no_grad()
def collapse_stats(agent, states: torch.Tensor, name: str = "psi"):
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
    stats = {
        f"{name}/mean_norm": mean.norm().item(),
        f"{name}/feature_norm": z.norm(dim=-1).mean().item(),
        f"{name}/offdiag_ratio": (offdiag_sq / diag_sq.clamp_min(1e-12)).sqrt().item(),
        f"{name}/effective_rank": eff_rank,
        f"{name}/condition_num": cond,
        f"{name}/top1_var_frac": top1,
        f"{name}/top5_var_frac": top5,
        f"{name}/dim": d,
    }
    return stats, eigvals.cpu()


def print_collapse_report(agent, states: torch.Tensor, name: str = "psi"):
    stats, eigvals = collapse_stats(agent, states, name)
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
def signal_to_noise(
    agent,
    pool_obs: torch.Tensor,
    pool_act: torch.Tensor,
    pool_next: torch.Tensor,
    n: int = 2048,
):
    """In-distribution vs shuffled-target bilinear ratio.

    Healthy: |q_phi(s, a, z) . psi(s')|_real >> |.|_shuffled.
    """
    idx = torch.randperm(pool_obs.shape[0], device=pool_obs.device)[:n]
    s = pool_obs[idx].to(agent.device)
    a = pool_act[idx].to(agent.device)
    sp = pool_next[idx].to(agent.device)
    shuf = sp[torch.randperm(n, device=agent.device)]
    z = _sample_z_uniform(agent, n)

    f = agent.q_phi(s, a, z)
    psi_sp = agent.psi(sp)
    psi_sh = agent.psi(shuf)

    ip_real = (f * psi_sp).sum(-1).abs().mean().item()
    ip_shuf = (f * psi_sh).sum(-1).abs().mean().item()
    print(
        f"        [S/N] real|<F,psi>|={ip_real:.3f}  shuffled={ip_shuf:.3f}  "
        f"ratio={ip_real / max(ip_shuf, 1e-6):.3f}  "
        f"(>3 ok, <1.5 collapsed)"
    )


# ---------- z geometry ----------


@torch.no_grad()
def print_z_geometry(z_per_task: Dict[int, torch.Tensor]):
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
def print_predictor_z_sensitivity(
    agent,
    pool: torch.Tensor,
    z_per_task: Dict[int, torch.Tensor],
    n_states: int = 64,
    n_random_zs: int = 4,
):
    """Q = q_phi(s, π(s,z), z) . z across different z's on a fixed batch."""
    idx = torch.randperm(pool.shape[0], device=pool.device)[:n_states]
    s = pool[idx].to(agent.device)

    zs = list(z_per_task.values())
    z_rand = _sample_z_uniform(agent, n_random_zs)
    zs += [z_rand[k] for k in range(n_random_zs)]

    qs = []
    for z in zs:
        z_b = z.unsqueeze(0).expand(n_states, -1)
        a = agent.act(s, z_b)
        pred = agent.q_phi(s, a, z_b)
        qs.append(_dot_z(pred, z_b).mean().item())

    qs_np = np.asarray(qs)
    spread = qs_np.max() - qs_np.min()
    mean_abs = np.abs(qs_np).mean()
    print(
        f"        [z-sens] Q across {len(qs)} z's ({len(z_per_task)} task + {n_random_zs} rand):"
    )
    print("                 values = " + "  ".join(f"{q:+.2f}" for q in qs_np))
    print(
        f"                 spread = {spread:.3f}  mean|Q| = {mean_abs:.3f}  "
        f"spread/|Q| = {spread / max(mean_abs, 1e-6):.3f}  "
        f"({'COLLAPSED' if spread / max(mean_abs, 1e-6) < 0.05 else 'ok'})"
    )


# ---------- Q calibration: predicted vs actual ----------


@torch.no_grad()
def q_calibration(
    agent,
    env,
    z: torch.Tensor,
    goal_xy: np.ndarray,
    tol: float,
    task_id: int,
    max_steps: int = 1000,
    verbose: bool = True,
):
    """Roll out under π(·, z), compare Q_pred(s_t, z) vs Q_actual(s_t, z).

    Q_actual uses both the indicator reward (env-true) and <psi(s), z>
    (the reward z was actually fit to). Big indicator/z-reward gap means
    z-inference is the bottleneck. Big pred/actual gap means the critic is.
    """
    obs, _ = env.reset(options=dict(task_id=task_id, render_goal=False))
    q_preds, r_inds, r_zs = [], [], []
    z_b = z.unsqueeze(0).to(agent.device)

    for _ in range(max_steps):
        o_t = torch.as_tensor(obs, dtype=torch.float32, device=agent.device).unsqueeze(
            0
        )
        a = agent.act(o_t, z_b)
        pred = agent.q_phi(o_t, a, z_b)
        q_preds.append(_dot_z(pred, z_b).item())
        r_inds.append(float(np.linalg.norm(obs[:2] - goal_xy) <= tol))
        r_zs.append((agent.psi(o_t) * z_b).sum(-1).item())

        obs, _, term, trunc, info = env.step(a.cpu().numpy().squeeze(0))
        if float(info.get("success", 0.0)) > 0.5 or term or trunc:
            r_inds[-1] = max(r_inds[-1], float(info.get("success", 0.0)))
            break

    gamma = getattr(agent.config, "gamma", 0.99)

    def _disc_return_to_go(rewards):
        out = np.zeros(len(rewards))
        running = 0.0
        for t in range(len(rewards) - 1, -1, -1):
            running = rewards[t] + gamma * running
            out[t] = running
        return out

    q_preds = np.asarray(q_preds)
    q_act_ind = _disc_return_to_go(r_inds)
    q_act_zr = _disc_return_to_go(r_zs)

    def _corr(a, b):
        if a.std() > 1e-6 and b.std() > 1e-6:
            return float(np.corrcoef(a, b)[0, 1])
        return float("nan")

    out = {
        "q_pred_mean": float(q_preds.mean()),
        "q_actual_indicator_mean": float(q_act_ind.mean()),
        "q_actual_zreward_mean": float(q_act_zr.mean()),
        "q_pred_s0": float(q_preds[0]),
        "q_actual_indicator_s0": float(q_act_ind[0]),
        "q_actual_zreward_s0": float(q_act_zr[0]),
        "bias_vs_indicator": float(q_preds.mean() - q_act_ind.mean()),
        "bias_vs_zreward": float(q_preds.mean() - q_act_zr.mean()),
        "corr_vs_indicator": _corr(q_preds, q_act_ind),
        "corr_vs_zreward": _corr(q_preds, q_act_zr),
        "indicator_return": float(q_act_ind[0]) if len(q_act_ind) else 0.0,
        "success": int(max(r_inds) > 0.5) if r_inds else 0,
        "steps": len(q_preds),
    }
    if verbose:
        print(
            f"        [q-cal t{task_id}] "
            f"Q_pred[s0]={out['q_pred_s0']:+.2f}  "
            f"Q_act_ind[s0]={out['q_actual_indicator_s0']:+.2f}  "
            f"Q_act_z[s0]={out['q_actual_zreward_s0']:+.2f}  "
            f"bias_ind={out['bias_vs_indicator']:+.2f}  "
            f"corr_ind={out['corr_vs_indicator']:+.2f}  "
            f"corr_z={out['corr_vs_zreward']:+.2f}  "
            f"ret={out['indicator_return']:+.2f}  "
            f"succ={out['success']}  "
            f"T={out['steps']}"
        )
    return out


# ---------- rollout for success rate ----------


@torch.no_grad()
def rollout_task(
    agent,
    env,
    z: torch.Tensor,
    task_id: int,
    num_episodes: int,
    max_steps: int,
) -> float:
    ok = 0
    z_b = z.unsqueeze(0)
    for _ in tqdm.tqdm(range(num_episodes), leave=False, desc=f"task {task_id}"):
        obs, _ = env.reset(options=dict(task_id=task_id, render_goal=False))
        for _ in range(max_steps):
            o_t = torch.as_tensor(
                obs, dtype=torch.float32, device=agent.device
            ).unsqueeze(0)
            a = agent.act(o_t, z_b).cpu().numpy().squeeze(0)
            obs, _, term, trunc, info = env.step(a)
            if float(info.get("success", 0.0)) > 0.5:
                ok += 1
                break
            if term or trunc:
                break
    return ok / num_episodes


# ---------- trajectory plotting (with walls) ----------


def _draw_maze_walls(ax, env, color: str = "0.6", alpha: float = 0.85):
    """Shade wall cells of an OGBench locomaze env. No-op for non-maze envs."""
    try:
        u = env.unwrapped
        maze_map = u.maze_map
        unit = u._maze_unit
        off_x = u._offset_x
        off_y = u._offset_y
    except AttributeError:
        return

    H, W = maze_map.shape
    for i in range(H):
        for j in range(W):
            if maze_map[i, j] != 1:
                continue
            cx = j * unit - off_x
            cy = i * unit - off_y
            ax.add_patch(
                Rectangle(
                    (cx - unit / 2, cy - unit / 2),
                    unit,
                    unit,
                    facecolor=color,
                    edgecolor="none",
                    alpha=alpha,
                    zorder=0,
                )
            )
    ax.set_xlim(-off_x - unit / 2, (W - 1) * unit - off_x + unit / 2)
    ax.set_ylim(-off_y - unit / 2, (H - 1) * unit - off_y + unit / 2)


@torch.no_grad()
def plot_trajectories(
    agent,
    env,
    z_per_task: Dict[int, torch.Tensor],
    step: int,
    save_dir: str = "trajectories",
    max_steps: int = 1000,
):
    os.makedirs(save_dir, exist_ok=True)
    task_ids = sorted(z_per_task.keys())
    n = len(task_ids)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), squeeze=False)
    axes = axes[0]

    summary = {}
    for ax, task_id in zip(axes, task_ids):
        obs, goal_xy = _goal_xy_from_reset(env, task_id)
        z = z_per_task[task_id].unsqueeze(0)
        _draw_maze_walls(ax, env)

        traj = [obs[:2].copy()]
        actions = []
        success = False
        for _ in range(max_steps):
            o_t = torch.as_tensor(
                obs, dtype=torch.float32, device=agent.device
            ).unsqueeze(0)
            a = agent.act(o_t, z).cpu().numpy().squeeze(0)
            actions.append(a.copy())
            obs, _, term, trunc, info = env.step(a)
            traj.append(obs[:2].copy())
            if float(info.get("success", 0.0)) > 0.5:
                success = True
                break
            if term or trunc:
                break

        traj = np.array(traj)
        actions = np.array(actions)
        sat = float((np.abs(actions) > 0.95).mean()) if actions.size else 0.0
        amag = float(np.linalg.norm(actions, axis=-1).mean()) if actions.size else 0.0
        final_dist = float(np.linalg.norm(traj[-1] - goal_xy))

        ax.plot(traj[:, 0], traj[:, 1], lw=0.8, alpha=0.9, zorder=2)
        ax.scatter(
            traj[0, 0], traj[0, 1], c="g", s=60, marker="o", label="start", zorder=3
        )
        ax.scatter(
            goal_xy[0], goal_xy[1], c="r", s=120, marker="*", label="goal", zorder=3
        )
        ax.scatter(
            traj[-1, 0], traj[-1, 1], c="b", s=40, marker="x", label="end", zorder=3
        )
        ax.set_title(
            f"t{task_id} {'✓' if success else '✗'} "
            f"d={final_dist:.2f} |a|={amag:.2f} sat={sat:.2f} T={len(actions)}",
            fontsize=9,
        )
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.15, zorder=1)
        if task_id == task_ids[0]:
            ax.legend(fontsize=7, loc="best")

        summary[task_id] = {
            "success": success,
            "final_dist": final_dist,
            "sat_frac": sat,
            "action_mag": amag,
            "n_steps": len(actions),
        }

    fig.suptitle(f"step {step:,}", fontsize=11)
    fig.tight_layout()
    path = os.path.join(save_dir, f"traj_step_{step:07d}.png")
    fig.savefig(path, dpi=80, bbox_inches="tight")
    plt.close(fig)
    print(f"        [traj] saved {path}")

    # Action diversity at a shared state.
    obs0, _ = env.reset(options=dict(task_id=task_ids[0]))
    s = torch.as_tensor(obs0, dtype=torch.float32, device=agent.device).unsqueeze(0)
    acts = np.stack(
        [
            agent.act(s, z_per_task[t].unsqueeze(0)).cpu().numpy().squeeze(0)
            for t in task_ids
        ]
    )
    pw = np.linalg.norm(acts[:, None] - acts[None, :], axis=-1)
    triu = pw[np.triu_indices(len(task_ids), k=1)]
    print(
        f"        [traj] action diversity across tasks at fixed s: "
        f"mean_pairwise_dist={triu.mean():.3f} max={triu.max():.3f}"
    )
    return summary


# ---------- orchestrator ----------


@torch.no_grad()
def evaluate_zero_shot(
    agent,
    env,
    buffer_obs: torch.Tensor,
    buffer_act: torch.Tensor,
    buffer_next: torch.Tensor,
    morph: str,
    num_tasks: int = 5,
    num_episodes: int = 10,
    max_steps: int = 1000,
    num_inference_samples: int = 10_000,
    shift_reward: float = 1.0,
    verbose: bool = True,
    step: int = 0,
    save_dir: str = "trajectories",
    run_calibration: bool = True,
) -> Dict[int, float]:
    agent.eval_mode()
    tol = GOAL_TOL[morph]

    n = min(num_inference_samples, buffer_next.shape[0])
    pool_idx = torch.randperm(buffer_next.shape[0], device=buffer_next.device)[:n]
    pool = buffer_next[pool_idx]

    if verbose:
        diag_idx = torch.randint(
            0, buffer_obs.shape[0], (2048,), device=buffer_obs.device
        )
        diag_states = buffer_obs[diag_idx]
        print_collapse_report(agent, diag_states, name="phi")
        print_collapse_report(agent, diag_states, name="psi")
        signal_to_noise(agent, buffer_obs, buffer_act, buffer_next, n=2048)

        psi_pool = agent.psi(pool.to(agent.device))
        mean_psi_norm = psi_pool.mean(0).norm().item()
        sqrt_d = math.sqrt(agent.config.psi_dim)
        print(
            f"        [psi-mean] ||mean psi(pool)||={mean_psi_norm:.3f}  "
            f"(sqrt(d)={sqrt_d:.2f}; collapse if mean~sqrt_d)"
        )

    z_per_task: Dict[int, torch.Tensor] = {}
    goal_per_task: Dict[int, np.ndarray] = {}
    out: Dict[int, float] = {}

    for task_id in range(1, num_tasks + 1):
        _, goal_xy = _goal_xy_from_reset(env, task_id)
        z, _ = infer_z_for_task(
            agent, pool, goal_xy, tol, shift_reward=shift_reward, verbose=verbose
        )
        z_per_task[task_id] = z
        goal_per_task[task_id] = goal_xy

        out[task_id] = rollout_task(agent, env, z, task_id, num_episodes, max_steps)

    if verbose:
        print_z_geometry(z_per_task)
        print_predictor_z_sensitivity(agent, pool, z_per_task)
        if run_calibration:
            for task_id in sorted(z_per_task.keys()):
                q_calibration(
                    agent,
                    env,
                    z_per_task[task_id],
                    goal_per_task[task_id],
                    tol,
                    task_id=task_id,
                    max_steps=max_steps,
                )
        plot_trajectories(agent, env, z_per_task, step=step, save_dir=save_dir)

    agent.train_mode()
    return out
