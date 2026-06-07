"""Evaluation suite for TD-JEPA. Decoupled from training.

Public entry: `evaluate_zero_shot(agent, env, buffer_obs, buffer_act, buffer_next,
                                  morph, ...)`.

Assumes `agent` exposes:
    agent.phi(s)              # (B, obs) -> (B, phi_dim), already normalized
    agent.psi(s)              # (B, obs) -> (B, psi_dim), already normalized
    agent.fit_reward(s, r)    # (N, obs), (N,) or (N,1) -> (psi_dim,), projected
    agent.act(s, z)           # (B, obs), (B, psi_dim) -> (B, act_dim), in [-1, 1]
    agent.T_phi.predict(s, a, z)      # (B, phi_dim), (B, act), (B, psi_dim) -> (B, psi_dim)
    agent.T_psi.predict(s, a, z)      # (B, psi_dim), (B, act), (B, psi_dim) -> (B, phi_dim)
    agent.device              # str
    agent.config.psi_dim      # int
    agent.config.act_dim      # int
    agent.eval_mode() / agent.train_mode()
"""

from __future__ import annotations

import math
import os
import random
from typing import Dict, Tuple

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
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
    """Q scalar from predictor vector. If T_phi already returns a scalar,
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
    tol: float,  # SUCCESS tol (per-env): only used if reward_tol is None
    shift_reward: float = 0.0,  # reference default is 0.0, NOT 1.0
    oracle_z: bool = False,  # if True, bypass lstsq: z = project(psi(goal))
    verbose: bool = False,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Faithful to facebookresearch/td_jepa OGBench eval (ant_reward_fn):

        reward = (||s'_xy - goal||_2 <= 0.5).float() - 1.0    # values in {-1, 0}
        reward = reward + shift_reward                        # shift defaults to 0.0
        z      = lstsq(psi(next_obs), reward).solution.T
        z      = sqrt(d) * F.normalize(z)                     # project_z, in fit_reward

    NOTE: the reference reward is the SPARSE indicator minus 1 (mostly -1, a few 0s);
    it is NOT dense. The {-1,0} convention differs from a {0,1} convention by a
    constant offset, which lstsq absorbs into the psi-mean direction -- so the
    convention is not cosmetic. We match the reference exactly.
    """
    goal_t = torch.as_tensor(goal_xy, dtype=torch.float32, device=agent.device)

    if oracle_z:
        # Goal-reaching z*: up to the cov correction, z* ∝ psi(goal).
        # Pointmaze obs == xy, so psi takes the 2-d goal directly; for higher-dim
        # obs this still works if psi was trained to embed states whose xy == goal.
        g = goal_t.view(1, -1)
        if g.shape[-1] < agent.phi(pool[:1].to(agent.device)).shape[-1]:
            pass  # caller is responsible for passing a full-obs goal if obs != xy
        z = agent._project_z(agent.psi(g)).squeeze(0)
        diag = {"oracle_z": True, "positives": -1, "reward_fit_cos": float("nan")}
        if verbose:
            print(
                f"        [z-inf] goal=({goal_xy[0]:+.2f},{goal_xy[1]:+.2f})  ORACLE z=psi(goal)"
            )
        return z, diag

    xy = pool[:, :2].to(agent.device)
    dist = (xy - goal_t).norm(dim=-1)

    # Reference convention: indicator - 1.0  ->  {-1, 0}
    relabel = (dist <= tol).float() - 1.0
    rew = relabel + shift_reward

    n_pos = int((dist <= tol).sum().item())
    z = agent.fit_reward(pool, rew)

    # Reward-fit quality, computed on the SAME (shifted) target we fit.
    psi_pool = agent.psi(pool.to(agent.device))
    r_hat = psi_pool @ z
    rew_t = rew.to(agent.device).view(1, -1)
    cos = F.cosine_similarity(r_hat.view(1, -1), rew_t).item()

    diag = {
        "positives": n_pos,
        "reward_tol": tol,
        "shift_reward": shift_reward,
        "min_dist": float(dist.min().item()),
        "rew_mean": float(rew.mean().item()),
        "reward_fit_cos": cos,
    }
    if verbose:
        print(
            f"        [z-inf] goal=({goal_xy[0]:+.2f},{goal_xy[1]:+.2f})  "
            f"positives={n_pos}/{rew.numel()} (rtol={tol:.2f})  "
            f"shift={shift_reward:+.1f}  rew_mean={diag['rew_mean']:+.3f}  "
            f"min_dist={diag['min_dist']:.3f}"
        )
        flag = "z-inference BOTTLENECK" if cos < 0.5 else "ok"
        print(f"        [z-inf]   reward_fit_cos={cos:+.3f}  ({flag})")

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

    Healthy: |<F(s,a,z), psi(s')>|_real >> |.|_shuffled. A low ratio means the
    control-relevant variation in F is buried under a state-independent (DC)
    component, i.e. the signal/noise problem cumulant-centering targets.
    """
    idx = torch.randperm(pool_obs.shape[0], device=pool_obs.device)[:n]
    s = pool_obs[idx].to(agent.device)
    a = pool_act[idx].to(agent.device)
    sp = pool_next[idx].to(agent.device)
    shuf = sp[torch.randperm(s.shape[0], device=agent.device)]
    z = _sample_z_uniform(agent, s.shape[0])

    f = agent.T_phi.predict(agent.phi(s), a, z)
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
    """Q = T_phi(s, π(s,z), z) . z across different z's on a fixed batch."""
    idx = torch.randperm(pool.shape[0], device=pool.device)[:n_states]
    s = pool[idx].to(agent.device)

    zs = list(z_per_task.values())
    z_rand = _sample_z_uniform(agent, n_random_zs)
    zs += [z_rand[k] for k in range(n_random_zs)]

    qs = []
    for z in zs:
        z_b = z.unsqueeze(0).expand(n_states, -1)
        a = agent.act(s, z_b)
        pred = agent.T_phi.predict(agent.phi(s), a, z_b)
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
    task_id: int,
    max_steps: int = 1000,
    verbose: bool = True,
):
    """Roll out under π(·, z), compare Q_pred(s_t, z) vs Q_actual(s_t, z).

    The cumulant is psi(s'), so T_phi predicts the discounted sum of future
    cumulants starting at s_{t+1}:  Q_pred(s_t) ~= sum_{k>=1} g^{k-1} r(s_{t+k}).
    The return-to-go includes r(s_t), so bias/corr are computed on the aligned
    pair (q_preds[:-1] vs return_to_go[1:]). s0 prints remain the raw values.

    Q_actual uses both the indicator reward (env-true) and <psi(s), z>
    (the reward z was actually fit to). Big indicator/z-reward gap means
    z-inference is the bottleneck. Big pred/actual gap means the critic is.
    """
    obs, _ = env.reset(options=dict(task_id=task_id, render_goal=False))
    q_preds, r_zs = [], []
    z_b = z.unsqueeze(0).to(agent.device)

    for _ in range(max_steps):
        o_t = torch.as_tensor(obs, dtype=torch.float32, device=agent.device).unsqueeze(
            0
        )
        a = agent.act(o_t, z_b)
        pred = agent.T_phi.predict(agent.phi(o_t), a, z_b)
        q_preds.append(_dot_z(pred, z_b).item())
        r_zs.append((agent.psi(o_t) * z_b).sum(-1).item())

        obs, _, _, _, _ = env.step(a.cpu().numpy().squeeze(0))
    gamma = getattr(agent.config, "gamma", 0.99)

    def _disc_return_to_go(rewards):
        out = np.zeros(len(rewards))
        running = 0.0
        for t in range(len(rewards) - 1, -1, -1):
            running = rewards[t] + gamma * running
            out[t] = running
        return out

    q_preds = np.asarray(q_preds)
    q_act_zr = _disc_return_to_go(r_zs)
    # Align: Q_pred(s_t) predicts the return from s_{t+1} onward.
    if len(q_preds) > 1:
        q_p_al = q_preds[:-1]
        zr_al = q_act_zr[1:]
    else:
        q_p_al, zr_al = q_preds, q_act_zr

    def _corr(a, b):
        if a.std() > 1e-6 and b.std() > 1e-6:
            return float(np.corrcoef(a, b)[0, 1])
        return float("nan")

    out = {
        "q_pred_mean": float(q_preds.mean()),
        "q_actual_zreward_mean": float(q_act_zr.mean()),
        "q_pred_s0": float(q_preds[0]),
        "q_actual_zreward_s0": float(q_act_zr[0]),
        "bias_vs_zreward": float(q_p_al.mean() - zr_al.mean()),
        "corr_vs_zreward": _corr(q_p_al, zr_al),
        "steps": len(q_preds),
    }
    if verbose:
        # corr_ind is nan until success > 0 (indicator return has no variance).
        print(
            f"        [q-cal t{task_id}] "
            f"Q_pred[s0]={out['q_pred_s0']:+.2f}  "
            f"Q_act_z[s0]={out['q_actual_zreward_s0']:+.2f}  "
            f"corr_z={out['corr_vs_zreward']:+.2f}  "
            f"bias_z={out['bias_vs_zreward']:+.2f}  "
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
    pool: torch.Tensor = None,
    n_bg: int = 4000,
    field: str = "both",  # "reward" | "value" | "both" | "none"
    value_action_mode: str = "policy",
    value_n_actions: int = 16,
):
    os.makedirs(save_dir, exist_ok=True)
    task_ids = sorted(z_per_task.keys())
    n = len(task_ids)

    rows = []
    if pool is not None and field != "none":
        if field in ("reward", "both"):
            rows.append("reward")
        if field in ("value", "both"):
            rows.append("value")
    if not rows:
        rows = ["traj"]  # plain trajectories, no background field
    nrow = len(rows)

    bg_xy = bg = None
    if pool is not None and rows != ["traj"]:
        bg = pool
        if bg.shape[0] > n_bg:
            bg = bg[torch.randperm(bg.shape[0], device=bg.device)[:n_bg]]
        bg_xy = bg[:, :2].cpu().numpy()
        bg_psi = agent.psi(bg.to(agent.device)) if "reward" in rows else None

    fig, axes = plt.subplots(nrow, n, figsize=(4 * n, 4 * nrow), squeeze=False)

    summary = {}
    # roll out once per task; reuse the trajectory across all rows
    for col, task_id in enumerate(task_ids):
        obs, goal_xy = _goal_xy_from_reset(env, task_id)
        z_vec = z_per_task[task_id]
        z = z_vec.unsqueeze(0)

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

        for r, kind in enumerate(rows):
            ax = axes[r, col]
            _draw_maze_walls(ax, env)

            pct = None
            if kind == "reward":
                pct = _draw_field(
                    ax, fig, bg_xy, (bg_psi * z_vec).sum(-1).cpu().numpy(), goal_xy
                )
            elif kind == "value":
                vals = _value_field(
                    agent,
                    bg,
                    z_vec,
                    n_actions=value_n_actions,
                    action_mode=value_action_mode,
                )
                pct = _draw_field(ax, fig, bg_xy, vals, goal_xy)

            ax.plot(traj[:, 0], traj[:, 1], lw=0.8, alpha=0.9, color="w", zorder=2)
            ax.scatter(
                traj[0, 0],
                traj[0, 1],
                c="lime",
                s=60,
                marker="o",
                label="start",
                zorder=3,
            )
            ax.scatter(
                goal_xy[0], goal_xy[1], c="r", s=120, marker="*", label="goal", zorder=3
            )
            ax.scatter(
                traj[-1, 0],
                traj[-1, 1],
                c="black",
                s=40,
                marker="x",
                label="end",
                zorder=3,
            )

            tag = {"reward": "r̂=⟨ψ,z⟩", "value": "E_a Q", "traj": ""}[kind]
            gp = f" gP={pct:.2f}" if pct is not None else ""
            ax.set_title(
                f"t{task_id} {'✓' if success else '✗'} {tag}{gp}  "
                f"d={final_dist:.2f} |a|={amag:.2f} sat={sat:.2f} T={len(actions)}",
                fontsize=8,
            )
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, alpha=0.15, zorder=1)
            if col == 0 and r == 0:
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


# ---------- principal-component color plots ----------


@torch.no_grad()
def _pca_scores(feats: torch.Tensor, k: int):
    """Top-k PCA of feats via SVD of the centered matrix.

    Returns (scores [B,k], var_frac [k], loadings [k,d]). scores[:,i] == Xc @ V_i,
    so each column is mean-zero -> a zero-centered diverging colormap is correct.
    A max-abs-loading-positive sign convention (sklearn's svd_flip) makes the sign
    deterministic, so colors stay consistent across checkpoints / across phi vs psi.
    """
    feats = feats.float()
    mean = feats.mean(0, keepdim=True)
    Xc = feats - mean
    U, S, Vh = torch.linalg.svd(Xc, full_matrices=False)
    k = min(k, S.shape[0])
    U, S, Vh = U[:, :k], S[:k], Vh[:k]
    sign = torch.sign(Vh[torch.arange(k, device=Vh.device), Vh.abs().argmax(dim=1)])
    sign[sign == 0] = 1.0
    Vh = Vh * sign.unsqueeze(1)
    U = U * sign.unsqueeze(0)
    scores = U * S.unsqueeze(0)  # == Xc @ Vh.T
    var_frac = S.pow(2) / S.pow(2).sum().clamp_min(1e-12)
    return scores, var_frac, Vh


@torch.no_grad()
def plot_principal_components(
    agent,
    states: torch.Tensor,
    step: int,
    name: str = "psi",
    n_components: int = 6,
    save_dir: str = "trajectories",
    env=None,
    max_points: int = 20_000,
    point_size: float = 3.0,
    robust_clip: float = 0.99,
):
    """Scatter buffer states in xy-space, colored by their projection onto each of
    the top-`n_components` PCs of {name}(s).

    For locomaze envs obs[:2] is xy, so this shows the *spatial* structure of the
    representation: a healthy SR/Laplacian-style encoder gives low-frequency modes
    over the reachable set in the top PCs and finer detail lower down. Red flags are
    one PC carrying nearly all the variance (cf. top1_var_frac), or sharp jumps that
    don't respect maze geometry. For ant the color is the PC of the *full*-state
    embedding drawn at the agent's xy, so pose/velocity variation shows up as color
    spread among nearby points -- expected, not a bug.
    """
    os.makedirs(save_dir, exist_ok=True)
    enc = agent.psi if name == "psi" else agent.phi

    B = states.shape[0]
    if B > max_points:
        states = states[torch.randperm(B, device=states.device)[:max_points]]
    xy = states[:, :2].cpu().numpy()

    feats = enc(states.to(agent.device))
    k = min(n_components, feats.shape[1])
    scores, var_frac, _ = _pca_scores(feats, k)
    scores = scores.cpu().numpy()
    var_frac = var_frac.cpu().numpy()

    ncol = min(k, 3)
    nrow = math.ceil(k / ncol)
    fig, axes = plt.subplots(
        nrow, ncol, figsize=(4.2 * ncol, 4.0 * nrow), squeeze=False
    )
    axes_flat = axes.ravel()

    for c in range(k):
        ax = axes_flat[c]
        if env is not None:
            _draw_maze_walls(ax, env)
        v = scores[:, c]
        lim = float(np.quantile(np.abs(v), robust_clip))  # symmetric, outlier-robust
        lim = lim if lim > 0 else 1.0
        sc = ax.scatter(
            xy[:, 0],
            xy[:, 1],
            c=v,
            cmap="RdBu_r",
            vmin=-lim,
            vmax=lim,
            s=point_size,
            alpha=0.6,
            linewidths=0,
            zorder=2,
        )
        ax.set_title(f"PC{c + 1}  ({100 * var_frac[c]:.1f}% var)", fontsize=9)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.12, zorder=1)
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)

    for c in range(k, len(axes_flat)):
        axes_flat[c].axis("off")

    cum = 100 * var_frac[:k].sum()
    fig.suptitle(
        f"{name}(s) principal components in xy-space — step {step:,}  "
        f"(top {k} = {cum:.1f}% var)",
        fontsize=11,
    )
    fig.tight_layout()
    path = os.path.join(save_dir, f"pca_{name}_step_{step:07d}.png")
    fig.savefig(path, dpi=90, bbox_inches="tight")
    plt.close(fig)
    print(
        f"        [pca-{name}] saved {path}  top{k} var = "
        + ", ".join(f"{100 * f:.1f}%" for f in var_frac[:k])
    )
    return path


@torch.no_grad()
def _value_field(
    agent,
    bg_states: torch.Tensor,
    z_vec: torch.Tensor,
    n_actions: int = 16,
    action_mode: str = "policy",  # "uniform": E_{a~U[-1,1]}; "policy": a=pi(s,z)
) -> np.ndarray:
    """V(s,z) = E_a <z, T_phi(phi(s), a, z)>  over background states.

    phi(s) is computed once; only the action varies. "uniform" averages over the
    action box, so it's actor-INDEPENDENT (a clean critic-only check); "policy"
    uses pi(s,z) for the on-policy value (critic+actor together). Either way this
    is the *future* value (T_phi predicts successor features from s' onward), so
    read it as a basin that should rise toward the goal, not as immediate reward.
    """
    s = bg_states.to(agent.device)
    phi_s = agent.phi(s)
    z_b = z_vec.unsqueeze(0).expand(s.shape[0], -1).to(agent.device)
    acc = torch.zeros(s.shape[0], device=agent.device)
    K = n_actions if action_mode == "uniform" else 1
    for _ in range(K):
        if action_mode == "uniform":
            a = torch.empty(
                s.shape[0], agent.config.act_dim, device=agent.device
            ).uniform_(-1.0, 1.0)
        else:
            a = agent.act(s, z_b)
        acc += (agent.T_phi.predict(phi_s, a, z_b) * z_b).sum(-1)
    return (acc / K).cpu().numpy()


def _draw_field(ax, fig, bg_xy, vals, goal_xy):
    """Scatter background colored by `vals`, mark the argmax, return goal percentile."""
    lo, hi = np.percentile(vals, [2, 98])
    sc = ax.scatter(
        bg_xy[:, 0],
        bg_xy[:, 1],
        c=vals,
        cmap="viridis",
        vmin=lo,
        vmax=hi,
        s=6,
        alpha=0.5,
        linewidths=0,
        zorder=1,
    )
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    peak = bg_xy[int(vals.argmax())]
    ax.scatter(
        peak[0],
        peak[1],
        facecolors="none",
        edgecolors="magenta",
        s=140,
        marker="D",
        linewidths=1.5,
        zorder=4,
        label="peak",
    )
    goal_i = int(np.linalg.norm(bg_xy - goal_xy, axis=-1).argmin())
    return float((vals <= vals[goal_i]).mean())


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
    num_inference_samples: int = 50_000,
    shift_reward: float = 1.0,
    oracle_z: bool = False,
    verbose: bool = True,
    step: int = 0,
    save_dir: str = "trajectories",
    run_calibration: bool = True,
    plot_pca: bool = True,
    pca_components: int = 6,
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
            agent,
            pool,
            goal_xy,
            tol,
            shift_reward=shift_reward,
            verbose=verbose,
            oracle_z=oracle_z,
        )
        z_per_task[task_id] = z
        goal_per_task[task_id] = goal_xy

        out[task_id] = rollout_task(agent, env, z, task_id, num_episodes, max_steps)

    if verbose:
        print_z_geometry(z_per_task)
        print_predictor_z_sensitivity(agent, pool, z_per_task)
        if run_calibration:
            for task_id in sorted(z_per_task.keys()):
                q_calibration(agent, env, z_per_task[task_id], task_id)

            latent_z = [
                agent._project_z(torch.randn(agent.config.psi_dim)) for _ in range(5)
            ]
            for z in latent_z:
                q_calibration(agent, env, z, random.randint(1, 5))

        plot_trajectories(
            agent, env, z_per_task, step=step, save_dir=save_dir, pool=pool
        )

        if plot_pca:
            for nm in ("phi", "psi"):
                plot_principal_components(
                    agent,
                    buffer_obs,
                    step=step,
                    name=nm,
                    n_components=pca_components,
                    save_dir=save_dir,
                    env=env,
                )

    agent.train_mode()
    return out
