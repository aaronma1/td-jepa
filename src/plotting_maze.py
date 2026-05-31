import os

import matplotlib.pyplot as plt
import numpy as np
import torch


def _draw_maze_walls(ax, env, color="0.6", alpha=0.85):
    """Shade wall cells of an OGBench locomaze env on `ax`.

    Uses env.unwrapped.maze_map (1 = wall, 0 = free) and the env's
    maze_unit / offset_x / offset_y so the wall rectangles line up
    exactly with the agent's world coordinates. No-op for non-maze envs.
    The color matches OGBench's mujoco renderer (rgba='.6 .6 .6 1').
    """
    try:
        u = env.unwrapped
        maze_map = u.maze_map
        unit = u._maze_unit
        off_x = u._offset_x
        off_y = u._offset_y
    except AttributeError:
        return

    from matplotlib.patches import Rectangle

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

    # Tight bounds on the maze so adjustable="box" keeps the cells square.
    ax.set_xlim(-off_x - unit / 2, (W - 1) * unit - off_x + unit / 2)
    ax.set_ylim(-off_y - unit / 2, (H - 1) * unit - off_y + unit / 2)


@torch.no_grad()
def plot_trajectories(
    agent,
    env,
    z_per_task,
    step: int,
    morph: str,
    save_dir: str = "trajectories",
    max_steps: int = 1000,
):
    """Render one rollout per task as an xy trajectory plot over the maze."""
    print(f"PLOTTING {step}")
    os.makedirs(save_dir, exist_ok=True)
    task_ids = sorted(z_per_task.keys())
    n = len(task_ids)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), squeeze=False)
    axes = axes[0]

    summary = {}
    for ax, task_id in zip(axes, task_ids):
        obs, info = env.reset(options=dict(task_id=task_id, render_goal=False))
        goal_xy = np.asarray(
            info.get("goal", getattr(env.unwrapped, "cur_goal_xy", np.zeros(2)))
        )[:2]
        z = z_per_task[task_id].unsqueeze(0)

        # Walls first so the trajectory overlays them.
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
        final_dist = float(np.linalg.norm(traj[-1] - goal_xy))

        ax.plot(traj[:, 0], traj[:, 1], lw=0.8, alpha=0.9, zorder=2)
        ax.scatter(
            traj[0, 0],
            traj[0, 1],
            c="g",
            s=60,
            marker="o",
            label="start",
            zorder=3,
        )
        ax.scatter(
            goal_xy[0],
            goal_xy[1],
            c="r",
            s=120,
            marker="*",
            label="goal",
            zorder=3,
        )
        ax.scatter(
            traj[-1, 0],
            traj[-1, 1],
            c="b",
            s=40,
            marker="x",
            label="end",
            zorder=3,
        )
        ax.set_title(
            f"t{task_id} "
            f"{'✓' if success else '✗'} "
            f"d={final_dist:.2f} "
            f"|a|sat={sat:.2f} "
            f"steps={len(actions)}",
            fontsize=9,
        )
        # adjustable="box" keeps the xlim/ylim from _draw_maze_walls intact;
        # "datalim" (the previous setting) would expand them to match aspect.
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.15, zorder=1)
        if task_id == task_ids[0]:
            ax.legend(fontsize=7, loc="best")

        summary[task_id] = {
            "success": success,
            "final_dist": final_dist,
            "sat_frac": sat,
            "n_steps": len(actions),
            "action_mean": actions.mean(0).tolist() if actions.size else [],
            "action_std": actions.std(0).tolist() if actions.size else [],
        }

    fig.suptitle(f"step {step:,}", fontsize=11)
    fig.tight_layout()
    path = os.path.join(save_dir, f"traj_step_{step:07d}.png")
    fig.savefig(path, dpi=80, bbox_inches="tight")
    plt.close(fig)

    print(f"        [traj] saved {path}")
    # action diversity across tasks at a single shared state
    obs0, _ = env.reset(options=dict(task_id=task_ids[0]))
    s = torch.as_tensor(obs0, dtype=torch.float32, device=agent.device).unsqueeze(0)
    task_actions = {}
    for tid in task_ids:
        a = agent.act(s, z_per_task[tid].unsqueeze(0)).cpu().numpy().squeeze(0)
        task_actions[tid] = a
    arr = np.stack([task_actions[t] for t in task_ids], axis=0)
    pairwise = np.linalg.norm(arr[:, None] - arr[None, :], axis=-1)
    triu = pairwise[np.triu_indices(len(task_ids), k=1)]
    print(
        f"        [traj] action diversity across tasks at fixed s: "
        f"mean_pairwise_dist={triu.mean():.3f} "
        f"max={triu.max():.3f} "
        f"(low → actor is z-independent)"
    )
    return summary
