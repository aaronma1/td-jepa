import argparse
import time

import numpy as np
import ogbench
import torch

from td_jepa import TD_JEPA, TD_JEPA_CONFIG


# Hyperparameters
def default_config(env_name):
    if env_name.startswith("pointmaze"):
        return TD_JEPA_CONFIG(
            obs_dim=2,
            act_dim=2,
            phi_dim=16,
            psi_dim=16,
            phi_hidden_layers=(64, 64, 64),
            psi_hidden_layers=(64, 64, 64),
            T_phi_hidden_layers=(64, 128, 128, 64),
            T_psi_hidden_layers=(64, 128, 128, 64),
            actor_hidden_layers=(64, 64, 64),
            # train_device="cpu",
        )


# Buffer and datasets
class Buffer:
    def __init__(self, s, a, s_prime, terminals, device="cuda"):
        self.s = s.to(device)
        self.a = a.to(device)
        self.s_prime = s_prime.to(device)
        self.terminals = terminals

    def sample_batch(self, batch_size):
        idx = torch.randint(0, self.s.shape[0], (batch_size,))
        return self.s[idx], self.a[idx], self.s_prime[idx], self.terminals[idx]


def get_dataset(env_name, device="cuda"):
    _, train_dataset, _ = ogbench.make_env_and_datasets(env_name)

    obs = train_dataset["observations"]
    act = train_dataset["actions"]
    term = train_dataset["terminals"]
    next_obs = np.concatenate([obs[1:], obs[-1:]], axis=0)
    valid = term == 0
    obs, act, next_obs = obs[valid], act[valid], next_obs[valid]
    term_valid = term[valid].astype(np.float32)
    return Buffer(
        torch.tensor(obs, dtype=torch.float32, device=device),
        torch.tensor(act, dtype=torch.float32, device=device),
        torch.tensor(next_obs, dtype=torch.float32, device=device),
        torch.tensor(term_valid, dtype=torch.float32, device=device).reshape(-1, 1),
        device=device,
    )


def train_env(args, config=None):

    if config is None:
        config = default_config(args.env_name)

    buffer = get_dataset(args.env_name, config.train_device)
    agent = TD_JEPA(config)  # pyright: ignore[reportArgumentType]
    t0 = time.time()

    log_buf = None
    n_info = 0

    for step in range(1, args.total_steps + 1):
        batch = buffer.sample_batch(args.batch_size)
        info = agent.train_step(*batch, log=step % 100 == 0)
        if log_buf is None:
            log_buf = info
        elif info is not None:
            for k in log_buf.keys():
                log_buf[k] += info[k]
            n_info += 1

        if step % args.log_every == 0:
            sps = step / (time.time() - t0)
            msg = "  ".join(f"{k}={log_buf[k] / n_info:+.3f}" for k in log_buf.keys())
            print(f"[{step:>7d}/{args.total_steps}] ({sps:>4.0f} sps)  {msg}")
            log_buf = {k: 0.0 for k in log_buf.keys()}
            n_info = 0

        if step % args.eval_every == 0:
            results = evaluate_zero_shot(args.env_name, agent)
            print(results)


@torch.no_grad()
def reward_fit_quality(agent, states, rewards):
    psi = agent.psi(states.to(agent.device))  # (B, psi_dim)
    z = agent.fit_reward(states, rewards)  # (psi_dim,)
    r_hat = psi @ z  # (B,)
    r = rewards.to(agent.device).view(-1)
    return torch.nn.functional.cosine_similarity(
        r_hat.unsqueeze(0), r.unsqueeze(0)
    ).item()


def evaluate_zero_shot(
    env_name,
    agent,
    task_ids=(1, 2, 3, 4, 5),
    n_goal_samples=1024,
    goal_noise_std=0.05,
    n_episodes=10,
    max_steps=1000,
    render=False,
):
    """Zero-shot evaluation on goal-conditioned OGBench tasks.

    For each task:
      1. Read the goal observation from env.reset().
      2. Generate a batch of samples by adding Gaussian noise around the goal,
         all labelled reward = 1 (i.e. "at the goal").
      3. Infer z = agent.fit_reward(states, rewards).
      4. Run n_episodes with action = agent.act(obs, z); count successes.
    """
    env = ogbench.make_env_and_datasets(env_name, env_only=True)
    agent.eval_mode()
    device = agent.device
    results = {}

    for task_id in task_ids:
        ob, info = env.reset(options=dict(task_id=task_id, render_goal=False))
        goal = np.asarray(info["goal"], dtype=np.float32)  # (obs_dim,)

        # --- 1. Generate samples around the goal, all with reward = 1 ---
        goal_t = torch.as_tensor(goal, device=device).unsqueeze(0)  # (1, obs_dim)
        noise = (
            torch.randn(n_goal_samples, goal.shape[0], device=device) * goal_noise_std
        )
        states_t = goal_t + noise  # (B, obs_dim)
        states_t[0] = goal_t  # keep the exact goal
        rewards_t = torch.ones(n_goal_samples, 1, device=device)

        # --- 2. Infer z ---
        z = agent.fit_reward(states_t, rewards_t)  # (psi_dim,)
        print(
            f"z and reward cosine similarity: {reward_fit_quality(agent, states_t, rewards_t)}"
        )
        z = z.unsqueeze(0)  # (1, psi_dim) for act()

        # --- 3. Roll out the policy ---
        successes = 0
        for _ in range(n_episodes):
            ob, info = env.reset(options=dict(task_id=task_id, render_goal=render))
            for _ in range(max_steps):
                ob_t = torch.as_tensor(
                    ob, dtype=torch.float32, device=device
                ).unsqueeze(0)
                a = agent.act(ob_t, z).cpu().numpy().squeeze(0)
                ob, _, term, trunc, info = env.step(a)
                if render:
                    env.render()
                if term or trunc:
                    break
            successes += float(info.get("success", 0.0))

        results[task_id] = successes / n_episodes

        print_collapse_report(agent, states_t, "phi")

    agent.train_mode()
    return results


def psi_collapse_diagnostics(agent, states, name="psi"):
    """Inspect the covariance of an encoder's outputs to detect collapse.

    Healthy: rank ≈ d, condition number modest, off-diag energy << diag energy.
    Collapsed: a few large eigenvalues dominate (or all zero), high condition
    number, effective rank << d.
    """
    enc = agent.psi if name == "psi" else agent.phi
    z = enc(states.to(agent.device))  # (B, d)
    B, d = z.shape

    mean = z.mean(0)  # (d,)
    zc = z - mean
    cov = (zc.T @ zc) / max(B - 1, 1)  # (d, d)

    diag = cov.diagonal()
    offdiag_sq = cov.pow(2).sum() - diag.pow(2).sum()
    diag_sq = diag.pow(2).sum()

    # Eigenvalue spectrum (symmetric → eigvalsh, sorted ascending)
    eigvals = torch.linalg.eigvalsh(cov).clamp_min(0)
    total = eigvals.sum().clamp_min(1e-12)
    p = eigvals / total

    # Effective rank (participation ratio): exp(entropy of normalized spectrum)
    eff_rank = torch.exp(-(p * (p + 1e-12).log()).sum()).item()
    cond = (eigvals[-1] / eigvals[0].clamp_min(1e-12)).item()
    top1_frac = (eigvals[-1] / total).item()
    top5_frac = (eigvals[-min(5, d) :].sum() / total).item()

    stats = {
        f"{name}/mean_norm": mean.norm().item(),
        f"{name}/feature_norm": z.norm(dim=-1).mean().item(),
        f"{name}/diag_mean": diag.mean().item(),
        f"{name}/diag_std": diag.std().item(),
        f"{name}/offdiag_ratio": (offdiag_sq / diag_sq.clamp_min(1e-12)).sqrt().item(),
        f"{name}/effective_rank": eff_rank,  # ∈ [1, d]; collapse → 1
        f"{name}/condition_num": cond,  # collapse → ∞
        f"{name}/top1_var_frac": top1_frac,  # collapse → 1
        f"{name}/top5_var_frac": top5_frac,
        f"{name}/dim": d,
    }

    return stats, eigvals.cpu()


@torch.no_grad()
def sample_buffer_states(buffer_obs, n=2048, device=None):
    idx = torch.randint(0, buffer_obs.size(0), (n,), device=buffer_obs.device)
    states = buffer_obs[idx]
    return states.to(device) if device is not None else states


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


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--env_name",
        default="pointmaze-medium-navigate-v0",
        help="OGBench env id.  See sweep_antmaze in the reference repo.",
    )
    p.add_argument("--total_steps", type=int, default=1_000_000)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--log_every", type=int, default=1000)
    p.add_argument("--eval_every", type=int, default=25000)

    args = p.parse_args()
    train_env(args)
