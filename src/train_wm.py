import argparse
import time

import numpy as np
import ogbench
import torch

# NOTE: adjust the module name below to whatever you saved the eval suite as
# (the file whose top docstring is "Evaluation suite for TD-JEPA").
from eval import detect_morph, evaluate_zero_shot
from networks import ActorConfig, EncoderConfig, FbConfig, WmConfig
from td_wm import TD_WM, TD_WM_CONFIG


# Hyperparameters
def default_config(env_name):
    if env_name.startswith("pointmaze"):
        return TD_WM_CONFIG(
            obs_dim=2,
            act_dim=2,
            phi_dim=128,
            psi_dim=128,
            phi_config=EncoderConfig(n_hidden=4, width=256, siren_encoder=True),
            psi_config=EncoderConfig(n_hidden=4, width=256, siren_encoder=True),
            T_phi_config=FbConfig(width=512, n_sa=3, n_sz=3, n_shared=3),
            T_psi_config=FbConfig(width=512, n_sa=3, n_sz=3, n_shared=3),
            P_phi_config=WmConfig(width=512, n_sa=3, n_shared=3),
            P_psi_config=WmConfig(width=512, n_sa=3, n_shared=3),
            actor_config=ActorConfig(width=256, depth=4),
            actor_bc_coeff=0.3,
            actor_use_raw_obs=True,
            lambda_phi=1.0,
            lambda_psi=1.0,
            tau_encoder=0.005,
            tau_predictor=0.005,
            phi_lr=1e-5,
            psi_lr=1e-5,
            T_phi_lr=1e-3,
            T_psi_lr=1e-3,
            actor_lr=1e-4,
            train_goal_ratio=0.125,
            latent_z_batch_size=4,
        )
    if env_name.startswith("antmaze"):
        return TD_WM_CONFIG(
            obs_dim=29,
            act_dim=8,
            phi_dim=256,
            psi_dim=256,
            phi_config=EncoderConfig(n_hidden=4, width=256),
            psi_config=EncoderConfig(n_hidden=4, width=256),
            T_phi_config=FbConfig(width=512, n_sa=2, n_sz=2, n_shared=2),
            T_psi_config=FbConfig(width=512, n_sa=2, n_sz=2, n_shared=2),
            P_phi_config=WmConfig(width=512, n_sa=3, n_shared=3),
            P_psi_config=WmConfig(width=512, n_sa=3, n_shared=3),
            actor_config=ActorConfig(width=256, depth=4),
            # train_device="cpu",
            actor_bc_coeff=0.3,
            actor_use_raw_obs=True,
            lambda_phi=1.0,
            lambda_psi=1.0,
            #
            phi_lr=1e-4,
            psi_lr=1e-4,
            T_phi_lr=1e-3,
            T_psi_lr=1e-3,
            actor_lr=1e-4,
        )
    raise ValueError(f"no default config for env {env_name!r}")


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
    # Persistent env reused across every eval (env_only avoids reloading the dataset).
    eval_env = ogbench.make_env_and_datasets(args.env_name, env_only=True)
    morph = detect_morph(args.env_name)
    agent = TD_WM(config)  # pyright: ignore[reportArgumentType]
    t0 = time.time()

    print(torch.max(buffer.s[:, 0]), torch.min(buffer.s[:, 0]))
    print(torch.max(buffer.s[:, 1]), torch.min(buffer.s[:, 1]))

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
            results = evaluate_zero_shot(
                agent,
                eval_env,
                buffer.s,
                buffer.a,
                buffer.s_prime,
                morph,
                num_tasks=args.eval_tasks,
                num_episodes=args.eval_episodes,
                max_steps=args.eval_max_steps,
                num_inference_samples=args.eval_inference_samples,
                shift_reward=args.shift_reward,
                verbose=args.eval_verbose,
                step=step,
                save_dir=args.traj_dir,
                run_calibration=args.eval_calibration,
            )
            mean_sr = sum(results.values()) / max(len(results), 1)
            print(f"[{step:>7d}] zero-shot success {results}  mean={mean_sr:+.3f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--env_name",
        default="antmaze-medium-navigate-v0",
        help="OGBench env id.  See sweep_antmaze in the reference repo.",
    )
    p.add_argument("--total_steps", type=int, default=1_000_000)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--log_every", type=int, default=1000)
    p.add_argument("--eval_every", type=int, default=10000)

    # Eval-suite knobs (forwarded to evaluate_zero_shot).
    p.add_argument("--eval_tasks", type=int, default=5)
    p.add_argument("--eval_episodes", type=int, default=10)
    p.add_argument("--eval_max_steps", type=int, default=1000)
    p.add_argument("--eval_inference_samples", type=int, default=50_000)
    p.add_argument("--shift_reward", type=float, default=1.0)
    p.add_argument("--traj_dir", default="trajectories")
    # Turn off with --no-eval-verbose / --no-eval-calibration for cheap evals.
    p.add_argument(
        "--eval_verbose", action=argparse.BooleanOptionalAction, default=True
    )
    p.add_argument(
        "--eval_calibration", action=argparse.BooleanOptionalAction, default=True
    )

    args = p.parse_args()
    train_env(args)
