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
    )


def train_env(args, config=None):

    if config is None:
        config = default_config(args.env_name)

    buffer = get_dataset(args.env_name)
    agent = TD_JEPA(config)  # pyright: ignore[reportArgumentType]
    t0 = time.time()

    log_buf = None

    for step in range(1, args.total_steps + 1):
        batch = buffer.sample_batch(args.batch_size)
        info = agent.train_step(*batch, log=True)
        if log_buf is None:
            log_buf = info
        else:
            for k in log_buf.keys():
                log_buf[k] += info[k]

        if step % args.log_every == 0:
            sps = step / (time.time() - t0)
            msg = "  ".join(
                f"{k}={log_buf[k] / args.log_every:+.3f}" for k in log_buf.keys()
            )
            print(f"[{step:>7d}/{args.total_steps}] ({sps:>4.0f} sps)  {msg}")
            log_buf = {k: 0.0 for k in log_buf.keys()}


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

    args = p.parse_args()
    train_env(args)
