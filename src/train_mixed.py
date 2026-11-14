"""Train TD-JEPA on a mixture of antmaze-medium-explore and antmaze-medium-navigate data.

Usage:
    uv run train_mixed.py --explore_frac 0.25 --total_steps 1000000 --eval_every 50000

--explore_frac: fraction of the combined dataset that should be explore data.
  0.0 = pure navigate (baseline, 0.98)
  1.0 = pure explore (hypothesis: worse)
  0.25/0.50/0.75 = mixed concentrations

Full navigate dataset is always loaded. Explore is subsampled to achieve
the desired fraction: n_exp = n_nav * frac / (1 - frac). Both are concatenated
into one flat buffer and sampled uniformly. Both datasets share the same
antmaze-medium maze, so eval goals (tasks 1-5) are identical.
"""
import argparse

import numpy as np
import torch

import params
from train import (
    Buffer,
    add_train_param_args,
    build_and_train,
    cli_overrides,
    get_dataset_ogbench,
    make_evaluator,
    train_loop,
)

NAV_ENV = "antmaze-medium-navigate-v0"
EXP_ENV = "antmaze-medium-explore-v0"


def get_mixed_dataset(explore_frac, device="cuda", seed=0):
    """Concatenate full navigate + subsampled explore into one flat buffer.

    Explore is subsampled (random rows) so it comprises ``explore_frac`` of the
    combined pool:  n_exp = n_nav * explore_frac / (1 - explore_frac).
    The combined buffer is then sampled uniformly.
    """
    rng = np.random.default_rng(seed)
    nav = get_dataset_ogbench(NAV_ENV, device=device)
    exp = get_dataset_ogbench(EXP_ENV, device=device)
    n_nav, n_exp_all = nav.s.shape[0], exp.s.shape[0]
    n_exp = n_exp_all if explore_frac >= 1.0 else min(int(n_nav * explore_frac / (1.0 - explore_frac)), n_exp_all)

    if n_exp < n_exp_all:
        idx = torch.as_tensor(rng.choice(n_exp_all, size=n_exp, replace=False), device=exp.s.device)
    else:
        idx = slice(None)
    cat = lambda a, b: torch.cat([a, b[idx]], dim=0)

    print(f"  navigate: {n_nav}  explore: {n_exp} (of {n_exp_all})  "
          f"total: {n_nav + n_exp}  (explore_frac={explore_frac:.2f})")
    return Buffer(cat(nav.s, exp.s), cat(nav.a, exp.a), cat(nav.s_prime, exp.s_prime),
                  cat(nav.terminals, exp.terminals), device=device)


def train_env(explore_frac, seed, tp: params.TrainParams):
    buffer = get_mixed_dataset(explore_frac, device=tp.device, seed=seed)
    # Eval on the navigate env — same maze, same 5 goal tasks.
    eval_fn = make_evaluator(NAV_ENV, buffer, tp)
    build_and_train(
        f"mixed_explore{explore_frac:.2f}_seed{seed}", NAV_ENV, buffer, eval_fn, tp,
        meta={"explore_frac": explore_frac, "seed": seed},
        eval_suffix=f"  (explore_frac={explore_frac:.2f})",
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--explore_frac", type=float, default=0.5,
                   help="Fraction of training data from explore (0=nav, 1=explore)")
    p.add_argument("--seed", type=int, default=0)
    add_train_param_args(p)
    args = p.parse_args()
    tp = params.train_params(NAV_ENV, **{"eval_every": 50_000, **cli_overrides(args)})
    print(tp)
    train_env(args.explore_frac, args.seed, tp)
