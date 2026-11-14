"""Single source of truth for TD-JEPA hyperparameters.

Every training script (train.py, train_mixed.py) reads its agent config and
its training/eval settings from here.  CLI flags only override these values.

  agent_config(env_name, obs_dim, act_dim) -> TD_JEPA_CONFIG
  train_params(env_name, **overrides)      -> TrainParams
"""

import os
from dataclasses import dataclass, replace

from agents.networks import ActorConfig, EncoderConfig, FbConfig
from agents.td_jepa import TD_JEPA_CONFIG

# ─── Environments ────────────────────────────────────────────────────────────

# ExORL / DMC lives in the BSynth checkout (exorl.dmc + exorl/datasets/...).
BSYNTH_ROOT = os.path.expanduser("~/Documents/BSynth")

DMC_TASKS = {
    "walker": ["walker_flip", "walker_run", "walker_stand", "walker_walk"],
    "cheetah": ["cheetah_run", "cheetah_run_backward", "cheetah_walk", "cheetah_walk_backward"],
    "quadruped": ["quadruped_jump", "quadruped_run", "quadruped_stand", "quadruped_walk"],
}


def dmc_domain(env_name):
    """'walker' / 'walker_run' -> 'walker'; None for non-DMC envs."""
    for key in DMC_TASKS:
        if env_name == key or env_name.startswith(key + "_"):
            return key
    return None


def is_dmc(env_name):
    return dmc_domain(env_name) is not None


# ─── Training / eval settings ───────────────────────────────────────────────


@dataclass
class TrainParams:
    total_steps: int = 1_000_000
    batch_size: int = 256
    log_every: int = 1000
    eval_every: int = 10_000
    # eval
    eval_tasks: int = 5  # OGBench only; DMC evals every task in DMC_TASKS
    eval_episodes: int = 10
    eval_max_steps: int = 1000
    eval_inference_samples: int = 50_000
    shift_reward: float = 1.0  # OGBench only
    traj_dir: str = "trajectories"
    eval_verbose: bool = True
    eval_calibration: bool = True
    # data
    expl_alg: str = "proto"  # DMC only: ExORL exploration dataset
    device: str = "cuda"
    # checkpointing
    checkpoint_every: int = 50_000  # 0 disables periodic checkpoints (final.pt is still written)
    checkpoint_dir: str = ""  # "" -> checkpoints/<run name>
    checkpoint_keep_all: bool = False  # also keep a weights-only step_XXXXXXX.pt per checkpoint
    resume: str = ""  # path to a checkpoint, or "latest" for <checkpoint_dir>/latest.pt


def train_params(env_name, **overrides):
    p = TrainParams()
    if is_dmc(env_name):
        p = replace(
            p,
            # Paper uses 1024; 512 to fit/finish on a GTX 1070.
            batch_size=512,
            eval_every=100_000,
            eval_verbose=False,
            eval_calibration=False,
        )
    return replace(p, **overrides)


# ─── Agent configs ──────────────────────────────────────────────────────────


def agent_config(env_name, obs_dim, act_dim):
    if env_name.startswith("pointmaze"):
        return TD_JEPA_CONFIG(
            obs_dim=2,
            act_dim=2,
            phi_dim=64,
            psi_dim=64,
            phi_config=EncoderConfig(n_hidden=4, width=256, siren_encoder=True),
            psi_config=EncoderConfig(n_hidden=4, width=256, siren_encoder=True),
            T_phi_config=FbConfig(width=512, n_sa=3, n_sz=3, n_shared=3),
            T_psi_config=FbConfig(width=512, n_sa=3, n_sz=3, n_shared=3),
            actor_config=ActorConfig(width=256, depth=4),
            actor_bc_coeff=0.3,
            actor_use_raw_obs=True,
            lambda_phi=2.0,
            lambda_psi=2.0,
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
        # Paper Table 4, OGBench proprio (the 0.98 antmaze-medium-navigate config).
        return TD_JEPA_CONFIG(
            obs_dim=29,
            act_dim=8,
            phi_dim=256,
            psi_dim=50,
            phi_config=EncoderConfig(n_hidden=4, width=512),
            psi_config=EncoderConfig(n_hidden=4, width=512),
            T_phi_config=FbConfig(width=512, n_sa=2, n_sz=2, n_shared=2),
            T_psi_config=FbConfig(width=512, n_sa=2, n_sz=2, n_shared=2),
            actor_config=ActorConfig(width=512, depth=4),
            actor_bc_coeff=0.3,
            actor_use_raw_obs=True,
            lambda_phi=1.0,
            lambda_psi=1.0,
        )
    if is_dmc(env_name):
        # Paper Table 4, DMC proprio.
        return TD_JEPA_CONFIG(
            obs_dim=obs_dim,
            act_dim=act_dim,
            phi_dim=256,
            psi_dim=50,
            phi_config=EncoderConfig(n_hidden=0, width=256),
            psi_config=EncoderConfig(n_hidden=2, width=256),
            T_phi_config=FbConfig(width=1024, n_sa=3, n_sz=3, n_shared=3),
            T_psi_config=FbConfig(width=1024, n_sa=3, n_sz=3, n_shared=3),
            actor_config=ActorConfig(width=256, depth=3),
            actor_bc_coeff=0.0,
            actor_use_raw_obs=True,
            lambda_phi=1.0,
            lambda_psi=1.0,
            gamma=0.98,
            tau_encoder=0.001,
            tau_predictor=0.001,
            eval_device="cuda",
        )
    raise ValueError(f"no default config for env {env_name!r}")
