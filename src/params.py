"""Params manager: hyperparameters live in .params/{bfm_name}_{env_name}_params.yaml.

Every training script (train.py, train_mixed.py, eval_checkpoint.py) reads its
agent config and its training/eval settings through here.  CLI flags only
override the values from the params file.

Lookup for (bfm_name, env_name), first hit wins:
  .params/{bfm}_{env_name}_params.yaml   e.g. tdj_antmaze-large-navigate-v0_params.yaml
  .params/{bfm}_{family}_params.yaml     e.g. tdj_antmaze_params.yaml, tdj_walker_params.yaml
where family is the DMC domain (walker_run -> walker) or the OGBench prefix
(antmaze-medium-navigate-v0 -> antmaze).

File layout:
  train:  TrainParams fields (anything omitted keeps its dataclass default)
  agent:  TD_JEPA_CONFIG kwargs, with sub-maps phi / psi (EncoderConfig),
          T_phi / T_psi (FbConfig) and actor (ActorConfig).  obs_dim / act_dim
          come from the dataset unless the file sets them.

  agent_config(env_name, obs_dim, act_dim, bfm_name="tdj") -> TD_JEPA_CONFIG
  train_params(env_name, bfm_name="tdj", **overrides)      -> TrainParams
"""

import copy
import dataclasses
import os
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path

import yaml

from agents.networks import ActorConfig, EncoderConfig, FbConfig
from agents.td_jepa import TD_JEPA_CONFIG

PARAMS_DIR = Path(__file__).resolve().parent / ".params"
DEFAULT_BFM = "tdj"

# ExORL / DMC lives in the BSynth checkout (exorl.dmc + exorl/datasets/...).
BSYNTH_ROOT = os.path.expanduser("~/Projects/BSynth")

# ─── Environments ────────────────────────────────────────────────────────────

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


def env_family(env_name):
    """'walker_run' -> 'walker', 'antmaze-medium-navigate-v0' -> 'antmaze'."""
    return dmc_domain(env_name) or env_name.split("-")[0]


# ─── Training / eval settings ───────────────────────────────────────────────


@dataclass
class TrainParams:
    batch_size: int  # required: must be set in the params file's train section
    bfm_name: str = DEFAULT_BFM  # selects .params/{bfm_name}_{env}_params.yaml
    total_steps: int = 1_000_000
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


# ─── Params files ───────────────────────────────────────────────────────────


def params_path(env_name, bfm_name=DEFAULT_BFM):
    names = dict.fromkeys([env_name, env_family(env_name)])  # ordered, deduped
    tried = [PARAMS_DIR / f"{bfm_name}_{n}_params.yaml" for n in names]
    for path in tried:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"no params file for bfm={bfm_name!r} env={env_name!r}; tried: "
        + ", ".join(str(p) for p in tried)
    )


@lru_cache(maxsize=None)
def _load(path):
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    unknown = set(data) - {"train", "agent"}
    if unknown:
        raise ValueError(f"{path}: unknown top-level keys {sorted(unknown)}")
    return data


def load_params(env_name, bfm_name=DEFAULT_BFM):
    """Raw {'train': {...}, 'agent': {...}} dict from the params file (a copy)."""
    return copy.deepcopy(_load(params_path(env_name, bfm_name)))


def train_params(env_name, bfm_name=DEFAULT_BFM, **overrides):
    file_train = load_params(env_name, bfm_name).get("train") or {}
    known = {f.name for f in dataclasses.fields(TrainParams)}
    unknown = set(file_train) - (known - {"bfm_name"})
    if unknown:
        raise ValueError(
            f"{params_path(env_name, bfm_name)}: unknown train keys {sorted(unknown)}"
        )
    if "batch_size" not in file_train:
        raise ValueError(f"{params_path(env_name, bfm_name)}: train.batch_size is required")
    return TrainParams(**{**file_train, **overrides, "bfm_name": bfm_name})


_SUB_CONFIGS = {
    "phi_config": ("phi", EncoderConfig),
    "psi_config": ("psi", EncoderConfig),
    "T_phi_config": ("T_phi", FbConfig),
    "T_psi_config": ("T_psi", FbConfig),
    "actor_config": ("actor", ActorConfig),
}


def agent_config(env_name, obs_dim, act_dim, bfm_name=DEFAULT_BFM):
    if bfm_name != "tdj":
        raise ValueError(f"no agent builder for bfm {bfm_name!r} (only 'tdj')")
    kw = load_params(env_name, bfm_name).get("agent") or {}
    path = params_path(env_name, bfm_name)
    for arg, (key, cls) in _SUB_CONFIGS.items():
        if key not in kw:
            raise ValueError(f"{path}: agent.{key} is required")
        kw[arg] = cls(**kw.pop(key))
    kw.setdefault("obs_dim", obs_dim)
    kw.setdefault("act_dim", act_dim)
    try:
        return TD_JEPA_CONFIG(**kw)
    except TypeError as e:
        raise ValueError(f"{path}: bad agent params ({e})") from e
