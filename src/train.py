"""TD-JEPA training for OGBench and DMC (ExORL) environments.

All hyperparameters come from params.py; CLI flags override them.

Usage:
  uv run train.py --env_name antmaze-medium-navigate-v0
  uv run train.py --env_name walker --expl_alg proto
"""

import argparse
import dataclasses
import glob
import os
import random
import sys
import time

import numpy as np
import torch

import params
from agents.td_jepa import TD_JEPA

# ─── Buffer ──────────────────────────────────────────────────────────────────


class Buffer:
    def __init__(self, s, a, s_prime, terminals, physics=None, device="cuda"):
        self.s = s.to(device)
        self.a = a.to(device)
        self.s_prime = s_prime.to(device)
        self.terminals = terminals.to(device)
        self.physics = physics.to(device) if physics is not None else None

    def sample_batch(self, batch_size):
        idx = torch.randint(0, self.s.shape[0], (batch_size,), device=self.s.device)
        return self.s[idx], self.a[idx], self.s_prime[idx], self.terminals[idx]


def _make_buffer(obs, act, next_obs, term, physics=None, device="cuda"):
    """Drop terminal transitions (term==1) and move everything to `device`."""
    valid = term == 0
    t = lambda x: torch.tensor(x[valid], dtype=torch.float32, device=device)
    return Buffer(
        t(obs),
        t(act),
        t(next_obs),
        t(term.astype(np.float32)).reshape(-1, 1),
        t(physics) if physics is not None else None,
        device=device,
    )


# ─── Dataset loading ────────────────────────────────────────────────────────


def get_dataset_ogbench(env_name, device="cuda"):
    import ogbench

    _, ds, _ = ogbench.make_env_and_datasets(env_name)
    obs = ds["observations"]
    next_obs = np.concatenate([obs[1:], obs[-1:]], axis=0)
    return _make_buffer(obs, ds["actions"], next_obs, ds["terminals"], device=device)


def get_dataset_dmc(env_name, expl_alg, device="cuda"):
    base = os.path.join(
        params.BSYNTH_ROOT, "exorl", "datasets", params.dmc_domain(env_name), expl_alg, "buffer"
    )
    paths = sorted(glob.glob(os.path.join(base, "episode_*.npz")))
    if not paths:
        raise FileNotFoundError(f"No ExORL episodes at {base}")

    obs, nobs, act, phy, term = [], [], [], [], []
    for p in paths:
        f = np.load(p)
        obs.append(f["observation"][:-1])
        nobs.append(f["observation"][1:])
        act.append(f["action"][:-1])
        phy.append(f["physics"][:-1])
        term.append((f["discount"].flatten()[:-1] == 0).astype(np.float32))
    obs, nobs, act, phy, term = map(np.concatenate, (obs, nobs, act, phy, term))
    buf = _make_buffer(obs, act, nobs, term, physics=phy, device=device)
    print(
        f"DMC {env_name}/{expl_alg}: {buf.s.shape[0]} transitions "
        f"(dropped {len(term) - buf.s.shape[0]} terminals)"
    )
    return buf


def get_dataset(env_name, tp: params.TrainParams):
    if params.is_dmc(env_name):
        return get_dataset_dmc(env_name, tp.expl_alg, tp.device)
    return get_dataset_ogbench(env_name, tp.device)


# ─── DMC eval ────────────────────────────────────────────────────────────────


def _import_exorl_dmc():
    if params.BSYNTH_ROOT not in sys.path:
        sys.path.insert(0, params.BSYNTH_ROOT)
    from exorl import dmc

    return dmc


def relabel_reward_dmc(env, physics_np, actions_np):
    """Recompute task rewards from recorded (physics, action) pairs."""
    import mujoco

    m, d = env._physics.model.ptr, env._physics.data.ptr
    rewards = []
    for i in range(physics_np.shape[0]):
        with env._physics.reset_context():
            env._physics.set_state(physics_np[i])
            env._physics.set_control(actions_np[i])
        mujoco.mj_forward(m, d)
        mujoco.mj_fwdPosition(m, d)
        mujoco.mj_sensorVel(m, d)
        mujoco.mj_subtreeVel(m, d)
        rewards.append(env._task.get_reward(env._physics))
    return np.array(rewards, dtype=np.float32)


@torch.no_grad()
def evaluate_dmc(agent, buffer, task_names, num_episodes=10, max_steps=1000,
                 num_inference_samples=50_000):
    """DMC zero-shot eval: reward relabelling + exp/ls z inference."""
    dmc = _import_exorl_dmc()
    agent.eval_mode()
    device = agent.device

    n = min(num_inference_samples, buffer.s_prime.shape[0])
    idx = torch.randperm(buffer.s_prime.shape[0], device=buffer.s_prime.device)[:n]
    pool_next = buffer.s_prime[idx].to(device)
    pool_phy = buffer.physics[idx].cpu().numpy()
    pool_act = buffer.a[idx].cpu().numpy()
    psi_pool = agent.psi(pool_next)

    out = {}
    for task_name in task_names:
        reward_env = dmc.make(task_name, seed=0)
        r = torch.as_tensor(
            relabel_reward_dmc(reward_env, pool_phy, pool_act), device=device
        )
        reward_env.close()

        z_exp = agent._project_z((psi_pool * r[:, None]).mean(0))
        z_ls = agent._project_z(torch.linalg.lstsq(psi_pool, r).solution.squeeze(-1))

        for label, z in [("exp", z_exp), ("ls", z_ls)]:
            returns = []
            for _ in range(num_episodes):
                env = dmc.make(task_name, seed=random.randint(0, 10000))
                ts = env.reset()
                ep_ret = 0.0
                for _ in range(max_steps):
                    o = torch.as_tensor(ts.observation, dtype=torch.float32, device=device)
                    a = agent.act(o.unsqueeze(0), z.unsqueeze(0)).cpu().numpy().squeeze(0)
                    ts = env.step(a)
                    ep_ret += ts.reward
                    if ts.last():
                        break
                returns.append(ep_ret)
                env.close()
            out[f"{task_name}_{label}"] = float(np.mean(returns))
            print(f"  [{task_name}] {label}: return={out[f'{task_name}_{label}']:.1f}")

    agent.train_mode()
    return out


# ─── Evaluator dispatch ──────────────────────────────────────────────────────


def make_evaluator(env_name, buffer, tp: params.TrainParams):
    """Returns eval_fn(agent, step) -> (results dict, summary str)."""
    if params.is_dmc(env_name):
        task_names = params.DMC_TASKS[params.dmc_domain(env_name)]
        if env_name in task_names:  # single-task env name, e.g. walker_run
            task_names = [env_name]

        def eval_fn(agent, step):
            res = evaluate_dmc(
                agent, buffer, task_names,
                num_episodes=tp.eval_episodes,
                max_steps=tp.eval_max_steps,
                num_inference_samples=tp.eval_inference_samples,
            )
            return res, f"DMC returns mean={np.mean(list(res.values())):.2f}"

        return eval_fn

    import ogbench
    from eval import detect_morph, evaluate_zero_shot

    # Persistent env reused across every eval (env_only avoids reloading the dataset).
    eval_env = ogbench.make_env_and_datasets(env_name, env_only=True)
    morph = detect_morph(env_name)

    def eval_fn(agent, step):
        res = evaluate_zero_shot(
            agent, eval_env, buffer.s, buffer.a, buffer.s_prime, morph,
            num_tasks=tp.eval_tasks,
            num_episodes=tp.eval_episodes,
            max_steps=tp.eval_max_steps,
            num_inference_samples=tp.eval_inference_samples,
            shift_reward=tp.shift_reward,
            verbose=tp.eval_verbose,
            step=step,
            save_dir=tp.traj_dir,
            run_calibration=tp.eval_calibration,
        )
        mean_sr = sum(res.values()) / max(len(res), 1)
        return res, f"zero-shot success {res}  mean={mean_sr:+.3f}"

    return eval_fn


# ─── Checkpointing ───────────────────────────────────────────────────────────


def _rng_state():
    st = {
        "torch": torch.get_rng_state(),
        "python": random.getstate(),
        "numpy": [torch.as_tensor(x.astype(np.int64)) if isinstance(x, np.ndarray) else x
                  for x in np.random.get_state()],
    }
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def _set_rng_state(st):
    torch.set_rng_state(st["torch"].cpu())
    random.setstate(st["python"])
    np.random.set_state(tuple(x.numpy().astype(np.uint32) if torch.is_tensor(x) else x
                             for x in st["numpy"]))
    if "cuda" in st and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in st["cuda"]])


class Checkpointer:
    """Writes <dir>/latest.pt (full, resumable) every `checkpoint_every` steps and
    <dir>/final.pt at the end; optionally weights-only <dir>/step_XXXXXXX.pt snapshots."""

    def __init__(self, run_name, tp: params.TrainParams, meta):
        self.dir = tp.checkpoint_dir or os.path.join("checkpoints", run_name)
        self.tp = tp
        self.meta = {**meta, "train_params": dataclasses.asdict(tp)}

    def resume_path(self):
        if not self.tp.resume:
            return None
        if self.tp.resume == "latest":
            path = os.path.join(self.dir, "latest.pt")
            if not os.path.exists(path):
                print(f"[ckpt] no {path}; starting from scratch")
                return None
            return path
        return self.tp.resume

    def maybe_resume(self, agent):
        """Load a checkpoint into `agent` if requested; returns the step to resume from."""
        path = self.resume_path()
        if path is None:
            return 0
        ckpt = agent.load_state(path)
        if "rng" in ckpt:
            _set_rng_state(ckpt["rng"])
        print(f"[ckpt] resumed from {path} at step {ckpt['step']}")
        return int(ckpt["step"])

    def save(self, agent, step, final=False):
        name = "final.pt" if final else "latest.pt"
        path = os.path.join(self.dir, name)
        agent.save_state(path, step=step, rng=_rng_state(), **self.meta)
        if self.tp.checkpoint_keep_all and not final:
            agent.save_state(os.path.join(self.dir, f"step_{step:07d}.pt"),
                             include_optimizers=False, step=step, **self.meta)
        print(f"[ckpt] saved {path} (step {step})")


# ─── Training loop ───────────────────────────────────────────────────────────


def train_loop(agent, buffer, eval_fn, tp: params.TrainParams, eval_suffix="",
               checkpointer=None, start_step=0):
    t0 = time.time()
    log_buf, n_info = {}, 0
    step = start_step
    for step in range(start_step + 1, tp.total_steps + 1):
        info = agent.train_step(*buffer.sample_batch(tp.batch_size), log=step % 100 == 0)
        if info is not None:
            for k, v in info.items():
                log_buf[k] = log_buf.get(k, 0.0) + v
            n_info += 1

        if step % tp.log_every == 0 and n_info > 0:
            sps = (step - start_step) / (time.time() - t0)
            msg = "  ".join(f"{k}={v / n_info:+.3f}" for k, v in log_buf.items())
            print(f"[{step:>7d}/{tp.total_steps}] ({sps:>4.0f} sps)  {msg}")
            log_buf, n_info = {}, 0

        if step % tp.eval_every == 0:
            _, summary = eval_fn(agent, step)
            print(f"[{step:>7d}] {summary}{eval_suffix}")

        if checkpointer and tp.checkpoint_every and step % tp.checkpoint_every == 0:
            checkpointer.save(agent, step)

    if checkpointer:
        checkpointer.save(agent, step, final=True)


def build_and_train(run_name, env_name, buffer, eval_fn, tp: params.TrainParams,
                    meta=None, eval_suffix=""):
    """Build the agent for `env_name`, resume if requested, train, checkpoint."""
    obs_dim, act_dim = buffer.s.shape[1], buffer.a.shape[1]
    config = params.agent_config(env_name, obs_dim, act_dim)
    config.train_device = tp.device
    agent = TD_JEPA(config)  # pyright: ignore[reportArgumentType]
    ckpt = Checkpointer(run_name, tp, {"env_name": env_name, "obs_dim": obs_dim,
                                       "act_dim": act_dim, **(meta or {})})
    start = ckpt.maybe_resume(agent)
    train_loop(agent, buffer, eval_fn, tp, eval_suffix, checkpointer=ckpt, start_step=start)
    return agent


def train(env_name, tp: params.TrainParams):
    buffer = get_dataset(env_name, tp)
    run_name = env_name if not params.is_dmc(env_name) else f"{env_name}_{tp.expl_alg}"
    return build_and_train(run_name, env_name, buffer, make_evaluator(env_name, buffer, tp), tp)


# ─── CLI ─────────────────────────────────────────────────────────────────────


def add_train_param_args(p: argparse.ArgumentParser):
    """One flag per TrainParams field, defaulting to None (= use params.py)."""
    for f in dataclasses.fields(params.TrainParams):
        if f.type in (bool, "bool"):
            p.add_argument(f"--{f.name}", action=argparse.BooleanOptionalAction, default=None)
        else:
            typ = {"int": int, "float": float, "str": str}.get(f.type, f.type)
            p.add_argument(f"--{f.name}", type=typ, default=None)
    return p


def cli_overrides(args):
    names = {f.name for f in dataclasses.fields(params.TrainParams)}
    return {k: v for k, v in vars(args).items() if k in names and v is not None}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--env_name", default="antmaze-medium-navigate-v0",
                   help="OGBench env id (e.g. antmaze-medium-navigate-v0) or DMC domain/task (walker, walker_run).")
    add_train_param_args(p)
    args = p.parse_args()
    tp = params.train_params(args.env_name, **cli_overrides(args))
    print(tp)
    train(args.env_name, tp)
