# TD-JEPA JAX Port

Porting TD-JEPA from PyTorch (`td_jepa.py` / `train.py`) to JAX/Flax NNX. The codebase lives at `~/Documents/BSynth`, PyTorch reference at `~/Documents/td-jepa`.

## Two Reference Variants

1. **`train.py` + `td_jepa.py`**: 256-dim phi/psi, no inverse dynamics loss, per-head TD targets, D=1, batch 256. This is the "original paper" config the user targets.
2. **`train_inverse_dynamics.py` + `td_jepa_inverse_dynamics.py`** (git commit `947a81c`): 50-dim, inverse dynamics auxiliary loss, ensemble-mean TD targets, D=2, batch 1024. The Justfile `run-antmaze` target uses this. Reached 0.79 mean at 700K.

**Always check which reference variant the user means before making changes.**

## Critical Architecture Details

### Encoder (StateEncoder / EncoderMLP)
- PyTorch: `Linear(obs, hidden) → LayerNorm → Tanh → [Linear→ReLU]*(n-1) → Linear(hidden, z_dim) → Norm(sqrt(d)*normalize)`
- **No obs normalization** — raw obs straight in. Only LayerNorm inside the network normalizes.
- Output: L2-normalized to `||z|| = sqrt(d)` via `NormZLayer`.
- Weight init: `orthogonal_(gain=1.0)` on ALL Linear layers, zero bias.

### Actor (CRITICAL — dual-head s/sz)
- The PyTorch `networks.py` Actor was **manually modified** to use dual-head embeddings instead of single `_build_mlp`:
  ```python
  self.s = _build_embedding(state_dim, hidden, hidden, depth=1)   # embeds obs alone
  self.sz = _build_embedding(state_dim + z_dim, hidden, hidden, depth=1)  # embeds obs+z
  self.net = _build_mlp(hidden*2, action_dim, hidden_dims)  # MLP body
  ```
- **`_build_embedding` does NOT apply orthogonal init** — only `_build_mlp` does. The s/sz heads use PyTorch default init (Kaiming/lecun_uniform). The body MLP uses orthogonal.
- JAX port: `PredictorEmbedding` with `use_orth_init=True` for predictor heads (correct — `FB_Predictor.apply(self._init_weights)` applies orthogonal). Actor s/sz heads should use `use_orth_init=False` (Kaiming/lecun_uniform) — matching PyTorch's `_build_embedding` which has NO orthogonal init. This was a root cause of the 0.40 vs 0.98 gap.
- `sample_action`: deterministic + clipped Gaussian noise (stddev=0.1, clip=0.3), clamped to [-1,1].
- **The s/sz dual-head actor is THE key knob** — PyTorch reached 0.845 with it, 0.10 without.

### Predictor (ForwardHead / ForwardEnsemble)
- `_build_embedding` with `n_sa=2, n_sz=2, n_shared=2` for antmaze.
- Two embedding heads: `(s+a)` and `(s+z)`, each output `hidden//2`, concatenated to `hidden`.
- Body: `[Linear→ReLU]*n_shared → Linear(out)`.
- Ensemble of 2 heads. `.predict()` = mean. `__call__()` = tuple.
- Weight init: orthogonal via `self.apply(self._init_weights)` on `FB_Predictor`.

### TD Targets
- `train.py` uses **per-head** targets: each head gets its own target from its own T_phi_t/T_psi_t output.
- `train_inverse_dynamics.py` uses **ensemble-mean** targets: all heads predict the SAME target (mean of K).

### Z Sampling
- `train.py`: uses **online** psi for goal sampling.
- `train_inverse_dynamics.py`: uses **psi_target** for goal sampling.
- `_sample_mixed_z`: `z = norm_z(randn(B, psi_dim))`; with prob `goal_ratio` replace by `norm_z(psi(next_obs[perm]) @ inv(z_cov + 1e-6*I))`.

### Eval (reward-fit z inference)
- `train.py`: `reward = (dist <= tol).float() - 1.0 + shift_reward` = {0, 1}. **No center_reward.**
- `train_inverse_dynamics.py` Justfile: `--center_reward` (mean-centers reward before lstsq).
- `fit_reward`: `z = lstsq(psi(pool), reward).solution → sqrt(d) * z / ||z||`.
- Pool: 50K random next_obs samples. 10 episodes per task, 5 tasks, max 1000 steps.
- Rollout uses **deterministic** `agent.act(obs, z)` — no noise.

## Hyperparameters (paper Table 4 OGBench proprio — CURRENT TARGET, ALL ALIGNED)
- phi_dim=256, psi_dim=50 (asymmetric)
- encoder hidden=**512**, 4 layers
- predictor hidden=512, n_sa=2, n_sz=2, n_shared=2 (4 total hidden layers: 2 embed + 2 body)
- actor hidden=**512**, depth=4 (→ body has 5 Linear layers: 1 first + 3 hidden + 1 final)
- batch=256, gamma=0.99
- lambda_phi=1.0, lambda_psi=1.0
- goal_ratio=0.5, scale_train_goals=True, z_cov_tau=0.005
- tau_encoder=0.005, tau_predictor=0.005
- lr=1e-4 for all, adam_eps=1e-8 (PyTorch config says 1e-5 but NEVER passes it to Adam — uses torch default 1e-8)
- bc_coeff=0.3, actor_explore_stddev=0.1, actor_explore_clip=0.3
- Actor body: `nnx.relu` first activation (NOT tanh — see pitfall #17)
- Actor grad clip: `optax.clip_by_global_norm(1.0)` — matches PyTorch `clip_grad_norm_(1.0)`
- Terminal transitions: dropped (`valid = term == 0`) — matches PyTorch. JAX uses `done_mask` in TD+actor loss (see pitfall #16)
- Actor s/sz init: Kaiming uniform (`use_orth_init=False`) — matches PyTorch `_build_embedding` default
- normalize_obs=False (raw obs, only LayerNorm inside)
- **max_samples=-1** (load full 1M transitions, NOT 100K cap — see pitfall #18)
- **eval_epi=10** (PyTorch default) — was 2, critical fix
- **n_seeds=1** — was 2, matches PyTorch single seed
- **eval_freq=100K** (pretrain_zs_eval_freq) — final eval at 1M. PyTorch evals every 10K during training.
- **Dataset: episodic (N, T, obs_dim)** via `load_dataset` + `sample_N_batches` with `done_mask` masking terminal transitions
- **Actor std: paper says 0.2, code has 0.1** — NOT YET FIXED (user deferred)
- **RESULT: 0.92 mean at 1M** (task1=0.90, task2=0.90, task3=0.80, task4=1.00, task5=1.00). PyTorch ref: 0.98.

## PAPER TABLE 4 — OGBench Proprio (from tdjepa.pdf, Appendix E.3)

**IMPORTANT: The paper's architecture differs from the `train.py` code in several dimensions.**
The code was modified by the user; the paper's Table 4 reports the canonical config.

| Component | Paper Table 4 (OGBench proprio) | train.py code (modified) | Match? |
|-----------|------|------|--------|
| SFs hidden layers | **4** | 2 (n_sa=2, n_sz=2, n_shared=2) | NO |
| SFs hidden width | **512** | 512 | YES |
| State Encoder phi hidden layers | **4** | 4 | YES |
| State Encoder phi hidden width | **512** | 256 | NO |
| State Encoder phi output dim | **256** | 256 | YES |
| Task Encoder psi hidden layers | **4** | 4 | YES |
| Task Encoder psi hidden width | **512** | 256 | NO |
| Task Encoder psi output dim | **50** | 50 | YES |
| Actor hidden layers | **4** | 4 | YES |
| Actor hidden width | **512** | 256 | NO |
| Actor fixed std | **0.2** | 0.1 (sample_action) | NO |
| gamma | 0.99 | 0.99 | YES |
| batch size | 256 | 256 | YES |
| lambda | 1.0 | 1.0 | YES |
| EMA tau | 0.005 | 0.005 | YES |
| pgoal | 0.5 | 0.5 | YES |

**Key paper-vs-code mismatches — ALL FIXED:**
1. **Encoder/actor/predictor width: 512** — FIXED in params.py (was 256)
2. **SFs (predictor) hidden layers: 4** — confirmed match (2 embed + 2 body = 4 total)
3. **Actor std: 0.2** — NOT YET FIXED (code has 0.1, user deferred)

The paper also says: "actor networks are Gaussian MLPs with a similar architecture to predictors, and fixed standard deviation of 0.2." And: "TD-targets and value estimates for SVG-like policy updates are computed as the mean of twin networks."

## Pitfalls

1. **nnx.vmap requires top-level functions, NOT methods.** `nnx.vmap(lambda model, x: model.phi(x))` fails if `phi` is a method — use a top-level `_encode_phi(model, x)` function instead.

2. **EncoderMLP.__call__ truncation**: Patch edits can accidentally remove the `return self.net(obs)` line, causing `None` returns. Always re-read the file after patching `__call__`.

3. **Obs normalization**: PyTorch does NOT normalize obs before the network. Our `normalize_obs` flag must be `False`. The only normalization is LayerNorm inside the network and NormZLayer on output.

4. **_build_embedding init**: PyTorch's `_build_embedding` does NOT apply orthogonal init — only `_build_mlp` does. The Actor Class also does NOT call `self.apply(weight_init)`. So the Actor's s/sz heads use PyTorch DEFAULT init (Kaiming uniform), and only the body MLP uses orthogonal. The user initially said "use orthogonal init like i do in pytorch" but the PyTorch code actually uses Kaiming default for `_build_embedding`. This was a root cause of the JAX vs PyTorch gap — JAX must use `use_orth_init=False` for actor s/sz heads.

5. **center_reward**: `train.py` does NOT center. Only `train_inverse_dynamics.py` with `--center_reward` does.

6. **Subagent edits**: Delegating file edits to subagents risks truncating method bodies. Re-read affected files after delegation. Prefer direct edits when context is already loaded.

7. **JAX cold-start**: ~25min compilation on first run after code changes (XLA cache invalidated). Use `tmux` + `PYTHONUNBUFFERED=1` for long runs. Don't poll every 3 min — use a 1h cron job instead.

8. **phi256/psi50 asymmetric**: Better than 256/256 — lower R² (better lstsq conditioning), but psi(goal) cosine higher (harder goal separation). Trade-off depends on whether inv-dyn loss is present.

9. **Mismatches vs modified PyTorch train.py (all FIXED 2026-07-15)**:
   - `lambda_phi/psi`: 1.0 (was 2.0)
   - Actor grad clip: `optax.clip_by_global_norm(1.0)` (user confirmed after self-correction)
   - Terminal transitions: dropped (`valid = term == 0`)
   - Actor s/sz init: Kaiming uniform (`use_orth_init=False`) — NOT orthogonal
   - eval_epi: 10 (was 2) — critical for seeing sporadic successes
   - n_seeds: 1 (was 2) — matches PyTorch single seed
   - eval_freq: 10K (was 100K) — catches transient spikes
   - Dataset: flat array (was episode-segmented) — correct (s,s') pairs
   - **Always `git diff` the PyTorch reference before assuming defaults** — the user manually modifies files

10. **Always check git diff on PyTorch reference**: The user manually modifies `train.py` and `networks.py`. Never assume the committed defaults match what was actually run. Run `git diff` on the reference repo to find manual modifications.

11. **eval_epi MUST match PyTorch (10)**: JAX default `eval_epi=2` gives only {0, 0.5, 1.0} success rates. PyTorch uses 10 episodes. With 2 eps, a task succeeding 30% of the time shows as 0.0 or 0.5 — making it look like the task isn't learning when it is. This was a major contributor to the JAX vs PyTorch gap.

12. **Dataset must be FLAT, not episode-segmented**: PyTorch uses a flat (N_valid, obs_dim) array of transitions sampled uniformly. The JAX port reshapes filtered transitions back into (n_eps, T) episodes, which creates incorrect (s, s') pairs at row boundaries where the last state of one episode connects to the first state of the next. Use a flat array and sample uniformly like PyTorch's `buffer.sample_batch(batch_size)`.

13. **User self-corrects**: The user will say "actually do X, i was wrong" — when this happens, update immediately without argument. Examples: grad clip ("actually do clip actor grads i was wrong"), init style, "inv dyn was unnecessary" (correcting a prior finding that inv-dyn was THE key ingredient). Don't assume prior session findings are still current — the user may have ruled them out.

14. **Diff methodology — do it yourself, don't delegate**: A delegated subagent diff timed out after 600s. Doing the diff manually (reading every file line-by-line in both repos) found two new differences that prior diffs over multiple sessions had missed. The manual diff took ~5 minutes of focused reading. Always prefer direct file reading over delegation for cross-repo code comparison.

15. **Encoder hidden ReLU init gain**: PyTorch `StateEncoder._init_weights` applies `orthogonal_(gain=1.0)` to ALL Linear layers including ReLU hidden. JAX `ORTH_RELU_INIT = orthogonal(scale=sqrt(2))` is the standard He init for ReLU but WRONG for matching PyTorch. Use `ORTH_INIT` (gain=1.0) for encoder hidden layers. This was missed for multiple sessions because the difference is subtle (gain=1.0 vs gain=sqrt(2)) and only visible in the `_init_weights` static method, not the `_build_embedding` function itself.

16. **Terminal transition masking**: PyTorch drops `term==1` rows from the dataset entirely (`valid = term == 0`). JAX `sample_N_batches` can sample terminal transitions. Fix: mask them with `done_mask = 1 - done` in both TD loss (`_td_masked`) and actor loss (Q normalization, BC, Q mean). The ortho loss and z_cov update still include terminal transitions — this is a minor difference left unmasked.

17. **Actor trunk first activation is ReLU, not Tanh**: PyTorch's `Actor.__init__` calls `_build_mlp(hidden*2, act_dim, hidden_dims, activation)` where `activation=nn.ReLU` is the 4th positional arg. In `_build_mlp`, this position is `input_activation` (NOT `activation`), so the trunk's first layer is `Linear -> LayerNorm -> ReLU`, not `Linear -> LayerNorm -> Tanh`. This is likely accidental in the reference (the `activation` arg was meant for hidden layers) but it's what produced 0.98. JAX must use `nnx.relu` as the actor trunk's first activation. The `PredictorEmbedding` heads still use `nnx.tanh` internally — that's correct because `_build_embedding` defaults `input_activation=nn.Tanh`.

18. **max_samples must be -1 (load ALL data)**: PyTorch `get_dataset` loads the full OGBench dataset (~5M transitions). The JAX default was `max_samples=100_000` (100K transitions — 50x less data). This is likely THE biggest single contributor to the JAX 0.42 vs PyTorch 0.98 gap. ALL prior JAX runs used 100K and never matched PyTorch. Fix: `max_samples: int = -1` in params.py. This loads ~5000 episodes × 1000 steps.

19. **adam_eps: PyTorch config vs runtime mismatch**: `TD_JEPA_CONFIG` defines `adam_eps=1e-5` but NEVER passes it to `torch.optim.Adam(...)`. Both optimizers use the torch default `eps=1e-8`. JAX must use `eps=1e-8` in all `optax.adam` calls, NOT the config value of 1e-5.

20. **PyTorch ref was OGBench-only — NOW has DMC support via `train_dmc.py`**: The original `train.py` hardcodes `import ogbench` and only has configs for `pointmaze` and `antmaze`. We wrote `~/Documents/td-jepa/src/train_dmc.py` which adds: (a) ExORL dataset loading from `episode_*.npz` files, (b) DMC env wrapper via `exorl.dmc.make()`, (c) DMC eval with reward relabelling from physics + exp/ls z inference, (d) auto-detect obs/act dims from data. Usage: `python src/train_dmc.py --env_name walker --expl_alg proto --total_steps 1000000`. DMC walker has 10M transitions (10K episodes × 1000 steps) vs OGBench's 1M — takes ~2x longer per run. **CRITICAL: DMC config is VERY different from OGBench** — see pitfall #22.

21. **subprocess.run with capture_output=True is fragile for long runs**: Using `subprocess.run(cmd, capture_output=True, timeout=...)` holds ALL stdout in memory and the log file is only written AFTER the subprocess completes. If the process is killed mid-run (OOM, timeout, manual kill), the log file is empty. For long training runs, use shell redirection (`> $LOG 2>&1`) or `subprocess.run(stdout=logf)` instead. This caused a sweep run to lose all output when the tmux session was killed.

22. **DMC config is VERY different from OGBench — don't reuse antmaze config**: Paper Table 4 specifies completely different architecture/hyperparameters for DMC vs OGBench. Using the antmaze config (w512, 4 layers, gamma=0.99, batch=256, 1M steps) on DMC walker produces ~48 raw return vs paper's 785.2 normalized. The DMC config is: SFs 1024 width/3 layers, state encoder 0 hidden layers/256 width, task encoder 2 layers/256 width/50 dim, actor 256 width/3 layers, gamma=0.98, batch=1024, 2M steps, EMA=0.001, no BC. Both PyTorch and JAX confirmed underperforming with the wrong config — it's a config issue, not a port issue. JAX overrides in `/tmp/tdj_sweep_runner.py`, PyTorch in `train_dmc.py:default_config()`. **On GTX 1070 (8GB), use batch=512 and 1M steps instead of paper's batch=1024 and 2M** — batch=1024 gives only 3 sps (7.5 days), batch=512 gives ~15 sps (18h). Both PyTorch and JAX run in parallel at ~7.9GB VRAM (tight but fits). **RESULT: DMC walker with tuned config peaks at ~700 mean at 400K then REGRESSES to ~607 at 1M. Paper target 785.2 NOT reached. The `run` task (300) is the weak spot. The config fix (not the port) was the key, but the regression at longer training suggests overfitting or LR schedule issues — consider early stopping at 400K or adding LR decay.**

23. **DMC walker zero_shot.csv has DUAL-RUN ambiguity**: `~/Documents/BSynth/.runs/tdj_walker/zero_shot.csv` contains TWO sets of step=1000000 rows — the old wrong-config run (returns 23-123) and the tuned run (returns 485-883). When reading results, disambiguate by checking the return magnitude: ~20-123 = old wrong config, ~300-883 = tuned config. The same file was appended to across runs. Always check return scale, not just step number, when parsing this file.

24. **DMC walker performance regresses after 400K — don't assume more steps = better**: The tuned DMC walker run peaked at ~700 mean at 400K (flip=632, run=344, stand=943, walk=880) but declined to ~607 by 1M. This is NOT noise — the decline is monotonic from 400K→600K→800K→1M. The `run` task drops from 344 to 282 to 299 to 300. For DMC walker, consider early-stopping at the 400K checkpoint or adding LR decay/schedule. This contrasts with OGBench antmaze where performance is stable or improving through 1M.

## Key Findings (Sessions 2026-07-13 through 2026-07-17)

### Run Scripts
- `run_tdj_1m_fulldata_v2.py` — current run script for all-fixes 1M run. Uses standard `main.py` pipeline (load_dataset, train_base_agent, run_zero_shot_eval) with `max_samples=-1`. Outputs to `.runs/tdj_1m_fulldata_v2/`. Run in tmux: `PYTHONUNBUFFERED=1 .venv/bin/python run_tdj_1m_fulldata_v2.py`.
- `run_tdj_1m_fulldata.py` — older script with its own terminal-dropping logic and diagnostics. Superseded by v2 which uses the standard pipeline.

### FINAL RESULTS — JAX no-inv-dyn 0.42, JAX inv-dyn 0.86, PyTorch 0.98 (2026-07-17, pre-new-fixes)
- PyTorch w512 (dual-head s/sz, psi=50, lambda=1.0): 0.98 stable at 1M — ALL 5 tasks working
- JAX inv-dyn (50-dim, batch 1024, D=2): 0.86 at 800K eval — tasks 3/4 working (0.75, 0.90)
- JAX no-inv-dyn (all prior fixes, eval10, n1, Kaiming, flat, w512): 0.42 at 1M — task3=0.15, task4=0.00
- The five prior fixes closed the measurement gap but NOT the no-inv-dyn performance gap
- **Inverse dynamics loss is NOT necessary** — user explicitly confirmed: PyTorch train.py (no inv-dyn) gets 0.98. Inv-dyn was a different path that was explored and ruled out.
- **NEW: All 10 fixes applied + RUN COMPLETE. JAX 0.92 mean at 1M (was 0.42 pre-fixes).**
  - Final 1M eval: task1=0.90, task2=0.90, task3=0.80, task4=1.00, task5=1.00, mean=0.92
  - PyTorch ref: 0.98 at 1M. Gap closed from 0.56 to 0.06.
  - The full-data fix (max_samples=-1, 1M transitions vs 100K) was the biggest single contributor
  - Branch `tdj`, commit `50afc6a`. Run script: `run_tdj_1m_fulldata_v2.py`.
  - Multi-env sweep running in tmux `tdj_sweep`: walker/proto (DMC), antmaze-large-navigate, antmaze-medium-stitch, antmaze-large-stitch. Each 1M steps, ~4-5h per env.
  - Pipeline integration verified: TDJ works with opti_bfm, synthesizer, DMC and OGBench paths (12/12 ad-hoc checks passed).

### The s/sz dual-head actor is THE critical knob
- PyTorch `train.py` with single `_build_mlp` actor: ~0.10 mean at 1M
- PyTorch `train.py` with dual-head s/sz actor (manual mod), w=256: 0.845 peak (noisy, most evals 0.10-0.50)
- PyTorch `train.py` with dual-head s/sz actor, w=512: **0.98 STABLE** at 1M (consistently 0.84-0.98 from 820K)
- JAX with dual-head actor, phi256/psi50, pre-fixes: **0.40-0.60** at 1M
- JAX with all fixes (w512, eval_epi=10, n_seeds=1, Kaiming init, flat dataset, eval_freq=10K): **run in progress**
- The dual-head actor separates the obs embedding from the obs+z embedding, giving the policy network richer task-conditioned features
- **This is not in the original paper code** — it's a manual modification to `networks.py` Actor class
- **The 0.845 with w=256 was noisy, but w=512 made it stable at 0.98 — NOT a lucky spike**

### Weight init on actor s/sz heads matters
- `_build_embedding` in PyTorch does NOT apply orthogonal init (no init loop)
- `_build_mlp` DOES apply orthogonal init explicitly
- The Actor's s/sz heads use `_build_embedding` → default Kaiming uniform init
- The Actor's body uses `_build_mlp` → orthogonal init
- JAX port: `PredictorEmbedding(use_orth_init=False)` for actor heads, `lecun_uniform()` as JAX equivalent of Kaiming

### Obs normalization: NO
- PyTorch `StateEncoder.forward` passes raw obs to `self.net(obs)` — no `(obs-mean)/std`
- Only normalization is LayerNorm inside the network + NormZLayer on output
- Setting `normalize_obs=True` in JAX actively hurts performance (0.10 vs 0.40+ with False)

### Eval convention: train.py does NOT center reward
- `train.py`: `reward = (dist <= tol).float() - 1.0 + shift_reward` = {0, 1}, no centering
- `train_inverse_dynamics.py` Justfile: `--center_reward` centers before lstsq
- Using `center_reward=True` when targeting `train.py` is wrong

### No actor grad clip — CORRECTED
- User initially said "no grad clip for actor that is not in pytorch"
- But final diff confirmed `train.py` DOES have `torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)`
- User then self-corrected: "actually do clip actor grads i was wrong"
- JAX now has `optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr, eps=1e-5))`

### Performance is noisy (especially with w=256)
- task1 bounces between 0.0 and 1.0 across evals (same in PyTorch, w=256)
- task3 is the hardest — often stuck at 0.00, sporadically spikes to 0.5-1.0
- eval every 100K misses spikes that eval-every-10K catches (PyTorch evals 10x more often)
- 2 parallel seeds with only 2 episodes each = high variance → FIXED to 1 seed, 10 episodes
- **The paper reports 14.81% (0.148) for antmaze-medium-navigate proprio** — mean ± SE across seeds
- **PyTorch w=256 was noisy (0.10-0.50, peak 0.845)** but **w=512 was stable (0.84-0.98)** — width matters
- **The 0.845 with w=256 was a peak, not stable. The 0.98 with w=512 is stable.**

### Width=512 (2026-07-14)
- Paper Table 4 confirms encoder/actor width=512 for OGBench proprio (was 256)
- Updated `tdj_phi_hid_dim=512`, `tdj_psi_hid_dim=512`, `tdj_actor_hid_dim=512`
- **Result: 0.40 mean at 1M — NO improvement over w=256.** Width is not the bottleneck.
- Actor std still 0.1 (paper says 0.2) — user deferred this fix

### Noisy eval rollouts (2026-07-13/14)
- Added stddev=0.1, clip=0.3 noise to actor output during eval rollouts (JAX-only, PyTorch is deterministic)
- **Result: 0.40 mean at 1M — NO improvement over deterministic eval.** Noise doesn't help.

### FINAL RESULTS (2026-07-17): PyTorch 0.98, JAX 0.92 at 1M (all fixes applied)
- **JAX tdj_1m_fulldata_v2 (ALL 10 fixes, no inv-dyn): 0.92 mean@1M** — task1=0.90, task2=0.90, task3=0.80, task4=1.00, task5=1.00
- **PyTorch w512 (no inv-dyn, dual-head s/sz, psi=50, lambda=1.0): 0.98 mean@1M** — task1=1.00, task2=1.00, task3=0.90, task4=1.00, task5=1.00 (STABLE from 810K)
- **JAX tdj_1m_invdyn (inv-dyn, 50-dim): 0.86 mean@800K** — task1=0.75, task2=0.90, task3=0.75, task4=0.90, task5=1.00
- **Inv-dyn is NOT necessary** — user explicitly confirmed: PyTorch train.py (no inv-dyn) gets 0.98. Inv-dyn was a different path that was explored and ruled out. The 0.86 JAX inv-dyn result is NOT the target — the target is matching PyTorch train.py without inv-dyn.
- The ten fixes (max_samples=-1, encoder init gain, actor relu activation, adam_eps, terminal masking, eval_epi=10, n_seeds=1, Kaiming init, flat dataset, eval_freq=10K) closed the gap from 0.42 to 0.92. Biggest contributors: full data (100K->1M) and eval_epi (2->10).
- **Paper reports 14.81% (0.148)** for antmaze-medium-navigate proprio — mean ± SE across seeds
- **Actor std: paper says 0.2, code has 0.1** — NOT YET FIXED (user deferred)
- Multi-env sweep running in tmux `tdj_sweep`: walker/proto, antmaze-large-navigate, antmaze-medium-stitch, antmaze-large-stitch
  - **COMPLETED**: antmaze-medium-navigate=0.92, antmaze-medium-stitch=0.86, antmaze-large-navigate=0.56, antmaze-large-stitch=0.42, walker/proto (DMC) returns 23-123
  - Medium mazes close to PyTorch (0.86-0.92 vs 0.98), large mazes lag (0.42-0.56)
  - **JAX antmaze results match or exceed paper Table 1**: mn=92% vs 70.4%, ms=86% vs 61.6%, ln=56% vs 57.2%, ls=42% vs 40.6%
  - **DMC walker underperforms vs paper with WRONG config**: JAX ~48 raw return vs paper 785.2 normalized. PyTorch also underperformed with the same wrong config (~61 at 500K) — config issue not port issue.
  - **DMC walker with PAPER-MATCHED config (Table 4): peaks at ~700 mean at 400K then REGRESSES to ~607 at 1M.** Paper target 785.2 NOT reached. The `run` task (300) is the weak spot; `stand` (884) and `walk` (759) are strong. Tuned config: SFs 1024/3, enc 0 layers/256, task 2/256/50, actor 256/3, gamma=0.98, batch=512 (1024 too slow), 1M steps, EMA=0.001, no BC. Regression suggests overfitting or LR schedule issues — may need shorter training (400K checkpoint) or LR decay.
  - PyTorch ref now supports DMC via `train_dmc.py` (OGBench-only originally). `train_dmc.py` adds ExORL npz loading, DMC env wrapper, reward relabelling eval, auto-detect obs/act dims.

### TWO NEW FIXES found 2026-07-17 (TESTED — 0.92 mean at 1M)

**6. Encoder hidden ReLU layer init gain — FIXED**
- PyTorch `StateEncoder._init_weights` applies `nn.init.orthogonal_(m.weight)` with default gain=1.0 to ALL Linear layers, including hidden ReLU layers (networks.py:215)
- JAX was using `ORTH_RELU_INIT = orthogonal(scale=sqrt(2))` for hidden ReLU layers — standard He init but WRONG for matching PyTorch
- FIX: Changed to `ORTH_INIT` (gain=1.0) in `agents/networks.py:49`

**7. Terminal transition masking — FIXED**
- PyTorch drops terminal transitions entirely from the dataset: `valid = term == 0` (train.py:83-84)
- JAX was including terminal transitions in training (sample_N_batches can sample t where term==1)
- FIX: Added `_td_masked()` that zero-weights terminal transitions in TD loss. Pass `done_mask = 1 - done` to `_critic_loss`. Also mask actor loss (Q normalization, BC, Q mean) using `n_valid = sum(done_mask)`.
- Files: `agents/td_jepa.py` — `_td_masked`, `_critic_loss`, `train_step`, `_actor_loss`

### FIVE NEW FIXES found 2026-07-17 session 2 (external diff analysis, ALL TESTED)

**8. max_samples: 100K -> -1 (full dataset) — BIGGEST SINGLE CONTRIBUTOR — FIXED**
- PyTorch loads ALL transitions. JAX default was `max_samples=100_000` (10x less data).
- Actual dataset size: 1M transitions (1000 episodes × 1000 steps), NOT ~5M as docstring claimed.
- ALL prior JAX runs used 100K and never exceeded 0.60 mean. This alone could explain the gap.
- FIX: `max_samples: int = -1` in params.py:35

**9. Actor trunk first activation: tanh -> relu — FIXED**
- PyTorch `Actor` passes `activation=nn.ReLU` as 4th positional to `_build_mlp`, landing on `input_activation`. Trunk is `Linear->LN->ReLU`, not Tanh.
- JAX `TDJActor` had `nnx.tanh` — changed to `nnx.relu`.
- FIX: `agents/networks.py:301` — `nnx.tanh` -> `nnx.relu`

**10. adam_eps: 1e-5 -> 1e-8 — FIXED**
- PyTorch config defines `adam_eps=1e-5` but never passes it to `torch.optim.Adam`. Both use default `eps=1e-8`.
- JAX had `eps=1e-5` in all 5 optimizers. Changed to `eps=1e-8`.
- FIX: `agents/td_jepa.py` — all `optax.adam(..., eps=1e-8)`

### Noisy eval rollouts — TRIED then REMOVED (2026-07-13/14)

User requested adding noise to eval rollouts. Tried stddev=0.1, clip=0.3.
Result: 0.40 mean — NO improvement. PyTorch eval is deterministic (`agent.act()` — no noise).
**Reverted to `noisy=False` to match PyTorch.** Noise was actively hurting task3
(narrow corridor) where precise navigation is needed.

### PyTorch w=512 hits 0.98 STABLE — NOT a lucky spike (2026-07-14/15)

**PyTorch w=512 (paper Table 4 widths) hit 0.98 mean at 1M** — consistently 0.84-0.98 from
810K onward. ALL 5 tasks working (task3: 0.7-1.0, task4: 0.8-1.0). This is STABLE, not a
lucky spike. The earlier w=256 PyTorch was noisy (0.10-0.50 with a 0.845 peak) — width 512
made it stable. The user confirmed: "i think that 0.98 was not lucky."

**JAX w=512 scored only 0.40** before all fixes were applied. task3/4 at 0.00.

### ROOT CAUSE FOUND: JAX 0.40 vs PyTorch 0.98 gap (2026-07-15)

Systematic diff revealed FIVE differences. All now FIXED:

**1. EVAL EPISODES: 2 (JAX) vs 10 (PyTorch) — BIGGEST IMPACT — FIXED**
- `eval_epi = 2` gives {0, 0.5, 1.0} only. PyTorch uses 10 → {0, 0.1, ..., 1.0}
- task3=0.00 was actually succeeding 10-20% but 2 eps missed it every time
- FIX: `eval_epi = 10` in params.py

**2. ACTOR S/SZ INIT: orthogonal (JAX) vs Kaiming uniform (PyTorch) — FIXED**
- PyTorch `_build_embedding` does NOT apply orthogonal init
- Actor class does NOT call `self.apply(weight_init)`
- `self.s`/`self.sz` use default Kaiming uniform; only `self.net` (from `_build_mlp`) has orthogonal
- FIX: `use_orth_init=False` (lecun_uniform as JAX Kaiming equivalent)

**3. DATASET: episode-segmented with reshape bug (JAX) vs flat array (PyTorch) — FIXED**
- PyTorch: flat (99900, 29) array, uniform sampling via `buffer.sample_batch(batch_size)`
- JAX was reshaping to (99, 1001) episodes, creating wrong (s,s') pairs at boundaries
- User initially declined ("actually not 3") then accepted in next round ("fix 1,2,3")
- FIX: reshape to (1, n_valid, obs_dim) for sample_N_batches compatibility

**4. N_SEEDS: 2 (JAX) vs 1 (PyTorch) — FIXED**
- 2 seeds averaged → bad seed drags down mean. PyTorch runs 1 seed.
- FIX: `tdj_n_seeds = 1` in params.py and run script

**5. EVAL FREQUENCY: 100K (JAX) vs 10K (PyTorch) — FIXED**
- 100K eval misses transient spikes. PyTorch catches them at 10K.
- FIX: `pretrain_zs_eval_freq = 10_000`

Other confirmed matches (NOT differences):
- Adam: 1 Adam/4 groups (PyTorch) vs 4 separate Adams (JAX) — equivalent, Adam moments are per-param
- LayerNorm eps: both 1e-5
- NormZLayer: both sqrt(d) * F.normalize
- Orthogonal init gain: both 1.0
- EMA: both tgt = (1-tau)*tgt + tau*src with tau=0.005
- z_cov: both (1-tau)*old + tau*(psi.T@psi/B)
- TD targets: both per-head, both use sample_action(stddev=0.1) for bootstrap
- Actor loss: both -Q/|Q|.mean() + 0.3*BC, both grad clip 1.0
- Eval: both deterministic, both reward={0,1}, both lstsq fit_reward

## Verification

Use ad-hoc scripts at `/tmp/hermes-verify-*.py` with `JAX_PLATFORMS=cpu` to avoid GPU contention with live training:
```python
jax.config.update("jax_platforms", "cpu")
```
Check: param values, agent construction, forward pass shapes, train_step metrics keys, act() + fit_reward() shapes.

## Multi-Env Sweeps

For evaluating TD-JEPA across multiple environments (DMC walker, antmaze variants):

- Use a **bash script** in tmux, NOT Python subprocess. `subprocess.run` with `capture_output=True` holds all stdout in memory — the log file is only written after the subprocess completes, so if it gets killed mid-run you lose everything. Use `stdout=logf` or shell redirection (`> $LOG 2>&1`) instead.
- Set `subprocess` timeout to at least 14400s (4h) per env if using Python. Bash scripts don't need timeouts.
- DMC walker dataset: 10K episodes × 1000 steps = 10M transitions (vs OGBench's 1M). Takes longer to load and train.
- OGBench datasets download automatically on first use (~234MB per env). Pre-download with `ogbench.make_env_and_datasets(env_name, dataset_only=True)` to avoid download time during sweep.
- Each 1M-step run takes ~4-5h on GTX 1070 for OGBench, ~18h for DMC walker (batch=512, 10M transitions).
- **Run sequentially, NOT in parallel.** Two processes on one 8GB GTX 1070 drops throughput from 15→3 sps (GPU contention). Sequential with a bash script in tmux is the right approach.
- Sweep script: `run_sequential_sweep.sh` — bash script that runs 4 envs sequentially in tmux. Usage: `tmux new-session -d -s sweep "bash run_sequential_sweep.sh"`.
- Runner script: `/tmp/tdj_sweep_runner.py` — per-env script that loads data, trains, evals. Handles both DMC and OGBench with config branching. Writes to `.runs/tdj_<env>/`.
- Monitor with: `tail .runs/tdj_<env>.log` or `tmux attach -t sweep`.
- **Set a cron job (every 2h) to check progress** instead of polling manually. User explicitly said "are you burning tokens watching?" — don't sit in sleep loops. Cron jobs on CLI are local-only (no delivery channel) — check with `cronjob action='list'` or let the user ask "progress?".

## CsvLogger / save_stage Integration (2026-07-17)

TDJ's `train_tdj` was using its own local `CsvLogger` and `save_stage` classes that didn't match the shared ones used by FB and other agents. Fixed to use:
- `from core.logger import CsvLogger, mean_seed_columns, metric_row` — shared logger with fixed columns, `logger.log(step, row)`, `logger.close()`
- `from core.checkpointing import save_stage` — `save_stage("joint", agents, ckpt_dir)` (was `save_stage(agents, "tdj_final.pkl")`)
- `training.py` updated: `file_stem="joint"` (was `"tdj_final"`) to match checkpoint naming
- Metrics file renamed from `tdj_metrics.csv` to `joint_train.csv` (consistent with FB)
- Added tqdm progress bar and periodic checkpointing (`args.checkpoint_freq`, default 50K)
- Uses `args.log_freq` (default 10K) for logging frequency
- Per-seed `metric_row` logging with `mean_seed_columns` (matches FB pattern)

## Paper Table 1 Comparison (2026-07-17)

JAX TDJ results vs paper TD-JEPA (proprioception, success rate %):

| Env | JAX TDJ (ours) | Paper TD-JEPA | Match? |
|-----|----------------|---------------|--------|
| antmaze-mn | 92% | 70.40 ± 3.72 | JAX > paper |
| antmaze-ms | 86% | 61.56 ± 4.53 | JAX > paper |
| antmaze-ln | 56% | 57.20 ± 4.25 | JAX ≈ paper |
| antmaze-ls | 42% | 40.60 ± 2.51 | JAX ≈ paper |
| walker (DMC) | ~607 raw (1M) | 785.2 normalized | below (regression after 400K peak of ~700) |

Paper DMC returns are normalized by 1000 (max achievable). The DMC walker tuned run peaks at ~700 raw at 400K but regresses to ~607 at 1M — the `run` task (300) is the bottleneck. The config fix (wrong config → tuned config) took returns from ~48 to ~700, but the paper's 785.2 is still not reached. The regression after 400K suggests overfitting or LR schedule issues — consider early stopping or LR decay for DMC.

**Paper Table 4 DMC config (proprioception):**
- SFs: 1024 width, 3 hidden layers
- State encoder: 0 hidden layers, 256 width, 256 dim
- Task encoder: 2 hidden layers, 256 width, 50 dim
- Actor: 3 hidden layers, 256 width
- gamma=0.98, batch=1024, 2M steps, EMA=0.001, no BC, pgoal=0.5

## Shared Infrastructure Integration (2026-07-17)

TDJ's `train_tdj` now uses the shared `CsvLogger` + `save_stage` from `core.logger` / `core.checkpointing`, matching the FB agent pattern. This was requested by the user: "can we integrate with csvlogger and save_stage like in the other BFMS under agents?"

Changes:
- Removed local `CsvLogger` class and `save_stage` function from `agents/td_jepa.py`
- Imports: `from core.logger import CsvLogger, mean_seed_columns, metric_row` and `from core.checkpointing import save_stage`
- Metrics file: `joint_train.csv` (was `tdj_metrics.csv`) — consistent with FB
- Checkpoint: `save_stage("joint", agents, ckpt_dir)` → `joint.pkl` (was `tdj_final.pkl`)
- `training.py`: `file_stem="joint"` (was `"tdj_final"`)
- Per-seed `metric_row` logging with `mean_seed_columns(aux_keys)` (matches FB)
- Added `tqdm` progress bar, periodic checkpointing at `args.checkpoint_freq` (default 50K)
- Uses `args.log_freq` (default 10K) for logging frequency
- Smoke tested end-to-end: `joint_train.csv` + `joint.pkl` both created, zero-shot eval runs

## Workflow Preferences (user)

- Run long training in tmux, NOT background terminal — survives disconnects
- Use `PYTHONUNBUFFERED=1` so output isn't buffered through tee
- Don't poll every 3 min — wastes tokens. User will ask "progress?" when they want an update. For long runs, set a cron job (every 2-4h, not 3 min) or wait for the user to ask. User explicitly said "are you burning tokens watching?" — this is a strong signal to stop polling and let cron or the user drive check-ins.
- When delegating to subagents, the user may say "make sure the subagent doesn't change anything" — this means the delegation should be read-only/analysis only, no file edits. Honor this immediately.
- "make it work" not "make it plan" — deliver running code, not descriptions
- When the user says "kill it", kill immediately, don't argue
- User prefers direct edits over subagent delegation when context is already loaded
- Always re-read files after subagent edits — they can truncate method bodies
- User blocks repeated `-c` flag python scripts — don't loop on `python -c` commands. If blocked, switch to a different approach (read source, write a temp script, use a different tool).