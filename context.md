# TD-JEPA v3 — Project Context

## What This Is

A faithful reimplementation of TD-JEPA (from facebookresearch/td_jepa / arXiv 2510.00739) for the `pointmaze-medium-navigate-v0` OGBench task. Goal: achieve non-zero zero-shot success rates.

## Files

| File | Description |
|------|-------------|
| `td_jepa3.py` | V3 model (6.03M params). Faithful reimplementation of metamotivo reference. |
| `train3.py` | V3 training script. batch_size=256, lr=1e-4, gamma=0.99, bc_coeff=0.3. |
| `td_jepa2.py` | V2 (abandoned — q_mean stuck near 0, 0% zero-shot) |
| `train2.py` | V2 training script |
| `td_jepa.py` | Original buggy implementation (8+ cataloged bugs) |
| `train.py` | Original training script |
| `/tmp/td_jepa/` | Cloned reference metamotivo repository |

## Architecture (td_jepa3.py)

Matches reference `metamotivo/agents/td_jepa/`:
- **Encoders** (BackwardMap): 4 hidden layers (512), 2-layer embedding with dim reduction (512→256), L2 normalization, output dim=50
- **Predictors** (ForwardMap): 2 hidden layers (512), 2-layer embedding with dim reduction, dual input heads:
  - `embed_sa(cat[obs, action])` → input dim = d_z + action_dim = 52
  - `embed_sz(cat[obs, z])` → input dim = d_z + d_z = 100
  - Ensemble of 2 members via `nn.ModuleList`
- **Actor**: 2 hidden layers (512), embedding layers=2, with separate `embed_s(obs)` and `embed_z(z)` pathways
- **Separate optimizers**: phi_enc, psi_enc, phi_pred, psi_pred, actor (all Adam lr=1e-4)
- **Target networks**: EMA with tau_enc=0.005, tau_pred=0.005
- **z sampling**: Mixture of psi(next_obs) goals (whitened) and uniform sphere, p_goal=0.5
- **Observation normalizer**: Identity (matching OGBench proprio config — no normalization)
- **No image encoder**: Raw 2D observations go directly to MLP encoders

## Bugs Found & Fixed (chronological)

### V2 bugs (caused 0% zero-shot):
1. **Architecture mismatch**: V2 used 1-layer embeddings, reference uses 2-layer with dim reduction
2. **z sampling source**: V2 used permuted current_obs, reference uses next_obs via psi
3. **Single optimizer**: V2 used 1 optimizer, reference uses 5 separate optimizers
4. **Actor architecture**: V2 had simple MLP, reference has embed_s + embed_z pathways
5. **ForwardMap argument order**: Signature `(obs, z, action)` but calls were `net(h, a, z)` → swapped z/action. Fixed with keyword args.

### V3 bugs (caused exploding td_loss):
6. **`batch_discount` shape bug** (CRITICAL): `discount` was `(B,)` instead of `(B, 1)`. Broadcasting `(B,) * (B, 50)` produced `(B, B, 50)` instead of `(B, 50)`, making TD targets completely wrong → positive feedback loop → exploding Q values and td_loss.
7. **Per-sample discount**: Used constant `self.gamma=0.99` instead of per-sample `batch["discount"]` (0 for terminal transitions). Now uses `batch_discount` with shape `(B, 1)`.
8. **z_cov update timing**: Was inside `_sample_z()` using next_obs goals; moved to `_update_z_stats()` called after backward with `psi_s.detach()` matching reference line 346.
9. **Duplicate `return z`**: Patching artifact left orphaned `return z` in `_update_z_stats()` — removed.
10. **Gradient clipping**: Added `clip_grad_norm_` with `max_norm=10` for both representation and actor updates (reference doesn't use explicit clipping, but this helps stability).

## Training Progress

### V1 (original code): 0% zero-shot at every eval (20K–240K). Bugs: inverted BC normalization, wrong dimensions, etc.
### V2 (rewrite): 0% zero-shot at 20K, 40K, 60K. q_mean stuck near 0 (-0.13), td_loss flat at ~102 (random baseline).
### V3 first run (z/action swap bug): td_loss exploded 32→1271, q_mean +13→+307 in 14K steps.
### V3 second run (keyword args, constant gamma): td_loss 34→205→461→879→... still exploding. Same pattern.
### V4 (all fixes including batch_discount shape): **Running now** in tmux `tdjepa`. Early results at step 4000: td_loss=205, q_mean=+50. **Still showing explosion pattern.**

```
tmux capture-pane -t tdjepa -p -S -30
tail -20 train3_v4_log.txt
```

## Key Remaining Differences from Reference

1. **Shared-embedding ensemble**: Reference uses `DenseParallel` where ensemble members share embedding layers but have separate body parameters. Our `nn.ModuleList` uses fully independent ForwardMap networks. This reduces implicit regularization and could contribute to instability.

2. **Reward-based z inference at eval**: Reference uses `reward_inference()` with least-squares (10K samples) to compute z from goal rewards. Our eval uses `psi(goal)` directly. This could give lower success rates even with a correctly trained model.

3. **`actor_use_full_encoder=False`**: Matches OGBench proprio config.

## What To Try If V4 Still Fails

1. **Stronger gradient clipping**: Reduce `max_grad_norm` from 10 to 1 or 5
2. **Lower learning rate**: Try lr=3e-5 instead of 1e-4
3. **Shared-embedding ensemble**: Implement `DenseParallel`-style forward pass where `embed_sa` and `embed_sz` are shared across ensemble members, only body layers differ
4. **Warm-up**: Add LR warmup for first 1-5K steps
5. **Actor Q normalization**: Reference divides actor loss by `|Qs|.mean()` — verify this is implemented correctly
6. **Predictor output normalization**: Add L2 normalization to predictor outputs (not in reference but could help stability)
7. **Direct reference comparison**: Run the actual metamotivo package on pointmaze to confirm it works, then binary-search for the difference

## Key Hyperparameters (matching reference)

| Parameter | Value |
|-----------|-------|
| batch_size | 256 |
| lr | 1e-4 (all optimizers) |
| gamma | 0.99 |
| bc_coeff | 0.3 |
| phi_ortho_coef | 1.0 |
| psi_ortho_coef | 1.0 |
| encoder_target_tau | 0.005 |
| predictor_target_tau | 0.005 |
| n_hidden_enc | 4 |
| n_hidden_pred | 2 |
| n_hidden_actor | 2 |
| n_embedding_pred | 2 |
| n_embedding_actor | 2 |
| hidden_dim | 512 |
| d_z | 50 (both phi and psi) |
| p_goal | 0.5 |
| scale_train_goals | True |
| z_whitening_decay | 0.01 |
| stddev_clip | 0.3 |
| actor_std | 1.0 |
| n_ensemble | 2 |

## Reference Code Structure

```
/tmp/td_jepa/metamotivo/agents/td_jepa/
├── agent.py          # Update logic, sampling, evaluation
├── model.py          # Network definitions, target networks
└── nn_models.py      # BackwardMap, ForwardMap, SimpleActorArchi, DenseParallel
```

Key reference update flow (agent.py):
1. `obs_normalizer(obs)`, `obs_normalizer(next_obs)` → update stats, then normalize (Identity for proprio)
2. `encode_image(obs, next_obs)` → for proprio, returns raw obs (Identity encoder)
3. `z = sample_mixed_z(train_goal=psi_next_obs).clone()` → goals from online psi(next_obs)
4. Representation update: TD loss + ortho loss, backprop, step 4 optimizers
5. `_update_z_stats(psi_enc)` → z_cov EMA using online psi(current_obs), AFTER backward
6. Actor update: detached phi(obs), Q normalization by |Qs|.mean(), bc_coeff*bc_loss
7. EMA target updates for all 4 target networks