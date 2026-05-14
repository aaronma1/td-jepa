# TD-JEPA v2 Bug Analysis: Comparison with metamotivo Reference

## Executive Summary

Zero-shot success at 0% despite stable losses. Six bugs found by comparing with the
official metamotivo reference (https://github.com/facebookresearch/td_jepa). Two are
critical, two are high impact, two are medium.

---

## BUG 1 (CRITICAL): Actor Loss — BC normalization is inverted

### Reference (agent.py lines 426-447):
```python
actor_loss = -Q.mean()
bc_error = F.mse_loss(actor_action, action)
bc_loss = self.cfg.train.bc_coeff * bc_error
actor_loss = (actor_loss / Qs.abs().mean().detach()) + bc_loss
# RESULT: (-Q/|Q|) + bc_coeff * MSE
```

### Our v2 (td_jepa2.py lines 370-382):
```python
actor_loss = -q.mean()
bc = F.mse_loss(mu, act)
scale = q.detach().abs().mean().clamp_min(1e-3)
actor_loss = actor_loss + self.bc_alpha * bc / scale
# RESULT: -Q + bc_coeff * MSE / |Q|
```

### Impact:
In the reference, the Q gradient is divided by |Q| to keep it bounded, while BC
is kept at face value. In our code, Q is left at raw scale while BC is divided by
|Q|. With |Q| ≈ 0.19, our BC term is amplified ~5.3x while Q is tiny at -0.19.

The actor essentially becomes almost-pure BC, which prevents it from learning
goal-directed behaviour via the Q signal. The actor optimizes `mean_action ≈ data_action`
instead of `max_a Q(s,z,a)`, so it can never solve novel tasks.

### Fix:
```python
# Change actor_loss computation to match reference
q_abs_mean = q.detach().abs().mean().clamp_min(1e-3)
actor_loss = (-q.mean() / q_abs_mean)  # normalize Q term
if self.bc_alpha > 0:
    mu = self.actor.mean(actor_in, z)
    bc = F.mse_loss(mu, act)
    actor_loss = actor_loss + self.bc_alpha * bc  # BC at face value
```

ALSO: the reference uses the SAMPLED action for Q computation and for BC loss,
not the mean action. The mean action is only used for evaluation. The reference
code does:
```python
actor_action = dist.sample(clip=0.3)  # sampled action
preds = self._model._phi_predictor(phi_enc, z, actor_action)  # Q with sampled action
Qs = (preds * z).sum(-1)
...
bc_error = F.mse_loss(actor_action, action)  # BC also uses sampled action
```

Our code uses `.mean()` for BC and `.sample()` for Q. For BC, we should use the
sampled action too (though using mean is also valid and both are used in practice).

---

## BUG 2 (HIGH): Encoder/Predictor depths are SWAPPED

### Reference (launch_td_jepa_ogbench.py):
```python
"phi_mlp_encoder": {"hidden_layers": 4, "hidden_dim": 512}  # DEEP
"psi_mlp_encoder": {"hidden_layers": 4, "hidden_dim": 512}  # DEEP
"phi_predictor": {"hidden_layers": 2, "hidden_dim": 512}      # SHALLOW
"psi_predictor": {"hidden_layers": 2, "hidden_dim": 512}      # SHALLOW
```

### Our v2 defaults (td_jepa2.py + train2.py):
```python
n_hidden_enc=2   # SHALLOW (should be 4)
n_hidden_pred=4   # DEEP (should be 2)
```

### Impact:
Encoders need to be expressive enough to map raw observations to useful representations.
With only 2 hidden layers, encoders may underfit. Predictors with 4 hidden layers may
overfit to noise. The reference design philosophy is: deep encoders (to learn good
representations from raw obs), shallow predictors (to learn simple transitions in
latent space).

### Fix:
Change defaults to `n_hidden_enc=4, n_hidden_pred=2` or pass them explicitly in train2.py.

---

## BUG 3 (MEDIUM): Z-whitening uses diagonal instead of full inverse covariance

### Reference (agent.py lines 166-167):
```python
inv_cov = torch.inverse(self._model._z_cov + 1e-6 * torch.eye(*self._model._z_cov.size(), device=z.device))
goals = torch.matmul(goals, inv_cov)
```

### Our v2 (td_jepa2.py lines 283-284):
```python
scale = (self._z_cov.diag().clamp_min(1e-6)).rsqrt()
z = z * scale.unsqueeze(0) * math.sqrt(self.d_z)
```

### Impact:
The reference does full Cholesky-like whitening (z @ Σ^{-1}), which decorrelates all
dimensions. Our diagonal whitening only normalizes the variance of each dimension
without decorrelating them. If off-diagonal correlations in z-space grow, diagonal
whitening won't undo them, leading to ill-conditioned representations.

Full whitening also rescales differently. If cov has eigenvalues [100, 1, 1, ...],
diagonal whitening scales all dimensions to unit variance, but doesn't rotate the
coordinate system. Full inverse covariance does both.

### Fix:
```python
# In _sample_z, replace with full inverse covariance whitening
inv_cov = torch.inverse(self._z_cov + 1e-6 * torch.eye(self.d_z, device=z.device))
goals = torch.matmul(goals, inv_cov)
# After whitening, project back to sphere
goals = self.project_z(goals)  # sqrt(d) * normalize
```

Actually, looking more carefully, the reference project_z is:
```python
def project_z(self, z):
    if self.cfg.archi.norm_z:
        z = math.sqrt(z.shape[-1]) * F.normalize(z, dim=-1)
    return z
```

So after full whitening, the reference also re-normalizes back to the sphere. This
means whitening is primarily about decorrelating, and then re-projecting to the sphere
ensures the vector norm is preserved.

For our diagonal whitening + renormalization, the effect is the same IF the covariance
is already diagonal (which ortho_reg tries to achieve). But during early training when
covariance is not yet diagonal, full whitening is important.

---

## BUG 4 (MEDIUM): Ensemble gradient flow differs

### Reference:
```python
phi_preds = self._model._phi_predictor(phi_enc, z, action)  # (2, B, d)
phi_tdjepa_loss = (phi_preds - td_target_phi).pow(2).sum(-1).mean()
# Gradient flows to EACH ensemble member through its own prediction
```

### Our v2:
```python
pred_phi = self._predictor_mean(self.T_phi, self.T_phi_t, phi_s, act, z)
# Takes mean of ensemble predictions FIRST, then computes loss
# Gradient flows to ALL members equally through the mean
```

### Impact:
In the reference, each ensemble member gets its own gradient signal from its individual
prediction error. This is standard ensemble training. In our code, taking the mean
before the loss means gradient is distributed equally regardless of individual quality.

This could reduce diversity of the ensemble and make it less effective as a target.

### Fix:
Compute loss per-ensemble member and average:
```python
preds = [net(phi_s, act, z) for net in self.T_phi]
all_preds = torch.stack(preds, dim=0)  # (n_ensemble, B, d)
td_loss_phi = (all_preds - tgt_phi.unsqueeze(0)).pow(2).sum(-1).mean()
```

---

## BUG 5 (LOW-MEDIUM): Action sampling difference

### Reference uses TruncatedNormal with noise clipping:
```python
# stddev_clip = 0.3 (default)
actor_action = dist.sample(clip=0.3)  # clips noise to [-0.3, 0.3] before adding
# Then _clamp to [-1+eps, 1-eps]
```

### Our v2:
```python
a_pi = self.actor.sample(actor_in, z)  # mu + randn*0.2, then .clamp(-1, 1)
```

### Impact:
Minor. With std=0.2, noise is mostly within [-0.3, 0.3] anyway. But the reference
also does `_clamp` which clips to `[low+eps, high-eps] = [-0.999999, 0.999999]` with
a straight-through estimator, which affects gradients differently from `.clamp(-1, 1)`.

---

## BUG 6 (INFORMATIONAL): Evaluation approach differs

### Reference uses reward inference for evaluation:
The OGBench evaluation code does NOT use `infer_z_from_goal`. Instead it uses
`reward_inference` which performs linear regression of task-specific rewards onto
psi(observations), then projects the resulting z to the unit sphere.

This gives a z vector that's specifically tuned for the task's reward structure,
rather than just being `psi(goal_obs)` which might not be the optimal z for the
task.

### Our v2 uses simple goal encoding:
```python
z = agent.infer_z_from_goal(g_t)  # = psi(goal)
```

### Impact:
For simple goal-reaching where reward is 1[at goal], `psi(g)` should be a reasonable z.
But for OGBench's multi-task evaluation, reward inference from dataset samples may be
more effective. This is likely not the primary cause of 0% success but could improve
performance once other bugs are fixed.

---

## Additional Differences (not bugs)

1. **Observation normalization**: The reference has an `obs_normalizer` (learned running
   stats). For OGBench proprio, it's Identity, but it could be important for other envs.

2. **Actor architecture**: The reference actor has `embedding_layers=2` (default) and
   `hidden_layers=1` (default, not overridden in OGBench config). The hidden dimension
   is 512. The `simple_embedding` with 2 layers produces:
   `[Linear(in, 512), LN, Tanh, Linear(512, 256), ReLU]`
   Then `policy` with 1 hidden layer: `[Linear(512, 512), ReLU, Linear(512, action_dim)]`
   Plus tanh on output.
   
   Our actor with `n_hidden_actor=4`:
   `[EmbeddingBlock(obs, 512), EmbeddingBlock(z, 512), MLP(1024, 512, action_dim, 4)]`
   That's much deeper. Reference actor has 1 hidden layer in the body, we have 4.

3. **Default hyperparameters differ**:
   - Reference: `batch_size=256`, `lr_phi=lr_psi=lr_predictor=lr_actor=1e-4`
   - Our v2: `batch_size=1024`, `lr=3e-4` (single optimizer for all repr params)
   - The reference uses SEPARATE optimizers for phi, psi predictors, phi encoder, psi encoder, and actor
   - Our v2 uses a SINGLE optimizer for all repr params and a separate one for actor

4. **Discount handling**: Reference uses `discount * ~terminated`, ours drops terminal
   transitions. Both approaches are valid for offline RL with fixed datasets.

---

## Priority Fix Order

1. **BUG 1** (critical): Fix actor loss BC normalization — swap which term is normalized
2. **BUG 2** (high): Fix encoder/predictor depths — encoders should be deep (4), predictors shallow (2)
3. **BUG 3** (medium): Use full inverse covariance whitening instead of diagonal-only
4. **BUG 4** (medium): Fix ensemble loss to compute per-member then average
5. Consider implementing reward inference for evaluation
6. Consider matching actor depth to reference (1 hidden layer in body, not 4)