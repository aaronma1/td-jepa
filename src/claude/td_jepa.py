"""TD-JEPA reimplementation matching the official facebookresearch/td_jepa code.

Reference: https://github.com/facebookresearch/td_jepa  (metamotivo/agents/td_jepa/)

Key faithful pieces of the reference that the earlier sketch got wrong:

* Both phi and psi encoders end in an L2-normalisation that rescales the output
  to ``sqrt(d_z)``.  This is what keeps the orthonormality regulariser bounded:
  with ‖phi(s)‖ constant, ``-diag(Cov).mean()`` is a constant ``-d_z`` and only
  the off-diagonal term is an active learning signal.  Without this, the diag
  term has no lower bound and the regulariser explodes.
* phi_dim = psi_dim = 50 by default.  The asymmetry is about *content*, not
  dimensionality.
* The TD loss is ``(pred - target).pow(2).sum(-1).mean()`` -- summed over the
  feature dimension, averaged over batch.  Not ``F.mse_loss`` which would
  additionally divide by d_z and make the regulariser dominate.
* Predictor architecture: two embedding heads, ``(state, z)`` and
  ``(state, action)``, concatenated into a body.  State appears in *both*
  branches.  Ensemble of 2 parallel predictors, averaged for targets.
* Actor BC normalisation: ``actor_loss = (-Q.mean() / |Q|.detach()) + bc_coeff * bc_mse``.
  The ``-Q/|Q|`` term has magnitude ~1 so ``bc_coeff`` in [0.3, 3.0] is sensible.
* Running ``z_cov`` (EMA of ``z.T @ z / batch``) used to whiten goal-sampled z's
  before projecting onto the sphere.
* Actor input is the raw observation (proprio) for OGBench, not phi(s).

The single-predictor / Gaussian-actor / TruncatedNormal version implemented here
matches ``TDJEPAAgent`` in the reference (the DMC default).  For OGBench
manipulation tasks the reference switches to ``TDJEPAFlowBCAgent`` with a
noise-conditioned flow-matching actor; that's a larger change and not included
here, but the rest of the agent is identical.
"""

from __future__ import annotations

import copy
import math
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as pyd
from torch.distributions.utils import _standard_normal

# --------------------------------------------------------------------------- #
# Distributions / helpers                                                      #
# --------------------------------------------------------------------------- #


class TruncatedNormal(pyd.Normal):
    """Reparameterisable truncated Gaussian on [-1, 1].

    Matches metamotivo/nn_models.py::TruncatedNormal.  Critical for SVG-style
    policy gradient: the ``x = x - x.detach() + clamped_x.detach()`` trick
    lets gradients flow as if the action were unclamped, while the forward
    pass actually clamps.
    """

    def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6):
        super().__init__(loc, scale, validate_args=False)
        self.low = low
        self.high = high
        self.eps = eps

    def _clamp(self, x):
        clamped_x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
        return x - x.detach() + clamped_x.detach()

    def sample(self, clip=None, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
        eps = eps * self.scale
        if clip is not None:
            eps = torch.clamp(eps, -clip, clip)
        return self._clamp(self.loc + eps)


class Norm(nn.Module):
    """L2-normalise then rescale to have norm ``sqrt(d)`` (the FB convention)."""

    def forward(self, x):
        return math.sqrt(x.shape[-1]) * F.normalize(x, dim=-1)


def weight_init(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


# --------------------------------------------------------------------------- #
# Encoder / Predictor / Actor                                                  #
# --------------------------------------------------------------------------- #


def simple_embedding(in_dim: int, hidden_dim: int, n_layers: int) -> nn.Sequential:
    """LN-Tanh head + ReLU MLP, output dim = hidden_dim // 2 (matches metamotivo)."""
    assert n_layers >= 2, "must have at least 2 embedding layers"
    seq = [nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Tanh()]
    for _ in range(n_layers - 2):
        seq += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
    seq += [nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU()]
    return nn.Sequential(*seq)


class BackwardMap(nn.Module):
    """State/task encoder.  Linear-LN-Tanh head, then ReLU hidden, then Linear + Norm."""

    def __init__(
        self, obs_dim: int, z_dim: int, hidden_dim: int = 512, n_hidden: int = 4
    ):
        super().__init__()
        seq = [nn.Linear(obs_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Tanh()]
        for _ in range(n_hidden - 1):
            seq += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        seq += [nn.Linear(hidden_dim, z_dim), Norm()]
        self.net = nn.Sequential(*seq)

    def forward(self, x):
        return self.net(x)


class ForwardMap(nn.Module):
    """Predictor T(state, z, action) -> out, with two embedding heads (s,z) and (s,a).

    Outputs ``num_parallel`` predictions (ensemble).  Forward returns a tensor of
    shape ``(num_parallel, batch, out_dim)``.
    """

    def __init__(
        self,
        in_dim: int,
        z_dim: int,
        action_dim: int,
        out_dim: int,
        hidden_dim: int = 512,
        n_hidden: int = 2,
        n_embed: int = 2,
        num_parallel: int = 2,
    ):
        super().__init__()
        self.num_parallel = num_parallel
        self.embed_z = simple_embedding(in_dim + z_dim, hidden_dim, n_embed)
        self.embed_sa = simple_embedding(in_dim + action_dim, hidden_dim, n_embed)
        body = []
        for _ in range(n_hidden):
            body += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        body += [nn.Linear(hidden_dim, out_dim)]
        self.body = nn.Sequential(*body)

    def forward(
        self, h: torch.Tensor, z: torch.Tensor, a: torch.Tensor
    ) -> torch.Tensor:
        # Stack K parallel copies (different params, same inputs) by sharing inputs
        # and running each parallel head separately. For simplicity we run the
        # ensemble as K forward passes through the same module replicated K
        # times; cleaner than the DenseParallel trick from the reference and
        # matches its behaviour exactly when ``num_parallel`` is small.
        e_z = self.embed_z(torch.cat([h, z], dim=-1))
        e_sa = self.embed_sa(torch.cat([h, a], dim=-1))
        x = torch.cat([e_sa, e_z], dim=-1)
        out = self.body(x)
        # We're not actually using an ensemble in this simplified version; the
        # outer code can wrap this module in nn.ModuleList for real ensembling.
        return out


class ForwardMapEnsemble(nn.Module):
    """Wraps ``num_parallel`` independent ForwardMap modules; output stacked along dim 0."""

    def __init__(self, num_parallel: int = 2, **kwargs):
        super().__init__()
        self.heads = nn.ModuleList(
            [ForwardMap(num_parallel=1, **kwargs) for _ in range(num_parallel)]
        )
        self.num_parallel = num_parallel

    def forward(self, h, z, a):
        return torch.stack([head(h, z, a) for head in self.heads], dim=0)  # (K, B, D)


class Actor(nn.Module):
    """Gaussian actor with fixed std, two embedding heads (s) and (s,z).

    Mean is tanh-squashed.  Returns a ``TruncatedNormal`` distribution.
    """

    def __init__(
        self,
        obs_dim: int,
        z_dim: int,
        action_dim: int,
        hidden_dim: int = 512,
        n_hidden: int = 2,
        n_embed: int = 2,
    ):
        super().__init__()
        self.embed_z = simple_embedding(obs_dim + z_dim, hidden_dim, n_embed)
        self.embed_s = simple_embedding(obs_dim, hidden_dim, n_embed)
        body = []
        for _ in range(n_hidden):
            body += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        body += [nn.Linear(hidden_dim, action_dim)]
        self.body = nn.Sequential(*body)

    def forward(
        self, obs: torch.Tensor, z: torch.Tensor, std: float
    ) -> TruncatedNormal:
        e_z = self.embed_z(torch.cat([obs, z], dim=-1))
        e_s = self.embed_s(obs)
        mu = torch.tanh(self.body(torch.cat([e_s, e_z], dim=-1)))
        return TruncatedNormal(mu, torch.full_like(mu, std))


# --------------------------------------------------------------------------- #
# Agent                                                                        #
# --------------------------------------------------------------------------- #


class TDJEPA:
    """Asymmetric TD-JEPA agent matching ``TDJEPAAgent`` in the reference.

    Defaults reproduce the OGBench config from
    ``scripts/train/proprio/launch_td_jepa_ogbench.py`` (BASE_CFG):
      phi_dim = psi_dim = 50
      encoder: hidden 512, 4 layers, L2-normalised output
      predictor: hidden 512, 2 layers, 2 embedding heads, ensemble of 2
      actor: hidden 512, 2 layers, takes raw obs (actor_use_full_encoder=False)
      discount 0.99, tau 0.005, batch 256, lr 1e-4, ortho coefs 1.0
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        phi_dim: int = 50,
        psi_dim: int = 50,
        enc_hidden_dim: int = 512,
        enc_n_hidden: int = 4,
        pred_hidden_dim: int = 512,
        pred_n_hidden: int = 2,
        actor_hidden_dim: int = 512,
        actor_n_hidden: int = 2,
        actor_use_full_encoder: bool = False,
        num_parallel: int = 2,
        gamma: float = 0.99,
        lr: float = 1e-4,
        tau: float = 0.005,
        phi_ortho_coef: float = 1.0,
        psi_ortho_coef: float = 1.0,
        train_goal_ratio: float = 0.5,
        scale_train_goals: bool = True,
        actor_std: float = 0.2,
        stddev_clip: float = 0.3,
        bc_coeff: float = 0.0,
        z_cov_tau: float = 0.005,
        device: str = "cuda",
    ):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.phi_ortho_coef = phi_ortho_coef
        self.psi_ortho_coef = psi_ortho_coef
        self.train_goal_ratio = train_goal_ratio
        self.scale_train_goals = scale_train_goals
        self.actor_std = actor_std
        self.stddev_clip = stddev_clip
        self.bc_coeff = bc_coeff
        self.z_cov_tau = z_cov_tau
        self.z_dim = psi_dim  # z lives in psi-space
        self.phi_dim = phi_dim
        self.psi_dim = psi_dim
        self.actor_use_full_encoder = actor_use_full_encoder
        self.actor_in_dim = phi_dim if actor_use_full_encoder else obs_dim

        # --- Networks
        self.phi = BackwardMap(obs_dim, phi_dim, enc_hidden_dim, enc_n_hidden).to(
            device
        )
        self.psi = BackwardMap(obs_dim, psi_dim, enc_hidden_dim, enc_n_hidden).to(
            device
        )

        # phi_predictor: (phi(s), z, a) -> psi-space
        self.T_phi = ForwardMapEnsemble(
            num_parallel=num_parallel,
            in_dim=phi_dim,
            z_dim=psi_dim,
            action_dim=action_dim,
            out_dim=psi_dim,
            hidden_dim=pred_hidden_dim,
            n_hidden=pred_n_hidden,
        ).to(device)
        # psi_predictor: (psi(s), z, a) -> phi-space
        self.T_psi = ForwardMapEnsemble(
            num_parallel=num_parallel,
            in_dim=psi_dim,
            z_dim=psi_dim,
            action_dim=action_dim,
            out_dim=phi_dim,
            hidden_dim=pred_hidden_dim,
            n_hidden=pred_n_hidden,
        ).to(device)

        self.actor = Actor(
            self.actor_in_dim,
            psi_dim,
            action_dim,
            actor_hidden_dim,
            actor_n_hidden,
        ).to(device)

        for m in (self.phi, self.psi, self.T_phi, self.T_psi, self.actor):
            m.apply(weight_init)

        # --- Target networks (EMA on encoders + predictors only; not actor)
        self.phi_t = copy.deepcopy(self.phi).requires_grad_(False)
        self.psi_t = copy.deepcopy(self.psi).requires_grad_(False)
        self.T_phi_t = copy.deepcopy(self.T_phi).requires_grad_(False)
        self.T_psi_t = copy.deepcopy(self.T_psi).requires_grad_(False)

        # --- Separate optimisers (one per network), matching the reference
        self.opt_phi = torch.optim.Adam(self.phi.parameters(), lr=lr)
        self.opt_psi = torch.optim.Adam(self.psi.parameters(), lr=lr)
        self.opt_T_phi = torch.optim.Adam(self.T_phi.parameters(), lr=lr)
        self.opt_T_psi = torch.optim.Adam(self.T_psi.parameters(), lr=lr)
        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=lr)

        # --- z covariance for goal whitening
        self.register_z_cov = torch.eye(psi_dim, device=device)

    # ------------------------------------------------------------------ utils
    def _project_z(self, z: torch.Tensor) -> torch.Tensor:
        return math.sqrt(self.z_dim) * F.normalize(z, dim=-1)

    @torch.no_grad()
    def _sample_z(self, B: int, psi_next: torch.Tensor | None = None) -> torch.Tensor:
        """Sample z mixture: half from psi(next_obs) (whitened, projected), half from sphere."""
        z = torch.randn(B, self.z_dim, device=self.device)
        z = self._project_z(z)
        if psi_next is not None:
            perm = torch.randperm(B, device=self.device)
            goals = psi_next[perm]
            if self.scale_train_goals:
                eye = torch.eye(self.z_dim, device=self.device)
                inv_cov = torch.linalg.inv(self.register_z_cov + 1e-6 * eye)
                goals = goals @ inv_cov
            goals = self._project_z(goals)
            mask = torch.rand((B, 1), device=self.device) < self.train_goal_ratio
            z = torch.where(mask, goals, z)
        return z

    @torch.no_grad()
    def _update_z_cov(self, z_batch: torch.Tensor) -> None:
        # z_batch should be the ψ(s) of the current batch (the "task" representation)
        new = z_batch.T @ z_batch / z_batch.shape[0]
        self.register_z_cov.mul_(1.0 - self.z_cov_tau).add_(new, alpha=self.z_cov_tau)

    def _orth_loss(self, enc: torch.Tensor):
        """Matches reference _orth_loss exactly.

        Cov = enc @ enc.T, shape (B, B).
        diag    = -mean(diag(Cov))
        offdiag = 0.5 * sum_{i!=j} Cov[i,j]^2 / (B*(B-1))
        Returns (loss, diag, offdiag) for logging.
        """
        B = enc.shape[0]
        cov = enc @ enc.T
        off_mask = 1.0 - torch.eye(B, device=enc.device)
        diag = -cov.diag().mean()
        offdiag = 0.5 * (cov * off_mask).pow(2).sum() / off_mask.sum()
        return offdiag + diag, diag.detach(), offdiag.detach()

    @staticmethod
    def _ensemble_target(preds: torch.Tensor) -> torch.Tensor:
        """Mean of ensemble predictions: preds shape (K, B, D) -> (B, D)."""
        return preds.mean(dim=0)

    def _ema(self, target: nn.Module, source: nn.Module) -> None:
        with torch.no_grad():
            for tp, sp in zip(target.parameters(), source.parameters()):
                tp.data.mul_(1.0 - self.tau).add_(sp.data, alpha=self.tau)

    # ------------------------------------------------------------------ update
    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        obs = batch["obs"]
        act = batch["action"]
        nxt = batch["next_obs"]
        discount = batch.get(
            "discount", self.gamma * torch.ones(obs.shape[0], 1, device=self.device)
        )

        # --- z sampling (using psi-target on next_obs as the "goal" source)
        with torch.no_grad():
            psi_next_target = self.psi_t(nxt)
        z = self._sample_z(obs.shape[0], psi_next=psi_next_target)

        # --- TD targets (no grad)
        with torch.no_grad():
            next_phi_t = self.phi_t(nxt)
            next_psi_t = self.psi_t(nxt)
            actor_in_next = next_phi_t if self.actor_use_full_encoder else nxt
            next_dist = self.actor(actor_in_next, z, self.actor_std)
            next_action = next_dist.sample(clip=self.stddev_clip)

            tgt_phi_preds = self.T_phi_t(next_phi_t, z, next_action)  # (K, B, psi_dim)
            tgt_psi_preds = self.T_psi_t(next_psi_t, z, next_action)  # (K, B, phi_dim)

            tgt_phi = next_psi_t + discount * self._ensemble_target(
                tgt_phi_preds
            )  # (B, psi_dim)
            tgt_psi = next_phi_t + discount * self._ensemble_target(
                tgt_psi_preds
            )  # (B, phi_dim)

        # --- Predictions
        phi_enc = self.phi(obs)
        psi_enc = self.psi(obs)
        phi_preds = self.T_phi(phi_enc, z, act)  # (K, B, psi_dim)
        psi_preds = self.T_psi(psi_enc, z, act)  # (K, B, phi_dim)

        # TD loss: sum over feature dim, mean over (parallel x batch).
        phi_td = (phi_preds - tgt_phi.unsqueeze(0)).pow(2).sum(-1).mean()
        psi_td = (psi_preds - tgt_psi.unsqueeze(0)).pow(2).sum(-1).mean()
        td_loss = phi_td + psi_td

        # Orthonormality
        phi_orth, phi_diag, phi_off = self._orth_loss(phi_enc)
        psi_orth, psi_diag, psi_off = self._orth_loss(psi_enc)

        total = (
            td_loss + self.phi_ortho_coef * phi_orth + self.psi_ortho_coef * psi_orth
        )

        for opt in (self.opt_phi, self.opt_psi, self.opt_T_phi, self.opt_T_psi):
            opt.zero_grad(set_to_none=True)
        total.backward()
        for opt in (self.opt_phi, self.opt_psi, self.opt_T_phi, self.opt_T_psi):
            opt.step()

        # Update running z covariance with current psi(s)
        self._update_z_cov(psi_enc.detach())

        # --- Actor update
        actor_metrics = self._update_actor(obs, act, z, phi_enc.detach())

        # --- EMA target networks
        self._ema(self.phi_t, self.phi)
        self._ema(self.psi_t, self.psi)
        self._ema(self.T_phi_t, self.T_phi)
        self._ema(self.T_psi_t, self.T_psi)

        return {
            "td_loss": td_loss.item(),
            "phi_td": phi_td.item(),
            "psi_td": psi_td.item(),
            "phi_orth": phi_orth.item(),
            "psi_orth": psi_orth.item(),
            "phi_diag": phi_diag.item(),
            "psi_diag": psi_diag.item(),
            "phi_off": phi_off.item(),
            "psi_off": psi_off.item(),
            "phi_norm": phi_enc.norm(dim=-1).mean().item(),
            "psi_norm": psi_enc.norm(dim=-1).mean().item(),
            "tgt_phi": tgt_phi.mean().item(),
            **actor_metrics,
        }

    def _update_actor(self, obs, act, z, phi_enc_detached) -> Dict[str, float]:
        actor_in = phi_enc_detached if self.actor_use_full_encoder else obs
        dist = self.actor(actor_in, z, self.actor_std)
        actor_action = dist.sample(clip=self.stddev_clip)
        preds = self.T_phi(phi_enc_detached, z, actor_action)  # (K, B, psi_dim)
        Qs = (preds * z.unsqueeze(0)).sum(-1)  # (K, B)
        Q = Qs.mean(dim=0)  # (B,)
        actor_loss = -Q.mean()

        bc_err = torch.zeros((), device=self.device)
        if self.bc_coeff > 0:
            bc_err = F.mse_loss(actor_action, act)
            # Crucial scaling: normalise the Q term, then add BC -- matches reference.
            actor_loss = (
                actor_loss / Qs.abs().mean().detach()
            ) + self.bc_coeff * bc_err

        self.opt_actor.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.opt_actor.step()
        return {
            "actor_loss": actor_loss.item(),
            "bc_err": bc_err.item() if self.bc_coeff > 0 else 0.0,
            "q_mean": Q.mean().item(),
        }

    # -------------------------------------------------------------- inference
    @torch.no_grad()
    def infer_z_from_rewards(
        self, obs: torch.Tensor, rewards: torch.Tensor
    ) -> torch.Tensor:
        """Closed-form least-squares of ``r`` onto ``psi(s)``.  Matches the reference's
        ``reward_inference``.

        rewards: (N,) or (N, 1).  Returns z of shape (1, psi_dim) on the sphere.
        """
        psi_s = self.psi(obs)  # (N, psi_dim)
        r = rewards.float().view(-1, 1)
        # torch.linalg.lstsq solves min_z ||psi_s @ z - r||^2
        z = torch.linalg.lstsq(psi_s, r).solution.T  # (1, psi_dim)
        return self._project_z(z)

    @torch.no_grad()
    def infer_z_from_goal(self, goal_obs: torch.Tensor) -> torch.Tensor:
        """For goal-reaching with sparse reward at the goal: z = psi(g) (already on sphere)."""
        return self.psi(goal_obs)

    @torch.no_grad()
    def act(
        self, obs: torch.Tensor, z: torch.Tensor, deterministic: bool = True
    ) -> torch.Tensor:
        actor_in = self.phi(obs) if self.actor_use_full_encoder else obs
        dist = self.actor(actor_in, z, self.actor_std)
        return dist.mean if deterministic else dist.sample(clip=self.stddev_clip)
