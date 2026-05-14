"""TD-JEPA v3: Faithful reimplementation aligned with the reference metamotivo code.

Key fixes from v2 (causing 0% zero-shot success)
------------------------------------------------
1. Actor loss: -Q/|Q| + bc_coeff * MSE(sampled_action, data_action)
   — v2 had -Q + bc_coeff * MSE / |Q| which inverted the normalization.
   With the v2 formula, BC dominated the actor loss (5x amplified) and
   prevented goal-directed behaviour.
2. Encoder/predictor depths were SWAPPED. Reference uses 4 hidden layers
   (5 total) for encoders and 2 for predictor bodies; v2 had 2/4.
   Now n_hidden_enc=3 (5 total layers) and n_hidden_pred=2.
3. Z-whitening now uses full inverse covariance (like reference), and only
   whitens goal-encoded samples, not the entire z mixture.
4. Ensemble predictor loss computed per-member then averaged (not mean-then-loss).
5. Actor uses sampled action for BC loss (not mean action).
6. Batch size 256 (not 1024), learning rate 1e-4 (not 3e-4).
7. Actor body depth 1 (not 4), matching reference default hidden_layers=1.

Key fixes from v1
-----------------
1. Both phi and psi are L2-normalised to sqrt(d).
2. d_phi = d_psi = 50 by default.
3. TD loss: .pow(2).sum(-1).mean() convention.
4. BC scaling now matches reference.
5. Predictor: two embedding heads (h⊕a) and (h⊕z).
6. Ensemble of 2 predictors.

Reference: Bagatella et al., "TD-JEPA: Latent-predictive Representations for
Zero-Shot RL" (arXiv 2510.00739, 2025).
"""

from __future__ import annotations

import copy
import math
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================================== #
# Networks                                                                     #
# =========================================================================== #

def _orthogonal_init(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class EmbeddingBlock(nn.Module):
    """Linear → LayerNorm → Tanh.  Input projection for all networks."""

    def __init__(self, in_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLP(nn.Module):
    """ReLU MLP.  No activation or LayerNorm on the output layer."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, n_hidden: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        d = in_dim
        for _ in range(n_hidden):
            layers += [nn.Linear(d, hidden_dim), nn.ReLU()]
            d = hidden_dim
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Encoder(nn.Module):
    """phi or psi: EmbeddingBlock → MLP(ReLU) → L2-normalise to sqrt(d_out).

    Both encoders are L2-normalised so that ||phi(s)|| = ||psi(s)|| = sqrt(d).
    This keeps the orthonormality regularisation stable:  diag.mean() = d is
    constant, so only the off-diagonal term contributes gradients.
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, n_hidden: int) -> None:
        super().__init__()
        self.emb = EmbeddingBlock(in_dim, hidden_dim)
        self.body = MLP(hidden_dim, hidden_dim, out_dim, n_hidden)
        self.out_dim = out_dim
        self.apply(_orthogonal_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.emb(x)
        h = self.body(h)
        norm = h.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return math.sqrt(self.out_dim) * h / norm


class Predictor(nn.Module):
    """Single predictor network T : R^{d_in} x A x Z -> R^{d_out}.

    Two LN-Tanh embedding heads:
      Head 1: (h ⊕ a)  —  state encoding combined with action
      Head 2: (h ⊕ z)  —  state encoding combined with task vector

    State information appears in BOTH heads (fix from v1 where state was
    only in one head and action‖z was in the other).
    """

    def __init__(
        self,
        in_dim: int,
        action_dim: int,
        z_dim: int,
        hidden_dim: int,
        out_dim: int,
        n_hidden: int,
    ) -> None:
        super().__init__()
        self.emb_sa = EmbeddingBlock(in_dim + action_dim, hidden_dim)
        self.emb_sz = EmbeddingBlock(in_dim + z_dim, hidden_dim)
        self.body = MLP(2 * hidden_dim, hidden_dim, out_dim, n_hidden)
        self.apply(_orthogonal_init)

    def forward(
        self, h: torch.Tensor, a: torch.Tensor, z: torch.Tensor
    ) -> torch.Tensor:
        e1 = self.emb_sa(torch.cat([h, a], dim=-1))
        e2 = self.emb_sz(torch.cat([h, z], dim=-1))
        return self.body(torch.cat([e1, e2], dim=-1))


class Actor(nn.Module):
    """Gaussian actor with fixed std.  Mean is tanh-squashed, noise added for
    exploration, action clamped to [-1, 1].

    By default (actor_uses_encoder=False for OGBench) the actor takes raw
    observations rather than phi(s).
    """

    def __init__(
        self,
        in_dim: int,
        z_dim: int,
        action_dim: int,
        hidden_dim: int,
        n_hidden: int,
        std: float = 0.2,
    ) -> None:
        super().__init__()
        self.emb_state = EmbeddingBlock(in_dim, hidden_dim)
        self.emb_z = EmbeddingBlock(z_dim, hidden_dim)
        self.body = MLP(2 * hidden_dim, hidden_dim, action_dim, n_hidden)
        self.std = std
        self.apply(_orthogonal_init)

    def mean(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        e1 = self.emb_state(h)
        e2 = self.emb_z(z)
        return torch.tanh(self.body(torch.cat([e1, e2], dim=-1)))

    def sample(
        self, h: torch.Tensor, z: torch.Tensor, deterministic: bool = False
    ) -> torch.Tensor:
        mu = self.mean(h, z)
        if deterministic:
            return mu
        return (mu + torch.randn_like(mu) * self.std).clamp(-1.0, 1.0)


# =========================================================================== #
# Orthonormality regularisation (Algorithm 1)                                 #
# =========================================================================== #

def ortho_reg(z: torch.Tensor) -> torch.Tensor:
    """L_REG: average squared off-diagonal covariance minus mean diagonal.

    With L2-normalised encodings (||z|| = sqrt(d)), diag = d is constant,
    so -diag.mean() = -d is a fixed offset and only the off-diagonal term
    provides a useful gradient that pushes different dimensions apart.
    """
    B = z.size(0)
    cov = z @ z.T  # (B, B)
    diag = torch.diagonal(cov)
    off_sq = (cov ** 2).sum() - (diag ** 2).sum()
    return off_sq / (2.0 * B * (B - 1)) - diag.mean()


# =========================================================================== #
# TD-JEPA Agent                                                                #
# =========================================================================== #

class TDJEPA2:
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        d_z: int = 50,
        hidden_dim: int = 512,
        n_hidden_enc: int = 3,
        n_hidden_pred: int = 2,
        n_hidden_actor: int = 1,
        n_ensemble: int = 2,
        gamma: float = 0.99,
        lr: float = 1e-4,
        tau: float = 0.005,
        lambda_reg: float = 1.0,
        bc_alpha: float = 0.3,
        actor_std: float = 0.2,
        actor_uses_encoder: bool = False,
        p_goal: float = 0.5,
        z_whitening_decay: float = 0.995,
        device: str = "cuda",
    ) -> None:
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.lambda_reg = lambda_reg
        self.bc_alpha = bc_alpha
        self.p_goal = p_goal
        self.d_z = d_z
        self.n_ensemble = n_ensemble
        self.actor_uses_encoder = actor_uses_encoder

        # -- Encoders (both L2-normalised to sqrt(d_z)) --
        self.phi = Encoder(obs_dim, hidden_dim, d_z, n_hidden_enc).to(device)
        self.psi = Encoder(obs_dim, hidden_dim, d_z, n_hidden_enc).to(device)

        # -- Predictor ensemble (2 networks, mean averaging) --
        self.T_phi = nn.ModuleList([
            Predictor(d_z, action_dim, d_z, hidden_dim, d_z, n_hidden_pred)
            for _ in range(n_ensemble)
        ]).to(device)
        self.T_psi = nn.ModuleList([
            Predictor(d_z, action_dim, d_z, hidden_dim, d_z, n_hidden_pred)
            for _ in range(n_ensemble)
        ]).to(device)

        # -- Actor --
        actor_in_dim = d_z if actor_uses_encoder else obs_dim
        self.actor = Actor(actor_in_dim, d_z, action_dim, hidden_dim, n_hidden_actor, std=actor_std).to(device)

        # -- Target networks (EMA copies) --
        self.phi_t = copy.deepcopy(self.phi).requires_grad_(False)
        self.psi_t = copy.deepcopy(self.psi).requires_grad_(False)
        self.T_phi_t = copy.deepcopy(self.T_phi).requires_grad_(False)
        self.T_psi_t = copy.deepcopy(self.T_psi).requires_grad_(False)

        # -- Z whitening: running covariance EMA --
        self._z_cov = torch.eye(d_z, device=device)
        self._z_whitening_decay = z_whitening_decay

        # -- Optimisers (repr = encoders + predictors, actor separate) --
        repr_params = (
            list(self.phi.parameters())
            + list(self.psi.parameters())
            + list(self.T_phi.parameters())
            + list(self.T_psi.parameters())
        )
        self.opt_repr = torch.optim.Adam(repr_params, lr=lr)
        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=lr)

    # --------------------------------------------------------------------- utils

    @torch.no_grad()
    def _sample_z(self, obs: torch.Tensor) -> torch.Tensor:
        """Mixture of psi(s_perm) (optionally whitened) and uniform-on-sphere.

        Matches the reference: only the goal-encoded samples are whitened,
        not the uniform spherical samples.
        """
        B = obs.size(0)
        # Spherical samples: uniform on sphere of radius sqrt(d_z)
        z_sph = torch.randn(B, self.d_z, device=obs.device)
        z_sph = math.sqrt(self.d_z) * z_sph / z_sph.norm(dim=-1, keepdim=True).clamp_min(1e-8)

        # Goal-encoded samples from psi(s_perm)
        z_goal = self.psi(obs[torch.randperm(B, device=obs.device)])

        # Whiten goals using full running inverse covariance (scale_train_goals=True)
        cov_reg = self._z_cov + 1e-6 * torch.eye(self.d_z, device=obs.device)
        inv_cov = torch.inverse(cov_reg)
        z_goal_white = torch.matmul(z_goal, inv_cov)
        # Project whitened goals back to sphere of radius sqrt(d_z)
        z_goal_white = math.sqrt(self.d_z) * z_goal_white / z_goal_white.norm(dim=-1, keepdim=True).clamp_min(1e-8)

        # Mix goals and spherical samples
        mask = (torch.rand(B, 1, device=obs.device) < self.p_goal).float()
        z = mask * z_goal_white + (1.0 - mask) * z_sph

        # Update running covariance EMA using the goal-encoded samples (before whitening)
        batch_cov = (z_goal.T @ z_goal) / B
        self._z_cov = self._z_whitening_decay * self._z_cov + (1.0 - self._z_whitening_decay) * batch_cov

        return z

    def _ema(self, target: nn.Module, source: nn.Module) -> None:
        with torch.no_grad():
            for tp, sp in zip(target.parameters(), source.parameters()):
                tp.data.mul_(1.0 - self.tau).add_(sp.data, alpha=self.tau)

    @torch.no_grad()
    def _predictor_target(
        self,
        predictors: nn.ModuleList,
        target_predictors: nn.ModuleList,
        h: torch.Tensor,
        a: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """Ensemble mean over target predictors for TD target computation."""
        preds = []
        for net in target_predictors:
            preds.append(net(h, a, z))
        return sum(preds) / len(preds)

    def _predictor_all(
        self,
        predictors: nn.ModuleList,
        h: torch.Tensor,
        a: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """Return stacked predictions from all ensemble members: (n_ensemble, B, dim)."""
        preds = []
        for net in predictors:
            preds.append(net(h, a, z))
        return torch.stack(preds, dim=0)

    # ------------------------------------------------------------------ update

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        obs, act, nxt = batch["obs"], batch["action"], batch["next_obs"]

        # Sample z with whitening
        z = self._sample_z(obs)

        # Encode current states (both L2-normalised to sqrt(d_z))
        phi_s = self.phi(obs)
        psi_s = self.psi(obs)

        # ---- TD targets (no grad) ----
        with torch.no_grad():
            phi_n_t = self.phi_t(nxt)
            psi_n_t = self.psi_t(nxt)
            # Actor uses raw obs or phi(s) depending on config
            actor_in_nxt = self.phi_t(nxt) if self.actor_uses_encoder else nxt
            a_next = self.actor.sample(actor_in_nxt, z)
            tgt_phi = psi_n_t + self.gamma * self._predictor_target(
                self.T_phi, self.T_phi_t, phi_n_t, a_next, z
            )
            tgt_psi = phi_n_t + self.gamma * self._predictor_target(
                self.T_psi, self.T_psi_t, psi_n_t, a_next, z
            )

        # ---- Predictions: compute per-ensemble-member loss (not mean-then-loss) ----
        phi_preds = self._predictor_all(self.T_phi, phi_s, act, z)    # (n_ens, B, d_z)
        psi_preds = self._predictor_all(self.T_psi, psi_s, act, z)    # (n_ens, B, d_z)

        # ---- TD loss: sum over feature dim, mean over ensemble and batch ----
        # This matches the reference: (preds - target).pow(2).sum(-1).mean()
        # which averages over both ensemble and batch dimensions
        td_loss_phi = (phi_preds - tgt_phi.unsqueeze(0)).pow(2).sum(dim=-1).mean()
        td_loss_psi = (psi_preds - tgt_psi.unsqueeze(0)).pow(2).sum(dim=-1).mean()
        td_loss = td_loss_phi + td_loss_psi

        # ---- Orthonormality regularisation (Algorithm 1) ----
        reg_loss = ortho_reg(phi_s) + ortho_reg(psi_s)

        # ---- Representation loss ----
        repr_loss = td_loss + self.lambda_reg * reg_loss

        self.opt_repr.zero_grad(set_to_none=True)
        repr_loss.backward()
        self.opt_repr.step()

        # ---- Actor: maximise T_phi(phi(s), a, z)^T z  (SVG-style, stop grad at phi) ----
        with torch.no_grad():
            phi_s_d = self.phi(obs)
        actor_in = phi_s_d if self.actor_uses_encoder else obs
        a_pi = self.actor.sample(actor_in, z)

        # Use online predictor for Q computation — take ensemble mean
        phi_preds_actor = self._predictor_all(self.T_phi, phi_s_d, a_pi, z)  # (n_ens, B, d_z)
        Qs = (phi_preds_actor * z.unsqueeze(0)).sum(dim=-1)  # (n_ens, B)
        q = Qs.mean(dim=0)  # (B,) — mean over ensemble

        # Normalize Q by its magnitude (reference convention)
        q_abs_mean = q.detach().abs().mean().clamp_min(1e-3)
        actor_loss = -q.mean() / q_abs_mean

        # BC regularisation: MSE between sampled action and data action, NOT divided by |Q|
        # Reference: actor_loss = (-Q / |Q|.detach()) + bc_coeff * MSE(sampled_action, data_action)
        bc_loss_val = 0.0
        if self.bc_alpha > 0:
            bc = F.mse_loss(a_pi, act)
            actor_loss = actor_loss + self.bc_alpha * bc
            bc_loss_val = bc.item()

        self.opt_actor.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.opt_actor.step()

        # ---- EMA on all target networks ----
        self._ema(self.phi_t, self.phi)
        self._ema(self.psi_t, self.psi)
        for t_net, o_net in zip(self.T_phi_t, self.T_phi):
            self._ema(t_net, o_net)
        for t_net, o_net in zip(self.T_psi_t, self.T_psi):
            self._ema(t_net, o_net)

        return {
            "td_loss": td_loss.item(),
            "reg_loss": reg_loss.item(),
            "actor_loss": actor_loss.item(),
            "bc_loss": bc_loss_val,
            "q_mean": q.mean().item(),
        }

    # -------------------------------------------------------------- inference

    @torch.no_grad()
    def infer_z_from_goal(self, goal_obs: torch.Tensor) -> torch.Tensor:
        """For goal-reaching tasks: z = psi(g).  Already L2-normalised."""
        return self.psi(goal_obs)

    @torch.no_grad()
    def infer_z_from_rewards(
        self,
        obs: torch.Tensor,
        rewards: torch.Tensor,
        eps: float = 1e-3,
    ) -> torch.Tensor:
        """Closed-form linear regression of r onto psi(s) (Section 3.3).

        This is the reference evaluation method — use it for OGBench tasks
        by regressing task rewards onto psi(observations).
        """
        psi_s = self.psi(obs)
        N = psi_s.size(0)
        cov = psi_s.T @ psi_s / N
        rhs = psi_s.T @ rewards / N
        z = torch.linalg.solve(cov + eps * torch.eye(self.d_z, device=psi_s.device), rhs)
        # Project back to sphere of radius sqrt(d_z)
        norm = z.norm(dim=0, keepdim=True).clamp_min(1e-8)
        return math.sqrt(self.d_z) * z / norm

    @torch.no_grad()
    def act(
        self, obs: torch.Tensor, z: torch.Tensor, deterministic: bool = True
    ) -> torch.Tensor:
        """Produce an action.  Uses raw obs unless actor_uses_encoder=True."""
        actor_in = self.phi(obs) if self.actor_uses_encoder else obs
        return self.actor.sample(actor_in, z, deterministic=deterministic)

    # ------------------------------------------------------------------- I/O

    def state_dict(self) -> dict:
        return {
            "phi": self.phi.state_dict(),
            "psi": self.psi.state_dict(),
            "T_phi": self.T_phi.state_dict(),
            "T_psi": self.T_psi.state_dict(),
            "actor": self.actor.state_dict(),
            "phi_t": self.phi_t.state_dict(),
            "psi_t": self.psi_t.state_dict(),
            "T_phi_t": self.T_phi_t.state_dict(),
            "T_psi_t": self.T_psi_t.state_dict(),
            "_z_cov": self._z_cov.cpu(),
        }

    def load_state_dict(self, sd: dict) -> None:
        self.phi.load_state_dict(sd["phi"])
        self.psi.load_state_dict(sd["psi"])
        self.T_phi.load_state_dict(sd["T_phi"])
        self.T_psi.load_state_dict(sd["T_psi"])
        self.actor.load_state_dict(sd["actor"])
        self.phi_t.load_state_dict(sd["phi_t"])
        self.psi_t.load_state_dict(sd["psi_t"])
        self.T_phi_t.load_state_dict(sd["T_phi_t"])
        self.T_psi_t.load_state_dict(sd["T_psi_t"])
        self._z_cov = sd["_z_cov"].to(self.device)