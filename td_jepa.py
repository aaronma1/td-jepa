"""TD-JEPA: temporal-difference latent-predictive representations for zero-shot RL.

Bagatella, Pirotta, Touati, Lazaric, Tirinzoni (2025).  arXiv:2510.00739

This is the *asymmetric* variant (separate state encoder ``phi`` and task encoder
``psi``).  The core objective is the TD loss of Eq. (9) for the two predictors
``T_phi`` and ``T_psi``, plus the orthonormality regulariser of Algorithm 1 and
an SVG-style actor loss on ``T_phi(phi(s), a, z)^T z``.

Notes
-----
* ``z`` is sampled from a 50/50 mixture of ``psi(s)`` (random batch element) and
  uniform-on-sphere of radius ``sqrt(d_psi)`` -- the same convention as Touati &
  Ollivier 2021 and inherited by TD-JEPA (Section E.4).
* Architecture follows Fig. 9: two LayerNorm-Tanh embedding blocks before the
  main MLP body of every successor-feature / predictor / actor network.
* BC regularisation: the paper uses FlowQ-style regularisation (Park 2025b).  I
  approximate it with a simple MSE-on-mean term scaled by ``|Q|`` so the two
  terms stay at comparable magnitudes; this is enough for pointmaze-style data
  but you'll want the full flow model for harder OGBench manipulation tasks.
"""
from __future__ import annotations

import copy
import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Networks                                                                    #
# --------------------------------------------------------------------------- #

def _orthogonal_init(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class EmbeddingBlock(nn.Module):
    """``Linear -> LayerNorm -> Tanh``.  Used at every input to predictors / actor."""

    def __init__(self, in_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.net(x)


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, n_hidden: int) -> None:
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(n_hidden):
            layers += [nn.Linear(d, hidden_dim), nn.ReLU()]
            d = hidden_dim
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.net(x)


class StateEncoder(nn.Module):
    """``phi : S -> R^{d_phi}``.  Plain MLP (no output normalisation)."""

    def __init__(self, obs_dim: int, hidden_dim: int, out_dim: int, n_hidden: int) -> None:
        super().__init__()
        self.net = MLP(obs_dim, hidden_dim, out_dim, n_hidden)
        self.apply(_orthogonal_init)

    def forward(self, s: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.net(s)


class TaskEncoder(nn.Module):
    """``psi : S -> R^{d_psi}``, L2-normalised and rescaled so ``||psi(s)|| = sqrt(d_psi)``."""

    def __init__(self, obs_dim: int, hidden_dim: int, out_dim: int, n_hidden: int) -> None:
        super().__init__()
        self.net = MLP(obs_dim, hidden_dim, out_dim, n_hidden)
        self.out_dim = out_dim
        self.apply(_orthogonal_init)

    def forward(self, s: torch.Tensor) -> torch.Tensor:  # noqa: D401
        z = self.net(s)
        norm = z.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return math.sqrt(self.out_dim) * z / norm


class Predictor(nn.Module):
    """``T : R^{d_in} x A x Z -> R^{d_out}`` with two LN-Tanh input embeddings."""

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
        self.emb_state = EmbeddingBlock(in_dim, hidden_dim)
        self.emb_az = EmbeddingBlock(action_dim + z_dim, hidden_dim)
        self.body = MLP(2 * hidden_dim, hidden_dim, out_dim, n_hidden)
        self.apply(_orthogonal_init)

    def forward(self, h: torch.Tensor, a: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        e1 = self.emb_state(h)
        e2 = self.emb_az(torch.cat([a, z], dim=-1))
        return self.body(torch.cat([e1, e2], dim=-1))


class Actor(nn.Module):
    """Gaussian actor with fixed std=0.2; mean is tanh-squashed.

    Gradients propagate through the deterministic mean (SVG-style).  Noise is
    added for exploration and the action clamped to [-1, 1].
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

    def sample(self, h: torch.Tensor, z: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        mu = self.mean(h, z)
        if deterministic:
            return mu
        return (mu + torch.randn_like(mu) * self.std).clamp(-1.0, 1.0)


# --------------------------------------------------------------------------- #
# Algorithm                                                                   #
# --------------------------------------------------------------------------- #

class TDJEPA:
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        d_phi: int = 256,
        d_psi: int = 50,
        hidden_dim: int = 512,
        n_hidden_enc: int = 4,
        n_hidden_pred: int = 4,
        n_hidden_actor: int = 4,
        gamma: float = 0.99,
        lr: float = 1e-4,
        tau: float = 0.005,
        lambda_reg: float = 1.0,
        p_goal: float = 0.5,
        bc_alpha: float = 0.0,
        actor_std: float = 0.2,
        device: str = "cuda",
    ) -> None:
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.lambda_reg = lambda_reg
        self.p_goal = p_goal
        self.d_psi = d_psi
        self.bc_alpha = bc_alpha

        self.phi = StateEncoder(obs_dim, hidden_dim, d_phi, n_hidden_enc).to(device)
        self.psi = TaskEncoder(obs_dim, hidden_dim, d_psi, n_hidden_enc).to(device)
        self.T_phi = Predictor(d_phi, action_dim, d_psi, hidden_dim, d_psi, n_hidden_pred).to(device)
        self.T_psi = Predictor(d_psi, action_dim, d_psi, hidden_dim, d_phi, n_hidden_pred).to(device)
        self.actor = Actor(d_phi, d_psi, action_dim, hidden_dim, n_hidden_actor, std=actor_std).to(device)

        self.phi_t = copy.deepcopy(self.phi).requires_grad_(False)
        self.psi_t = copy.deepcopy(self.psi).requires_grad_(False)
        self.T_phi_t = copy.deepcopy(self.T_phi).requires_grad_(False)
        self.T_psi_t = copy.deepcopy(self.T_psi).requires_grad_(False)

        repr_params = (
            list(self.phi.parameters())
            + list(self.psi.parameters())
            + list(self.T_phi.parameters())
            + list(self.T_psi.parameters())
        )
        self.opt_repr = torch.optim.Adam(repr_params, lr=lr)
        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=lr)

    # ------------------------------------------------------------------ utils
    @torch.no_grad()
    def _sample_z(self, obs: torch.Tensor) -> torch.Tensor:
        """Mixture of ``psi(s_perm)`` and uniform-on-sphere of radius ``sqrt(d_psi)``."""
        B = obs.size(0)
        z_goal = self.psi(obs[torch.randperm(B, device=obs.device)])
        z_sph = torch.randn(B, self.d_psi, device=obs.device)
        z_sph = math.sqrt(self.d_psi) * z_sph / z_sph.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        mask = (torch.rand(B, 1, device=obs.device) < self.p_goal).float()
        return mask * z_goal + (1.0 - mask) * z_sph

    @staticmethod
    def _ortho_reg(z: torch.Tensor) -> torch.Tensor:
        """Algorithm 1's L_REG: off-diag (z_i^T z_j)^2 averaged, minus mean diagonal."""
        B = z.size(0)
        cov = z @ z.T                                       # (B, B)
        diag = torch.diagonal(cov)
        off_sq = (cov ** 2).sum() - (diag ** 2).sum()
        return off_sq / (2.0 * B * (B - 1)) - diag.mean()

    def _ema(self, target: nn.Module, source: nn.Module) -> None:
        with torch.no_grad():
            for tp, sp in zip(target.parameters(), source.parameters()):
                tp.data.mul_(1.0 - self.tau).add_(sp.data, alpha=self.tau)

    # ----------------------------------------------------------------- update
    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        obs, act, nxt = batch["obs"], batch["action"], batch["next_obs"]
        z = self._sample_z(obs)

        # --- TD targets (no grad)
        with torch.no_grad():
            phi_n_t = self.phi_t(nxt)
            psi_n_t = self.psi_t(nxt)
            a_next = self.actor.sample(phi_n_t, z)
            tgt_phi = psi_n_t + self.gamma * self.T_phi_t(phi_n_t, a_next, z)
            tgt_psi = phi_n_t + self.gamma * self.T_psi_t(psi_n_t, a_next, z)

        # --- Predictions + TD loss
        phi_s = self.phi(obs)
        psi_s = self.psi(obs)
        pred_phi = self.T_phi(phi_s, act, z)
        pred_psi = self.T_psi(psi_s, act, z)

        td_loss = F.mse_loss(pred_phi, tgt_phi) + F.mse_loss(pred_psi, tgt_psi)
        reg_loss = self._ortho_reg(phi_s) + self._ortho_reg(psi_s)
        repr_loss = td_loss + self.lambda_reg * reg_loss

        self.opt_repr.zero_grad(set_to_none=True)
        repr_loss.backward()
        self.opt_repr.step()

        # --- Actor: maximise T_phi(phi(s), a, z)^T z   (gradient stops at phi)
        with torch.no_grad():
            phi_s_d = self.phi(obs)
        a_pi = self.actor.sample(phi_s_d, z)
        q = (self.T_phi(phi_s_d, a_pi, z) * z).sum(dim=-1)
        actor_loss = -q.mean()

        bc_loss_val = 0.0
        if self.bc_alpha > 0:
            mu = self.actor.mean(phi_s_d, z)
            bc = F.mse_loss(mu, act)
            scale = q.detach().abs().mean().clamp_min(1e-3)
            actor_loss = actor_loss + self.bc_alpha * scale * bc
            bc_loss_val = bc.item()

        self.opt_actor.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.opt_actor.step()

        # --- EMA on all four target nets
        self._ema(self.phi_t, self.phi)
        self._ema(self.psi_t, self.psi)
        self._ema(self.T_phi_t, self.T_phi)
        self._ema(self.T_psi_t, self.T_psi)

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
        """For goal-reaching tasks ``r(s) = 1[s = g]``: ``z = psi(g)``.

        ``TaskEncoder`` already normalises onto the sphere of radius ``sqrt(d_psi)``.
        """
        return self.psi(goal_obs)

    @torch.no_grad()
    def infer_z_from_rewards(
        self,
        obs: torch.Tensor,
        rewards: torch.Tensor,
        eps: float = 1e-3,
    ) -> torch.Tensor:
        """Closed-form linear regression of ``r`` onto ``psi(s)`` (Section 3.3)."""
        psi_s = self.psi(obs)
        N = psi_s.size(0)
        cov = psi_s.T @ psi_s / N
        rhs = psi_s.T @ rewards / N
        z = torch.linalg.solve(cov + eps * torch.eye(self.d_psi, device=psi_s.device), rhs)
        return math.sqrt(self.d_psi) * z / z.norm(dim=0, keepdim=True).clamp_min(1e-8)

    @torch.no_grad()
    def act(self, obs: torch.Tensor, z: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        return self.actor.sample(self.phi(obs), z, deterministic=deterministic)

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
        }

    def load_state_dict(self, sd: dict) -> None:
        for k, mod in [
            ("phi", self.phi), ("psi", self.psi),
            ("T_phi", self.T_phi), ("T_psi", self.T_psi),
            ("actor", self.actor),
            ("phi_t", self.phi_t), ("psi_t", self.psi_t),
            ("T_phi_t", self.T_phi_t), ("T_psi_t", self.T_psi_t),
        ]:
            mod.load_state_dict(sd[k])