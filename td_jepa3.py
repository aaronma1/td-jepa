"""TD-JEPA v3: Precise reimplementation matching the metamotivo reference architecture.

All architectural details now exactly match the reference:
1. Encoder: BackwardMap — [Lin, LN, Tanh] + (hidden_layers-1)*[Lin, ReLU] + [Lin, Norm]
   With hidden_layers=4: [Lin(2,512), LN, Tanh, Lin(512,512), ReLU, Lin(512,512), ReLU,
   Lin(512,512), ReLU, Lin(512,50), L2_norm]
2. Predictor: ForwardMap — 2-layer embedding (Lin, LN, Tanh, Lin, ReLU) with dim reduction
   (hidden_dim → hidden_dim//2) + body with hidden_layers=2
3. Actor: SimpleActor — 2-layer embedding (Lin, LN, Tanh, Lin, ReLU) with dim reduction
   (hidden_dim → hidden_dim//2) + policy with hidden_layers

Critical fixes from v2:
- Actor loss: -Q.mean()/|Q|.mean() + bc_coeff * MSE  (not -Q + bc*MSE/|Q|)
- z sampling uses next_obs (not permuted current obs)
- Architecture depth/width matches reference exactly
- Scale_train_goals (z whitening) uses full inverse covariance
- Ensemble loss: per-member then average (not mean-then-loss)
- Actor uses TruncatedNormal-like sampling with noise clip
- Separate optimizers for phi_enc, psi_enc, phi_pred, psi_pred, actor
- discount applied per-sample (not constant gamma)

Reference: https://github.com/facebookresearch/td_jepa (metamotivo package)
"""

from __future__ import annotations

import copy
import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================================== #
# Networks (matching reference architecture exactly)                          #
# =========================================================================== #

class Norm(nn.Module):
    """L2-normalize to sqrt(d), matching reference Norm()."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return math.sqrt(x.shape[-1]) * F.normalize(x, dim=-1)


def _orthogonal_init(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class BackwardMap(nn.Module):
    """Encoder matching reference BackwardArchi.

    Architecture: [Linear(in, hidden), LayerNorm(hidden), Tanh]
                  + (hidden_layers-1) * [Linear(hidden, hidden), ReLU]
                  + [Linear(hidden, out)]
                  + Norm (L2-normalize to sqrt(d))
    """
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, hidden_layers: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        # Embedding block
        layers += [nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Tanh()]
        # Hidden layers
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        # Output
        layers += [nn.Linear(hidden_dim, out_dim)]
        # L2-normalize
        layers += [Norm()]
        self.net = nn.Sequential(*layers)
        self.apply(_orthogonal_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def simple_embedding(in_dim: int, hidden_dim: int, embedding_layers: int = 2,
                     num_parallel: int = 1) -> nn.Module:
    """Reference simple_embedding: [Lin, LN, Tanh] + (embedding_layers-2)*[Lin, ReLU]
    + [Lin, hidden_dim//2, ReLU].

    Always outputs hidden_dim // 2.
    Requires embedding_layers >= 2.
    """
    assert embedding_layers >= 2, "embedding_layers must be >= 2"
    if num_parallel > 1:
        layers = [DenseParallel(in_dim, hidden_dim, n_parallel=num_parallel),
                  ParallelLayerNorm([hidden_dim], n_parallel=num_parallel),
                  nn.Tanh()]
        for _ in range(embedding_layers - 2):
            layers += [DenseParallel(hidden_dim, hidden_dim, n_parallel=num_parallel), nn.ReLU()]
        layers += [DenseParallel(hidden_dim, hidden_dim // 2, n_parallel=num_parallel), nn.ReLU()]
    else:
        layers = [nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Tanh()]
        for _ in range(embedding_layers - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers += [nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU()]
    return nn.Sequential(*layers)


class DenseParallel(nn.Module):
    """Parallel linear layer for ensemble forward pass."""
    def __init__(self, in_dim: int, out_dim: int, n_parallel: int = 2) -> None:
        super().__init__()
        self.n_parallel = n_parallel
        self.weight = nn.Parameter(torch.empty(n_parallel, out_dim, in_dim))
        self.bias = nn.Parameter(torch.empty(n_parallel, out_dim))
        nn.init.orthogonal_(self.weight, gain=math.sqrt(2))
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (n_parallel, B, in_dim) or (B, in_dim)
        if x.dim() == 2:
            x = x.unsqueeze(0).expand(self.n_parallel, -1, -1)
        return torch.einsum('pbi,poi->pbo', x, self.weight) + self.bias.unsqueeze(1)


class ParallelLayerNorm(nn.Module):
    """Parallel LayerNorm for ensemble."""
    def __init__(self, shapes: list, n_parallel: int = 2) -> None:
        super().__init__()
        self.norms = nn.ModuleList([nn.LayerNorm(s) for s in shapes for _ in range(n_parallel)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (n_parallel, B, *shapes)
        results = []
        for i, norm in enumerate(self.norms):
            results.append(norm(x[i]))
        return torch.stack(results, dim=0)


class ForwardMap(nn.Module):
    """Predictor matching reference ForwardArchi.

    Two embedding heads with dim reduction (hidden_dim -> hidden_dim//2):
      embed_sa: simple_embedding(obs_dim+action_dim, hidden_dim, embedding_layers)
      embed_sz: simple_embedding(obs_dim+z_dim, hidden_dim, embedding_layers)
    Body:
      hidden_layers * [Linear(hidden_dim, hidden_dim), ReLU] + Linear(hidden_dim, out_dim)
    Supports num_parallel > 1 for ensemble-in-one-forward-pass.
    """
    def __init__(self, obs_dim: int, z_dim: int, action_dim: int, hidden_dim: int,
                 hidden_layers: int = 2, embedding_layers: int = 2,
                 num_parallel: int = 2, out_dim: int | None = None) -> None:
        super().__init__()
        self.num_parallel = num_parallel
        self.hidden_dim = hidden_dim
        if out_dim is None:
            out_dim = z_dim
        self.embed_sa = simple_embedding(obs_dim + action_dim, hidden_dim, embedding_layers)
        self.embed_z = simple_embedding(obs_dim + z_dim, hidden_dim, embedding_layers)

        layers: list[nn.Module] = []
        for _ in range(hidden_layers):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers += [nn.Linear(hidden_dim, out_dim)]
        self.Fs = nn.Sequential(*layers)
        self.apply(_orthogonal_init)

    def forward(self, obs: torch.Tensor, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        sa_embedding = self.embed_sa(torch.cat([obs, action], dim=-1))
        z_embedding = self.embed_z(torch.cat([obs, z], dim=-1))
        return self.Fs(torch.cat([sa_embedding, z_embedding], dim=-1))


class ForwardMapParallel(nn.Module):
    """Ensemble predictor using DenseParallel for efficient forward pass.

    Same as ForwardMap but uses DenseParallel linear layers for num_parallel > 1.
    """
    def __init__(self, obs_dim: int, z_dim: int, action_dim: int, hidden_dim: int,
                 hidden_layers: int = 2, embedding_layers: int = 2,
                 num_parallel: int = 2, out_dim: int | None = None) -> None:
        super().__init__()
        self.num_parallel = num_parallel
        self.hidden_dim = hidden_dim
        if out_dim is None:
            out_dim = z_dim
        # Embedding heads are shared across ensemble members
        self.embed_sa = simple_embedding(obs_dim + action_dim, hidden_dim, embedding_layers)
        self.embed_z = simple_embedding(obs_dim + z_dim, hidden_dim, embedding_layers)

        # Body uses DenseParallel for ensemble
        body_layers: list[nn.Module] = []
        for _ in range(hidden_layers):
            body_layers += [DenseParallel(hidden_dim, hidden_dim, num_parallel), nn.ReLU()]
        body_layers += [DenseParallel(hidden_dim, out_dim, num_parallel)]
        self.Fs = nn.Sequential(*body_layers)
        self.apply(_orthogonal_init)

    def forward(self, obs: torch.Tensor, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if obs.dim() == 2:
            obs = obs.unsqueeze(0).expand(self.num_parallel, -1, -1)
            z = z.unsqueeze(0).expand(self.num_parallel, -1, -1)
            action = action.unsqueeze(0).expand(self.num_parallel, -1, -1)
        sa_embedding = self.embed_sa(torch.cat([obs, action], dim=-1))
        z_embedding = self.embed_z(torch.cat([obs, z], dim=-1))
        return self.Fs(torch.cat([sa_embedding, z_embedding], dim=-1))


class Actor(nn.Module):
    """Actor matching reference Actor.

    Architecture:
      embed_s: simple_embedding(obs_dim, hidden_dim, embedding_layers=2) → hidden_dim//2
      embed_z: simple_embedding(obs_dim+z_dim, hidden_dim, embedding_layers=2) → hidden_dim//2
      policy: hidden_layers * [Linear(hidden_dim, hidden_dim), ReLU] + Linear(hidden_dim, action_dim)
      output: tanh(policy(concat(embed_s, embed_z)))
    """
    def __init__(self, obs_dim: int, z_dim: int, action_dim: int, hidden_dim: int,
                 hidden_layers: int = 2, embedding_layers: int = 2, stddev: float = 0.2,
                 stddev_clip: float = 0.3) -> None:
        super().__init__()
        self.embed_s = simple_embedding(obs_dim, hidden_dim, embedding_layers)
        self.embed_z = simple_embedding(obs_dim + z_dim, hidden_dim, embedding_layers)
        layers: list[nn.Module] = []
        for _ in range(hidden_layers):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers += [nn.Linear(hidden_dim, action_dim)]
        self.policy = nn.Sequential(*layers)
        self.stddev = stddev
        self.stddev_clip = stddev_clip
        self.apply(_orthogonal_init)

    def forward(self, obs: torch.Tensor, z: torch.Tensor, std: float) -> "TruncatedNormal":
        s_embedding = self.embed_s(obs)
        z_embedding = self.embed_z(torch.cat([obs, z], dim=-1))
        mu = torch.tanh(self.policy(torch.cat([s_embedding, z_embedding], dim=-1)))
        std = torch.ones_like(mu) * std
        return TruncatedNormal(mu, std)


class TruncatedNormal:
    """Minimal TruncatedNormal matching reference dist.sample(clip=0.3)."""
    def __init__(self, mu: torch.Tensor, std: torch.Tensor):
        self.mu = mu
        self.std = std

    def sample(self, clip: float = 0.3) -> torch.Tensor:
        noise = torch.randn_like(self.mu) * self.std
        noise = noise.clamp(-clip, clip)
        return (self.mu + noise).clamp(-1.0 + 1e-6, 1.0 - 1e-6)

    @property
    def mean(self) -> torch.Tensor:
        return self.mu


# =========================================================================== #
# Orthogonality loss                                                           #
# =========================================================================== #

def ortho_reg(z: torch.Tensor, off_diag: torch.Tensor | None = None,
              off_diag_sum: float | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """L_orth: off-diagonal squared + (-diagonal mean).

    Returns (loss, diag_loss, offdiag_loss).
    """
    B = z.size(0)
    cov = z @ z.T
    diag = torch.diagonal(cov)
    offdiag = 0.5 * (cov ** 2)
    # Mask out diagonal
    mask = 1 - torch.eye(B, device=z.device)
    offdiag_loss = (offdiag * mask).sum() / max(B * (B - 1), 1)
    diag_loss = -diag.mean()
    return offdiag_loss + diag_loss, diag_loss, offdiag_loss


# =========================================================================== #
# TD-JEPA Agent                                                                #
# =========================================================================== #

class TDJEPA3:
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        d_z: int = 50,
        hidden_dim: int = 512,
        n_hidden_enc: int = 4,
        n_hidden_pred: int = 2,
        n_hidden_actor: int = 2,
        n_embedding_pred: int = 2,
        n_embedding_actor: int = 2,
        n_ensemble: int = 2,
        gamma: float = 0.99,
        lr: float = 1e-4,
        tau_enc: float = 0.005,
        tau_pred: float = 0.005,
        phi_ortho_coef: float = 1.0,
        psi_ortho_coef: float = 1.0,
        bc_coeff: float = 0.3,
        actor_std: float = 0.2,
        stddev_clip: float = 0.3,
        actor_uses_encoder: bool = False,
        p_goal: float = 0.5,
        z_whitening_decay: float = 0.995,
        scale_train_goals: bool = True,
        device: str = "cuda",
    ) -> None:
        self.device = device
        self.gamma = gamma
        self.tau_enc = tau_enc
        self.tau_pred = tau_pred
        self.phi_ortho_coef = phi_ortho_coef
        self.psi_ortho_coef = psi_ortho_coef
        self.bc_coeff = bc_coeff
        self.p_goal = p_goal
        self.d_z = d_z
        self.n_ensemble = n_ensemble
        self.actor_uses_encoder = actor_uses_encoder
        self.scale_train_goals = scale_train_goals
        self.stddev_clip = stddev_clip
        self.actor_std = actor_std

        # -- Encoders (both L2-normalised to sqrt(d_z)) --
        self.phi = BackwardMap(obs_dim, hidden_dim, d_z, n_hidden_enc).to(device)
        self.psi = BackwardMap(obs_dim, hidden_dim, d_z, n_hidden_enc).to(device)

        # -- Predictor ensemble (using ModuleList for simplicity) --
        self.T_phi = nn.ModuleList([
            ForwardMap(d_z, d_z, action_dim, hidden_dim, n_hidden_pred, n_embedding_pred, out_dim=d_z)
            for _ in range(n_ensemble)
        ]).to(device)
        self.T_psi = nn.ModuleList([
            ForwardMap(d_z, d_z, action_dim, hidden_dim, n_hidden_pred, n_embedding_pred, out_dim=d_z)
            for _ in range(n_ensemble)
        ]).to(device)

        # -- Actor --
        actor_in_dim = d_z if actor_uses_encoder else obs_dim
        self.actor = Actor(actor_in_dim, d_z, action_dim, hidden_dim,
                           n_hidden_actor, n_embedding_actor, actor_std, stddev_clip).to(device)

        # -- Target networks (EMA copies) --
        self.phi_t = copy.deepcopy(self.phi).requires_grad_(False)
        self.psi_t = copy.deepcopy(self.psi).requires_grad_(False)
        self.T_phi_t = copy.deepcopy(self.T_phi).requires_grad_(False)
        self.T_psi_t = copy.deepcopy(self.T_psi).requires_grad_(False)

        # -- Z whitening: running covariance EMA --
        self.register_buffer('_z_cov', torch.eye(d_z, device=device))
        self.z_whitening_decay = z_whitening_decay

        # -- Separate optimizers (matching reference) --
        self.phi_enc_opt = torch.optim.Adam(self.phi.parameters(), lr=lr)
        self.psi_enc_opt = torch.optim.Adam(self.psi.parameters(), lr=lr)
        self.phi_pred_opt = torch.optim.Adam(self.T_phi.parameters(), lr=lr)
        self.psi_pred_opt = torch.optim.Adam(self.T_psi.parameters(), lr=lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)

        # Gradient clipping for stability
        self.max_grad_norm = 10.0

    def register_buffer(self, name: str, tensor: torch.Tensor):
        # Simple attribute (not nn.Module, so no register_buffer mechanism)
        setattr(self, name, tensor)

    @torch.no_grad()
    def _sample_z(self, obs: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
        """Sample z: mixture of psi(next_obs) goals (whitened) and uniform sphere.

        Uses next_obs for goals, matching the reference.
        """
        B = obs.size(0)
        # Spherical samples
        z_sph = torch.randn(B, self.d_z, device=obs.device)
        z_sph = math.sqrt(self.d_z) * F.normalize(z_sph, dim=-1)

        # Goal-encoded samples from next_obs (not permuted, already random batch)
        z_goal = self.psi(next_obs)

        # Whiten goals
        if self.scale_train_goals:
            cov_reg = self._z_cov + 1e-6 * torch.eye(self.d_z, device=obs.device)
            inv_cov = torch.linalg.inv(cov_reg)
            z_goal_white = torch.matmul(z_goal, inv_cov)
            z_goal_white = math.sqrt(self.d_z) * F.normalize(z_goal_white, dim=-1)
        else:
            z_goal_white = z_goal

        # Mix
        mask = (torch.rand(B, 1, device=obs.device) < self.p_goal).float()
        z = mask * z_goal_white + (1.0 - mask) * z_sph

        return z

    def _update_z_stats(self, psi_enc: torch.Tensor) -> None:
        """Update z_cov EMA after the backward pass, matching reference."""
        B = psi_enc.size(0)
        batch_cov = (psi_enc.T @ psi_enc) / B
        self._z_cov = self.z_whitening_decay * self._z_cov + (1.0 - self.z_whitening_decay) * batch_cov

    def _ema(self, target: nn.Module, source: nn.Module, tau: float) -> None:
        with torch.no_grad():
            for tp, sp in zip(target.parameters(), source.parameters()):
                tp.data.mul_(1.0 - tau).add_(sp.data, alpha=tau)

    @torch.no_grad()
    def _predictor_target_mean(
        self, predictors: nn.ModuleList, h: torch.Tensor, a: torch.Tensor, z: torch.Tensor
    ) -> torch.Tensor:
        """Ensemble mean over target predictors."""
        preds = [net(obs=h, z=z, action=a) for net in predictors]
        return sum(preds) / len(preds)

    def _predictor_all(
        self, predictors: nn.ModuleList, h: torch.Tensor, a: torch.Tensor, z: torch.Tensor
    ) -> torch.Tensor:
        """Stacked predictions from all ensemble members: (n_ensemble, B, d_z)."""
        return torch.stack([net(obs=h, z=z, action=a) for net in predictors], dim=0)

    # ------------------------------------------------------------------ update

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        obs, act, nxt = batch["obs"], batch["action"], batch["next_obs"]
        batch_discount = batch["discount"]
        if batch_discount.dim() == 1:
            batch_discount = batch_discount.unsqueeze(-1)  # (B, 1)

        # Sample z using next_obs for goals
        z = self._sample_z(obs, nxt)

        # Encode current states
        phi_s = self.phi(obs)
        psi_s = self.psi(obs)

        # ---- TD targets (no grad) ----
        with torch.no_grad():
            phi_n_t = self.phi_t(nxt)
            psi_n_t = self.psi_t(nxt)
            # Action for next state
            actor_in_nxt = self.phi_t(nxt) if self.actor_uses_encoder else nxt
            dist_nxt = self.actor(actor_in_nxt, z, self.actor_std)
            a_next = dist_nxt.sample(clip=self.stddev_clip)

            tgt_phi_pred = self._predictor_target_mean(self.T_phi_t, phi_n_t, a_next, z)
            tgt_psi_pred = self._predictor_target_mean(self.T_psi_t, psi_n_t, a_next, z)

            tgt_phi = psi_n_t + batch_discount * tgt_phi_pred
            tgt_psi = phi_n_t + batch_discount * tgt_psi_pred

        # ---- Predictions (per-ensemble, then loss) ----
        phi_preds = self._predictor_all(self.T_phi, phi_s, act, z)   # (n_ens, B, d_z)
        psi_preds = self._predictor_all(self.T_psi, psi_s, act, z)   # (n_ens, B, d_z)

        td_loss_phi = (phi_preds - tgt_phi.unsqueeze(0)).pow(2).sum(dim=-1).mean()
        td_loss_psi = (psi_preds - tgt_psi.unsqueeze(0)).pow(2).sum(dim=-1).mean()
        td_loss = td_loss_phi + td_loss_psi

        # ---- Orthonormality regularisation ----
        phi_orth_loss, phi_diag, phi_off = ortho_reg(phi_s)
        psi_orth_loss, psi_diag, psi_off = ortho_reg(psi_s)

        total_loss = td_loss + self.phi_ortho_coef * phi_orth_loss + self.psi_ortho_coef * psi_orth_loss

        # ---- Update representation (encoders + predictors) ----
        self.phi_enc_opt.zero_grad(set_to_none=True)
        self.psi_enc_opt.zero_grad(set_to_none=True)
        self.phi_pred_opt.zero_grad(set_to_none=True)
        self.psi_pred_opt.zero_grad(set_to_none=True)
        total_loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(
            list(self.phi.parameters()) + list(self.psi.parameters())
            + list(self.T_phi.parameters()) + list(self.T_psi.parameters()),
            self.max_grad_norm
        )
        self.phi_enc_opt.step()
        self.psi_enc_opt.step()
        self.phi_pred_opt.step()
        self.psi_pred_opt.step()

        # Update z_cov after backward (matching reference: uses online psi encoding of current obs)
        self._update_z_stats(psi_s.detach())
        with torch.no_grad():
            phi_s_d = self.phi(obs)
        actor_in = phi_s_d if self.actor_uses_encoder else obs
        dist = self.actor(actor_in, z, self.actor_std)
        actor_action = dist.sample(clip=self.stddev_clip)

        # Q = T_phi(phi(s), a, z)^T z, ensemble mean
        preds = self._predictor_all(self.T_phi, phi_s_d, actor_action, z)  # (n_ens, B, d_z)
        Qs = (preds * z.unsqueeze(0)).sum(dim=-1)  # (n_ens, B)
        Q = Qs.mean(dim=0)  # (B,) ensemble mean

        actor_loss = -Q.mean()

        bc_error = torch.tensor(0.0, device=z.device)
        if self.bc_coeff > 0:
            bc_error = F.mse_loss(actor_action, act)
            bc_loss = self.bc_coeff * bc_error
            # Normalize Q term by |Q|, keep BC at face value
            actor_loss = (actor_loss / Qs.abs().mean().detach().clamp_min(1e-3)) + bc_loss

        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_opt.step()

        # ---- EMA target updates ----
        self._ema(self.phi_t, self.phi, self.tau_enc)
        self._ema(self.psi_t, self.psi, self.tau_enc)
        for t_net, o_net in zip(self.T_phi_t, self.T_phi):
            self._ema(t_net, o_net, self.tau_pred)
        for t_net, o_net in zip(self.T_psi_t, self.T_psi):
            self._ema(t_net, o_net, self.tau_pred)

        return {
            "td_loss": td_loss.item(),
            "reg_loss": (phi_orth_loss + psi_orth_loss).item(),
            "actor_loss": actor_loss.item(),
            "bc_loss": bc_error.item(),
            "q_mean": Q.mean().item(),
        }

    # -------------------------------------------------------------- inference

    @torch.no_grad()
    def infer_z_from_goal(self, goal_obs: torch.Tensor) -> torch.Tensor:
        """z = psi(g), already L2-normalised."""
        return self.psi(goal_obs)

    @torch.no_grad()
    def infer_z_from_rewards(
        self, obs: torch.Tensor, rewards: torch.Tensor, eps: float = 1e-3
    ) -> torch.Tensor:
        """Closed-form z from reward inference (matching reference)."""
        psi_s = self.psi(obs)
        z = torch.linalg.lstsq(psi_s, rewards).solution.T
        return math.sqrt(self.d_z) * F.normalize(z, dim=-1)

    @torch.no_grad()
    def act(self, obs: torch.Tensor, z: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        actor_in = self.phi(obs) if self.actor_uses_encoder else obs
        dist = self.actor(actor_in, z, self.actor_std)
        if deterministic:
            return dist.mean
        return dist.sample(clip=self.stddev_clip)

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