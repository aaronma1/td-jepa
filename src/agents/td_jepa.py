import copy
import math
import os

import torch
import torch.nn as nn

from agents.networks import ActorConfig, EncoderConfig, FbConfig

torch.set_default_dtype(torch.float32)


class TD_JEPA_CONFIG:
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        phi_dim: int,
        psi_dim: int,
        # architecture
        phi_config: EncoderConfig,
        psi_config: EncoderConfig,
        T_phi_config: FbConfig,
        T_psi_config: FbConfig,
        actor_config: ActorConfig,
        # learning rates
        phi_lr=1e-4,
        psi_lr=1e-4,
        T_phi_lr=1e-4,
        T_psi_lr=1e-4,
        actor_lr=1e-4,
        adam_eps=1e-8,  # torch default; what every run so far actually used
        tau_predictor=0.005,
        tau_encoder=0.005,
        tau_z_cov=0.005,
        # main hypers
        gamma=0.99,
        actor_use_raw_obs=False,
        # other hypers
        lambda_phi=1.0,
        lambda_psi=1.0,
        actor_bc_coeff=0.3,
        actor_explore_stddev=0.1,
        actor_explore_clip=0.3,
        actor_grad_norm=1.0,  # <=0 disables actor grad clipping
        ensemble_mean_targets=False,  # True: both heads regress the target-ensemble mean (BSynth/metamotivo)
        actor_reencode_phi=False,  # True: actor sees phi(s) from the just-updated phi (BSynth/metamotivo)
        latent_z_batch_size=1,  # (D)
        train_goal_ratio=0.5,
        scale_train_goals=True,
        train_device="cuda",
        eval_device="cpu",
    ) -> None:
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.phi_dim = phi_dim
        self.psi_dim = psi_dim

        self.phi_config = phi_config
        self.psi_config = psi_config
        self.T_phi_config = T_phi_config
        self.T_psi_config = T_psi_config
        self.actor_config = actor_config

        self.actor_use_raw_obs = actor_use_raw_obs

        self.phi_lr = phi_lr
        self.psi_lr = psi_lr
        self.T_phi_lr = T_phi_lr
        self.T_psi_lr = T_psi_lr
        self.actor_lr = actor_lr
        self.adam_eps = adam_eps
        # ema weights
        self.tau_predictor = tau_predictor
        self.tau_encoder = tau_encoder
        self.tau_z_cov = tau_z_cov

        self.gamma = gamma
        self.latent_z_batch_size = latent_z_batch_size
        self.train_goal_ratio = train_goal_ratio
        self.lambda_phi = lambda_phi
        self.lambda_psi = lambda_psi
        self.scale_train_goals = scale_train_goals
        self.actor_bc_coeff = actor_bc_coeff
        self.actor_explore_stddev = actor_explore_stddev
        self.actor_explore_clip = actor_explore_clip
        self.actor_grad_norm = actor_grad_norm
        self.ensemble_mean_targets = ensemble_mean_targets
        self.actor_reencode_phi = actor_reencode_phi
        self.train_device = train_device
        self.eval_device = eval_device

    def make_networks(self):
        phi = self.phi_config.build_network(self.obs_dim, self.phi_dim)
        psi = self.psi_config.build_network(self.obs_dim, self.psi_dim)
        T_phi = self.T_phi_config.build_network(
            self.phi_dim, self.act_dim, self.psi_dim, self.psi_dim
        )
        T_psi = self.T_psi_config.build_network(
            self.psi_dim, self.act_dim, self.psi_dim, self.phi_dim
        )

        if self.actor_use_raw_obs:
            actor = self.actor_config.build_network(
                self.obs_dim, self.psi_dim, self.act_dim
            )
        else:
            actor = self.actor_config.build_network(
                self.phi_dim, self.psi_dim, self.act_dim
            )
        return phi, psi, T_phi, T_psi, actor


def _tile(x, D):
    B = x.size(0)
    return x.unsqueeze(0).expand(D, B, -1).reshape(B * D, -1)


def _ortho_loss(enc):
    B = enc.shape[0]
    cov = enc @ enc.T
    diag = cov.diag()
    offdiag = torch.square(cov).sum() - torch.square(diag).sum()
    return (1 / (2 * B * (B - 1))) * offdiag - (1 / B) * diag.sum()


class TD_JEPA:
    def __init__(
        self,
        config: TD_JEPA_CONFIG,
        phi=None,
        psi=None,
        T_phi=None,
        T_psi=None,
        actor=None,
    ) -> None:
        self.config = config
        self.train_device = config.train_device
        self.eval_device = config.eval_device
        self.device = None

        c_phi, c_psi, c_T_phi, c_T_psi, c_actor = config.make_networks()

        self.phi = c_phi if not phi else phi
        self.psi = c_psi if not psi else psi
        self.T_phi = c_T_phi if not T_phi else T_phi
        self.T_psi = c_T_psi if not T_psi else T_psi
        self.actor = c_actor if not actor else actor

        self.train_mode()
        # --- Target networks (EMA on encoders + predictors only; not actor)
        #
        self.phi_t = copy.deepcopy(self.phi).requires_grad_(False)
        self.psi_t = copy.deepcopy(self.psi).requires_grad_(False)
        self.T_phi_t = copy.deepcopy(self.T_phi).requires_grad_(False)
        self.T_psi_t = copy.deepcopy(self.T_psi).requires_grad_(False)
        self._z_cov = torch.eye(
            self.config.psi_dim, device=self.train_device, requires_grad=False
        )

        # --- Separate optimisers (one per network), matching the reference
        self.opt_actor = torch.optim.Adam(
            self.actor.parameters(), lr=config.actor_lr, eps=config.adam_eps
        )
        self.opt_critic = torch.optim.Adam(
            [
                {"params": self.phi.parameters(), "lr": config.phi_lr},
                {"params": self.psi.parameters(), "lr": config.psi_lr},
                {"params": self.T_phi.parameters(), "lr": config.T_phi_lr},
                {"params": self.T_psi.parameters(), "lr": config.T_psi_lr},
            ],
            eps=config.adam_eps,
        )

    def train_mode(self):
        self.phi.to(self.train_device)
        self.psi.to(self.train_device)
        self.T_phi.to(self.train_device)
        self.T_psi.to(self.train_device)
        self.actor.to(self.train_device)
        self.device = self.train_device

    def eval_mode(self):
        self.phi.to(self.eval_device)
        self.psi.to(self.eval_device)
        self.T_phi.to(self.eval_device)
        self.T_psi.to(self.eval_device)
        self.actor.to(self.eval_device)
        self.device = self.eval_device

    @torch.no_grad()
    def _project_z(self, z):
        z = (math.sqrt(self.config.psi_dim) * z) / z.norm(
            dim=-1, keepdim=True
        ).clamp_min(1e-8)
        return z

    @torch.no_grad()
    def compute_td_targets(self, next_states, terminals, z):
        D = self.config.latent_z_batch_size
        phi_embed_t = _tile(self.phi_t(next_states), D)
        psi_embed_t = _tile(self.psi_t(next_states), D)
        ns = _tile(next_states, D) if self.config.actor_use_raw_obs else phi_embed_t
        next_actions = self.actor.sample_action(
            ns, z, self.config.actor_explore_stddev, self.config.actor_explore_clip
        )
        bootstrap = 1 - _tile(terminals, D)

        F_phi = self.T_phi_t(phi_embed_t, next_actions, z)  # (2, BD, psi_dim)
        F_psi = self.T_psi_t(psi_embed_t, next_actions, z)  # (2, BD, phi_dim)
        if self.config.ensemble_mean_targets:
            F_phi = F_phi.mean(0, keepdim=True).expand_as(F_phi)
            F_psi = F_psi.mean(0, keepdim=True).expand_as(F_psi)

        phi_target = (
            psi_embed_t.unsqueeze(0).repeat(2, 1, 1)
            + self.config.gamma * bootstrap * F_phi
        )
        psi_target = (
            phi_embed_t.unsqueeze(0).repeat(2, 1, 1)
            + self.config.gamma * bootstrap * F_psi
        )

        return phi_target, psi_target

    def train_step(self, states, actions, next_states, terminals, log=False):
        """
        states: (B, s)
        actions: (B, s)
        next_states: (B, s)
        terminals: (B, 1)
        """
        states = states.contiguous()
        actions = actions.contiguous()
        next_states = next_states.contiguous()
        terminals = terminals.contiguous()
        B = states.shape[0]
        D = self.config.latent_z_batch_size

        with torch.no_grad():
            z = self._sample_mixed_z(next_states, D)
            phi_target, psi_target = self.compute_td_targets(next_states, terminals, z)

        phi_embed = _tile(self.phi(states), D)
        psi_embed = _tile(self.psi(states), D)

        F_phi = self.T_phi(phi_embed, _tile(actions, D), z)
        F_psi = self.T_psi(psi_embed, _tile(actions, D), z)
        TD_phi = torch.square(phi_target - F_phi).sum(-1).mean()
        TD_psi = torch.square(psi_target - F_psi).sum(-1).mean()
        Ortho_phi = self.config.lambda_phi * _ortho_loss(phi_embed[:B])
        Ortho_psi = self.config.lambda_psi * _ortho_loss(psi_embed[:B])

        self.opt_critic.zero_grad(set_to_none=True)
        L = TD_phi + Ortho_phi + TD_psi + Ortho_psi
        L.backward()
        self.opt_critic.step()

        with torch.no_grad():
            self._z_cov = (
                psi_embed[:B].T @ psi_embed[:B]
            ) / B * self.config.tau_z_cov + (1 - self.config.tau_z_cov) * self._z_cov

        self.ema()
        # compute actor loss
        self.T_phi.requires_grad_(False)
        if self.config.actor_reencode_phi:
            with torch.no_grad():
                phi_actor = _tile(self.phi(states), D)
        else:
            phi_actor = phi_embed.detach()
        actor_input = _tile(states, D) if self.config.actor_use_raw_obs else phi_actor
        actor_actions = self.actor.sample_action(
            actor_input, z, self.config.actor_explore_stddev, self.config.actor_explore_clip
        )
        F_actor = self.T_phi.predict(phi_actor, actor_actions, z)
        Qs = torch.einsum("Bi,Bi->B", F_actor, z)
        if self.config.actor_bc_coeff > 0:
            Qs = Qs / (Qs.abs().mean().detach())

        # compute BC regularization
        data_actions = _tile(actions, D)
        bc = ((actor_actions - data_actions) ** 2).mean()

        # full actor loss
        L_actor = -Qs.mean() + self.config.actor_bc_coeff * bc
        self.opt_actor.zero_grad(set_to_none=True)
        L_actor.backward()
        if self.config.actor_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.actor.parameters(), self.config.actor_grad_norm
            )
        self.opt_actor.step()
        self.T_phi.requires_grad_(True)

        if log:
            return {
                "phi td error": TD_phi.detach().item(),
                "psi td error": TD_psi.detach().item(),
                "phi ortho loss": Ortho_phi.detach().item(),
                "psi ortho loss": Ortho_psi.detach().item(),
                "actor loss": L_actor.detach().item(),
                "actor q loss": -Qs.mean().detach().item(),
                "var q": Qs.var().detach().item(),
                "actor bc loss": (bc.mean() * self.config.actor_bc_coeff)
                .detach()
                .item(),
                "mean q actor": torch.mean(torch.einsum("Bi, Bi -> B", F_actor, z))
                .detach()
                .item(),
                "mean q data": torch.mean(
                    torch.einsum("Bi, Bi -> B", (F_phi[0] + F_phi[1]) / 2, z)
                )
                .detach()
                .item(),
            }

    @torch.no_grad()
    def fit_reward(self, states, rewards):
        states = states.to(self.device)
        rewards = rewards.to(self.device).view(-1, 1)
        psi = self.psi(states)
        z = torch.linalg.lstsq(psi, rewards).solution.squeeze(-1)  # (psi_dim,)
        return (self.config.psi_dim**0.5) * z / z.norm().clamp_min(1e-8)

    @torch.no_grad()
    def act(self, states, z):
        states = states.to(self.device)
        z = z.to(self.device)
        if not self.config.actor_use_raw_obs:
            states = self.phi(states)
        return self.actor(states, z)

    @torch.no_grad()
    def _sample_mixed_z(self, next_states, D):
        """
        Sample latent z mixed pwith psi(next_states)
        """
        B = next_states.shape[0]
        BD = B * D
        z = self._project_z(torch.randn(BD, self.config.psi_dim, device=self.device))
        if self.config.train_goal_ratio <= 0.0:
            return z
        next_tiled = _tile(next_states, D)
        perm = torch.randperm(BD, device=self.device)
        goals = self.psi(next_tiled[perm])
        if self.config.scale_train_goals:
            eye = torch.eye(self.config.psi_dim, device=self.device)
            inv_cov = torch.linalg.inv(self._z_cov + 1e-6 * eye)
            goals = goals @ inv_cov
        goals = self._project_z(goals)

        mask = torch.rand((BD, 1), device=self.device) < self.config.train_goal_ratio
        return torch.where(mask, goals, z)

    def ema(self):

        def _ema(target: nn.Module, source: nn.Module, tau) -> None:
            with torch.no_grad():
                for tgt, src in zip(target.parameters(), source.parameters()):
                    tgt.data.mul_(1.0 - tau).add_(src.data, alpha=tau)

        _ema(self.phi_t, self.phi, self.config.tau_encoder)
        _ema(self.psi_t, self.psi, self.config.tau_encoder)
        _ema(self.T_phi_t, self.T_phi, self.config.tau_predictor)
        _ema(self.T_psi_t, self.T_psi, self.config.tau_predictor)

    # --- Checkpointing -----------------------------------------------------

    _MODULES = ("phi", "psi", "T_phi", "T_psi", "actor", "phi_t", "psi_t", "T_phi_t", "T_psi_t")

    def state_dict(self, include_optimizers=True):
        sd = {
            "modules": {n: getattr(self, n).state_dict() for n in self._MODULES},
            "z_cov": self._z_cov.detach().clone(),
        }
        if include_optimizers:
            sd["optimizers"] = {
                "opt_actor": self.opt_actor.state_dict(),
                "opt_critic": self.opt_critic.state_dict(),
            }
        return sd

    def load_state_dict(self, sd):
        for n, msd in sd["modules"].items():
            getattr(self, n).load_state_dict(msd)
        self._z_cov = sd["z_cov"].to(self.train_device)
        for n, osd in sd.get("optimizers", {}).items():
            getattr(self, n).load_state_dict(osd)

    def save_state(self, save_path, include_optimizers=True, **extra):
        """Atomically write agent state (+ any `extra` entries) to `save_path`."""
        ckpt = {**extra, **self.state_dict(include_optimizers)}
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        tmp = save_path + ".tmp"
        torch.save(ckpt, tmp)
        os.replace(tmp, save_path)

    def load_state(self, save_path):
        """Load agent state from `save_path`; returns the full checkpoint dict."""
        ckpt = torch.load(save_path, map_location=self.train_device, weights_only=True)
        self.load_state_dict(ckpt)
        self.train_mode()
        return ckpt
