import copy
from collections.abc import Iterable

import torch
from typing_extensions import override

from networks import Actor, FB_Predictor, StateEncoder


class TD_JEPA_CONFIG:
    def __init__(
        self,
        obs_dim,
        act_dim,
        phi_dim=64,
        psi_dim=64,
        # architecture
        phi_hidden_layers=(256, 256, 256, 256),
        psi_hidden_layers=(256, 256, 256, 256),
        T_phi_hidden_layers=(512, 512, 512, 512),
        T_psi_hidden_layers=(512, 512, 512, 512),
        actor_hidden_layers=(256, 256, 56),
        # learning rates
        phi_lr=1e-4,
        psi_lr=1e-4,
        T_phi_lr=1e-4,
        T_psi_lr=1e-4,
        actor_lr=1e-4,
        ema_coiff=0.99,
        # other hypers
        gamma=0.99,
        reg_lambda=1.0,
        actor_use_raw_obs=False,
        latent_z_batch_size=1,  # (D)
        train_device="cuda",
        eval_device="cpu",
    ) -> None:
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.phi_dim = phi_dim
        self.psi_dim = psi_dim

        self.phi_hidden_layers = phi_hidden_layers
        self.psi_hidden_layers = psi_hidden_layers
        self.T_phi_hidden_layers = T_phi_hidden_layers
        self.T_psi_hidden_layers = T_psi_hidden_layers

        self.phi_lr = phi_lr
        self.psi_lr = psi_lr
        self.T_phi_lr = T_phi_lr
        self.T_psi_lr = T_psi_lr
        self.actor_lr = actor_lr
        self.ema_coiff = ema_coiff

        self.gamma = gamma
        self.reg_lambda = reg_lambda
        self.actor_use_raw_obs = actor_use_raw_obs
        self.latent_z_batch_size = latent_z_batch_size
        self.train_device = train_device
        self.eval_device = eval_device


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

        self.phi = (
            StateEncoder(config.obs_dim, config.phi_dim, config.phi_hidden_layers)
            if not phi
            else phi
        )
        self.psi = StateEncoder(config.obs_dim, config.psi_dim) if not psi else psi
        self.T_phi = (
            FB_Predictor(
                input_encoder_dim=config.phi_dim,
                action_dim=config.act_dim,
                z_dim=config.psi_dim,
                output_encoder_dim=config.psi_dim,
                hidden_dims=config.T_psi_hidden_layers,
            )
            if not T_phi
            else T_phi
        )
        self.T_psi = (
            FB_Predictor(
                input_encoder_dim=config.psi_dim,
                action_dim=config.act_dim,
                z_dim=config.psi_dim,
                output_encoder_dim=config.phi_dim,
                hidden_dims=config.T_psi_hidden_layers,
            )
            if not T_psi
            else T_psi
        )  # (D_psi, D_a, D_z) -> (D_phi)

        if actor is not None:
            self.actor = actor
        elif config.actor_use_raw_obs:
            self.actor = Actor(config.obs_dim, config.psi_dim, config.act_dim)
        else:
            self.actor = Actor(config.phi_dim, config.psi_dim, config.act_dim)

        self.train_mode()
        # --- Target networks (EMA on encoders + predictors only; not actor)
        self.phi_t = copy.deepcopy(self.phi).requires_grad_(False)
        self.psi_t = copy.deepcopy(self.psi).requires_grad_(False)
        self.T_phi_t = copy.deepcopy(self.T_phi).requires_grad_(False)
        self.T_psi_t = copy.deepcopy(self.T_psi).requires_grad_(False)

        # --- Separate optimisers (one per network), matching the reference
        self.opt_phi = torch.optim.Adam(self.phi.parameters(), lr=config.phi_lr)
        self.opt_psi = torch.optim.Adam(self.psi.parameters(), lr=config.psi_lr)
        self.opt_T_phi = torch.optim.Adam(self.T_phi.parameters(), lr=config.T_phi_lr)
        self.opt_T_psi = torch.optim.Adam(self.T_psi.parameters(), lr=config.T_psi_lr)
        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)

    def train_mode(
        self,
    ):
        self.phi.to(self.train_device)
        self.psi.to(self.train_device)
        self.T_phi.to(self.train_device)
        self.T_psi.to(self.train_device)
        self.actor.to(self.train_device)
        self.device = self.train_device

    @torch.no_grad()
    def _sample_latent_z(self, B):

        z = torch.rand(B, self.config.psi_dim, device=self.device)
        z = z / torch.sum(torch.square(z), dim=-1, keepdim=True)

        return z

    def ema(self):
        def _ema(source, target, tau):
            with torch.no_grad():
                for tp, sp in zip(target.parameters(), source.parameters()):
                    tp.data.mul_(1.0 - tau).add_(sp.data, alpha=tau)

        tau = self.config.ema_coiff
        _ema(self.phi, self.phi_t, tau)
        _ema(self.psi, self.psi_t, tau)
        _ema(self.T_phi, self.T_phi_t, tau)
        _ema(self.T_psi, self.T_psi_t, tau)

    def train_step(self, states, actions, next_states, terminals, log=False):
        """
        states: (B, s)
        actions: (B, s)
        next_states: (B, s)
        terminals: (B, 1)
        """
        B = states.shape[0]
        D = self.config.latent_z_batch_size
        z = self._sample_latent_z(B * D)
        # compute EMA targets
        with torch.no_grad():
            phi_embed_t = self.phi_t(next_states).repeat(
                self.config.latent_z_batch_size, 1
            )
            psi_embed_t = self.psi_t(next_states).repeat(
                self.config.latent_z_batch_size, 1
            )

            if self.config.actor_use_raw_obs:
                next_actions = self.actor.sample_action(next_states.repeat(D, 1), z)
            else:
                next_actions = self.actor.sample_action(
                    self.psi(next_states).repeat(D, 1), z
                )
            bootstrap = 1 - terminals.repeat(D, 1)
            phi_target = psi_embed_t + bootstrap * self.config.gamma * self.T_phi_t(
                phi_embed_t, next_actions, z
            )
            psi_target = phi_embed_t + bootstrap * self.config.gamma * self.T_psi_t(
                psi_embed_t, next_actions, z
            )

        phi_embed = self.phi(states).repeat(D, 1)
        psi_embed = self.psi(states).repeat(D, 1)
        F_phi = self.T_phi(phi_embed, actions, z)
        F_psi = self.T_psi(psi_embed, actions, z)

        TD_phi = 0.5 * torch.mean(torch.square(phi_target - F_phi))
        TD_psi = 0.5 * torch.mean(torch.square(psi_target - F_psi))

        def _ortho_loss(enc):
            B = enc.shape[0]
            cov = enc @ enc.T
            off_mask = 1.0 - torch.eye(B, device=enc.device)
            diag = cov.diag()
            offdiag = torch.square(cov * off_mask)

            return (1 / (2 * B * (B - 1))) * offdiag.sum() - (1 / B) * diag.sum()

        Ortho_phi = self.config.reg_lambda * _ortho_loss(phi_embed[:B])
        Ortho_psi = self.config.reg_lambda * _ortho_loss(psi_embed[:B])

        L_phi = TD_phi + Ortho_phi
        self.opt_phi.zero_grad()
        self.opt_T_phi.zero_grad()
        L_phi.backward()
        self.opt_phi.step()
        self.opt_T_phi.step()

        L_psi = TD_psi + Ortho_psi
        self.opt_psi.zero_grad()
        self.opt_T_psi.zero_grad()
        L_psi.backward()
        self.opt_psi.step()
        self.opt_T_psi.step()

        self.ema()

        if self.config.actor_use_raw_obs:
            actor_actions = self.actor.sample_action(
                states.repeat(self.config.latent_z_batch_size, 1), z
            )
        else:
            phi_embed = phi_embed.detach()
            actor_actions = self.actor.sample_action(phi_embed, z)

        L_actor = torch.mean(
            torch.einsum(
                "Bi, Bi -> B",
                self.T_phi(phi_embed, actor_actions, z),
                z,
            )
        )
        self.opt_actor.zero_grad()
        L_actor.backward()
        self.opt_actor.step()

        if log:
            return {
                "phi td error": TD_phi.detach().item(),
                "psi td error": TD_psi.detach().item(),
                "phi ortho loss": Ortho_phi.detach().item(),
                "psi ortho loss": Ortho_psi.detach().item(),
                "actor loss": L_actor.detach().item(),
                "mean q": torch.mean(torch.einsum("Bi, Bi -> B", F_phi, z))
                .detach()
                .item(),
            }

    def eval_mode(
        self,
    ):
        self.phi.to(self.eval_device)
        self.psi.to(self.eval_device)
        self.T_phi.to(self.eval_device)
        self.T_psi.to(self.eval_device)
        self.actor.to(self.eval_device)
        self.device = self.eval_device

    @torch.no_grad
    def fit_reward(self, states, rewards):
        psi = self.psi(states.to(self.device)).to(rewards.device)
        return torch.mean((psi * rewards) / torch.einsum("Bi, Bi -> B", psi, psi))

    @torch.no_grad
    def act(self, states, z):
        states = states.to(self.device)
        z = z.to(self.device)
        if not self.config.actor_use_raw_obs:
            states = self.psi(states)
        return self.actor(states, z)
