import copy

import torch

from networks import Actor, FB_Predictor, StateEncoder

torch.set_default_dtype(torch.float32)


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
        actor_hidden_layers=(256, 256, 256),
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
        self.actor_hidden_layers = actor_hidden_layers

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


def _tile(x, D):
    B = x.size(0)
    return x.unsqueeze(0).expand(D, B, -1).reshape(B * D, -1)


def _ortho_loss(enc):
    B = enc.shape[0]
    cov = enc @ enc.T
    diag = cov.diag()
    offdiag = torch.square(cov - torch.diag_embed(diag))
    return (1 / (2 * B * (B - 1))) * offdiag.sum() - (1 / B) * diag.sum()


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
            StateEncoder(
                config.obs_dim, config.phi_dim, hidden_dims=config.phi_hidden_layers
            )
            if not phi
            else phi
        )
        self.psi = (
            StateEncoder(
                config.obs_dim, config.psi_dim, hidden_dims=config.psi_hidden_layers
            )
            if not psi
            else psi
        )
        self.T_phi = (
            FB_Predictor(
                input_encoder_dim=config.phi_dim,
                action_dim=config.act_dim,
                z_dim=config.psi_dim,
                output_encoder_dim=config.psi_dim,
                hidden_dims=config.T_phi_hidden_layers,
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
            self.actor = Actor(
                config.obs_dim,
                config.psi_dim,
                config.act_dim,
                hidden_dims=config.actor_hidden_layers,
            )
        else:
            self.actor = Actor(
                config.phi_dim,
                config.psi_dim,
                config.act_dim,
                hidden_dims=config.actor_hidden_layers,
            )

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

        z = 2 * torch.rand(B, self.config.psi_dim, device=self.device) - 1
        z = z / torch.norm(z, dim=-1, keepdim=True)

        return z

    def ema(self):
        def _ema(source, target, tau):
            with torch.no_grad():
                for tp, sp in zip(target.parameters(), source.parameters()):
                    tp.data.mul_(1.0 - tau).add_(sp.data, alpha=tau)

        tau = 1 - self.config.ema_coiff
        _ema(self.phi, self.phi_t, tau)
        _ema(self.psi, self.psi_t, tau)
        _ema(self.T_phi, self.T_phi_t, tau)
        _ema(self.T_psi, self.T_psi_t, tau)

    @torch.no_grad()
    def compute_td_targets(self, next_states, terminals, z):
        D = self.config.latent_z_batch_size

        phi_embed_t = _tile(self.phi_t(next_states), D)
        psi_embed_t = _tile(self.psi_t(next_states), D)

        ns = _tile(next_states, D) if self.config.actor_use_raw_obs else phi_embed_t
        next_actions = self.actor.sample_action(ns, z)

        bootstrap = 1 - _tile(terminals, D)
        phi_target = psi_embed_t + bootstrap * self.config.gamma * self.T_phi_t(
            phi_embed_t, next_actions, z
        )
        psi_target = phi_embed_t + bootstrap * self.config.gamma * self.T_psi_t(
            psi_embed_t, next_actions, z
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
        z = self._sample_latent_z(B * D)
        # compute EMA targets
        with torch.no_grad():
            phi_target, psi_target = self.compute_td_targets(next_states, terminals, z)

        phi_embed = _tile(self.phi(states), D)
        psi_embed = _tile(self.psi(states), D)
        F_phi = self.T_phi(phi_embed, _tile(actions, D), z)
        F_psi = self.T_psi(psi_embed, _tile(actions, D), z)
        TD_phi = 0.5 * torch.mean(torch.square(phi_target - F_phi))
        TD_psi = 0.5 * torch.mean(torch.square(psi_target - F_psi))
        Ortho_phi = self.config.reg_lambda * _ortho_loss(phi_embed[:B])
        Ortho_psi = self.config.reg_lambda * _ortho_loss(psi_embed[:B])
        optim = [
            self.opt_phi,
            self.opt_psi,
            self.opt_T_phi,
            self.opt_T_psi,
        ]
        for opt in optim:
            opt.zero_grad(set_to_none=True)
        L = TD_phi + Ortho_phi + TD_psi + Ortho_psi
        L.backward()
        for opt in optim:
            opt.step()
        self.ema()

        # compute actor loss
        phi_embed = phi_embed.detach()

        actor_input = (
            _tile(states, D) if self.config.actor_use_raw_obs else phi_embed.detach()
        )
        actor_actions = self.actor.sample_action(actor_input, z)
        self.T_phi.requires_grad_(False)
        F_actor = self.T_phi(phi_embed.detach(), actor_actions, z)
        L_actor = -torch.mean(torch.einsum("Bi,Bi->B", F_actor, z))
        self.opt_actor.zero_grad(set_to_none=True)
        L_actor.backward()
        self.opt_actor.step()
        self.T_phi.requires_grad_(True)

        if log:
            return {
                "phi td error": TD_phi.detach().item(),
                "psi td error": TD_psi.detach().item(),
                "phi ortho loss": Ortho_phi.detach().item(),
                "psi ortho loss": Ortho_psi.detach().item(),
                "actor loss": L_actor.detach().item(),
                "mean q actor": torch.mean(torch.einsum("Bi, Bi -> B", F_actor, z))
                .detach()
                .item(),
                "mean q data": torch.mean(torch.einsum("Bi, Bi -> B", F_phi, z))
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

    @torch.no_grad()
    def fit_reward(self, states, rewards):
        psi = self.psi(states.to(self.device)).to(rewards.device)
        return torch.mean(
            (psi * rewards) / torch.einsum("Bi, Bi -> B", psi, psi), dim=0
        )

    @torch.no_grad()
    def act(self, states, z):
        states = states.to(self.device)
        z = z.to(self.device)
        if not self.config.actor_use_raw_obs:
            states = self.phi(states)
        return self.actor(states, z)
