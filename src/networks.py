import math
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from sympy.ntheory.residue_ntheory import n_order
from typing_extensions import override


class EncoderConfig:
    def __init__(
        self, n_hidden: int, width: int, normalize_out: bool = True, siren_encoder=False
    ):
        self.n_hidden = n_hidden
        self.width = width
        self.normalize_out = normalize_out
        self.siren_encoder = siren_encoder

    def build_network(self, s_dim, out_dim):

        if self.siren_encoder:
            return SIRENEncoder(
                s_dim,
                out_dim,
                self.width,
                self.n_hidden,
                normalize_representation=self.normalize_out,
            )

        return StateEncoder(
            s_dim,
            out_dim,
            self.width,
            self.n_hidden,
            normalize_representation=self.normalize_out,
        )


class FbConfig:
    def __init__(self, width: int, n_sa: int, n_sz: int, n_shared: int):
        self.width = width
        self.n_sa = n_sa
        self.n_sz = n_sz
        self.n_shared = n_shared

    def build_network(self, s_dim, a_dim, z_dim, out_dim):
        return FB_Predictor(
            s_dim,
            a_dim,
            z_dim,
            out_dim,
            hidden_width=self.width,
            n_sa=self.n_sa,
            n_sz=self.n_sz,
            n_shared=self.n_shared,
        )


class ActorConfig:
    def __init__(self, width: int, depth: int):
        self.width = width
        self.depth = depth

    def build_network(self, s_dim, z_dim, act_dim):
        layers = [self.width] * self.depth
        return Actor(s_dim, z_dim, act_dim, hidden_dims=layers)


def _build_mlp(
    input_dim: int,
    output_dim: int,
    hidden_dims,
    input_activation=nn.Tanh,
    activation=nn.ReLU,
):
    layers = []
    layers += [
        nn.Linear(input_dim, hidden_dims[0]),
        nn.LayerNorm(hidden_dims[0]),
        input_activation(),
    ]

    last_dim = hidden_dims[0]
    for i in hidden_dims[1:]:
        layers.append(nn.Linear(last_dim, i))
        layers.append(activation())
        last_dim = i
    layers.append(torch.nn.Linear(last_dim, output_dim))

    for m in layers:
        if isinstance(m, torch.nn.Linear):
            nn.init.orthogonal_(m.weight)
            nn.init.zeros_(m.bias)
    return nn.Sequential(*layers)


def _build_embedding(
    input_dim: int,
    output_dim: int,
    width: int,
    depth: int,
    input_activation=nn.Tanh,
    hidden_activation=nn.ReLU,
):

    seq = [
        nn.Linear(input_dim, width),
        nn.LayerNorm(width),
        input_activation(),
    ]
    for i in range(1, depth):
        seq += [nn.Linear(width, width), hidden_activation()]
    seq += [nn.Linear(width, output_dim)]

    return nn.Sequential(*seq)


class SIRENEncoder(nn.Module):
    class SineLayer(nn.Module):
        """SIREN layer: sin(omega_0 * (Wx+b)) with Sitzmann et al. (2020) init."""

        def __init__(self, in_features, out_features, is_first=False, omega_0=16.0):
            super().__init__()
            self.in_features = in_features
            self.omega_0 = omega_0
            self.linear = nn.Linear(in_features, out_features)
            with torch.no_grad():
                bound = (
                    (1.0 / in_features)
                    if is_first
                    else (math.sqrt(6.0 / in_features) / omega_0)
                )
                self.linear.weight.uniform_(-bound, bound)
                self.linear.bias.zero_()

        def forward(self, x):
            return torch.sin(self.omega_0 * self.linear(x))

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        width: int,
        depth: int,
        activation=nn.ReLU,
        normalize_representation=True,
    ):
        super().__init__()

        self.embed = nn.Sequential(
            SIRENEncoder.SineLayer(input_dim, width, is_first=True),
            SIRENEncoder.SineLayer(width, width),
        )

        self.net = nn.Sequential(
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, output_dim),
        )

        def init_orthogonal(module):
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.net.apply(init_orthogonal)

        self.output_dim = output_dim
        self.normalize_represenation = normalize_representation

    @override
    def forward(self, obs):
        obs = 2 * (((obs + 2) / 25) - 0.5)
        z = self.net(self.embed(obs))

        if self.normalize_represenation:
            z = math.sqrt(self.output_dim) * F.normalize(z, dim=-1)
        return z


class StateEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        width: int,
        depth: int,
        activation=nn.ReLU,
        normalize_representation=True,
    ):
        super().__init__()
        self.net = _build_embedding(
            input_dim,
            output_dim,
            width,
            depth,
        )
        self.output_dim = output_dim
        self.normalize_represenation = normalize_representation
        self.apply(self._init_weights)

    @override
    def forward(self, obs):
        z = self.net(obs)
        if self.normalize_represenation:
            z = math.sqrt(self.output_dim) * F.normalize(z, dim=-1)
        return z

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


class FB_Predictor(nn.Module):
    def __init__(
        self,
        input_encoder_dim: int,
        action_dim: int,
        z_dim: int,
        output_encoder_dim: int,
        hidden_width: int = 512,
        n_sa: int = 2,
        n_sz: int = 2,
        n_shared: int = 2,
        n_parallel: int = 2,
    ):
        super().__init__()
        self.embed_sa = _build_embedding(
            input_encoder_dim + action_dim,
            hidden_width // 2,
            hidden_width,
            n_sa,
        )
        self.embed_sa2 = _build_embedding(
            input_encoder_dim + action_dim,
            hidden_width // 2,
            hidden_width,
            n_sa,
        )

        self.embed_sz = _build_embedding(
            input_encoder_dim + z_dim,
            hidden_width // 2,
            hidden_width,
            n_sz,
        )
        self.embed_sz2 = _build_embedding(
            input_encoder_dim + z_dim,
            hidden_width // 2,
            hidden_width,
            n_sz,
        )

        self.body = nn.Sequential(
            nn.Linear(hidden_width, hidden_width),
            nn.ReLU(),
            nn.Linear(hidden_width, hidden_width),
            nn.ReLU(),
            nn.Linear(hidden_width, output_encoder_dim),
        )
        self.body2 = nn.Sequential(
            nn.Linear(hidden_width, hidden_width),
            nn.ReLU(),
            nn.Linear(hidden_width, hidden_width),
            nn.ReLU(),
            nn.Linear(hidden_width, output_encoder_dim),
        )

        self.output_dim = output_encoder_dim
        self.apply(self._init_weights)

    @override
    def forward(self, latent_obs, action, z):
        sa = torch.cat([latent_obs, action], dim=-1)
        sz = torch.cat([latent_obs, z], dim=-1)
        h1 = torch.cat([self.embed_sa(sa), self.embed_sz(sz)], dim=-1)
        h2 = torch.cat([self.embed_sa2(sa), self.embed_sz2(sz)], dim=-1)
        out = torch.stack([self.body(h1), self.body2(h2)])
        return out

    def predict(self, latent_obs, action, z):
        out = self.forward(latent_obs, action, z)
        return (out[0] + out[1]) / 2

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


class Actor(nn.Module):
    def __init__(
        self,
        state_dim,
        z_dim,
        action_dim,
        hidden_dims=(128, 128, 128),
        activation=nn.ReLU,
    ):
        super().__init__()
        self.net = _build_mlp(state_dim + z_dim, action_dim, hidden_dims, activation)

    @override
    def forward(self, obs, z):
        input = torch.cat([obs, z], dim=-1)
        return torch.tanh(self.net(input))

    def sample_action(self, obs, z, stddev=0.01, stddev_clip=0.03):
        act = self.forward(obs, z)
        noise = torch.randn_like(act) * stddev
        noise = noise.clamp(-stddev_clip, stddev_clip)
        return (act + noise).clamp(-1.0, 1.0)
