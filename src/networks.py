import math
from typing import Iterable

import torch
from typing_extensions import override


def _build_mlp(
    input_dim: int,
    output_dim: int,
    hidden_dims: Iterable[int],
    activation=torch.nn.Tanh,
    normalize=True,
):
    print(hidden_dims)
    layers = []
    last_dim = input_dim
    for i in hidden_dims:
        layers.append(torch.nn.Linear(last_dim, i))
        if normalize:
            layers.append(torch.nn.LayerNorm(i))
        layers.append(activation())
        last_dim = i
    layers.append(torch.nn.Linear(last_dim, output_dim))
    for m in layers:
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.orthogonal_(m.weight, gain=1.0)
            torch.nn.init.zeros_(m.bias)

    return torch.nn.Sequential(*layers)


class StateEncoder(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims=(256, 256, 256, 256),
        activation=torch.nn.ReLU,
    ):
        super().__init__()
        self.net = _build_mlp(input_dim, output_dim, hidden_dims)
        self.output_dim = output_dim

    @override
    def forward(self, obs):
        z = self.net(obs)
        norm = z.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        z = math.sqrt(self.output_dim) * z / norm
        return z


class FB_Predictor(torch.nn.Module):
    def __init__(
        self,
        input_encoder_dim: int,
        action_dim: int,
        z_dim: int,
        output_encoder_dim: int,
        hidden_dims=(512, 512, 512, 512),
        activation=torch.nn.ReLU,
    ):
        super().__init__()
        self.net = _build_mlp(
            input_encoder_dim + action_dim + z_dim, output_encoder_dim, hidden_dims
        )
        self.net2 = _build_mlp(
            input_encoder_dim + action_dim + z_dim, output_encoder_dim, hidden_dims
        )
        self.output_dim = output_encoder_dim

    @override
    def forward(self, latent_obs, action, z):
        input = torch.cat([latent_obs, action, z], dim=-1)
        out = 0.5 * (self.net(input) + self.net2(input))

        norm = out.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        out = math.sqrt(self.output_dim) * out / norm
        return out


class Actor(torch.nn.Module):
    def __init__(
        self,
        state_dim,
        z_dim,
        action_dim,
        hidden_dims=(128, 128, 128),
        activation=torch.nn.ReLU,
    ):
        super().__init__()
        self.net = _build_mlp(state_dim + z_dim, action_dim, hidden_dims)

    @override
    def forward(self, obs, z):
        input = torch.cat([obs, z], dim=-1)
        return torch.tanh(self.net(input))

    def sample_action(self, obs, z, stddev=0.1, stddev_clip=0.3):
        act = self.forward(obs, z)
        noise = torch.randn_like(act) * stddev
        noise = noise.clamp(-stddev_clip, stddev_clip)
        return (act + noise).clamp(-1.0, 1.0)
