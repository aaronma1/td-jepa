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
    layers = []
    last_dim = input_dim
    for i in hidden_dims:
        layers.append(torch.nn.Linear(last_dim, i))
        layers.append(activation())

        if normalize:
            layers.append(torch.nn.LayerNorm(i))
        last_dim = i

    layers.append(torch.nn.Linear(last_dim, output_dim))
    for m in layers:
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.orthogonal(m.weight, gain=1.0)
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

    @override
    def forward(self, obs):
        return self.net(obs)


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

    @override
    def forward(self, latent_obs, action, z):
        input = torch.cat([latent_obs, action, z], dim=-1)
        return self.net(input)


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
        return self.net(input)

    def sample_action(self, obs, z):
        return self.forward(obs, z)
