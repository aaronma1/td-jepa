"""TD-JEPA networks: encoders (phi / psi), twin-head predictors (T_phi / T_psi), actor.

Every architectural difference between the two flavours we train is a config knob, so
the params file fully determines the network (.params/tdj_*_params.yaml):

  OGBench (this repo's tuned antmaze/pointmaze config — the knob defaults)
    * predictor embeddings end in a Linear (no activation), orthogonal gain 1
    * actor heads: full-width embeddings with torch default init
    * actor body: Linear -> LayerNorm -> ReLU -> ..., hard clamp of noisy actions

  DMC (BSynth JAX port == official metamotivo td_jepa)
    * encoder with n_hidden=0 is a single Linear (+ norm)
    * predictor embeddings end in Linear -> ReLU, orthogonal gain sqrt(2)
    * actor heads: half-width embeddings ending in ReLU, orthogonal init
    * actor body without LayerNorm, straight-through clamp of noisy actions
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing_extensions import override


# ─── Building blocks ────────────────────────────────────────────────────────


def mlp(in_dim, out_dim, width, depth, first_act=nn.Tanh, layernorm=True, out_relu=False):
    """Linear(in, w) [-> LayerNorm] -> first_act -> [Linear(w, w) -> ReLU] * (depth - 1)
    -> Linear(w, out) [-> ReLU].  depth < 1 behaves like depth 1."""
    layers = [nn.Linear(in_dim, width)]
    if layernorm:
        layers.append(nn.LayerNorm(width))
    layers.append(first_act())
    for _ in range(depth - 1):
        layers += [nn.Linear(width, width), nn.ReLU()]
    layers.append(nn.Linear(width, out_dim))
    if out_relu:
        layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def orthogonal_init_(module, gain=1.0):
    """Orthogonal weights / zero biases on every Linear in `module`; LayerNorm reset."""
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=gain)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


def norm_z(z):
    """Project to ||z|| = sqrt(d)."""
    return math.sqrt(z.shape[-1]) * F.normalize(z, dim=-1)


# ─── Encoders ────────────────────────────────────────────────────────────────


class StateEncoder(nn.Module):
    """Raw obs -> R^out_dim, orthogonal init (gain 1), optionally normed to sqrt(d).

    n_hidden=0: single Linear (official BackwardMap(hidden_layers=0)).
    n_hidden>=1: mlp(depth=n_hidden) = Linear -> LN -> Tanh -> [Linear -> ReLU]*(n-1) -> Linear.
    """

    def __init__(self, in_dim, out_dim, width, n_hidden, normalize_out=True):
        super().__init__()
        if n_hidden == 0:
            self.net = nn.Sequential(nn.Linear(in_dim, out_dim))
        else:
            self.net = mlp(in_dim, out_dim, width, n_hidden)
        self.normalize_out = normalize_out
        orthogonal_init_(self)

    @override
    def forward(self, obs):
        z = self.net(obs)
        return norm_z(z) if self.normalize_out else z


class SIRENEncoder(nn.Module):
    """Pointmaze encoder: 2 sine layers -> [Linear -> ReLU] * 2 -> Linear (fixed depth)."""

    class SineLayer(nn.Module):
        """sin(omega_0 * (Wx + b)) with Sitzmann et al. (2020) init."""

        def __init__(self, in_dim, out_dim, is_first=False, omega_0=16.0):
            super().__init__()
            self.omega_0 = omega_0
            self.linear = nn.Linear(in_dim, out_dim)
            bound = (1.0 / in_dim) if is_first else (math.sqrt(6.0 / in_dim) / omega_0)
            with torch.no_grad():
                self.linear.weight.uniform_(-bound, bound)
                self.linear.bias.zero_()

        def forward(self, x):
            return torch.sin(self.omega_0 * self.linear(x))

    def __init__(self, in_dim, out_dim, width, normalize_out=True):
        super().__init__()
        self.embed = nn.Sequential(
            SIRENEncoder.SineLayer(in_dim, width, is_first=True),
            SIRENEncoder.SineLayer(width, width),
        )
        self.net = nn.Sequential(
            nn.Linear(width, width), nn.ReLU(),
            nn.Linear(width, width), nn.ReLU(),
            nn.Linear(width, out_dim),
        )
        orthogonal_init_(self.net)
        self.normalize_out = normalize_out

    @override
    def forward(self, obs):
        obs = 2 * (((obs + 2) / 25) - 0.5)  # pointmaze xy range -> ~[-1, 1]
        z = self.net(self.embed(obs))
        return norm_z(z) if self.normalize_out else z


# ─── Predictor ───────────────────────────────────────────────────────────────


class FB_Predictor(nn.Module):
    """Twin-head F(s_emb, a, z) -> out_dim.

    Each head: embed_sa(s, a) and embed_sz(s, z), both mlp(depth=n_sa / n_sz) -> width//2,
    concatenated -> [Linear -> ReLU] * n_shared -> Linear(out).  forward() stacks the two
    heads (2, B, out); predict() is their mean.
    """

    def __init__(self, s_dim, a_dim, z_dim, out_dim, width=512, n_sa=2, n_sz=2,
                 n_shared=2, embed_out_relu=False, init_gain=1.0):
        super().__init__()
        half = width // 2
        sa = lambda: mlp(s_dim + a_dim, half, width, n_sa, out_relu=embed_out_relu)
        sz = lambda: mlp(s_dim + z_dim, half, width, n_sz, out_relu=embed_out_relu)
        self.embed_sa, self.embed_sa2 = sa(), sa()
        self.embed_sz, self.embed_sz2 = sz(), sz()
        self.body, self.body2 = self._body(width, out_dim, n_shared), self._body(width, out_dim, n_shared)
        orthogonal_init_(self, init_gain)

    @staticmethod
    def _body(width, out_dim, n_shared):
        layers = []
        for _ in range(n_shared):
            layers += [nn.Linear(width, width), nn.ReLU()]
        return nn.Sequential(*layers, nn.Linear(width, out_dim))

    @override
    def forward(self, s_emb, action, z):
        sa = torch.cat([s_emb, action], dim=-1)
        sz = torch.cat([s_emb, z], dim=-1)
        h1 = torch.cat([self.embed_sa(sa), self.embed_sz(sz)], dim=-1)
        h2 = torch.cat([self.embed_sa2(sa), self.embed_sz2(sz)], dim=-1)
        return torch.stack([self.body(h1), self.body2(h2)])

    def predict(self, s_emb, action, z):
        out = self.forward(s_emb, action, z)
        return (out[0] + out[1]) / 2


# ─── Actor ───────────────────────────────────────────────────────────────────


class Actor(nn.Module):
    """Dual-head deterministic actor: tanh(net([s(obs), sz(obs, z)])).

    Heads: mlp(depth=1) -> embed_dim (+ ReLU if embed_out_relu); torch default init
    unless embed_orth_init.  Body: mlp(2*embed_dim -> act, depth, first_act=ReLU,
    layernorm=body_layernorm), orthogonal init (gain 1).
    """

    def __init__(self, s_dim, z_dim, act_dim, width, depth, embed_dim=None,
                 embed_out_relu=False, embed_orth_init=False, body_layernorm=True,
                 straight_through_clamp=False):
        super().__init__()
        embed_dim = embed_dim or width
        self.s = mlp(s_dim, embed_dim, width, 1, out_relu=embed_out_relu)
        self.sz = mlp(s_dim + z_dim, embed_dim, width, 1, out_relu=embed_out_relu)
        self.net = mlp(2 * embed_dim, act_dim, width, depth, first_act=nn.ReLU,
                       layernorm=body_layernorm)
        orthogonal_init_(self.net)
        if embed_orth_init:
            orthogonal_init_(self.s)
            orthogonal_init_(self.sz)
        self.straight_through_clamp = straight_through_clamp

    @override
    def forward(self, obs, z):
        h = torch.cat([self.s(obs), self.sz(torch.cat([obs, z], dim=-1))], dim=-1)
        return torch.tanh(self.net(h))

    def sample_action(self, obs, z, stddev, stddev_clip):
        """Action + clipped Gaussian noise, clamped to [-1, 1].

        straight_through_clamp (metamotivo): clamp to [-1+1e-6, 1-1e-6] in the forward
        pass but pass gradients straight through, so saturated dims still train the actor.
        """
        x = self.forward(obs, z)
        x = x +(torch.randn_like(x) * stddev).clamp(-stddev_clip, stddev_clip)
        if self.straight_through_clamp:
            eps = 1e-6
            return x + (x.clamp(-1.0 + eps, 1.0 - eps) - x).detach()
        return x.clamp(-1.0, 1.0)


# ─── Configs (one per network; fields map 1:1 onto the params file) ────────


@dataclass
class EncoderConfig:
    n_hidden: int
    width: int
    normalize_out: bool = True
    siren_encoder: bool = False  # pointmaze; ignores n_hidden (fixed depth)

    def build_network(self, s_dim, out_dim):
        if self.siren_encoder:
            return SIRENEncoder(s_dim, out_dim, self.width, self.normalize_out)
        return StateEncoder(s_dim, out_dim, self.width, self.n_hidden, self.normalize_out)


@dataclass
class FbConfig:
    width: int
    n_sa: int
    n_sz: int
    n_shared: int
    embed_out_relu: bool = False
    init_gain: float = 1.0

    def build_network(self, s_dim, a_dim, z_dim, out_dim):
        return FB_Predictor(s_dim, a_dim, z_dim, out_dim, self.width, self.n_sa, self.n_sz,
                            self.n_shared, self.embed_out_relu, self.init_gain)


@dataclass
class ActorConfig:
    width: int
    depth: int
    embed_dim: int | None = None  # None -> width
    embed_out_relu: bool = False
    embed_orth_init: bool = False
    body_layernorm: bool = True
    straight_through_clamp: bool = False

    def build_network(self, s_dim, z_dim, act_dim):
        return Actor(s_dim, z_dim, act_dim, self.width, self.depth, self.embed_dim,
                     self.embed_out_relu, self.embed_orth_init, self.body_layernorm,
                     self.straight_through_clamp)
