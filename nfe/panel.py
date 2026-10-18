"""The panel form of the filter: alternating attention over the two axes
of a bars-by-names panel.

Where :mod:`nfe.model` filters one name's window with a GRU, this
alternates time attention within each name (order-aware, with a learned
position encoding) and asset attention within each bar
(permutation-equivariant over names), then reads a mixture-density
posterior off each name's final state. On panels whose names share a
common factor in levels, the asset step can learn the factor removal
that a classical pipeline performs by rolling PCA.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .model import NFE_KMIX
from .prior import BURN_IN, L, SCALE, VOL_SD

__all__ = ["RowColumnBlock", "PanelNFE", "build_panel_nfe", "panel_simulate",
           "panel_featurize", "panel_mdn_nll", "panel_mdn_pit"]


def panel_simulate(n_panels: int, N: int, rng: np.random.Generator,
                   shared: bool = True, L: int = L) -> tuple[np.ndarray, np.ndarray]:
    """Panels of switching-OU paths. With ``shared`` the innovation
    hyper-parameters (phi_vol, s_log) are drawn once per panel, giving the
    names a common structure; otherwise once per name (the control).
    Returns ``W (P, N, L)`` and ``y (P, N)`` log2 half-life."""
    P = n_panels
    shape = (P, 1) if shared else (P, N)
    phi = rng.uniform(0.0, 0.5, shape) * np.ones((P, N))
    slog = rng.uniform(0.02, 0.30, shape) * np.ones((P, N))
    hl = np.exp(rng.uniform(np.log(3), np.log(400), (P, N)))
    k = np.log(2) / hl
    m_ln = np.exp(slog ** 2 / 2)
    sd_ln = np.sqrt((np.exp(slog ** 2) - 1) * np.exp(slog ** 2))
    x = np.zeros((P, N)); lv = np.zeros((P, N)); m = np.zeros((P, N))
    W = np.empty((P, N, L), dtype=np.float32)
    for t in range(BURN_IN + L):
        lv = phi * lv + VOL_SD * np.sqrt(1 - phi ** 2) * rng.standard_normal((P, N))
        e = (np.exp(slog * rng.standard_normal((P, N))) - m_ln) / sd_ln
        m = m + 0.05 * SCALE * rng.standard_normal((P, N))
        x = x + k * (m - x) + SCALE * np.exp(lv) * e
        if t >= BURN_IN:
            W[:, :, t - BURN_IN] = x
    return W, np.log2(hl).astype(np.float32)


def panel_featurize(W: np.ndarray) -> np.ndarray:
    """``(P, N, L)`` -> standardized ``(P, N, L, 2)``: level and increments."""
    mu = W.mean(-1, keepdims=True)
    sd = W.std(-1, keepdims=True) + 1e-12
    Z = (W - mu) / sd
    dZ = np.diff(Z, axis=-1, prepend=Z[..., :1])
    return np.stack([Z, dZ], axis=-1).astype(np.float32)


class RowColumnBlock(nn.Module):
    """One alternating block: pre-LN time attention within each name, then
    pre-LN asset attention within each bar, then a feed-forward. Submodules
    are created in a fixed order so a seeded init is reproducible."""

    def __init__(self, d: int, heads: int):
        super().__init__()
        self.ln_t = nn.LayerNorm(d)
        self.attn_time = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ln_n = nn.LayerNorm(d)
        self.attn_asset = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ln_f = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(),
                                nn.Linear(2 * d, d))

    def forward(self, x):                          # x: (B, N, L, d)
        B, N, T, D = x.shape
        h = x.reshape(B * N, T, D)                 # time step: within a name
        q = self.ln_t(h)
        h = h + self.attn_time(q, q, q, need_weights=False)[0]
        h = h.reshape(B, N, T, D).permute(0, 2, 1, 3).reshape(B * T, N, D)
        q = self.ln_n(h)                           # asset step: within a bar
        h = h + self.attn_asset(q, q, q, need_weights=False)[0]
        h = h.reshape(B, T, N, D).permute(0, 2, 1, 3)
        return h + self.ff(self.ln_f(h))


class PanelNFE(nn.Module):
    """Cell embedding + a learned time-position encoding + alternating
    row--column blocks + a mixture-density head on each name's final-bar
    state. Input ``(B, N, L, 2)``; output ``(B, N, 3*kmix)`` mixture
    parameters over the log2 half-life of each name."""

    def __init__(self, d: int = 64, heads: int = 4, blocks: int = 2,
                 window: int = L, kmix: int = NFE_KMIX):
        super().__init__()
        self.cell_embed = nn.Linear(2, d)
        self.pos_time = nn.Parameter(torch.zeros(1, 1, window, d))
        self.blocks = nn.ModuleList(
            [RowColumnBlock(d, heads) for _ in range(blocks)])
        self.head = nn.Linear(d, 3 * kmix)
        self.kmix = kmix

    def forward(self, x):                          # (B, N, L, 2)
        h = self.cell_embed(x) + self.pos_time
        for b in self.blocks:
            h = b(h)
        return self.head(h[:, :, -1])              # (B, N, 3*kmix), final bar


def build_panel_nfe(d: int = 64, heads: int = 4, blocks: int = 2,
                    window: int = L, kmix: int = NFE_KMIX) -> PanelNFE:
    """Factory for :class:`PanelNFE`, for symmetry with
    :func:`nfe.model.build_net`."""
    return PanelNFE(d, heads, blocks, window, kmix)


def panel_mdn_nll(out: torch.Tensor, y: torch.Tensor,
                  kmix: int = NFE_KMIX) -> torch.Tensor:
    """Mixture negative log-likelihood over the panel-shaped head output
    (including the ``0.5 log 2 pi`` constant)."""
    o = out.view(*out.shape[:-1], 3, kmix)
    lw = torch.log_softmax(o[..., 0, :], -1)
    mu = o[..., 1, :]
    ls = o[..., 2, :].clamp(-4, 2)
    ll = lw - ls - 0.5 * ((y.unsqueeze(-1) - mu) / ls.exp()) ** 2 \
        - 0.5 * float(np.log(2 * np.pi))
    return -torch.logsumexp(ll, -1)


def panel_mdn_pit(out: torch.Tensor, y: torch.Tensor,
                  kmix: int = NFE_KMIX) -> np.ndarray:
    """Probability-integral transform of ``y`` under the panel posterior
    (needs ``scipy``)."""
    from scipy.stats import norm
    o = out.view(*out.shape[:-1], 3, kmix)
    w = torch.softmax(o[..., 0, :], -1).numpy()
    mu = o[..., 1, :].numpy()
    sd = o[..., 2, :].clamp(-4, 2).exp().numpy()
    return (w * norm.cdf((y.numpy()[..., None] - mu) / sd)).sum(-1)
