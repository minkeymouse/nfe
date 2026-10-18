"""The single-name filter and its training.

A GRU encoder over the standardized window and a mixture-density head
over the log2 half-life, trained by neural posterior estimation on draws
from a generative model. Trained once; applied to any window in a single
forward pass, without ever seeing a real return.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .prior import (DEFAULT_PHI_VOL, DEFAULT_S_LOG, HL_HI, HL_LO, Simulator,
                    featurize, switching_ou)

__all__ = ["NFE_KMIX", "NFE_NTRAIN", "NFE_STEPS", "HIDDEN", "LR", "BATCH",
           "build_net", "mdn_nll", "train_nfe"]

NFE_KMIX: int = 5             # mixture components
NFE_NTRAIN: int = 60_000      # training draws
NFE_STEPS: int = 4_000        # optimizer steps
HIDDEN: int = 64              # GRU width
LR: float = 3e-3              # Adam learning rate, cosine-annealed to zero
BATCH: int = 512              # draws per step


def build_net(hidden: int = HIDDEN, kmix: int = NFE_KMIX) -> nn.ModuleDict:
    """The bare filter: a GRU encoder of the (level, increment) window and
    a linear mixture-density head emitting {log-weight, mean, log-scale}
    for each of ``kmix`` components. One definition shared by training and
    loading."""
    return nn.ModuleDict({
        "gru": nn.GRU(2, hidden, batch_first=True),
        "mdn": nn.Linear(hidden, 3 * kmix),
    })


def mdn_nll(out: torch.Tensor, y: torch.Tensor, kmix: int = NFE_KMIX) -> torch.Tensor:
    """Negative log-likelihood of ``y`` under the mixture the head emits
    (up to the constant ``0.5 log 2 pi``)."""
    o = out.view(-1, 3, kmix)
    lw = torch.log_softmax(o[:, 0], -1)
    mu = o[:, 1]
    ls = o[:, 2].clamp(-4, 2)
    ll = lw - ls - 0.5 * ((y[:, None] - mu) / ls.exp()) ** 2
    return -torch.logsumexp(ll, -1)


def train_nfe(seed: int = 0, phi_vol: float = DEFAULT_PHI_VOL,
              s_log: float = DEFAULT_S_LOG, *, simulator: Simulator | None = None,
              ntrain: int = NFE_NTRAIN, steps: int = NFE_STEPS,
              hl_lo: float = HL_LO, hl_hi: float = HL_HI,
              hidden: int = HIDDEN, kmix: int = NFE_KMIX,
              lr: float = LR, batch: int = BATCH) -> nn.ModuleDict:
    """Amortize the filtering posterior: train the filter by neural
    posterior estimation on ``ntrain`` draws from a generative model.
    Returns the trained module.

    ``simulator`` is any callable ``(n, rng) -> (windows, targets)``; the
    default is :func:`nfe.prior.switching_ou` built from the keyword
    arguments. Both the draws and the initialization derive from ``seed``,
    so the same arguments on the same CPU build reproduce the same weights.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    if simulator is None:
        simulator = switching_ou(phi_vol=phi_vol, s_log=s_log,
                                 hl_lo=hl_lo, hl_hi=hl_hi)
    Wtr, ytr = simulator(ntrain, rng)
    X = torch.tensor(featurize(Wtr))
    Y = torch.tensor(ytr, dtype=torch.float32)
    net = build_net(hidden=hidden, kmix=kmix)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for _ in range(steps):
        i = torch.randint(0, len(X), (batch,))
        out = net["mdn"](net["gru"](X[i])[0][:, -1])
        loss = mdn_nll(out, Y[i], kmix=kmix).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    net.eval()
    return net
