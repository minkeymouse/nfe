"""The decision layer: posterior draws of the reversion rate and the
uncertainty-penalized sizing rule.

No gradient from profit reaches the filter; the only interface between
the two is the posterior. For a window with standardized deviation ``z``
from its mean and posterior draws of the reversion rate ``kappa`` over a
hold of ``H`` bars, with ``a = exp(-kappa H)`` the expected fraction of
the deviation remaining,

    w = -z * E[1 - a] / ( E[1 - a^2] + z^2 Var(a) )

The ``z^2 Var(a)`` term is the parameter-uncertainty penalty: identically
zero for a point estimate, it is what a posterior adds over a plug-in.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import torch

from .model import NFE_KMIX
from .prior import H, featurize

__all__ = ["NFE_NSAMP", "nfe_kappa_samples", "sig_nfe"]

NFE_NSAMP: int = 64           # posterior draws per name


def nfe_kappa_samples(net, hist: np.ndarray, nsamp: int = NFE_NSAMP,
                      seed: int = 0, kmix: int = NFE_KMIX) -> np.ndarray:
    """``hist`` (N, L) cumulated windows -> (N, nsamp) posterior draws of
    the reversion rate kappa, drawn from the mixture with a generator seeded
    by ``seed`` so repeated calls agree."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        o = net["mdn"](net["gru"](torch.tensor(featurize(hist)))[0][:, -1])
        o = o.view(-1, 3, kmix)
        w = torch.softmax(o[:, 0], -1)
        mu = o[:, 1]
        sd = o[:, 2].clamp(-4, 2).exp()
        c = torch.multinomial(w, nsamp, replacement=True, generator=g)
        y = mu.gather(1, c) + \
            sd.gather(1, c) * torch.randn(mu.shape[0], nsamp, generator=g)
    return np.log(2) / np.clip(2.0 ** y.numpy(), 1e-3, 1e4)


def sig_nfe(hist: np.ndarray, net, H: int = H,
            prior: Callable[[np.ndarray], np.ndarray] | None = None) -> np.ndarray:
    """Posterior-sized positions for a window of residual returns ``hist``
    (N, L): the rule in the module docstring, with the deviation ``z``
    clipped to ``[-3, 3]`` and the expectations taken over posterior draws.

    ``prior`` optionally supplies a deployment prior on the log2 half-life
    as a callable ``f(y) -> log density`` over an ``(N, S)`` array of
    draws. The head is trained under a prior flat in log2 half-life, so it
    is likelihood-shaped and reweighting its draws by ``f`` is the
    posterior under that prior, with no retraining; the prior may differ
    per name or per deployment. ``None`` keeps the training prior. See
    :func:`nfe.prior.gaussian_prior` and :func:`nfe.prior.column_prior`.
    """
    X = np.cumsum(hist, axis=-1)
    m, sd = X.mean(-1), X.std(-1) + 1e-12
    z = np.clip((X[:, -1] - m) / sd, -3, 3)
    ks = nfe_kappa_samples(net, X)
    if prior is None:
        a, b = np.exp(-ks * H), np.exp(-2 * ks * H)
        num = (1.0 - a).mean(1)
        den = (1.0 - b).mean(1) + z ** 2 * a.var(1)
        return -z * num / (den + 1e-12)
    y = np.log2(np.log(2) / np.clip(ks, 1e-12, None))     # draws back to log2
    lw = np.asarray(prior(y), float)
    lw = lw - lw.max(1, keepdims=True)
    w = np.exp(lw)
    w /= w.sum(1, keepdims=True)
    a, b = np.exp(-ks * H), np.exp(-2 * ks * H)
    Ea, Eb = (w * a).sum(1), (w * b).sum(1)
    Va = (w * a * a).sum(1) - Ea ** 2
    return -z * (1.0 - Ea) / ((1.0 - Eb) + z ** 2 * Va + 1e-12)
