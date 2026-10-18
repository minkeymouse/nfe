"""The generative model that is the filter's label factory, and the
calibrations that fit it to real windows.

A generative model is any callable with the signature

    simulator(n, rng) -> (windows (n, L), targets (n,))

whose draws carry the latent target by construction. Real data never
does, which is what makes a posterior trained on such draws checkable
for calibration. :func:`switching_ou` is the default: a two-state
switching Ornstein--Uhlenbeck spread with stochastic volatility, a
drifting level and centred lognormal innovations. :func:`train_nfe`
accepts any other simulator with the same signature.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

__all__ = ["L", "H", "HL_LO", "HL_HI", "Simulator", "simulate", "switching_ou",
           "featurize", "gaussian_prior", "column_prior", "calib_halflife",
           "calib_from"]

L: int = 60                   # causal window, in bars
H: int = 20                   # non-overlapping hold horizon the sizing integrates over
HL_LO: float = 3.0            # default log-uniform half-life prior, in bars
HL_HI: float = 400.0

# Simulator internals. These scale the process, not its identifying
# parameters; the filter sees standardized windows, so only their ratios matter.
BURN_IN: int = 200            # bars simulated and discarded before recording
SCALE: float = 0.01           # innovation and level-drift scale
VOL_SD: float = 0.5           # stationary sd of the log-volatility AR(1)

DEFAULT_PHI_VOL: float = 0.19  # volatility persistence
DEFAULT_S_LOG: float = 0.14    # lognormal innovation asymmetry
DEFAULT_TAU: float = 0.05      # level drift, relative to SCALE

Simulator = Callable[[int, np.random.Generator], tuple[np.ndarray, np.ndarray]]


def simulate(n: int, rng: np.random.Generator, phi_vol: float = DEFAULT_PHI_VOL,
             s_log: float = DEFAULT_S_LOG, tau: float = DEFAULT_TAU,
             hl_lo: float = HL_LO, hl_hi: float = HL_HI) -> tuple[np.ndarray, np.ndarray]:
    """Draw ``n`` windows from the switching-OU process.

    Returns ``(windows (n, L), log2 half-life (n,))``. The half-life is
    drawn log-uniformly on ``[hl_lo, hl_hi]``; training keeps the wide
    default on purpose, because a head trained under a prior flat in log2
    half-life is likelihood-shaped and a deployment prior can then be
    applied by reweighting its draws (:func:`column_prior`,
    :func:`nfe.filter.sig_nfe`) instead of by retraining.
    """
    hl = np.exp(rng.uniform(np.log(hl_lo), np.log(hl_hi), n))
    k = np.log(2) / hl
    m_ln = np.exp(s_log ** 2 / 2)
    sd_ln = np.sqrt((np.exp(s_log ** 2) - 1) * np.exp(s_log ** 2))
    x = np.zeros(n); lv = np.zeros(n); m = np.zeros(n)
    W = np.empty((n, L))
    for t in range(BURN_IN + L):
        lv = phi_vol * lv + VOL_SD * np.sqrt(1 - phi_vol ** 2) * rng.standard_normal(n)
        e = (np.exp(s_log * rng.standard_normal(n)) - m_ln) / sd_ln
        m = m + tau * rng.standard_normal(n) * SCALE
        x = x + k * (m - x) + SCALE * np.exp(lv) * e
        if t >= BURN_IN:
            W[:, t - BURN_IN] = x
    return W, np.log2(hl)


def switching_ou(phi_vol: float = DEFAULT_PHI_VOL, s_log: float = DEFAULT_S_LOG,
                 tau: float = DEFAULT_TAU, hl_lo: float = HL_LO,
                 hl_hi: float = HL_HI) -> Simulator:
    """Package :func:`simulate` with fixed parameters as a simulator
    ``(n, rng) -> (W, y)``, the form :func:`nfe.model.train_nfe` takes."""
    def simulator(n: int, rng: np.random.Generator):
        return simulate(n, rng, phi_vol=phi_vol, s_log=s_log, tau=tau,
                        hl_lo=hl_lo, hl_hi=hl_hi)
    return simulator


def featurize(W: np.ndarray) -> np.ndarray:
    """``(..., L)`` windows -> standardized ``(..., L, 2)`` features: the
    level and its increments. The filter sees shape, not units, at train
    and deploy alike."""
    Z = (W - W.mean(-1, keepdims=True)) / (W.std(-1, keepdims=True) + 1e-12)
    return np.stack([Z, np.diff(Z, axis=-1, prepend=Z[:, :1])],
                    -1).astype(np.float32)


# ---------------------------------------------------------------------------
# Plug-in half-lives on non-overlapping blocks (used by the calibrations).

def _ols_block(W: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """OLS AR(1) reversion rate per row of ``W`` (rows, L), clipped to the
    default half-life range. Returns ``(good, kappa)``; ``good`` marks rows
    that are finite and non-degenerate."""
    good = np.isfinite(W).all(-1) & (W.std(-1) > 1e-10)
    Z = W - W.mean(-1, keepdims=True)
    dz = np.diff(Z, axis=-1)
    x = Z[:, :-1]
    beta = (x * dz).sum(-1) / ((x * x).sum(-1) + 1e-18)
    k = np.clip(-beta, np.log(2) / HL_HI, np.log(2) / HL_LO)
    return good, k


def _block_halflives(E: np.ndarray) -> np.ndarray:
    """Per-(block, column) log2 half-life from the OLS AR(1) plug-in on
    non-overlapping L-bar blocks of the cumulated panel ``E`` (T, N).
    Returns (K, N), NaN where degenerate.

    Gaps are zero-filled and masked per column, not per row: on a real
    panel one missing name would otherwise discard the whole cross-section
    for that bar.
    """
    X = np.cumsum(np.nan_to_num(E), axis=0)
    out = []
    for t0 in range(0, len(X) - L + 1, L):
        good, k = _ols_block(X[t0:t0 + L].T)
        out.append(np.where(good, np.log2(np.log(2) / k), np.nan))
    return np.array(out) if out else np.zeros((0, E.shape[1]))


def _trim_leading_gaps(E: np.ndarray) -> np.ndarray:
    """Drop leading rows until the first bar at which every column is finite
    (a rolling-residual burn-in)."""
    ok = np.isfinite(E).all(1)
    return E[int(np.argmax(ok)) if ok.any() else len(E):]


# ---------------------------------------------------------------------------
# Deployment priors on the log2 half-life.

def gaussian_prior(centre, sd) -> Callable[[np.ndarray], np.ndarray]:
    """A Gaussian prior on the log2 half-life, as the callable the sizing
    rule expects: ``f(y) -> log density`` for ``y`` an ``(N, S)`` array of
    draws for N names. ``centre`` and ``sd`` may be scalars or per-name
    arrays. Any callable with that signature is an equally valid prior."""
    centre = np.asarray(centre, float).reshape(-1, 1) if np.ndim(centre) \
        else float(centre)
    sd = np.asarray(sd, float).reshape(-1, 1) if np.ndim(sd) else float(sd)

    def log_density(y):
        return -0.5 * ((y - centre) / sd) ** 2

    return log_density


def column_prior(panel: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """Fit one prior per column of a ``(T, N)`` residual panel, from that
    slice alone, by empirical Bayes.

    Each column's block half-lives are averaged and pulled toward the panel
    mean by the one-way shrinkage factor ``between / (between + within / k)``,
    so a column with few or noisy blocks is pulled further. Fit it on the
    training slice only, never on the slice being evaluated.
    """
    Y = _block_halflives(panel)
    n = panel.shape[1]
    if len(Y) < 2:
        return gaussian_prior(np.full(n, np.log2(20.0)), 1.0)
    k = np.isfinite(Y).sum(0).astype(float)
    m = np.nanmean(Y, 0)
    gm = float(np.nanmean(m))
    within = float(np.nanmean(np.nanvar(Y, 0, ddof=1)))
    between = max(float(np.nanvar(m, ddof=1))
                  - within / max(float(np.nanmean(k)), 1.0), 1e-4)
    lam = between / (between + within / np.maximum(k, 1.0))
    centre = gm + lam * (np.nan_to_num(m, nan=gm) - gm)
    return gaussian_prior(centre, np.sqrt(between * (1.0 - lam) + 1e-6))


# ---------------------------------------------------------------------------
# Calibrating the training prior to a substrate (training slice only).

def calib_halflife(E: np.ndarray, lo_q: float = 0.10, hi_q: float = 0.90,
                   pad: float = 1.5) -> dict:
    """Fit the half-life prior's range to a ``(T, N)`` residual panel.
    Returns ``{'hl_lo', 'hl_hi'}``: the ``lo_q``/``hi_q`` quantiles of the
    plug-in block half-lives, widened by ``pad`` and clipped to the default
    range. Use it to bake a fitted range into training; the default keeps
    the wide range and conditions at deployment instead."""
    E = _trim_leading_gaps(E)
    X = np.cumsum(np.nan_to_num(E), axis=0)
    T = len(X)
    if T <= L:
        return dict(hl_lo=HL_LO, hl_hi=HL_HI)
    ys = []
    for t0 in range(0, T - L + 1, L):
        good, k = _ols_block(X[t0:t0 + L].T)
        ys.append(np.where(good, np.log(2) / k, np.nan))
    y = np.concatenate(ys)
    y = y[np.isfinite(y)]
    if len(y) < 8:
        return dict(hl_lo=HL_LO, hl_hi=HL_HI)
    lo, hi = np.quantile(y, [lo_q, hi_q])
    return dict(hl_lo=float(np.clip(lo / pad, HL_LO, HL_HI)),
                hl_hi=float(np.clip(hi * pad, HL_LO, HL_HI)))


def calib_from(E: np.ndarray) -> dict:
    """Moment-match the innovation law of :func:`simulate` to a ``(T, N)``
    residual panel. Returns ``{'phi_vol', 's_log'}``.

    Statistics are taken per L-bar window and pooled by the median: the
    lag-1 autocorrelation of absolute standardized increments sets
    ``phi_vol``, their skewness sets ``s_log``. Measuring on the whole
    series instead of per window drives ``phi_vol`` to its cap, and the
    leading burn-in must be trimmed or its zero-filled windows dominate a
    short slice; both guards are from observed failures. Slices shorter
    than one window return the defaults.
    """
    E = _trim_leading_gaps(E)
    X = np.cumsum(np.nan_to_num(E), axis=0)
    T = len(X)
    if T <= L:
        return dict(phi_vol=DEFAULT_PHI_VOL, s_log=DEFAULT_S_LOG)
    W = np.stack([X[t - L:t].T for t in range(L, T, L)], 0).reshape(-1, L)
    W = W[np.isfinite(W).all(-1)]
    W = W[W.std(-1) > 1e-10]
    if len(W) < 8:
        return dict(phi_vol=DEFAULT_PHI_VOL, s_log=DEFAULT_S_LOG)
    Z = (W - W.mean(-1, keepdims=True)) / (W.std(-1, keepdims=True) + 1e-12)
    ds = np.diff(Z, axis=-1)
    ds = ds / (ds.std(-1, keepdims=True) + 1e-12)
    ad = np.abs(ds)
    m = ad.mean(-1, keepdims=True)
    ac1 = float(np.median(((ad[:, :-1] - m) * (ad[:, 1:] - m)).mean(-1)
                          / (ad.var(-1) + 1e-12)))
    skew = float(np.median((ds ** 3).mean(-1)))
    gs = np.linspace(0.02, 1.2, 60)
    sk = (np.exp(gs ** 2) + 2) * np.sqrt(np.exp(gs ** 2) - 1)
    return dict(phi_vol=float(np.clip(ac1 * 4.0, 0.0, 0.97)),
                s_log=float(gs[np.argmin(np.abs(sk - abs(skew)))]))
