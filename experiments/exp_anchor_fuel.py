"""Diagnostic: does the KRX residual panel still carry a common component?

    uv run python experiments/exp_anchor_fuel.py

[exploration] (2026-07-24). MEASUREMENT, not a verdict experiment. The
anchored-NFE idea (pin each name's level with the cross-section, per the
factor-sharing confirmation) has fuel on KRX only if the book's residual
construction left common structure behind: the spreads the book trades
are PCA-3 residuals, and a perfect residualization would leave the panel
cross-sectionally independent, exactly the control condition in which
pooling was measured to be worthless. This script measures what is left.

STATISTICS, per non-overlapping 60d window of residual increments:
  lam1   top-eigenvalue share of the cross-name correlation matrix
  rbar   mean signed pairwise correlation
NULL: each name's increments time-shuffled INDEPENDENTLY within the
window (destroys common timing, preserves marginals); 20 shuffles per
window. CONTEXTS: raw returns (factor present, positive control),
market-adjusted (one factor removed), PCA-3 residuals (the book's
stream). Era-2 windows only (the reported era). Fuel = PCA-3 lam1
above its shuffled null by > 2 SE across windows.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments"))

from exp_e3_book import L, load_panel, r_demean, r_pca

N_SHUF = 20


def win_stats(W):
    """W (L, N) increments -> (lam1 share, mean pairwise corr)."""
    sd = W.std(0)
    live = sd > 1e-12
    if live.sum() < 10:
        return np.nan, np.nan
    C = np.corrcoef(W[:, live].T)
    ev = np.linalg.eigvalsh(C)
    lam1 = float(ev[-1] / ev.sum())
    n = C.shape[0]
    rbar = float((C.sum() - n) / (n * (n - 1)))
    return lam1, rbar


def survey(E, t0, rng):
    outs, nulls = [], []
    for t in range(t0, len(E) - L, L):
        W = E[t:t + L]
        if not np.isfinite(W).all():
            continue
        lam1, rbar = win_stats(W)
        if np.isnan(lam1):
            continue
        nl = []
        for _ in range(N_SHUF):
            Ws = np.take_along_axis(
                W, rng.permuted(np.tile(np.arange(L)[:, None],
                                        (1, W.shape[1])), axis=0), axis=0)
            nl.append(win_stats(Ws)[0])
        outs.append((lam1, rbar))
        nulls.append(np.mean(nl))
    lam = np.array([o[0] for o in outs])
    rb = np.array([o[1] for o in outs])
    nul = np.array(nulls)
    d = lam - nul
    se = d.std(ddof=1) / np.sqrt(len(d))
    return lam.mean(), rb.mean(), nul.mean(), d.mean(), se, len(d)


def main():
    rng = np.random.default_rng(0)
    R, RF, dates = load_panel()
    mid = len(R) // 2
    print("[exploration] anchor fuel: common structure left in the "
          "residual panel (era-2 60d windows)\n", flush=True)
    print(f"  {'stream':16s} {'lam1':>6s} {'rbar':>7s} {'null':>6s}"
          f" {'lam1-null':>10s} {'windows':>8s}", flush=True)
    streams = [("raw returns", R, mid),
               ("market-adjusted", r_demean(R, RF), mid),
               ("PCA-3 residual", r_pca(R, RF), mid)]
    cache = ROOT / "results" / "run_260724" / "afm_stream_era2.npz"
    if cache.exists():   # 2026-07-24 v4 round: the NFE's own stream
        streams.append(("AFM residual", np.load(cache)["eps"], 252))
    for name, E, t0 in streams:
        lam, rb, nul, d, se, n = survey(E, t0, rng)
        print(f"  {name:16s} {lam:6.3f} {rb:+7.4f} {nul:6.3f}"
              f" {d:+7.3f} +- {se:.3f} {n:>5d}", flush=True)
    print("\n  fuel = (lam1 - null) > 2 SE on the traded stream.",
          flush=True)


if __name__ == "__main__":
    main()
