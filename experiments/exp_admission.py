"""Substrate admission: does this cross-section's residual mean-revert at all?

    uv run python experiments/exp_admission.py

[exploration] (2026-07-22). PRE-REGISTRATION — frozen before the first run on
these substrates.

WHY. A book comparison on a cross-section whose residuals do not mean-revert
returns near-zeros for every arm, and that table is indistinguishable from one
where the methods are simply bad. So the substrate is tested BEFORE the book,
with the simplest possible OU trade and no estimation at all: if the residual is
genuinely OU, the causal window mean alone should work, and if that reads zero
there is nothing a better estimator can find.

PROBE (no foresight, no fitting). At each non-overlapping H-day hold, form the
displacement from the trailing L-day mean and measure the cross-sectional
correlation between it and the realized forward move, averaged over holds:
    stat = mean_t  corr_i( m_i(t) - X_i(t) ,  X_i(t+H) - X_i(t) )

CONTROLS, identical pipeline, both required before the statistic is read:
    NULL      random walk + stochastic volatility, per-name vol matched,
              NO mean reversion. Resampled NDRAW times so the excess has an SE.
    POSITIVE  true switching OU, half-life 10d vs 120d, 250d spells — a strong
              regime. Tells us what the instrument reads when there IS something.

RESIDUAL CONSTRUCTIONS. demean (cross-sectional) and PCA-3 (rolling 252d,
causal). The Fama-French arm was dropped with the earlier industry-portfolio
substrate (design decision 2026-07-22), so the ladder is two constructions,
not three.

SUBSTRATES. KRX daily (120 names, complete 2010-2026) and Binance daily (45
names, 2021-2026). Each is also split into halves, because the earlier
substrate's result was era-dependent and a substrate that is admitted in only one era must
not be reported as admitted.

GATE. Admitted if (real - null) exceeds 2 SE of the null draw spread combined
with the hold-to-hold SE of the real statistic. Report the ratio to the positive
control alongside, since clearing a null says nothing about magnitude.
NOTE ON MULTIPLICITY: 2 substrates x 2 constructions x 3 windows (full + 2 eras)
= 12 tests. Any single 2-SE result is uncorrected and must be reported as such.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments"))

from exp_e3_book import load_panel, r_demean, r_pca, L, H, C_TURN   # noqa: E402

NDRAW = 20
PHI_VOL, SIG_VOL = 0.19, 0.5


def probe(X):
    """-> (per-hold correlations, per-hold book P&L)."""
    ok = np.isfinite(X).all(1)
    X = X[int(np.argmax(ok)):] if ok.any() else X
    T, N = X.shape
    cs, pnl, prev = [], [], np.zeros(N)
    for t0 in np.arange(L, T - H, H):
        W = X[t0 - L:t0]
        m, sd = W.mean(0), W.std(0) + 1e-12
        dx = X[t0 + H] - X[t0]
        d = m - X[t0 - 1]
        if d.std() < 1e-15 or dx.std() < 1e-15:
            continue
        cs.append(np.corrcoef(d, dx)[0, 1])
        w = -np.clip((X[t0 - 1] - m) / sd, -3, 3)
        w = w - w.mean(); w = w / (np.abs(w).sum() + 1e-12)
        pnl.append(w @ dx - C_TURN * np.abs(w - prev).sum()); prev = w
    return np.array(cs), np.array(pnl)


def gen(T, N, rng, regime, vol):
    """rw-sv null (regime=False) or true switching OU (regime=True)."""
    pf, ps = 1 - np.log(2) / 10.0, 1 - np.log(2) / 120.0
    st = rng.integers(0, 2, N); x = np.zeros(N); lv = np.zeros(N)
    X = np.empty((T, N))
    for t in range(T):
        st = np.where(rng.random(N) < 1 / 250.0, 1 - st, st)
        p = np.where(st == 0, pf, ps) if regime else 1.0
        lv = PHI_VOL * lv + SIG_VOL * np.sqrt(1 - PHI_VOL ** 2) * rng.standard_normal(N)
        x = p * x + np.exp(lv) * rng.standard_normal(N)
        X[t] = x
    r = np.diff(X, axis=0, prepend=X[:1])
    return np.cumsum(r * (vol / (r.std(0) + 1e-18)), axis=0)


def main():
    print(f"[exploration] substrate admission — L={L}, H={H}, "
          f"null resampled {NDRAW}x\n")
    print(f"  {'substrate':<10}{'window':<20}{'residual':<9}{'REAL':>17}"
          f"{'NULL':>17}{'excess':>17}{'ctrl':>7}  verdict")
    for sub in ("krx", "binance"):
        R, _RF, dates = load_panel(sub)
        halves = [("full", slice(None)),
                  (f"era1 {str(dates[0])[:7]}", slice(0, len(R) // 2)),
                  (f"era2 {str(dates[len(R)//2])[:7]}", slice(len(R) // 2, None))]
        for wlab, sl in halves:
            Re = R[sl]
            for cname, fn in (("demean", r_demean), ("PCA-3", r_pca)):
                E = fn(Re, None)
                X = np.cumsum(np.nan_to_num(E), axis=0)
                cs, _ = probe(X)
                if len(cs) < 5:
                    print(f"  {sub:<10}{wlab:<20}{cname:<9}  (too few holds)")
                    continue
                se_r = cs.std(ddof=1) / np.sqrt(len(cs))
                vol = np.nanstd(E, axis=0)
                Tn, Nn = X.shape
                nulls = [probe(gen(Tn, Nn, np.random.default_rng(300 + d),
                                   False, vol))[0].mean() for d in range(NDRAW)]
                nulls = np.array(nulls)
                ctrl = probe(gen(Tn, Nn, np.random.default_rng(2), True, vol))[0].mean()
                ex = cs.mean() - nulls.mean()
                sd = np.sqrt(se_r ** 2 + nulls.std(ddof=1) ** 2)
                v = "ADMIT" if ex > 2 * sd else ("marginal" if ex > sd else "reject")
                print(f"  {sub:<10}{wlab:<20}{cname:<9}"
                      f"{cs.mean():>10.4f} +-{se_r:.4f}"
                      f"{nulls.mean():>10.4f} +-{nulls.std(ddof=1):.4f}"
                      f"{ex:>10.4f} +-{sd:.4f}{ctrl:>7.3f}  {v}", flush=True)
        print()


if __name__ == "__main__":
    main()
