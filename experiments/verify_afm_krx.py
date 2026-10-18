"""Verify the Attention Factor Model paper claim on REAL KRX daily data.

Tests Epstein et al. (2025)'s central claim -- that jointly-learned attention
factors + a LongConv residual filter beat the PCA + Avellaneda-Lee (OU) recipe
after costs, out-of-sample -- on the KOSPI large-cap daily panel behind the
paper (data/raw/krx/, the files exp_cross_sectional.load_krx reads). This
lives in the PROJECT and
imports the `dfm_python` package (PyPI: dfm-python); no verification code goes
in the package.

Honesty / scope (report with the numbers):
  - Cross-section is TINY (~13 stocks) vs the paper's 500 -> high estimation
    noise; a single period, a few seeds -> a demonstration, not a powered study.
  - Characteristics are RETURN-BASED only (multi-horizon momentum, realized vol),
    a strict subset of the paper's 79 (no Compustat fundamentals) -> attention
    has far thinner input than in the paper.
  - Costs are the paper's 5bps turnover + 1bp shorting. KRX microstructure and
    period differ from the paper's US sample.
  - Strictly causal: characteristics use only past returns; the model lags them
    internally; OOS = train on the first 70%, evaluate on the held-out tail.

    uv run python experiments/verify_afm_krx.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from dfm_python import AFM, AFMDataset

ROOT = Path(__file__).resolve().parent.parent

RAW = ROOT / "data" / "raw" / "krx"
ANN = np.sqrt(252.0)
K = 3
HIST = 25
COST_TURN, COST_SHORT = 5e-4, 1e-4
WARMUP = 252
MIN_ROWS = 4000                         # keep symbols with long history (drops newest listings)


def load_returns():
    """Common-date daily log-return panel for the long-history KOSPI names."""
    frames = {}
    for f in sorted(RAW.glob("*.parquet")):
        if "master" in f.name:
            continue
        d = pd.read_parquet(f)[["date", "close"]].dropna()
        if len(d) < MIN_ROWS:
            continue
        d = d.drop_duplicates("date").set_index("date")["close"].astype(float)
        frames[f.stem] = d
    panel = pd.DataFrame(frames).dropna()               # intersection of dates
    prices = panel.to_numpy(float)
    R = np.diff(np.log(prices), axis=0)                 # (T, N) log returns
    return R, list(panel.columns), panel.index.to_numpy()[1:]


def _xrank(a):
    """Cross-sectional rank in [-0.5, 0.5] per row (day)."""
    order = a.argsort(axis=1).argsort(axis=1).astype(float)
    return order / max(a.shape[1] - 1, 1) - 0.5


def build_characteristics(R):
    """Causal return-based characteristics (T, N, M), rank-normalized per day."""
    T, N = R.shape
    df = pd.DataFrame(R)
    feats = []
    for w in (5, 21, 63, 126, 252):                     # trailing cumulative returns
        feats.append(df.rolling(w).sum().to_numpy())
    for w in (21, 63):                                  # realized volatility
        feats.append(df.rolling(w).std().to_numpy())
    feats.append(np.vstack([np.zeros((1, N)), R[:-1]])) # prior-day return (short reversal)
    X = np.stack([_xrank(np.nan_to_num(f)) for f in feats], axis=2)  # (T, N, M)
    return X


def oos_sharpe(model, R, X, tr):
    ds = AFMDataset(returns=R[tr - HIST:], characteristics=X[tr - HIST:])
    br = model.predict(data=ds)["book_returns"].ravel()
    return float(br.mean() / (br.std() + 1e-12) * ANN)


def train_eval(factor_model, trading, R, X, tr, seed):
    ds = AFMDataset(returns=R[:tr], characteristics=X[:tr])
    epochs = 1 if trading == "ou" else 70
    m = AFM(dataset=ds, n_factors=K, hist_len=HIST, embed_dim=16, n_kernels=16,
            factor_model=factor_model, trading=trading, lambda_var=0.05,
            turnover_cost=COST_TURN, short_cost=COST_SHORT, learning_rate=3e-3,
            max_epochs=epochs, patience=15, seed=seed)
    m.fit()
    return oos_sharpe(m, R, X, tr)


def main():
    R, names, dates = load_returns()
    X = build_characteristics(R)
    R, X = R[WARMUP:], X[WARMUP:]                        # drop characteristic warmup
    T, N = R.shape
    tr = int(T * 0.7)
    print(f"KRX panel: N={N} stocks {names}")
    print(f"  {T} common trading days {str(dates[WARMUP])[:8]}..{str(dates[-1])[:8]}, "
          f"train {tr} / test {T - tr}; costs {COST_TURN}/{COST_SHORT}; K={K}, s={HIST}")
    print(f"  (return-based characteristics only, M={X.shape[2]}; tiny cross-section)\n")

    seeds = [0, 1]
    print(f"{'model':>26} {'OOS after-cost Sharpe (ann.)':>30}", flush=True)
    # learned models: average over seeds (training is stochastic)
    for label, fm, tr_pol in (("AFM (attention+LongConv)", "attention", "longconv"),
                              ("PCA + LongConv", "pca", "longconv")):
        s = [train_eval(fm, tr_pol, R, X, tr, sd) for sd in seeds]
        print(f"{label:>26} {np.mean(s):>16.3f}  (seeds {min(s):+.2f}..{max(s):+.2f})",
              flush=True)
    # PCA + OU (Avellaneda-Lee): no learnable params -> deterministic
    ou = train_eval("pca", "ou", R, X, tr, 0)
    print(f"{'PCA + OU (Avellaneda-Lee)':>26} {ou:>16.3f}", flush=True)

    print("\nPaper claim = learned (attention+LongConv) >= PCA+OU after costs, OOS.")
    print("Read the ORDERING, not the level: this is a tiny KRX cross-section with")
    print("return-only characteristics, not the paper's 500-stock 79-characteristic US panel.")


if __name__ == "__main__":
    main()
