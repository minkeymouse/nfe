"""Further AFM verification: does the paper's edge require BREADTH?

The KRX test (experiments/verify_afm_krx.py) found the paper's claim did not
reproduce on 13 stocks -- diagnosed as over-flexibility overfitting a tiny, thin
cross-section. This tests that diagnosis directly: on synthetic markets with a
planted residual-arbitrage signal (ground truth), sweep the cross-section size N
and measure AFM's out-of-sample after-cost Sharpe and its edge over the
parametric PCA+OU baseline. If the edge GROWS with N, the paper's method is
breadth-dependent and the KRX failure (N=13) is explained rather than a bug.

Not the package -- lives in the project, imports the `dfm_python` package
(PyPI: dfm-python).

    uv run python experiments/verify_afm_breadth.py
"""
from __future__ import annotations

import numpy as np
from dfm_python import AFM, AFMDataset

ANN = np.sqrt(252.0)
KTRUE = 5
HIST = 25
COST_TURN, COST_SHORT = 5e-4, 1e-4


def make_market(T, N, K, seed, phi=0.9):
    rng = np.random.default_rng(seed)
    beta = rng.standard_normal((N, K))
    M = K + 4
    char = np.empty((T, N, M))
    char[:, :, :K] = beta[None]                         # characteristics carry loadings
    char[:, :, K:] = rng.standard_normal((T, N, M - K)) * 0.5
    char += rng.standard_normal((T, N, M)) * 0.1
    F = rng.standard_normal((T, K)) * 0.02
    innov = rng.standard_normal((T, N)) * 0.01
    eps = np.zeros((T, N))
    for t in range(1, T):
        eps[t] = phi * eps[t - 1] + innov[t]            # OU residual arbitrage signal
    R = F @ beta.T + eps
    return R.astype(np.float64), char.astype(np.float64)


def oos(model, R, X, tr):
    ds = AFMDataset(returns=R[tr - HIST:], characteristics=X[tr - HIST:])
    br = model.predict(data=ds)["book_returns"].ravel()
    return float(br.mean() / (br.std() + 1e-12) * ANN)


def fit_eval(fm, trading, R, X, tr, seed, epochs):
    ds = AFMDataset(returns=R[:tr], characteristics=X[:tr])
    m = AFM(dataset=ds, n_factors=KTRUE, hist_len=HIST, embed_dim=16, n_kernels=16,
            factor_model=fm, trading=trading, lambda_var=0.05,
            turnover_cost=COST_TURN, short_cost=COST_SHORT, learning_rate=3e-3,
            max_epochs=epochs, patience=15, seed=seed)
    m.fit()
    return oos(m, R, X, tr)


def main():
    T = 900
    tr = int(T * 0.7)
    Ns = [13, 25, 50, 100]
    seeds = [0, 1]
    print(f"Synthetic breadth sweep (planted OU signal, K_true={KTRUE}, T={T}, "
          f"train {tr}/test {T-tr}, costs {COST_TURN}/{COST_SHORT})", flush=True)
    print(f"{'N':>5} {'AFM OOS Sharpe':>16} {'PCA+OU OOS':>12} {'edge (AFM-OU)':>14}",
          flush=True)
    for N in Ns:
        afm_s, ou_s = [], []
        for sd in seeds:
            R, X = make_market(T, N, KTRUE, sd)
            afm_s.append(fit_eval("attention", "longconv", R, X, tr, sd, 70))
            ou_s.append(fit_eval("pca", "ou", R, X, tr, sd, 1))
        a, o = np.mean(afm_s), np.mean(ou_s)
        print(f"{N:>5} {a:>16.2f} {o:>12.2f} {a - o:>+14.2f}", flush=True)
    print("\nDiagnosis confirmed if AFM Sharpe and edge grow with N: the paper's "
          "edge is breadth-dependent, so the KRX N=13 negative is expected, not a bug.",
          flush=True)


if __name__ == "__main__":
    main()
