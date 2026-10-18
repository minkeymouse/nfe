"""Fair test of the AFM edge: conditional (characteristic-driven) factors.

The breadth sweep showed AFM does NOT beat PCA+OU on markets with STATIC,
PCA-recoverable factors -- but that is not the paper's claim. Epstein et al.
(2025) argue LEARNED conditional factors help when loadings are time-varying
functions of firm characteristics that static PCA cannot track. This tests
exactly that: a market where the true loadings drift as a nonlinear function of
evolving characteristics, so attention (which maps current characteristics ->
loadings each day) should track them while PCA (a static trailing-window
covariance) blurs them. If AFM beats the PCA baselines here, the paper's edge
MECHANISM reproduces (even if its numbers cannot, for lack of data).

Not the package -- lives in the project, imports the `dfm_python` package
(PyPI: dfm-python).

    uv run python experiments/verify_afm_conditional.py
"""
from __future__ import annotations

import numpy as np
from dfm_python import AFM, AFMDataset

ANN = np.sqrt(252.0)
K = 3
HIST = 25
COST_TURN, COST_SHORT = 5e-4, 1e-4


def make_conditional_market(T, N, M, seed, phi=0.9, drift=0.15):
    """Loadings drift as a nonlinear function of evolving characteristics."""
    rng = np.random.default_rng(seed)
    Wtrue = rng.standard_normal((M, K))                 # characteristic -> loading map
    char = np.empty((T, N, M))
    char[0] = rng.standard_normal((N, M))
    for t in range(1, T):
        char[t] = 0.98 * char[t - 1] + drift * rng.standard_normal((N, M))  # persistent drift
    beta = np.tanh(char @ Wtrue)                         # (T, N, K) time-varying, nonlinear
    F = rng.standard_normal((T, K)) * 0.02
    innov = rng.standard_normal((T, N)) * 0.01
    eps = np.zeros((T, N))
    for t in range(1, T):
        eps[t] = phi * eps[t - 1] + innov[t]            # OU arbitrage signal
    R = np.empty((T, N))
    R[0] = eps[0]
    for t in range(1, T):
        R[t] = (beta[t - 1] * F[t][None, :]).sum(1) + eps[t]   # uses beta_{t-1} (causal)
    return R.astype(np.float64), char.astype(np.float64)


def oos(model, R, X, tr):
    ds = AFMDataset(returns=R[tr - HIST:], characteristics=X[tr - HIST:])
    br = model.predict(data=ds)["book_returns"].ravel()
    return float(br.mean() / (br.std() + 1e-12) * ANN)


def fit_eval(fm, trading, R, X, tr, seed, epochs):
    ds = AFMDataset(returns=R[:tr], characteristics=X[:tr])
    m = AFM(dataset=ds, n_factors=K, hist_len=HIST, embed_dim=16, n_kernels=16,
            factor_model=fm, trading=trading, lambda_var=0.05,
            turnover_cost=COST_TURN, short_cost=COST_SHORT, learning_rate=3e-3,
            max_epochs=epochs, patience=15, seed=seed)
    m.fit()
    return oos(m, R, X, tr)


def main():
    T, M = 1000, 6
    tr = int(T * 0.7)
    seeds = [0, 1]
    print(f"Conditional-factor market (time-varying nonlinear loadings, K_true={K}, "
          f"M={M}, T={T}, train {tr}/test {T-tr}, costs {COST_TURN}/{COST_SHORT})",
          flush=True)
    print(f"{'N':>5} {'AFM OOS':>10} {'PCA+LongConv':>14} {'PCA+OU':>10} "
          f"{'edge(AFM-best PCA)':>20}", flush=True)
    for N in (50, 100):
        afm, pcl, ou = [], [], []
        for sd in seeds:
            R, X = make_conditional_market(T, N, M, sd)
            afm.append(fit_eval("attention", "longconv", R, X, tr, sd, 80))
            pcl.append(fit_eval("pca", "longconv", R, X, tr, sd, 80))
            ou.append(fit_eval("pca", "ou", R, X, tr, 0, 1))
        a, p, o = np.mean(afm), np.mean(pcl), np.mean(ou)
        print(f"{N:>5} {a:>10.2f} {p:>14.2f} {o:>10.2f} {a - max(p, o):>+20.2f}",
              flush=True)
    print("\nPaper mechanism reproduces if AFM > best PCA baseline here (learned "
          "conditional factors track drifting characteristic-driven loadings that "
          "static PCA cannot).", flush=True)


if __name__ == "__main__":
    main()
