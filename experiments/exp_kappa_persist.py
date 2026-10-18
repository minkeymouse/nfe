"""Does the reversion speed have a NAME-level persistent component, and do
observable characteristics explain it?

    uv run python experiments/exp_kappa_persist.py

PRE-REGISTRATION (frozen 2026-07-27, before the first run).

WHY.  The training prior is one (phi, gamma) per era shared by all 120
names, so every name is told its half-life is log-uniform on 3-400 days.
run_260727's screen showed the filter CAN exploit a per-name prior when
one exists: conditioning on a type that genuinely shifts the prior moved
AUC 0.652 -> 0.891, a gain the size of the whole level cost (0.240 in
CEILING.md).  That was an upper bound on a simulator.  The open question
is whether REAL names differ in reversion speed in a way real
characteristics predict.  This is label-free, so it runs on real data
with no simulator and no misspecification gate.

DATA.  The cached AFM residual streams (results/run_260724/
afm_stream_era{1,2}.npz, 2036 days x 120 names each) and the
point-in-time characteristic panel (data/processed/fundamentals_daily.npz,
7 features).  Nothing is refit; no returns are predicted.

METHOD.  Per name, cumulate the residual, cut non-overlapping L=60 blocks,
and fit the conventional OLS AR(1) plug-in on the demeaned block to get
kappa-hat; work in log2 half-life, the filter's own target.

    P1 PERSISTENCE   corr(y_k, y_{k+1}) pooled over names and blocks.
    P2 NAME EFFECT   share of the variance of y explained by name identity
                     (a one-way ICC).
    P3 PREDICTABLE   fit name-level mean y on characteristics in era 1 and
                     predict era 2 name-level mean y, out of sample.

CONFOUND, AND THE CONTROL FOR IT.  kappa-hat from a short window is biased
and the bias tracks the block's realized volatility, so a "name effect"
could be nothing but the well-known persistence of volatility.  Every
statistic is therefore reported twice: raw, and after residualizing y on
log realized volatility within block.  P3 is reported for volatility
alone, for the seven characteristics alone, and for both, so the question
"do the characteristics add anything beyond volatility" has a number.

NULLS.  P1/P2: name labels shuffled within block, which destroys name
identity and preserves every marginal.  P3: the characteristic-to-name
mapping shuffled.  Both resampled for an SE.

READ-OUT (this is a measurement, not a gate).  The conditioned prior has
real footing only if the name effect survives the volatility control AND
the characteristics predict it out of sample beyond volatility.  If it
does not, the lever is dead on this substrate and is reported as such.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

L = 60
HL_LO, HL_HI = 3.0, 400.0
NBOOT = 2000


def blocks(eps):
    """(T,N) residual returns -> per-(block,name) log2 half-life and log vol."""
    S = np.cumsum(np.nan_to_num(eps), axis=0)
    T, N = S.shape
    ys, vs = [], []
    for t0 in range(0, T - L + 1, L):
        W = S[t0:t0 + L].T                                  # (N, L)
        ok = np.isfinite(W).all(-1) & (W.std(-1) > 1e-10)
        Z = W - W.mean(-1, keepdims=True)
        dz = np.diff(Z, axis=-1)
        x = Z[:, :-1]
        beta = (x * dz).sum(-1) / ((x * x).sum(-1) + 1e-18)  # dS = beta*S
        kap = np.clip(-beta, np.log(2) / HL_HI, np.log(2) / HL_LO)
        y = np.log2(np.log(2) / kap)
        v = np.log(np.abs(np.diff(W, axis=-1)).mean(-1) + 1e-12)
        y[~ok] = np.nan; v[~ok] = np.nan
        ys.append(y); vs.append(v)
    return np.array(ys), np.array(vs)                       # (B, N)


def resid_on(y, v):
    """Residualize y on v within each block (removes the vol confound)."""
    out = np.full_like(y, np.nan)
    for b in range(len(y)):
        m = np.isfinite(y[b]) & np.isfinite(v[b])
        if m.sum() < 10:
            continue
        X = np.stack([np.ones(m.sum()), v[b][m]], 1)
        c, *_ = np.linalg.lstsq(X, y[b][m], rcond=None)
        out[b, m] = y[b][m] - X @ c
    return out


def persistence(y):
    a, b = y[:-1].ravel(), y[1:].ravel()
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 20:
        return float("nan"), float("nan")
    r = float(np.corrcoef(a[m], b[m])[0, 1])
    return r, float((1 - r ** 2) / np.sqrt(m.sum() - 2))


def icc(y):
    """Share of variance explained by name identity."""
    m = np.isfinite(y)
    if m.sum() < 20:
        return float("nan")
    gm = np.nanmean(y)
    nm = np.nanmean(y, 0)                                   # (N,)
    between = np.nansum(m.sum(0) * (nm - gm) ** 2)
    total = np.nansum((y - gm) ** 2)
    return float(between / (total + 1e-18))


def shuffle_names(y, rng):
    out = y.copy()
    for b in range(len(out)):
        out[b] = out[b][rng.permutation(out.shape[1])]
    return out


def oos_predict(Xtr, ytr, Xte, yte):
    """Ridge fit on era 1 name means -> era 2 name means. Returns held-out
    R^2 and Spearman rank correlation."""
    from scipy.stats import spearmanr
    m1 = np.isfinite(ytr) & np.isfinite(Xtr).all(1)
    m2 = np.isfinite(yte) & np.isfinite(Xte).all(1)
    if m1.sum() < 20 or m2.sum() < 20:
        return float("nan"), float("nan")
    mu, sd = Xtr[m1].mean(0), Xtr[m1].std(0) + 1e-12
    A = np.c_[np.ones(m1.sum()), (Xtr[m1] - mu) / sd]
    lam = 1.0 * np.eye(A.shape[1]); lam[0, 0] = 0.0
    c = np.linalg.solve(A.T @ A + lam, A.T @ ytr[m1])
    B = np.c_[np.ones(m2.sum()), (Xte[m2] - mu) / sd]
    p = B @ c
    ss = ((yte[m2] - p) ** 2).sum()
    st = ((yte[m2] - ytr[m1].mean()) ** 2).sum()
    return float(1 - ss / st), float(spearmanr(p, yte[m2]).statistic)


def main():
    rng = np.random.default_rng(0)
    F = np.load(ROOT / "data/processed/fundamentals_daily.npz", allow_pickle=True)
    chars_all = F["F"]                                      # (4072, 120, 7)
    feats = list(F["feats"])
    print(f"characteristics: {feats}", flush=True)

    res = {}
    Y, V = {}, {}
    for era in (1, 2):
        eps = np.load(ROOT / f"results/run_260724/afm_stream_era{era}.npz")["eps"]
        y, v = blocks(eps)
        Y[era], V[era] = y, v
        yr = resid_on(y, v)
        r, se = persistence(y); rr, rse = persistence(yr)
        i0, i1 = icc(y), icc(yr)
        nulls = [icc(shuffle_names(y, rng)) for _ in range(30)]
        nullsr = [icc(shuffle_names(yr, rng)) for _ in range(30)]
        res[f"era{era}"] = {
            "blocks": int(y.shape[0]), "names": int(y.shape[1]),
            "P1_persistence_raw": [r, se],
            "P1_persistence_vol_resid": [rr, rse],
            "P2_icc_raw": i0,
            "P2_icc_null": [float(np.mean(nulls)), float(np.std(nulls, ddof=1))],
            "P2_icc_vol_resid": i1,
            "P2_icc_vol_resid_null": [float(np.mean(nullsr)),
                                      float(np.std(nullsr, ddof=1))],
        }
        print(f"era{era}: P1 raw {r:+.3f}+-{se:.3f}  vol-resid {rr:+.3f}+-{rse:.3f}"
              f" | P2 ICC {i0:.3f} (null {np.mean(nulls):.3f})"
              f"  vol-resid {i1:.3f} (null {np.mean(nullsr):.3f})", flush=True)

    # ---- P3: do characteristics predict the name-level speed, era1 -> era2?
    T = chars_all.shape[0]
    half = T // 2
    C1 = np.nanmean(chars_all[:half], 0)                    # (120, 7)
    C2 = np.nanmean(chars_all[half:], 0)
    lv1 = np.nanmean(V[1], 0)[:, None]                      # (120, 1) log vol
    lv2 = np.nanmean(V[2], 0)[:, None]
    n1, n2 = np.nanmean(Y[1], 0), np.nanmean(Y[2], 0)       # name-level speed
    nr1 = np.nanmean(resid_on(Y[1], V[1]), 0)
    nr2 = np.nanmean(resid_on(Y[2], V[2]), 0)

    sets = {"vol_only": (lv1, lv2),
            "chars_only": (C1, C2),
            "vol_plus_chars": (np.c_[lv1, C1], np.c_[lv2, C2])}
    p3 = {}
    for name, (A, B) in sets.items():
        r2, rho = oos_predict(A, n1, B, n2)
        r2r, rhor = oos_predict(A, nr1, B, nr2)
        nulls = []
        for _ in range(200):
            p = rng.permutation(len(n1))
            nulls.append(oos_predict(A[p], n1, B, n2)[1])
        p3[name] = {"R2_oos": r2, "spearman": rho,
                    "R2_oos_vol_resid": r2r, "spearman_vol_resid": rhor,
                    "null_spearman": [float(np.nanmean(nulls)),
                                      float(np.nanstd(nulls, ddof=1))]}
        print(f"P3 {name:15s} rho {rho:+.3f} (R2 {r2:+.3f}) | "
              f"vol-resid rho {rhor:+.3f} | null {np.nanmean(nulls):+.3f}"
              f"+-{np.nanstd(nulls, ddof=1):.3f}", flush=True)
    res["P3_era1_to_era2"] = p3

    out = ROOT / "results" / "run_260727"
    out.mkdir(parents=True, exist_ok=True)
    (out / "kappa_persist.json").write_text(json.dumps(res, indent=2))
    print(f"\n-> {out / 'kappa_persist.json'}")


if __name__ == "__main__":
    main()
