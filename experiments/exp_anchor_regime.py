"""The anchored regime: X = h(C) + u, with converge/diverge states on u.

    uv run python experiments/exp_anchor_regime.py [--smoke]

[exploration] (2026-07-23). PRE-REGISTRATION — frozen before the first full run.

REDEFINITION. The regime is no longer fast-vs-slow reversion
(a speed contrast the measured ceiling caps at 0.65) but the SIGN of the
deviation dynamics: does u return to the anchor (C) or trend away (D)?
Measured grounds: anchor KNOWN gives 0.80-0.90 sign detection, anchor
estimated from the window 0.56-0.69 — the same ~0.24 level cost as the speed
regime. The anchor is the universal lever, so it gets its own model
h(C;psi), gradient-boosted on characteristics, fitted causally then frozen:
GBM carries the level, OU the deviation dynamics.

STAGES AND GATES
  S1  fit h on the train slice: X_t ~ GBM(C_{t-1}).  Controls:
      - feature-shuffle null (same GBM, shuffled characteristics): its R^2 is
        the capacity floor; h must clear it.
      - report R^2_anchor = 1 - Var(X - h)/Var(X) on HELD-OUT data.
  S2  ceiling repair: on SYNTHETIC panels with a planted anchor of the same
      R^2, measure converge/diverge detection on u = X - h vs on X - mean.
      Prediction from the decomposition: detection rises toward the
      anchor-known bound in proportion to R^2_anchor.
  S3  THE EXISTENCE TEST (the gate that matters). On real data, classify each
      window C vs D from u, then check OUT OF SAMPLE whether the label
      predicts the NEXT window's fade P&L: fade should earn on C-labelled
      windows and lose on D-labelled ones. Controls:
      - null: same pipeline on a no-regime surrogate (rw-sv, vol-matched);
        the C-D fade spread must vanish.
      - positive control: planted C/D switching panel; the spread must appear.
      GATE: real C-D spread > 2 SE AND positive control > 4x the null.
      If S3 fails, the redefinition is recorded as not supported on this
      substrate and the paper's claims do not change.

Substrate: KRX 120 names (the E3 panel), PCA-3 residual returns cumulated,
characteristics = the causal window statistics of exp_predictive (momentum,
vol, skew, kurt, liquidity proxies from |r|). L=60, H=20, era split as E3.

AMENDMENT 2026-07-23: +6 point-in-time DART fundamentals join the anchor
characteristics. All six are level/quality features, direction-blind by
construction — the axis the first S1 run found missing. The rw NULL keeps
the REAL fundamentals attached, so a fundamental-channel artifact must show
up there too; the planted positive control is fully synthetic, returns-only.

CONFIRMATION PASS --era1 (registered 2026-07-23 BEFORE the run). era2's S3
found the OPPOSITE of the registration at 3.7 SE: fade earns MORE on
diverge-labelled windows (overshoot). That inversion was measured twice on
the same era2 windows, so it needs an independent sample. --era1 fits h on
[300, mid/2 - H] and evaluates S3 on [mid/2 + L, mid - H], no overlap with
era2. GATE (frozen): the direction confirms iff the era1 REAL spread
< -2 SE (same sign as era2) with the null flat. PASS -> the overshoot
reading gains an independent sample; FAIL -> the finding is era2-scoped and
the paper may not state it as a property of the substrate.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments"))

from sklearn.ensemble import HistGradientBoostingRegressor          # noqa: E402
from exp_e3_book import load_panel, r_pca, L, H                     # noqa: E402

SMOKE = "--smoke" in sys.argv
ERA1 = "--era1" in sys.argv          # registered confirmation pass (docstring)
KC = np.log(2) / 20.0
NBOOT = 2000


# ------------------------------------------------------------------ features
def load_fund():
    """(T, N, 6) point-in-time fundamentals (prep_fundamentals.py), aligned to
    the panel, or None. All six (roe/de/margin/rev_yoy/bp/ep) are LEVEL and
    QUALITY features -- direction-blind by construction, i.e. exactly the axis
    S1 found missing when the anchor had only returns-derived inputs. Coverage
    starts 2016-03 (DART floor); earlier days fall back to the daily median."""
    p = ROOT / "data" / "processed" / "fundamentals_daily.npz"
    if not p.exists():
        return None
    return np.load(p, allow_pickle=True)["F"]


def characteristics(X, t, Fnd=None):
    """Causal per-name features at time t (anchor inputs). (N, F).

    SLOW features only. The first smoke run included 5d/20d trailing means and
    the "anchor" became a lagged copy of X itself (held-out R^2 0.991, and the
    planted-panel S3 control INVERTED because u collapsed to a few days of
    innovations). Home must be built from things that move slower than the
    deviation we want to define on top of it.
    """
    base = X[max(0, t - 250):t].mean(0)                       # the slow level
    f = [base]
    # DIRECTION-BLIND features only. The second smoke run kept momentum and the
    # planted-panel control inverted at 3.7 SE: during a trend the GBM anchor
    # EXTRAPOLATES the trend past the current level, flipping the sign of the
    # deviation, so fade "won" on diverge windows. Principle: the anchor may
    # know how volatile, fat-tailed or liquid a name is, but not which way it
    # has been going -- directional information belongs to the regime, not to
    # home. (Skew is directional and is excluded with momentum.)
    r = np.diff(X[max(0, t - 60):t], axis=0)
    v = r.std(0) + 1e-12
    f.append(v)
    rs = (r - r.mean(0)) / v
    f.append((rs ** 4).mean(0))
    f.append(np.abs(r).mean(0))                               # liquidity proxy
    if Fnd is not None:                       # valuation/quality anchor axes
        for fi in range(Fnd.shape[2]):
            c = Fnd[t - 1, :, fi].copy()
            fin = np.isfinite(c)
            c[~fin] = np.median(c[fin]) if fin.any() else 0.0
            f.append(c)
    return np.column_stack(f)


def fit_anchor(X, t0s_fit, shuffle=False, seed=0, Fnd=None):
    """GBM anchor: X_t ~ h(C_{t-1}). Returns the fitted model."""
    rng = np.random.default_rng(seed)
    Cs, ys = [], []
    for t in t0s_fit:
        C = characteristics(X, t, Fnd)
        if shuffle:
            C = C[rng.permutation(len(C))]
        Cs.append(C); ys.append(X[t])
    Cs, ys = np.vstack(Cs), np.concatenate(ys)
    m = HistGradientBoostingRegressor(max_iter=60 if SMOKE else 300,
                                      random_state=seed)
    m.fit(Cs, ys)
    return m


def anchor_of(model, X, t, Fnd=None):
    return model.predict(characteristics(X, t, Fnd))


# ------------------------------------------------------------------ synthetic
def gen_cd(T, N, rng, c_drift=0.2, anchor_sd=0.0):
    """Converge/diverge switching panel; optional smooth latent anchor."""
    st = rng.integers(0, 2, N)
    dirn = np.sign(rng.standard_normal(N))
    x = rng.standard_normal(N) * 0.01 / np.sqrt(2 * KC)
    a = np.zeros(N)
    X = np.empty((T, N)); Z = np.empty((T, N), int); A = np.empty((T, N))
    for t in range(T):
        flip = rng.random(N) < 1 / 250.0
        st = np.where(flip, 1 - st, st)
        dirn = np.where(flip & (st == 1), np.sign(rng.standard_normal(N)), dirn)
        a = a + anchor_sd * rng.standard_normal(N)            # slow anchor walk
        drift = np.where(st == 0, -KC * x, c_drift * 0.01 * dirn)
        x = x + drift + 0.01 * rng.standard_normal(N)
        X[t] = a + x; Z[t] = st; A[t] = a
    return X, Z, A


def detect_cd(U):
    """Window score for DIVERGE: drift LR with the anchor already removed."""
    d = np.diff(U, axis=-1)
    rc = d + KC * U[:, :-1]
    llc = -(rc ** 2).sum(-1)
    c = 0.2 * 0.01
    return np.maximum(-((d - c) ** 2).sum(-1), -((d + c) ** 2).sum(-1)) - llc


# ------------------------------------------------------------------ stages
def s1_s2(X, mid, Fnd=None):
    t_fit = np.arange(300, mid - H, H)
    t_oos = np.arange(mid + L, len(X) - H, H)
    h = fit_anchor(X, t_fit, Fnd=Fnd)
    hs = fit_anchor(X, t_fit, shuffle=True, Fnd=Fnd)
    def sse(pred, ts):
        num = den = 0.0
        for t in ts:
            e = X[t] - pred(t)
            b = X[t] - X[max(0, t - 250):t].mean(0)           # slow-level baseline
            num += (e ** 2).sum(); den += (b ** 2).sum()
        return 1.0 - num / den
    r2h = sse(lambda t: anchor_of(h, X, t, Fnd), t_oos)
    r2s = sse(lambda t: anchor_of(hs, X, t, Fnd), t_oos)
    print(f"  S1 anchor skill vs slow-level baseline (held-out): h(C) {r2h:+.3f} "
          f"| feature-shuffle null {r2s:+.3f} | "
          f"{'CLEARS' if r2h > max(r2s, 0) + 0.05 else 'FAILS'}", flush=True)

    # S2: synthetic ceiling repair at the SAME anchor R^2
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(1)
    Tn, Nn = 2000, 120
    Xs, Zs, As = gen_cd(Tn, Nn, rng, anchor_sd=0.002)
    ts = np.arange(300, Tn - H, 7)
    Wm, Wu, y = [], [], []
    for t in ts:
        w = Xs[t - L:t].T
        Wm.append(w - w.mean(-1, keepdims=True))              # window-mean anchor
        Wu.append(w - As[t - 1][:, None])                     # TRUE anchor
        y.append(Zs[t - 1])
    Wm, Wu, y = np.vstack(Wm), np.vstack(Wu), np.concatenate(y)
    a_mean = roc_auc_score(y, detect_cd(Wm))
    a_true = roc_auc_score(y, detect_cd(Wu))
    print(f"  S2 synthetic C/D detection: window-mean anchor {a_mean:.3f} -> "
          f"true anchor {a_true:.3f} (the recoverable gap)", flush=True)
    return h


def s3(X, h, mid, label, Fnd=None, t_end=None):
    """Existence: does the C/D label predict NEXT-window fade P&L out of sample?"""
    t0s = np.arange(mid + L, (len(X) if t_end is None else t_end) - H, H)
    scores, pnl, wgt = [], [], []
    for t in t0s:
        a = anchor_of(h, X, t, Fnd)
        u = (X[t - L:t] - a[None, :]).T                       # anchored deviation
        s = detect_cd(u)                                      # high = diverge
        z = u[:, -1] / (u.std(-1) + 1e-12)
        w = -np.clip(z, -3, 3)
        w = w - w.mean(); w = w / (np.abs(w).sum() + 1e-12)
        dx = X[t + H] - X[t]
        scores.append(s); pnl.append(w * dx * len(w))         # per-name fade pnl
        wgt.append(w)
    S, P = np.vstack(scores), np.vstack(pnl)
    # CONFOUNDER CONTROL: D-labelled names carry larger |u| hence larger |w|,
    # so if fade earns on average their pnl is mechanically larger. Also test
    # the per-unit-weight (directional) pnl, which removes the sizing channel.
    W = np.vstack(wgt)
    Pn = P / (np.abs(W) * P.shape[1] + 1e-12) * np.sign(np.abs(W))
    thr = np.median(S)
    conv, div = P[S <= thr], P[S > thr]
    d = conv.mean() - div.mean()
    boot = np.random.default_rng(0)
    ds = []
    for _ in range(NBOOT):
        i = boot.integers(0, len(P), len(P))
        Si, Pi = S[i], P[i]
        t2 = np.median(Si)
        ds.append(Pi[Si <= t2].mean() - Pi[Si > t2].mean())
    se = np.std(ds, ddof=1)
    cu, du_ = Pn[S <= thr], Pn[S > thr]
    print(f"  S3 {label}: fade pnl C {conv.mean()*1e4:+.1f}bp vs D "
          f"{div.mean()*1e4:+.1f}bp | spread {d*1e4:+.1f} +- {se*1e4:.1f}bp "
          f"[{'PASS' if abs(d) > 2*se else 'fail'}] | per-unit-weight C "
          f"{np.nanmean(cu)*1e4:+.1f} vs D {np.nanmean(du_)*1e4:+.1f}bp", flush=True)
    return d, se


def main():
    tag = "smoke" if SMOKE else ("era1 CONFIRMATION" if ERA1 else "exploration")
    print(f"[{tag}] anchored converge/diverge regime\n")
    R, _, dates = load_panel()
    E = r_pca(R, None)
    X = np.cumsum(np.nan_to_num(E), axis=0)
    mid = len(X) // 2
    Fnd = load_fund()
    print(f"  fundamentals: {'attached (+6 anchor features)' if Fnd is not None else 'not on disk (returns-only anchor)'}\n",
          flush=True)

    if ERA1:                 # fit on era1's first half, evaluate on its second
        fit_hi, ev_lo, ev_hi, lab = mid // 2, mid // 2, mid, "REAL (era1 oos)"
    else:
        fit_hi, ev_lo, ev_hi, lab = mid, mid, len(X), "REAL (era2 oos)"

    if ERA1:
        h = fit_anchor(X, np.arange(300, fit_hi - H, H), Fnd=Fnd)
    else:
        h = s1_s2(X, mid, Fnd=Fnd)
        print()
    s3(X, h, ev_lo, lab, Fnd=Fnd, t_end=ev_hi)

    # null: no-regime surrogate through the identical pipeline. The REAL
    # fundamentals stay attached: if the anchor manufactures a spurious C-D
    # spread through the fundamental channel, it must appear here too.
    rng = np.random.default_rng(9)
    r0 = np.diff(X, axis=0, prepend=X[:1])
    Xn = np.cumsum(rng.standard_normal(X.shape) * r0.std(0), axis=0)
    hn = fit_anchor(Xn, np.arange(300, fit_hi - H, H), Fnd=Fnd)
    s3(Xn, hn, ev_lo, "NULL rw (no regime)", Fnd=Fnd, t_end=ev_hi)

    # positive control: planted C/D switching (fully synthetic names, so no
    # fundamentals exist for it; it validates the returns-side pipeline)
    Xs, Zs, _ = gen_cd(len(X), X.shape[1], np.random.default_rng(3),
                       anchor_sd=0.002)
    hp = fit_anchor(Xs, np.arange(300, fit_hi - H, H))
    s3(Xs, hp, ev_lo, "POSITIVE (planted C/D)", t_end=ev_hi)


if __name__ == "__main__":
    main()
