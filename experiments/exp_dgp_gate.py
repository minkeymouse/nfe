"""Which residual process actually matches real cross-sections? (G4 two-sample gate)

    uv run python experiments/exp_dgp_gate.py [binance|krx|both]

[exploration] (2026-07-21). Everything in SIM_LADDER.md is tier A: it holds
where the switching-OU DGP is true. REAL_LADDER.md showed the mechanism does
not transfer, and the G4 gate already said why -- a classifier separates
simulated OU windows from real residual windows at AUC 0.884 (KF) / 0.963
(crypto), far above the 0.75 pass bar. an earlier ladder script (since removed) localized the driver:
KURTOSIS (per-feature AUC 0.851). So the Gaussian switching OU is the wrong
process, and this script asks which process is less wrong.

CANDIDATES (all keep the switching mean-reversion skeleton; they differ in the
innovation law, which is where the gate says the mismatch lives):
  ou-gauss   switching OU, Gaussian innovations            <- the current DGP
  ou-t       Student-t(nu) innovations                     <- kurtosis
  ou-sv      stochastic volatility, log-vol AR(1)          <- vol clustering
  ou-t-sv    both
  ou-jump    Gaussian + compound Poisson jumps
  ou-sv-x    stochastic vol with a COMMON (cross-sectional) vol factor, so
             windows share a market-wide vol state as real residuals do
  ou-tsv-het t+SV with per-path (half-life, sigma, spell) drawn from wide priors
  ou-tsv-emp same, with half-life and scale resampled from the REAL empirical
             distributions -- the most aggressive calibration available
  rw-gauss / rw-tsv / rw-tsv-het   NO mean reversion at all (kappa == 0). These
             test the SKELETON rather than the innovation law: if a pure random
             walk sits closer to real residuals than any OU variant, the
             mean-reverting premise itself is what the gate is rejecting.

SPLIT PROTOCOL -- audited 2026-07-22, detail in DGP_GATE.md. Random splits of
stride-1 windows leak: neighbours share 59/60 points, and a same-DGP null that
must score 0.5 reached 0.738 on binance from that artefact alone. Windows are
therefore split BY TIME with an L-bar embargo, keeping the full stride-1
sample; both controls pass (same-DGP null 0.50, known-wrong ou-gauss 0.90), so
the cell is neither inflated nor deflated. Subsampling happens at the INDEX
level, before any window is materialised.

Free parameters are moment-matched to the real substrate, NOT fitted to the
gate -- otherwise the gate would score an optimizer, not a process. Targets:
within-window excess kurtosis of the increments, lag-1 autocorrelation of
|increment| (vol clustering), and lag-1 autocorrelation of the increment.

GATE (pre-registered, the project's standing G4). A classifier is trained to
tell simulated windows from real windows using scale-invariant window
statistics only.
  PASS  AUC <= 0.75 -- the process is close enough that tier-A mechanism
        results may be argued to transfer to that substrate.
  FAIL  AUC  > 0.75 -- simulator-dependent posterior claims stay
        diagnostic-only for that universe.
Per-feature AUCs are reported so a failure names its own cause, as before.
All windows are standardized, so the gate cannot win on scale.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from exp_cross_sectional import load_binance, load_krx, fixed_panel

L, H = 60, 20
NWIN = 20_000
GATE = 0.75
WHICH = next((a for a in sys.argv[1:] if a in ("binance", "krx", "both")), "both")


# ------------------------------------------------------------------ features
def stdz(W):
    return (W - W.mean(-1, keepdims=True)) / (W.std(-1, keepdims=True) + 1e-12)


def feats(W):
    """Scale-invariant window statistics (n, F) + their names."""
    Z = stdz(W)
    d = np.diff(Z, axis=-1)
    ds = d / (d.std(-1, keepdims=True) + 1e-12)
    ad = np.abs(ds)
    ac = lambda a, k: ((a[:, :-k] - a.mean(-1, keepdims=True)) *
                       (a[:, k:] - a.mean(-1, keepdims=True))).mean(-1) / \
                      (a.var(-1) + 1e-12)
    F = {
        "kurtosis(d)": (ds ** 4).mean(-1),
        "skew(d)": (ds ** 3).mean(-1),
        "maxabs(d)": ad.max(-1),
        "p99/p50 |d|": (np.percentile(ad, 99, -1) /
                        (np.percentile(ad, 50, -1) + 1e-12)),
        "volclust ac1|d|": ac(ad, 1),
        "volclust ac5|d|": ac(ad, 5),
        "ac1(d)": ac(ds, 1),
        "ac5(d)": ac(ds, 5),
        "zero crossings": (np.diff(np.sign(Z), axis=-1) != 0).sum(-1) * 1.0,
        "range": Z.max(-1) - Z.min(-1),
        "frac |d|>2sd": (ad > 2).mean(-1),
        "half-vol ratio": (d[:, L // 2:].std(-1) / (d[:, :L // 2].std(-1) + 1e-12)),
    }
    return np.column_stack([np.nan_to_num(v) for v in F.values()]), list(F)


# ------------------------------------------------------------------ DGPs
def simulate(kind, n, T, rng, cal):
    """Switching OU skeleton; `kind` changes the innovation law and, for the
    `-het` variants, the per-path heterogeneity of (half-life, sigma, spell)."""
    rw = kind.startswith("rw")
    het = kind.endswith("-het") or kind.endswith("-emp")
    if kind.endswith("-emp"):                 # match the REAL kappa/sd spread
        hlf = np.exp(rng.choice(cal["log_hl_emp"], n))
        hls = hlf * np.exp(rng.uniform(1.0, 3.0, n))
        scale = np.exp(rng.choice(cal["log_sd_emp"], n))
        spell = rng.uniform(100, 500, n)
    elif het:
        lu = lambda a, b: np.exp(rng.uniform(np.log(a), np.log(b), n))
        hlf, hls = lu(2, 30), lu(40, 400)
        scale = lu(0.002, 0.05)
        spell = rng.uniform(80, 600, n)
    else:
        hlf, hls, spell = 10.0, 120.0, 250.0
        scale = np.full(n, 1.0)
    kf, ks = np.log(2) / hlf, np.log(2) / hls
    if np.isscalar(spell):
        spell = np.full(n, spell)
    st = rng.integers(0, 2, n)
    x = np.zeros(n)
    X = np.zeros((T, n))
    nu = cal["nu"]
    phiv, sv = cal["phi_vol"], cal["sig_vol"]
    lam, jsz = cal["jump_rate"], cal["jump_size"]
    lv = np.zeros(n)                      # log vol state
    lvc = 0.0                             # common vol factor
    for t in range(T):
        st = np.where(rng.random(n) < 1.0 / spell, 1 - st, st)
        k = np.zeros(n) if rw else np.where(st == 0, kf, ks)
        if kind in ("ou-sv", "ou-t-sv", "ou-tsv-het", "ou-tsv-emp",
                    "rw-sv", "rw-tsv", "rw-tsv-het",
                    "ou-sv-lognorm", "rw-sv-lognorm"):
            lv = phiv * lv + sv * np.sqrt(1 - phiv ** 2) * rng.standard_normal(n)
        elif kind == "ou-sv-x":
            lvc = phiv * lvc + sv * np.sqrt(1 - phiv ** 2) * rng.standard_normal()
            lv = phiv * lv + 0.5 * sv * np.sqrt(1 - phiv ** 2) * \
                rng.standard_normal(n) + 0.0
            lv = lv * 0.5 + lvc
        if kind.endswith("-lognorm"):
            sl = cal["s_log"]
            m_ln = np.exp(sl ** 2 / 2)
            sd_ln = np.sqrt((np.exp(sl ** 2) - 1) * np.exp(sl ** 2))
            e = cal["skew_sign"] * (np.exp(sl * rng.standard_normal(n)) - m_ln) / sd_ln
        elif kind in ("ou-t", "ou-t-sv", "ou-tsv-het", "ou-tsv-emp",
                      "rw-tsv", "rw-tsv-het"):
            e = rng.standard_t(nu, n) / np.sqrt(nu / (nu - 2))
        else:
            e = rng.standard_normal(n)
        if kind == "ou-jump":
            j = (rng.random(n) < lam) * rng.standard_normal(n) * jsz
            e = e + j
        x = x + k * (0.0 - x) + 0.01 * scale * np.exp(lv) * e
        X[t] = x
    return X


def calibrate(Wr):
    """Moment-match the free parameters to the real windows (never to the gate)."""
    Z = stdz(Wr)
    d = np.diff(Z, axis=-1)
    ds = d / (d.std(-1, keepdims=True) + 1e-12)
    ad = np.abs(ds)
    kurt = float(np.median((ds ** 4).mean(-1)))
    ac1 = float(np.median(((ad[:, :-1] - ad.mean(-1, keepdims=True)) *
                           (ad[:, 1:] - ad.mean(-1, keepdims=True))).mean(-1) /
                          (ad.var(-1) + 1e-12)))
    nu = max(2.5, min(30.0, 4.0 + 6.0 / max(kurt - 3.0, 0.05)))   # t kurtosis 3+6/(nu-4)
    phi = float(np.clip(ac1 * 4.0, 0.0, 0.97))
    dX = np.diff(Wr, axis=-1); Xl = Wr[:, :-1]
    mx = Xl.mean(-1, keepdims=True)
    b = ((Xl - mx) * (dX - dX.mean(-1, keepdims=True))).sum(-1) / \
        (((Xl - mx) ** 2).sum(-1) + 1e-12)
    hl = np.log(2) / np.clip(-b, 1e-3, 1.0)
    sub = lambda a: np.log(a[np.isfinite(a) & (a > 0)])[:5000]
    skew = float(np.median((ds ** 3).mean(-1)))          # increment asymmetry
    gs = np.linspace(0.02, 1.2, 60)                     # match a centred lognormal
    sk = (np.exp(gs ** 2) + 2) * np.sqrt(np.exp(gs ** 2) - 1)
    s_log = float(gs[np.argmin(np.abs(sk - abs(skew)))])
    return {"nu": nu, "phi_vol": phi, "sig_vol": 0.5,
            "target_skew": skew, "s_log": s_log,
            "skew_sign": float(np.sign(skew) or 1.0),
            "jump_rate": 0.02, "jump_size": 4.0,
            "target_kurt": kurt, "target_ac1absd": ac1,
            "log_hl_emp": sub(hl), "log_sd_emp": sub(Wr.std(-1) / 0.01)}


# ------------------------------------------------------------------ gate
OFF = np.arange(-L, 0)


def residual_panel(panel):
    """(T, N) cross-sectionally demeaned cumulative log-return residuals."""
    panel = panel.dropna(axis=1, how="any")
    r = np.diff(np.log(panel.to_numpy(float)), axis=0)
    return np.cumsum(r - r.mean(1, keepdims=True), axis=0)


def windows(X, rng, cap=2 * NWIN, stride=1):
    """Sample window (end-time, asset) indices FIRST, then materialise only
    those. Never build the full (n_t, N, L) tensor -- it is tens of GB."""
    T, N = X.shape
    ts = np.arange(L, T, stride)
    gt, gn = np.repeat(ts, N), np.tile(np.arange(N), len(ts))
    if cap < len(gt):
        s = rng.choice(len(gt), cap, replace=False)
        gt, gn = gt[s], gn[s]
    W = X[gt[:, None] + OFF[None, :], gn[:, None]]
    ok = np.isfinite(W).all(-1)
    return W[ok], gt[ok]


def gate(Wr, tr, Ws, ts):
    """Two-sample AUC under a TIME split with an L-bar embargo (no window in
    train shares a point with any window in test)."""
    n = min(len(Wr), len(Ws))
    Fr, names = feats(Wr[:n])
    Fs, _ = feats(Ws[:n])
    Xf = np.vstack([Fr, Fs])
    y = np.r_[np.zeros(n), np.ones(n)]
    t = np.r_[tr[:n], ts[:n]]
    cut = np.median(t)
    a, b = t < cut - L, t > cut + L
    clf = HistGradientBoostingClassifier(max_iter=200, random_state=0)
    clf.fit(Xf[a], y[a])
    auc = roc_auc_score(y[b], clf.predict_proba(Xf[b])[:, 1])
    per = {}
    for j, nm in enumerate(names):
        s = roc_auc_score(y[b], Xf[b, j])
        per[nm] = float(max(s, 1 - s))
    return float(auc), per


NSEED = 5


def verdict(mean, se):
    """PASS only if the whole 2-SE band clears the bar; FAIL if none of it does."""
    if mean + 2 * se <= GATE:
        return "PASS"
    return "MARGINAL" if mean - 2 * se <= GATE else "FAIL"


def sweep(Xr, tr_src, cal, kinds, T, N, label):
    rows = {}
    print(f"  {'DGP':<12}{'G4 AUC':>16}{'verdict':>10}   worst-separating features")
    for kind in kinds:
        Wr, tr = tr_src()
        aucs, per = [], {}
        for sd in range(NSEED):
            Xs = simulate(kind, N, T, np.random.default_rng(100 + sd), cal)
            Ws, ts = windows(Xs, np.random.default_rng(22 + sd))
            a, per = gate(Wr, tr, Ws, ts)
            aucs.append(a)
        a = np.array(aucs)
        m, se = float(a.mean()), float(a.std(ddof=1) / np.sqrt(NSEED))
        top = sorted(per.items(), key=lambda kv: -kv[1])[:3]
        rows[kind] = {"auc": m, "se": se, "seeds": aucs, "per_feature": per}
        print(f"  {label(kind):<12}{m:>9.3f} +-{se:5.3f}{verdict(m, se):>10}   "
              + ", ".join(f"{k} {v:.2f}" for k, v in top), flush=True)
    return rows


def run(name, panel):
    Xr = residual_panel(panel)
    T, N = Xr.shape
    Wcal, _ = windows(Xr, np.random.default_rng(7), cap=NWIN, stride=L)
    cal = calibrate(Wcal)
    print(f"\n=== {name.upper()}: panel T={T} N={N} | calibrated "
          f"nu={cal['nu']:.1f}, phi_vol={cal['phi_vol']:.2f} "
          f"(targets: kurt {cal['target_kurt']:.1f}, ac1|d| "
          f"{cal['target_ac1absd']:.3f}, skew {cal['target_skew']:+.3f} "
          f"-> s_log {cal['s_log']:.2f}) ===", flush=True)
    out = {"calibration": {k: v for k, v in cal.items()
                           if not k.endswith("_emp")}, "T": T, "N": N}

    # NULL FIRST: two independent draws of the SAME DGP must score ~0.5. If they
    # do not, the split protocol is separating, not the process.
    def null_src(kind):
        return lambda: windows(simulate(kind, N, T, np.random.default_rng(11), cal),
                               np.random.default_rng(21))
    print("  -- null (same DGP, independent seeds; must be ~0.50)")
    out["null"] = {k: sweep(Xr, null_src(k), cal, [k], T, N,
                            lambda c: f"NULL {c}")[k] for k in ("ou-gauss", "ou-sv")}

    print("  -- candidates vs real")
    kinds = ["ou-gauss", "ou-t", "ou-sv", "ou-t-sv", "ou-jump", "ou-sv-x",
             "ou-lognorm", "ou-sv-lognorm",
             "ou-tsv-het", "ou-tsv-emp",
             "rw-gauss", "rw-sv", "rw-sv-lognorm", "rw-tsv", "rw-tsv-het"]
    out["dgp"] = sweep(Xr, lambda: windows(Xr, np.random.default_rng(21)),
                       cal, kinds, T, N, lambda c: c)
    best = min(out["dgp"], key=lambda k: out["dgp"][k]["auc"])
    b = out["dgp"][best]
    print(f"  -> closest: {best} ({b['auc']:.3f} +-{b['se']:.3f}, "
          f"{verdict(b['auc'], b['se'])}; baseline ou-gauss "
          f"{out['dgp']['ou-gauss']['auc']:.3f})", flush=True)
    out["best"] = best
    return out


def main():
    OUT = Path("results/run_260722"); OUT.mkdir(parents=True, exist_ok=True)
    print(f"[exploration] G4 two-sample gate over candidate residual processes "
          f"(pass <= {GATE})", flush=True)
    res = {}
    if WHICH in ("binance", "both"):
        res["binance"] = run("binance", fixed_panel(load_binance(), 45))
    if WHICH in ("krx", "both"):
        res["krx"] = run("krx", fixed_panel(load_krx(), 150))
    (OUT / "dgp_gate.json").write_text(json.dumps(res, indent=1))
    print(f"\n[exploration] artifact: {OUT}/dgp_gate.json")


if __name__ == "__main__":
    main()
