"""Verification pass: is the promise-gap reduction conditional or uniform?

    uv run python experiments/verify_promise_gap_sim.py

(2026-10-13, camera-ready check: a shrinkage-matched control for the
promise gap.) The real-data version is the two control arms added to
exp_predictive.py on the same day; this is the same control on truth-known
draws of the admitted process, where the mechanism can be read without the
real panel. Positions are normalised to unit gross per hold, so scaling a
forecast changes the promise and not one trade; every point-ou variant below
trades the identical book.

Setup (mirrors exp_predictive.py line for line where it matters): windows
and standardised H-day forward moves from the admitted process with
regime switching (its `_sim_windows_dx`); holds of N=120 independent names;
the first 30% of holds are the train slice; costs 5bp turnover + 1bp short;
the promise is w . E[dX] - cost, the gap its mean excess over realised P&L.

Arms
    gauss             mu = 0 (promise = -cost)
    point-ou          plug-in: z60 (1 - exp(-kappa_hat H)); positions = forecast
    point-ou-matched  point forecast x one constant so its mean |promise| on
                      the train slice equals nfe's (uniform caution)
    point-ou-recal    point forecast x its OLS slope on realised train outcomes
    nfe               posterior predictive mean, positions mu/var (as real)
    oracle-kappa      the point forecast with the TRUE kappa (mean-correct
                      reference; its gap is the floor a right mean leaves)

Registered reading (before the run). Under the true model the nfe mean is
close to the Bayes predictive mean, so its gap sits near oracle-kappa's and
well below point-ou's. point-ou-recal matches the mean by construction on
the train slice, so its test gap is also small. The decisive comparison is
point-ou-matched against nfe: nfe's gap smaller => the posterior's shrinkage
is conditional; equal within SE => the reduction is uniform caution. Three
draw seeds; one trained filter (seed 0, package defaults).

Measured 2026-10-13 (cloud, CPU; results/run_261013/PROMISE_GAP_SIM.md),
gap in bp per hold, bootstrap SE about 4-5, three draw seeds:
    point-ou          +388 / +394 / +385
    point-ou-matched  +167 / +170 / +163    (matched scale 0.52 on every seed)
    nfe               +165 / +165 / +162
    point-ou-recal      -0 /  +11 /  -15    (slope 0.13-0.17)
    oracle-kappa      +213 / +215 / +219
    gauss              -68 /  -68 /  -74
Reading: nfe and point-ou-matched agree within SE on every seed, so on
truth-known draws the posterior's promise-gap reduction over the point
plug-in is reproduced by one uniform scale factor; it is average caution,
not conditional caution. Every forecast that treats the window mean as the
level over-promises, the true-kappa oracle included (+213), so the residual
gap is the unknown level, the same lever the Bayes-bound analysis names.
The posterior's sizing does lift the book (Sharpe 3.75-3.94 vs 3.26-3.88
for the point's positions), a separate effect from the gap. The real-data
verdict is the two control arms in exp_predictive.py.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results" / "run_261013"
H, L = 20, 60
C_TURN, C_SHORT = 5e-4, 1e-4
N_NAMES, N_HOLDS, TRAIN_FRAC, NBOOT = 120, 300, 0.30, 2000


def sim_windows_dx(n, rng, phi_vol=0.19, s_log=0.14, tau=0.05):
    """exp_predictive._sim_windows_dx, with the true kappa path's final
    value returned as well (for the oracle arm)."""
    hl = np.exp(rng.uniform(np.log(3), np.log(400), n))
    k = np.log(2) / hl
    st = (rng.random(n) < 0.5)
    m_ln = np.exp(s_log ** 2 / 2)
    sd_ln = np.sqrt((np.exp(s_log ** 2) - 1) * np.exp(s_log ** 2))
    x = np.zeros(n); lv = np.zeros(n); m = np.zeros(n)
    W = np.empty((n, L))
    T = 200 + L + H
    for t in range(T):
        flip = rng.random(n) < 1 / 250.0
        st = np.where(flip, ~st, st)
        ks = np.where(st, k, np.log(2) / 400.0)
        lv = phi_vol * lv + 0.5 * np.sqrt(1 - phi_vol ** 2) * rng.standard_normal(n)
        e = (np.exp(s_log * rng.standard_normal(n)) - m_ln) / sd_ln
        m = m + tau * 0.01 * rng.standard_normal(n)
        x = x + ks * (m - x) + 0.01 * np.exp(lv) * e
        if 200 <= t < 260:
            W[:, t - 200] = x
        if t == 259:
            x0 = x.copy(); k_true = ks.copy()
    vol = np.abs(np.diff(W, axis=-1)).std(-1) + 1e-12
    y = (x - x0) / (vol * np.sqrt(H))
    return W, np.clip(y, -20, 20), vol * np.sqrt(H), k_true


def features(W):
    """z60 and kappa_hat exactly as exp_predictive.build forms them."""
    x = W[:, -1]
    z60 = -(x - W.mean(-1)) / (W.std(-1) + 1e-12)
    d, xl = np.diff(W, axis=-1), W[:, :-1]
    xcm = xl - xl.mean(-1, keepdims=True)
    b = (xcm * (d - d.mean(-1, keepdims=True))).sum(-1) / ((xcm ** 2).sum(-1) + 1e-18)
    return z60, np.clip(-b, -1, 1)


def nfe_mean_var(net, W):
    """Predictive mean and variance of the nfe arm (closed form over
    posterior draws, as exp_predictive.nfe_quantiles; moments taken
    analytically instead of from the quantile grid)."""
    from nfe import nfe_kappa_samples
    ks = nfe_kappa_samples(net, W, seed=0)
    m = W.mean(-1); sd = W.std(-1) + 1e-12
    z = np.clip((W[:, -1] - m) / sd, -3, 3)
    mu_d = (-z)[:, None] * (1.0 - np.exp(-ks * H))
    var_d = (1.0 - np.exp(-2 * ks * H)) / (2 * ks * H)
    mu = mu_d.mean(1)
    var = var_d.mean(1) + mu_d.var(1)
    return mu, np.maximum(var, 1e-9)


def book(sig, scale, y, tt, mu_pred):
    """exp_predictive.book: per hold, demeaned unit-gross positions, costs
    on turnover and shorts; returns (pnl, promise) per hold."""
    pnl, prom, prev = [], [], None
    for t in np.unique(tt):
        k = tt == t
        w = sig[k] - sig[k].mean()
        n = np.abs(w).sum()
        if n < 1e-12:
            pnl.append(0.0); prom.append(0.0); continue
        w = w / n
        if prev is None:
            prev = np.zeros_like(w)
        cost = C_TURN * np.abs(w - prev).sum() + C_SHORT * np.maximum(-w, 0).sum()
        pnl.append(w @ (y[k] * scale[k]) - cost)
        prom.append(w @ (mu_pred[k] * scale[k]) - cost)
        prev = w
    return np.array(pnl), np.array(prom)


def main():
    from nfe import train_nfe
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    net = train_nfe(seed=0, phi_vol=0.19, s_log=0.14)
    print(f"filter trained ({time.time()-t0:.0f}s)", flush=True)
    lines = []
    for draw_seed in (11, 12, 13):
        rng = np.random.default_rng(draw_seed)
        n = N_NAMES * N_HOLDS
        W, y, sc, k_true = sim_windows_dx(n, rng)
        tt = np.repeat(np.arange(N_HOLDS), N_NAMES)
        tr = tt < int(N_HOLDS * TRAIN_FRAC)
        te = ~tr
        z60, kap = features(W)
        m_pt = z60 * (1 - np.exp(-np.clip(kap, 1e-3, 1) * H))
        mu_nfe, var_nfe = nfe_mean_var(net, W)
        c_m = float(np.mean(np.abs(mu_nfe[tr])) / (np.mean(np.abs(m_pt[tr])) + 1e-12))
        c_r = float(max(np.dot(y[tr], m_pt[tr]) / (np.dot(m_pt[tr], m_pt[tr]) + 1e-12), 0.0))
        m_or = z60 * (1 - np.exp(-k_true * H))
        arms = {
            "gauss": (m_pt, np.zeros_like(m_pt)),
            "point-ou": (m_pt, m_pt),
            "point-ou-matched": (m_pt, c_m * m_pt),
            "point-ou-recal": (m_pt, c_r * m_pt),
            "nfe": (mu_nfe / var_nfe, mu_nfe),
            "oracle-kappa": (m_or, m_or),
        }
        nh = int(te.sum() // N_NAMES)
        bi = np.random.default_rng(1).integers(0, nh, (NBOOT, nh))
        hdr = (f"draw seed {draw_seed}: matched scale {c_m:.3f}, recal slope {c_r:.3f}, "
               f"{nh} test holds")
        print(hdr, flush=True); lines.append(hdr)
        for name, (sig, mu) in arms.items():
            p, pr = book(sig[te], sc[te], y[te], tt[te], mu[te])
            d = pr - p
            gap = float(d.mean() * 1e4)
            se = float(np.array([d[i].mean() for i in bi]).std(ddof=1) * 1e4)
            sh = float(p.mean() / (p.std(ddof=1) + 1e-15) * np.sqrt(252.0 / H))
            msg = (f"  {name:<18} gap {gap:+8.1f} +- {se:5.1f} bp/hold   "
                   f"promise {pr.mean()*1e4:+8.1f}   realised {p.mean()*1e4:+8.1f}   "
                   f"Sharpe {sh:6.2f}")
            print(msg, flush=True); lines.append(msg)
    (OUT / "PROMISE_GAP_SIM.md").write_text(
        "# Promise-gap shrinkage controls on truth-known draws\n\n```\n"
        + "\n".join(lines) + "\n```\n")
    print(f"-> {OUT / 'PROMISE_GAP_SIM.md'}")


if __name__ == "__main__":
    main()
