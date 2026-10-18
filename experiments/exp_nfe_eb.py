"""Empirical-Bayes NFE: a per-name prior from the name's own train slice.

    uv run python experiments/exp_nfe_eb.py

PRE-REGISTRATION (frozen 2026-07-27, before the first run).

WHY.  The filter is trained once on a pooled prior: every one of the 120
names is told its half-life is log-uniform on 3-400 days.  Two run_260727
measurements say that leaves information on the table -- the screen shows
the filter can use a per-name prior when one genuinely exists (AUC 0.652
-> 0.891), and exp_kappa_persist shows one does, label-free on the real
AFM stream under the paper's own 30/70 split: train-slice half-life
predicts evaluation-slice half-life at Spearman +0.504 / +0.325 against a
shuffled-name null of -0.00 +- 0.09.  That same experiment killed the
characteristic route (own past kappa +0.245 out of sample, the seven
characteristics -0.142), so the lever is empirical Bayes on the name's own
recent speed, not what the name looks like.

METHOD.  Causal throughout.  On the train slice only, cut L=60 blocks, fit
the OLS AR(1) plug-in per (name, block), and shrink the per-name mean
toward the panel mean by the usual one-way factor
lambda_i = tau_b^2 / (tau_b^2 + sigma_w^2 / k_i), giving a per-name prior
N(c_i, tau_b^2 (1 - lambda_i)) on the true log2 half-life.  The trained
filter's MDN was fitted under a prior uniform in log2 half-life, so
reweighting its draws by that Gaussian IS the posterior under the per-name
prior; no retraining, and the filter itself is untouched.

ARMS (identical stream, costs, holds, benchmark):
    base      the fade rule on the AFM stream           (the paper's AFM row)
    nfe       posterior sizing, pooled prior            (the paper's NFE row)
    nfe_eb    posterior sizing, per-name prior
    nfe_ebsh  per-name prior with the centres SHUFFLED across names --
              matches the amount of shrinkage exactly and destroys only the
              name-to-prior correspondence.  THE ATTRIBUTION CONTROL.

GATES (verdicts only at these):
    G0 HARNESS  the nfe arm reproduces the paper's era-2 Sharpe (1.42) to
                within 0.02; otherwise the run is void and nothing is read.
    G1 PRIMARY  era-2 paired dSR(nfe_eb - nfe) > 0 at 2 SE.
    G2 ATTRIB   era-2 paired dSR(nfe_eb - nfe_ebsh) > 0 at 2 SE.  Without
                this the effect is shrinkage, not conditioning -- the null
                that killed the sizing overlays in VALUE_2X2.
    G3 ERA 1    era 1 not degraded: dSR(nfe_eb - nfe) >= -1 SE.
    G4 PATH     report the cumulative paths; a differentiated arm must be
                visibly separable from AFM, not merely a lower-risk copy.

RETRACTION.  If G2 fails, the gain is reported as unconditional shrinkage
and no per-name claim is made.

SCOPE.  Real KRX AFM residual streams, [exploration], one seed per arm.
Nothing is refit; the nfe/ package is not modified.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import torch

from nfe import L, H, NFE_KMIX, calib_from, featurize, sig_nfe, train_nfe

C_TURN, C_SHORT = 5e-4, 1e-4
NBOOT = 2000
HL_LO, HL_HI = 3.0, 400.0


# ------------------------------------------------------------------ book
def book(E, policy, rng=None, floor=False):
    T, N = E.shape
    pnl, tov, prev, w = [], [], np.zeros(N), np.zeros(N)
    for t in range(L, T):
        if (t - L) % H == 0:
            hist = E[t - L:t].T
            if not np.isfinite(hist).all():
                pnl.append(0.0); tov.append(0.0); continue
            w = rng.standard_normal(N) if floor else policy(hist)
            w = w - w.mean()
            n = np.abs(w).sum()
            w = w / n if n > 1e-12 else w
        p = w @ E[t]
        turn = np.abs(w - prev).sum()
        pnl.append(p - C_TURN * turn - C_SHORT * np.maximum(-w, 0).sum())
        tov.append(turn); prev = w
    return np.array(pnl), np.array(tov)


def metrics(p, tov, ann):
    eq = np.cumprod(1.0 + p)
    mdd = float(-(eq / np.maximum.accumulate(eq) - 1.0).min())
    return dict(sharpe=float(p.mean() / p.std(ddof=1) * ann),
                mu=float(p.mean() * ann ** 2 * 100),
                sd=float(p.std(ddof=1) * ann * 100),
                mdd=mdd, turn=float(tov.mean()),
                cvar5=float(np.mean(np.sort(p)[:max(1, len(p) // 20)]) * 1e4))


# ------------------------------------------------------- empirical Bayes
def train_slice_prior(E_tr):
    """(T,N) train-slice residuals -> per-name prior (centre, sd) on the true
    log2 half-life, by one-way shrinkage.  Causal: train slice only."""
    S = np.cumsum(np.nan_to_num(E_tr), axis=0)
    T, N = S.shape
    ys = []
    for t0 in range(0, T - L + 1, L):
        W = S[t0:t0 + L].T
        Z = W - W.mean(-1, keepdims=True)
        dz = np.diff(Z, axis=-1); x = Z[:, :-1]
        beta = (x * dz).sum(-1) / ((x * x).sum(-1) + 1e-18)
        kap = np.clip(-beta, np.log(2) / HL_HI, np.log(2) / HL_LO)
        y = np.log2(np.log(2) / kap)
        y[~(np.isfinite(W).all(-1) & (W.std(-1) > 1e-10))] = np.nan
        ys.append(y)
    Y = np.array(ys)                                    # (K, N)
    k = np.isfinite(Y).sum(0).astype(float)
    m = np.nanmean(Y, 0)
    gm = float(np.nanmean(m))
    sw2 = float(np.nanmean(np.nanvar(Y, 0, ddof=1)))    # within-name noise
    tb2 = max(float(np.nanvar(m, ddof=1)) - sw2 / max(np.nanmean(k), 1.0), 1e-4)
    lam = tb2 / (tb2 + sw2 / np.maximum(k, 1.0))
    c = gm + lam * (np.nan_to_num(m, nan=gm) - gm)
    sd = np.sqrt(tb2 * (1.0 - lam) + 1e-6)
    return c, sd, dict(grand_mean=gm, within_var=sw2, between_var=tb2,
                       mean_lambda=float(np.nanmean(lam)))


def sig_nfe_eb(hist, net, c, s, nsamp=64, seed=0):
    """Posterior sizing with the MDN draws reweighted by the per-name prior."""
    X = np.cumsum(hist, axis=-1)
    m, sd = X.mean(-1), X.std(-1) + 1e-12
    z = np.clip((X[:, -1] - m) / sd, -3, 3)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        o = net["mdn"](net["gru"](torch.tensor(featurize(X)))[0][:, -1])
        o = o.view(-1, 3, NFE_KMIX)
        w_ = torch.softmax(o[:, 0], -1)
        mu, sdm = o[:, 1], o[:, 2].clamp(-4, 2).exp()
        idx = torch.multinomial(w_, nsamp, replacement=True, generator=g)
        y = mu.gather(1, idx) + sdm.gather(1, idx) * \
            torch.randn(mu.shape[0], nsamp, generator=g)
    y = y.numpy()                                        # (N, nsamp) log2 HL
    lw = -0.5 * ((y - c[:, None]) / s[:, None]) ** 2     # log prior weight
    lw -= lw.max(1, keepdims=True)
    pw = np.exp(lw); pw /= pw.sum(1, keepdims=True)      # self-normalized
    ks = np.log(2) / np.clip(2.0 ** y, 1e-3, 1e4)
    a, b = np.exp(-ks * H), np.exp(-2 * ks * H)
    Ea = (pw * a).sum(1)
    Eb = (pw * b).sum(1)
    Va = (pw * a * a).sum(1) - Ea ** 2
    return -z * (1.0 - Ea) / ((1.0 - Eb) + z ** 2 * Va + 1e-12)


def sig_fade(hist):
    X = np.cumsum(hist, axis=-1)
    z = (X[:, -1] - X.mean(-1)) / (X.std(-1) + 1e-12)
    return -np.clip(z, -3, 3)


def main():
    ann = np.sqrt(247.0)
    out = {}
    paths = {}
    for era in (1, 2):
        eps = np.load(ROOT / f"results/run_260724/afm_stream_era{era}.npz")["eps"]
        T = len(eps)
        ntr = int(round(T * 0.30))
        cal = calib_from(eps[:ntr])
        net = train_nfe(**cal)
        c, s, info = train_slice_prior(eps[:ntr])
        rs = np.random.default_rng(0)
        csh = c[rs.permutation(len(c))]
        print(f"era{era}: T={T} train={ntr} phi_vol={cal['phi_vol']:.2f} "
              f"s_log={cal['s_log']:.2f} | prior centre {c.mean():.2f}"
              f" sd {s.mean():.3f} lambda {info['mean_lambda']:.2f}", flush=True)

        arms = {
            "base": lambda h: sig_fade(h),
            "nfe": lambda h, _n=net: sig_nfe(h, _n),
            "nfe_eb": lambda h, _n=net: sig_nfe_eb(h, _n, c, s),
            "nfe_ebsh": lambda h, _n=net: sig_nfe_eb(h, _n, csh, s),
        }
        rows = {k: book(eps, f) for k, f in arms.items()}
        rows["random"] = book(eps, None, rng=np.random.default_rng(7), floor=True)

        ev = np.arange(len(rows["base"][0])) >= (ntr - L)
        res = {}
        bi = np.random.default_rng(1).integers(0, int(ev.sum()), (NBOOT, int(ev.sum())))
        for k, (p, tv) in rows.items():
            res[k] = metrics(p[ev], tv[ev], ann)
        def gap(a, b):
            d = rows[a][0][ev] - rows[b][0][ev]
            g = d.mean() / (d.std(ddof=1) + 1e-15) * ann
            se = np.array([d[i].mean() / (d[i].std(ddof=1) + 1e-15)
                           for i in bi]).std(ddof=1) * ann
            return float(g), float(se)
        res["gaps"] = {"eb_minus_nfe": gap("nfe_eb", "nfe"),
                       "eb_minus_shuffled": gap("nfe_eb", "nfe_ebsh"),
                       "eb_minus_base": gap("nfe_eb", "base"),
                       "nfe_minus_base": gap("nfe", "base")}
        res["prior_info"] = info
        out[f"era{era}"] = res
        paths[f"era{era}"] = {k: np.cumsum(rows[k][0][ev]).tolist()
                              for k in ("base", "nfe", "nfe_eb", "nfe_ebsh")}
        print(f"  {'arm':<10}{'SR':>7}{'mu%':>7}{'sd%':>7}{'MDD':>7}{'turn':>8}")
        for k in ("base", "nfe", "nfe_eb", "nfe_ebsh", "random"):
            m = res[k]
            print(f"  {k:<10}{m['sharpe']:>7.2f}{m['mu']:>7.2f}{m['sd']:>7.2f}"
                  f"{m['mdd']*100:>7.1f}{m['turn']:>8.3f}", flush=True)
        for k, (g, se) in res["gaps"].items():
            print(f"    {k:<20}{g:+.3f} +- {se:.3f}", flush=True)

    d = ROOT / "results" / "run_260727"
    d.mkdir(parents=True, exist_ok=True)
    (d / "nfe_eb.json").write_text(json.dumps(out, indent=2))
    (d / "nfe_eb_paths.json").write_text(json.dumps(paths))

    print("\n=== GATES ===")
    s2 = out["era2"]
    hn = s2["nfe"]["sharpe"]
    print(f"G0 harness: era2 nfe SR {hn:.2f} vs paper 1.42  "
          f"{'PASS' if abs(hn - 1.42) <= 0.02 else 'FAIL — run is void'}")
    g, se = s2["gaps"]["eb_minus_nfe"]
    print(f"G1 eb - nfe (era2)      : {g:+.3f} +- {se:.3f}  "
          f"{'PASS' if g > 2 * se else 'FAIL'}")
    g2, se2 = s2["gaps"]["eb_minus_shuffled"]
    print(f"G2 eb - shuffled (era2) : {g2:+.3f} +- {se2:.3f}  "
          f"{'PASS' if g2 > 2 * se2 else 'FAIL'}  <- attribution")
    g3, se3 = out["era1"]["gaps"]["eb_minus_nfe"]
    print(f"G3 eb - nfe (era1)      : {g3:+.3f} +- {se3:.3f}  "
          f"{'PASS' if g3 >= -se3 else 'FAIL'}")
    print(f"\n-> {d / 'nfe_eb.json'}")


if __name__ == "__main__":
    main()
