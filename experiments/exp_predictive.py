"""Route (2): a PREDICTIVE posterior p(dX | window) — no simulator, no G4 gate.

    uv run python experiments/exp_predictive.py [--smoke] [--seed N]

[exploration] (2026-07-22). PRE-REGISTRATION — frozen before the first full run.

WHY. Targeting p(theta, z | window) needs a simulator, which puts every
real-data posterior claim behind the G4 misspecification gate. The decision
layer never consumes theta, only a distribution over the traded quantity, so
target p(dX_{t+H} | causal window) directly, on REAL data against REALIZED
outcomes: no simulator, no gate, and calibration checkable on held-out data
rather than by SBC on a simulator we know is only approximately right. The
target need not be Bayesian — the claim rests on verifiable calibration — so
boosted quantile regression is a first-class arm.

ARMS (identical causal features; only the output object differs)
    point-ou     OU plug-in, POINT forecast — no distribution at all
    gauss        N(0, vol60*sqrt(H)) — distribution, unlearned mean
    gb-quantile  9 boosted quantile regressions
    mdn          5-component mixture density network

METRICS. PIT KS and coverage at 50/80/90 (calibration); CRPS (accuracy);
mean 80% width (sharpness); book Sharpe sizing by E[dX]/Var[dX] (decision);
and the promise-realized gap per hold — the book's own forecast minus what it
earned. That gap is THE instrument: a method that cannot form it has no
self-audit.

HYPOTHESES (registered)
  H1  gb-quantile and mdn are calibrated out of sample: |empirical - nominal|
      coverage within 2 SE at the 80% and 90% levels.
  H2  gb-quantile beats gauss on CRPS.
  H3  sizing by the predictive distribution beats the point plug-in in the book.
  H4  the promise-realized gap is bounded for the calibrated arms and inflates
      for point-ou.
  H5  (NULL) on a no-mean-reversion surrogate the calibration results survive
      but every CRPS/decision advantage vanishes. If an advantage persists on
      the surrogate, it is pipeline, not signal.

GATES. Verdicts only at H1-H5, on the held-out split only, 2 SE against a
bootstrap over holds. If H3 fails, no decision claim is made — calibration is
then reported as honesty without alpha, consistent with the rest of the project.

FROZEN PROTOCOL. KRX daily, 120 names, rolling-252d PCA-3 residuals (the
admitted construction), eras split at 2018-04, per era train = first 30% /
evaluate = last 70%. Features are causal window statistics: z at 5/20/60/250d,
momentum at 60/250d, OLS kappa, vol, skew, kurtosis. Target = dX over H=20
days, standardised by vol60*sqrt(H). Book conventions identical to E3
(dollar-neutral, unit gross, 5bp turnover + 1bp shorting).

SCOPE. `[exploration]`, one seed. Sharpe levels are not a profitability claim.

AMENDMENTS (registered predictions and verdicts in run_260723/RUN_LOG.md)
  2026-07-23a  nfe-direct gains a split-conformal width layer (CQR family,
               multiplicative around the median), train slice only.
               Predicted: cov90 -> ~0.90, CRPSS toward 0, resolution unmoved.
  2026-07-23b  +6 point-in-time DART features on every learner arm (the
               same-dataset requirement); the NULL surrogate keeps them, so a
               fundamental-channel artifact would have to show up there too.
  2026-07-23c  (supersedes 23a's single-arm scope) every learned arm gets the
               conformal layer: outcome-fitted arms calibrate on the train
               slice's embargoed tail, simulator arms on the full slice they
               never saw; nfe-direct's simulator target scale is matched to
               the train slice (sim 1.47 vs real 1.08). Predicted: every arm
               reaches cov90 ~ 0.90, mdn's CRPS improves, resolution moves
               nowhere, no arm attains positive CRPSS.
  2026-07-28   --seed N: training-seed sweep for the honesty table. Reseeds
               only the learners; data-side rngs stay fixed, so the sweep
               isolates training randomness on identical data. seed 0
               reproduces the ledger run and keeps its run_260723 path.
               Predicted: the honesty ordering holds on every seed; any seed
               that breaks it is reported, not dropped.
  2026-10-13   Two SHRINKAGE CONTROLS for the promise gap (camera-ready check:
               is the amortized arms' smaller gap uncertainty quantification
               or timidity?). Positions are
               normalised to unit gross per hold, so scaling a forecast
               changes the promise and not one trade: both controls hold
               point-ou's positions exactly and change only the promise.
                 point-ou-matched  the point forecast scaled by one constant
                                   so that its mean absolute promise on the
                                   TRAIN slice equals the nfe arm's -- the
                                   posterior's average caution, applied
                                   uniformly, with no conditional content.
                 point-ou-recal    the point forecast scaled by its OLS slope
                                   against realised outcomes on the TRAIN
                                   slice -- the strongest scalar an
                                   outcome-fitted recalibration can buy.
               Registered reading. point-ou-recal's gap will be small, likely
               below every amortized arm's, because it fits a scalar to
               outcomes the amortized arms never see; that is a bound, not a
               rival method. The comparison that decides the paper's claim
               is point-ou-matched against nfe: if nfe's gap is smaller, the
               posterior's shrinkage is conditional (it is cautious on the
               right windows); if the two gaps agree within their bootstrap
               SE, the gap reduction is uniform caution and the paper must
               say so. Sharpe is identical to point-ou for both by
               construction and is not read.
               MEASURED 2026-10-15 on the PUBLIC proxy panel (NFE_PANEL=public,
               NFE_BATCH=2048, seed 0; results/run_261015/PROMISE_GAP_PUBLIC.md;
               gap in bp/hold, bootstrap SE over holds, paired difference
               nfe minus point-ou-matched):
                 era2 REAL  point-ou 1113+-35  matched 602+-25  nfe 589+-24
                            nfe-direct 160+-22  gb 259+-32  recal 41+-21
                            paired nfe - matched  -13 +- 15
                 era1 REAL  point-ou  972+-25  matched 507+-19  nfe 503+-20
                            paired nfe - matched   -4 +- 11
                 NULL (no MR) era1/era2: paired -22 +- 11 / -36 +- 10
               Reading: on both real eras nfe and the uniformly shrunk point
               agree within one SE (matched scale 0.53-0.55), so the
               amortized arm's gap reduction over the point plug-in is
               uniform caution; the only paired difference beyond 2 SE is on
               the no-mean-reversion null, where the posterior is slightly
               more cautious still. point-ou-recal's slope is 0.00-0.05: the
               point forecast's magnitude is nearly unrelated to realised
               outcomes on every panel, so the residual gap shared by every
               arm is the level taken from the window, not the estimator.
               The honesty ordering of the paper's Table 2 reproduces on this
               panel (point-ou > mdn > nfe > gb > nfe-direct; CRPSS <= 0 for
               every arm). Not the paper's universe: within-panel reading only.
  2026-10-17   Same two controls on NFE_PANEL=vendor2: the paper's universe RULE
               (KOSPI common, top-200 by median trading value, the 120 unbroken
               names from 2010-01-04, panel end 2026-07-20 = the 4,072nd
               trading day of Table 4) rebuilt from a second vendor's adjusted
               closing prices, so the same rule on different source prices
               (not redistributable; not shipped). Registered
               reading, before the run: identical to 2026-10-13 -- the
               decisive comparison is point-ou-matched against nfe; agreement
               within bootstrap SE on both real eras = uniform caution, and the
               paper's sentence stands; a paired gap beyond 2 SE in nfe's
               favour on a real era = conditional content, and the paper's
               sentence must be revised. Secondary: the Table 2 ordering and
               CRPSS <= 0 are expected to reproduce. Artifact:
               results/run_261017/predictive_vendor2_seed0.json.
               MEASURED 2026-10-17 (results/run_261017/PROMISE_GAP_VENDOR2.md; gap in
               bp/hold, bootstrap SE over holds, paired = nfe minus matched):
                 era2 REAL  point-ou 1102+-41  matched 577+-29  nfe 579+-29
                            mdn 591+-70  gb 227+-35  direct 138+-26  recal -11+-23
                            paired  +1 +- 17
                 era1 REAL  point-ou 1003+-27  matched 528+-19  nfe 515+-20
                            paired -13 +- 12
                 NULL (no MR) era1/era2: paired -42 +- 10 / -38 +- 10
               Reading: neither real era shows a paired gap beyond 2 SE in nfe's
               favour (era2 within 1 SE, era1 at 1.05 SE), so on the paper's
               own universe rule the reduction is uniform caution; the Table 2
               ordering and CRPSS <= 0 reproduce (point-ou > mdn > nfe > gb >
               direct). Null panels: the posterior is more cautious than the
               matched point beyond 2 SE, as on the public panel. Recal slope
               0.015-0.026.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT))

from sklearn.ensemble import HistGradientBoostingRegressor            # noqa: E402
from exp_e3_book import (load_panel, load_fundamentals, r_pca,          # noqa: E402
                         C_TURN, C_SHORT, L, H)

FUND_NAMES = ["roe", "de", "margin", "rev_yoy", "bp", "ep"]

SMOKE = "--smoke" in sys.argv
SEED = int(sys.argv[sys.argv.index("--seed") + 1]) if "--seed" in sys.argv else 0
QS = [0.05, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 0.95]
SPLIT = "2008-01-01"
NBOOT = 2000
ARMS = ["gauss", "point-ou", "gb-quantile", "mdn", "nfe", "nfe-direct",
        "point-ou-matched", "point-ou-recal"]   # the last two: 2026-10-13 controls
# gauss (the climatological reference) is scored FIRST: every other arm's
# skill score is relative to it.
# seed 0 is the ledger run and keeps its run_260723 path; the robustness
# sweep (amendment 2026-07-28) lands beside the other run_260728 outputs.
import os                                                              # noqa: E402
PANEL_SRC = os.environ.get("NFE_PANEL", "krx")
ARTIFACT = ((ROOT / "results" / "run_260723" / "predictive.json" if SEED == 0
             else ROOT / "results" / "run_260728" / f"predictive_seed{SEED}.json")
            if PANEL_SRC == "krx" else
            ROOT / "results" / ("run_261017" if PANEL_SRC == "vendor2" else "run_261015")
            / f"predictive_{PANEL_SRC}_seed{SEED}.json")
ROWS = []                          # every printed cell is also written to disk


# --------------------------------------------------------------- features
def build(E, stride=5, FUND=None):
    """E (T,N) residual returns -> X (n,F), y (n,), plus book bookkeeping.

    The residual constructions carry their own burn-in (rolling PCA emits NaN for
    the first 252 days). Those rows must be TRIMMED, not zero-filled: a window
    lying inside them has zero volatility, and the standardised target
    dX/(vol*sqrt(H)) then explodes (measured: 8.8e9 at the 99.9th percentile,
    which blew up the boosted quantile fit). Samples whose scale is still
    degenerate are dropped."""
    ok = np.isfinite(E).all(1)
    first = int(np.argmax(ok)) if ok.any() else len(E)
    E = E[first:]
    Fn = None if FUND is None else FUND[first:]
    Xc = np.cumsum(E, axis=0)
    T, N = Xc.shape
    t0s = np.arange(max(L, 250), T - H, stride)
    F, y, sc, tt, aa, WIN = [], [], [], [], [], []
    for t in t0s:
        x = Xc[t - 1]
        f = {}
        for w in (5, 20, 60, 250):
            W = Xc[t - w:t]
            f[f"z{w}"] = -(x - W.mean(0)) / (W.std(0) + 1e-12)
        for w in (60, 250):
            f[f"mom{w}"] = x - Xc[t - w]
        W = Xc[t - 60:t].T
        d, xl = np.diff(W, axis=-1), W[:, :-1]
        xcm = xl - xl.mean(-1, keepdims=True)
        b = (xcm * (d - d.mean(-1, keepdims=True))).sum(-1) / ((xcm ** 2).sum(-1) + 1e-18)
        f["kap"] = np.clip(-b, -1, 1)
        r = np.diff(Xc[t - 60:t], axis=0)
        v = r.std(0) + 1e-12
        f["vol"] = v
        rs = (r - r.mean(0)) / v
        f["skew"] = (rs ** 3).mean(0)
        f["kurt"] = (rs ** 4).mean(0)
        if Fn is not None:            # point-in-time fundamentals, same dataset
            # cross-sectional RANK in [-0.5, 0.5], as the AFM side consumes
            # them: raw de/ep carry O(100) outliers that blew up the MDN on
            # the first launch (CRPS 3.2e7); ranks are a monotone transform,
            # so the tree arm loses nothing and the neural arm survives.
            for k, nmf in enumerate(FUND_NAMES):
                c = Fn[t - 1, :, k].copy()
                fin = np.isfinite(c)
                c[~fin] = np.median(c[fin]) if fin.any() else 0.0
                o = c.argsort().argsort().astype(float)
                f[nmf] = o / max(len(c) - 1, 1) - 0.5
        s = v * np.sqrt(H)                                  # per-name scale
        WIN.append(Xc[t - 60:t].T.copy())                   # raw window for NFE
        F.append(np.column_stack([np.nan_to_num(f[k]) for k in f]))
        y.append((Xc[t + H] - Xc[t]) / s)                   # standardised target
        sc.append(s); tt.append(np.full(N, t)); aa.append(np.arange(N))
    F, y, sc = np.vstack(F), np.concatenate(y), np.concatenate(sc)
    tt, aa = np.concatenate(tt), np.concatenate(aa)
    WIN = np.vstack([w for w in WIN])
    keep = (np.isfinite(F).all(1) & np.isfinite(y) &
            (sc > np.quantile(sc, 0.001)) & (np.abs(y) < 20.0))
    return F[keep], y[keep], sc[keep], tt[keep], aa[keep], list(f), WIN[keep]


# --------------------------------------------------------------- predictors
_NFE_NET = None
_NFED_NET = None
# NFE_BATCH=<rows>: compute nfe_quantiles in batches of that many windows.
# Off by default (0) because the posterior draws are generated per call, so
# batching changes which Monte Carlo draws each row receives: statistically
# equivalent, not bit-identical to the ledger run. Set it on machines where
# the unbatched (N, 64, 301) intermediates do not fit (2026-10-15: 15 GB).
NFE_BATCH = int(os.environ.get("NFE_BATCH", "0"))


def _sim_windows_dx(n, rng, phi_vol=0.19, s_log=0.14, tau=0.05):
    """Draws from the ADMITTED process (2-state switching + SV + lognormal +
    level drift), returning (window, standardised H-day forward move). The
    labels are the object the decision layer consumes -- no theta detour."""
    hl = np.exp(rng.uniform(np.log(3), np.log(400), n))
    k = np.log(2) / hl
    st = (rng.random(n) < 0.5)                      # start regime; spell 250d
    m_ln = np.exp(s_log ** 2 / 2)
    sd_ln = np.sqrt((np.exp(s_log ** 2) - 1) * np.exp(s_log ** 2))
    x = np.zeros(n); lv = np.zeros(n); m = np.zeros(n)
    W = np.empty((n, 60))
    T = 200 + 60 + H
    ks = np.where(st, k, np.log(2) / 400.0)          # slow state near-RW
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
            x0 = x.copy()
    vol = np.abs(np.diff(W, axis=-1)).std(-1) + 1e-12
    y = (x - x0) / (vol * np.sqrt(H))                # standardised, same as real
    return W, np.clip(y, -20, 20)


def train_nfe_direct(seed=0, n=120_000, steps=4000, target_sd=None):
    """Amortize p(dX | window) DIRECTLY: same GRU-64 + MDN, but the MDN target
    is the standardised forward move, not log2 half-life. Frees the predictive
    mean from the OU functional form -z(1-e^{-kH}) -- which the feature audit
    (what_pays) showed carries ~zero real-data signal -- and lets the variance
    respond to the volatility state the simulator carries. The TabPFN recipe,
    amortized over our admitted process instead of generic tabular tasks."""
    import torch, torch.nn as nn
    from nfe import featurize as _feat
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    W, y = _sim_windows_dx(n, rng)
    if target_sd is not None:          # match the sim target scale to the
        y = y * (target_sd / (y.std() + 1e-12))   # train slice (one scalar)
    X = torch.tensor(_feat(W)); Y = torch.tensor(y, dtype=torch.float32)
    K = 5
    net = nn.ModuleDict({"gru": nn.GRU(2, 64, batch_first=True),
                         "mdn": nn.Linear(64, 3 * K)})
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for _ in range(steps):
        i = torch.randint(0, len(X), (512,))
        o = net["mdn"](net["gru"](X[i])[0][:, -1]).view(-1, 3, K)
        lw = torch.log_softmax(o[:, 0], -1); mu = o[:, 1]; ls = o[:, 2].clamp(-4, 3)
        ll = lw - ls - 0.5 * ((Y[i, None] - mu) / ls.exp()) ** 2
        loss = -torch.logsumexp(ll, -1).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    return net


def nfed_quantiles(Xwin, qs=QS, target_sd=None):
    global _NFED_NET
    import torch
    from nfe import featurize as _feat
    if _NFED_NET is None:
        _NFED_NET = train_nfe_direct(seed=SEED, target_sd=target_sd)
    with torch.no_grad():
        o = _NFED_NET["mdn"](_NFED_NET["gru"](torch.tensor(_feat(Xwin)))[0][:, -1])
        o = o.view(-1, 3, 5)
        w = torch.softmax(o[:, 0], -1).numpy()
        mu, sd = o[:, 1].numpy(), o[:, 2].clamp(-4, 3).exp().numpy()
    return mix_quantiles(w, mu, sd, qs)


def conformal_lam(Qtr, ytr, qs=QS):
    """Split-conformal width calibration in the CQR family (Romano et al.
    2019), multiplicative around the median. Fitted on the TRAIN slice only.

    WHY multiplicative: the diagnosed defect is SCALE -- the simulator's
    standardised H-day target has sd 1.47 vs 1.08 on real windows, so the
    amortized predictive is uniformly too wide (cov90 0.96-0.99). A per-level
    factor lambda_q on the offset from the median corrects a scale error
    exactly and leaves the (already dead, measured reso 0.02) conditional
    ordering untouched -- so any resolution gain after this is real, not an
    artifact of the width fix."""
    med = Qtr[:, len(qs) // 2]
    lam = np.ones(len(qs))
    for j, q in enumerate(qs):
        off = Qtr[:, j] - med
        ok = np.abs(off) > 1e-9
        if abs(q - 0.5) < 1e-9 or ok.sum() < 50:
            continue
        s = (ytr[ok] - med[ok]) / off[ok]
        lam[j] = np.quantile(s, q if q > 0.5 else 1 - q)
    return np.clip(lam, 0.1, 10.0)


def conformal_apply(Q, lam, qs=QS):
    med = Q[:, len(qs) // 2]
    return np.sort(med[:, None] + lam[None, :] * (Q - med[:, None]), axis=1)


def nfe_quantiles(Xwin, qs=QS, seed=0):
    """The amortized filter as a PREDICTIVE distribution, via the closed form

        dX | theta, x  ~  N( (m - x)(1 - e^{-kH}),  sigma^2 (1-e^{-2kH})/(2k) )

    theta samples come from the NFE posterior p(theta | window) -- trained ONCE
    on the simulator, applied in ONE forward pass, no per-name refitting. The
    mixture over 64 posterior draws is the predictive density; its quantiles
    enter the same scoring as every other arm.

    sigma is not in the posterior (the MDN outputs half-life only), so the
    conditional variance uses the window's own increment scale -- recorded as a
    design choice: theta-uncertainty enters through kappa, scale is conditioned
    on. In standardised target units sigma_daily^2 = 1/H exactly.
    """
    global _NFE_NET
    from nfe import train_nfe, nfe_kappa_samples
    if _NFE_NET is None:
        _NFE_NET = train_nfe(seed=SEED, phi_vol=0.19,
                             s_log=0.14)   # gate-calibrated values
    if NFE_BATCH and len(Xwin) > NFE_BATCH:
        return np.vstack([nfe_quantiles(Xwin[i:i + NFE_BATCH], qs, seed)
                          for i in range(0, len(Xwin), NFE_BATCH)])
    ks = nfe_kappa_samples(_NFE_NET, Xwin, seed=seed)     # (N, nsamp)
    m = Xwin.mean(-1); sd = Xwin.std(-1) + 1e-12
    z = np.clip((Xwin[:, -1] - m) / sd, -3, 3)
    mu_d = (-z)[:, None] * (1.0 - np.exp(-ks * H))        # target units
    var_d = (1.0 - np.exp(-2 * ks * H)) / (2 * ks * H)
    sd_d = np.sqrt(np.maximum(var_d, 1e-10))
    from scipy.stats import norm
    lo = (mu_d - 4 * sd_d).min(1, keepdims=True)
    hi = (mu_d + 4 * sd_d).max(1, keepdims=True)
    g = lo + (hi - lo) * np.linspace(0, 1, 301)[None, :]
    cdf = norm.cdf((g[:, None, :] - mu_d[:, :, None]) / sd_d[:, :, None]).mean(1)
    out = np.empty((len(Xwin), len(qs)))
    for j, q in enumerate(qs):
        idx = np.argmin(np.abs(cdf - q), axis=1)
        out[:, j] = g[np.arange(len(Xwin)), idx]
    return out


def fit_gb(Xtr, ytr, Xte, n_iter):
    qs = np.empty((len(Xte), len(QS)))
    for j, q in enumerate(QS):
        m = HistGradientBoostingRegressor(loss="quantile", quantile=q,
                                          max_iter=n_iter, random_state=SEED)
        qs[:, j] = m.fit(Xtr, ytr).predict(Xte)
    return np.sort(qs, axis=1)                              # enforce monotonicity


def fit_mdn(Xtr, ytr, Xte, steps):
    import torch
    import torch.nn as nn
    torch.manual_seed(SEED)
    Kc = 5
    mu_x, sd_x = Xtr.mean(0), Xtr.std(0) + 1e-9
    tx = lambda A: torch.tensor((A - mu_x) / sd_x, dtype=torch.float32)
    net = nn.Sequential(nn.Linear(Xtr.shape[1], 64), nn.ReLU(),
                        nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 3 * Kc))
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    Xt, yt = tx(Xtr), torch.tensor(ytr, dtype=torch.float32)
    n = len(Xt)
    for i in range(steps):
        idx = torch.randint(0, n, (min(2048, n),))
        o = net(Xt[idx]).view(-1, 3, Kc)
        lw = torch.log_softmax(o[:, 0], -1)
        mu, ls = o[:, 1], o[:, 2].clamp(-4, 3)
        ll = lw - ls - 0.5 * ((yt[idx, None] - mu) / ls.exp()) ** 2
        loss = -torch.logsumexp(ll, -1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        o = net(tx(Xte)).view(-1, 3, Kc)
        w = torch.softmax(o[:, 0], -1).numpy()
        mu, sd = o[:, 1].numpy(), o[:, 2].clamp(-4, 3).exp().numpy()
    return w, mu, sd


def mix_quantiles(w, mu, sd, qs=QS, grid=401):
    """Numeric quantiles of a Gaussian mixture."""
    lo = (mu - 4 * sd).min(1)[:, None]
    hi = (mu + 4 * sd).max(1)[:, None]
    g = lo + (hi - lo) * np.linspace(0, 1, grid)[None, :]
    from scipy.stats import norm
    cdf = np.einsum("nk,nkg->ng", w,
                    norm.cdf((g[:, None, :] - mu[:, :, None]) / sd[:, :, None]))
    out = np.empty((len(w), len(qs)))
    for j, q in enumerate(qs):
        idx = np.argmin(np.abs(cdf - q), axis=1)
        out[:, j] = g[np.arange(len(w)), idx]
    return out


# --------------------------------------------------------------- scoring
def crps_from_q(qs_pred, y, qs=QS):
    """Pinball-loss average = an unbiased CRPS proxy on the quantile grid."""
    d = y[:, None] - qs_pred
    return float(np.mean(np.maximum(np.array(qs) * d, (np.array(qs) - 1) * d)) * 2)


def crpss(cr, cr_clim):
    """Skill against the CLIMATOLOGICAL reference (Wilks; Gneiting & Raftery).
    The unconditional N(0, sigma_hat sqrt(H)) forecast uses no information from
    the window, so it is the climatology for this problem. Scoring against it
    is the standard fix for the fact that a proper score alone rewards a
    forecast that never commits: climatology scores exactly 0 skill."""
    return 1.0 - cr / (cr_clim + 1e-15)


def resolution(qs_pred, y, qs=QS, nbin=10):
    """Hersbach-style split of the proper score into RELIABILITY and
    RESOLUTION. Climatology is perfectly reliable and has zero resolution: it
    is the same forecast every time. Resolution here is the variance of the
    predictive mean across cases, normalised by the variance of the outcome --
    how much the forecast actually moves with the input."""
    qa = np.array(qs)
    mu = np.trapezoid(qs_pred, qa, axis=1) / (qa[-1] - qa[0])
    return float(np.var(mu) / (np.var(y) + 1e-15))


def pit_and_cov(qs_pred, y, qs=QS):
    below = (qs_pred < y[:, None]).sum(1)
    u = np.clip(below / len(qs), 1e-6, 1 - 1e-6)
    ks = float(np.max(np.abs(np.sort(u) - np.linspace(0, 1, len(u)))))
    cov = {}
    for lvl, (a, b) in {50: (2, 6), 80: (1, 7), 90: (0, 8)}.items():
        cov[lvl] = float(np.mean((y >= qs_pred[:, a]) & (y <= qs_pred[:, b])))
    return ks, cov


def moments(qs_pred, qs=QS):
    """Mean and variance from the quantile grid (trapezoid over q)."""
    qa = np.array(qs)
    m = np.trapezoid(qs_pred, qa, axis=1) / (qa[-1] - qa[0])
    v = np.trapezoid((qs_pred - m[:, None]) ** 2, qa, axis=1) / (qa[-1] - qa[0])
    return m, np.maximum(v, 1e-9)


def book(sig, scale, y, tt, ann, mu_pred=None):
    """sig = position per (hold, name); y = realised standardised move.

    Returns (net P&L per hold, Sharpe, promised P&L per hold). The promise is the
    book's OWN forecast of what it will earn, w . E[dX] — the self-audit
    quantity. A method with no predictive mean cannot form it.
    """
    ts = np.unique(tt)
    pnl, prom, prev = [], [], None
    for t in ts:
        k = tt == t
        w = sig[k]
        w = w - w.mean()
        n = np.abs(w).sum()
        if n < 1e-12:
            pnl.append(0.0); prom.append(0.0); continue
        w = w / n
        if prev is None or len(prev) != len(w):
            prev = np.zeros_like(w)
        turn = np.abs(w - prev).sum()
        cost = C_TURN * turn + C_SHORT * np.maximum(-w, 0).sum()
        pnl.append(w @ (y[k] * scale[k]) - cost)
        prom.append((w @ (mu_pred[k] * scale[k]) - cost) if mu_pred is not None
                    else np.nan)
        prev = w
    p = np.array(pnl)
    return p, float(p.mean() / (p.std(ddof=1) + 1e-15) * np.sqrt(252.0 / H)), np.array(prom)


def main():
    global _NFED_NET
    R, RF, dates = load_panel()
    ann = np.sqrt(len(R) / ((dates[-1] - dates[0]).days / 365.25))
    mid = len(R) // 2
    n_iter, steps, stride = (30, 400, 20) if SMOKE else (200, 4000, 5)
    print(f"[{'smoke' if SMOKE else 'exploration'}] predictive posterior "
          f"p(dX|window) — no simulator, no G4 gate (training seed {SEED})")
    print(f"  KRX ({PANEL_SRC} panel) {len(R)}d x {R.shape[1]} | PCA-3 residuals | H={H} | "
          f"stride={stride}\n", flush=True)

    eras = [(f"era1 {str(dates[0])[:7]}", slice(0, mid)),
            (f"era2 {str(dates[mid])[:7]}", slice(mid, None))]
    if SMOKE:
        eras = eras[1:]
    rng = np.random.default_rng(0)

    FUND = load_fundamentals(dates)
    print(f"  fundamentals: {'attached (+6 features)' if FUND is not None else 'not on disk'}\n", flush=True)
    for elab, sl in eras:
        Re = R[sl]
        Fe = None if FUND is None else FUND[sl]
        E = r_pca(Re, None)
        # NULL surrogate: no mean reversion, vol matched per name. The REAL
        # fundamentals stay attached: a fundamental-channel artifact must
        # show up on the null too.
        En = rng.standard_normal(Re.shape) * np.nanstd(E, axis=0)
        for src, Ecur in [("REAL", E), ("NULL rw (no MR)", En)]:
            _NFED_NET = None       # one direct net per panel (23c: its target
            X, y, sc, tt, aa, fn, WINS = build(Ecur, stride, FUND=Fe)  # scale is panel-causal)
            ts = np.unique(tt)
            cut = ts[int(len(ts) * 0.3)]
            tr, te = tt < cut, tt >= cut
            print(f"  === {elab} · {src} === train {tr.sum():,} / test {te.sum():,}"
                  f" rows, {len(np.unique(tt[te]))} holds", flush=True)
            # BOOK EVALUATION USES NON-OVERLAPPING HOLDS ONLY. Training keeps
            # every stride-`stride` sample, but consecutive samples share 15 of
            # 20 target days, so a book scored on all of them has dependent
            # "holds", an inflated Sharpe and a meaningless SE.
            tte = np.unique(tt[te])
            keep_t = set(tte[::max(1, H // stride)].tolist())
            bk = te & np.array([t in keep_t for t in tt])
            nh = len(np.unique(tt[bk]))
            bi = np.random.default_rng(1).integers(0, nh, (NBOOT, nh))
            # FLOOR: turnover-matched — one random draw HELD across the hold,
            # not redrawn each time (a redrawn floor loses on cost alone; the
            # same mistake was found and fixed in exp_e3_book.py).
            fl = np.empty(bk.sum())
            tb = tt[bk]
            for i, t in enumerate(np.unique(tb)):
                fl[tb == t] = rng.standard_normal((tb == t).sum())
            pf, shf, _ = book(fl, sc[bk], y[bk], tb, ann)
            print(f"  {'arm':<12}{'CRPS':>8}{'CRPSS':>8}{'reso':>7}{'PIT KS':>8}"
                  f"{'cov90':>7}{'wid80':>7}{'Sharpe':>8}{'promise':>10}")
            preds = {}
            kap, vol = X[:, fn.index("kap")], X[:, fn.index("vol")]
            z60 = X[:, fn.index("z60")]
            # uniform split-conformal (amendment 23c): outcome-fitted arms
            # fit on the first 80% of the train slice and calibrate on its
            # L-embargoed tail; simulator-trained arms calibrate on the full
            # train slice (they never saw it).
            c8 = ts[int(len(ts) * 0.24)]
            trf = tt < c8
            trc = tr & (tt >= c8 + L)
            m_pt = z60 * (1 - np.exp(-np.clip(kap, 1e-3, 1) * H))   # point-ou forecast
            for arm in ARMS:
                if arm == "point-ou":
                    m = m_pt
                    q = np.repeat(m[:, None], len(QS), 1)
                elif arm == "point-ou-matched":
                    # 2026-10-13 control: uniform shrinkage matched to the nfe
                    # arm's mean absolute predictive mean on the train slice.
                    mu_nfe_tr = moments(nfe_quantiles(WINS[tr]))[0]
                    c_m = float(np.mean(np.abs(mu_nfe_tr))
                                / (np.mean(np.abs(m_pt[tr])) + 1e-12))
                    q = np.repeat((c_m * m_pt)[:, None], len(QS), 1)
                    print(f"  [control] point-ou-matched: scale {c_m:.3f}", flush=True)
                elif arm == "point-ou-recal":
                    # 2026-10-13 control: OLS slope of realised outcomes on the
                    # point forecast, train slice only, clipped at zero.
                    c_r = float(max(np.dot(y[tr], m_pt[tr])
                                    / (np.dot(m_pt[tr], m_pt[tr]) + 1e-12), 0.0))
                    q = np.repeat((c_r * m_pt)[:, None], len(QS), 1)
                    print(f"  [control] point-ou-recal: slope {c_r:.3f}", flush=True)
                elif arm == "gauss":
                    from scipy.stats import norm
                    q = norm.ppf(np.array(QS))[None, :] * np.ones((len(X), 1))
                elif arm == "gb-quantile":
                    q = np.full((len(X), len(QS)), np.nan)
                    Qct = fit_gb(X[trf], y[trf], np.vstack([X[trc], X[te]]),
                                 n_iter)
                    nc = int(trc.sum())
                    lam = conformal_lam(Qct[:nc], y[trc])
                    q[te] = conformal_apply(Qct[nc:], lam)
                elif arm == "nfe":
                    q = np.full((len(X), len(QS)), np.nan)
                    lam = conformal_lam(nfe_quantiles(WINS[tr]), y[tr])
                    q[te] = conformal_apply(nfe_quantiles(WINS[te]), lam)
                elif arm == "nfe-direct":
                    q = np.full((len(X), len(QS)), np.nan)
                    tsd = float(np.std(y[tr]))
                    lam = conformal_lam(nfed_quantiles(WINS[tr],
                                                       target_sd=tsd), y[tr])
                    q[te] = conformal_apply(nfed_quantiles(WINS[te]), lam)
                else:
                    w_, mu_, sd_ = fit_mdn(X[trf], y[trf],
                                           np.vstack([X[trc], X[te]]), steps)
                    Qct = mix_quantiles(w_, mu_, sd_)
                    nc = int(trc.sum())
                    lam = conformal_lam(Qct[:nc], y[trc])
                    q = np.full((len(X), len(QS)), np.nan)
                    q[te] = conformal_apply(Qct[nc:], lam)
                qt = q[te]
                cr = crps_from_q(qt, y[te])
                ks, cov = pit_and_cov(qt, y[te])
                mu, var = moments(qt)
                if arm.startswith("point-ou"):
                    sig = m_pt[te]            # the controls trade point-ou's book
                else:
                    sig = mu / var
                kb = bk[te]
                pb, sh, pr = book(sig[kb], sc[bk], y[bk], tt[bk], ann,
                                  mu_pred=mu[kb])
                d = pb - pf
                gap = d.mean() / (d.std(ddof=1) + 1e-15) * ann
                se = np.array([d[i].mean() / (d[i].std(ddof=1) + 1e-15)
                               for i in bi]).std(ddof=1) * ann
                pg = float(np.nanmean(pr - pb)) * 1e4
                ROWS.append(dict(era=elab, panel=src, arm=arm, crps=cr, pit_ks=ks,
                                 resolution=resolution(qt, y[te]),
                                 cov50=cov[50], cov80=cov[80], cov90=cov[90],
                                 width80=float(np.mean(qt[:, 7] - qt[:, 1])),
                                 sharpe=sh, vs_floor=gap, vs_floor_se=se,
                                 promise_gap_bp=pg, n_test=int(te.sum()),
                                 n_holds=int(nh),
                                 # per-hold series (2026-10-15), so paired
                                 # bootstrap SEs between arms can be formed
                                 # offline: holds are in the same order for
                                 # every arm of a panel.
                                 pnl_per_hold=[float(v) for v in pb],
                                 promise_per_hold=[float(v) for v in pr]))
                if arm == "gauss":
                    preds["clim"] = cr
                _clim = preds.get("clim", cr)
                print(f"  {arm:<12}{cr:>8.3f}{crpss(cr, _clim):>8.3f}"
                      f"{resolution(qt, y[te]):>7.3f}{ks:>8.3f}"
                      f"{cov[90]:>7.2f}{float(np.mean(qt[:,7]-qt[:,1])):>7.2f}"
                      f"{sh:>8.2f}{pg:>9.1f}bp", flush=True)
            ROWS.append(dict(era=elab, panel=src, arm="floor(rand-slow)",
                             sharpe=shf, n_holds=int(nh)))
            print(f"  {'floor(rand)':<12}{'':>8}{'':>9}{'':>7}{'':>7}{'':>7}{'':>8}"
                  f"{shf:>8.2f}\n", flush=True)


if __name__ == "__main__":
    import json
    main()
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT.write_text(json.dumps(ROWS, indent=1))
    print(f"\n  artifact: {ARTIFACT.relative_to(ROOT)}  ({len(ROWS)} rows)")
