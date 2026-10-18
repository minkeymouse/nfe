"""E3 — the book: which residual construction produces a tradeable stat-arb book?

    uv run python experiments/exp_e3_book.py [--smoke] [--nfe] [--afm]

[exploration] (registered 2026-07-22; docstring brought current 2026-07-23,
the original text having described the KF-49 substrate, since retired.
The hypotheses/gates below are the ones actually run, on KRX; history in
results/run_260722/E3_BOOK.md.)

SUBSTRATE. KRX daily, 120 single names (top-200 by median traded value,
unbroken 2010-2026, 4,072 days), split into two eras at 2018-04. Both
universe filters look forward (survivorship, disclosed in the paper), so
paired gaps to the turnover-matched random benchmark are the readout.
Admission (stage 0): PCA-3 residuals admitted (+0.047±0.016 vs null
+0.001±0.009); market-adjusted rejected -> negative-control arm.

ARMS (code name -> paper name; the npz/log keys keep the code names):
    PCA          rolling PCA-3 residuals + base rule        -> "PCA / base"
    AL           same residuals + AL threshold rule         -> "PCA / AL"
    PCA+NFE      same residuals + NFE overlay on base       -> "PCA / base+NFE"
    AFM+fade     AFM residual stream + base rule            -> "AFM / base"
    AFM+NFE      AFM residual stream + NFE overlay          -> "AFM / base+NFE"
    AFM+LongConv AFM residuals + its own learned filter     -> "AFM / LongConv"
    demean       market-adjusted residuals + base (control)
    random-slow  turnover-matched random benchmark (held H days)

POLICY TERMS. base = fade: w ∝ -clip(d, ±3) on the standardized deviation d
(paper renamed fade->base 2026-07-23; function names below keep sig_fade).
NFE is an OVERLAY on base, not a separate policy: same signal, re-sized by
the amortized posterior (variance term = parameter-uncertainty penalty).

HYPOTHESES (as run). H1: best arm beats the random benchmark by >2 paired
SE in the admitted era (era2). H4: in era1 (admission fails) nothing does —
the negative-control era. Verdicts only at these gates; if H1 fails, no
ordering is claimed.

FROZEN PROTOCOL. Per era: train = first 30% (AFM fit + AL entry tuning +
NFE calibration), evaluate = last 70%. L=60, H=20 non-overlapping holds,
K=3, PCA window 252d, costs 5bp/unit turnover + 1bp short, dollar-neutral
unit-gross weights, annualization from the substrate calendar.

OUTPUT. Per-era table incl. annualized mu%/sd% + eval-period daily P&L dump
to results/run_260728/book_pnl_{era}.npz (consumed by fig_book_nfe.py).

SCOPE. [exploration], one seed per arm; only paired gaps read.

AMENDMENTS (dated)
  2026-07-23a  calib_from trims the rolling-PCA burn-in and guards
               degenerate windows (was collapsing to phi_vol=0/s_log=0.02);
               era2 train now 0.284/0.12, in the gate family.
  2026-07-23b  +6 point-in-time DART fundamentals offered to every arm
               (chars_with_fund, 8->14 characteristics); returns-only
               constructions consume none by definition.
  2026-07-23c  AFM's ingested characteristic set chosen by its own
               validation on the train slice (select_fund_set); era2
               declines the fundamentals -> returns-only configuration.
  2026-07-24   TABLE-AS-LINEAGE (user): one arm per generation. FF3 =
               characteristic residuals from a daily cross-sectional
               regression on [1, size, bp] (size NOT offered to AFM),
               base rule. Table 1's NFE = AFM's learned factors under
               posterior sizing (rows["AFM+NFE"], --afm --nfe). --panel
               adds two reported side results: NFE-selfrow (a self-built
               row step, -0.03 at the floor) and the 250-day window
               audit, whose fairness controls retract the 250d gap as a
               window effect. Five registered iteration rounds and their
               verdicts are in results/run_260724/E3_LINEAGE.md.
  2026-07-28   COLUMN-WISE PRIOR (provenance re-run). The AFM+NFE arm
               adopts the column-wise half-life prior (nfe.column_prior:
               one per name, shrunk toward the panel, fitted on its own
               train slice) -- the design confirmed at the pre-registered
               gates of exp_nfe_prior.py and made the package default.
               PCA+NFE keeps the shared prior; that confirmation covered
               the AFM stream only. P&L dumps move to run_260728 so the
               run_260723 ledger files stay intact.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT))

# the NFE architecture (simulator, calibration, training, posterior sizing)
# lives in its own module; the book is one consumer of it.
from nfe import L, H, calib_from, column_prior, sig_nfe, train_nfe   # noqa: E402

SMOKE = "--smoke" in sys.argv
NFE = "--nfe" in sys.argv
AFMX = "--afm" in sys.argv
PANEL = "--panel" in sys.argv   # add the self-built row-step arm and the
                                # 250-day window audit (exp_nfe_panel)
K, PCAWIN = 3, 252
C_TURN, C_SHORT = 5e-4, 1e-4
ENTRY = 1.25
NBOOT = 2000


# ------------------------------------------------------------------ data
SUBSTRATE = "krx"      # single substrate (decision 2026-07-22): KF-49
                       # and Binance both dropped; KRX is the only panel the
                       # admission test admits, and the negative control moves
                       # from a rejected SUBSTRATE to a rejected CONSTRUCTION.


def load_panel(name=None):
    """KRX daily log returns for the fixed 120-name panel. (The FF3 arm
    builds its own market/size/value factors point-in-time from the
    fundamentals in r_char3, so no external factor file is needed.)"""
    from exp_cross_sectional import load_krx, fixed_panel
    name = name or SUBSTRATE
    raw, n = load_krx(), 120
    P = fixed_panel(raw, n)
    R = np.diff(np.log(P.to_numpy(float)), axis=0)
    return R, None, P.index[1:]


def r_demean(R, RF):
    return R - R.mean(1, keepdims=True)


def r_ff3(R, RF, win=PCAWIN):
    """Causal: betas from the trailing window, applied to day t."""
    out = np.full_like(R, np.nan)
    for t in range(win, len(R)):
        Z = np.column_stack([np.ones(win), RF[t - win:t]])
        B = np.linalg.lstsq(Z, R[t - win:t], rcond=None)[0]
        out[t] = R[t] - np.r_[1.0, RF[t]] @ B
    return out


def r_char3(R, Fe):
    """FF3-style characteristic residuals (2026-07-24 arm): daily
    cross-sectional regression of returns on [1, size, bp] — the
    market/size/value generation, characteristics fixed in advance and
    point-in-time. Names lacking characteristics that day carry the
    demeaned residual."""
    out = np.empty_like(R)
    for t in range(len(R)):
        e = R[t] - R[t].mean()
        sz, bp = Fe[t, :, 6], Fe[t, :, 4]
        live = np.isfinite(sz) & np.isfinite(bp)
        if live.sum() > 10:
            Z = np.column_stack([np.ones(live.sum()),
                                 (sz[live] - sz[live].mean()) /
                                 (sz[live].std() + 1e-12),
                                 (bp[live] - bp[live].mean()) /
                                 (bp[live].std() + 1e-12)])
            b = np.linalg.lstsq(Z, R[t][live], rcond=None)[0]
            e[live] = R[t][live] - Z @ b
        out[t] = e
    return out


def r_pca(R, RF, win=PCAWIN, k=K):
    """Causal rolling PCA residuals.

    Names with EXACTLY zero variance inside the window (a halted or
    limit-locked single name; 5 such windows in the KRX panel) make
    np.corrcoef produce NaN and lstsq fail. Industry portfolios never freeze,
    so KF-49 never triggered this. Degenerate names are excluded from the
    factor estimation for that window and carry a demeaned residual."""
    out = np.full_like(R, np.nan)
    for t in range(win, len(R)):
        Hh = R[t - win:t]
        sd = Hh.std(0)
        live = sd > 1e-12
        if live.sum() <= k + 1:
            out[t] = R[t] - R[t].mean()
            continue
        Hl, sdl = Hh[:, live], sd[live]
        mu = Hl.mean(0)
        V = np.linalg.eigh(np.corrcoef(Hl.T))[1][:, ::-1][:, :k]
        Fh = ((Hl - mu) / sdl) @ V
        B = np.linalg.lstsq(Fh, Hl - mu, rcond=None)[0]
        e = R[t] - R[t].mean()
        e[live] = R[t][live] - mu - (((R[t][live] - mu) / sdl) @ V) @ B
        out[t] = e
    return out


def load_fundamentals(dates):
    """(T, N, 6) point-in-time fundamental panel (prep_fundamentals.py),
    aligned to load_panel's dates/name order, or None if not built yet.
    Coverage starts 2016-03 (DART's own floor): era 2 is fully covered,
    era 1 only from 2016 -- earlier days rank to neutral 0 downstream."""
    p = ROOT / "data" / "processed" / "fundamentals_daily.npz"
    if not p.exists():
        return None
    z = np.load(p, allow_pickle=True)
    fd = pd.to_datetime(pd.Index(z["dates"].astype(str)))
    assert len(fd) == len(dates) and fd[0] == dates[0] and fd[-1] == dates[-1]
    return z["F"]


def chars_with_fund(Re, Fe):
    """AFM characteristics: 8 return-based + 6 fundamental columns, all
    rank-normalized per day. A fundamental column on a day where fewer than a
    quarter of names have a disclosed value ranks to 0 (neutral) -- ranking a
    median-filled near-constant row would otherwise hand the model an
    index-order artifact."""
    def _xrank(a):        # inlined from verify_afm_krx (whose import needs the
        order = a.argsort(axis=1).argsort(axis=1).astype(float)   # dfm_python package)
        return order / max(a.shape[1] - 1, 1) - 0.5
    Xr = build_characteristics(Re)
    if Fe is None:
        return Xr
    cols = []
    for fi in range(Fe.shape[2]):
        f = Fe[:, :, fi]
        fin = np.isfinite(f)
        med = np.zeros((f.shape[0], 1))
        anyf = fin.any(1)
        med[anyf, 0] = np.nanmedian(np.where(fin, f, np.nan)[anyf], axis=1)
        rk = _xrank(np.where(fin, f, med))
        rk[fin.sum(1) < f.shape[1] // 4] = 0.0
        cols.append(rk)
    return np.concatenate([Xr, np.stack(cols, 2)], axis=2)


# returns-only constructions run in the era loop; FF3 (r_char3) and AFM
# need the fundamentals / the dfm-python package and are added separately.
RESIDUALS = {"demean": r_demean, "PCA": r_pca, "AL": r_pca}
try:                                     # AFM needs torch + the dfm_python package
    from dfm_python import AFM, AFMDataset
    from dfm_python.models.afm.factors import residuals as afm_residuals
    from verify_afm_krx import build_characteristics
    import torch
    HAVE_AFM = True
except Exception as _e:                  # noqa: BLE001
    HAVE_AFM = False
    print(f"  [AFM unavailable: {type(_e).__name__} — needs dfm_python (PyPI: dfm-python)]")
# plug point for DFM / DDFM / iVDFM: add `name: fn(R, RF) -> (T, N) residual returns`


# ------------------------------------------------------------------ policies
def sig_fade(hist, rng=None):
    """OU s-score fade on the cumulative residual: the frozen common policy."""
    X = np.cumsum(hist, axis=-1)
    s = (X[:, -1] - X.mean(-1)) / (X.std(-1) + 1e-12)
    return -np.clip(s, -3, 3)


def sig_al(hist, rng=None, entry=ENTRY):
    """Avellaneda-Lee: unit fade only when the s-score clears the entry band."""
    X = np.cumsum(hist, axis=-1)
    s = (X[:, -1] - X.mean(-1)) / (X.std(-1) + 1e-12)
    return -np.sign(s) * (np.abs(s) >= entry)


def tune_entry(E, Ts, grid=(0.5, 0.75, 1.0, 1.25, 1.5, 2.0)):
    """AL carries one free parameter and every other arm carries none, so it is
    selected on the TRAIN SLICE -- the same courtesy AFM's hyper-parameters get.
    Fixing it at the published default while tuning nothing else is not neutral."""
    best, bs = grid[0], -9e9
    for e in grid:
        p, _ = book(E[:Ts], lambda h, _e=e: sig_al(h, entry=_e))
        sh = p.mean() / (p.std(ddof=1) + 1e-15) if p.std(ddof=1) > 0 else -9e9
        if sh > bs:
            best, bs = e, sh
    return best


POLICY = {"demean": sig_fade, "PCA": sig_fade, "AL": sig_al}


# ------------------------------------------------------------------ the book
def book(E, policy, rng=None, floor_hold=None):
    """E (T, N) residual returns -> daily net P&L and turnover."""
    T, N = E.shape
    pnl, tov = [], []
    prev = np.zeros(N)
    held = None
    w = np.zeros(N)
    for t in range(L, T):
        if (t - L) % H == 0:                     # rebalance ONLY at hold starts
            hist = E[t - L:t].T
            if not np.isfinite(hist).all():
                pnl.append(0.0); tov.append(0.0); continue
            w = rng.standard_normal(N) if floor_hold is not None else policy(hist)
            w = w - w.mean()
            n = np.abs(w).sum()
            w = w / n if n > 1e-12 else w
        p = w @ E[t]                       # traded on the residual stream
        turn = np.abs(w - prev).sum()
        pnl.append(p - C_TURN * turn - C_SHORT * np.maximum(-w, 0).sum())
        tov.append(turn)
        prev = w
    return np.array(pnl), np.array(tov)


def metrics(p, tov, ann):
    if p.std(ddof=1) < 1e-15:
        return dict(sharpe=0.0, sortino=0.0, mdd=0.0, calmar=0.0,
                    cvar5=0.0, turn=float(tov.mean()))
    dn = p[p < 0].std(ddof=1) if (p < 0).sum() > 2 else p.std(ddof=1)
    # compounded equity: an arithmetic cumsum gives "drawdowns" above 100% and a
    # Calmar that is not the standard one (found in the 2026-07-22 audit).
    eq = np.cumprod(1.0 + p)
    mdd = float(-(eq / np.maximum.accumulate(eq) - 1.0).min())
    yrs = len(p) / (ann ** 2)
    a = float(eq[-1] ** (1.0 / max(yrs, 1e-9)) - 1.0) if eq[-1] > 0 else -1.0
    return dict(sharpe=float(p.mean() / p.std(ddof=1) * ann),
                sortino=float(p.mean() / (dn + 1e-15) * ann),
                mdd=mdd, calmar=float(a / (mdd + 1e-12)),
                cvar5=float(np.mean(np.sort(p)[:max(1, len(p) // 20)])),
                turn=float(tov.mean()),
                mu=float(p.mean() * ann ** 2 * 100),      # annualized mean %
                sd=float(p.std(ddof=1) * ann * 100))      # annualized vol %


def afm_stream(Re, Ts, epochs, K_afm=5, lv=1.0, Fe=None):
    """AFM's RESIDUALS, forced into the E3 convention.

    AFM ships with its own protocol -- daily rebalancing, a 30-day look-back,
    its own factor-neutralised weights. Running it that way makes its numbers
    incomparable with every other arm (its earlier run turned over ~0.34/day
    against 0.23-0.39 per 20-day hold here). So we take only what AFM is
    actually good at -- the learned conditional factors, hence the residual
    stream -- and put it through the SAME book as everything else: L=60 window,
    H=20 holds, same costs, same floor. Hyper-parameters are the ones its own
    validation protocol selected on this substrate (K=5, lambda_Var=1.0), with
    the look-back forced to L.

    Returns (residual returns (T,N), the fitted model).
    """
    C = chars_with_fund(Re, Fe)
    m = AFM(dataset=AFMDataset(returns=Re[:Ts], characteristics=C[:Ts]),
            n_factors=K_afm, hist_len=L, embed_dim=32, n_kernels=32,
            factor_model="attention", trading="longconv", lambda_var=lv,
            turnover_cost=C_TURN, short_cost=C_SHORT, learning_rate=3e-3,
            max_epochs=epochs, patience=15, seed=0)
    m.fit()
    with torch.no_grad():
        ret, ch = AFMDataset(returns=Re, characteristics=C).get_tensors(m.device)
        ret, ch = ret.unsqueeze(0), ch.unsqueeze(0)
        of, bt = m._factor_weights(ret, ch)
        eps = afm_residuals(of, bt, ret)[0].cpu().numpy()
    return eps, m


def afm_stream_rolling(Re, epochs, step=246, Fe=None):
    """Refit AFM once per year, as its own protocol does. Fitting once on a
    train slice and never again leaves its factors to go stale over 8 years
    while the PCA benchmark is re-estimated on a trailing 252 days -- which is
    unfair to AFM, not to us."""
    T = len(Re)
    eps = np.full_like(Re, np.nan)
    last = None
    for a in range(step * 2, T, step):           # need >=2y before the first fit
        e, m = afm_stream(Re[:a], a, epochs,
                          Fe=None if Fe is None else Fe[:a])
        eps[a:min(a + step, T)] = e[a - min(a, len(e)):][:min(step, T - a)]             if len(e) >= a else np.nan
        last = m
    ok = np.isfinite(eps).all(1)
    if not ok.any():
        return afm_stream(Re, int(len(Re) * 0.3), epochs, Fe=Fe)
    return eps, last


def select_fund_set(Re, Ts, Fe, epochs):
    """Which fundamentals does AFM ingest? Chosen by ITS OWN validation on
    the train slice (amendment 2026-07-23c). Fit on the first 70% of the
    train slice, score the fade book on the remaining 30%. On eras where
    coverage is thin the neutral-filled columns cannot win and selection
    falls back to returns-only by itself."""
    menu = {"none": [], "value(bp,ep)": [4, 5], "quality(roe,de)": [0, 1],
            "all6": [0, 1, 2, 3, 4, 5]}
    fit = int(Ts * 0.7)
    best, bs, blab = [], -9e9, "none"
    for lab, idx in menu.items():
        Fs = None if not idx else Fe[:Ts][:, :, idx]
        eps, _ = afm_stream(Re[:Ts], fit, epochs, Fe=Fs)
        p, _ = book(eps, sig_fade)
        pv = p[max(fit - L, 0):]
        sh = float(pv.mean() / (pv.std(ddof=1) + 1e-15))
        print(f"    char-set {lab:<16} val Sharpe {sh*np.sqrt(246):+.2f}",
              flush=True)
        if sh > bs:
            best, bs, blab = idx, sh, lab
    print(f"  AFM characteristic set selected on train: {blab}", flush=True)
    return best


def sig_longconv(hist, model):
    """AFM's own learned filter, evaluated at the E3 rebalance times."""
    with torch.no_grad():
        h = torch.tensor(hist, dtype=torch.float32, device=model.device)
        return model.filter(h).cpu().numpy()


def main():
    R, RF, dates = load_panel()
    FUND = load_fundamentals(dates)
    yrs = (dates[-1] - dates[0]).days / 365.25
    ann = np.sqrt(len(R) / yrs)
    mid = len(R) // 2          # KF-49's fixed 2008 split is gone with KF-49;
                               # each substrate is halved into two eras instead
    print(f"[{'smoke' if SMOKE else 'exploration'}] E3 book — {SUBSTRATE} daily")
    print(f"  {len(R)} days x {R.shape[1]} industries "
          f"{dates[0].date()}..{dates[-1].date()} | ann sqrt({len(R)/yrs:.0f})")
    print(f"  eras: {str(dates[0])[:7]}..{str(dates[mid-1])[:7]} (n={mid})  |  "
          f"{str(dates[mid])[:7]}..{str(dates[-1])[:7]} (n={len(R)-mid})")
    print(f"  L={L} H={H} K={K} pcawin={PCAWIN} entry={ENTRY} "
          f"costs {C_TURN}/{C_SHORT}")
    print(f"  fundamentals: {'attached (M=14 characteristics)' if FUND is not None else 'NOT on disk (M=8, returns only)'}\n",
          flush=True)

    e1, e2 = str(dates[0])[:7], str(dates[mid])[:7]
    eras = [(f"era 1 ({e1}..{str(dates[mid-1])[:7]})", slice(0, mid)),
            (f"era 2 ({e2}..{str(dates[-1])[:7]})", slice(mid, None))]
    if SMOKE:
        eras = [(eras[1][0], slice(mid, mid + 900))]

    for elab, sl in eras:
        Re, RFe = R[sl], (None if RF is None else RF[sl])
        Fe = None if FUND is None else FUND[sl]
        Ts = int(len(Re) * 0.3)                  # train slice (AFM only); all arms
        rows = {}                                # are still scored on the tail
        for name, rfn in RESIDUALS.items():
            E = rfn(Re, RFe)
            if name == "AL":
                ent = tune_entry(E, Ts)
                print(f"  AL entry tuned on the train slice: {ent}", flush=True)
                p, tv = book(E, lambda h, _e=ent: sig_al(h, entry=_e))
            else:
                p, tv = book(E, POLICY[name])
            rows[name] = (p, tv)
        net = None
        cal = None
        if NFE or PANEL:
            cal = calib_from(r_pca(Re[:Ts], None))     # TRAIN SLICE only
        if NFE or PANEL:                         # train FIRST: the arms reuse it
            net = train_nfe(**cal)
            print(f"  NFE trained on the admitted DGP "
                  f"(phi_vol={cal['phi_vol']:.2f}, s_log={cal['s_log']:.2f}, "
                  f"calibrated causally)", flush=True)
        if NFE:
            rows["PCA+NFE"] = book(r_pca(Re, RFe),
                                   lambda h, _n=net: sig_nfe(h, _n))
        if Fe is not None:
            rows["FF3"] = book(r_char3(Re, Fe), sig_fade)
        if PANEL and Fe is not None:
            # the self-built row step (reaches -0.03, at the floor): a
            # reconstruction factor model on the six slow direction-blind
            # fundamentals, then the simulator posterior sizes its stream.
            from exp_nfe_panel import nfe_stream_rolling, rank_chars
            ch = rank_chars(Fe[:, :, :6])
            ch_lag = np.concatenate([ch[:1], ch[:-1]], axis=0)
            eps_sr = nfe_stream_rolling(Re, ch_lag)
            print("  self-built row-step stream extracted "
                  "(reconstruction, annual refits)", flush=True)
            rows["NFE-selfrow"] = book(eps_sr,
                                       lambda h, _n=net: sig_nfe(h, _n))
        if AFMX and HAVE_AFM:
            etag = "era1" if sl.start in (0, None) else "era2"
            cachep = ROOT / "results" / "run_260724" / \
                f"afm_stream_{etag}.npz"
            mdl = None
            if cachep.exists() and not SMOKE:
                eps = np.load(cachep)["eps"]
                print("  AFM residual stream loaded from cache "
                      "(LongConv row skipped; number recorded)",
                      flush=True)
            else:
                ep_afm = 8 if SMOKE else 40
                fsel = select_fund_set(Re, Ts, Fe, ep_afm) \
                    if Fe is not None else []
                Fs = None if not fsel else Fe[:, :, fsel]
                eps, mdl = afm_stream_rolling(Re, ep_afm, Fe=Fs)
                print("  AFM residual stream extracted (E3 convention "
                      "forced)", flush=True)
                if not SMOKE:
                    cachep.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(cachep, eps=eps)
            if mdl is not None:
                rows["AFM+LongConv"] = book(
                    eps, lambda h, _m=mdl: sig_longconv(h, _m))
            rows["AFM+fade"] = book(eps, sig_fade)
            if net is not None:
                # the NFE of Table 1: AFM's compression + the posterior,
                # under the column-wise half-life prior (2026-07-28
                # amendment; confirmed in exp_nfe_prior, package default).
                cp = column_prior(eps[:Ts])
                rows["AFM+NFE"] = book(
                    eps, lambda h, _n=net, _p=cp: sig_nfe(h, _n, prior=_p))
                if PANEL:                      # the 250-day window audit
                    from exp_nfe_panel import (book_states, states_audit,
                                               states_nfe250, train_nfe_L)
                    seed250 = 1 if "--audit-confirm" in sys.argv else 0
                    net250 = train_nfe_L(250, cal["phi_vol"], cal["s_log"],
                                         seed=seed250)
                    print("  250d posterior trained", flush=True)
                    rows["NFE250"] = book_states(
                        eps, states_nfe250(eps, net, net250),
                        C_TURN, C_SHORT)
                    rows["base250"] = book_states(          # fairness control
                        eps, states_audit(eps, "base250"), C_TURN, C_SHORT)
                    rows["plug250"] = book_states(          # fairness control
                        eps, states_audit(eps, "plug250"), C_TURN, C_SHORT)
        rng = np.random.default_rng(0)
        Ef = r_pca(Re, RFe)
        rows["random-slow"] = book(Ef, None, rng=rng, floor_hold=H)

        ev = slice(max(Ts - L, 0), None)         # evaluation days
        base = rows["random-slow"][0][ev]
        bi = np.random.default_rng(1).integers(0, len(base), (NBOOT, len(base)))
        print(f"  === {elab} — evaluated on {len(base)} days ===")
        print(f"  {'arm':<13}{'Sharpe':>8}{'mu%':>7}{'sd%':>7}{'Sortino':>9}"
              f"{'MDD':>8}{'Calmar':>8}{'CVaR5':>9}{'turn':>7}{'vs floor':>16}")
        res = {}
        for name in list(rows):
            p, tv = rows[name]
            m = metrics(p[ev], tv[ev], ann)
            d = p[ev] - base
            gap = d.mean() / (d.std(ddof=1) + 1e-15) * ann
            se = np.array([d[i].mean() / (d[i].std(ddof=1) + 1e-15)
                           for i in bi]).std(ddof=1) * ann
            res[name] = (m, gap, se)
            tag = "" if name == "random-slow" else f"{gap:>8.2f} +-{se:.2f}"
            print(f"  {name:<13}{m['sharpe']:>8.2f}{m['mu']:>7.2f}{m['sd']:>7.2f}"
                  f"{m['sortino']:>9.2f}"
                  f"{m['mdd']*100:>7.1f}%{m['calmar']:>8.2f}{m['cvar5']*1e4:>8.1f}bp"
                  f"{m['turn']:>7.3f}{tag:>16}")
        real = [n for n in rows if n != "random-slow"]
        best = max(real, key=lambda n: res[n][0]["sharpe"])
        g, se = res[best][1], res[best][2]
        v = "PASS" if g > 2 * se else ("FAIL" if g < -2 * se else "NULL")
        print(f"  -> best arm `{best}` vs floor: {g:+.2f} +-{se:.2f}  [{v}]")
        if v != "PASS":
            print("     G: no arm clears the turnover-matched floor -> "
                  "NO ordering among arms is claimed for this era.")
        print(flush=True)
        # dump evaluation-period daily P&L per arm for the paper figure
        # (2026-07-28: moved from run_260723 so that ledger stays intact)
        out = ROOT / "results" / "run_260728"
        out.mkdir(parents=True, exist_ok=True)
        dts = dates[sl][L:][ev]
        era_tag = "era1" if sl.start in (0, None) else "era2"
        np.savez(out / f"book_pnl_{era_tag}.npz",
                 dates=np.array([str(d)[:10] for d in dts]),
                 **{name: rows[name][0][ev] for name in rows})
if __name__ == "__main__":
    main()
