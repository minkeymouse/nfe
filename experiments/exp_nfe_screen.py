"""Triage screen: three untested levers for the NFE.

    uv run python experiments/exp_nfe_screen.py --smoke
    uv run python experiments/exp_nfe_screen.py

THIS IS A SCREEN, NOT A REGISTERED TEST.  No claim is made from it and no
gate is declared.  Its only job is to say which of three levers is worth a
properly pre-registered round.  Truth-known synthetic draws throughout;
nothing here touches real data or the book.  The nfe/ package is not
modified.

WHY THESE THREE.  What has already been ruled out is narrow and specific:
sizing overlays (five registered configurations; VALUE_2X2 showed an
unconditional-shrinkage null reproduces the whole gain), input
re-representation (run_260727 exp_nfe_level: +0.0002 against +0.139 for
the true level), and cross-sectional residualization of an
already-residualized stream (exp_nfe_xs: +0.0087 at the KRX-like 5% fuel).
None of those touched the three levers below.

A. EXIT TIMING.  sig_nfe sizes a position and Appendix B holds it for
   exactly H=20 days; there is no exit rule anywhere in the project (no
   script, no log).  The posterior over the half-life implies an optimal
   hold: tau=5d is finished in ten days and the rest is noise exposure,
   tau=200d captures almost nothing in twenty while paying full cost.
   Crucially the unconditional null that killed the sizing overlays cannot
   apply here -- a prior-only exit gives every name the same hold, which
   IS the current fixed-H design, so any gain is conditional by
   construction.  Arms: fixed H, posterior-chosen h, plug-in-chosen h, and
   a SHUFFLED-h control that matches the holding-period distribution
   exactly while destroying the conditioning.
   Objective: h* maximizes E_post[z(1-e^{-kappa h}) - c] / sqrt(h).

B. CHARACTERISTIC-CONDITIONED PRIOR.  The training prior is one (phi,
   gamma) per era shared by all 120 names.  Screen the upper bound: give
   each simulated name a type c that genuinely shifts its half-life prior,
   and ask whether a filter that sees c beats one that does not.  If it
   fails HERE, where c is informative by construction, the idea is dead;
   if it works, the open question is whether real characteristics carry
   such information -- which the honesty table makes doubtful.

C. ADAPTIVE SPAN.  LEVEL.md's finding was not that a longer window fails
   but that the optimal span depends on how fast the level drifts (1000 ->
   250 -> 60 as drift grows) and a fixed choice cannot track it.  Screen
   whether a multi-scale input (short and long together) matches the best
   fixed span at BOTH drift rates -- the precondition for a learned span
   being worth building.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn

from nfe import NFE_KMIX, mdn_nll

L, HMAX = 60, 60
COST = 0.05          # round-trip cost in units of the window sd (fill-pessimistic)


# ---------------------------------------------------------------- simulators
def sim_fwd(n, rng, L=L, hmax=HMAX, phi_vol=0.19, s_log=0.14, tau=0.05,
            hl_lo=3.0, hl_hi=400.0, c=None):
    """Window + forward path from the same switching-OU draw.

    Returns W (n,L), FWD (n,hmax) continuation of the spread, y = log2 HL.
    ``c`` optionally supplies a per-name type in [0,1] that shifts the
    half-life prior (screen B); None draws the pooled prior.
    """
    if c is None:
        hl = np.exp(rng.uniform(np.log(hl_lo), np.log(hl_hi), n))
    else:                       # type shifts the prior's centre by 2 octaves
        lo = np.log(hl_lo) + c * (np.log(hl_hi) - np.log(hl_lo)) * 0.55
        hl = np.exp(rng.uniform(lo, lo + np.log(8.0), n))
        hl = np.clip(hl, hl_lo, hl_hi)
    k = np.log(2) / hl
    m_ln = np.exp(s_log ** 2 / 2)
    sd_ln = np.sqrt((np.exp(s_log ** 2) - 1) * np.exp(s_log ** 2))
    x = np.zeros(n); lv = np.zeros(n); m = np.zeros(n)
    W = np.empty((n, L)); F = np.empty((n, hmax))
    for t in range(200 + L + hmax):
        lv = phi_vol * lv + 0.5 * np.sqrt(1 - phi_vol ** 2) * rng.standard_normal(n)
        e = (np.exp(s_log * rng.standard_normal(n)) - m_ln) / sd_ln
        m = m + tau * rng.standard_normal(n) * 0.01
        x = x + k * (m - x) + 0.01 * np.exp(lv) * e
        if 200 <= t < 200 + L:
            W[:, t - 200] = x
        elif t >= 200 + L:
            F[:, t - 200 - L] = x
    return W, F, np.log2(hl)


def featurize(W):
    Z = (W - W.mean(-1, keepdims=True)) / (W.std(-1, keepdims=True) + 1e-12)
    return np.stack([Z, np.diff(Z, axis=-1, prepend=Z[:, :1])], -1).astype(np.float32)


def feat_multi(W, spans):
    """Concatenate the standardized level+increment channels computed over
    several trailing spans (screen C)."""
    ch = []
    for s in spans:
        V = W[:, -s:]
        Z = (V - V.mean(-1, keepdims=True)) / (V.std(-1, keepdims=True) + 1e-12)
        Zp = np.pad(Z, ((0, 0), (W.shape[1] - s, 0)), mode="edge")
        ch += [Zp, np.diff(Zp, axis=-1, prepend=Zp[:, :1])]
    return np.stack(ch, -1).astype(np.float32)


# ---------------------------------------------------------------- net
def train(X, Y, seed, steps, dev, extra=0):
    torch.manual_seed(seed)
    Xt = torch.tensor(X); Yt = torch.tensor(Y, dtype=torch.float32)
    net = nn.ModuleDict({
        "gru": nn.GRU(X.shape[-1], 64, batch_first=True),
        "mdn": nn.Linear(64 + extra, 3 * NFE_KMIX)}).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    return _loop(net, opt, sch, Xt, Yt, None, steps, dev)


def train_cond(X, C, Y, seed, steps, dev):
    torch.manual_seed(seed)
    Xt = torch.tensor(X); Ct = torch.tensor(C, dtype=torch.float32)[:, None]
    Yt = torch.tensor(Y, dtype=torch.float32)
    net = nn.ModuleDict({
        "gru": nn.GRU(X.shape[-1], 64, batch_first=True),
        "mdn": nn.Linear(65, 3 * NFE_KMIX)}).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    return _loop(net, opt, sch, Xt, Yt, Ct, steps, dev)


def _loop(net, opt, sch, X, Y, C, steps, dev):
    for _ in range(steps):
        i = torch.randint(0, len(X), (512,))
        h = net["gru"](X[i].to(dev))[0][:, -1]
        if C is not None:
            h = torch.cat([h, C[i].to(dev)], -1)
        loss = mdn_nll(net["mdn"](h), Y[i].to(dev)).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    return net


def post(net, X, dev, C=None, bs=8192):
    ws, ms, ss = [], [], []
    Xt = torch.tensor(X)
    Ct = None if C is None else torch.tensor(C, dtype=torch.float32)[:, None]
    with torch.no_grad():
        for j in range(0, len(Xt), bs):
            h = net["gru"](Xt[j:j + bs].to(dev))[0][:, -1]
            if Ct is not None:
                h = torch.cat([h, Ct[j:j + bs].to(dev)], -1)
            o = net["mdn"](h).view(-1, 3, NFE_KMIX).cpu()
            ws.append(torch.softmax(o[:, 0], -1).numpy())
            ms.append(o[:, 1].numpy())
            ss.append(o[:, 2].clamp(-4, 2).exp().numpy())
    return map(np.concatenate, (ws, ms, ss))


def kappa_draws(w, mu, sd, nsamp, rng):
    k = rng.random((len(w), nsamp))
    cum = np.cumsum(w, 1)
    idx = (k[:, :, None] > cum[:, None, :]).sum(-1).clip(0, w.shape[1] - 1)
    y = np.take_along_axis(mu, idx, 1) + \
        np.take_along_axis(sd, idx, 1) * rng.standard_normal((len(w), nsamp))
    return np.log(2) / np.clip(2.0 ** y, 1e-3, 1e4)


def auc(score, lab):
    from scipy.stats import rankdata
    n1 = int(lab.sum()); n0 = len(lab) - n1
    r = rankdata(score)
    return float((r[lab == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


# ---------------------------------------------------------------- screens
def screen_A(dev, ntr, nev, steps, rng):
    """Exit timing. One filter, four exit rules on the same positions."""
    Wtr, _, ytr = sim_fwd(ntr, np.random.default_rng(1))
    net = train(featurize(Wtr), ytr, 0, steps, dev)
    W, F, y = sim_fwd(nev, np.random.default_rng(777))
    w, mu, sd = post(net, featurize(W), dev)
    ks = kappa_draws(w, mu, sd, 64, rng)                       # (n,64)

    sdw = W.std(-1) + 1e-12
    z = np.clip((W[:, -1] - W.mean(-1)) / sdw, -3, 3)
    # forward move in window-sd units, cumulative
    move = (F - W[:, -1:]) / sdw[:, None]                      # (n,HMAX)
    hs = np.arange(1, HMAX + 1)

    def obj(kap):                                              # (n,K)->(n,HMAX)
        cap = np.abs(z)[:, None] * (1.0 - np.exp(-kap[..., None] * hs)).mean(1)
        return (cap - COST) / np.sqrt(hs)

    kap_hat = np.log(2) / np.clip(2.0 ** ((w * mu).sum(1)), 1e-3, 1e4)
    h_post = hs[np.argmax(obj(ks), 1)]
    h_plug = hs[np.argmax(obj(kap_hat[:, None]), 1)]
    h_shuf = h_post[rng.permutation(len(h_post))]

    def pnl(h):
        r = move[np.arange(len(h)), h - 1]
        return -np.sign(z) * np.minimum(np.abs(z), 3.0) * r - COST * np.abs(z) / 3.0

    out = {}
    for name, h in [("fixed20", np.full(len(z), 20)), ("posterior", h_post),
                    ("plugin", h_plug), ("shuffled", h_shuf)]:
        p = pnl(h)
        mh = float(np.mean(h))
        spt = float(p.mean() / (p.std() + 1e-12))
        out[name] = {"mean_bp": float(p.mean() * 1e4),
                     "sharpe_per_trade": spt,
                     "mean_hold": mh,
                     # time-normalized: longer holds are not free
                     "sharpe_ann": spt * float(np.sqrt(252.0 / mh))}
    base = pnl(np.full(len(z), 20))
    for name, h in [("posterior", h_post), ("plugin", h_plug),
                    ("shuffled", h_shuf)]:
        d = pnl(h) - base
        out[name]["vs_fixed_bp"] = float(d.mean() * 1e4)
        out[name]["vs_fixed_se"] = float(d.std(ddof=1) / np.sqrt(len(d)) * 1e4)
    d = pnl(h_post) - pnl(h_shuf)
    out["posterior"]["vs_shuffled_bp"] = float(d.mean() * 1e4)
    out["posterior"]["vs_shuffled_se"] = float(d.std(ddof=1) / np.sqrt(len(d)) * 1e4)
    return out


def screen_B(dev, ntr, nev, steps):
    """Characteristic-conditioned prior, upper bound."""
    r1 = np.random.default_rng(2)
    ctr = r1.random(ntr)
    Wtr, _, ytr = sim_fwd(ntr, r1, c=ctr)
    r2 = np.random.default_rng(888)
    cev = r2.random(nev)
    Wev, _, yev = sim_fwd(nev, r2, c=cev)
    lab = (yev > np.log2(60.0)).astype(int)
    res = {}
    for name, C in [("pooled", None), ("conditioned", ctr)]:
        Ce = None if C is None else cev
        net = (train(featurize(Wtr), ytr, 0, steps, dev) if C is None
               else train_cond(featurize(Wtr), C, ytr, 0, steps, dev))
        w, mu, sd = post(net, featurize(Wev), dev, Ce)
        from scipy.special import ndtr
        tail = (w * (1 - ndtr((np.log2(60.0) - mu) / sd))).sum(1)
        res[name] = {"auc": auc(tail, lab)}
    res["gain"] = res["conditioned"]["auc"] - res["pooled"]["auc"]
    return res


def screen_C(dev, ntr, nev, steps):
    """Adaptive span: does multi-scale match the best fixed span at both
    drift rates?"""
    LONG = 250
    res = {}
    for dname, tau in [("stable", 0.0), ("drifting", 0.15)]:
        Wtr, _, ytr = sim_fwd(ntr, np.random.default_rng(3), L=LONG, tau=tau)
        Wev, _, yev = sim_fwd(nev, np.random.default_rng(555), L=LONG, tau=tau)
        lab = (yev > np.log2(60.0)).astype(int)
        row = {}
        for aname, spans in [("short60", (60,)), ("long250", (250,)),
                             ("multi", (60, 250))]:
            net = train(feat_multi(Wtr, spans), ytr, 0, steps, dev)
            w, mu, sd = post(net, feat_multi(Wev, spans), dev)
            from scipy.special import ndtr
            tail = (w * (1 - ndtr((np.log2(60.0) - mu) / sd))).sum(1)
            row[aname] = auc(tail, lab)
        row["best_fixed"] = max(row["short60"], row["long250"])
        row["multi_minus_bestfixed"] = row["multi"] - row["best_fixed"]
        res[dname] = row
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    ntr, nev, steps = (4_000, 3_000, 400) if args.smoke else (60_000, 40_000, 3_000)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(11)
    print(f"[screen] device={dev} ntrain={ntr} steps={steps}\n", flush=True)

    res = {}
    print("== A: exit timing ==", flush=True)
    res["A_exit"] = screen_A(dev, ntr, nev, steps, rng)
    print(json.dumps(res["A_exit"], indent=2), flush=True)

    print("\n== B: characteristic-conditioned prior ==", flush=True)
    res["B_prior"] = screen_B(dev, ntr, nev, steps)
    print(json.dumps(res["B_prior"], indent=2), flush=True)

    print("\n== C: adaptive span ==", flush=True)
    res["C_span"] = screen_C(dev, ntr, nev, steps)
    print(json.dumps(res["C_span"], indent=2), flush=True)

    out = ROOT / "results" / "run_260727"
    out.mkdir(parents=True, exist_ok=True)
    tag = "smoke" if args.smoke else "full"
    (out / f"nfe_screen_{tag}.json").write_text(json.dumps(res, indent=2))

    a = res["A_exit"]["posterior"]
    print("\n=== SCREEN READ-OUT (no gates, no claims) ===")
    for nm in ("fixed20", "posterior", "plugin", "shuffled"):
        r = res["A_exit"][nm]
        print(f"A exit  {nm:10s} hold {r['mean_hold']:5.1f}d  "
              f"SR/trade {r['sharpe_per_trade']:+.4f}  "
              f"SR annualized {r['sharpe_ann']:+.3f}")
    print(f"A exit  posterior vs fixed-20 : {a['vs_fixed_bp']:+.2f} "
          f"+- {a['vs_fixed_se']:.2f} bp/trade   (mean hold "
          f"{res['A_exit']['posterior']['mean_hold']:.1f}d)")
    print(f"A exit  posterior vs shuffled : {a['vs_shuffled_bp']:+.2f} "
          f"+- {a['vs_shuffled_se']:.2f} bp/trade  <- the attribution one")
    print(f"B prior conditioned - pooled  : {res['B_prior']['gain']:+.4f} AUC")
    for d in ("stable", "drifting"):
        c = res["C_span"][d]
        print(f"C span  {d:9s} multi {c['multi']:.4f} vs best fixed "
              f"{c['best_fixed']:.4f} ({c['multi_minus_bestfixed']:+.4f})")
    print(f"\n-> {out / f'nfe_screen_{tag}.json'}")


if __name__ == "__main__":
    main()
