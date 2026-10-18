"""Can the contemporaneous cross-section supply the level the window cannot?

    uv run python experiments/exp_nfe_xs.py --smoke
    uv run python experiments/exp_nfe_xs.py

PRE-REGISTRATION (frozen 2026-07-27, before the first full run).

QUESTION.  exp_nfe_level established that the level's identification loss
is INFORMATIONAL, not representational (+0.0002 / +0.0010 AUC for level
channels against +0.139 for the true level), so re-encoding the window
cannot help: the level is not in it.  Two sources sit outside -- the
name's own longer history, which run_260722/LEVEL.md found does not
transfer to real data, and the CONTEMPORANEOUS CROSS-SECTION, untested at
this scale.  run_260724/NFE_ANCHOR.md showed the cross-sectional
mechanism works in NLL but never measured AUC, so it cannot be placed on
the ceiling ladder.  This repeats it in AUC with an oracle arm, so the
answer is a fraction of a measured gap rather than a bare improvement.

SIMULATOR (identical parameterization to exp_nfe_anchor).  Panels of
N=64 names; each name is a switching-OU spread x, contaminated by a
shared random-walk level factor F with loadings b ~ U(0.5, 1.5), scaled
so F takes the target `share` of increment variance:  S_i = x_i + b_i F.
The label is the half-life of x_i, which F does not touch.

ARMS (identical net, budget, seeds and eval panels):
    plain   feat_plain(S)     the current NFE input: per-window demeaning
    xs      feat_anchor(S)    within-window residualization against the
                              cross-sectional mean; beta_i from an OLS of
                              own increments on mean increments, then the
                              plain features of S_i - beta_i*Abar
    oracle  feat_plain(x)     the clean panel: the shared factor removed
                              exactly.  Upper bound on what ANY use of the
                              cross-section could recover.

SHARES: 0.00 (no shared factor at all), 0.05 (the measured KRX-like fuel
on an already-residualized stream), 0.50 (strong).

HYPOTHESES AND GATES (verdicts only at these):
    G1 PRIMARY (realistic)  at share=0.05, AUC(xs) - AUC(plain) > 0 at 2 SE.
    G2 MAGNITUDE (realistic) at share=0.05, xs recovers >= 25% of the
                 AUC(oracle) - AUC(plain) gap.
    G3 SPECIFICITY  at share=0.00, |AUC(xs) - AUC(plain)| <= 1 SE.  With no
                 shared factor the cross-sectional mean is noise; if
                 subtracting it still "helps", the effect is not pooling
                 and G1/G2 are void whatever they say.
    G4 DOSE-RESPONSE  the xs advantage is monotone in share
                 (0.00 <= 0.05 <= 0.50), each step at 2 SE for the
                 0.50-vs-0.05 contrast.
    G5 CALIBRATION  PIT KS of xs not worse than plain by more than 0.02.

READ-OUT that matters regardless of the gates: the recovered fraction at
share=0.05.  The book's own residual stream carries 3-5% (PCA +0.033, AFM
+0.058); raw returns carry +0.234.  If the fraction at 0.05 is negligible
the cross-sectional route is only worth taking upstream, on raw returns,
which is a different and much larger experiment.

RETRACTION CLAUSE.  If G3 fails, no cross-sectional claim is made from
this run at any share.

SCOPE.  Truth-known panel draws only; one architecture (per-name GRU-64 +
5-MDN over the residualized window), one contamination model (a single
random-walk factor in levels).  Nothing here touches real data or the
book.  The nfe/ package is NOT modified.
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

from nfe import L, NFE_KMIX, mdn_nll, panel_simulate

THR = np.log2(60.0)
N_PANEL = 64
SHARES = {"none": 0.00, "matched": 0.05, "strong": 0.50}
ARMS = ("plain", "xs", "oracle")


def sim_xs(n_panels, rng, share):
    """S = x + b*F with F a random-walk level factor at the target share of
    increment variance.  Returns S (contaminated), X (clean), y, realized."""
    X, y = panel_simulate(n_panels, N_PANEL, rng, shared=False)
    dx = np.diff(X, axis=-1)
    sx2 = float(np.mean(dx ** 2))
    b = rng.uniform(0.5, 1.5, (n_panels, N_PANEL, 1))
    if share <= 0.0:
        S = X.astype(np.float32)
        return S, X.astype(np.float32), y, 0.0
    sF = np.sqrt(share / (1.0 - share) * sx2 / float(np.mean(b ** 2)))
    F = np.cumsum(sF * rng.standard_normal((n_panels, 1, L)), -1)
    S = (X + b * F).astype(np.float32)
    real = 1.0 - sx2 / float(np.mean(np.diff(S, axis=-1) ** 2))
    return S, X.astype(np.float32), y, real


def feat_plain(S):
    mu = S.mean(-1, keepdims=True)
    sd = S.std(-1, keepdims=True) + 1e-12
    Z = (S - mu) / sd
    dZ = np.diff(Z, axis=-1, prepend=Z[..., :1])
    return np.stack([Z, dZ], -1).astype(np.float32)


def feat_xs(S):
    """Residualize each name against the cross-sectional mean, beta from an
    OLS of own increments on mean increments (levels of two random walks
    would be spurious).  Causal and deterministic."""
    A = S.mean(1, keepdims=True)
    dS = np.diff(S, axis=-1)
    dA = np.diff(A, axis=-1)
    beta = (dS * dA).sum(-1) / ((dA * dA).sum(-1) + 1e-18)
    return feat_plain((S - beta[..., None] * A).astype(np.float32))


def train(X, Y, seed, steps, dev):
    torch.manual_seed(seed)
    X = torch.tensor(X.reshape(-1, L, X.shape[-1]))
    Y = torch.tensor(Y.reshape(-1))
    net = nn.ModuleDict({"gru": nn.GRU(X.shape[-1], 64, batch_first=True),
                         "mdn": nn.Linear(64, 3 * NFE_KMIX)}).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for _ in range(steps):
        i = torch.randint(0, len(X), (512,))
        loss = mdn_nll(net["mdn"](net["gru"](X[i].to(dev))[0][:, -1]),
                       Y[i].to(dev)).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    return net


def evaluate(net, Xe, ye, dev):
    from scipy.special import ndtr
    Xe = torch.tensor(Xe.reshape(-1, L, Xe.shape[-1]))
    yf = ye.reshape(-1)
    ws, ms, ss, nlls = [], [], [], []
    with torch.no_grad():
        for j in range(0, len(Xe), 8192):
            o = net["mdn"](net["gru"](Xe[j:j + 8192].to(dev))[0][:, -1])
            nlls.append(mdn_nll(o, torch.tensor(yf[j:j + 8192]).to(dev)).cpu().numpy())
            o = o.view(-1, 3, NFE_KMIX).cpu()
            ws.append(torch.softmax(o[:, 0], -1).numpy())
            ms.append(o[:, 1].numpy())
            ss.append(o[:, 2].clamp(-4, 2).exp().numpy())
    w, mu, sd = map(np.concatenate, (ws, ms, ss))
    tail = (w * (1.0 - ndtr((THR - mu) / sd))).sum(1)
    pit = (w * ndtr((yf[:, None] - mu) / sd)).sum(1)
    return tail, pit, np.concatenate(nlls)


def auc(score, lab):
    from scipy.stats import rankdata
    n1 = int(lab.sum()); n0 = len(lab) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(score)
    return float((r[lab == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def ks_uniform(p):
    p = np.sort(np.clip(p, 0, 1)); n = len(p)
    return float(np.max(np.abs(p - np.arange(1, n + 1) / n)))


def paired_boot(a, b, lab, nboot, rng):
    n = len(lab); out = np.empty(nboot)
    for j in range(nboot):
        i = rng.integers(0, n, n)
        out[j] = auc(a[i], lab[i]) - auc(b[i], lab[i])
    return float(out.mean()), float(out.std(ddof=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seeds", type=int, default=2)
    args = ap.parse_args()
    ntr, steps, nev, nboot = (1_500, 400, 300, 200) if args.smoke \
        else (15_000, 3_000, 2_000, 1_000)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[{'smoke' if args.smoke else 'exploration'}] device={dev} "
          f"panels={ntr} steps={steps} seeds={args.seeds}", flush=True)

    res = {}
    for cond, share in SHARES.items():
        Se, Xe, yev, real = sim_xs(nev, np.random.default_rng(999), share)
        lab = (yev.reshape(-1) > THR).astype(int)
        FE = {"plain": feat_plain(Se), "xs": feat_xs(Se),
              "oracle": feat_plain(Xe)}
        per = {a: {"tail": [], "nll": [], "ks": []} for a in ARMS}
        for sd_ in range(args.seeds):
            Str, Xtr, ytr, _ = sim_xs(ntr, np.random.default_rng(sd_ + 1), share)
            FT = {"plain": feat_plain(Str), "xs": feat_xs(Str),
                  "oracle": feat_plain(Xtr)}
            for a in ARMS:
                net = train(FT[a], ytr, sd_, steps, dev)
                tail, pit, nll = evaluate(net, FE[a], yev, dev)
                per[a]["tail"].append(tail)
                per[a]["nll"].append(float(nll.mean()))
                per[a]["ks"].append(ks_uniform(pit))
                print(f"  {cond:8s} seed={sd_} {a:6s} AUC {auc(tail, lab):.4f} "
                      f"NLL {nll.mean():.4f} KS {per[a]['ks'][-1]:.4f}", flush=True)
            del FT, Str, Xtr
        pooled = {a: np.mean(per[a]["tail"], 0) for a in ARMS}
        rng = np.random.default_rng(4242)
        res[cond] = {
            "target_share": share, "realized_share": real,
            "auc": {a: auc(pooled[a], lab) for a in ARMS},
            "nll": {a: float(np.mean(per[a]["nll"])) for a in ARMS},
            "ks": {a: float(np.mean(per[a]["ks"])) for a in ARMS},
            "xs_minus_plain": paired_boot(pooled["xs"], pooled["plain"],
                                          lab, nboot, rng),
            "oracle_minus_plain": paired_boot(pooled["oracle"], pooled["plain"],
                                              lab, nboot, rng),
        }
        print(json.dumps(res[cond], indent=2), flush=True)
        del FE

    out = ROOT / "results" / "run_260727"
    out.mkdir(parents=True, exist_ok=True)
    tag = "smoke" if args.smoke else "full"
    (out / f"nfe_xs_{tag}.json").write_text(json.dumps(res, indent=2))

    print("\n=== GATES ===")
    m, n0, s = res["matched"], res["none"], res["strong"]
    g, ge = m["xs_minus_plain"]; o, _ = m["oracle_minus_plain"]
    n_g, n_se = n0["xs_minus_plain"]
    s_g, s_se = s["xs_minus_plain"]
    print(f"G1 xs-plain @share=.05   : {g:+.4f} +- {ge:.4f}  "
          f"{'PASS' if g > 2 * ge else 'FAIL'}")
    frac = g / o if o > 0 else float("nan")
    print(f"G2 fraction of oracle gap: {frac:.2%} (oracle {o:+.4f})  "
          f"{'PASS' if frac >= 0.25 else 'FAIL'}")
    print(f"G3 specificity @share=0  : {n_g:+.4f} +- {n_se:.4f}  "
          f"{'PASS' if abs(n_g) <= n_se else 'FAIL'}")
    dd, dse = s_g - g, float(np.hypot(s_se, ge))
    print(f"G4 dose-response .50-.05 : {dd:+.4f} +- {dse:.4f}  "
          f"{'PASS' if dd > 2 * dse else 'FAIL'}  (strong {s_g:+.4f})")
    dks = m["ks"]["xs"] - m["ks"]["plain"]
    print(f"G5 PIT KS xs-plain       : {dks:+.4f}  "
          f"{'PASS' if dks <= 0.02 else 'FAIL'}")
    print(f"\n-> {out / f'nfe_xs_{tag}.json'}")


if __name__ == "__main__":
    main()
