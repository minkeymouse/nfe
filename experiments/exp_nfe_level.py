"""Is the window-mean demeaning in featurize() throwing away the level?

    uv run python experiments/exp_nfe_level.py --smoke
    uv run python experiments/exp_nfe_level.py

PRE-REGISTRATION (frozen 2026-07-27, before the first full run).

QUESTION.  The filter's input is Z = (W - mean(W)) / std(W): the level is
profiled out of the same 60 points the state is read from, and
run_260722/CEILING.md finds that exact operation to be the dominant loss
in the identification ceiling (0.884 scale-invariant -> 0.644 once m is
profiled out, -0.240 or two thirds of the whole loss; pinning m at 0 buys
0.644 -> 0.798).  The demeaning happens in PREPROCESSING, so no encoder
capacity can undo it.  This asks whether keeping the level in the input
recovers any of it.

The increments channel is invariant to the choice (diff of (W-c)/s does
not depend on c), so the arms below differ ONLY in the level channel.

ARMS (identical net, budget, seeds and eval draws):
    demean  Z = (W - mean(W)) / std(W)     the current NFE input
    anchor  Z = (W - W[0])    / std(W)     level kept, relative to window start
    both    3 channels: demean + anchor + increments   (strict superset)
    oracle  Z = (W - m_true)  / std(W)     positive control, upper bound

SETTINGS.  Level drift tau in {0.00, 0.05}; 0.05 is the package default
(a drifting level), 0.00 is a constant level.

HYPOTHESES AND GATES (verdicts only at these):
    G1 PRIMARY   at tau=0.05, AUC(anchor) - AUC(demean) > 0 at 2 SE
                 (paired bootstrap over shared eval windows, pooled over seeds).
    G2 MAGNITUDE anchor closes >= 25% of the AUC(oracle) - AUC(demean) gap
                 at tau=0.05.
    G3 NOT-A-SWAP  AUC(both) >= AUC(anchor) - 1 SE.  `both` is a strict
                 superset of `demean`, so if `both` <= `demean` the anchored
                 channel is noise and the hypothesis is rejected outright.
    G4 SPECIFICITY / DOSE-RESPONSE  the anchor advantage is LARGER at
                 tau=0.00 than at tau=0.05.  A constant level is the case
                 where location is genuinely informative; if the advantage
                 does not shrink as the level drifts, it is not level
                 recovery.
    G5 CALIBRATION  PIT KS of anchor not worse than demean by more than 0.02.

NULL.  Shuffled labels give AUC 0.50 by construction; reported as a floor.

RETRACTION CLAUSE.  If G1 passes but G4 fails, the result is reported as
an input-representation effect of unknown origin, NOT as level recovery,
and no mechanism claim is made.

SCOPE.  Truth-known draws of the paper's own prior only; single substrate,
single architecture.  Nothing here touches real data or the book.  The
nfe/ package is NOT modified -- the paper's numbers must reproduce.
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

from nfe import L, NFE_KMIX, mdn_nll

THR = np.log2(60.0)          # the paper's regime threshold: half-life > 60d
ARMS = ("demean", "anchor", "both", "oracle")
TAUS = (0.00, 0.05)


def simulate_lv(n, rng, phi_vol=0.19, s_log=0.14, tau=0.05):
    """As nfe.simulate, but also records the true level path M (n, L).

    Kept byte-compatible with nfe/prior.py:simulate in draw order so
    the windows are the same given the same rng and tau.
    """
    hl = np.exp(rng.uniform(np.log(3), np.log(400), n))
    k = np.log(2) / hl
    m_ln = np.exp(s_log ** 2 / 2)
    sd_ln = np.sqrt((np.exp(s_log ** 2) - 1) * np.exp(s_log ** 2))
    x = np.zeros(n); lv = np.zeros(n); m = np.zeros(n)
    W = np.empty((n, L)); M = np.empty((n, L))
    for t in range(200 + L):
        lv = phi_vol * lv + 0.5 * np.sqrt(1 - phi_vol ** 2) * rng.standard_normal(n)
        e = (np.exp(s_log * rng.standard_normal(n)) - m_ln) / sd_ln
        m = m + tau * rng.standard_normal(n) * 0.01
        x = x + k * (m - x) + 0.01 * np.exp(lv) * e
        if t >= 200:
            W[:, t - 200] = x
            M[:, t - 200] = m
    return W, M, np.log2(hl)


def feats(arm, W, M):
    """Level channel(s) + the (shared) increment channel."""
    s = W.std(-1, keepdims=True) + 1e-12
    zd = (W - W.mean(-1, keepdims=True)) / s          # demeaned level
    za = (W - W[:, :1]) / s                           # start-anchored level
    zo = (W - M) / s                                  # true deviation
    inc = np.diff(zd, axis=-1, prepend=zd[:, :1])     # invariant to the offset
    if arm == "demean":
        ch = [zd, inc]
    elif arm == "anchor":
        ch = [za, inc]
    elif arm == "both":
        ch = [zd, za, inc]
    elif arm == "oracle":
        ch = [zo, inc]
    else:
        raise ValueError(arm)
    return np.stack(ch, -1).astype(np.float32)


def build(in_dim, hidden=64, kmix=NFE_KMIX):
    return nn.ModuleDict({"gru": nn.GRU(in_dim, hidden, batch_first=True),
                          "mdn": nn.Linear(hidden, 3 * kmix)})


def train(X, Y, seed, steps, hidden=64):
    torch.manual_seed(seed)
    net = build(X.shape[-1], hidden)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for _ in range(steps):
        i = torch.randint(0, len(X), (512,))
        loss = mdn_nll(net["mdn"](net["gru"](X[i])[0][:, -1]), Y[i]).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    return net


def posterior(net, X, bs=4096):
    """-> (w, mu, sd) of the mixture, numpy."""
    ws, ms, ss = [], [], []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            o = net["mdn"](net["gru"](X[i:i + bs])[0][:, -1]).view(-1, 3, NFE_KMIX)
            ws.append(torch.softmax(o[:, 0], -1).numpy())
            ms.append(o[:, 1].numpy())
            ss.append(o[:, 2].clamp(-4, 2).exp().numpy())
    return np.concatenate(ws), np.concatenate(ms), np.concatenate(ss)


def _ncdf(z):
    from scipy.special import ndtr
    return ndtr(z)


def tail_and_pit(w, mu, sd, y):
    """P(y > THR) under the mixture, and the PIT of the true y."""
    tail = (w * (1.0 - _ncdf((THR - mu) / sd))).sum(1)
    pit = (w * _ncdf((y[:, None] - mu) / sd)).sum(1)
    return tail, pit


def auc(score, lab):
    """Rank AUC; ties averaged."""
    from scipy.stats import rankdata
    n1 = lab.sum(); n0 = len(lab) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(score)
    return float((r[lab == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def ks_uniform(p):
    p = np.sort(np.clip(p, 0, 1))
    n = len(p)
    e = (np.arange(1, n + 1)) / n
    return float(np.max(np.abs(p - e)))


def paired_boot(scores, lab, a, b, nboot, rng):
    """Paired bootstrap over eval windows of AUC(a) - AUC(b)."""
    n = len(lab)
    out = np.empty(nboot)
    for j in range(nboot):
        i = rng.integers(0, n, n)
        out[j] = auc(scores[a][i], lab[i]) - auc(scores[b][i], lab[i])
    return float(out.mean()), float(out.std(ddof=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    ntrain, steps, neval, nboot = (12_000, 600, 8_000, 200) if args.smoke \
        else (60_000, 4_000, 40_000, 2_000)
    seeds = range(1) if args.smoke else range(args.seeds)

    res = {}
    for tau in TAUS:
        # one shared eval set per tau, identical across arms and seeds
        Wev, Mev, yev = simulate_lv(neval, np.random.default_rng(9_000), tau=tau)
        lab = (yev > THR).astype(int)
        Yev = torch.tensor(yev, dtype=torch.float32)
        Xev = {a: torch.tensor(feats(a, Wev, Mev)) for a in ARMS}

        per = {a: {"auc": [], "nll": [], "ks": [], "tail": []} for a in ARMS}
        for sd_ in seeds:
            Wtr, Mtr, ytr = simulate_lv(ntrain, np.random.default_rng(sd_), tau=tau)
            Ytr = torch.tensor(ytr, dtype=torch.float32)
            for a in ARMS:
                net = train(torch.tensor(feats(a, Wtr, Mtr)), Ytr, sd_, steps)
                w, mu, s = posterior(net, Xev[a])
                tail, pit = tail_and_pit(w, mu, s, yev)
                with torch.no_grad():
                    nll = float(mdn_nll(
                        net["mdn"](net["gru"](Xev[a])[0][:, -1]), Yev).mean())
                per[a]["auc"].append(auc(tail, lab))
                per[a]["nll"].append(nll)
                per[a]["ks"].append(ks_uniform(pit))
                per[a]["tail"].append(tail)
                print(f"  tau={tau:.2f} seed={sd_} {a:7s} "
                      f"AUC {per[a]['auc'][-1]:.4f}  NLL {nll:.4f}  "
                      f"KS {per[a]['ks'][-1]:.4f}", flush=True)

        # pool the seeds by averaging the scores, then paired-bootstrap once
        pooled = {a: np.mean(per[a]["tail"], 0) for a in ARMS}
        rng = np.random.default_rng(4_242)
        gaps = {f"{a}-demean": paired_boot(pooled, lab, a, "demean", nboot, rng)
                for a in ARMS if a != "demean"}
        gaps["both-anchor"] = paired_boot(pooled, lab, "both", "anchor", nboot, rng)
        lab_sh = lab.copy(); np.random.default_rng(7).shuffle(lab_sh)

        res[f"tau={tau:.2f}"] = {
            "auc": {a: float(np.mean(per[a]["auc"])) for a in ARMS},
            "auc_pooled": {a: auc(pooled[a], lab) for a in ARMS},
            "nll": {a: float(np.mean(per[a]["nll"])) for a in ARMS},
            "ks": {a: float(np.mean(per[a]["ks"])) for a in ARMS},
            "gaps": {k: {"mean": v[0], "se": v[1]} for k, v in gaps.items()},
            "null_shuffled_auc": auc(pooled["demean"], lab_sh),
        }
        print(json.dumps(res[f"tau={tau:.2f}"], indent=2), flush=True)

    out = ROOT / "results" / "run_260727"
    out.mkdir(parents=True, exist_ok=True)
    tag = "smoke" if args.smoke else "full"
    (out / f"nfe_level_{tag}.json").write_text(json.dumps(res, indent=2))

    # ---- verdicts at the pre-registered gates ----
    r5, r0 = res["tau=0.05"], res["tau=0.00"]
    g = r5["gaps"]
    a_gap, a_se = g["anchor-demean"]["mean"], g["anchor-demean"]["se"]
    o_gap = g["oracle-demean"]["mean"]
    b_gap = g["both-demean"]["mean"]
    ba = g["both-anchor"]["mean"]
    a0 = r0["gaps"]["anchor-demean"]["mean"]
    print("\n=== GATES ===")
    print(f"G1 anchor-demean @tau=.05 : {a_gap:+.4f} +- {a_se:.4f}  "
          f"{'PASS' if a_gap > 2 * a_se else 'FAIL'}")
    frac = a_gap / o_gap if o_gap > 0 else float('nan')
    print(f"G2 fraction of oracle gap : {frac:.2%} (oracle {o_gap:+.4f})  "
          f"{'PASS' if frac >= 0.25 else 'FAIL'}")
    print(f"G3 both >= anchor - 1 SE  : both-anchor {ba:+.4f}  "
          f"{'PASS' if ba >= -a_se else 'FAIL'}  (both-demean {b_gap:+.4f})")
    print(f"G4 dose-response tau0>tau05: {a0:+.4f} vs {a_gap:+.4f}  "
          f"{'PASS' if a0 > a_gap else 'FAIL'}")
    dks = r5["ks"]["anchor"] - r5["ks"]["demean"]
    print(f"G5 PIT KS anchor-demean   : {dks:+.4f}  "
          f"{'PASS' if dks <= 0.02 else 'FAIL'}")
    print(f"null (shuffled labels)    : {r5['null_shuffled_auc']:.4f}")
    print(f"\n-> {out / f'nfe_level_{tag}.json'}")


if __name__ == "__main__":
    main()
