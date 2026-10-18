"""Calibrating the half-life prior to the substrate (confirmation round).

    uv run python experiments/exp_nfe_prior.py --smoke
    uv run python experiments/exp_nfe_prior.py

PRE-REGISTRATION (frozen 2026-07-27, before this round was run).

WHAT THIS CONFIRMS.  The exploration in run_260727 (exp_nfe_eb, one seed)
found that the filter's half-life prior is the one part of the generative
model that is never fitted to the substrate.  calib_from moment-matches the
innovation law -- phi_vol and gamma -- on the train slice, but the half-life
is left at the package default, log-uniform on 3 to 400 days, while the
train slice says the panel's blocks sit near 10 days.  Decomposing the
exploration with a pooled-centre control gave, on the AFM stream:

    era 2   nfe 1.405 -> tight prior 1.478 (+0.072) -> per-name 1.488 (+0.011)
    era 1   nfe -0.284 -> tight prior -0.256 (+0.028) -> per-name -0.195 (+0.061)

so most of the gain is the pooled calibration and a smaller, era-dependent
part is per-name.  One seed proves nothing; this round re-runs it on fresh
training seeds with the levels, not the paired dSR, as the read-out.

WHY LEVELS AND NOT dSR.  These arms share a signal and differ only in
sizing, so their P&L series are nearly collinear and the difference series
has tiny variance.  In the exploration dSR(eb - shuffled) came out +1.408
for a Sharpe level difference of 0.012.  The paper's dSR convention is
sound against the random benchmark, where the correlation is low, and
misleading here.  Gates below are therefore on Sharpe LEVELS, with the
seed spread as the uncertainty.

ARMS (identical stream, costs, holds; SEEDS is the only thing resampled):
    nfe       package default prior, log-uniform 3-400d   (the paper's row)
    nfe_cal   half-life prior calibrated on the train slice, POOLED
    nfe_eb    the same, with a per-name centre by one-way shrinkage
    nfe_ebsh  per-name centres SHUFFLED across names -- attribution control

GATES (verdicts only at these; SEEDS fresh, none used in the exploration):
    G0 HARNESS  the nfe arm reproduces the paper's era-2 Sharpe 1.42 within
                0.05 on every seed, else the round is void.
    G1 PRIMARY  mean over seeds of SR(nfe_cal) - SR(nfe) > 0 on BOTH eras.
    G2 CONSISTENCY  that sign holds on every seed, both eras.
    G3 PER-NAME  mean SR(nfe_eb) - SR(nfe_cal) > 0 on both eras AND
                 SR(nfe_eb) - SR(nfe_ebsh) > 0 on both eras.  If the second
                 fails, the per-name layer is dropped and only the pooled
                 calibration is carried forward.
    G4 NO-COST   turnover of nfe_cal does not exceed nfe's by more than 20%.

REGISTERED EXPECTATION.  G1/G2 pass, G3 is marginal.  The exploration's
per-name gain was +0.011 in era 2, well inside seed noise.

SCOPE.  Real KRX AFM residual streams (cached, not refit), [confirmation]
for G1/G2 on fresh seeds.  The nfe/ package is read, not modified, by this
script; if G1 passes the calibration is promoted into the package in a
separate commit and the paper's affected numbers are re-run.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from exp_nfe_eb import (book, metrics, sig_fade, sig_nfe_eb,  # noqa: E402
                        train_slice_prior)
from nfe import L, calib_from, sig_nfe, train_nfe             # noqa: E402

SEEDS = (11, 12, 13)          # fresh: the exploration used seed 0
ANN = np.sqrt(247.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    seeds = SEEDS[:1] if args.smoke else SEEDS

    out = {}
    for era in (1, 2):
        eps = np.load(ROOT / f"results/run_260724/afm_stream_era{era}.npz")["eps"]
        T = len(eps); ntr = int(round(T * 0.30))
        cal = calib_from(eps[:ntr])
        c, s, info = train_slice_prior(eps[:ntr])
        gm = np.full_like(c, info["grand_mean"])
        csh = c[np.random.default_rng(0).permutation(len(c))]
        ev = None
        per = {a: [] for a in ("nfe", "nfe_cal", "nfe_eb", "nfe_ebsh")}
        turn = {a: [] for a in per}
        for sd_ in seeds:
            net = train_nfe(seed=sd_, **cal)
            arms = {"nfe": lambda h, _n=net: sig_nfe(h, _n),
                    "nfe_cal": lambda h, _n=net: sig_nfe_eb(h, _n, gm, s),
                    "nfe_eb": lambda h, _n=net: sig_nfe_eb(h, _n, c, s),
                    "nfe_ebsh": lambda h, _n=net: sig_nfe_eb(h, _n, csh, s)}
            for a, f in arms.items():
                p, tv = book(eps, f)
                if ev is None:
                    ev = np.arange(len(p)) >= (ntr - L)
                m = metrics(p[ev], tv[ev], ANN)
                per[a].append(m["sharpe"]); turn[a].append(m["turn"])
            print(f"  era{era} seed={sd_}  " + "  ".join(
                f"{a} {per[a][-1]:+.3f}" for a in per), flush=True)
        base = metrics(*[x[ev] for x in book(eps, sig_fade)], ANN)
        out[f"era{era}"] = {
            "base_sharpe": base["sharpe"],
            "sharpe": {a: {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1))
                           if len(v) > 1 else 0.0, "per_seed": v}
                       for a, v in per.items()},
            "turn": {a: float(np.mean(v)) for a, v in turn.items()},
            "prior_info": info,
            "grand_mean_halflife_days": float(2.0 ** info["grand_mean"]),
        }

    d = ROOT / "results" / "run_260727"
    d.mkdir(parents=True, exist_ok=True)
    tag = "smoke" if args.smoke else "full"
    (d / f"nfe_prior_{tag}.json").write_text(json.dumps(out, indent=2))

    print("\n=== GATES ===")
    ok0 = all(abs(v - 1.42) <= 0.05 for v in out["era2"]["sharpe"]["nfe"]["per_seed"])
    print(f"G0 harness (era2 nfe within 0.05 of 1.42 every seed): "
          f"{'PASS' if ok0 else 'FAIL — round void'}")
    g1 = {e: out[f"era{e}"]["sharpe"]["nfe_cal"]["mean"] -
             out[f"era{e}"]["sharpe"]["nfe"]["mean"] for e in (1, 2)}
    print(f"G1 cal - nfe  era1 {g1[1]:+.3f}  era2 {g1[2]:+.3f}  "
          f"{'PASS' if min(g1.values()) > 0 else 'FAIL'}")
    cons = all(a > b for e in (1, 2)
               for a, b in zip(out[f"era{e}"]["sharpe"]["nfe_cal"]["per_seed"],
                               out[f"era{e}"]["sharpe"]["nfe"]["per_seed"]))
    print(f"G2 consistency (every seed, both eras): {'PASS' if cons else 'FAIL'}")
    g3a = {e: out[f"era{e}"]["sharpe"]["nfe_eb"]["mean"] -
              out[f"era{e}"]["sharpe"]["nfe_cal"]["mean"] for e in (1, 2)}
    g3b = {e: out[f"era{e}"]["sharpe"]["nfe_eb"]["mean"] -
              out[f"era{e}"]["sharpe"]["nfe_ebsh"]["mean"] for e in (1, 2)}
    print(f"G3 eb - cal   era1 {g3a[1]:+.3f}  era2 {g3a[2]:+.3f} | "
          f"eb - shuffled era1 {g3b[1]:+.3f}  era2 {g3b[2]:+.3f}  "
          f"{'PASS' if min(g3a.values()) > 0 and min(g3b.values()) > 0 else 'FAIL'}")
    tg = {e: out[f"era{e}"]["turn"]["nfe_cal"] / out[f"era{e}"]["turn"]["nfe"]
          for e in (1, 2)}
    print(f"G4 turnover ratio era1 {tg[1]:.2f} era2 {tg[2]:.2f}  "
          f"{'PASS' if max(tg.values()) <= 1.20 else 'FAIL'}")
    for e in (1, 2):
        r = out[f"era{e}"]
        print(f"\nera{e}: base {r['base_sharpe']:+.3f} | " + " | ".join(
            f"{a} {r['sharpe'][a]['mean']:+.3f}+-{r['sharpe'][a]['sd']:.3f}"
            for a in ("nfe", "nfe_cal", "nfe_eb", "nfe_ebsh")))
    print(f"\n-> {d / f'nfe_prior_{tag}.json'}")


if __name__ == "__main__":
    main()
