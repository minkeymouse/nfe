"""Verification pass: Figure 3's reliability curve across training seeds.

    uv run python experiments/verify_fig3_seeds.py

(2026-10-12, camera-ready check: calibration across training seeds, not
only the single seed of the figure.) Trains the NFE
at package defaults (NFE_NTRAIN x NFE_STEPS) for seeds 0, 1, 2 -- seed 0 is
the figure's -- and scores the PIT of the true log2 half-life on the same
20,000 fresh draws fig_calibration.py uses (rng 202), for the posterior and
for the half-width point-estimate stand-in. Reports the maximum
|empirical - nominal| coverage over the unit grid and the coverage at
0.50/0.80/0.90/0.95. Simulator draws only; nothing touches real data.
CPU-only; ~7 min per seed on 4 cores.

Registered reading (before the run): the posterior's max deviation stays
within 0.03 on every seed, the half-width stand-in exceeds 0.05 on every
seed. Measured 2026-10-12 (cloud, CPU, torch 2.12): posterior 0.022 / 0.009
/ 0.010, half-width 0.115 / 0.082 / 0.085 -- both hold; see RUN_LOG.md.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results" / "run_261012"
SEEDS = (0, 1, 2)


def main():
    import torch
    from scipy.stats import norm
    from nfe import featurize as _feat, simulate as _sim_train, train_nfe, NFE_KMIX
    OUT.mkdir(parents=True, exist_ok=True)
    grid = np.linspace(0, 1, 101)
    lines, curves = [], {}
    for seed in SEEDS:
        t0 = time.time()
        torch.manual_seed(seed)
        net = train_nfe(seed=seed, phi_vol=0.19, s_log=0.14)
        rng = np.random.default_rng(202)
        W, y = _sim_train(20_000, rng, phi_vol=0.19, s_log=0.14)
        with torch.no_grad():
            o = net["mdn"](net["gru"](torch.tensor(_feat(W)))[0][:, -1])
        o = o.view(-1, 3, NFE_KMIX)
        pi = torch.softmax(o[:, 0], -1).numpy()
        mu = o[:, 1].numpy()
        sd = o[:, 2].clamp(-4, 2).exp().numpy()
        row = {}
        for scale, lab in [(1.0, "posterior"), (0.5, "half-width")]:
            u = np.sort((pi * norm.cdf((y[:, None] - mu) / (sd * scale))).sum(1))
            emp = np.searchsorted(u, grid, side="right") / len(u)
            row[lab] = emp
            curves[f"{lab}_{seed}"] = emp
        dev = {k: float(np.abs(v - grid).max()) for k, v in row.items()}
        msg = (f"seed {seed}: posterior max|dev| {dev['posterior']:.4f} | "
               f"half-width max|dev| {dev['half-width']:.4f} | {time.time()-t0:.0f}s")
        print(msg, flush=True); lines.append(msg)
        for q in (0.5, 0.8, 0.9, 0.95):
            i = int(round(q * 100))
            m = (f"   nominal {q:.2f}: posterior {row['posterior'][i]:.3f}  "
                 f"half-width {row['half-width'][i]:.3f}")
            print(m, flush=True); lines.append(m)
    np.savez(OUT / "fig3_seedcheck.npz", grid=grid, **curves)
    (OUT / "FIG3_SEEDS.md").write_text(
        "# Figure 3 reliability across training seeds\n\n" + "\n".join(lines) + "\n")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
