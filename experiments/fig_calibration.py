"""Paper figure: the posterior is calibrated (the title claim).

    uv run python experiments/fig_calibration.py

(2026-07-24). Replaces the identification-ladder dot plot as Fig 3.
Reliability on truth-known draws of the admitted process: the
probability-integral transform (PIT) of the true log2 half-life under
each predictive, plotted as its empirical CDF against the diagonal. A
calibrated predictive gives PIT ~ Uniform, so its curve sits on the
45-degree line; an over-confident one (intervals too narrow) bows away.
The amortized NFE posterior tracks the diagonal; the same means with
the spread halved -- a stand-in for a point estimate reporting false
precision -- bows off it. Simulator draws only; no real data.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.size": 7, "axes.labelsize": 7,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "axes.linewidth": 0.6,
})
import matplotlib.pyplot as plt

OUT = ROOT / "results" / "figures" / "calibration.pdf"


def main():
    import torch
    from scipy.stats import norm
    from nfe import featurize as _feat, simulate as _sim_train, train_nfe, NFE_KMIX
    torch.manual_seed(0)
    net = train_nfe(seed=0, phi_vol=0.19, s_log=0.14)
    rng = np.random.default_rng(202)
    W, y = _sim_train(20_000, rng, phi_vol=0.19, s_log=0.14)
    with torch.no_grad():
        o = net["mdn"](net["gru"](torch.tensor(_feat(W)))[0][:, -1])
    o = o.view(-1, 3, NFE_KMIX)
    pi = torch.softmax(o[:, 0], -1).numpy()
    mu = o[:, 1].numpy()
    sd = o[:, 2].clamp(-4, 2).exp().numpy()

    def pit(scale):
        # mixture CDF of the true y under the (optionally shrunk) posterior
        return (pi * norm.cdf((y[:, None] - mu) / (sd * scale))).sum(1)

    grid = np.linspace(0, 1, 101)
    fig, ax = plt.subplots(figsize=(2.5, 2.2))
    ax.plot([0, 1], [0, 1], color="black", lw=0.6, ls=":",
            label="perfect calibration")
    for scale, lab, col in [(1.0, "NFE posterior", "#0072B2"),
                            (0.5, "point estimate (half-width)",
                             "#D55E00")]:
        u = np.sort(pit(scale))
        emp = np.searchsorted(u, grid, side="right") / len(u)
        ax.plot(grid, emp, color=col, lw=1.3, label=lab)
    ax.set_xlabel("nominal coverage")
    ax.set_ylabel("empirical coverage")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.legend(frameon=False, fontsize=6, loc="upper left",
              handlelength=1.6, borderaxespad=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(pad=0.3)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
