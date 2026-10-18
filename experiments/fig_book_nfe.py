"""Paper figure: cumulative returns of the era-2 books, with/without NFE.

    uv run python experiments/fig_book_nfe.py

(2026-07-23). AFM's Figure 2 analog, but the contrast is the paper's point:
not the number of factors K, but the SAME policy with and without the
amortized posterior attached. Encoding: color = residual model (Okabe-Ito,
colorblind-safe), line style = policy (solid base, dashed base+NFE, dotted
the model's own alternative). Source: results/run_260728/book_pnl_era2.npz
(exp_e3_book dump; evaluation days only, net of costs; 2026-07-28 = the
column-wise-prior provenance re-run, NFE arm only changed).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.size": 7, "axes.labelsize": 7, "legend.fontsize": 5.8,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "axes.linewidth": 0.6,
})
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "results" / "run_260728" / "book_pnl_era2.npz"
OUT = ROOT / "results" / "figures" / "book_cumret.pdf"

BLUE, VERM, GREEN, ORAN, GRAY = ("#0072B2", "#D55E00", "#009E73",
                                 "#E69F00", "#7f7f7f")
# lineage view (2026-07-24): one line per generation, the base rule for
# the three factor constructions, the self-contained panel arm for NFE.
SERIES = [  # (npz key, label, color, linestyle, linewidth)
    ("FF3",          "FF3",              ORAN,  "-", 1.1),
    ("PCA",          "PCA",              BLUE,  "-", 1.2),
    ("AFM+fade",     "AFM",              VERM,  "-", 1.2),
    ("AFM+NFE",      "NFE",              GREEN, "-", 1.2),
    ("random-slow",  "random benchmark", GRAY,  "-", 0.9),
]


def main():
    z = np.load(SRC, allow_pickle=True)
    dts = pd.to_datetime(z["dates"].astype(str))
    # Volatility-matched: scale each book to a common annualized vol
    # (10%) so the plot shows the risk-adjusted ordering (the paper's
    # claim) rather than raw return; the random benchmark is left at
    # its own scale as the floor.
    TARGET = 0.10 / np.sqrt(246.0)
    fig, ax = plt.subplots(figsize=(3.35, 1.5))
    for key, lab, c, ls, lw in SERIES:
        r = z[key].astype(float)
        if key != "random-slow":
            r = r * (TARGET / (r.std() + 1e-12))
        eq = np.cumprod(1.0 + r)
        ax.plot(dts, eq, color=c, linestyle=ls, linewidth=lw, label=lab)
    ax.axhline(1.0, color="black", linewidth=0.4, alpha=0.4)
    ax.set_ylabel("Cumulative return\n(vol-matched)")
    ax.legend(loc="upper left", frameon=False, handlelength=2.4,
              borderaxespad=0.2, labelspacing=0.35)
    ax.spines[["top", "right"]].set_visible(False)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.margins(x=0.01)
    fig.tight_layout(pad=0.3)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
