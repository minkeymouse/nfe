"""Paper figure: the geometry of the amortized posterior.

    uv run python experiments/fig_posterior_tsne.py

(2026-07-23). Panel (a): t-SNE of the NFE encoder embeddings on fresh
truth-known draws of the admitted process, colored by the true log2
half-life -- shows the representation organizing by the latent state.
Panel (b): example posterior densities q_psi(log2 HL | w) for a fast, an
ambiguous, and a slow window, with the slow-state tail shaded (regime
probability = tail mass; threshold drawn at 60 days for illustration).
Simulator draws only; nothing here touches real data.
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

OUT = ROOT / "results" / "figures" / "posterior_tsne.pdf"


def main():
    import torch
    from nfe import train_nfe, simulate as _sim_train, featurize as _feat, NFE_KMIX
    torch.manual_seed(0)
    net = train_nfe(seed=0, phi_vol=0.19, s_log=0.14)   # gate-calibrated
    rng = np.random.default_rng(123)
    W, y = _sim_train(4000, rng, phi_vol=0.19, s_log=0.14)
    X = torch.tensor(_feat(W))
    with torch.no_grad():
        H = net["gru"](X)[0][:, -1]
        o = net["mdn"](H).view(-1, 3, NFE_KMIX)
        pi = torch.softmax(o[:, 0], -1).numpy()
        mu = o[:, 1].numpy()
        sd = o[:, 2].clamp(-4, 2).exp().numpy()
    H = H.numpy()

    from sklearn.manifold import TSNE
    Z = TSNE(n_components=2, perplexity=40, init="pca",
             random_state=0).fit_transform(H)

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(7.2, 2.32), gridspec_kw={"width_ratios": [1.4, 1]})
    fig.subplots_adjust(wspace=0.32)

    sc = ax1.scatter(Z[:, 0], Z[:, 1], c=y, s=2.5, cmap="viridis",
                     linewidths=0, rasterized=True)
    cb = fig.colorbar(sc, ax=ax1, fraction=0.045, pad=0.02)
    cb.set_label(r"true $\log_2$ half-life (days)", fontsize=6.5)
    cb.ax.tick_params(labelsize=6)
    ax1.set_xticks([]); ax1.set_yticks([])
    for sp in ax1.spines.values():
        sp.set_visible(False)
    ax1.set_title("(a) encoder embeddings, t-SNE", fontsize=7)

    # panel (b): fast / ambiguous / slow example posteriors
    order = np.argsort(y)
    idx_fast = order[60]
    idx_slow = order[-60]
    # ambiguous: highest predictive sd among mid-half-life draws
    mid = order[len(order) // 3: 2 * len(order) // 3]
    mix_var = (pi * (sd ** 2 + mu ** 2)).sum(1) - ((pi * mu).sum(1)) ** 2
    idx_amb = mid[np.argmax(mix_var[mid])]
    grid = np.linspace(np.log2(2), np.log2(1200), 400)
    colors = {"fast": "#0072B2", "ambiguous": "#7f7f7f", "slow": "#D55E00"}
    thr = np.log2(60.0)
    for lab, i in [("fast", idx_fast), ("ambiguous", idx_amb),
                   ("slow", idx_slow)]:
        dens = np.zeros_like(grid)
        for k in range(NFE_KMIX):
            dens += pi[i, k] / (sd[i, k] * np.sqrt(2 * np.pi)) * \
                np.exp(-0.5 * ((grid - mu[i, k]) / sd[i, k]) ** 2)
        ax2.plot(grid, dens, color=colors[lab], lw=1.1,
                 label=f"{lab} (true HL {2**y[i]:.0f}d)")
        if lab == "ambiguous":
            m = grid >= thr
            ax2.fill_between(grid[m], 0, dens[m], color=colors[lab],
                             alpha=0.25, lw=0)
    ax2.axvline(thr, color="black", lw=0.5, ls=":")
    # headroom above the densities so the legend never overlaps a curve
    ax2.set_ylim(0, ax2.get_ylim()[1] * 1.34)
    ax2.text(thr + 0.10, ax2.get_ylim()[1] * 0.60,
             r"$P(z{=}\mathrm{D})$ = tail mass", fontsize=6)
    ax2.set_xlabel(r"$\log_2$ half-life (days)")
    ax2.set_ylabel(r"$q_\psi$")
    ax2.legend(frameon=False, fontsize=6, loc="upper left",
               borderaxespad=0.2, labelspacing=0.3, handlelength=1.5)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.set_title("(b) example posteriors", fontsize=7)

    fig.tight_layout(pad=0.4)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", dpi=300)
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
