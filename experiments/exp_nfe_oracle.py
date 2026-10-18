"""Oracle diagnostic: what is the volatility-sharing channel worth?

    uv run python experiments/exp_nfe_oracle.py [--smoke]

[exploration] (2026-07-23). PRE-REGISTRATION — frozen before the first run.
Follow-up forced by exp_nfe_attention's H2 retraction (the raw NFE-vs-GRU
gap survived the independent-panel control -> capacity, not cross-section).
That verdict has two readings: (a) the shared-innovation channel is itself
worthless, or (b) the channel exists but was masked (per-window
standardization erases the common scale; learning may miss it). This
experiment separates them with an oracle upper bound.

DESIGN. Per-name GRU filter (exp_e3_book architecture, identical MDN head
and loss), trained and evaluated twice on the SAME shared-hyperparameter
panel draws as exp_nfe_attention:
    plain   input (z, dz)                       — 2 channels
    oracle  input (z, dz, phi_vol, s_log)       — 4 channels; the TRUE
            panel-level hyper-parameters, standardized, broadcast over the
            window. No estimation: this is knowledge by fiat.
The oracle gain upper-bounds ANY cross-sectional pooling of the innovation
hyper-parameters: attention can at best infer what the oracle is handed.

GATE (registered): paired panel-level NLL gap (plain - oracle) at 2 SE.
    gap < 2 SE  -> the channel is WORTHLESS; the H2 retraction is a
                   property of the design, not of the architecture.
    gap > 2 SE  -> the channel exists and was masked; featurization
                   (panel-scale features) is the next lever.

PROTOCOL. Same constants as exp_nfe_attention: N=64, L=60, train 30k
panels (smoke 2k), 3k steps (smoke 300) batch 512, eval 2,000 panels
(seed 999, the same eval draws), one seed [exploration].
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments"))

from exp_nfe_attention import KMIX, L, N_PANEL, feat, mdn_nll, mdn_pit

SMOKE = "--smoke" in sys.argv
TRAIN_PANELS = 2_000 if SMOKE else 30_000
STEPS = 300 if SMOKE else 3_000
EVAL_PANELS = 400 if SMOKE else 2_000


def sim_shared_with_hp(n_panels, rng):
    """Shared-condition panels plus the true panel hyper-parameters,
    standardized to zero-mean unit-range for the oracle channels."""
    P = n_panels
    phi = rng.uniform(0.0, 0.5, (P, 1)) * np.ones((P, N_PANEL))
    slog = rng.uniform(0.02, 0.30, (P, 1)) * np.ones((P, N_PANEL))
    hl = np.exp(rng.uniform(np.log(3), np.log(400), (P, N_PANEL)))
    k = np.log(2) / hl
    m_ln = np.exp(slog ** 2 / 2)
    sd_ln = np.sqrt((np.exp(slog ** 2) - 1) * np.exp(slog ** 2))
    x = np.zeros((P, N_PANEL)); lv = np.zeros((P, N_PANEL)); m = np.zeros((P, N_PANEL))
    W = np.empty((P, N_PANEL, L), dtype=np.float32)
    for t in range(200 + L):
        lv = phi * lv + 0.5 * np.sqrt(1 - phi ** 2) * rng.standard_normal((P, N_PANEL))
        e = (np.exp(slog * rng.standard_normal((P, N_PANEL))) - m_ln) / sd_ln
        m = m + 0.05 * 0.01 * rng.standard_normal((P, N_PANEL))
        x = x + k * (m - x) + 0.01 * np.exp(lv) * e
        if t >= 200:
            W[:, :, t - 200] = x
    hp = np.stack([(phi - 0.25) / 0.25, (slog - 0.16) / 0.14], -1)
    return W, np.log2(hl).astype(np.float32), hp.astype(np.float32)


def windows(W, hp=None):
    """(P,N,L)->(P*N,L,C): standardized features, oracle channels appended."""
    X = feat(W).reshape(-1, L, 2)
    if hp is None:
        return X
    H = np.repeat(hp.reshape(-1, 1, 2), L, axis=1)
    return np.concatenate([X, H], axis=-1)


def train_gru(Xtr, Ytr, seed=0):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Xtr = torch.tensor(Xtr); Ytr = torch.tensor(Ytr)
    net = nn.ModuleDict({"gru": nn.GRU(Xtr.shape[-1], 64, batch_first=True),
                         "mdn": nn.Linear(64, 3 * KMIX)}).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
    for it in range(STEPS):
        i = torch.randint(0, len(Xtr), (512,))
        out = net["mdn"](net["gru"](Xtr[i].to(dev))[0][:, -1])
        loss = mdn_nll(out, Ytr[i].to(dev)).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    return net, dev


def evaluate(net, dev, Xe, ye):
    import torch
    with torch.no_grad():
        Xe = torch.tensor(Xe)
        outs = [net["mdn"](net["gru"](Xe[j:j + 4096].to(dev))[0][:, -1]).cpu()
                for j in range(0, len(Xe), 4096)]
        out = torch.cat(outs)
        nll = mdn_nll(out, torch.tensor(ye.reshape(-1))).numpy()
        pit = mdn_pit(out, torch.tensor(ye.reshape(-1)))
    return nll, pit


def main():
    from scipy.stats import kstest
    print(f"[{'smoke' if SMOKE else 'exploration'}] oracle diagnostic — "
          f"value of the shared innovation hyper-parameters\n", flush=True)
    rng = np.random.default_rng(0)
    Wtr, ytr, hptr = sim_shared_with_hp(TRAIN_PANELS, rng)
    rng_e = np.random.default_rng(999)
    We, ye, hpe = sim_shared_with_hp(EVAL_PANELS, rng_e)
    res = {}
    for lab, hp_tr, hp_e in [("plain", None, None), ("oracle", hptr, hpe)]:
        net, dev = train_gru(windows(Wtr, hp_tr), ytr.reshape(-1))
        nll, pit = evaluate(net, dev, windows(We, hp_e), ye)
        res[lab] = (nll.reshape(EVAL_PANELS, -1).mean(1),
                    kstest(pit, "uniform").statistic)
        print(f"  {lab:6s}  NLL {res[lab][0].mean():.3f}  "
              f"PIT KS {res[lab][1]:.3f}", flush=True)
    d = res["plain"][0] - res["oracle"][0]
    se = d.std(ddof=1) / np.sqrt(len(d))
    verdict = ("channel EXISTS and was masked -> featurization is the lever"
               if d.mean() > 2 * se else
               "channel WORTHLESS -> H2 retraction is the design's property")
    print(f"\n  oracle gain (plain - oracle) {d.mean():+.3f} +- {se:.3f}"
          f"  [{'PASS' if d.mean() > 2*se else 'null'}]\n  -> {verdict}",
          flush=True)


if __name__ == "__main__":
    main()
