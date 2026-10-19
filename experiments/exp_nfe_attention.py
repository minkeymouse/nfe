"""Neural Filter Estimator: row-column attention over the panel.

    uv run python experiments/exp_nfe_attention.py [--smoke]

[exploration] (2026-07-23). PRE-REGISTRATION — hypotheses and gates frozen
before the first full run. Design decision: the attention implementation
belongs to THIS paper; the ICLR follow-up generalizes beyond finance.

LINEAGE. FF3 -> DFM -> AFM -> TabFM -> NFE. A factor model compresses a
day's cross-section into a few numbers; TabFM does the
same with attention, so its row embedding IS the generalized factor. What
it lacks for a financial panel is dynamics: its ICL stage attends over
compressed rows as an EXCHANGEABLE set (no position encoding on the row
axis — see .../pytorch/model.py, ICLearning rope=None). The NFE keeps the
row-column factorization and the synthetic-prior recipe and adds an
ordered time axis with a forced dynamic prior, so the output is a
filtering posterior rather than a zero-shot label. Hence the one
structural change: position encoding on TIME, none on ASSETS —
permutation-equivariant across names, order-aware across days.

QUESTION. Does the row-column attention NFE beat the per-name GRU filter
at the same task — the posterior over each name's mean-reversion time —
on truth-known panel draws?

WHY IT CAN. The per-name filter sees one window; a panel filter can pool
the cross-section. Here the shared structure is panel-level innovation
hyper-parameters (phi_vol, s_log drawn per panel), which asset attention
can infer to sharpen every name's posterior — a learned empirical Bayes.
This is the smallest honest cross-sectional structure; common FACTORS in
levels are the follow-up object, not this experiment.

HYPOTHESES (registered, gates at 2 SE over eval panels)
  H1  On SHARED-hyperparameter panels, the NFE beats the per-name GRU on
      held-out NLL of log2 half-life.
  H2  SPECIFICITY: on panels with per-name INDEPENDENT hyper-parameters,
      the gain vanishes (|gap| < 2 SE) — asset attention has nothing to
      pool; if a gain persists, it is capacity, not cross-section, and
      the cross-sectional claim is retracted.
  H3  Calibration (PIT of true log2-HL under the posterior) stays
      uniform (KS not worse than the GRU's by more than 0.02).

PROTOCOL. Panels N=64 names, L=60; train 30k panels (smoke 2k), Adam 3k
steps (smoke 300) batch 16; eval 2,000 fresh panels per condition, one
seed [exploration]. Same prior ranges as exp_e3_book._sim_train, with
panel-level phi_vol ~ U(0, 0.5), s_log ~ U(0.02, 0.30) in the shared
condition and per-name draws of the same ranges in the independent one.
The GRU baseline (exp_e3_book architecture, identical MDN head and loss)
trains on the SAME panel budget reshaped to per-name windows.

NOTE (post-smoke fix, before the first full run). The smoke baseline
trained the GRU on TRAIN_PANELS//10 panels, an unregistered shortcut that
confounds any gap with data budget; the smoke signature (equal gap in
both conditions) is what such a confound produces. Fixed to the full
matched pool, hypotheses and gates unchanged. Smoke verdicts are
mechanics-only.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT))

# the panel-form architecture lives in the nfe package; this experiment
# tests it (H1/H2/H3). Re-export under the local names its dependents use.
from nfe import build_panel_nfe                                # noqa: E402
from nfe.panel import (panel_featurize as feat,            # noqa: E402
                           panel_mdn_nll as mdn_nll,
                           panel_mdn_pit as mdn_pit,
                           panel_simulate as sim_panels)

SMOKE = "--smoke" in sys.argv
L, KMIX = 60, 5
N_PANEL = 64
TRAIN_PANELS = 2_000 if SMOKE else 30_000
STEPS = 300 if SMOKE else 3_000
EVAL_PANELS = 400 if SMOKE else 2_000
EVAL_CHUNK = 100


def train_panel(shared, seed=0):
    import torch
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(seed)
    W, y = sim_panels(TRAIN_PANELS, N_PANEL, rng, shared=shared)
    X = torch.tensor(feat(W)); Y = torch.tensor(y)
    net = build_panel_nfe().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=3e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
    t0 = time.time()
    for it in range(STEPS):
        i = torch.randint(0, len(X), (16,))
        loss = mdn_nll(net(X[i].to(dev)), Y[i].to(dev)).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        if it % max(1, STEPS // 5) == 0:
            print(f"    step {it:>5} nll {loss.item():.3f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    return net


_GRU_CACHE = {}


def gru_eval_nll(Wp, yp):
    """Per-name GRU baseline (exp_e3_book architecture, identical head and
    loss) on the same eval draws, trained on the SAME panel budget
    reshaped to per-name windows. The per-name marginal law of a single
    window is identical across the shared/independent conditions, so one
    trained baseline serves both."""
    import torch
    import torch.nn as nn
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if "net" not in _GRU_CACHE:
        torch.manual_seed(0)
        rng = np.random.default_rng(50)
        Wtr, ytr = sim_panels(TRAIN_PANELS, N_PANEL, rng, shared=True)
        Xtr = torch.tensor(feat(Wtr).reshape(-1, L, 2))
        Ytr = torch.tensor(ytr.reshape(-1))
        net = nn.ModuleDict({"gru": nn.GRU(2, 64, batch_first=True),
                             "mdn": nn.Linear(64, 3 * KMIX)}).to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=3e-3)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
        for it in range(STEPS):
            i = torch.randint(0, len(Xtr), (512,))
            out = net["mdn"](net["gru"](Xtr[i].to(dev))[0][:, -1])
            loss = mdn_nll(out, Ytr[i].to(dev)).mean()
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        _GRU_CACHE["net"] = net
    net = _GRU_CACHE["net"]
    with torch.no_grad():
        Xe = torch.tensor(feat(Wp).reshape(-1, L, 2))
        outs = []
        for j in range(0, len(Xe), 4096):
            outs.append(net["mdn"](net["gru"](
                Xe[j:j + 4096].to(dev))[0][:, -1]).cpu())
        out = torch.cat(outs)
        nll = mdn_nll(out, torch.tensor(yp.reshape(-1)))
        pit = mdn_pit(out, torch.tensor(yp.reshape(-1)))
    return nll.numpy(), pit


def main():
    import torch
    from scipy.stats import kstest
    print(f"[{'smoke' if SMOKE else 'exploration'}] row-column attention NFE "
          f"vs per-name GRU  (panels N={N_PANEL}, L={L})\n", flush=True)
    res = {}
    for shared, lab in [(True, "SHARED hyper-params (H1)"),
                        (False, "INDEPENDENT hyper-params (H2 specificity)")]:
        print(f"  == {lab} ==", flush=True)
        net = train_panel(shared=shared)
        dev = next(net.parameters()).device
        rng = np.random.default_rng(999)
        Wp, yp = sim_panels(EVAL_PANELS, N_PANEL, rng, shared=shared)
        with torch.no_grad():
            Xe = torch.tensor(feat(Wp))
            out = torch.cat([net(Xe[j:j + EVAL_CHUNK].to(dev)).cpu()
                             for j in range(0, len(Xe), EVAL_CHUNK)])
            nll_p = mdn_nll(out, torch.tensor(yp)).numpy().ravel()
            pit_p = mdn_pit(out, torch.tensor(yp)).ravel()
        nll_g, pit_g = gru_eval_nll(Wp, yp)
        # panel-level paired gap (panels are the independent unit)
        gp = nll_p.reshape(EVAL_PANELS, -1).mean(1)
        gg = nll_g.reshape(EVAL_PANELS, -1).mean(1)
        d = gg - gp
        se = d.std(ddof=1) / np.sqrt(len(d))
        ks_p = kstest(pit_p, "uniform").statistic
        ks_g = kstest(pit_g, "uniform").statistic
        res[shared] = (d.mean(), se, ks_g, ks_p)
        print(f"  NLL  per-name GRU {gg.mean():.3f} | NFE {gp.mean():.3f}"
              f" | gap {d.mean():+.3f} +- {se:.3f} "
              f"[{'PASS' if abs(d.mean()) > 2*se else 'null'}]", flush=True)
        print(f"  PIT KS  GRU {ks_g:.3f} | NFE {ks_p:.3f}\n", flush=True)

    gs, ses, _, _ = res[True]
    gi, sei, _, _ = res[False]
    ctr, sec = gs - gi, float(np.hypot(ses, sei))
    print(f"  H2 contrast (shared gap - indep gap) {ctr:+.3f} +- {sec:.3f}"
          f" -> {'cross-sectional pooling' if ctr > 2 * sec else 'capacity, not cross-section'}",
          flush=True)
    dk_s = res[True][3] - res[True][2]
    dk_i = res[False][3] - res[False][2]
    print(f"  H3 PIT KS worsening vs GRU: shared {dk_s:+.3f}, "
          f"indep {dk_i:+.3f} (gate 0.02)", flush=True)


if __name__ == "__main__":
    main()
