"""NFE factor-sharing: does asset attention learn the row step?

    uv run python experiments/exp_nfe_factor.py [--smoke]

[exploration] (2026-07-23). PRE-REGISTRATION — frozen before the first
full run. Design decision: after the vol-sharing design retracted its
pooling claim (exp_nfe_attention H2; oracle bound in exp_nfe_oracle),
move the shared component to where it enters the target's identification
FIRST-ORDER: a common factor in levels. This is the ladder's measured
lesson (the unknown level costs 0.240) made cross-sectional, and it is
the succession's own claim run as an experiment — the row step (FF3 ->
DFM -> AFM's learned factor -> TabFM's row embedding) executed by asset
attention inside the filter.

DESIGN. Observed level S_i = beta_i F_t + x_i: F a random walk with the
innovation scale (0.01 daily), beta_i ~ U(0.5, 1.5) per name, x_i the
admitted switching-OU spread of exp_nfe_attention.sim_panels with
PER-NAME innovation hyper-parameters in BOTH conditions (the vol channel
is closed; the factor is the only shared component).
    SHARED       one F path per panel — the cross-section identifies F.
    INDEPENDENT  each name its own F path, identical marginal law — a
                 single window's distribution is unchanged, so the
                 per-name problem is equally hard; there is only nothing
                 to pool.
Target: log2 half-life of x_i. One window cannot separate a random walk
from a slow reversion around a drifting level; sixty-three neighbours
sharing the walk can.

HYPOTHESES (registered, gates at 2 SE over eval panels)
  H1  On shared-factor panels the panel NFE beats the per-name GRU on
      held-out NLL of log2 half-life.
  H2  POOLING CONTRAST — the lesson of the vol experiment, where
      capacity confounded the raw gap: gap_shared - gap_independent
      > 2 SE. The contrast, not the raw gap, is the pooling measure.
  H3  Calibration: the NFE's PIT KS within 0.02 of the GRU's on the
      shared condition.

PROTOCOL. As exp_nfe_attention: N=64, L=60, train 30k panels (smoke 2k),
Adam 3k steps (smoke 300) batch 16, eval 2,000 fresh panels per
condition (seed 999), GRU baseline on the same panel budget reshaped to
windows (per-name marginal identical across conditions -> one baseline
serves both), one seed [exploration].

CONFIRMATION (registered 2026-07-23 BEFORE the confirm run; decided
after the exploration pass H1 +0.326 / contrast +0.179 / H3 pass).
--confirm re-runs the identical protocol on disjoint seeds: panel train
seed 1, baseline pool seed 151, eval seed 1999. Gates UNCHANGED: H1 gap
> 2 SE, H2 contrast > 2 SE, H3 KS within 0.02 of the GRU's. Output tags
[confirmation]; a failure at any gate is reported as such.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments"))

from exp_nfe_attention import (KMIX, L, N_PANEL, build_panel_nfe, feat,
                               mdn_nll, mdn_pit, sim_panels)

SMOKE = "--smoke" in sys.argv
CONFIRM = "--confirm" in sys.argv
TRAIN_PANELS = 2_000 if SMOKE else 30_000
STEPS = 300 if SMOKE else 3_000
EVAL_PANELS = 400 if SMOKE else 2_000
EVAL_CHUNK = 100
F_VOL = 0.01
# fresh disjoint seeds for the registered confirmation pass
TRAIN_SEED = 1 if CONFIRM else 0
POOL_SEED = 151 if CONFIRM else 50
EVAL_SEED = 1999 if CONFIRM else 999


def sim_factor_panels(n_panels, rng, shared=True):
    """S = beta*F + x. x: admitted spreads, per-name hyper-params in both
    conditions. F: random walk, one path per panel (shared) or per name
    (independent control, identical marginal). Returns S (P,N,L), y."""
    W, y = sim_panels(n_panels, N_PANEL, rng, shared=False)
    beta = rng.uniform(0.5, 1.5, (n_panels, N_PANEL, 1))
    if shared:
        F = np.cumsum(F_VOL * rng.standard_normal((n_panels, 1, L)), -1)
    else:
        F = np.cumsum(F_VOL * rng.standard_normal((n_panels, N_PANEL, L)), -1)
    return (W + beta * F).astype(np.float32), y


def train_panel(shared, seed=TRAIN_SEED):
    import torch
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(seed)
    W, y = sim_factor_panels(TRAIN_PANELS, rng, shared=shared)
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
    """Per-name GRU baseline on the same eval draws, trained on the same
    panel budget of shared-condition draws (per-name marginal identical
    across conditions)."""
    import torch
    import torch.nn as nn
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if "net" not in _GRU_CACHE:
        torch.manual_seed(POOL_SEED if CONFIRM else 0)
        rng = np.random.default_rng(POOL_SEED)
        Wtr, ytr = sim_factor_panels(TRAIN_PANELS, rng, shared=True)
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
        outs = [net["mdn"](net["gru"](Xe[j:j + 4096].to(dev))[0][:, -1]).cpu()
                for j in range(0, len(Xe), 4096)]
        out = torch.cat(outs)
        nll = mdn_nll(out, torch.tensor(yp.reshape(-1)))
        pit = mdn_pit(out, torch.tensor(yp.reshape(-1)))
    return nll.numpy(), pit


def main():
    import torch
    from scipy.stats import kstest
    tag = "smoke" if SMOKE else ("confirmation" if CONFIRM else "exploration")
    print(f"[{tag}] factor-sharing NFE "
          f"vs per-name GRU  (S = beta*F + x; N={N_PANEL}, L={L})\n",
          flush=True)
    res = {}
    for shared, lab in [(True, "SHARED factor (H1)"),
                        (False, "INDEPENDENT factors (control)")]:
        print(f"  == {lab} ==", flush=True)
        net = train_panel(shared=shared)
        dev = next(net.parameters()).device
        rng = np.random.default_rng(EVAL_SEED)
        Wp, yp = sim_factor_panels(EVAL_PANELS, rng, shared=shared)
        with torch.no_grad():
            Xe = torch.tensor(feat(Wp))
            out = torch.cat([net(Xe[j:j + EVAL_CHUNK].to(dev)).cpu()
                             for j in range(0, len(Xe), EVAL_CHUNK)])
            nll_p = mdn_nll(out, torch.tensor(yp)).numpy().ravel()
            pit_p = mdn_pit(out, torch.tensor(yp)).ravel()
        nll_g, pit_g = gru_eval_nll(Wp, yp)
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

    gs, ses, _, ksp_s = res[True]
    gi, sei, ksg_s, _ = res[False]
    ctr, sec = gs - gi, float(np.hypot(ses, sei))
    print(f"  H2 pooling contrast (shared gap - indep gap) "
          f"{ctr:+.3f} +- {sec:.3f} "
          f"[{'PASS: cross-sectional pooling' if ctr > 2 * sec else 'null: capacity only'}]",
          flush=True)
    print(f"  H3 KS worsening vs GRU (shared): "
          f"{res[True][3] - res[True][2]:+.3f} (gate 0.02)", flush=True)


if __name__ == "__main__":
    main()
