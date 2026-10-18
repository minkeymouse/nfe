"""Anchored NFE: does a cross-sectional anchor channel improve the filter?

    uv run python experiments/exp_nfe_anchor.py [--smoke]

[exploration] (2026-07-24). PRE-REGISTRATION — frozen before the first
full run. Motivation chain, all measured: the identification bottleneck
is the level (ladder, −0.240); pooling pays when the shared component is
a factor in levels (exp_nfe_factor, confirmed); the KRX residual panel
retains common structure above its shuffled null (exp_anchor_fuel,
lam1−null = +0.033 ± 0.002, vs +0.234 for raw returns) — small but
unambiguous fuel. This experiment asks whether handing the per-name
filter a cross-sectional ANCHOR CHANNEL converts that structure into a
better posterior.

DESIGN. Panels S_i = x_i + b_i F with x_i the admitted per-name spread
(per-name innovation hyper-parameters), b_i ~ U(0.5, 1.5), F a common
random-walk level factor scaled so the factor's share of increment
variance is
    matched  ~5%   (the order the KRX residual panel retains)
    strong   ~50%  (the regime where pooling was confirmed)
ARMS, identical GRU+MDN; target log2 half-life of x:
    plain     filter on the own standardized window
    anchored  filter on the window RESIDUALIZED within-window against
              the cross-sectional mean: beta_i from an OLS of own
              increments on mean increments, input S_i − beta_i·Abar.
              Causal, deterministic, per window — a window-level row
              step.
AMENDMENT (post-smoke, pre-full-run): the first registered form gave
the filter the anchor as extra input channels and let it learn the
integration; at the protocol budget the learned integration was inert
(gap ~0.001 in both conditions). Replaced by the explicit
residualization above BEFORE the first full run; gates unchanged.
GATES (registered; paired panel-level NLL gaps, 2 SE)
  H1  strong: anchored beats plain (> 2 SE) — the mechanism works.
  H2  dose-response: gap(strong) > gap(matched) (> 2 SE on the
      difference) — the gain scales with the fuel.
  H3  matched: the gap is REPORTED either way; expectation recorded in
      advance: small, possibly unresolvable — this is the honest bound
      on what the anchor buys at KRX-like fuel, and a null here is a
      finding, not a failure.
PROTOCOL. N=64, L=60, train 30k panels (smoke 2k) per condition per
arm, 3k steps (300) batch 512 windows, eval 2,000 fresh panels
(seed 999), one seed [exploration].
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments"))

from exp_nfe_attention import KMIX, L, N_PANEL, mdn_nll, sim_panels

SMOKE = "--smoke" in sys.argv
TRAIN_PANELS = 2_000 if SMOKE else 30_000
STEPS = 300 if SMOKE else 3_000
EVAL_PANELS = 400 if SMOKE else 2_000
SHARES = {"matched": 0.05, "strong": 0.50}


def sim_anchor_panels(n_panels, rng, share):
    """S = x + b*F, F a random-walk level factor with the given share of
    increment variance. Returns S (P,N,L), y (P,N) log2 HL of x, and the
    realized share (verification)."""
    W, y = sim_panels(n_panels, N_PANEL, rng, shared=False)
    dx = np.diff(W, axis=-1)
    sx2 = float(np.mean(dx ** 2))
    b = rng.uniform(0.5, 1.5, (n_panels, N_PANEL, 1))
    sF = np.sqrt(share / (1.0 - share) * sx2 / float(np.mean(b ** 2)))
    F = np.cumsum(sF * rng.standard_normal((n_panels, 1, L)), -1)
    S = (W + b * F).astype(np.float32)
    dS = np.diff(S, axis=-1)
    real = 1.0 - sx2 / float(np.mean(dS ** 2))
    return S, y, real


def feat_plain(S):
    mu = S.mean(-1, keepdims=True)
    sd = S.std(-1, keepdims=True) + 1e-12
    Z = (S - mu) / sd
    dZ = np.diff(Z, axis=-1, prepend=Z[..., :1])
    return np.stack([Z, dZ], axis=-1).astype(np.float32)


def feat_anchor(S):
    """Explicit within-window residualization against the cross-sectional
    mean: beta_i from OLS of own increments on mean increments (levels
    of two random walks would be spurious), then the plain features of
    S_i − beta_i·Abar. Causal and deterministic."""
    A = S.mean(1, keepdims=True)                       # (P, 1, L)
    dS = np.diff(S, axis=-1)
    dA = np.diff(A, axis=-1)
    va = (dA * dA).sum(-1) + 1e-18                     # (P, 1)
    beta = (dS * dA).sum(-1) / va                      # (P, N)
    R = S - beta[..., None] * A
    return feat_plain(R.astype(np.float32))


def train_gru(Xtr, Ytr, seed=0):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Xtr = torch.tensor(Xtr.reshape(-1, L, Xtr.shape[-1]))
    Ytr = torch.tensor(Ytr.reshape(-1))
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


def eval_nll(net, dev, Xe, ye):
    import torch
    with torch.no_grad():
        Xe = torch.tensor(Xe.reshape(-1, L, Xe.shape[-1]))
        outs = [net["mdn"](net["gru"](Xe[j:j + 4096].to(dev))[0][:, -1]).cpu()
                for j in range(0, len(Xe), 4096)]
        out = torch.cat(outs)
        return mdn_nll(out, torch.tensor(ye.reshape(-1))).numpy()


def main():
    print(f"[{'smoke' if SMOKE else 'exploration'}] anchored NFE — "
          f"cross-sectional anchor channel  (N={N_PANEL}, L={L})\n",
          flush=True)
    gaps = {}
    for cond, share in SHARES.items():
        rng = np.random.default_rng(1)
        Str, ytr, real = sim_anchor_panels(TRAIN_PANELS, rng, share)
        rng_e = np.random.default_rng(999)
        Se, ye, _ = sim_anchor_panels(EVAL_PANELS, rng_e, share)
        print(f"  == {cond} (target share {share:.2f}, realized "
              f"{real:.2f}) ==", flush=True)
        res = {}
        for arm, ffn in [("plain", feat_plain), ("anchored", feat_anchor)]:
            net, dev = train_gru(ffn(Str), ytr)
            nll = eval_nll(net, dev, ffn(Se), ye)
            res[arm] = nll.reshape(EVAL_PANELS, -1).mean(1)
            print(f"    {arm:9s} NLL {res[arm].mean():.3f}", flush=True)
        d = res["plain"] - res["anchored"]
        se = d.std(ddof=1) / np.sqrt(len(d))
        gaps[cond] = (d.mean(), se)
        print(f"    gap (plain - anchored) {d.mean():+.3f} +- {se:.3f} "
              f"[{'PASS' if d.mean() > 2*se else 'null'}]\n", flush=True)

    gs, ses = gaps["strong"]
    gm, sem = gaps["matched"]
    dd, sed = gs - gm, float(np.hypot(ses, sem))
    print(f"  H1 strong gap    {gs:+.3f} +- {ses:.3f} "
          f"[{'PASS' if gs > 2*ses else 'FAIL'}]", flush=True)
    print(f"  H2 dose-response {dd:+.3f} +- {sed:.3f} "
          f"[{'PASS' if dd > 2*sed else 'FAIL'}]", flush=True)
    print(f"  H3 matched gap   {gm:+.3f} +- {sem:.3f} "
          f"[{'resolved' if abs(gm) > 2*sem else 'null'}] "
          f"(the honest bound at KRX-like fuel)", flush=True)


if __name__ == "__main__":
    main()
