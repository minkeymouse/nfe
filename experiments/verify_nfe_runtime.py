"""Runtime of the amortized posterior against its unamortized counterpart.

    uv run python experiments/verify_nfe_runtime.py

[measurement] (2026-07-28). A verification pass, not an experiment: no
hypotheses, no gates. The paper's premise is that amortization is what
makes the filtering posterior affordable at the cross-section; this
script puts wall-clock numbers on that premise, which the draft
currently asserts without measuring.

THREE COSTS, one 120-name cross-section (the book's size), L=60:

  amortized   sig_nfe: one batched forward pass + posterior sizing.
              (Training is a one-time cost, timed separately.)
  plug-in     the classical per-window OLS AR(1) refit (Avellaneda-Lee
              practice), vectorized. Carries NO distribution -- included
              as the practice baseline, not as a posterior.
  unamortized the same posterior WITHOUT amortization: on the same
              generative model (switching_ou defaults, the values the
              net is trained with), p(tau | window) by a G-point
              half-life grid, each grid point's marginal likelihood by
              a bootstrap particle filter over the latent (log-vol,
              level) pair, P particles, fully vectorized across
              names x grid x particles (best effort for the comparator,
              so the contrast is conservative).

SANITY. A timing for a broken filter is meaningless, so the unamortized
posterior must first be sane: across the 120 truth-known draws its
posterior mean log2 half-life must correlate positively with the truth,
at a level comparable to the NFE's. Both correlations are printed and
saved; if the PF's is not positive the timing is void.

HARDWARE is recorded in the artifact (CPU model, thread count). Median
of NREP repeats after a warm-up; the one-time training cost is a single
wall-clock measurement. Artifact: results/run_260728/nfe_runtime.json.
"""
from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from nfe import L, simulate, sig_nfe, train_nfe          # noqa: E402
from nfe.prior import HL_HI, HL_LO                   # noqa: E402

N_NAMES = 120        # the book's cross-section
NREP = 20            # timing repeats (median)
GRID = 25            # half-life grid points (unamortized comparator)
NPART = 256          # particles per grid point
PHI_VOL, S_LOG, TAU = 0.19, 0.14, 0.05   # switching_ou defaults


def pf_grid_posterior(W, phi_vol=PHI_VOL, s_log=S_LOG, tau=TAU,
                      grid=GRID, npart=NPART, seed=0):
    """p(log2 half-life | window) on a grid, marginal likelihood per grid
    point by a bootstrap particle filter over (log-vol, level).

    Mirrors nfe.prior.simulate exactly: lv AR(1) with innovation sd
    0.5*sqrt(1-phi^2); level random walk with step sd tau*0.01;
    increment = k*(m - x) + 0.01*exp(lv)*zeta with zeta the centred,
    unit-variance lognormal. Vectorized over names x grid x particles.
    Returns (posterior weights (n, grid), grid of log2 half-lives).
    """
    rng = np.random.default_rng(seed)
    n = len(W)
    ys = np.linspace(np.log2(HL_LO), np.log2(HL_HI), grid)   # flat prior
    k = np.log(2) / 2.0 ** ys                                 # (G,)
    m_ln = np.exp(s_log ** 2 / 2)
    sd_ln = np.sqrt((np.exp(s_log ** 2) - 1) * np.exp(s_log ** 2))
    # particles: (n, G, P) latent log-vol and level, both from their
    # generative marginals at the window start (lv stationary; m a
    # 200-step random walk from 0, exactly as simulate's burn-in leaves it).
    # x_0 is taken as given (the filter starts at the first observation).
    lv = 0.5 * rng.standard_normal((n, grid, npart))          # stationary sd 0.5
    m = tau * 0.01 * np.sqrt(200.0) * rng.standard_normal((n, grid, npart))
    loglik = np.zeros((n, grid))
    x = W[:, 0]
    for t in range(1, L):
        lv = phi_vol * lv + 0.5 * np.sqrt(1 - phi_vol ** 2) \
            * rng.standard_normal((n, grid, npart))
        m = m + tau * 0.01 * rng.standard_normal((n, grid, npart))
        dx = (W[:, t] - x)[:, None, None]                     # observed increment
        a = k[None, :, None] * (m - x[:, None, None])         # reversion pull
        b = 0.01 * np.exp(lv) / sd_ln                         # innovation scale
        w = (dx - a) / b + m_ln                               # back to e^{gamma u}
        good = w > 1e-12
        lw = np.where(good, np.log(np.maximum(w, 1e-12)), 0.0)
        logp = np.where(
            good,
            -lw - 0.5 * (lw / s_log) ** 2 - np.log(b) - np.log(s_log)
            - 0.5 * np.log(2 * np.pi),
            -np.inf)
        mx = logp.max(-1, keepdims=True)
        pw = np.exp(logp - np.where(np.isfinite(mx), mx, 0.0))
        tot = pw.sum(-1)
        loglik += np.where(np.isfinite(mx[..., 0]),
                           mx[..., 0] + np.log(tot / npart + 1e-300), -1e3)
        # systematic resampling, chunked over the grid axis so the
        # comparison tensor stays (n, P, P) rather than (n, G, P, P)
        cdf = np.cumsum(pw, -1)
        cdf /= np.maximum(cdf[..., -1:], 1e-300)
        pos = (np.arange(npart) + rng.random((n, grid, 1))) / npart
        idx = np.empty((n, grid, npart), dtype=np.int64)
        for g in range(grid):
            idx[:, g] = np.minimum(
                (cdf[:, g][:, None, :] < pos[:, g][:, :, None]).sum(-1),
                npart - 1)
        lv = np.take_along_axis(lv, idx, -1)
        m = np.take_along_axis(m, idx, -1)
        x = W[:, t]
    lw = loglik - loglik.max(1, keepdims=True)
    post = np.exp(lw)
    post /= post.sum(1, keepdims=True)
    return post, ys


def plugin_fit(hist):
    """The classical per-window refit: OLS AR(1) kappa per name, plus the
    point-estimate sizing (the variance term is identically zero)."""
    X = np.cumsum(hist, axis=-1)
    m, sd = X.mean(-1), X.std(-1) + 1e-12
    z = np.clip((X[:, -1] - m) / sd, -3, 3)
    Z = X - X.mean(-1, keepdims=True)
    dz = np.diff(Z, axis=-1)
    xl = Z[:, :-1]
    beta = (xl * dz).sum(-1) / ((xl * xl).sum(-1) + 1e-18)
    k = np.clip(-beta, np.log(2) / HL_HI, np.log(2) / HL_LO)
    a = np.exp(-k * 20)
    return -z * (1.0 - a) / (1.0 - a ** 2 + 1e-12)


def med_time(fn, nrep=NREP):
    fn()                                                      # warm-up
    ts = []
    for _ in range(nrep):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def main():
    import torch
    cpu = platform.processor() or "unknown"
    try:
        cpu = [ln.split(":", 1)[1].strip()
               for ln in Path("/proc/cpuinfo").read_text().splitlines()
               if ln.startswith("model name")][0]
    except Exception:
        pass
    print(f"[measurement] NFE runtime — {N_NAMES}-name cross-section, L={L}")
    print(f"  cpu: {cpu} | torch threads {torch.get_num_threads()}\n")

    rng = np.random.default_rng(0)
    W, ytrue = simulate(N_NAMES, rng)                 # truth-known cross-section
    hist = np.diff(W, axis=-1, prepend=W[:, :1])      # residual-return windows

    t0 = time.perf_counter()
    net = train_nfe(seed=0)
    t_train = time.perf_counter() - t0
    print(f"  one-time training: {t_train:8.1f} s")

    t_amort = med_time(lambda: sig_nfe(hist, net))
    print(f"  amortized posterior + sizing, whole cross-section: "
          f"{t_amort*1e3:8.1f} ms")

    t_plug = med_time(lambda: plugin_fit(hist))
    print(f"  plug-in OLS refit + sizing (no distribution):      "
          f"{t_plug*1e3:8.1f} ms")

    t0 = time.perf_counter()
    post, ys = pf_grid_posterior(W)
    t_pf = time.perf_counter() - t0
    print(f"  unamortized grid x particle-filter posterior:      "
          f"{t_pf:8.1f} s   (grid {GRID} x {NPART} particles)")

    # sanity: both posteriors must track the truth for the timing to count.
    # The deployed sizing consumes 64 posterior draws, so that estimate
    # carries Monte-Carlo noise; the analytic mixture mean is the
    # like-for-like fidelity comparison against the grid posterior's mean.
    import torch
    from scipy.stats import spearmanr
    from nfe import featurize, nfe_kappa_samples
    mean_pf = post @ ys
    ks = nfe_kappa_samples(net, np.cumsum(hist, axis=-1))
    mean_64 = np.log2(np.log(2) / np.clip(ks, 1e-12, None)).mean(1)
    with torch.no_grad():
        o = net["mdn"](net["gru"](torch.tensor(featurize(
            np.cumsum(hist, axis=-1))))[0][:, -1]).view(-1, 3, 5)
        wmix = torch.softmax(o[:, 0], -1)
        mean_an = (wmix * o[:, 1]).sum(-1).numpy()
    r_pf = float(spearmanr(mean_pf, ytrue).statistic)
    r_64 = float(spearmanr(mean_64, ytrue).statistic)
    r_an = float(spearmanr(mean_an, ytrue).statistic)
    print(f"\n  sanity (Spearman vs truth over {N_NAMES} draws): "
          f"pf {r_pf:+.3f} | nfe analytic {r_an:+.3f} | nfe 64-draw {r_64:+.3f}"
          + ("" if r_pf > 0 else "   ** PF NOT SANE — timing void **"))

    out = ROOT / "results" / "run_260728" / "nfe_runtime.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(
        cpu=cpu, torch_threads=torch.get_num_threads(),
        n_names=N_NAMES, L=L, nrep=NREP, grid=GRID, npart=NPART,
        train_s=t_train, amortized_ms=t_amort * 1e3,
        plugin_ms=t_plug * 1e3, pf_s=t_pf,
        ratio_pf_over_amortized=t_pf / t_amort,
        spearman_pf=r_pf, spearman_nfe_analytic=r_an,
        spearman_nfe_64draw=r_64), indent=1))
    print(f"  artifact: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
