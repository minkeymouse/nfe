"""What is the NFE overlay worth where the prior is true, and what survives
deployment? The simplified 2x2: substrates {synthetic, KRX} x policies
{base, base+NFE}.

    uv run python experiments/exp_value_2x2.py [--smoke]

[exploration] (2026-07-24). PRE-REGISTRATION — frozen before the first full
run. Design: add a synthetic substrate, simplify the policy axis to
base / base+NFE, and measure what the overlay is worth on each substrate.

WHY THIS DESIGN. On KRX the overlay pays in risk, not return, and two of
our own measurements say that is what SHOULD happen there: the
identification ladder (no estimator escapes the 0.61-0.65 band at L=60)
and the honesty table (conditional resolution ~0 on real data). The
missing cell is the world where the prior IS the truth, where the
posterior is correctly specified and uncertainty-shaped sizing must show
up as Sharpe if it is worth anything. run_260722/SIM_LADDER measured
this once in a different harness; this re-measures it under the paper's
own policies so both substrates share one table.

SYNTHETIC SUBSTRATE. N=120 names, T=2,036 days (era length), each an
independent switching-OU spread: kappa alternates between fast
(HL ~ logU(5,20)d) and slow (HL ~ logU(60,250)d) states with geometric
spells of mean 250d; level m=0; innovations are the ADMITTED law (SV +
centred lognormal) at phi_vol=0.28, s_log=0.12 (era-2 gate-calibrated).
The NFE arm is the paper's own train_nfe at that calibration, trained on
prior draws only.

ARMS (all on the same paths; base and overlay identical to the paper's)
  base            w ~ -clip(z, +-3)                        (sig_fade)
  base+NFE        eq:sizing with moments under q(.|w)      (sig_nfe)
  base+prior      eq:sizing with kappa sampled from the PRIOR, not the
                  posterior — same average shrinkage, no conditioning.
                  This is the attribution null.
  base+oracle     eq:sizing with the true instantaneous kappa — the
                  ceiling; identically zero variance term.
  random          turnover-matched random benchmark        (floor)

KRX SUBSTRATE. No rerun: the era-2 book P&L dump of the full E3 run
(results/run_260723/book_pnl_era2.npz) already contains base and
base+NFE on both residual streams; this script recomputes the PAIRED
hold-level gaps from it.

FROZEN CONVENTIONS. As the paper: L=60, H=20 non-overlapping holds,
dollar-neutral unit gross, costs 5bp/unit turnover + 1bp shorting,
ann = sqrt(246); paired bootstrap over holds (2,000 resamples), the
same resample indices for every arm.

GATES (registered)
  G1  VALUE: synthetic SR(base+NFE) - SR(base) > 2 paired SE.
  G2  ATTRIBUTION: synthetic SR(base+prior) - SR(base) NOT > 2 paired
      SE; if the unconditional-shrinkage null also clears, the G1 gain
      is retracted from "conditional information" to "deleveraging".
  G3  COHERENCE: SR(base+oracle) >= SR(base+NFE) >= SR(base) ordering.
  KRX side is descriptive (numbers already in Table 1); recorded
  expectation: SR gap within 2 paired SE, risk metrics improve.

CONFIRMATION (registered 2026-07-24 BEFORE the confirm run; after the
exploration pass read G1 +0.31 +- 0.19 [FAIL at 2 SE], G2 quiet, G3
pass — an effect of the expected size but an under-powered sample of
~99 holds). --confirm re-runs the synthetic substrate on FIVE fresh
panels (seeds 101-105, disjoint from the exploration seed 7), the SAME
trained filter, pooling hold-level P&L across panels (~495 holds);
gates G1-G3 UNCHANGED; the outcome is reported whichever way it falls.
This is a powered fresh-seed confirmation, not resampling-to-
significance: the sample is fixed in advance and disjoint.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT))

from nfe import H, L, sig_nfe, train_nfe
from exp_e3_book import metrics, sig_fade

SMOKE = "--smoke" in sys.argv
CONFIRM = "--confirm" in sys.argv
N_NAMES = 30 if SMOKE else 120
T_DAYS = 600 if SMOKE else 2_036
NBOOT = 200 if SMOKE else 2_000
PANEL_SEEDS = (101, 102, 103, 104, 105) if CONFIRM else (7,)
PHI_VOL, S_LOG = 0.28, 0.12          # era-2 gate calibration
SPELL = 250.0
ANN = np.sqrt(246.0)
PER_YEAR = 246.0 / H


def sim_switch_panel(N, T, rng, phi_vol=PHI_VOL, s_log=S_LOG):
    """Independent switching-OU spreads under the admitted innovation law.
    Returns E (T, N) increments and KAP (T, N) true instantaneous kappa."""
    hl_f = np.exp(rng.uniform(np.log(5), np.log(20), N))
    hl_s = np.exp(rng.uniform(np.log(60), np.log(250), N))
    burn = 200
    m_ln = np.exp(s_log ** 2 / 2)
    sd_ln = np.sqrt((np.exp(s_log ** 2) - 1) * np.exp(s_log ** 2))
    slow = rng.random(N) < 0.5
    x = np.zeros(N); lv = np.zeros(N)
    X = np.empty((T, N)); KAP = np.empty((T, N))
    for t in range(burn + T):
        flip = rng.random(N) < 1.0 / SPELL
        slow = np.where(flip, ~slow, slow)
        kap = np.where(slow, np.log(2) / hl_s, np.log(2) / hl_f)
        lv = phi_vol * lv + 0.5 * np.sqrt(1 - phi_vol ** 2) * \
            rng.standard_normal(N)
        e = (np.exp(s_log * rng.standard_normal(N)) - m_ln) / sd_ln
        x = x + kap * (0.0 - x) + 0.01 * np.exp(lv) * e
        if t >= burn:
            X[t - burn] = x; KAP[t - burn] = kap
    E = np.diff(np.vstack([np.zeros(N), X]), axis=0)
    return E, KAP


def book_t(E, policy_t, rng=None, floor_hold=False):
    """exp_e3_book.book with the rebalance time passed to the policy
    (needed by the oracle arm); otherwise identical conventions."""
    from exp_e3_book import C_SHORT, C_TURN
    T, N = E.shape
    pnl, tov = [], []
    prev = np.zeros(N); w = np.zeros(N)
    for t in range(L, T):
        if (t - L) % H == 0:
            hist = E[t - L:t].T
            w = rng.standard_normal(N) if floor_hold else policy_t(hist, t)
            w = w - w.mean()
            n = np.abs(w).sum()
            w = w / n if n > 1e-12 else w
        p = w @ E[t]
        turn = np.abs(w - prev).sum()
        pnl.append(p - C_TURN * turn - C_SHORT * np.maximum(-w, 0).sum())
        tov.append(turn)
        prev = w
    return np.array(pnl), np.array(tov)


def holds(p):
    n = len(p) // H
    return p[:n * H].reshape(n, H).sum(1)


def sr_holds(hp):
    return float(hp.mean() / (hp.std(ddof=1) + 1e-15) * np.sqrt(PER_YEAR))


def paired_gaps_h(hp, rng, nboot=NBOOT):
    """dict name->hold-level pnl -> (sr, gap-vs-base, paired SE) per arm."""
    n = len(hp["base"])
    idx = rng.integers(0, n, (nboot, n))
    out = {}
    for k, v in hp.items():
        gap = sr_holds(v) - sr_holds(hp["base"])
        boots = np.array([sr_holds(v[i]) - sr_holds(hp["base"][i])
                          for i in idx])
        out[k] = (sr_holds(v), gap, float(boots.std(ddof=1)))
    return out


def main():
    tag = "smoke" if SMOKE else ("confirmation" if CONFIRM else "exploration")
    print(f"[{tag}] 2x2 value of the overlay"
          f"  (N={N_NAMES}, T={T_DAYS}, panels {PANEL_SEEDS}, admitted "
          f"innovations phi_vol={PHI_VOL}, s_log={S_LOG})\n", flush=True)

    # ---------------- synthetic substrate ----------------
    net = train_nfe(seed=0, phi_vol=PHI_VOL, s_log=S_LOG)
    rng0 = np.random.default_rng(7)
    prior_hl = np.exp(rng0.uniform(np.log(3), np.log(400), 256))
    prior_k = np.log(2) / prior_hl

    def z_of(hist):
        X = np.cumsum(hist, axis=-1)
        m, sd = X.mean(-1), X.std(-1) + 1e-12
        return np.clip((X[:, -1] - m) / sd, -3, 3)

    def sig_prior(hist, t):
        z = z_of(hist)
        a, b = np.exp(-prior_k * H), np.exp(-2 * prior_k * H)
        num = (1.0 - a).mean()
        den = (1.0 - b).mean() + z ** 2 * a.var()
        return -z * num / (den + 1e-12)

    names5 = ["base", "base+NFE", "base+prior", "base+oracle", "random"]
    hold_pool = {k: [] for k in names5}
    daily_pool = {k: [] for k in names5}
    tov_pool = {k: [] for k in names5}
    for ps in PANEL_SEEDS:
        rng = np.random.default_rng(ps)
        E, KAP = sim_switch_panel(N_NAMES, T_DAYS, rng)

        def sig_oracle(hist, t, _K=KAP):
            z = z_of(hist)
            a = np.exp(-_K[t] * H)
            return -z * (1.0 - a) / ((1.0 - a ** 2) + 1e-12)

        arms = {
            "base": lambda h, t: sig_fade(h),
            "base+NFE": lambda h, t: sig_nfe(h, net),
            "base+prior": sig_prior,
            "base+oracle": sig_oracle,
        }
        for name, pol in arms.items():
            p, tv = book_t(E, pol)
            daily_pool[name].append(p); tov_pool[name].append(tv)
            hold_pool[name].append(holds(p))
        p, tv = book_t(E, None, rng=np.random.default_rng(99 + ps),
                       floor_hold=True)
        daily_pool["random"].append(p); tov_pool["random"].append(tv)
        hold_pool["random"].append(holds(p))
        print(f"  panel seed {ps} done", flush=True)

    pnls = {k: np.concatenate(v) for k, v in daily_pool.items()}
    tovs = {k: np.concatenate(v) for k, v in tov_pool.items()}
    hp = {k: np.concatenate(v) for k, v in hold_pool.items()}

    print("\n  === SYNTHETIC (admitted process, truth known; "
          f"{len(hp['base'])} holds) ===", flush=True)
    print(f"  {'arm':13s} {'SR':>6s} {'mu%':>6s} {'sd%':>5s} {'MDD':>6s}"
          f" {'turn':>5s}   {'dSR vs base':>12s}", flush=True)
    res = paired_gaps_h(hp, np.random.default_rng(11))
    for name in names5:
        m = metrics(pnls[name], tovs[name], ANN)
        s, g, se = res[name]
        tail = "" if name == "base" else f"{g:+.2f} +- {se:.2f}"
        print(f"  {name:13s} {m['sharpe']:6.2f} {m['mu']:6.1f} {m['sd']:5.1f}"
              f" {m['mdd']*100:5.1f}% {m['turn']:5.3f}   {tail:>12s}",
              flush=True)

    g1, se1 = res["base+NFE"][1], res["base+NFE"][2]
    g2, se2 = res["base+prior"][1], res["base+prior"][2]
    go = res["base+oracle"][1]
    print(f"\n  G1 value        dSR(base+NFE)   {g1:+.2f} +- {se1:.2f}  "
          f"[{'PASS' if g1 > 2 * se1 else 'FAIL'}]", flush=True)
    print(f"  G2 attribution  dSR(base+prior) {g2:+.2f} +- {se2:.2f}  "
          f"[{'PASS (null quiet)' if g2 <= 2 * se2 else 'FAIL (null fires)'}]",
          flush=True)
    print(f"  G3 coherence    oracle {go:+.2f} >= NFE {g1:+.2f} >= 0: "
          f"[{'PASS' if go >= g1 >= 0 else 'FAIL'}]", flush=True)

    # ---------------- KRX substrate (from the E3 dump) ----------------
    npz = ROOT / "results" / "run_260723" / "book_pnl_era2.npz"
    if npz.exists() and not SMOKE:
        d = np.load(npz)
        print("\n  === KRX era 2 (from the full E3 run; descriptive) ===",
              flush=True)
        for stream, b, o in [("PCA", "PCA", "PCA+NFE"),
                             ("AFM", "AFM+fade", "AFM+NFE")]:
            pb, po = d[b], d[o]
            hb, ho = holds(pb), holds(po)
            n = len(hb)
            idx = np.random.default_rng(13).integers(0, n, (NBOOT, n))
            boots = np.array([sr_holds(ho[i]) - sr_holds(hb[i]) for i in idx])
            mb = metrics(pb, np.zeros_like(pb), ANN)
            mo = metrics(po, np.zeros_like(po), ANN)
            print(f"  {stream}: base SR {mb['sharpe']:.2f} -> +NFE "
                  f"{mo['sharpe']:.2f}  dSR {sr_holds(ho)-sr_holds(hb):+.2f}"
                  f" +- {boots.std(ddof=1):.2f} | sd {mb['sd']:.1f}->"
                  f"{mo['sd']:.1f} MDD {mb['mdd']*100:.1f}->"
                  f"{mo['mdd']*100:.1f} CVaR {mb['cvar5']*1e4:.0f}->"
                  f"{mo['cvar5']*1e4:.0f}bp", flush=True)


if __name__ == "__main__":
    main()
