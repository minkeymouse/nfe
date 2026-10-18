"""Cross-sectional regime detection on Binance + KRX daily cross-sections.

    uv run python experiments/exp_cross_sectional.py [binance|krx|both] [--smoke]

[exploration] (2026-07-21). Per asset i, residual spread z_{i,t} (market-neutral
cumulative residual, standardized). Task: causally nowcast the BROKEN state --
does the stretched residual FAIL to revert over the next H days
(|z_{t+H}| > |z_t|)? A realized (hindsight) label; truth exists.

Contestants (same label, different feature extractor):
  temporal-only   gated recurrent filter over each asset's z-sequence
                  (selective = Mamba family), no cross-section.
  cross-sectional temporal filter + multi-head attention ACROSS assets at each
                  time (non-causal in the asset dim, causal in time).
  AFM             attention-factor encoder (softmax(Q Xtilde^T) factors over
                  firm characteristics, ridge loadings, residual projection --
                  Epstein et al. 2025) with the trading head REPLACED by a
                  regime head over the AFM residual.
Train 60% of dates / test 40% (H-day embargo). Complete-history names only
(survivorship caveat). Metrics: AUC/AUPRC/MCC/Brier + per-asset flicker/jitter.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import torch
import torch.nn as nn

SMOKE = "--smoke" in sys.argv
WHICH = next((a for a in sys.argv[1:] if a in ("binance", "krx", "both")), "both")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
L, H = 60, 20
EPOCHS = 4 if SMOKE else 25


def load_binance():
    frames = {}
    for p in sorted(Path("data/raw/binance").glob("*.parquet")):
        if p.stem == "manifest":
            continue
        d = pd.read_parquet(p)[["date", "close"]].dropna()
        frames[p.stem] = d.set_index("date")["close"].astype(float)
    return pd.DataFrame(frames).sort_index()


def load_krx():
    """The KRX panel. NFE_PANEL=krx (default): the daily closing prices
    behind the paper's numbers (KOSPI common stocks ranked by median daily
    trading value over the two years to 2026-06, top 200, daily closes from
    2010-01-04); not redistributed. NFE_PANEL=public: a proxy panel built
    from public daily closes (same layout; not the paper's universe).
    NFE_PANEL=vendor2: the same universe rule and layout rebuilt from a
    second price source (daily closes 2010-01-04..2026-07-20); not
    redistributed; the robustness check in the paper's Appendix B used
    it."""
    import os
    src = os.environ.get("NFE_PANEL", "krx")
    base = {"krx": ("data/raw/krx", "manifest.parquet"),
            "public": ("data/raw/krx_public", "manifest.parquet"),
            "vendor2": ("data/raw/krx_vendor2", "manifest.parquet")}[src]
    man = pd.read_parquet(f"{base[0]}/{base[1]}")
    frames = {}
    for sh in man["shcode"]:
        p = Path(f"{base[0]}/{sh}.parquet")
        if not p.exists():
            continue
        d = pd.read_parquet(p)[["date", "close"]].dropna()
        d["date"] = pd.to_datetime(d["date"].astype(str), format="%Y%m%d")
        frames[sh] = d.set_index("date")["close"].astype(float)
    return pd.DataFrame(frames).sort_index()


def fixed_panel(raw, target_n):
    cnt = raw.notna().sum().sort_values(ascending=False)
    return raw[list(cnt.index[:target_n])].dropna()


def build(panel):
    """-> z (T,N), r (T,N returns), C (T,N,M characteristics), broken (T,N)."""
    panel = panel.dropna(axis=1, how="any")
    lp = np.log(panel.to_numpy(float))
    r = np.diff(lp, axis=0)
    e = r - r.mean(1, keepdims=True)
    X = np.cumsum(e, axis=0)
    T, N = X.shape
    z = np.full_like(X, np.nan)
    for t in range(L, T):
        w = X[t - L:t]
        z[t] = (X[t] - w.mean(0)) / (w.std(0) + 1e-8)
    # GENERATIVE regime (theta-defined), identified with hindsight: fit the OU
    # (AR(1)) on a FORWARD window and read the reversion speed kappa. broken =
    # the residual is NOT dynamically mean-reverting (half-life > 60d, i.e.
    # kappa below ln2/60) -> drift/random-walk; reverting otherwise. A parameter
    # of the process, not a single realized outcome, so it is not the fade P&L.
    FWD = 2 * H
    kth = np.log(2) / 60.0
    broken = np.full_like(X, np.nan)
    for t in range(L, T - FWD):
        Xw = X[t:t + FWD]
        dX = np.diff(Xw, axis=0); Xl = Xw[:-1]
        mx = Xl.mean(0)
        b = ((Xl - mx) * (dX - dX.mean(0))).sum(0) / (((Xl - mx) ** 2).sum(0) + 1e-12)
        broken[t] = ((-b) < kth).astype(float)         # weak/no forward reversion
    # OU-OPTIMAL fade signal (closed-form given estimated theta): the expected
    # H-day reversion = -s * (1 - 2^{-H/HL}), s the standardized deviation,
    # HL the AR(1) half-life estimated on the window. This is the optimal OU
    # solution; models supply the regime posterior p(z|x) that modulates it.
    g = np.full_like(X, np.nan)
    for t in range(L, T):
        Xw = X[t - L:t]
        dX = np.diff(Xw, axis=0); Xl = Xw[:-1]
        mx = Xl.mean(0)
        b = ((Xl - mx) * (dX - dX.mean(0))).sum(0) / (((Xl - mx) ** 2).sum(0) + 1e-12)
        kap = np.clip(-b, 1e-3, None)
        hl = np.log(2) / kap
        g[t] = -z[t] * (1 - 2.0 ** (-H / np.maximum(hl, 1e-3)))
    rdf = pd.DataFrame(r)
    C = np.stack([r, rdf.rolling(5).sum().to_numpy(),
                  rdf.rolling(20).sum().to_numpy(),
                  rdf.rolling(20).std().to_numpy(),
                  np.nan_to_num(z)], axis=-1)              # (T,N,5)
    return z, r.astype(np.float32), np.nan_to_num(C).astype(np.float32), broken, g


# ------------------------------------------------------------ contestants
class CrossSec(nn.Module):
    """Temporal selective filter (+ optional cross-sectional attention)."""
    def __init__(s, nin, d=32, heads=4, use_cross=True):
        super().__init__()
        s.temporal = nn.GRU(nin, d, batch_first=True)
        s.use_cross = use_cross
        if use_cross:
            s.attn = nn.MultiheadAttention(d, heads, batch_first=True)
            s.ln = nn.LayerNorm(d)
        s.head = nn.Sequential(nn.Linear(d * (2 if use_cross else 1), d),
                               nn.ReLU(), nn.Linear(d, 1))

    def forward(s, X):                                     # X: (N, T, nin)
        Hs, _ = s.temporal(X)
        if not s.use_cross:
            return s.head(Hs).squeeze(-1)
        N, T, d = Hs.shape
        C = torch.empty_like(Hs)
        for a in range(0, T, 256):
            h = Hs[:, a:a + 256].transpose(0, 1)
            att, _ = s.attn(h, h, h)
            C[:, a:a + 256] = s.ln(att).transpose(0, 1)
        return s.head(torch.cat([Hs, C], -1)).squeeze(-1)


class AFMRegime(nn.Module):
    """AFM attention as the cross-sectional module + regime head. On the SHARED
    z representation (fair vs the other contestants), AFM's characteristic-based
    softmax(Q Xtilde^T) factor weights (Epstein 2025) aggregate the temporal
    states into K factors and back to per-asset context, replacing the generic
    attention of the cross-sectional model with AFM's factor construction."""
    def __init__(s, m_char, k=4, d=32):
        super().__init__()
        s.temporal = nn.GRU(1, d, batch_first=True)        # over shared z
        s.embed = nn.Linear(m_char, d)
        s.query = nn.Parameter(0.1 * torch.randn(k, d))
        s.ln = nn.LayerNorm(d)
        s.head = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU(), nn.Linear(d, 1))

    def forward(s, Xz, C):                                 # Xz:(N,T,1)  C:(T,N,M)
        Hs, _ = s.temporal(Xz)                             # (N,T,d) temporal state
        Xt = s.embed(C)                                    # (T,N,d) embedded chars
        sc = torch.einsum("kd,tnd->tkn", s.query, Xt) / (Xt.shape[-1] ** 0.5)
        omega = torch.softmax(sc, -1)                      # (T,K,N) AFM factor weights
        Ht = Hs.transpose(0, 1)                            # (T,N,d)
        Ft = torch.einsum("tkn,tnd->tkd", omega, Ht)       # (T,K,d) factor states
        ctx = torch.einsum("tkn,tkd->tnd", omega, Ft)      # (T,N,d) per-asset factor context
        ctx = s.ln(ctx).transpose(0, 1)                    # (N,T,d)
        return s.head(torch.cat([Hs, ctx], -1)).squeeze(-1)


def masks(z, broken):
    T, N = z.shape
    y = broken.T.astype(np.float32)
    split = int(T * 0.6)
    valid = ~np.isnan(broken.T) & ~np.isnan(z.T)
    trm = valid.copy(); trm[:, split - 2 * H:] = False     # embargo the forward-label lookahead
    tem = valid.copy(); tem[:, :split] = False
    return y, trm, tem


def train_eval(net, forward_inputs, y, trm, seed):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    bce = nn.BCEWithLogitsLoss()
    Y = torch.tensor(y, device=DEV); M = torch.tensor(trm, device=DEV)
    net.train()
    for _ in range(EPOCHS):
        logit = net(*forward_inputs)
        loss = bce(logit[M], Y[M])
        opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        return torch.sigmoid(net(*forward_inputs)).cpu().numpy()


COST = 1e-3                                               # fill-pessimistic, /unit turnover


def book(g, X, Xm, pT, tmask):
    """Detect-then-trade, HORIZON-ALIGNED: rebalance every H days, hold H days
    (non-overlapping), so realized P&L = w . (residual level change over H) is
    on the SAME horizon as the H-day regime label and OU signal. Position =
    OU-optimal fade g modulated by (1 - p_broken); dollar-neutral, gross=1;
    turnover cost. Returns per-rebalance net returns, market H-day returns, and
    mean turnover."""
    T, N = g.shape
    w_prev = np.zeros(N)
    rets, mkt, turns = [], [], []
    for t0 in range(0, T - H, H):                        # non-overlapping H-day holds
        m = tmask[t0] & ~np.isnan(g[t0])
        if m.sum() < 5:
            continue
        sig = np.where(m, np.clip(np.nan_to_num(g[t0]), -3, 3), 0.0)
        w = sig * np.where(m, 1.0 - pT[t0], 0.0)
        w = w / (np.abs(w).sum() + 1e-12)
        dX = np.where(m, X[t0 + H] - X[t0], 0.0)         # residual reversion over hold
        rets.append(float(np.nansum(w * dX)) - COST * float(np.abs(w - w_prev).sum()))
        mkt.append(float(Xm[t0 + H] - Xm[t0]))
        turns.append(float(np.abs(w - w_prev).sum())); w_prev = w
    return np.array(rets), np.array(mkt), float(np.mean(turns) if turns else 0)


def fin_metrics(rets, mkt, ann):
    r = np.asarray(rets, float)
    if len(r) < 5:
        return {k: float("nan") for k in
                ("ann_ret", "sharpe", "sortino", "mdd", "calmar",
                 "jensen_a", "treynor", "hit", "turn")}
    mean, sd = r.mean(), r.std() + 1e-12
    dn = r[r < 0]; dsd = (dn.std() if len(dn) > 1 else sd) + 1e-12
    cum = np.cumsum(r); mdd = float((np.maximum.accumulate(cum) - cum).max())
    annret = mean * ann
    beta = float(np.cov(r, mkt)[0, 1] / (np.var(mkt) + 1e-12))
    alpha = (mean - beta * mkt.mean()) * ann
    return {"ann_ret": annret, "sharpe": mean / sd * np.sqrt(ann),
            "sortino": mean / dsd * np.sqrt(ann), "mdd": mdd,
            "calmar": annret / (mdd + 1e-12),
            "jensen_a": alpha,
            "treynor": (annret / beta) if abs(beta) > 1e-3 else float("nan"),
            "hit": float((r > 0).mean())}


def run_substrate(name, panel, ann):
    z, r, C, broken, g = build(panel)
    T, N = z.shape
    e = r - r.mean(1, keepdims=True)                      # residual returns
    X = np.cumsum(e, 0)                                   # residual level
    Xm = np.cumsum(r.mean(1))                             # market cumulative
    y, trm, tem = masks(z, broken)
    temT = tem.T                                          # (T,N) test mask
    Xz = torch.tensor(np.nan_to_num(z).T[:, :, None].astype(np.float32), device=DEV)
    Ct = torch.tensor(C, device=DEV)
    print(f"\n=== {name.upper()}: {N} names, {T} days (ann={ann}), "
          f"broken(theta) rate {np.nanmean(broken):.2f} ===", flush=True)
    print(f"  {'strategy':<16}{'annRet':>8}{'Sharpe':>7}{'Sortino':>8}{'MDD':>7}"
          f"{'Calmar':>7}{'Jensen a':>9}{'Treynor':>8}{'hit':>6}{'turn':>6}")

    # gates: raw fade (no regime), each model's p(broken), oracle (true broken)
    gates = {"fade (no gate)": np.zeros((N, T))}
    contestants = [
        ("temporal-only", lambda: CrossSec(1, use_cross=False).to(DEV), (Xz,)),
        ("cross-sectional", lambda: CrossSec(1, use_cross=True).to(DEV), (Xz,)),
        ("AFM", lambda: AFMRegime(C.shape[-1]).to(DEV), (Xz, Ct)),
    ]
    for tag, make, inputs in contestants:
        gates["+" + tag] = train_eval(make(), inputs, y, trm, seed=0)   # (N,T)
    gates["+oracle"] = np.nan_to_num(broken).T                          # (N,T)

    out = {}
    for tag, p in gates.items():
        rets, mkt, turn = book(g, X, Xm, p.T, temT)
        m = fin_metrics(rets, mkt, ann / H); m["turn"] = turn   # per-rebalance -> annualize
        out[tag] = m
        print(f"  {tag:<16}{m['ann_ret']*100:>7.1f}%{m['sharpe']:>7.2f}"
              f"{m['sortino']:>8.2f}{m['mdd']*100:>6.1f}%{m['calmar']:>7.2f}"
              f"{m['jensen_a']*100:>8.1f}%{m['treynor']:>8.2f}"
              f"{m['hit']*100:>5.0f}%{m['turn']:>6.2f}", flush=True)
    return out


def main():
    OUT = Path("results/run_260722"); OUT.mkdir(parents=True, exist_ok=True)
    print(f"[exploration] cross-sectional detect-then-trade on {WHICH}"
          f"{' [SMOKE]' if SMOKE else ''} (dev={DEV}); after-cost book metrics",
          flush=True)
    res = {}
    if WHICH in ("binance", "both"):
        res["binance"] = run_substrate("binance", fixed_panel(load_binance(), 45), 365)
    if WHICH in ("krx", "both"):
        res["krx"] = run_substrate("krx", fixed_panel(load_krx(), 150), 252)
    tag = "smoke" if SMOKE else "explore"
    (OUT / f"cross_sectional_{tag}.json").write_text(json.dumps(res, indent=1))
    print(f"\n[exploration] artifact: {OUT}/cross_sectional_{tag}.json")


if __name__ == "__main__":
    main()
