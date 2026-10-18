"""Point-in-time daily fundamental features for the E3 KRX panel.

    uv run python experiments/prep_fundamentals.py

(2026-07-23). Inputs: DART statements (data/raw/dart/panel_fundamentals.parquet),
DART share counts (data/raw/dart/shares.parquet), unadjusted daily closes
(data/raw/krx_raw/<shcode>.parquet, same layout as the panel) and the
panel's adjusted closes (data/raw/krx/). Output: data/processed/fundamentals_daily.npz
with F (T, N, 6) aligned to exp_e3_book.load_panel's dates and name order.

FEATURES (all direction-blind: level/quality/valuation, no momentum/skew —
the anchor experiment's constraint):
  roe      ttm net income / equity            quality
  de       total debt / equity                leverage
  margin   ttm net income / ttm revenue       quality
  rev_yoy  ttm revenue / ttm revenue 1y ago   growth (scale-free)
  bp       equity / mktcap(t)                 value (book-to-price)
  ep       ttm net income / mktcap(t)         value (earnings yield)

POINT-IN-TIME RULES (the reason DART was chosen over the KRX portal):
  - A statement becomes usable on the first panel date STRICTLY AFTER its
    disclosure date (rcept). Later disclosures supersede earlier ones.
  - Quarterly flows: Q1..Q3 statements carry the quarter alone (verified on
    Samsung 2016: Q1+Q2+Q3 = 15.6T vs annual 22.7T); the annual statement is
    full-year, so Q4 flow = annual - (Q1+Q2+Q3), NaN if any quarter missing.
    TTM = sum of 4 consecutive quarterly flows.
  - Market cap: shares are AS OF period end (Samsung Q1-2018 still shows the
    pre-split count 11 days post-split), so mcap0 = raw_close(period end) x
    shares(period end) -- contemporaneous, split-consistent -- then
    mktcap(t) = mcap0 x adj(t)/adj(period end): adjusted returns carry every
    capital event forward, so bp/ep update daily without raw daily data.
  - equity <= 0 -> roe/de/bp NaN.
NaN policy: left in place; consumers impute (rank layers use the daily
cross-sectional median).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments"))

FEATS = ["roe", "de", "margin", "rev_yoy", "bp", "ep", "size"]
# size (added 2026-07-24, for the FF3-style characteristic-residual arm):
# log point-in-time market capitalization, the same split-consistent
# mcap used as the bp/ep denominator.
PE_MD = {1: "0331", 2: "0630", 3: "0930", 4: "1231"}


def load_series(dirname, sh):
    p = ROOT / "data" / "raw" / dirname / f"{sh}.parquet"
    if not p.exists():
        return None
    d = pd.read_parquet(p)[["date", "close"]].dropna()
    d["date"] = pd.to_datetime(d["date"].astype(str), format="%Y%m%d")
    return d.set_index("date")["close"].astype(float)


def close_at_or_before(s, day, max_back=15):
    """Last close <= day, within max_back calendar days."""
    if s is None:
        return None, None
    w = s.loc[day - pd.Timedelta(days=max_back):day]
    if len(w) == 0:
        return None, None
    return float(w.iloc[-1]), w.index[-1]


def statements(fund, shares, adj, raw, sh):
    """Derive per-statement feature bases for one name; rows sorted by rcept."""
    f = fund[fund.shcode == sh].copy()
    if len(f) == 0:
        return None
    f["qi"] = f.year * 4 + f.q - 1
    f = f.sort_values("qi").drop_duplicates("qi", keep="last").set_index("qi")
    # quarterly flows: Q4 = annual - (Q1+Q2+Q3), per metric independently
    flow = {}
    for m in ("ni", "revenue"):
        v = f[m] if m in f else pd.Series(dtype=float)
        fl = pd.Series(np.nan, index=f.index)
        for qi in f.index:
            q = qi % 4 + 1
            if q < 4:
                fl[qi] = v.get(qi, np.nan)
            else:
                prior = [v.get(qi - k, np.nan) for k in (1, 2, 3)]
                a = v.get(qi, np.nan)
                fl[qi] = a - np.nansum(prior) if (
                    np.isfinite(a) and all(np.isfinite(p) for p in prior)
                    and all(qi - k in f.index for k in (1, 2, 3))) else np.nan
        flow[m] = fl
    ttm = {m: pd.Series({qi: (flow[m].reindex([qi - k for k in range(4)]).sum()
                              if flow[m].reindex([qi - k for k in range(4)])
                              .notna().all() else np.nan)
                         for qi in f.index}) for m in flow}
    sh_rows = shares[shares.shcode == sh].copy()
    sh_rows["qi"] = sh_rows.year * 4 + sh_rows.q - 1
    sh_map = dict(zip(sh_rows.qi, sh_rows.shares))
    rows = []
    for qi, r in f.iterrows():
        year, q = qi // 4, qi % 4 + 1
        eq = r.get("equity", np.nan)
        if not np.isfinite(eq) or eq <= 0:
            eq = np.nan
        tni, trv = ttm["ni"].get(qi, np.nan), ttm["revenue"].get(qi, np.nan)
        trv4 = ttm["revenue"].get(qi - 4, np.nan)
        pe = pd.Timestamp(f"{year}{PE_MD[q]}")
        mcap0 = adj_pe = np.nan
        n_sh = sh_map.get(qi, np.nan)
        if np.isfinite(n_sh):
            rc, rc_day = close_at_or_before(raw, pe)
            if rc is not None:
                ac, _ = close_at_or_before(adj, rc_day, max_back=0)
                if ac is not None:
                    mcap0, adj_pe = n_sh * rc, ac
        rows.append({
            "rcept": pd.Timestamp(str(r["rcept"])), "qi": qi,
            "roe": tni / eq if np.isfinite(tni) else np.nan,
            "de": r.get("debt", np.nan) / eq,
            "margin": tni / trv if np.isfinite(tni) and np.isfinite(trv)
                      and abs(trv) > 0 else np.nan,
            "rev_yoy": trv / trv4 - 1 if np.isfinite(trv) and np.isfinite(trv4)
                       and trv4 > 0 else np.nan,
            "equity": eq, "ttm_ni": tni, "mcap0": mcap0, "adj_pe": adj_pe})
    st = pd.DataFrame(rows).sort_values(["rcept", "qi"])
    return st.drop_duplicates("rcept", keep="last")   # later period wins same-day


def main():
    from exp_cross_sectional import load_krx, fixed_panel
    P = fixed_panel(load_krx(), 120)
    dates, names = P.index[1:], [str(c) for c in P.columns]
    T, N = len(dates), len(names)
    fund = pd.read_parquet(ROOT / "data/raw/dart/panel_fundamentals.parquet")
    shares = pd.read_parquet(ROOT / "data/raw/dart/shares.parquet")
    F = np.full((T, N, len(FEATS)), np.nan)
    miss = []
    for j, sh in enumerate(names):
        adj, raw = load_series("krx", sh), load_series("krx_raw", sh)
        st = statements(fund, shares, adj, raw, sh)
        if st is None or len(st) == 0:
            miss.append(sh)
            continue
        # effective from the first panel date strictly after disclosure
        eff = np.searchsorted(dates.values, st["rcept"].values, side="right")
        keep = eff < T
        st, eff = st[keep], eff[keep]
        idx = np.full(T, -1)
        for k, e in enumerate(eff):
            idx[e:] = k
        have = idx >= 0
        for fi, feat in enumerate(["roe", "de", "margin", "rev_yoy"]):
            F[have, j, fi] = st[feat].values[idx[have]]
        adj_t = adj.reindex(dates).ffill().to_numpy()
        mcap = st["mcap0"].values[idx[have]] * adj_t[have] / \
            st["adj_pe"].values[idx[have]]
        F[have, j, 4] = st["equity"].values[idx[have]] / mcap
        F[have, j, 5] = st["ttm_ni"].values[idx[have]] / mcap
        F[have, j, 6] = np.log(np.maximum(mcap, 1.0))
        if j % 20 == 0:
            print(f"  {j+1}/{N} {sh}: {len(st)} statements", flush=True)
    out = ROOT / "data/processed/fundamentals_daily.npz"
    np.savez_compressed(out, F=F, dates=dates.strftime("%Y%m%d").to_numpy(),
                        names=np.array(names), feats=np.array(FEATS))
    cov = np.isfinite(F).mean(axis=(0, 1))
    e2 = dates >= pd.Timestamp("2018-01-01")
    cov2 = np.isfinite(F[e2]).mean(axis=(0, 1))
    print(f"\nno statements: {miss or 'none'}")
    print("coverage all | era2 (2018+):")
    for fi, feat in enumerate(FEATS):
        print(f"  {feat:8s} {cov[fi]*100:5.1f}% | {cov2[fi]*100:5.1f}%")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
