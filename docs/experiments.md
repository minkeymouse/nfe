# Experiment scripts

One row per script, from its docstring. Every script runs in the one
environment, `uv sync --extra experiments --group dev`, as
`uv run python experiments/<script>.py`, from the repository root. Each
script's docstring is its pre-registration record: question, hypotheses,
gates and frozen protocol, written before the first full run.

**Artifact chain.** Some scripts read what another script wrote, so order
matters. `exp_e3_book.py --afm` writes the AFM residual cache
`results/run_260724/afm_stream_era{1,2}.npz`, and the book dumps each
era's evaluation-period P&L per arm to
`results/run_260728/book_pnl_era{1,2}.npz`. `exp_nfe_eb.py`,
`exp_nfe_prior.py`, `exp_anchor_fuel.py` and `exp_kappa_persist.py` read
the cache (`exp_anchor_fuel.py` drops its AFM row when the cache is
absent); `fig_book_nfe.py` reads the era-2 dump and needs its `AFM+fade`
and `AFM+NFE` arms, so the dump must come from an `--afm --nfe` run. Every
`exp_e3_book.py` run rewrites the dumps, `--smoke` included (a smoke run
replaces `book_pnl_era2.npz` with its 900-day slice), while the cache is
written only by a full `--afm` run. `exp_predictive.py`,
`exp_admission.py`, `exp_cross_sectional.py` and `exp_dgp_gate.py` read
the panel directly and honour `NFE_PANEL` (`krx`, the default; `public`;
`vendor2`).

**Panel format.** The KRX loader (`load_krx` in `exp_cross_sectional.py`,
which the other scripts import) reads a directory `data/raw/<panel>/`
holding a manifest parquet whose `shcode` column holds the file-name
stems as strings, and one `<shcode>.parquet` per stock with columns
`date` (YYYYMMDD, string or int) and `close` (split-adjusted float);
other columns are ignored, and a listed stem with no file is skipped.
The per-stock series are outer-joined on date; `fixed_panel` then keeps
the N stocks with the most non-missing days (N = 120 in `exp_e3_book.py`,
150 in `exp_cross_sectional.py`) and drops every date on which any of
them has a gap. The manifests are
`data/raw/krx/manifest.parquet` (`krx`),
`data/raw/krx_public/manifest.parquet` (`public`) and
`data/raw/krx_vendor2/manifest.parquet` (`vendor2`). No data ships
with the repository (see the README); `prep_fundamentals.py` also reads
unadjusted closes from `data/raw/krx_raw/<shcode>.parquet`, in the same
layout.

| script | what it does |
|---|---|
| `exp_admission.py` | Substrate admission: does this cross-section's residual mean-revert at all? |
| `exp_anchor_fuel.py` | Diagnostic: does the KRX residual panel still carry a common component? |
| `exp_anchor_regime.py` | The anchored regime: X = h(C) + u, with converge/diverge states on u |
| `exp_cross_sectional.py` | Cross-sectional regime detection on Binance + KRX daily cross-sections |
| `exp_dgp_gate.py` | Which residual process actually matches real cross-sections? (G4 two-sample gate) |
| `exp_e3_book.py` | E3 — the book: which residual construction produces a tradeable stat-arb book? |
| `exp_kappa_persist.py` | Does the reversion speed have a NAME-level persistent component, and do observable characteristics explain it? |
| `exp_nfe_anchor.py` | Anchored NFE: does a cross-sectional anchor channel improve the filter? |
| `exp_nfe_attention.py` | Neural Filter Estimator: row-column attention over the panel |
| `exp_nfe_eb.py` | Empirical-Bayes NFE: a per-name prior from the name's own train slice |
| `exp_nfe_factor.py` | NFE factor-sharing: does asset attention learn the row step? |
| `exp_nfe_level.py` | Is the window-mean demeaning in featurize() throwing away the level? |
| `exp_nfe_oracle.py` | Oracle diagnostic: what is the volatility-sharing channel worth? |
| `exp_nfe_panel.py` | Panel-form NFE arms for the book: the self-built row step and the 250-day window audit |
| `exp_nfe_prior.py` | Calibrating the half-life prior to the substrate (confirmation round) |
| `exp_nfe_screen.py` | Triage screen: three untested levers for the NFE |
| `exp_nfe_xs.py` | Can the contemporaneous cross-section supply the level the window cannot? |
| `exp_predictive.py` | Route (2): a PREDICTIVE posterior p(dX / window) — no simulator, no G4 gate |
| `exp_value_2x2.py` | What is the NFE overlay worth where the prior is true, and what survives deployment? The 2x2: substrates {synthetic, KRX} x policies {base, base+NFE} |
| `fig_book_nfe.py` | Paper figure: cumulative returns of the era-2 books, with/without NFE |
| `fig_calibration.py` | Paper figure: the posterior is calibrated (the title claim) |
| `fig_posterior_tsne.py` | Paper figure: the geometry of the amortized posterior |
| `prep_fundamentals.py` | Point-in-time daily fundamental features for the E3 KRX panel |
| `verify_afm_breadth.py` | Further AFM verification: does the paper's edge require BREADTH? |
| `verify_afm_conditional.py` | Fair test of the AFM edge: conditional (characteristic-driven) factors |
| `verify_afm_krx.py` | Verify the Attention Factor Model paper claim on REAL KRX daily data |
| `verify_fig3_seeds.py` | Verification pass: Figure 3's reliability curve across training seeds |
| `verify_nfe_runtime.py` | Runtime of the amortized posterior against its unamortized counterpart |
| `verify_promise_gap_sim.py` | Verification pass: is the promise-gap reduction conditional or uniform? |
