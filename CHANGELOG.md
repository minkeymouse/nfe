# Changelog

## 0.3.1

Public release of the repository alongside the paper.

- `CITATION.cff` added; docstrings no longer refer to internal substrate
  names or working notes.

- `dfm-python` (the Attention Factor Model and PCA baselines) is an ordinary
  dependency in the `experiments` extra (PyPI, `>=0.7.0`) instead of a git
  submodule with its own environment; every script runs under `uv run`.
- Scripts import `dfm_python` directly; `verify_afm_*` use repository-relative
  paths. Numerics unchanged: the package core fingerprints bit for bit before
  and after, and the installed AFM equals the former submodule source bit for
  bit in the same environment.

## 0.3.0

A packaging and layout release; the numerics are unchanged, verified
against a fixed-seed fingerprint of 0.2.0 (`simulate`, `featurize`, the
calibrations, `column_prior`, a short `train_nfe`, `sig_nfe` with and
without a prior, `nfe_kappa_samples`, `mdn_nll`, the panel simulator and
`PanelNFE`).

- The package is flat: `nfe/{prior,model,filter,estimator,panel}.py`
  replaces `nfe/src/`. `nfe.src.*` imports are gone; everything public
  is exported from `nfe`.
- `nfe` is an installable project (`pip install -e .`, `uv sync`), with
  only `numpy` and `torch` as dependencies. The experiment scripts are
  an optional extra.
- `train_nfe` / `NFE.fit` / `NFE.fit_to` take the training budget and
  hyper-parameters (`ntrain`, `steps`, `hidden`, `kmix`, `lr`, `batch`)
  as arguments; the defaults are the previous constants. `NFEConfig`
  records them, and the package version, with the weights.
- `train_nfe` returns the module in eval mode.
- The simulator's internal constants (burn-in, scale, volatility sd) are
  named; the panel simulator shares them.
- Tests moved to `tests/` and run in seconds (the estimator tests use a
  small budget); new tests for seed reproducibility, the custom-simulator
  path, the flat-prior limit of the sizing rule and the deployment-prior
  signature. A GitHub Actions workflow runs them.
- Docstrings describe the method in its own terms; paper cross-references
  and project-specific measurements were moved out of the package.

## 0.2.0

Two things the filter had baked in are now choices, and one design
commitment is now explicit.

**The generative model is swappable.** A generative model is any callable
`(n, rng) -> (windows, targets)`. `switching_ou(...)` packages the
default process (switching Ornstein-Uhlenbeck with stochastic volatility,
a drifting level and lognormal innovations), so existing calls are
unchanged; `train_nfe(simulator=...)` and `NFE.fit(simulator=...)` take
any other. Everything downstream is untouched by the substitution.

**The deployment prior is swappable, and conditioning is per column.** The
head is trained under a prior flat in log2 half-life, which makes it
likelihood-shaped, so a prior can be applied by reweighting its draws
instead of retraining:

- `sig_nfe(..., prior=f)` takes any callable `f(y) -> log density` over an
  `(N, S)` array of draws; `None` keeps the training prior.
- `gaussian_prior(centre, sd)` builds one from scalars or per-column arrays.
- `column_prior(panel)` fits one prior per column of the panel, per name,
  from its training slice by empirical Bayes, each column's own
  half-lives pulled toward the panel mean by `between / (between + within/k)`.
- `NFE.fit_to(panel)` applies it by default. Pass
  `column_conditioning=False` to share one prior across the cross-section,
  or a callable of your own.
- `calib_halflife(train_slice)` fits the prior's range instead, for callers
  who would rather bake it into training; `simulate` / `train_nfe` take
  `hl_lo` / `hl_hi`.

Also fixed: `_block_halflives` masks gaps per column rather than per row.

## 0.1.0

Initial release of the Neural Filter Estimator as a standalone package:
the generative model and its calibration (`prior`), the GRU + mixture
density filter and its training (`model`), the decision layer (`filter`),
the panel form (`panel`), the `NFE` estimator with `save` /
`from_pretrained`, tests and a quick-start example.
