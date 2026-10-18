# nfe: the Neural Filter Estimator

An amortized filter for the latent state of a mean-reverting spread,
returning a calibrated posterior rather than a point estimate.

A generative model supplies training draws whose latent half-life is
known by construction. A GRU encoder with a mixture-density head is
trained on those draws by neural posterior estimation. A frozen decision
layer sizes positions from the posterior, shrinking a position when the
posterior is diffuse. Trained once, the filter answers an entire
cross-section in one forward pass and never sees a real return, which is
what makes its calibration checkable on truth-known draws.

The package is the reference implementation for *Amortized Neural
Filtering for Calibrated Posteriors in Statistical Arbitrage*
(ICAIF '26). The `experiments/` directory holds the scripts behind that
paper's numbers; the paper itself lives in its own repository.

## Install

```bash
pip install -e .                 # numpy + torch
pip install -e ".[hub]"          # + huggingface_hub, for from_pretrained("<repo id>")
pip install -e ".[experiments]"  # + scipy, pandas, matplotlib, scikit-learn, dfm-python
```

or, with [uv](https://docs.astral.sh/uv/): `uv sync` (add
`--extra experiments --group dev` for the scripts and the tests).

## Quick start

```python
from nfe import NFE

f = NFE.fit_to(train_residuals)          # calibrate the prior, then train
weights = f.size(window)                 # posterior-sized positions

f.save("my-filter")                      # -> my-filter/config.json + filter.pt
g = NFE.from_pretrained("my-filter")     # a local directory ...
g = NFE.from_pretrained("user/my-filter")  # ... or a Hugging Face Hub repo id
```

`train_residuals` is a `(T, N)` array of residual returns (bars by names);
`window` is an `(N, L)` slice for the current cross-section. The default
training budget is 60,000 draws for 4,000 steps, a few minutes on a CPU;
pass `ntrain=` and `steps=` to `fit` or `fit_to` for a quicker fit.
`examples/quickstart.py` runs the whole loop on synthetic data.

### Swapping the pieces

- **The generative model.** Any callable `(n, rng) -> (windows, targets)`
  works: `NFE.fit(simulator=...)` or `train_nfe(simulator=...)`. The
  default, `switching_ou`, is a two-state switching Ornstein-Uhlenbeck
  spread with stochastic volatility, a drifting level and lognormal
  innovations; `calib_from` moment-matches its innovation law to a panel.
- **The deployment prior.** The head is trained under a prior flat in log2
  half-life, so it is likelihood-shaped and a prior can be applied by
  reweighting its draws instead of retraining: `sig_nfe(..., prior=f)`
  for any `f(y) -> log density`. `gaussian_prior` and `column_prior` (one
  empirical-Bayes prior per name, the `fit_to` default) are provided.

## Layout

| path | role |
|---|---|
| `nfe/prior.py` | the generative model (`simulate`, `switching_ou`), the feature map, the calibrations (`calib_from`, `calib_halflife`) and the deployment priors (`gaussian_prior`, `column_prior`) |
| `nfe/model.py` | the single-name filter and its training: GRU encoder + mixture-density head (`build_net`, `mdn_nll`, `train_nfe`) |
| `nfe/filter.py` | the decision layer: posterior draws of the reversion rate (`nfe_kappa_samples`) and the uncertainty-penalized sizing rule (`sig_nfe`) |
| `nfe/estimator.py` | the `NFE` class and its `NFEConfig`, which records the training budget with the weights |
| `nfe/panel.py` | the panel form: alternating time and asset attention over a bars-by-names panel (`PanelNFE`) |
| `tests/` | `pytest tests/`, seconds on a CPU |
| `examples/` | `quickstart.py` |
| `experiments/` | the scripts behind the paper's tables and figures; see `docs/experiments.md` |

The package is CPU-only by design and deterministic for a given seed on a
given build. `dfm-python`, installed with the `experiments` extra,
provides the factor-model baselines (the Attention Factor Model and PCA)
some experiments compare against; the package itself does not depend on
it.

## Data

No data ships with the repository, and the scripts that collected it are
not included. The inputs the experiments read, in the layout
`docs/experiments.md` describes, are available from the author on
request: the KRX daily closing-price panels (`NFE_PANEL=krx`, the
paper's panel; `public`, a proxy on a different universe; `vendor2`, the
Appendix B robustness panel), the point-in-time fundamentals that
`prep_fundamentals.py` builds from DART statements, and the crypto and
factor series. The run ledgers that the script docstrings cite
(`results/run_*/RUN_LOG.md` and the per-run notes) are written under the
gitignored `results/` directory when a script runs.

## Reproducing the paper

```bash
uv sync --extra experiments --group dev  # includes dfm-python
uv run pytest tests -q                                # package suite, seconds on CPU
uv run python examples/quickstart.py                  # end to end on synthetic data
```

All scripts run in that one environment, as
`uv run python experiments/<script>.py`; each script's docstring usage
line is authoritative, and `docs/experiments.md` indexes every script with the
question it answers and the artifacts it reads. Run from the repository
root; outputs go to `results/run_YYMMDD/`.

Reproduction is bit for bit for the `nfe` package on a fixed seed and a
given torch build. The AFM baseline's numbers (Table 1's AFM and NFE rows)
were computed under torch 2.9.1; other torch builds reproduce them to
about 1e-6 relative, not bit for bit.

## Citation

```bibtex
@inproceedings{chang2026amortized,
  author    = {Minkey Chang},
  title     = {Amortized Neural Filtering for Calibrated Posteriors in Statistical Arbitrage},
  booktitle = {7th ACM International Conference on AI in Finance (ICAIF '26)},
  year      = {2026},
  address   = {Milan, Italy},
  publisher = {ACM},
  doi       = {10.1145/3858063.3858894}
}
```

## License

MIT; see `LICENSE`. `dfm-python` is a separate MIT-licensed package.
