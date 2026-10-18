"""Tests for the nfe package. Every training call uses a tiny budget so the
whole suite runs in seconds on a CPU.

    pytest tests/ -q
"""
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

import nfe
from nfe import (NFE, L, NFE_KMIX, build_net, build_panel_nfe, calib_from,
                 calib_halflife, column_prior, featurize, gaussian_prior,
                 mdn_nll, nfe_kappa_samples, panel_featurize, panel_mdn_nll,
                 panel_simulate, sig_nfe, simulate, switching_ou, train_nfe)

TINY = dict(ntrain=800, steps=40)


# ---- prior (the label factory) ---------------------------------------
def test_simulate_shapes():
    W, y = simulate(64, np.random.default_rng(0))
    assert W.shape == (64, L) and y.shape == (64,)
    assert np.isfinite(W).all() and np.isfinite(y).all()


def test_simulate_is_seed_reproducible():
    a = simulate(16, np.random.default_rng(5))
    b = simulate(16, np.random.default_rng(5))
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_switching_ou_matches_simulate():
    W1, y1 = simulate(8, np.random.default_rng(1), phi_vol=0.3, s_log=0.2)
    W2, y2 = switching_ou(phi_vol=0.3, s_log=0.2)(8, np.random.default_rng(1))
    assert np.array_equal(W1, W2) and np.array_equal(y1, y2)


def test_featurize_shapes():
    W, _ = simulate(8, np.random.default_rng(0))
    F = featurize(W)
    assert F.shape == (8, L, 2) and F.dtype == np.float32


def test_calibrate_returns_valid_params():
    W, _ = simulate(400, np.random.default_rng(1))
    cal = calib_from(W.T)                       # (L, N) residual-like panel
    assert set(cal) == {"phi_vol", "s_log"}
    assert 0.0 <= cal["phi_vol"] <= 0.97 and 0.02 <= cal["s_log"] <= 1.2


def test_calibrate_short_slice_falls_back_to_defaults():
    assert calib_from(np.zeros((L - 1, 3))) == {"phi_vol": 0.19, "s_log": 0.14}
    assert calib_halflife(np.zeros((L - 1, 3))) == {"hl_lo": nfe.HL_LO,
                                                     "hl_hi": nfe.HL_HI}


def test_priors_have_log_density_signature():
    W, _ = simulate(300, np.random.default_rng(2))
    y = np.linspace(1, 9, 20)[None].repeat(300, 0)
    for f in (gaussian_prior(4.0, 1.0), gaussian_prior(np.full(300, 4.0), 1.0),
              column_prior(W.T)):
        out = f(y)
        assert out.shape == y.shape and np.isfinite(out).all()


# ---- model (the filter) ----------------------------------------------
def test_build_net_output_shape():
    net = build_net()
    W, _ = simulate(5, np.random.default_rng(0))
    out = net["mdn"](net["gru"](torch.tensor(featurize(W)))[0][:, -1])
    assert out.shape == (5, 3 * NFE_KMIX)


def test_mdn_nll_finite():
    net = build_net()
    W, y = simulate(16, np.random.default_rng(0))
    out = net["mdn"](net["gru"](torch.tensor(featurize(W)))[0][:, -1])
    assert torch.isfinite(mdn_nll(out, torch.tensor(y, dtype=torch.float32))).all()


def test_train_reduces_loss():
    net = train_nfe(seed=0, **TINY)
    W, y = simulate(256, np.random.default_rng(9))
    X = torch.tensor(featurize(W)); Y = torch.tensor(y, dtype=torch.float32)
    with torch.no_grad():
        trained = mdn_nll(net["mdn"](net["gru"](X)[0][:, -1]), Y).mean()
    untrained = mdn_nll(build_net()["mdn"](
        build_net()["gru"](X)[0][:, -1]), Y).mean()
    assert trained < untrained          # NPE actually learns


def test_train_is_seed_reproducible():
    a = train_nfe(seed=3, ntrain=400, steps=10)
    b = train_nfe(seed=3, ntrain=400, steps=10)
    for pa, pb in zip(a.parameters(), b.parameters()):
        assert torch.equal(pa, pb)


def test_train_accepts_custom_simulator():
    def sim(n, rng):
        return simulate(n, rng, phi_vol=0.0, s_log=0.05)
    net = train_nfe(seed=0, simulator=sim, ntrain=400, steps=10)
    assert isinstance(net, torch.nn.ModuleDict)


# ---- filter (the decision layer) -------------------------------------
def test_sizing_shapes_and_finite():
    net = train_nfe(seed=0, ntrain=400, steps=20)
    window = np.random.default_rng(3).standard_normal((30, L)) * 0.01
    w = sig_nfe(window, net)
    assert w.shape == (30,) and np.isfinite(w).all()


def test_sizing_opposes_deviation():
    """A spread stretched positive should be faded (negative weight)."""
    net = train_nfe(seed=0, **TINY)
    up = np.linspace(0, 0.05, L)[None].repeat(4, 0)        # rising residual
    assert (sig_nfe(np.diff(up, prepend=0.0, axis=-1), net) <= 0).all()


def test_sizing_with_prior_matches_flat_prior_limit():
    """A constant prior reweights nothing, so the two code paths agree."""
    net = train_nfe(seed=0, ntrain=400, steps=20)
    window = np.random.default_rng(4).standard_normal((12, L)) * 0.01
    flat = sig_nfe(window, net, prior=lambda y: np.zeros_like(y))
    assert np.allclose(flat, sig_nfe(window, net), atol=1e-10)


def test_kappa_samples_positive():
    net = train_nfe(seed=0, ntrain=400, steps=20)
    W, _ = simulate(12, np.random.default_rng(0))
    ks = nfe_kappa_samples(net, np.cumsum(W, -1))
    assert ks.shape == (12, 64) and (ks > 0).all()


# ---- estimator (save / from_pretrained) ------------------------------
def test_estimator_roundtrip():
    W, _ = simulate(120, np.random.default_rng(0))
    f = NFE.fit(seed=0, phi_vol=0.2, s_log=0.1, **TINY)
    window = W.T[-L:].T
    w = f.size(window)
    with tempfile.TemporaryDirectory() as d:
        f.save(d)
        assert (Path(d) / "config.json").exists()
        assert (Path(d) / "filter.pt").exists()
        g = NFE.from_pretrained(d)
    assert g.config.phi_vol == 0.2 and g.config.steps == TINY["steps"]
    assert g.config.nfe_version == nfe.__version__
    assert np.allclose(g.size(window), w)


def test_estimator_fit_to_conditions_per_column():
    W, _ = simulate(300, np.random.default_rng(2))
    f = NFE.fit_to(W.T, seed=0, **TINY)
    assert 0.0 <= f.config.phi_vol <= 0.97
    assert callable(f.prior)
    g = NFE.fit_to(W.T, seed=0, column_conditioning=False, **TINY)
    assert g.prior is None
    with pytest.raises(ValueError):
        NFE.fit_to(W.T, seed=0, column_conditioning="yes", **TINY)


# ---- panel form (alternating time and asset attention) ---------------
def test_panel_forward_shape():
    W, _ = panel_simulate(3, 8, np.random.default_rng(0))
    net = build_panel_nfe(d=16, heads=2, blocks=1)
    out = net(torch.tensor(panel_featurize(W)))
    assert out.shape == (3, 8, 3 * NFE_KMIX)


def test_panel_is_permutation_equivariant():
    """Reordering names permutes the outputs the same way: the asset axis
    is exchangeable, the architecture's defining property."""
    W, _ = panel_simulate(2, 6, np.random.default_rng(0))
    net = build_panel_nfe(d=16, heads=2, blocks=1).eval()
    X = torch.tensor(panel_featurize(W))
    perm = torch.tensor([3, 1, 5, 0, 4, 2])
    with torch.no_grad():
        a = net(X)[:, perm]
        b = net(X[:, perm])
    assert torch.allclose(a, b, atol=1e-5)


def test_panel_nll_finite():
    W, y = panel_simulate(3, 8, np.random.default_rng(0))
    net = build_panel_nfe(d=16, heads=2, blocks=1)
    out = net(torch.tensor(panel_featurize(W)))
    assert torch.isfinite(panel_mdn_nll(out, torch.tensor(y))).all()
