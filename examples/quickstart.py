"""Minimal end-to-end use of the NFE estimator on synthetic data.

    python examples/quickstart.py

Calibrates the generative model to a residual panel, trains the filter,
sizes a cross-section, and round-trips the trained filter through
save / from_pretrained. Uses a reduced training budget so it finishes in
well under a minute on a CPU; drop the ``ntrain`` / ``steps`` arguments
for the full default budget.
"""
import tempfile

import numpy as np

from nfe import NFE, simulate

# A stand-in residual panel (T bars x N names). In practice these are the
# factor residuals of your book; here they are drawn from the simulator.
rng = np.random.default_rng(0)
W, _ = simulate(120, rng)                     # 120 windows of length L
residuals = W.T                               # (L, 120): bars x names

# Fit: calibrate the generative model to the panel, then train the filter.
f = NFE.fit_to(residuals, seed=0, ntrain=8_000, steps=400)
print("calibrated prior:", f.config.phi_vol, f.config.s_log)

# Deploy: posterior-sized positions for the latest window.
window = residuals[-f.config.L:].T            # (N, L) residual returns
weights = f.size(window)
print("weights:", weights.shape, "gross", np.abs(weights).sum().round(3))

# Persist and reload (the layout the Hugging Face Hub expects). The
# checkpoint holds the trained filter and its config; the deployment prior
# belongs to the substrate, so it is re-attached after loading.
with tempfile.TemporaryDirectory() as d:
    f.save(d)
    g = NFE.from_pretrained(d)
    g.prior = f.prior
    assert np.allclose(g.size(window), weights)
    print("save / from_pretrained round-trip OK")
