"""nfe: the Neural Filter Estimator.

An amortized filter for the latent state of a mean-reverting spread. A
generative model (:mod:`nfe.prior`) supplies draws whose latent
half-life is known by construction; a GRU encoder with a mixture-density
head (:mod:`nfe.model`) is trained on those draws by neural posterior
estimation; a frozen decision layer (:mod:`nfe.filter`) sizes positions
from the posterior. Trained once, the filter answers any window in one
forward pass and never sees a real return, which is what makes its
calibration checkable.

Modules
    nfe.prior       the generative model, feature map, calibrations, priors
    nfe.model       the GRU+MDN filter and its training
    nfe.filter      posterior draws and the sizing rule
    nfe.estimator   the NFE class: fit / fit_to / size / save / from_pretrained
    nfe.panel       the panel form (alternating time and asset attention)

Quick start::

    from nfe import NFE
    f = NFE.fit_to(train_residuals)          # calibrate the prior, then train
    w = f.size(window)                       # posterior-sized positions
    f.save("my-filter")                      # -> config.json + filter.pt
    NFE.from_pretrained("my-filter")         # a local dir or a Hub repo id

The functional API (``simulate``, ``calib_from``, ``train_nfe``,
``sig_nfe``, ...) is the same implementation the estimator wraps.
"""
from .prior import (L, H, HL_LO, HL_HI, Simulator, simulate, featurize,
                    calib_from, calib_halflife, gaussian_prior, column_prior,
                    switching_ou)
from .model import (NFE_KMIX, NFE_NTRAIN, NFE_STEPS, HIDDEN, build_net,
                    mdn_nll, train_nfe)
from .filter import NFE_NSAMP, nfe_kappa_samples, sig_nfe
from .estimator import NFE, NFEConfig
from .panel import (PanelNFE, RowColumnBlock, build_panel_nfe,
                    panel_simulate, panel_featurize, panel_mdn_nll,
                    panel_mdn_pit)

__version__ = "0.3.0"

__all__ = [
    "__version__",
    # estimator
    "NFE", "NFEConfig",
    # prior
    "L", "H", "HL_LO", "HL_HI", "Simulator", "simulate", "featurize",
    "calib_from", "calib_halflife", "gaussian_prior", "column_prior",
    "switching_ou",
    # model
    "NFE_KMIX", "NFE_NTRAIN", "NFE_STEPS", "HIDDEN", "build_net", "mdn_nll",
    "train_nfe",
    # filter
    "NFE_NSAMP", "nfe_kappa_samples", "sig_nfe",
    # panel form
    "PanelNFE", "RowColumnBlock", "build_panel_nfe", "panel_simulate",
    "panel_featurize", "panel_mdn_nll", "panel_mdn_pit",
]
