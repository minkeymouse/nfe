"""The user-facing estimator.

:class:`NFE` wraps prior calibration, amortized training and posterior
sizing behind ``fit`` / ``fit_to`` / ``size`` / ``save`` /
``from_pretrained``, and carries in :class:`NFEConfig` everything needed
to rebuild a trained filter from a checkpoint, including the training
budget that produced it.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch

from . import prior
from .filter import NFE_NSAMP, nfe_kappa_samples, sig_nfe
from .model import (BATCH, HIDDEN, LR, NFE_KMIX, NFE_NTRAIN, NFE_STEPS,
                    build_net, train_nfe)

__all__ = ["NFE", "NFEConfig", "CONFIG_FILE", "WEIGHTS_FILE"]

CONFIG_FILE, WEIGHTS_FILE = "config.json", "filter.pt"


@dataclass
class NFEConfig:
    """Everything needed to rebuild a trained filter and deploy it."""
    phi_vol: float = prior.DEFAULT_PHI_VOL   # generative model: volatility persistence
    s_log: float = prior.DEFAULT_S_LOG       # generative model: innovation asymmetry
    hl_lo: float = prior.HL_LO               # half-life prior range used in training
    hl_hi: float = prior.HL_HI
    L: int = prior.L                         # causal window
    H: int = prior.H                         # hold horizon the sizing integrates over
    kmix: int = NFE_KMIX                     # mixture components
    hidden: int = HIDDEN                     # GRU width
    seed: int = 0
    ntrain: int = NFE_NTRAIN                 # training budget that produced the weights
    steps: int = NFE_STEPS
    lr: float = LR
    batch: int = BATCH
    nfe_version: str = ""                    # package version that wrote the checkpoint


class NFE:
    """An amortized filter for the latent state of a mean-reverting spread.

    Typical use::

        from nfe import NFE
        f = NFE.fit_to(train_residuals)     # calibrate the prior, then train
        w = f.size(window)                  # posterior-sized positions

    ``train_residuals`` is a ``(T, N)`` array of residual returns (bars by
    names); ``window`` is an ``(N, L)`` slice of residual returns for the
    current cross-section.
    """

    def __init__(self, net: torch.nn.Module, config: NFEConfig,
                 prior_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None):
        self.net = net
        self.config = config
        # Deployment prior on the log2 half-life: any f(y) -> log density, or
        # None to keep the training prior. Not part of the checkpoint; it
        # belongs to the substrate, not to the trained filter.
        self.prior = prior_fn

    # ---- construction -------------------------------------------------
    @staticmethod
    def calibrate(residuals: np.ndarray) -> dict:
        """Fit the generative model's innovation law to a residual panel's
        training slice; returns ``{phi_vol, s_log}`` for :meth:`fit`."""
        return prior.calib_from(residuals)

    @classmethod
    def fit(cls, phi_vol: float = prior.DEFAULT_PHI_VOL,
            s_log: float = prior.DEFAULT_S_LOG,
            hl_lo: float = prior.HL_LO, hl_hi: float = prior.HL_HI,
            seed: int = 0, simulator: Optional[prior.Simulator] = None,
            ntrain: int = NFE_NTRAIN, steps: int = NFE_STEPS,
            hidden: int = HIDDEN, kmix: int = NFE_KMIX,
            lr: float = LR, batch: int = BATCH) -> "NFE":
        """Train the filter on draws from a generative model; the default
        is :func:`nfe.prior.switching_ou` built from the given parameters.
        ``ntrain`` and ``steps`` set the training budget (the defaults are
        the full budget; pass smaller values for a quick fit)."""
        from . import __version__
        net = train_nfe(seed=seed, phi_vol=phi_vol, s_log=s_log,
                        hl_lo=hl_lo, hl_hi=hl_hi, simulator=simulator,
                        ntrain=ntrain, steps=steps, hidden=hidden, kmix=kmix,
                        lr=lr, batch=batch)
        cfg = NFEConfig(phi_vol=phi_vol, s_log=s_log, hl_lo=hl_lo, hl_hi=hl_hi,
                        kmix=kmix, hidden=hidden, seed=seed, ntrain=ntrain,
                        steps=steps, lr=lr, batch=batch, nfe_version=__version__)
        return cls(net, cfg)

    @classmethod
    def fit_to(cls, panel: np.ndarray, seed: int = 0,
               column_conditioning=True,
               simulator: Optional[prior.Simulator] = None, **train_kwargs) -> "NFE":
        """Calibrate the generative model to a ``(T, N)`` panel's training
        slice and train, in one step. Extra keyword arguments go to
        :meth:`fit` (for instance ``ntrain`` and ``steps``).

        ``column_conditioning`` is on by default: each column of the panel,
        each name, gets its own prior on the half-life, fitted from the same
        slice by :func:`nfe.prior.column_prior`, which :meth:`size` then
        applies. Pass ``False`` to share the training prior across the
        cross-section, or a callable ``f(y) -> log density`` of your own.
        """
        f = cls.fit(**cls.calibrate(panel), seed=seed, simulator=simulator,
                    **train_kwargs)
        if column_conditioning is True:
            f.prior = prior.column_prior(panel)
        elif callable(column_conditioning):
            f.prior = column_conditioning
        elif column_conditioning is not False:
            raise ValueError("column_conditioning must be True, False, "
                             "or a callable")
        return f

    # ---- deployment ---------------------------------------------------
    def posterior_kappa(self, window: np.ndarray, nsamp: int = NFE_NSAMP,
                        seed: int = 0) -> np.ndarray:
        """Posterior draws of the reversion rate kappa for each name in an
        ``(N, L)`` residual-return ``window``."""
        return nfe_kappa_samples(self.net, np.cumsum(window, axis=-1),
                                 nsamp=nsamp, seed=seed, kmix=self.config.kmix)

    def size(self, window: np.ndarray) -> np.ndarray:
        """Posterior-sized positions for an ``(N, L)`` residual-return
        ``window``: an uncertain posterior shrinks the position, a confident
        one does not (see :func:`nfe.filter.sig_nfe`)."""
        return sig_nfe(window, self.net, H=self.config.H, prior=self.prior)

    __call__ = size

    # ---- persistence --------------------------------------------------
    def save(self, path) -> None:
        """Write ``config.json`` and ``filter.pt`` to ``path``, the layout
        :meth:`from_pretrained` and the Hugging Face Hub expect."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / CONFIG_FILE).write_text(json.dumps(asdict(self.config),
                                                   indent=2))
        torch.save(self.net.state_dict(), path / WEIGHTS_FILE)

    @classmethod
    def from_pretrained(cls, path_or_repo: str,
                        revision: Optional[str] = None,
                        map_location: str = "cpu") -> "NFE":
        """Load a trained filter from a local directory or a Hugging Face
        Hub repo id (``"user/nfe-xyz"``). The Hub path needs
        ``huggingface_hub`` installed."""
        local = Path(path_or_repo)
        if not (local / CONFIG_FILE).exists():
            local = Path(_download_from_hub(path_or_repo, revision))
        config = NFEConfig(**json.loads((local / CONFIG_FILE).read_text()))
        net = build_net(hidden=config.hidden, kmix=config.kmix)
        net.load_state_dict(torch.load(local / WEIGHTS_FILE,
                                       map_location=map_location,
                                       weights_only=True))
        net.eval()
        return cls(net, config)


def _download_from_hub(repo_id: str, revision: Optional[str]) -> str:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:                       # pragma: no cover
        raise ImportError(
            "loading from the Hugging Face Hub needs `huggingface_hub` "
            "(`pip install nfe[hub]`), or pass a local path."
        ) from e
    return snapshot_download(repo_id, revision=revision,
                             allow_patterns=[CONFIG_FILE, WEIGHTS_FILE])
