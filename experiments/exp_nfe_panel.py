"""Panel-form NFE arms for the book: the self-built row step and the
250-day window audit.

    imported by exp_e3_book.py --panel [--audit-confirm]
    uv run python experiments/exp_nfe_panel.py --smoke

[exploration] (2026-07-24). Two things this module supplies to the book,
both reported in the paper; the full iteration ledger (five registered
rounds, v1--v5, with verdicts) is in results/run_260724/E3_LINEAGE.md.

1. SELF-BUILT ROW STEP (nfe_stream_rolling). The fourth generation
   learning its own compression on real data: a conditional
   reconstruction factor model (signed attention weights over lagged
   rank characteristics, conditional betas, K factors) fit by
   minimizing ||R - beta F||^2 -- NOT the trading objective, so the
   posterior stays the product -- with AFM-style annual walk-forward
   refits. The residual stream is then sized by the simulator-trained
   posterior. Conditioning is the six slow, direction-blind
   fundamentals (the anchor principle: return-based columns let the
   betas absorb the reversion signal). Reaches -0.03, at the floor:
   the deployed NFE therefore adopts the AFM compression instead.

2. THE 250-DAY WINDOW AUDIT. The posterior conditioned on a 250-day
   window (the identification peak) beats AFM on the paired gap, but
   two registered fairness controls (states_audit) show the gain is a
   window effect, not a posterior effect, so the design keeps 60 days.
   What survives, at matched signal and window, is that the
   distribution beats the point plug-in.
     states_nfe250  z from the 60d window, kappa from the 250d window.
     base250        the base rule on the 250d deviation (control).
     plug250        a 250d point-kappa plug-in on the 60d signal.
"""
from __future__ import annotations

import sys

import numpy as np

SMOKE = "--smoke" in sys.argv
K_FACT, STEP, HID, WDECAY = 3, 246, 32, 1e-4
EPOCHS = 8 if SMOKE else 60


# ------------------------------------------------ self-built row step
def rank_chars(Fe):
    """Per-day cross-sectional rank of the fundamental columns to
    [-0.5, 0.5]: median-fill, neutral when under a quarter disclose."""
    T, N, M = Fe.shape
    out = np.zeros((T, N, M), dtype=np.float32)
    for fi in range(M):
        f = Fe[:, :, fi]
        fin = np.isfinite(f)
        med = np.zeros((T, 1))
        anyf = fin.any(1)
        med[anyf, 0] = np.nanmedian(np.where(fin, f, np.nan)[anyf], axis=1)
        order = np.where(fin, f, med).argsort(1).argsort(1).astype(float)
        rk = order / max(N - 1, 1) - 0.5
        rk[fin.sum(1) < N // 4] = 0.0
        out[:, :, fi] = rk
    return out


def _fit_row_step(Rw, Xw, epochs=EPOCHS, seed=0):
    """Conditional reconstruction factor model on one trailing window.
    Rw (T, N) returns, Xw (T, N, M) lagged rank characteristics ->
    a module mapping (R_t, X_{t-1}) to the residual. Signed, L1-normed
    factor weights (AFM's softmax portfolios cannot span signed
    loadings for reconstruction)."""
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    T, M = Rw.shape[0], Xw.shape[2]

    class RowStep(nn.Module):
        def __init__(self):
            super().__init__()
            self.q = nn.Linear(M, K_FACT, bias=False)
            self.b = nn.Sequential(nn.Linear(M, HID), nn.GELU(),
                                   nn.Linear(HID, K_FACT))

        def forward(self, R, X):                # R (B,N), X (B,N,M)
            w = self.q(X)
            w = w - w.mean(1, keepdim=True)
            w = w / (w.abs().sum(1, keepdim=True) + 1e-8)
            F = torch.einsum("bn,bnk->bk", R, w)
            fit = torch.einsum("bk,bnk->bn", F, self.b(X))
            return R - fit

    net = RowStep().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=WDECAY)
    Rt = torch.tensor(Rw, dtype=torch.float32, device=dev)
    Xt = torch.tensor(Xw, dtype=torch.float32, device=dev)
    for _ in range(epochs):
        perm = torch.randperm(T, device=dev)
        for j in range(0, T, 64):
            i = perm[j:j + 64]
            loss = (net(Rt[i], Xt[i]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    return net


def nfe_stream_rolling(Re, chars, epochs=EPOCHS):
    """Walk-forward residual stream: refit every STEP days on the
    trailing window, apply causally until the next refit. chars
    (T, N, M) are already lagged rank characteristics."""
    import torch
    T = len(Re)
    eps = np.full_like(Re, np.nan)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    first = 252
    for start in range(first, T, STEP):
        net = _fit_row_step(Re[max(0, start - 3 * STEP):start],
                            chars[max(0, start - 3 * STEP):start], epochs)
        hi = min(T, start + STEP)
        with torch.no_grad():
            resid = net(torch.tensor(Re[start:hi], dtype=torch.float32,
                                     device=dev),
                        torch.tensor(chars[start:hi], dtype=torch.float32,
                                     device=dev))
        eps[start:hi] = resid.cpu().numpy()
    eps[:first] = Re[:first] - Re[:first].mean(1, keepdims=True)
    return eps


# ------------------------------------------------ the 250-day window audit
def _sim_train_L(n, L2, rng, phi_vol, s_log, tau=0.05):
    """exp_e3_book._sim_train at a configurable window length L2."""
    hl = np.exp(rng.uniform(np.log(3), np.log(400), n))
    k = np.log(2) / hl
    m_ln = np.exp(s_log ** 2 / 2)
    sd_ln = np.sqrt((np.exp(s_log ** 2) - 1) * np.exp(s_log ** 2))
    x = np.zeros(n); lv = np.zeros(n); m = np.zeros(n)
    W = np.empty((n, L2), dtype=np.float32)
    for t in range(200 + L2):
        lv = phi_vol * lv + 0.5 * np.sqrt(1 - phi_vol ** 2) * \
            rng.standard_normal(n)
        e = (np.exp(s_log * rng.standard_normal(n)) - m_ln) / sd_ln
        m = m + tau * rng.standard_normal(n) * 0.01
        x = x + k * (m - x) + 0.01 * np.exp(lv) * e
        if t >= 200:
            W[:, t - 200] = x
    return W, np.log2(hl).astype(np.float32)


def _feat_w(W):
    Z = (W - W.mean(-1, keepdims=True)) / (W.std(-1, keepdims=True) + 1e-12)
    return np.stack([Z, np.diff(Z, axis=-1, prepend=Z[:, :1])],
                    -1).astype(np.float32)


def train_nfe_L(L2, phi_vol, s_log, seed=0):
    """The GRU+MDN posterior of exp_e3_book, trained on L2-day windows."""
    import torch
    import torch.nn as nn
    from exp_nfe_attention import mdn_nll
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(seed)
    ntrain, steps = (4_000, 400) if SMOKE else (60_000, 4_000)
    W, y = _sim_train_L(ntrain, L2, rng, phi_vol, s_log)
    X = torch.tensor(_feat_w(W)); Y = torch.tensor(y)
    net = nn.ModuleDict({"gru": nn.GRU(2, 64, batch_first=True),
                         "mdn": nn.Linear(64, 3 * 5)}).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for _ in range(steps):
        i = torch.randint(0, len(X), (256,))
        loss = mdn_nll(net["mdn"](net["gru"](X[i].to(dev))[0][:, -1]),
                       Y[i].to(dev)).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    return net


def _kappa_samples_L(net, X, nsamp=64, seed=0):
    """MDN kappa samples for arbitrary-length standardized windows."""
    import torch
    dev = next(net.parameters()).device
    with torch.no_grad():
        o = net["mdn"](net["gru"](
            torch.tensor(_feat_w(X)).to(dev))[0][:, -1]).cpu().view(-1, 3, 5)
    pi = torch.softmax(o[:, 0], -1).numpy()
    mu, sd = o[:, 1].numpy(), o[:, 2].clamp(-4, 2).exp().numpy()
    g = np.random.default_rng(seed)
    comp = np.stack([g.choice(5, nsamp, p=pi[i] / pi[i].sum())
                     for i in range(len(pi))])
    y = np.take_along_axis(mu, comp, 1) + \
        np.take_along_axis(sd, comp, 1) * g.standard_normal((len(pi), nsamp))
    return np.log(2) / np.clip(2.0 ** y, 1e-3, 1e4)


def _mv_weights(z, ks, H):
    a, b = np.exp(-ks * H), np.exp(-2 * ks * H)
    return -z * (1.0 - a).mean(1) / \
        ((1.0 - b).mean(1) + z ** 2 * a.var(1) + 1e-12)


def _rebalances(E, L=60, H=20):
    """Yield (t, 60d cumulative-residual window S) at each hold start."""
    for t in range(L, len(E)):
        if (t - L) % H == 0:
            yield t, np.cumsum(E[t - L:t].T, axis=-1)


def states_nfe250(E, net60, net250, L2=250, L=60, H=20):
    """z from the 60d window, kappa from the 250d window once available;
    plain 60d posterior weights before that."""
    from nfe import sig_nfe
    N = E.shape[1]
    X = np.cumsum(np.nan_to_num(E), axis=0)
    out = []
    for t, S60 in _rebalances(E, L, H):
        if not np.isfinite(S60).all():
            out.append((t, np.zeros(N))); continue
        if t >= L2:
            z = np.clip((S60[:, -1] - S60.mean(-1)) / (S60.std(-1) + 1e-12),
                        -3, 3)
            W250 = (X[t - L2:t] - X[t - L2]).T
            w = _mv_weights(z, _kappa_samples_L(net250, W250), H)
        else:
            w = sig_nfe(E[t - L:t].T, net60)
        out.append((t, _unit(w)))
    return out


def states_audit(E, mode, L2=250, L=60, H=20):
    """'base250' trades -clip(z250); 'plug250' sizes the z60 signal by a
    point kappa from a 250d AR(1) OLS. Registered fairness controls."""
    N = E.shape[1]
    X = np.cumsum(np.nan_to_num(E), axis=0)
    out = []
    for t, S60 in _rebalances(E, L, H):
        if not np.isfinite(S60).all() or t < L2:
            out.append((t, np.zeros(N))); continue
        W250 = (X[t - L2:t] - X[t - L2]).T
        if mode == "base250":
            w = -np.clip((W250[:, -1] - W250.mean(-1)) /
                         (W250.std(-1) + 1e-12), -3, 3)
        else:                                        # plug250
            z = np.clip((S60[:, -1] - S60.mean(-1)) / (S60.std(-1) + 1e-12),
                        -3, 3)
            dm, dS = W250[:, :-1] - W250.mean(-1, keepdims=True), \
                np.diff(W250, axis=-1)
            kap = np.clip(-(dm * dS).sum(-1) / ((dm ** 2).sum(-1) + 1e-18),
                          1e-4, 1.0)
            a = np.exp(-kap * H)
            w = -z * (1.0 - a) / ((1.0 - a ** 2) + 1e-12)
        out.append((t, _unit(w)))
    return out


def _unit(w):
    w = w - w.mean()
    n = np.abs(w).sum()
    return w / n if n > 1e-12 else w


def book_states(E, states, c_turn, c_short, L=60, H=20):
    """Book precomputed unit-gross rebalance weights under the frozen
    conventions (matches exp_e3_book.book)."""
    N = E.shape[1]
    st = dict(states)
    pnl, tov, prev, w = [], [], np.zeros(N), np.zeros(N)
    for t in range(L, len(E)):
        if t in st:
            w = st[t]
        p = w @ E[t]
        turn = np.abs(w - prev).sum()
        pnl.append(p - c_turn * turn - c_short * np.maximum(-w, 0).sum())
        tov.append(turn); prev = w
    return np.array(pnl), np.array(tov)


def main():
    print("smoke: self-built row step on a synthetic factor panel")
    rng = np.random.default_rng(0)
    T, N, M = 700, 40, 6
    F, B = rng.standard_normal((T, 2)), rng.standard_normal((N, 2))
    R = F @ B.T + 0.5 * rng.standard_normal((T, N))
    X = np.repeat(np.concatenate([B, rng.standard_normal((N, M - 2))],
                                 1)[None], T, 0).astype(np.float32)
    eps = nfe_stream_rolling(R, X, epochs=30)
    vi, vo = np.nanvar(R[252:]), np.nanvar(eps[252:])
    print(f"  variance share removed {1 - vo / vi:.2f} "
          f"(factor share {np.var(F @ B.T) / np.var(R):.2f})")


if __name__ == "__main__":
    main()
