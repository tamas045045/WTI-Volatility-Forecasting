"""LSTM-család: HAR-X + LSTM-korrekció (docs/09_lstm_protokoll.md, 3b. pont).

    y_hat = y_lin + y_nl
    y_lin = b0 + b' x_t                     lineáris rész: pontosan a vele azonos HAR-X modell OLS-e
    y_nl  = LSTM(x_(t-L+1), …, x_t)         nemlineáris korrekció, az OLS-reziduumokra tanítva

Minden újrabecslési napon:
  1. OLS a tanulóablakon (i−ablak … i−1), ugyanúgy, mint a HAR-X-ben;
  2. az LSTM a tanulóablak OLS-reziduumain tanul: első alkalommal és negyedévente nulláról
     (cold start), egyébként az előző súlyokból folytatva (warm start);
     KORAI LEÁLLÍTÁS: a tanulóablak utolsó 20%-a (időrendben) validációs rész; a legjobb állapotot tartjuk meg;
     a kimeneti réteg nullával indul, így a modell pontosan a HAR-X-ből indul, és csak akkor tér el tőle,
     ha az a validációs részen javít (túlillesztés ellen);
  3. az előrejelzés = lineáris rész + az 5 seed korrekciójának átlaga.
A visszaváltás ugyanazzal az s²-tel történik, mint a HAR-X-nél (az OLS reziduális varianciája).
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import torch
from torch import nn

from config import FREQUENCIES, LSTM_PARAMS, MEASURES, TRANSFORMATIONS, WINDOWS
from forecasting import refit_flags


class CorrectionNet(nn.Module):
    """Egy LSTM-réteg és egy lineáris kimenet: a szekvencia utolsó rejtett állapotából a korrekció."""

    def __init__(self, n_features: int, hidden: int):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, batch_first=True)
        self.out = nn.Linear(hidden, 1)
        nn.init.zeros_(self.out.weight)       # nulla kezdő korrekció: a modell a HAR-X-ből indul
        nn.init.zeros_(self.out.bias)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        h, _ = self.lstm(seq)                 # (batch, L, hidden)
        return self.out(h[:, -1, :]).squeeze(-1)


def sequences(X: np.ndarray, idx: np.ndarray, L: int) -> np.ndarray:
    """Az idx soraihoz tartozó szekvenciák: a sor és az előtte lévő L−1 sor jellemzői -> (len(idx), L, k)."""
    return X[idx[:, None] + np.arange(-L + 1, 1)[None, :]]


def _train(net, opt, xs, ys, max_steps, params):
    """Teljes batch-es tanítás korai leállítással.

    A minta utolsó `holdout` hányada (időrendben a legfrissebb napok) validációs rész. `eval_every`
    lépésenként mérjük rajta a hibát; ha `patience` mérésen át nem javul, leállunk, és a legjobb
    állapotot (akár a kiindulót) állítjuk vissza.
    """
    n_val = max(1, int(len(ys) * params["holdout"]))
    xt, yt, xv, yv = xs[:-n_val], ys[:-n_val], xs[-n_val:], ys[-n_val:]

    def val_loss():
        net.eval()
        with torch.no_grad():
            return float(torch.mean((net(xv) - yv) ** 2))

    best, best_state, bad = val_loss(), {k: v.clone() for k, v in net.state_dict().items()}, 0
    for step in range(1, max_steps + 1):
        net.train()
        opt.zero_grad()
        loss = torch.mean((net(xt) - yt) ** 2)
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        if step % params["eval_every"] == 0:
            v = val_loss()
            if v < best - 1e-12:
                best, best_state, bad = v, {k: t.clone() for k, t in net.state_dict().items()}, 0
            else:
                bad += 1
                if bad >= params["patience"]:
                    break
    net.load_state_dict(best_state)


def walk_forward(X: np.ndarray, y: np.ndarray, dates: pd.DatetimeIndex, window: int, frequency: str,
                 params: dict = LSTM_PARAMS, start: int | None = None, stop: int | None = None):
    """HAR-X + LSTM-korrekció walk-forward előrejelzés. Visszaad: mu, s2, mu_lin (a lineáris rész)."""
    L, hidden, lr = params["seq_len"], params["hidden"], params["lr"]
    n, k = X.shape
    Xc = np.column_stack([np.ones(n), X])
    refit = refit_flags(dates, frequency)
    first = max(window, start or 0)
    last = n if stop is None else stop
    mu, mu_lin, s2 = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)

    nets, opts = [None] * params["seeds"], [None] * params["seeds"]
    beta, var, n_refit = None, np.nan, 0
    zx_mu = zx_sd = None
    e_sd = 1.0
    for i in range(first, last):
        if beta is None or refit[i]:
            # --- 1. Lineáris rész: OLS a tanulóablakon (azonos a HAR-X-szel) ------------------
            rows = np.arange(i - window, i)
            beta, *_ = np.linalg.lstsq(Xc[rows], y[rows], rcond=None)
            resid = y[rows] - Xc[rows] @ beta
            var = float(resid @ resid) / (window - Xc.shape[1])

            # --- 2. Nemlineáris rész: az LSTM a reziduumokra tanul ----------------------------
            rows_seq = rows[rows >= L - 1]                    # teljes szekvencia csak innen van
            zx_mu = X[rows].mean(axis=0)                      # standardizálás csak a tanulóablakból
            zx_sd = np.where(X[rows].std(axis=0) > 1e-12, X[rows].std(axis=0), 1.0)
            e = resid[rows >= L - 1]
            e_sd = float(e.std()) if e.std() > 1e-12 else 1.0
            xs = torch.tensor((sequences(X, rows_seq, L) - zx_mu) / zx_sd, dtype=torch.float32)
            ys = torch.tensor(e / e_sd, dtype=torch.float32)
            cold = n_refit % params["cold_every"][frequency] == 0
            for s in range(params["seeds"]):
                if cold:
                    torch.manual_seed(params["base_seed"] + s)
                    nets[s] = CorrectionNet(k, hidden)
                    opts[s] = torch.optim.AdamW(nets[s].parameters(), lr=lr, weight_decay=params["weight_decay"])
                _train(nets[s], opts[s], xs, ys, params["epochs_cold"] if cold else params["epochs_warm"], params)
            n_refit += 1

        # --- 3. Előrejelzés: lineáris rész + a seedek átlagos korrekciója -----------------------
        lin = float(Xc[i] @ beta)
        xi = torch.tensor((sequences(X, np.array([i]), L) - zx_mu) / zx_sd, dtype=torch.float32)
        with torch.no_grad():
            corr = np.mean([float(net.eval()(xi)) for net in nets]) * e_sd
        mu_lin[i], mu[i], s2[i] = lin, lin + corr, var
    return mu, s2, mu_lin


def _run_one(args):
    """Egy LSTM-modell egy processzormagon; ugyanaz a hívás, mint a modell saját model.py-jában."""
    from runner import run_model
    level, tr, m, w, f = args
    torch.set_num_threads(1)                     # egy modell = egy processzormag
    run_model("LSTM", level, tr, w, f, m)
    return f"LSTM_{level} {tr} {w}_{f} {m}"


def _done(job) -> bool:
    from config import model_dir
    level, tr, m, w, f = job
    return (model_dir("LSTM", level, tr, w, f, m) / "forecasts.csv").exists()


def run_jobs(jobs, workers: int | None = None) -> list[str]:
    """Modellek párhuzamos futtatása; a már kész modelleket kihagyja (megszakítás után folytatható)."""
    from multiprocessing import Pool
    todo = [j for j in jobs if not _done(j)]
    todo.sort(key=lambda j: (j[4] != "Weekly", j[3] != "3Y"))   # a leglassabbak indulnak először
    with Pool(workers or max(1, os.cpu_count() - 2)) as pool:
        out = []
        for name in pool.imap_unordered(_run_one, todo):
            print(name, "kész", flush=True)
            out.append(name)
        return out


def level_jobs(level: str):
    """Egy lépcső 32 modellje: 2 transzformáció × 4 mérték × 2 ablak × 2 gyakoriság."""
    return [(level, tr, m, w, f) for tr in TRANSFORMATIONS for m in MEASURES for w in WINDOWS for f in FREQUENCIES]


def run_level(level: str, workers: int | None = None) -> list[str]:
    return run_jobs(level_jobs(level), workers)
