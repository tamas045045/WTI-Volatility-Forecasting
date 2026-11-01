"""Tiszta LSTM (HAR-X nélkül) – kiegészítő összevetés, egyetlen modell, a 256-os kereten kívül.

    y_hat = átlag(y a tanulóablakon) + LSTM(x_(t-L+1), …, x_t)

A felépítés lépésről lépésre a src/lstm.py walk_forward() függvényét követi, és a közös építőelemeit
(CorrectionNet, sequences, _train) változtatás nélkül használja. A meglévő kódhoz ez a fájl nem nyúl.
Az eltérések (ELTÉRÉS kommentek):
  1. nincs OLS: a háló a teljes célt tanulja, a tanulóablak átlagától vett eltérésként; a nulla kezdő
     kimenet így az ablak átlagát jelzi előre (a hibridnél a HAR-X-et);
  2. a visszaváltás s²-e a háló saját hibáinak varianciája a tanulóablakon (a hibridnél az OLS-reziduumoké).
Minden más azonos: szekvencia, standardizálás, korai leállítás, súlycsökkentés, 5 seed, cold/warm start.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))   # a projekt közös motorja

import numpy as np
import pandas as pd
import torch

from forecasting import refit_flags
from lstm import CorrectionNet, _train, sequences


def walk_forward_pure(X: np.ndarray, y: np.ndarray, dates: pd.DatetimeIndex, window: int, frequency: str,
                      params: dict, start: int | None = None, stop: int | None = None, log=None):
    """Tiszta LSTM walk-forward előrejelzés. Visszaad: mu (transzformált előrejelzés), s2.

    log: opcionális függvény, amely minden újrabecslés után megkapja (hányadik, összesen, dátum) – haladásnapló.
    """
    L, hidden, lr = params["seq_len"], params["hidden"], params["lr"]
    n, k = X.shape
    refit = refit_flags(dates, frequency)
    first = max(window, start or 0)
    last = n if stop is None else stop
    mu, s2 = np.full(n, np.nan), np.full(n, np.nan)

    nets, opts = [None] * params["seeds"], [None] * params["seeds"]
    base, var, n_refit = np.nan, np.nan, 0
    n_total = sum(1 for i in range(first, last) if i == first or refit[i])   # az újrabecslések száma
    zx_mu = zx_sd = None
    e_sd = 1.0
    for i in range(first, last):
        if n_refit == 0 or refit[i]:
            rows = np.arange(i - window, i)
            # --- 1. ELTÉRÉS: nincs lineáris rész; kiindulópont a tanulóablak átlaga -----------------
            base = float(y[rows].mean())
            resid = y[rows] - base

            # --- 2. Az LSTM a célra tanul (azonos eljárás, mint a hibridben) --------------------------
            rows_seq = rows[rows >= L - 1]
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

            # --- ELTÉRÉS: s² a háló saját hibáiból a tanulóablakon (itt nincs OLS-reziduum) ----------
            with torch.no_grad():
                fit = np.mean([net.eval()(xs).numpy().astype(float) for net in nets], axis=0) * e_sd
            var = float(np.mean((e - fit) ** 2))
            n_refit += 1
            if log:
                log(n_refit, n_total, dates[i])

        # --- 3. Előrejelzés: az ablak átlaga + a seedek átlagos kimenete ----------------------------
        xi = torch.tensor((sequences(X, np.array([i]), L) - zx_mu) / zx_sd, dtype=torch.float32)
        with torch.no_grad():
            out = np.mean([float(net.eval()(xi)) for net in nets]) * e_sd
        mu[i], s2[i] = base + out, var
    return mu, s2
