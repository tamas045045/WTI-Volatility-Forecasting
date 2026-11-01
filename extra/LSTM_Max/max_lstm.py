"""Maximalizált LSTM – kísérlet, a 256-os kereten kívül. A meglévő kódhoz nem nyúl.

Két változat (külön almappa, külön hangolás):
  Strict/  szigorú maximum: a háló a szintet tanulja (nincs horgony), 3 éves gördülő ablak – mint a tiszta LSTM;
           csak a háló felépítése és a tanítás maximalizált.
  Free/    szabad maximum: a horgony a mai volatilitáshoz jelölt (a validáció dönti el), és bővülő ablak
           (mindig a közös minta elejétől); a horgony nélküli változatok is a jelöltek között vannak.

    horgony nélkül:  y_hat = átlag(y a tanulóablakon) + LSTM(...)
    horgonnyal:      y_hat = har_d (a mérték mai, transzformált szórása) + átlag(y − har_d) + LSTM(...)
A horgonyos változat a holnapi szórás VÁLTOZÁSÁT tanulja a mai szinthez képest (szabványos „skip connection”);
OLS/HAR-regresszió egyik változatban sincs.

Ami azonos a többi LSTM-mel: input tábla és változók, cél, standardizálás csak a tanulóablakból, korai leállítás
(src/lstm.py _train, változatlanul), súlycsökkentés, cold/warm start, seedek átlaga. A visszaváltás s²-e a háló
saját tanulóablakbeli hibáinak varianciája (mint a tiszta LSTM-nél).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))   # a projekt közös motorja (csak olvasva)

import numpy as np
import pandas as pd
import torch
from torch import nn

from forecasting import refit_flags
from lstm import _train, sequences


class MaxNet(nn.Module):
    """1–2 rétegű LSTM, dropouttal; nulla kezdő kimenet (a modell a kiindulópontból indul)."""

    def __init__(self, n_features: int, hidden: int, layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers=layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        h, _ = self.lstm(seq)
        return self.out(self.drop(h[:, -1, :])).squeeze(-1)


def walk_forward_max(X: np.ndarray, y: np.ndarray, dates: pd.DatetimeIndex, window: int | None, frequency: str,
                     params: dict, anchor: int | None = None, start: int | None = None, stop: int | None = None,
                     log=None):
    """Walk-forward előrejelzés. window=None: bővülő ablak (a minta elejétől); anchor: a horgonyoszlop indexe X-ben.

    Visszaad: mu (transzformált előrejelzés), s2 (a visszaváltás varianciája).
    """
    L, hidden, lr = params["seq_len"], params["hidden"], params["lr"]
    n, k = X.shape
    refit = refit_flags(dates, frequency)
    first = max(window or 0, start or 0)
    last = n if stop is None else stop
    mu, s2 = np.full(n, np.nan), np.full(n, np.nan)
    a = X[:, anchor] if anchor is not None else np.zeros(n)        # horgony nélkül: 0

    nets, opts = [None] * params["seeds"], [None] * params["seeds"]
    c, var, n_refit = np.nan, np.nan, 0
    n_total = sum(1 for i in range(first, last) if i == first or refit[i])
    zx_mu = zx_sd = None
    e_sd = 1.0
    for i in range(first, last):
        if n_refit == 0 or refit[i]:
            rows = np.arange(0 if window is None else i - window, i)       # csak múltbeli sorok
            rows = rows[rows >= L - 1]                                      # teljes szekvencia csak innen van
            d = y[rows] - a[rows]                  # horgonnyal: változás a mai szinthez képest; nélküle: a szint
            c = float(d.mean())                    # kiindulópont: az ablak átlaga
            e = d - c
            zx_mu = X[rows].mean(axis=0)
            zx_sd = np.where(X[rows].std(axis=0) > 1e-12, X[rows].std(axis=0), 1.0)
            e_sd = float(e.std()) if e.std() > 1e-12 else 1.0
            xs = torch.tensor((sequences(X, rows, L) - zx_mu) / zx_sd, dtype=torch.float32)
            ys = torch.tensor(e / e_sd, dtype=torch.float32)
            cold = n_refit % params["cold_every"][frequency] == 0
            for s in range(params["seeds"]):
                if cold:
                    torch.manual_seed(params["base_seed"] + s)
                    nets[s] = MaxNet(k, hidden, params["layers"], params["dropout"])
                    opts[s] = torch.optim.AdamW(nets[s].parameters(), lr=lr, weight_decay=params["weight_decay"])
                _train(nets[s], opts[s], xs, ys, params["epochs_cold"] if cold else params["epochs_warm"], params)
            with torch.no_grad():
                fit = np.mean([net.eval()(xs).numpy().astype(float) for net in nets], axis=0) * e_sd
            var = float(np.mean((e - fit) ** 2))
            n_refit += 1
            if log:
                log(n_refit, n_total, dates[i])

        xi = torch.tensor((sequences(X, np.array([i]), L) - zx_mu) / zx_sd, dtype=torch.float32)
        with torch.no_grad():
            out = np.mean([float(net.eval()(xi)) for net in nets]) * e_sd
        mu[i], s2[i] = a[i] + c + out, var
    return mu, s2
