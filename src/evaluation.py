"""Kiértékelés: hibamérők, Giacomini–White teszt, MCS és a kritikus időszakok.

Minden mérés szórás-skálán történik: sigma_true = sqrt(RV) a következő napon, sigma_hat = a modell
visszaváltott előrejelzése (docs/02_keretrendszer_vazlat.md).

  QLIKE  = sigma/sigma_hat − ln(sigma/sigma_hat) − 1    (fő mérő; Patton 2011)
  MSE    = (sigma − sigma_hat)^2
  R2_OOS = 1 − MSE(modell) / MSE(benchmark)             (Campbell–Thompson 2008)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

# --- Kritikus időszakok (docs/02_keretrendszer_vazlat.md) -----------------------------------------
CRISES = {
    "COVID": ("2020-02-20", "2020-06-30"),
    "Ukraine_war": ("2022-02-24", "2022-06-30"),
}
TOP_SHARE = 0.05      # a legmagasabb realizált volatilitású napok aránya


def qlike(sigma, sigma_hat):
    r = np.asarray(sigma) / np.asarray(sigma_hat)
    return r - np.log(r) - 1.0


def mse(sigma, sigma_hat):
    return (np.asarray(sigma) - np.asarray(sigma_hat)) ** 2


def subsamples(dates: pd.DatetimeIndex, sigma: np.ndarray) -> dict[str, np.ndarray]:
    """A kiértékelés részmintái logikai maszkként: teljes, 2020 nélkül, válságok, legviharosabb 5%."""
    masks = {"full": np.ones(len(dates), bool), "ex_2020": dates.year != 2020}
    for name, (a, b) in CRISES.items():
        masks[name] = (dates >= a) & (dates <= b)
    masks["top5pct_days"] = sigma >= np.quantile(sigma, 1 - TOP_SHARE)
    return masks


def newey_west_var(d: np.ndarray, lag: int | None = None) -> float:
    """A d átlagának hosszú távú varianciája (Newey–West, Bartlett-súlyok)."""
    d = d - d.mean()
    T = len(d)
    lag = int(np.floor(4 * (T / 100.0) ** (2.0 / 9.0))) if lag is None else lag
    v = d @ d / T
    for k in range(1, lag + 1):
        v += 2 * (1 - k / (lag + 1)) * (d[k:] @ d[:-k]) / T
    return v / T


def gw_test(loss_small: np.ndarray, loss_big: np.ndarray) -> dict:
    """Giacomini–White (2006) feltétel nélküli teszt, egyoldalú: H1 = a `big` modell kisebb hibájú.

    d = loss_small − loss_big; t = átlag(d) / sqrt(NW-variancia); p = 1 − Phi(t).
    Gördülő ablakos becslésnél és egymásba ágyazott modelleknél is érvényes.
    """
    d = np.asarray(loss_small) - np.asarray(loss_big)
    se = np.sqrt(newey_west_var(d))
    t = d.mean() / se if se > 0 else 0.0
    return {"mean_diff": float(d.mean()), "t": float(t), "p_one_sided": float(1 - norm.cdf(t)),
            "ratio": float(np.mean(loss_big) / np.mean(loss_small))}
