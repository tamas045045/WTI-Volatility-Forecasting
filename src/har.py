"""HAR-X: gördülő ablakos OLS, egynapos előrejelzés.

    y_(t+1) = b0 + b' x_t + e,   x_t = a lépcső változói (HAR-tagok + makró + cross-asset + hír)

A becslés a tanulóablak (az utolsó `window` sor) adatain történik, és csak a heti/havi újrabecslési
napokon frissül; a kettő között az együtthatók változatlanok.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import FREQUENCIES, MEASURES, TRANSFORMATIONS, WINDOWS
from forecasting import refit_flags


def rolling_ols(X: np.ndarray, y: np.ndarray, dates: pd.DatetimeIndex, window: int, frequency: str,
                names: list[str]):
    """Visszaad: mu (transzformált előrejelzés), s2 (reziduális variancia), coefs (újrabecslésenként)."""
    n, k = X.shape
    Xc = np.column_stack([np.ones(n), X])            # konstans + magyarázó változók
    refit = refit_flags(dates, frequency)
    mu = np.full(n, np.nan)
    s2 = np.full(n, np.nan)
    coef_rows = []
    beta, var = None, np.nan
    for i in range(window, n):
        if beta is None or refit[i]:
            # Tanulás csak a múltbeli sorokon: i-window … i-1
            Xt, yt = Xc[i - window:i], y[i - window:i]
            beta, *_ = np.linalg.lstsq(Xt, yt, rcond=None)
            resid = yt - Xt @ beta
            var = float(resid @ resid) / (window - Xc.shape[1])   # torzítatlan reziduális variancia
            coef_rows.append([dates[i], *beta, var])
        mu[i] = Xc[i] @ beta
        s2[i] = var
    coefs = pd.DataFrame(coef_rows, columns=["refit_date", "const", *names, "resid_var"]).set_index("refit_date")
    return mu, s2, coefs


def run_level(level: str) -> list[str]:
    """Egy HAR-lépcső mind a 32 modellje: 2 transzformáció × 2 ablak × 2 gyakoriság × 4 mérték."""
    from runner import run_model                 # ugyanaz a hívás, mint a modell saját model.py-jában
    done = []
    for tr in TRANSFORMATIONS:
        for m in MEASURES:
            for w in WINDOWS:
                for f in FREQUENCIES:
                    run_model("HAR", level, tr, w, f, m)
                    done.append(f"HAR_{level} {tr} {w}_{f} {m}")
    return done
