"""Egyetlen modell futtatása a beállításai alapján (ezt hívja minden models/.../model.py).

    run_model("HAR", "L2", "Log", "3Y", "Weekly", "RV")

A HAR és az LSTM itt ugyanazzal a mentéssel és ugyanazzal az input táblával fut, mint a
lépcsőnkénti futtatókban (models/<modell>/run.py), így az eredmény bitre azonos.
"""
from __future__ import annotations

import pandas as pd

from config import WINDOWS
from forecasting import load_model_data, save_forecasts


def run_model(family: str, level: str, transformation: str, window: str, frequency: str, measure: str) -> pd.DataFrame:
    if family == "HAR":
        from har import rolling_ols
        X, y, sigma, dates, names = load_model_data(level, transformation, measure)
        mu, s2, coefs = rolling_ols(X, y, dates, WINDOWS[window], frequency, names)
        return save_forecasts("HAR", level, transformation, window, frequency, measure, dates, sigma, y, mu, s2,
                              extra={"coefficients.csv": coefs})
    if family == "LSTM":
        from lstm import walk_forward
        X, y, sigma, dates, _ = load_model_data(level, transformation, measure)
        mu, s2, mu_lin = walk_forward(X, y, dates, WINDOWS[window], frequency)
        lin = pd.DataFrame({"y_lin": mu_lin}, index=pd.DatetimeIndex(dates, name="date")).dropna()
        return save_forecasts("LSTM", level, transformation, window, frequency, measure, dates, sigma, y, mu, s2,
                              extra={"linear_part.csv": lin})
    raise ValueError(family)
