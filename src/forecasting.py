"""Közös előrejelzési eszközök a HAR és az LSTM számára.

  * load_model_data : az input tábla betöltése egy modellhez (a lépcső oszlopaival)
  * refit_flags     : mely sorokon kell újrabecsülni (heti / havi naptári váltás)
  * back_transform  : a transzformált előrejelzés visszaváltása napi szórásra, torzításkorrekcióval
  * save_forecasts  : az előrejelzések mentése a modell levélmappájába

Időzítési konvenció (mindkét családnál azonos): az i. sor jellemzői a t_i nap zárásakor ismertek,
a célja a következő kereskedési nap szórása. Az i. sor előrejelzéséhez a modell csak az
i−ablak … i−1 sorokon tanulhat; ezek célja már a t_i napon ismert.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import FREQUENCIES, INPUT, level_columns, model_dir


def load_model_data(level: str, transformation: str, measure: str):
    """Visszaad: X (jellemzők a lépcső szerint), y (transzformált cél), sigma_next (szint), dátumok."""
    t = pd.read_csv(INPUT / transformation / f"{measure}.csv", index_col="date", parse_dates=True)
    cols = level_columns(level, measure)
    return t[cols].to_numpy(float), t["y"].to_numpy(float), t["sigma_next"].to_numpy(float), t.index, cols


def refit_flags(dates: pd.DatetimeIndex, frequency: str) -> np.ndarray:
    """True ott, ahol új naptári hét (Weekly) vagy hónap (Monthly) kezdődik."""
    period = dates.to_period(FREQUENCIES[frequency]).asi8
    flags = np.ones(len(dates), dtype=bool)
    flags[1:] = period[1:] != period[:-1]
    return flags


def back_transform(transformation: str, mu: np.ndarray, s2: np.ndarray) -> np.ndarray:
    """Transzformált előrejelzés -> napi szórás, normális hibát feltételező torzításkorrekcióval.

    Log:          sigma_hat = exp(mu + s2/2)        (lognormális várható érték)
    QuarticRoot:  sigma_hat = mu^2 + s2             (E[(mu + e)^2], mert sigma = (sqrt sigma)^2)
    s2 = a tanulóablak reziduumainak varianciája.
    """
    if transformation == "Log":
        return np.exp(mu + 0.5 * s2)
    if transformation == "QuarticRoot":
        mu = np.maximum(mu, 0.0)        # sqrt(sigma) nem lehet negatív
        return mu ** 2 + s2
    raise ValueError(transformation)


def save_forecasts(family, level, transformation, window, frequency, measure, dates, sigma_true, y_true,
                   mu, s2, extra: dict | None = None) -> pd.DataFrame:
    """forecasts.csv a modell mappájába: dátum, valódi és előrejelzett szórás, transzformált értékek."""
    out = pd.DataFrame({
        "sigma_true": sigma_true, "sigma_hat": back_transform(transformation, mu, s2),
        "y_true": y_true, "y_hat": mu, "resid_var": s2,
    }, index=pd.DatetimeIndex(dates, name="date"))
    out = out.dropna(subset=["y_hat"])
    d = model_dir(family, level, transformation, window, frequency, measure)
    d.mkdir(parents=True, exist_ok=True)
    out.to_csv(d / "forecasts.csv")
    for name, frame in (extra or {}).items():
        frame.to_csv(d / name)
    return out
