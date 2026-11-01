"""A közös input táblák: data/input/<Transformation>/<Measure>.csv

Egy tábla = egy transzformáció × egy mérték. Az L1–L4 összes oszlopát tartalmazza; a modell lépcsője
csak azt dönti el, mely oszlopokat használja (config.level_columns). Így a HAR és az LSTM garantáltan
ugyanazt az adatot kapja.

Feldolgozási lánc (docs/02_keretrendszer_vazlat.md):
  napi variancia -> sqrt -> napi szórás (sigma) -> transzformáció: log(sigma) vagy sqrt(sigma)
Cél:  y = T(sigma_next), ahol sigma_next = sqrt(RV) a következő kereskedési napon.
A kiértékelés mindig szórás-skálán történik (sigma_next vs. a visszaváltott előrejelzés).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import CROSSASSET_COLUMNS, MACRO_COLUMNS, NEWS_COLUMNS

# Transzformációk szórás-skálán
FORWARD = {
    "Log": np.log,           # log(sigma)
    "QuarticRoot": np.sqrt,  # sqrt(sigma) = variancia^(1/4)
}
HAR_WINDOWS = {"d": 1, "w": 5, "m": 22}   # Corsi (2009): napi, heti, havi tag


def har_sigma(variance: pd.Series, horizon: int) -> pd.Series:
    """HAR-tag: az utolsó `horizon` érvényes nap átlagos varianciájának gyöke (szórás-skála)."""
    return np.sqrt(variance.rolling(horizon).mean())


def build_input(measures: pd.DataFrame, target_sigma_next: pd.Series, external: pd.DataFrame,
                news: pd.DataFrame, transformation: str, measure: str) -> pd.DataFrame:
    """Egy input tábla (még a közös mintára szűrés előtt), index: trading_day (érvényes napok)."""
    T = FORWARD[transformation]
    out = pd.DataFrame(index=measures.index)
    out["sigma_next"] = target_sigma_next          # a kiértékelés célja (szint)
    out["y"] = T(target_sigma_next)                # a modell célja (transzformált)

    # --- L1: HAR-tagok a mérték saját múltjából -------------------------------------------
    if measure == "RV_CJ":
        # ELTÉRÉS: az RV-CJ a folytonos (C) és az ugrás (J) részt külön tagként kapja.
        # A C ugyanazt a transzformációt kapja; a J sqrt(J)-ként, transzformáció nélkül kerül be,
        # mert a napok többségén J = 0 (log(0) nem értelmezett).
        for k, h in HAR_WINDOWS.items():
            out[f"har_c_{k}"] = T(har_sigma(measures["C"], h))
            out[f"har_j_{k}"] = har_sigma(measures["J"], h)
    else:
        for k, h in HAR_WINDOWS.items():
            out[f"har_{k}"] = T(har_sigma(measures[measure], h))

    # --- L2 és L3: külső piacok napi szórása, ugyanazzal a transzformációval ----------------
    for c in MACRO_COLUMNS + CROSSASSET_COLUMNS:
        out[c] = T(external[c].reindex(out.index))

    # --- L4: hírváltozók (nem volatilitások, ezért transzformáció nélkül) ----------------------
    for c in NEWS_COLUMNS:
        out[c] = news[c].reindex(out.index)
    return out


def common_sample(tables: dict) -> pd.Index:
    """A közös minta: azok a napok, ahol MINDEN tábla minden oszlopa megvan.

    Így minden modell pontosan ugyanazokon a sorokon tanul és ugyanazokon értékelődik;
    a lépcsők közti különbség csak a változókból fakadhat.
    """
    idx = None
    for t in tables.values():
        ok = t.dropna().index
        idx = ok if idx is None else idx.intersection(ok)
    return idx.sort_values()
