"""A négy napi volatilitásmérték számítása a WTI 5 perces session-bárjaiból.

Bemenet: a WTI-projekt session-tábla (egy sor = egy 5 perces bár, `trading_day` szerint címkézve,
`r` = sessionön belüli log-hozam; a session-határon átnyúló hozam nincs benne).

Kimenet napi bontásban (csak az érvényes napokra értelmezzük):
  RV            = sum r^2                                     (realizált variancia)
  BPV           = (pi/2) * n/(n-1) * sum |r_i||r_(i-1)|      (bipower variation, BNS 2004)
  J, C          = szignifikáns ugrás és folytonos rész: J = max(RV - BPV, 0), ha a z-teszt
                  szignifikáns (ABD 2007, alpha = 0,999), különben 0; C = RV - J
  Parkinson     = sum_bárok ln(H/L)^2 / (4 ln 2)              (5 perces bárokból összegezve)
  Garman_Klass  = sum_bárok [0,5 ln(H/L)^2 - (2 ln 2 - 1) ln(C/O)^2]

A Parkinson és a GK az 5 perces bárokból összegzett („realized range”) változat;
a döntés oka: docs/02_keretrendszer_vazlat.md (projekten kívüli előpróba, 2026-10-10).
"""
from __future__ import annotations

from math import gamma, pi

import numpy as np
import pandas as pd
from scipy.stats import norm

LN2 = np.log(2.0)
MU1 = np.sqrt(2.0 / pi)                                    # E|Z|, Z ~ N(0,1)
MU43 = 2.0 ** (2.0 / 3.0) * gamma(7.0 / 6.0) / gamma(0.5)  # E|Z|^(4/3)
JUMP_ALPHA = 0.999                                         # az ugrásteszt szintje (ABD 2007)


def _per_day(g: pd.DataFrame) -> pd.Series:
    """Egy kereskedési nap mértékei a nap bárjaiból."""
    r = g["r"].to_numpy()
    n = len(r)
    ar = np.abs(r)
    rv = float(np.sum(r * r))
    # Bipower variation: a szomszédos abszolút hozamok szorzata kiszűri az ugrásokat
    bpv = (1.0 / MU1 ** 2) * (n / (n - 1.0)) * float(np.sum(ar[1:] * ar[:-1]))
    # Tri-power quarticity: az ugrásteszt varianciájához
    a43 = ar ** (4.0 / 3.0)
    tpq = n * (1.0 / MU43 ** 3) * (n / (n - 2.0)) * float(np.sum(a43[2:] * a43[1:-1] * a43[:-2]))
    # Ugrásteszt (ratio-statisztika, Huang–Tauchen 2005; ABD 2007): z ~ N(0,1) ugrás nélkül
    z = ((rv - bpv) / rv) / np.sqrt(((pi ** 2) / 4.0 + pi - 5.0) * max(1.0, tpq / bpv ** 2) / n)
    jump = max(rv - bpv, 0.0) if z > norm.ppf(JUMP_ALPHA) else 0.0
    # Range-alapú mértékek az 5 perces bárok terjedelméből, napon belül összegezve
    hl = np.log(g["high"].to_numpy() / g["low"].to_numpy())
    co = np.log(g["close"].to_numpy() / g["open"].to_numpy())
    parkinson = float(np.sum(hl * hl) / (4.0 * LN2))
    garman_klass = float(np.sum(0.5 * hl * hl - (2.0 * LN2 - 1.0) * co * co))
    return pd.Series({"RV": rv, "BPV": bpv, "J": jump, "C": rv - jump, "jump_z": z,
                      "Parkinson": parkinson, "Garman_Klass": garman_klass, "n_bars": n})


def daily_measures(bars: pd.DataFrame, valid_days: pd.Index) -> pd.DataFrame:
    """Napi mértékek az érvényes kereskedési napokra (index: trading_day)."""
    if (bars[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("Nem pozitív ár a bárokban: a log-alapú mértékek nem értelmezhetők.")
    b = bars[bars["trading_day"].isin(valid_days)]
    out = b.groupby("trading_day", sort=True).apply(_per_day, include_groups=False)
    out.index.name = "trading_day"
    return out
