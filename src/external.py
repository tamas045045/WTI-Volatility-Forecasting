"""Külső napi adatok (L2 makró, L3 cross-asset, L4 hírek) beolvasása és kereskedési naphoz rendelése.

Időzítési szabály (a WTI-projekt protokollja szerint): minden megfigyelésnek van egy elérhetőségi
időpontja (tau = a záróérték időpontja, New York-i idő). A D kereskedési nap előrejelzése t_D-kor
készül (normál napon 17:00); D-hez a legutolsó olyan megfigyelés tartozik, amelyre tau < t_D (szigorúan).
Így egyetlen előrejelzésbe sem kerülhet jövőbeli adat.

Minden külső piacból napi szórás készül, a célváltozóval azonos skálán:
  OVX, VIX     : implikált éves volatilitás (%) -> napi szórás = érték / 100 / sqrt(252)
  többi piac   : Parkinson-szórás a napi OHLC-ből = |ln(H/L)| / sqrt(4 ln 2)
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd

from config import NY_TZ, RAW_CROSSASSET, RAW_MARKET

LN2 = np.log(2.0)
MAX_STALENESS_DAYS = 7   # ennél régebbi megfigyelést már nem használunk (pl. tartós adathiány)

# Fájl, típus és záróidő (tau, New York-i idő) sorozatonként.
# A cross-asset futures-nél a napi „Price” időpontja nem dokumentált, ezért konzervatívan 17:00-t
# veszünk: így az érték mindig csak a KÖVETKEZŐ kereskedési napon kerül a modellbe (nincs szivárgás).
SERIES = {
    # név:      (mappa, fájl, formátum, mit mér, záróidő)
    "ovx":      ("market", "OVXCLS.csv", "fred", "implied", "16:30"),
    "vix":      ("market", "Volatility_Index.csv", "investing", "implied", "16:15"),
    "sp500":    ("market", "SP500.csv", "investing", "range", "16:00"),
    "usd":      ("market", "USD_Index.csv", "investing", "range", "17:00"),
    "copper":   ("crossasset", "Copper_Futures.csv", "investing", "range", "17:00"),
    "gold":     ("crossasset", "Gold_Futures.csv", "investing", "range", "17:00"),
    "natgas":   ("crossasset", "NaturalGas_Futures.csv", "investing", "range", "17:00"),
    "wheat":    ("crossasset", "Wheat_Futures.csv", "investing", "range", "17:00"),
    "corn":     ("crossasset", "Corn_Futures.csv", "investing", "range", "17:00"),
    "soybeans": ("crossasset", "Soybeans_Futures.csv", "investing", "range", "17:00"),
}


def _number(s: pd.Series) -> pd.Series:
    """' 4,140.80 ' -> 4140.80; üres vagy '-' -> NaN."""
    s = s.astype(str).str.strip().str.replace(",", "", regex=False)
    return pd.to_numeric(s.replace({"": np.nan, "-": np.nan, ".": np.nan}), errors="raise")


def read_series(name: str) -> pd.DataFrame:
    """Egy sorozat beolvasása: date + sigma (napi szórás). Az értelmezhetetlen napok kimaradnak."""
    folder, fname, fmt, kind, _ = SERIES[name]
    path = (RAW_MARKET if folder == "market" else RAW_CROSSASSET) / fname
    text = path.read_text(encoding="utf-8").lstrip("﻿")
    if fmt == "fred":
        # FRED-formátum: observation_date,OVXCLS
        raw = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)
        date = pd.to_datetime(raw.iloc[:, 0].str.strip(), format="%Y-%m-%d")
        close = high = low = _number(raw.iloc[:, 1])
    else:
        # Investing.com-formátum: Date;Price;Open;High;Low;Vol.;Change %
        raw = pd.read_csv(io.StringIO(text), sep=";", dtype=str, keep_default_na=False)
        raw.columns = [c.strip() for c in raw.columns]
        date = pd.to_datetime(raw["Date"].str.strip(), format="%m/%d/%Y")
        close, high, low = _number(raw["Price"]), _number(raw["High"]), _number(raw["Low"])
    if kind == "implied":
        # Implikált éves volatilitás (%) -> napi szórás
        sigma = close / 100.0 / np.sqrt(252.0)
    else:
        # Parkinson-szórás a napi terjedelemből; a nulla terjedelmű nap (H <= L) hiányzó adatnak számít
        sigma = (np.abs(np.log(high / low)) / np.sqrt(4.0 * LN2)).where(high > low)
    out = pd.DataFrame({"date": date, "sigma": sigma}).dropna()
    if out["date"].duplicated().any():
        raise ValueError(f"{name}: duplikált dátum")
    return out.sort_values("date").reset_index(drop=True)


def align_to_trading_days(obs: pd.DataFrame, close_hhmm: str, t_d: pd.Series) -> pd.DataFrame:
    """Minden D kereskedési naphoz a legutolsó megfigyelés, amelyre tau < t_D (szigorúan).

    obs: date + sigma; t_d: a kereskedési napok előrejelzési időpontja (tz-aware, index: trading_day).
    Visszaad: sigma, obs_date (melyik napi értéket használtuk) a kereskedési napok indexén.
    """
    hh, mm = (int(x) for x in close_hhmm.split(":"))
    tau = (obs["date"] + pd.Timedelta(hours=hh, minutes=mm)).dt.tz_localize(NY_TZ).dt.tz_convert("UTC")
    utc_ns = "datetime64[ns, UTC]"   # egységes időfelbontás a két oldalon
    left = pd.DataFrame({"trading_day": t_d.index, "t": t_d.dt.tz_convert("UTC").astype(utc_ns)}).sort_values("t")
    right = pd.DataFrame({"t": tau.astype(utc_ns).to_numpy(), "sigma": obs["sigma"].to_numpy(),
                          "obs_date": obs["date"].to_numpy()})
    # allow_exact_matches=False -> tau < t_D (a t_D-vel egyidejű érték még nem ismert)
    m = pd.merge_asof(left, right.sort_values("t"), on="t", direction="backward", allow_exact_matches=False)
    m = m.set_index("trading_day").sort_index()
    # Túl régi megfigyelés helyett hiányzó érték
    stale = (m.index - pd.DatetimeIndex(m["obs_date"])).days > MAX_STALENESS_DAYS
    m.loc[stale, "sigma"] = np.nan
    return m[["sigma", "obs_date"]]


def external_daily(t_d: pd.Series) -> pd.DataFrame:
    """Az L2 és L3 összes változója (napi szórás) kereskedési naponként, plusz a használt dátumok."""
    cols = {}
    for name, (*_, close) in SERIES.items():
        a = align_to_trading_days(read_series(name), close, t_d)
        cols[name] = a["sigma"]
        cols[f"{name}_obs_date"] = a["obs_date"]
    return pd.DataFrame(cols)


def news_daily(gkg: pd.DataFrame) -> pd.DataFrame:
    """Az L4 három hírváltozója (a WTI-projektben már kereskedési naphoz rendelve, t_D előtti cikkekből).

    Ha a nap hírablaka nem volt elérhető (gkg_missing), a változók hiányzók (nem töltjük ki).
    """
    out = pd.DataFrame({
        "news_abnormal": gkg["gkg_oil_abnormal"],
        "news_tone": gkg["gkg_oil_tone"],
        "news_tone_disp": gkg["gkg_oil_tone_disp"],
    })
    return out.where(~gkg["gkg_missing"].astype(bool))
