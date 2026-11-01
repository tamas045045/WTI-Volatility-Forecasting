"""Az adatréteg tesztjei: képletek, időzítés (nincs jövőbeli adat), egyezés a WTI-projekttel.

Futtatás:  python -m pytest tests
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import pytest

from config import INPUT, MEASURES, PROCESSED, TRANSFORMATIONS, WTI_PIPELINE
from external import SERIES, align_to_trading_days, read_series
from inputs import FORWARD, build_input
from measures import daily_measures

WTI_MASTER = Path(r"C:\Users\Tamás\WTI_News_Absorption\data\processed\master_daily_gkg.parquet")


@pytest.fixture(scope="module")
def wti():
    return pd.read_parquet(WTI_PIPELINE / "wti_daily.parquet")


# --- 1. Képletek ------------------------------------------------------------------------------
def test_range_formulas_on_synthetic_bars():
    # Két bár egy napon: a Parkinson és a GK kézzel számolt értéke
    bars = pd.DataFrame({"trading_day": pd.Timestamp("2020-01-02"), "r": [0.01, -0.02, 0.005],
                         "open": [100, 101, 99], "high": [102, 101.5, 100], "low": [99, 98.5, 98.8],
                         "close": [101, 99, 99.5]})
    m = daily_measures(bars, pd.Index([pd.Timestamp("2020-01-02")])).iloc[0]
    hl = np.log(bars["high"] / bars["low"]); co = np.log(bars["close"] / bars["open"])
    assert m["Parkinson"] == pytest.approx((hl ** 2).sum() / (4 * np.log(2)))
    assert m["Garman_Klass"] == pytest.approx((0.5 * hl ** 2 - (2 * np.log(2) - 1) * co ** 2).sum())
    assert m["RV"] == pytest.approx(0.01 ** 2 + 0.02 ** 2 + 0.005 ** 2)
    assert m["C"] + m["J"] == pytest.approx(m["RV"])


def test_jump_split_adds_up():
    meas = pd.read_csv(PROCESSED / "measures_daily.csv", index_col=0)
    assert np.allclose(meas["C"] + meas["J"], meas["RV"])
    assert (meas["J"] >= 0).all() and (meas["C"] > 0).all()
    assert (meas[["Parkinson", "Garman_Klass"]] > 0).all().all()


# --- 2. Időzítés: nincs jövőbeli adat ---------------------------------------------------------
@pytest.mark.parametrize("name", list(SERIES))
def test_external_value_known_before_forecast(wti, name):
    # Minden felhasznált megfigyelés záróideje szigorúan a t_D előtt van
    a = align_to_trading_days(read_series(name), SERIES[name][-1], wti["t_D"]).dropna()
    hh, mm = (int(x) for x in SERIES[name][-1].split(":"))
    tau = (pd.DatetimeIndex(a["obs_date"]) + pd.Timedelta(hours=hh, minutes=mm)).tz_localize("America/New_York")
    assert (tau < wti.loc[a.index, "t_D"].to_numpy()).all()


@pytest.mark.skipif(not WTI_MASTER.exists(), reason="a WTI-projekt master táblája nem elérhető")
@pytest.mark.parametrize("name", ["ovx", "sp500", "usd"])
def test_alignment_matches_wti_project(wti, name):
    # Ugyanazt a napi értéket rendeljük minden naphoz, mint a WTI-projekt auditált pipeline-ja
    master = pd.read_parquet(WTI_MASTER)
    a = align_to_trading_days(read_series(name), SERIES[name][-1], wti["t_D"])
    both = master[f"{name}_obs_date"].notna() & a["obs_date"].notna()
    assert (pd.DatetimeIndex(master.loc[both, f"{name}_obs_date"]) == pd.DatetimeIndex(a.loc[both, "obs_date"])).mean() > 0.995


def test_inputs_ignore_future_data():
    # „Mérgezési” teszt: ha egy dátum utáni adatot elrontunk, a dátumig tartó input sorok nem változhatnak
    meas = pd.read_csv(PROCESSED / "measures_daily.csv", index_col=0, parse_dates=True)
    target = pd.read_csv(PROCESSED / "target_daily.csv", index_col=0, parse_dates=True)
    ext = pd.read_csv(PROCESSED / "external_daily.csv", index_col=0, parse_dates=True)
    news = pd.read_csv(PROCESSED / "news_daily.csv", index_col=0, parse_dates=True)
    cut = pd.Timestamp("2020-06-15")
    clean = build_input(meas, target["sigma_next"], ext, news, "Log", "RV_CJ")
    poisoned_meas = meas.copy(); poisoned_meas.loc[poisoned_meas.index > cut, ["RV", "C", "J"]] *= 50
    poisoned_ext = ext.copy(); poisoned_ext.loc[poisoned_ext.index > cut, "ovx"] *= 50
    dirty = build_input(poisoned_meas, target["sigma_next"], poisoned_ext, news, "Log", "RV_CJ")
    features = [c for c in clean.columns if c not in ("y", "sigma_next")]
    pd.testing.assert_frame_equal(clean.loc[:cut, features], dirty.loc[:cut, features])


# --- 3. A kész input táblák -------------------------------------------------------------------
def test_input_tables_consistent():
    meas = pd.read_csv(PROCESSED / "measures_daily.csv", index_col=0, parse_dates=True)
    target = pd.read_csv(PROCESSED / "target_daily.csv", index_col=0, parse_dates=True)
    dates = None
    for tr in TRANSFORMATIONS:
        for m in MEASURES:
            t = pd.read_csv(INPUT / tr / f"{m}.csv", index_col=0, parse_dates=True)
            assert not t.isna().any().any()
            # minden tábla pontosan ugyanazokon a napokon van
            dates = t.index if dates is None else dates
            assert t.index.equals(dates)
            # a cél a következő kereskedési nap szórása, transzformálva
            nxt = pd.DatetimeIndex(target.loc[t.index, "next_day"])
            assert np.allclose(t["sigma_next"], np.sqrt(meas.loc[nxt, "RV"].to_numpy()))
            assert np.allclose(t["y"], FORWARD[tr](t["sigma_next"]))
