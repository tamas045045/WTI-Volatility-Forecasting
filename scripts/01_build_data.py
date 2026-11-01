"""1. lépés: napi táblák építése -> data/processed/

  measures_daily.csv : a négy volatilitásmérték alkotórészei (RV, C, J, Parkinson, GK) érvényes napokra
  target_daily.csv   : sigma_next = sqrt(RV) a következő kereskedési napon (ha az érvényes)
  external_daily.csv : L2 + L3 változók napi szórása, point-in-time hozzárendeléssel
  news_daily.csv     : L4 hírváltozók

Futtatás:  python scripts/01_build_data.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from config import PROCESSED, WTI_PIPELINE
from external import external_daily, news_daily
from measures import daily_measures


def main():
    PROCESSED.mkdir(parents=True, exist_ok=True)
    wti = pd.read_parquet(WTI_PIPELINE / "wti_daily.parquet")          # naptár: minden kereskedési nap
    bars = pd.read_parquet(WTI_PIPELINE / "wti_5min_sessions.parquet")
    gkg = pd.read_parquet(WTI_PIPELINE / "gkg_news_daily.parquet")
    valid = wti.index[wti["valid"]]

    # --- Mértékek az érvényes napokra ---------------------------------------------------------
    meas = daily_measures(bars, valid)
    # Ellenőrzés: az RV egyezik a WTI-projekt auditált RV-jével
    diff = np.max(np.abs(meas["RV"] / wti.loc[meas.index, "rv"] - 1))
    assert diff < 1e-9, f"RV eltér a WTI-projekttől: {diff}"
    meas.to_csv(PROCESSED / "measures_daily.csv")

    # --- Cél: a következő naptári kereskedési nap szórása, ha az a nap érvényes ----------------
    next_day = pd.Series(wti.index[1:].append(pd.DatetimeIndex([pd.NaT])), index=wti.index)
    next_valid = next_day.map(lambda d: bool(wti["valid"].get(d, False)) if pd.notna(d) else False)
    sigma_next = pd.Series(np.nan, index=wti.index)
    ok = next_valid.to_numpy()
    sigma_next[ok] = np.sqrt(wti.loc[next_day[ok], "rv"].to_numpy())
    target = pd.DataFrame({"next_day": next_day, "sigma_next": sigma_next}).loc[valid]
    target.to_csv(PROCESSED / "target_daily.csv")

    # --- Külső piacok (L2, L3) és hírek (L4) ---------------------------------------------------
    ext = external_daily(wti["t_D"])
    ext.to_csv(PROCESSED / "external_daily.csv")
    news = news_daily(gkg)
    news.to_csv(PROCESSED / "news_daily.csv")

    # Rövid összefoglaló
    print(f"érvényes napok: {len(valid)}  ({valid.min().date()} – {valid.max().date()})")
    print(f"cél elérhető: {target['sigma_next'].notna().sum()} nap")
    print(f"ugrásnapok aránya: {(meas['J'] > 0).mean():.1%}")
    print("külső változók hiánya (érvényes napokon):")
    print(ext.loc[valid, [c for c in ext.columns if not c.endswith('_obs_date')]].isna().mean().round(3).to_string())
    print(f"hírváltozók elérhetők: {news.loc[news.index.intersection(valid)].dropna().shape[0]} nap")


if __name__ == "__main__":
    main()
