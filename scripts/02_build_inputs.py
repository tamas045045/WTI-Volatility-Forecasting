"""2. lépés: a közös input táblák -> data/input/<Transformation>/<Measure>.csv

Mind a 8 tábla (2 transzformáció × 4 mérték) ugyanarra a közös mintára van szűrve.
Futtatás:  python scripts/02_build_inputs.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from config import INPUT, MEASURES, PROCESSED, TRANSFORMATIONS
from inputs import build_input, common_sample


def main():
    meas = pd.read_csv(PROCESSED / "measures_daily.csv", index_col=0, parse_dates=True)
    target = pd.read_csv(PROCESSED / "target_daily.csv", index_col=0, parse_dates=True)
    ext = pd.read_csv(PROCESSED / "external_daily.csv", index_col=0, parse_dates=True)
    news = pd.read_csv(PROCESSED / "news_daily.csv", index_col=0, parse_dates=True)

    tables = {(tr, m): build_input(meas, target["sigma_next"], ext, news, tr, m)
              for tr in TRANSFORMATIONS for m in MEASURES}
    sample = common_sample(tables)

    for (tr, m), t in tables.items():
        (INPUT / tr).mkdir(parents=True, exist_ok=True)
        t.loc[sample].to_csv(INPUT / tr / f"{m}.csv", index_label="date")

    print(f"közös minta: {len(sample)} nap ({sample.min().date()} – {sample.max().date()})")
    print(f"ebből 2020-as nap: {(sample.year == 2020).sum()}")


if __name__ == "__main__":
    main()
