"""HAR_L4_News – HAR + makró + cross-asset + hírek: GKG figyelem, hangulat, megosztottság.

ELTÉRÉS az L3-hoz képest: + 3 hírváltozó (config.NEWS_COLUMNS).
A modell minden más tekintetben azonos a többi HAR-lépcsővel: ugyanaz az input tábla, ugyanaz
a gördülő OLS (src/har.py), ugyanaz a visszaváltás és ugyanaz a mentés.

Ebben a mappában 32 modell van: 2 transzformáció × 2 ablak × 2 gyakoriság × 4 mérték.
Futtatás:  python models/HAR_L4_News/run.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from har import run_level

LEVEL = "L4"   # <- az egyetlen különbség a négy HAR-futtató között

if __name__ == "__main__":
    done = run_level(LEVEL)
    print(f"HAR_{LEVEL}: {len(done)} modell kész")
