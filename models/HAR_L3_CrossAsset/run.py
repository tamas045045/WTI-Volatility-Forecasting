"""HAR_L3_CrossAsset – HAR + makró + cross-asset: réz, arany, földgáz, búza, kukorica, szója napi szórása.

ELTÉRÉS az L2-höz képest: + 6 nyersanyag (config.CROSSASSET_COLUMNS).
A modell minden más tekintetben azonos a többi HAR-lépcsővel: ugyanaz az input tábla, ugyanaz
a gördülő OLS (src/har.py), ugyanaz a visszaváltás és ugyanaz a mentés.

Ebben a mappában 32 modell van: 2 transzformáció × 2 ablak × 2 gyakoriság × 4 mérték.
Futtatás:  python models/HAR_L3_CrossAsset/run.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from har import run_level

LEVEL = "L3"   # <- az egyetlen különbség a négy HAR-futtató között

if __name__ == "__main__":
    done = run_level(LEVEL)
    print(f"HAR_{LEVEL}: {len(done)} modell kész")
