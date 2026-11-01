"""HAR_L2_Macro – HAR + makró: OVX, VIX, S&P 500 és USD index napi szórása.

ELTÉRÉS az L1-hez képest: + 4 makróváltozó (config.MACRO_COLUMNS).
A modell minden más tekintetben azonos a többi HAR-lépcsővel: ugyanaz az input tábla, ugyanaz
a gördülő OLS (src/har.py), ugyanaz a visszaváltás és ugyanaz a mentés.

Ebben a mappában 32 modell van: 2 transzformáció × 2 ablak × 2 gyakoriság × 4 mérték.
Futtatás:  python models/HAR_L2_Macro/run.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from har import run_level

LEVEL = "L2"   # <- az egyetlen különbség a négy HAR-futtató között

if __name__ == "__main__":
    done = run_level(LEVEL)
    print(f"HAR_{LEVEL}: {len(done)} modell kész")
