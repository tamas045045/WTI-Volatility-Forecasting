"""3. lépés: mind a 128 HAR-X modell (a négy lépcső run.py-jának egyben futtatása, néhány másodperc).

Ugyanaz, mintha a models/HAR_L1_Technical/run.py … models/HAR_L4_News/run.py fájlokat egymás után futtatnánk.
Futtatás:  python scripts/03_run_har.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import LEVELS
from har import run_level

if __name__ == "__main__":
    for level in LEVELS:
        print(f"HAR_{level}: {len(run_level(level))} modell kész")
