"""5. lépés: mind a 128 LSTM-modell, egyetlen közös processzor-készleten (a leggyorsabb teljes futás).

A már kész modelleket kihagyja, így megszakítás után újraindítható.
Futtatás:  python scripts/05_run_lstm.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import LEVELS
from lstm import level_jobs, run_jobs

if __name__ == "__main__":
    t0 = time.time()
    jobs = [j for level in LEVELS for j in level_jobs(level)]
    done = run_jobs(jobs)
    print(f"{len(done)} LSTM-modell kész, {(time.time() - t0) / 60:.0f} perc")
