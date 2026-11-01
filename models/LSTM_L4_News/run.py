"""LSTM_L4_News – HAR-X + LSTM-korrekció, a(z) HAR_L4_News változóival.

ELTÉRÉS az L3-hoz képest: + 3 hírváltozó (config.NEWS_COLUMNS).
A lineáris rész pontosan a HAR_L4_News modell; az LSTM-rész ugyanezen változók utolsó 22 napjából
tanulja a nemlineáris korrekciót (src/lstm.py, docs/09_lstm_protokoll.md). Minden más azonos a többi LSTM-lépcsővel.

Ebben a mappában 32 modell van: 2 transzformáció × 2 ablak × 2 gyakoriság × 4 mérték.
Futtatás:  python models/LSTM_L4_News/run.py   (mind a 128 együtt: python scripts/05_run_lstm.py)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from lstm import run_level

LEVEL = "L4"   # <- az egyetlen különbség a négy LSTM-futtató között

if __name__ == "__main__":
    done = run_level(LEVEL)
    print(f"LSTM_{LEVEL}: {len(done)} modell kész")
