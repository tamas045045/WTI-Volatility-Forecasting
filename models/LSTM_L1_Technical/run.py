"""LSTM_L1_Technical – HAR-X + LSTM-korrekció, a(z) HAR_L1_Technical változóival.

ALAPMODELL – csak a mérték saját napi, heti és havi tagja.
A lineáris rész pontosan a HAR_L1_Technical modell; az LSTM-rész ugyanezen változók utolsó 22 napjából
tanulja a nemlineáris korrekciót (src/lstm.py, docs/09_lstm_protokoll.md). Minden más azonos a többi LSTM-lépcsővel.

Ebben a mappában 32 modell van: 2 transzformáció × 2 ablak × 2 gyakoriság × 4 mérték.
Futtatás:  python models/LSTM_L1_Technical/run.py   (mind a 128 együtt: python scripts/05_run_lstm.py)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from lstm import run_level

LEVEL = "L1"   # <- az egyetlen különbség a négy LSTM-futtató között

if __name__ == "__main__":
    done = run_level(LEVEL)
    print(f"LSTM_{LEVEL}: {len(done)} modell kész")
