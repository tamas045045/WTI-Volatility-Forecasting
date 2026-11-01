"""4. lépés: az LSTM hiperparamétereinek kiválasztása a tesztminta ELŐTT (docs/09_lstm_protokoll.md, 4. pont).

Validációs beállítás: L4 (minden változó), Log, RV, 2 éves ablak, heti újrabecslés.
Validációs időszak: a közös minta 500–749. sora, azaz a kiértékelés kezdete (a 750. sor) előtti napok.
Így a választás a tesztidőszakból semmit nem lát.
Mérőszám: QLIKE szórás-skálán; összevetés a lineáris résszel (= HAR-X).

Kimenet: results/tables/lstm_tuning.csv. A nyertest kézzel írjuk be a config.LSTM_PARAMS-ba (dokumentálva).
Futtatás:  python scripts/04_tune_lstm.py
"""
import itertools
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import torch

from config import LSTM_PARAMS, TABLES
from evaluation import qlike as qlike_terms
from forecasting import back_transform, load_model_data
from lstm import walk_forward

VAL_START, VAL_STOP, WINDOW = 500, 750, 500
GRID = {"seq_len": [5, 22], "hidden": [16, 32], "lr": [1e-3, 3e-3]}   # az epochszámot a korai leállítás dönti el


def qlike(s, f):
    return float(np.mean(qlike_terms(s, f)))   # a közös QLIKE-definíció (src/evaluation.py)


def evaluate(combo):
    torch.set_num_threads(1)
    params = {**LSTM_PARAMS, **combo}
    X, y, sigma, dates, _ = load_model_data("L4", "Log", "RV")
    t0 = time.time()
    mu, s2, mu_lin = walk_forward(X, y, dates, WINDOW, "Weekly", params, VAL_START, VAL_STOP)
    sl = slice(VAL_START, VAL_STOP)
    return {**combo, "QLIKE_lstm": qlike(sigma[sl], back_transform("Log", mu[sl], s2[sl])),
            "QLIKE_har": qlike(sigma[sl], back_transform("Log", mu_lin[sl], s2[sl])),
            "seconds": round(time.time() - t0, 1)}


def main():
    combos = [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())]
    with Pool(min(len(combos), max(1, os.cpu_count() - 2))) as pool:
        rows = pool.map(evaluate, combos)
    res = pd.DataFrame(rows)
    res["ratio_vs_har"] = res["QLIKE_lstm"] / res["QLIKE_har"]
    res = res.sort_values("QLIKE_lstm")
    TABLES.mkdir(parents=True, exist_ok=True)
    res.to_csv(TABLES / "lstm_tuning.csv", index=False)
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
