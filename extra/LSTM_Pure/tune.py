"""A tiszta LSTM hiperparamétereinek kiválasztása – PONTOSAN a hibriddel azonos eljárással.

Ugyanaz, mint a scripts/04_tune_lstm.py: ugyanaz a rács, ugyanaz a validációs beállítás
(L4, Log, RV, 2 éves ablak, heti újrabecslés) és ugyanaz a validációs időszak (a közös minta 500–749. sora,
a kiértékelés kezdete előtt). A tesztidőszakból semmit nem lát. Mérőszám: QLIKE szórás-skálán.

Kimenet: extra/LSTM_Pure/tuning.csv (a legjobb az első sor; a model.py innen olvassa be).
Futtatás:  python extra/LSTM_Pure/tune.py      (kb. 5–10 perc, 8 kombináció párhuzamosan)
"""
import itertools
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
sys.path.insert(0, str(HERE))

import numpy as np
import pandas as pd
import torch

from config import LSTM_PARAMS
from evaluation import qlike as qlike_terms
from forecasting import back_transform, load_model_data
from har import rolling_ols
from lstm_pure import walk_forward_pure

# Ugyanazok az értékek, mint a scripts/04_tune_lstm.py-ban
VAL_START, VAL_STOP, WINDOW = 500, 750, 500
GRID = {"seq_len": [5, 22], "hidden": [16, 32], "lr": [1e-3, 3e-3]}


def qlike(s, f):
    return float(np.mean(qlike_terms(s, f)))


def evaluate(combo):
    torch.set_num_threads(1)
    params = {**LSTM_PARAMS, **combo}
    X, y, sigma, dates, names = load_model_data("L4", "Log", "RV")
    t0 = time.time()
    mu, s2 = walk_forward_pure(X, y, dates, WINDOW, "Weekly", params, VAL_START, VAL_STOP)
    mu_har, s2_har, _ = rolling_ols(X, y, dates, WINDOW, "Weekly", names)     # összevetési alap
    sl = slice(VAL_START, VAL_STOP)
    return {**combo, "QLIKE_lstm": qlike(sigma[sl], back_transform("Log", mu[sl], s2[sl])),
            "QLIKE_har": qlike(sigma[sl], back_transform("Log", mu_har[sl], s2_har[sl])),
            "seconds": round(time.time() - t0, 1)}


def main():
    combos = [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())]
    with Pool(min(len(combos), max(1, os.cpu_count() - 2))) as pool:
        rows = pool.map(evaluate, combos)
    res = pd.DataFrame(rows)
    res["ratio_vs_har"] = res["QLIKE_lstm"] / res["QLIKE_har"]
    res = res.sort_values("QLIKE_lstm")
    res.to_csv(HERE / "tuning.csv", index=False)
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
