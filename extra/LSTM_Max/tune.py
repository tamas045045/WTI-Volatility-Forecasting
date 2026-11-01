"""A maximalizált LSTM hangolása – csak a tesztidőszak ELŐTTI validáción.

Validáció: a közös minta 500–749. sora (2017-08 – 2018-08), mint minden más hangolásnál; a modell saját
beállításán (L4, Log, Garman_Klass, heti újrabecslés), 500 napos ablakkal. Mérőszám: QLIKE szórás-skálán.
Rács (horgonyonként 48 kombináció):
  seq_len {5, 22} × hidden {16, 32, 64} × layers {1, 2} × dropout {0, 0,2} × lr {1e-3, 3e-3}
  horgony: nincs (Strict és Free jelöltjei) / a mai szint, har_d (csak Free)
A bővülő ablakot a validáció nem tudja eldönteni (a minta elején a bővülő és a 2 éves ablak szinte
ugyanannyi adatot lát), ezért az előre rögzített tervezési döntés a Free változatban.

Kimenet: Strict/tuning.csv (horgony nélküli sorok), Free/tuning.csv (minden sor); a legjobb az első sor.
Futtatás:  python extra/LSTM_Max/tune.py
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
from max_lstm import walk_forward_max

LEVEL, TRANSFORMATION, MEASURE, FREQUENCY = "L4", "Log", "Garman_Klass", "Weekly"
VAL_START, VAL_STOP, WINDOW = 500, 750, 500
GRID = {"seq_len": [5, 22], "hidden": [16, 32, 64], "layers": [1, 2], "dropout": [0.0, 0.2], "lr": [1e-3, 3e-3]}
ANCHORS = ["none", "today"]


def qlike(s, f):
    return float(np.mean(qlike_terms(s, f)))


def evaluate(job):
    anchor, combo = job
    torch.set_num_threads(1)
    params = {**LSTM_PARAMS, **combo}
    X, y, sigma, dates, names = load_model_data(LEVEL, TRANSFORMATION, MEASURE)
    t0 = time.time()
    mu, s2 = walk_forward_max(X, y, dates, WINDOW, FREQUENCY, params,
                              names.index("har_d") if anchor == "today" else None, VAL_START, VAL_STOP)
    sl = slice(VAL_START, VAL_STOP)
    return {"anchor": anchor, **combo, "QLIKE_lstm": qlike(sigma[sl], back_transform(TRANSFORMATION, mu[sl], s2[sl])),
            "seconds": round(time.time() - t0, 1)}


def main():
    combos = [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())]
    jobs = [(a, c) for a in ANCHORS for c in combos]
    jobs.sort(key=lambda j: -(j[1]["seq_len"] * j[1]["hidden"] * j[1]["layers"]))   # a lassúak indulnak először
    # Összevetési alap: a HAR-X ugyanazon a validációs időszakon
    X, y, sigma, dates, names = load_model_data(LEVEL, TRANSFORMATION, MEASURE)
    mu_h, s2_h, _ = rolling_ols(X, y, dates, WINDOW, FREQUENCY, names)
    sl = slice(VAL_START, VAL_STOP)
    q_har = qlike(sigma[sl], back_transform(TRANSFORMATION, mu_h[sl], s2_h[sl]))

    rows, t0 = [], time.time()
    with Pool(max(1, os.cpu_count() - 2)) as pool:
        for k, r in enumerate(pool.imap_unordered(evaluate, jobs), 1):
            rows.append(r)
            print(f"HANGOLAS {k}/{len(jobs)} ({k / len(jobs):.0%}) | eltelt {(time.time() - t0) / 60:.0f} perc", flush=True)
    res = pd.DataFrame(rows)
    res["QLIKE_har"] = q_har
    res["ratio_vs_har"] = res["QLIKE_lstm"] / q_har
    res = res.sort_values("QLIKE_lstm")
    for variant, sel in [("Strict", res["anchor"] == "none"), ("Free", res["anchor"].notna())]:
        (HERE / variant).mkdir(exist_ok=True)
        res[sel].to_csv(HERE / variant / "tuning.csv", index=False)
    print(res.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
