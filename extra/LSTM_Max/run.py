"""A két maximalizált LSTM végső futtatása és kiértékelése, majd az összevetés a többi modellel.

  Strict: a tuning.csv legjobb (horgony nélküli) beállítása, 3 éves gördülő ablak
  Free:   a tuning.csv legjobb beállítása (horgonnyal vagy anélkül, amit a validáció választott), bővülő ablak
Mindkettő 10 seed átlaga (a többi LSTM-nél 5): a több független indítás átlaga csak a véletlen ingadozást csökkenti.
A két változat párhuzamosan fut, változatonként 4 processzorszálon.

Kimenet: <változat>/forecasts.csv, <változat>/metrics.csv, és comparison.csv (HAR_L4, hibrid LSTM_L4,
tiszta LSTM, Strict, Free – ugyanazon az 1841 napon, ugyanazokkal a részmintákkal és hibamérőkkel).
Futtatás:  python extra/LSTM_Max/run.py
"""
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

from config import LSTM_PARAMS, WINDOWS, model_dir
from evaluation import mse, qlike, subsamples
from forecasting import back_transform, load_model_data
from max_lstm import walk_forward_max

LEVEL, TRANSFORMATION, WINDOW, FREQUENCY, MEASURE = "L4", "Log", "3Y", "Weekly", "Garman_Klass"
SEEDS = 10


def params_of(variant: str) -> tuple[dict, str]:
    best = pd.read_csv(HERE / variant / "tuning.csv").iloc[0]
    p = {**LSTM_PARAMS, "seq_len": int(best["seq_len"]), "hidden": int(best["hidden"]), "layers": int(best["layers"]),
         "dropout": float(best["dropout"]), "lr": float(best["lr"]), "seeds": SEEDS}
    return p, best["anchor"]


def run(variant: str) -> str:
    torch.set_num_threads(4)
    params, anchor = params_of(variant)
    X, y, sigma, dates, names = load_model_data(LEVEL, TRANSFORMATION, MEASURE)
    window = WINDOWS[WINDOW] if variant == "Strict" else None          # Free: bővülő ablak
    t0 = time.time()

    def log(k, total, date):
        el = (time.time() - t0) / 60
        print(f"{variant} HALADAS {k}/{total} ({k / total:.0%}) | {date.date()} | eltelt {el:.0f} perc | "
              f"hátra kb. {el / k * (total - k):.0f} perc", flush=True)

    print(f"{variant} beállítás: horgony={anchor}, ablak={'bővülő' if window is None else window}, "
          + ", ".join(f"{k}={params[k]}" for k in ["seq_len", "hidden", "layers", "dropout", "lr", "seeds"]), flush=True)
    mu, s2 = walk_forward_max(X, y, dates, window, FREQUENCY, params,
                              names.index("har_d") if anchor == "today" else None, start=WINDOWS[WINDOW], log=log)
    out = pd.DataFrame({"sigma_true": sigma, "sigma_hat": back_transform(TRANSFORMATION, mu, s2),
                        "y_true": y, "y_hat": mu, "resid_var": s2}, index=pd.DatetimeIndex(dates, name="date"))
    out.dropna(subset=["y_hat"]).to_csv(HERE / variant / "forecasts.csv")
    return variant


def compare() -> pd.DataFrame:
    load = lambda p: pd.read_csv(p, index_col="date", parse_dates=True)
    d = lambda fam, lv: model_dir(fam, lv, TRANSFORMATION, WINDOW, FREQUENCY, MEASURE) / "forecasts.csv"
    models = {"HAR_L4": load(d("HAR", "L4")), "LSTM_L4 (hibrid)": load(d("LSTM", "L4")),
              "LSTM_Pure (tiszta)": load(HERE.parent / "LSTM_Pure" / "forecasts.csv"),
              "LSTM_Max Strict": load(HERE / "Strict" / "forecasts.csv"), "LSTM_Max Free": load(HERE / "Free" / "forecasts.csv")}
    bench = load(d("HAR", "L1"))
    dates = models["HAR_L4"].index                                   # a közös kiértékelési napok (1841)
    sigma = bench.loc[dates, "sigma_true"].to_numpy()
    rows = []
    for name, f in models.items():
        h = f.loc[dates, "sigma_hat"].to_numpy()
        for sample, m in subsamples(dates, sigma).items():
            b = bench.loc[dates, "sigma_hat"].to_numpy()[m]
            rows.append({"model": name, "sample": sample, "n": int(m.sum()), "QLIKE": qlike(sigma[m], h[m]).mean(),
                         "MSE": mse(sigma[m], h[m]).mean(), "R2_OOS": 1 - mse(sigma[m], h[m]).mean() / mse(sigma[m], b).mean()})
    res = pd.DataFrame(rows)
    for v in ["Strict", "Free"]:
        res[res.model == f"LSTM_Max {v}"].drop(columns="model").to_csv(HERE / v / "metrics.csv", index=False)
    res.to_csv(HERE / "comparison.csv", index=False)
    return res


if __name__ == "__main__":
    t0 = time.time()
    with Pool(2) as pool:
        for v in pool.imap_unordered(run, ["Strict", "Free"]):
            print(f"{v} KESZ ({(time.time() - t0) / 60:.0f} perc)", flush=True)
    res = compare()
    q = res.pivot(index="model", columns="sample", values="QLIKE")
    print("OSSZEVETES (QLIKE / HAR_L4):")
    print((q / q.loc["HAR_L4"]).round(3).to_string())
