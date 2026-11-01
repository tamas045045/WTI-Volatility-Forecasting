"""LSTM_Pure_L4 | Log | 3Y_Weekly | Garman_Klass – kiegészítő összevetés (nem része a 256 modellnek)

MODELL: tiszta LSTM, HAR-X nélkül (extra/LSTM_Pure/lstm_pure.py).
    y_hat = átlag(y a tanulóablakon) + LSTM(x_(t-21), …, x_t)

BEÁLLÍTÁSOK: pontosan a legjobb modell (HAR_L4 | Log | 3Y_Weekly | Garman_Klass) beállításai:
  ugyanaz az input tábla és a 16 változó (L4), ugyanaz a cél (a holnapi szórás logja), 750 napos ablak,
  heti újrabecslés, és ugyanaz a kiértékelés (1841 nap, szórás-skála, QLIKE, MSE, R²_OOS a HAR_L1-hez).
  A hiperparaméterek a tuning.csv legjobb sorából jönnek (extra/LSTM_Pure/tune.py: a hibriddel azonos
  rács és validáció). Minden más LSTM-beállítás azonos a hibridével (config.LSTM_PARAMS).

EBBEN A MAPPÁBAN
  model.py         ez a fájl: futtatás és kiértékelés
  lstm_pure.py     a tiszta LSTM kódja (a közös motor építőelemeiből, a meglévő kód módosítása nélkül)
  tune.py          a hiperparaméterek kiválasztása -> tuning.csv
  forecasts.csv    előrejelzések: date, sigma_true, sigma_hat, y_true, y_hat, resid_var
  metrics.csv      hibamérők részmintánként (ugyanaz a formátum, mint a többi modellnél)
  comparison.csv   összevetés: HAR_L4, LSTM_L4 (hibrid) és LSTM_Pure, azonos beállítással

FUTTATÁS:  python extra/LSTM_Pure/model.py      (egy processzormagon kb. 1–3 óra)
"""
import sys
import time
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
from lstm_pure import walk_forward_pure

# --- A modell beállításai (a legjobb modellé) ---------------------------------------------------
LEVEL, TRANSFORMATION, WINDOW, FREQUENCY, MEASURE = "L4", "Log", "3Y", "Weekly", "Garman_Klass"


def tuned_params() -> dict:
    """A hiperparaméterek a hangolás legjobb sorából (tuning.csv), minden más a hibridé."""
    best = pd.read_csv(HERE / "tuning.csv").iloc[0]
    return {**LSTM_PARAMS, "seq_len": int(best["seq_len"]), "hidden": int(best["hidden"]), "lr": float(best["lr"])}


def run() -> pd.DataFrame:
    torch.set_num_threads(1)          # ugyanúgy, mint a 128 LSTM-modell futásánál
    params = tuned_params()
    X, y, sigma, dates, _ = load_model_data(LEVEL, TRANSFORMATION, MEASURE)
    t0 = time.time()

    def log(k, total, date):
        # Haladásnapló: hányadik újrabecslés, dátum, eltelt és becsült hátralévő idő
        el = (time.time() - t0) / 60
        print(f"HALADAS {k}/{total} ({k / total:.0%}) | {date.date()} | eltelt {el:.0f} perc | "
              f"hátra kb. {el / k * (total - k):.0f} perc", flush=True)

    print(f"hiperparaméterek: seq_len={params['seq_len']}, hidden={params['hidden']}, lr={params['lr']}", flush=True)
    mu, s2 = walk_forward_pure(X, y, dates, WINDOWS[WINDOW], FREQUENCY, params, log=log)
    out = pd.DataFrame({"sigma_true": sigma, "sigma_hat": back_transform(TRANSFORMATION, mu, s2),
                        "y_true": y, "y_hat": mu, "resid_var": s2}, index=pd.DatetimeIndex(dates, name="date"))
    out = out.dropna(subset=["y_hat"])
    out.to_csv(HERE / "forecasts.csv")
    return out


def evaluate(fc: pd.DataFrame) -> pd.DataFrame:
    """Hibamérők ugyanazon a kiértékelési mintán és részmintákon, mint a 256 modellnél (scripts/06_evaluate.py)."""
    load = lambda fam, lv: pd.read_csv(model_dir(fam, lv, TRANSFORMATION, WINDOW, FREQUENCY, MEASURE) / "forecasts.csv",
                                      index_col="date", parse_dates=True)
    models = {"HAR_L4": load("HAR", "L4"), "LSTM_L4 (hibrid)": load("LSTM", "L4"), "LSTM_Pure": fc}
    bench = load("HAR", "L1")
    dates = models["HAR_L4"].index                       # a közös kiértékelési napok (1841)
    sigma = bench.loc[dates, "sigma_true"].to_numpy()
    rows = []
    for name, f in models.items():
        h = f.loc[dates, "sigma_hat"].to_numpy()
        for sample, m in subsamples(dates, sigma).items():
            b = bench.loc[dates, "sigma_hat"].to_numpy()[m]
            rows.append({"model": name, "sample": sample, "n": int(m.sum()), "QLIKE": qlike(sigma[m], h[m]).mean(),
                         "MSE": mse(sigma[m], h[m]).mean(), "R2_OOS": 1 - mse(sigma[m], h[m]).mean() / mse(sigma[m], b).mean()})
    res = pd.DataFrame(rows)
    res[res.model == "LSTM_Pure"].drop(columns="model").to_csv(HERE / "metrics.csv", index=False)
    res.to_csv(HERE / "comparison.csv", index=False)
    return res


if __name__ == "__main__":
    t0 = time.time()
    fc = run()
    res = evaluate(fc)
    print(f"LSTM_Pure: {len(fc)} előrejelzés, {(time.time() - t0) / 60:.0f} perc")
    print(res.pivot(index="model", columns="sample", values="QLIKE").round(4).to_string())
