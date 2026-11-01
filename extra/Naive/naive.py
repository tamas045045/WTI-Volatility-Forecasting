"""Naiv előrejelzés (viszonyítási alap, a 256 modell keretén kívül): a holnapi napi szórás = a mai.

    sigma_hat(t+1) = sqrt(RV_t)

Nincs becslés, nincs paraméter. A kiértékelés ugyanaz, mint minden más modellnél: ugyanaz az 1841 nap,
ugyanazok a részminták, QLIKE, MSE és R²_OOS a HAR_L1-hez (src/evaluation.py, változatlanul).
A meglévő kódhoz és eredményekhez nem nyúl.

Kimenet: forecasts.csv, metrics.csv és comparison.csv (a naiv modell, HAR_L1, HAR_L4, hibrid LSTM_L4,
tiszta LSTM és a két maximalizált LSTM egymás mellett).
Futtatás:  python extra/Naive/naive.py
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))

import numpy as np
import pandas as pd

from config import PROCESSED, model_dir
from evaluation import mse, qlike, subsamples

CFG = ("Log", "3Y", "Weekly", "Garman_Klass")      # az összevetett modellek közös beállítása


def main():
    load = lambda p: pd.read_csv(p, index_col="date", parse_dates=True)
    har = lambda lv: load(model_dir("HAR", lv, *CFG) / "forecasts.csv")
    bench, h4 = har("L1"), har("L4")
    dates = h4.index                                     # a közös kiértékelési napok (1841)
    sigma = bench.loc[dates, "sigma_true"].to_numpy()    # a holnapi valódi szórás

    # A naiv előrejelzés: a t napon ismert mai sqrt(RV)
    rv = pd.read_csv(PROCESSED / "measures_daily.csv", index_col=0, parse_dates=True)["RV"]
    naive = pd.DataFrame({"sigma_true": sigma, "sigma_hat": np.sqrt(rv.loc[dates].to_numpy())},
                         index=pd.DatetimeIndex(dates, name="date"))
    naive.to_csv(HERE / "forecasts.csv")

    models = {"Naiv (holnap = ma)": naive, "HAR_L1": bench, "HAR_L4": h4,
              "LSTM_L4 (hibrid)": load(model_dir("LSTM", "L4", *CFG) / "forecasts.csv"),
              "LSTM_Pure (tiszta)": load(HERE.parent / "LSTM_Pure" / "forecasts.csv"),
              "LSTM_Max Strict": load(HERE.parent / "LSTM_Max" / "Strict" / "forecasts.csv"),
              "LSTM_Max Free": load(HERE.parent / "LSTM_Max" / "Free" / "forecasts.csv")}
    rows = []
    for name, f in models.items():
        h = f.loc[dates, "sigma_hat"].to_numpy()
        for sample, m in subsamples(dates, sigma).items():
            b = bench.loc[dates, "sigma_hat"].to_numpy()[m]
            rows.append({"model": name, "sample": sample, "n": int(m.sum()), "QLIKE": qlike(sigma[m], h[m]).mean(),
                         "MSE": mse(sigma[m], h[m]).mean(), "R2_OOS": 1 - mse(sigma[m], h[m]).mean() / mse(sigma[m], b).mean()})
    res = pd.DataFrame(rows)
    res[res.model == "Naiv (holnap = ma)"].drop(columns="model").to_csv(HERE / "metrics.csv", index=False)
    res.to_csv(HERE / "comparison.csv", index=False)
    q = res.pivot(index="model", columns="sample", values="QLIKE").loc[list(models)]
    print((q / q.loc["HAR_L4"]).round(3).to_string())


if __name__ == "__main__":
    main()
