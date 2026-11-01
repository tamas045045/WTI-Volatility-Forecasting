"""Érzékenységi futtatás: a horgonyos maximalizált LSTM (Free/tuning.csv legjobb beállítása) 3 éves GÖRDÜLŐ ablakon.

A Free változat a horgonyt és a bővülő ablakot egyszerre változtatja; ez a futtatás a kettőt szétválasztja:
ugyanaz a beállítás és a horgony, de a HAR-X-szel (és a Strict változattal) azonos, 750 napos gördülő ablak.
Kimenet: FreeRolling/forecasts.csv és FreeRolling/metrics.csv (QLIKE a HAR_L4-hez viszonyítva, részmintánként,
valamint GW-teszt a HAR_L4 ellen). Futtatás:  python extra/LSTM_Max/run_horgony_gordulo.py
"""
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
sys.path.insert(0, str(HERE))

import pandas as pd
import torch

from config import WINDOWS, model_dir
from evaluation import gw_test, qlike, subsamples
from forecasting import back_transform, load_model_data
from max_lstm import walk_forward_max
from run import FREQUENCY, LEVEL, MEASURE, TRANSFORMATION, WINDOW, params_of

OUT = HERE / "FreeRolling"


def main() -> None:
    torch.set_num_threads(8)
    params, anchor = params_of("Free")
    X, y, sigma, dates, names = load_model_data(LEVEL, TRANSFORMATION, MEASURE)
    t0 = time.time()

    def log(k, total, date):
        el = (time.time() - t0) / 60
        print(f"FreeRolling HALADAS {k}/{total} ({k / total:.0%}) | {date.date()} | eltelt {el:.0f} perc | "
              f"hátra kb. {el / k * (total - k):.0f} perc", flush=True)

    print(f"FreeRolling: horgony={anchor}, ablak={WINDOWS[WINDOW]}, "
          + ", ".join(f"{k}={params[k]}" for k in ["seq_len", "hidden", "layers", "dropout", "lr", "seeds"]), flush=True)
    mu, s2 = walk_forward_max(X, y, dates, WINDOWS[WINDOW], FREQUENCY, params,
                              names.index("har_d") if anchor == "today" else None, start=WINDOWS[WINDOW], log=log)
    OUT.mkdir(exist_ok=True)
    out = pd.DataFrame({"sigma_true": sigma, "sigma_hat": back_transform(TRANSFORMATION, mu, s2),
                        "y_true": y, "y_hat": mu, "resid_var": s2}, index=pd.DatetimeIndex(dates, name="date"))
    out = out.dropna(subset=["y_hat"])
    out.to_csv(OUT / "forecasts.csv")

    har = pd.read_csv(model_dir("HAR", "L4", TRANSFORMATION, WINDOW, FREQUENCY, MEASURE) / "forecasts.csv",
                      index_col="date", parse_dates=True)
    idx = har.index
    s = har.loc[idx, "sigma_true"].to_numpy()
    lh, lf = qlike(s, har.loc[idx, "sigma_hat"].to_numpy()), qlike(s, out.loc[idx, "sigma_hat"].to_numpy())
    rows = [{"sample": k, "n": int(m.sum()), "QLIKE": lf[m].mean(), "ratio_vs_HAR_L4": lf[m].mean() / lh[m].mean()}
            for k, m in subsamples(idx, s).items()]
    res = pd.DataFrame(rows)
    g = gw_test(lf, lh)                      # alap: a horgonyos LSTM, vizsgált: HAR_L4 (t > 0: a HAR_L4 jobb)
    res["GW_t_HAR_better"] = g["t"]
    res.to_csv(OUT / "metrics.csv", index=False)
    print(res.to_string(index=False), flush=True)
    print(f"vége, {(time.time() - t0) / 60:.0f} perc", flush=True)


if __name__ == "__main__":
    main()
