"""Kiegészítő számítás: HAR + csak OVX (a makrolépcsőből csak az implikált olajvolatilitás).

Mind a 32 beállításon (2 transzformáció × 2 ablak × 2 gyakoriság × 4 mérték) ugyanaz a gördülő OLS, mint a
HAR-X-nél (src/har.py), csak az L2 változói közül kizárólag az OVX marad a HAR-tagok mellett. A QLIKE-arányt a
vele azonos beállítású HAR_L1-hez és HAR_L2-höz mérjük, ugyanazon az 1841 napon; mellé a GW-teszt (5%).
Kimenet: extra/HAR_OVX/comparison.csv (beállításonként) és összesítés a képernyőn.
Futtatás:  python extra/HAR_OVX/har_ovx.py
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))

import numpy as np
import pandas as pd

from config import FREQUENCIES, MEASURES, TRANSFORMATIONS, WINDOWS, model_dir
from evaluation import gw_test, qlike, subsamples
from forecasting import back_transform, load_model_data
from har import rolling_ols

DROP = {"vix", "sp500", "usd"}


def main() -> None:
    rows = []
    for tr in TRANSFORMATIONS:
        for m in MEASURES:
            X, y, sigma, dates, names = load_model_data("L2", tr, m)
            keep = [i for i, c in enumerate(names) if c not in DROP]
            for w, win in WINDOWS.items():
                for f in FREQUENCIES:
                    mu, s2, _ = rolling_ols(X[:, keep], y, dates, win, f, [names[i] for i in keep])
                    hat = pd.Series(back_transform(tr, mu, s2), index=dates)
                    l1 = pd.read_csv(model_dir("HAR", "L1", tr, w, f, m) / "forecasts.csv", index_col="date", parse_dates=True)
                    l2 = pd.read_csv(model_dir("HAR", "L2", tr, w, f, m) / "forecasts.csv", index_col="date", parse_dates=True)
                    idx = l1.index[l1.index >= "2018-08-30"]          # a közös, 1841 napos kiértékelés
                    l1 = l1.loc[idx]
                    s = l1["sigma_true"].to_numpy()
                    q0, q1, q2 = qlike(s, l1["sigma_hat"].to_numpy()), qlike(s, hat.loc[idx].to_numpy()), \
                        qlike(s, l2.loc[idx, "sigma_hat"].to_numpy())
                    covid = subsamples(idx, s)["COVID"]
                    rows.append({"transformation": tr, "measure": m, "window": w, "frequency": f,
                                 "ratio_vs_L1": q1.mean() / q0.mean(), "ratio_vs_L2": q1.mean() / q2.mean(),
                                 "ratio_vs_L1_COVID": q1[covid].mean() / q0[covid].mean(),
                                 "sig_better_than_L1": gw_test(q0, q1)["p_one_sided"] < 0.05,
                                 "sig_worse_than_L2": gw_test(q1, q2)["p_one_sided"] < 0.05})
    res = pd.DataFrame(rows)
    res.to_csv(HERE / "comparison.csv", index=False)
    print(res.to_string(index=False))
    print("\nÁtlagos QLIKE-arány a HAR_L1-hez: %.3f (min %.3f, max %.3f); a HAR_L2-höz: %.3f"
          % (res.ratio_vs_L1.mean(), res.ratio_vs_L1.min(), res.ratio_vs_L1.max(), res.ratio_vs_L2.mean()))
    print("COVID alatt a HAR_L1-hez: %.3f" % res.ratio_vs_L1_COVID.mean())
    print("Szignifikánsan jobb a HAR_L1-nél: %.0f%%; szignifikánsan rosszabb a HAR_L2-nél: %.0f%%"
          % (100 * res.sig_better_than_L1.mean(), 100 * res.sig_worse_than_L2.mean()))


if __name__ == "__main__":
    main()
