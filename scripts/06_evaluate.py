"""6. lépés: kiértékelés -> minden modell mappájába metrics.csv, összesítők a results/tables/ mappába.

  results/tables/metrics_all.csv     : minden modell hibái minden részmintán (teljes, 2020 nélkül, COVID, háború, top 5%)
  results/tables/hypotheses.csv      : a H1–H5 döntési szabályainak számai (docs/08_kutatasi_kerdesek_hipotezisek.md)
  results/tables/hypothesis_pairs.csv: a páronkénti Giacomini–White tesztek
  results/tables/main_effects.csv    : „fő hatás” tábla (H5)
  results/tables/crisis_best.csv     : a legjobb modellek a kritikus időszakokban
  results/tables/mcs.csv             : Model Confidence Set (K5)

Futtatás:  python scripts/06_evaluate.py
"""
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from config import FAMILIES, FREQUENCIES, LEVELS, MEASURES, TABLES, TRANSFORMATIONS, WINDOWS, all_models, model_dir
from evaluation import gw_test, mse, qlike, subsamples

DIMS = ["family", "level", "transformation", "window", "frequency", "measure"]
ALPHA = 0.05


def load_forecasts():
    """Az összes elkészült modell előrejelzése a KÖZÖS kiértékelési napokon."""
    fc = {}
    for key in all_models():
        p = model_dir(*key) / "forecasts.csv"
        if p.exists():
            fc[key] = pd.read_csv(p, index_col="date", parse_dates=True)
    common = None
    for f in fc.values():
        common = f.index if common is None else common.intersection(f.index)
    sigma = next(iter(fc.values())).loc[common, "sigma_true"].to_numpy()
    hat = {k: f.loc[common, "sigma_hat"].to_numpy() for k, f in fc.items()}
    return common, sigma, hat


def model_metrics(dates, sigma, hat):
    """Hibamérők modellenként és részmintánként; R2_OOS a vele azonos beállítású HAR_L1-hez képest."""
    masks = subsamples(dates, sigma)
    rows = []
    for key, h in hat.items():
        bench = hat.get(("HAR", "L1") + key[2:])
        for name, m in masks.items():
            r = dict(zip(DIMS, key), sample=name, n=int(m.sum()),
                     QLIKE=qlike(sigma[m], h[m]).mean(), MSE=mse(sigma[m], h[m]).mean())
            r["R2_OOS"] = 1 - r["MSE"] / mse(sigma[m], bench[m]).mean() if bench is not None else np.nan
            rows.append(r)
    return pd.DataFrame(rows)


def pair_tests(sigma, hat, mask, label, pairs):
    """Páronkénti GW tesztek: pairs = [(kisebb/alap kulcs, bővebb/vizsgált kulcs, csoport)]."""
    rows = []
    for small, big, group in pairs:
        if small in hat and big in hat:
            g = gw_test(qlike(sigma[mask], hat[small][mask]), qlike(sigma[mask], hat[big][mask]))
            g_rev = gw_test(qlike(sigma[mask], hat[big][mask]), qlike(sigma[mask], hat[small][mask]))
            rows.append({"hypothesis": label, "group": group, "base": "|".join(small), "test": "|".join(big),
                         **g, "test_better_sig": g["p_one_sided"] < ALPHA,
                         "test_worse_sig": g_rev["p_one_sided"] < ALPHA})
    return rows


def hypotheses(dates, sigma, hat, mask):
    others = lambda fams, lvls: itertools.product(fams, lvls, TRANSFORMATIONS, WINDOWS, FREQUENCIES)
    pairs = []
    # H1: Parkinson és GK az RV-hez képest, minden más fix
    for fam, lv, tr, w, f in others(FAMILIES, LEVELS):
        for meas in ["Parkinson", "Garman_Klass"]:
            pairs += pair_tests(sigma, hat, mask, "H1", [((fam, lv, tr, w, f, "RV"), (fam, lv, tr, w, f, meas), meas)])
    # H2, H4 (HAR) és H3, H4 (LSTM): lépcsők és családok összevetése
    for tr, w, f, meas in itertools.product(TRANSFORMATIONS, WINDOWS, FREQUENCIES, MEASURES):
        for fam in FAMILIES:
            for a, b in [("L1", "L2"), ("L2", "L3"), ("L3", "L4"), ("L1", "L4")]:
                pairs += pair_tests(sigma, hat, mask, f"{fam}_{a}_{b}",
                                    [((fam, a, tr, w, f, meas), (fam, b, tr, w, f, meas), fam)])
        for lv in LEVELS:
            pairs += pair_tests(sigma, hat, mask, f"LSTMvsHAR_{lv}",
                                [(("HAR", lv, tr, w, f, meas), ("LSTM", lv, tr, w, f, meas), lv)])
    P = pd.DataFrame(pairs)
    if P.empty:
        return P, pd.DataFrame()

    def share(label, col):
        s = P[P["hypothesis"] == label]
        return (s[col].mean() if len(s) else np.nan), len(s)

    out = []
    # H1
    s = P[P["hypothesis"] == "H1"]
    if len(s):
        out.append({"H": "H1", "mean_ratio_meas_vs_RV": s["ratio"].mean(),
                    "share_not_sig_worse": 1 - s["test_worse_sig"].mean(), "n_pairs": len(s),
                    "supported": bool(s["ratio"].mean() <= 1 and 1 - s["test_worse_sig"].mean() > 0.5)})
    # H2 (HAR): az L1→L2 javulás aránya a teljes L1→L4 javulásban
    g12, g14 = P[P["hypothesis"] == "HAR_L1_L2"], P[P["hypothesis"] == "HAR_L1_L4"]
    if len(g12) and len(g14):
        frac = g12["mean_diff"].mean() / g14["mean_diff"].mean()
        sig, n = share("HAR_L1_L2", "test_better_sig")
        out.append({"H": "H2", "share_of_total_gain_from_L2": frac, "share_L1L2_sig": sig, "n_pairs": n,
                    "supported": bool(frac > 0.5 and sig >= 0.5)})
    # H3: L1-en az LSTM nem jobb; az LSTM/HAR arány L4-en kisebb, mint L1-en
    l1, l4 = P[P["hypothesis"] == "LSTMvsHAR_L1"], P[P["hypothesis"] == "LSTMvsHAR_L4"]
    if len(l1) and len(l4):
        out.append({"H": "H3", "share_L1_lstm_sig_better": l1["test_better_sig"].mean(),
                    "ratio_L1": l1["ratio"].mean(), "ratio_L4": l4["ratio"].mean(), "n_pairs": len(l1),
                    "supported": bool(l1["test_better_sig"].mean() < 0.5 and l4["ratio"].mean() < l1["ratio"].mean())})
    # H4: a hírek (L3→L4) a HAR-ban nem, az LSTM-ben igen
    h34, n34 = P[P["hypothesis"] == "HAR_L3_L4"], P[P["hypothesis"] == "LSTM_L3_L4"]
    if len(h34) and len(n34):
        out.append({"H": "H4", "share_HAR_sig": h34["test_better_sig"].mean(), "share_LSTM_sig": n34["test_better_sig"].mean(),
                    "ratio_HAR": h34["ratio"].mean(), "ratio_LSTM": n34["ratio"].mean(), "n_pairs": len(h34),
                    "supported": bool(h34["test_better_sig"].mean() < 0.5 and n34["test_better_sig"].mean() >= 0.5)})
    return P, pd.DataFrame(out)


def main_effects(metrics: pd.DataFrame, sample: str = "full") -> pd.DataFrame:
    """H5: dimenziónként a QLIKE átlagos terjedelme (max − min a dimenzió szintjei között, a többi fix)."""
    m = metrics[metrics["sample"] == sample]
    rows = []
    for dim in DIMS:
        rest = [d for d in DIMS if d != dim]
        grp = m.groupby(rest)["QLIKE"]
        spread = (grp.max() - grp.min()) / grp.mean()        # relatív terjedelem
        if grp.size().min() > 1:
            rows.append({"dimension": dim, "mean_relative_spread": spread.mean(),
                         "levels": ", ".join(sorted(m[dim].unique()))})
    return pd.DataFrame(rows).sort_values("mean_relative_spread", ascending=False)


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    dates, sigma, hat = load_forecasts()
    print(f"modellek: {len(hat)}/256, közös kiértékelési napok: {len(dates)} ({dates.min().date()} – {dates.max().date()})")

    metrics = model_metrics(dates, sigma, hat)
    metrics.to_csv(TABLES / "metrics_all.csv", index=False)
    for key in hat:
        sel = (metrics[DIMS] == pd.Series(dict(zip(DIMS, key)))).all(axis=1)
        metrics.loc[sel, ["sample", "n", "QLIKE", "MSE", "R2_OOS"]].to_csv(model_dir(*key) / "metrics.csv", index=False)

    masks = subsamples(dates, sigma)
    pairs, hyp = [], []
    for name in ["full", "ex_2020"]:
        P, H = hypotheses(dates, sigma, hat, masks[name])
        P["sample"], H["sample"] = name, name
        pairs.append(P), hyp.append(H)
    pd.concat(pairs).to_csv(TABLES / "hypothesis_pairs.csv", index=False)
    hyp = pd.concat(hyp)

    eff = main_effects(metrics)
    eff.to_csv(TABLES / "main_effects.csv", index=False)
    full = metrics[metrics["sample"] == "full"]
    q_ratio = (full[full.transformation == "QuarticRoot"].set_index([d for d in DIMS if d != "transformation"])["QLIKE"]
               / full[full.transformation == "Log"].set_index([d for d in DIMS if d != "transformation"])["QLIKE"]).mean()
    soft = eff[eff.dimension.isin(["transformation", "window", "frequency"])]["mean_relative_spread"]
    core = eff[eff.dimension.isin(["measure", "level"])]["mean_relative_spread"]
    hyp = pd.concat([hyp, pd.DataFrame([{"H": "H5", "sample": "full", "quartic_vs_log_ratio": q_ratio,
                                         "max_soft_effect": soft.max(), "min_core_effect": core.min(),
                                         "supported": bool(soft.max() < core.min() and q_ratio <= 1)}])])
    hyp.to_csv(TABLES / "hypotheses.csv", index=False)

    # Kritikus időszakok: a legjobb 10 modell részmintánként
    best = (metrics.sort_values("QLIKE").groupby("sample").head(10))
    best.to_csv(TABLES / "crisis_best.csv", index=False)

    # K5: Model Confidence Set a teljes mintán (QLIKE)
    try:
        from arch.bootstrap import MCS
        L = pd.DataFrame({"|".join(k): qlike(sigma, h) for k, h in hat.items()})
        mcs = MCS(L, size=0.10, reps=1000, block_size=20, method="R", bootstrap="stationary", seed=0)
        mcs.compute()
        mcs.pvalues.sort_values("Pvalue", ascending=False).to_csv(TABLES / "mcs.csv")
        print(f"MCS (10%): {len(mcs.included)} modell a legjobbak halmazában")
    except Exception as exc:   # az MCS nagy modellszámnál lassú lehet; a többi eredményt nem veszélyezteti
        print("MCS hiba:", exc)

    pd.set_option("display.width", 200)
    print(hyp.round(3).to_string(index=False))
    print(eff.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
