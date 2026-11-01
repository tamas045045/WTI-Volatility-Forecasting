"""7. lépés: a dolgozat ábrái -> results/figures/*.png

Minden ábra a results/tables/ és a models/ fájljaiból készül, így minden szám ellenőrizhető.
Színek: validált kategorikus paletta első három eleme (kék, narancs, akvamarin).
Futtatás:  python scripts/07_make_figures.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import FIGURES, TABLES, WTI_PIPELINE, model_dir
from evaluation import CRISES, qlike

OUT = FIGURES
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
EVAL_START = "2018-08-30"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False})


def plain_log_axis(ax, ticks):
    """Log-skálás y-tengely sima számokkal (nem 10-hatvány alakban)."""
    from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_formatter(NullFormatter())


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight")
    plt.close(fig)


def shade_crises(ax):
    for name, (a, b) in CRISES.items():
        ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color=GRID, alpha=0.8, lw=0)


def fig_volatility():
    """1. ábra: a WTI napi szórása (sqrt RV), 2009–2026."""
    wti = pd.read_parquet(WTI_PIPELINE / "wti_daily.parquet")
    s = np.sqrt(wti.loc[wti["valid"], "rv"]) * 100
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    shade_crises(ax)
    ax.plot(s.index, s, color=BLUE, lw=0.6, label="napi szórás")
    ax.plot(s.index, s.rolling(22).mean(), color=INK, lw=1.2, label="22 napos átlag")
    ax.axvline(pd.Timestamp(EVAL_START), color=MUTED, lw=1, ls="--")
    ax.set_yscale("log")
    plain_log_axis(ax, [0.5, 1, 2, 5, 10, 20, 50])
    ax.set_ylabel("napi szórás, % (log skála)")
    ax.text(pd.Timestamp(EVAL_START), 45, " a kiértékelés kezdete →", color=MUTED, fontsize=8)
    ax.text(pd.Timestamp("2020-07-10"), 30, "COVID", color=MUTED, fontsize=8)
    ax.text(pd.Timestamp("2022-07-10"), 30, "háború", color=MUTED, fontsize=8)
    ax.legend(loc="lower right", ncol=2)
    save(fig, "f1_wti_volatility.png")


def fig_main_effects():
    """Fő hatások: a QLIKE átlagos relatív terjedelme dimenziónként."""
    e = pd.read_csv(TABLES / "main_effects.csv").sort_values("mean_relative_spread")
    names = {"level": "információs lépcső", "transformation": "transzformáció", "measure": "volatilitásmérték",
             "frequency": "újrabecslési gyakoriság", "window": "tanulóablak", "family": "modellcsalád (HAR / LSTM)"}
    fig, ax = plt.subplots(figsize=(6.0, 2.4))
    y = np.arange(len(e))
    ax.barh(y, e["mean_relative_spread"] * 100, color=BLUE, height=0.55)
    for yi, v in zip(y, e["mean_relative_spread"] * 100):
        ax.text(v + 0.2, yi, f"{v:.1f}%", va="center", color=INK, fontsize=8)
    ax.set_yticks(y, [names[d] for d in e["dimension"]])
    ax.set_xlabel("a QLIKE átlagos változása a dimenzió mentén (%)")
    ax.grid(axis="y", visible=False)
    save(fig, "f2_main_effects.png")


def daily_losses():
    """Minden modell napi QLIKE-ja a közös kiértékelési napokon (oszlop = modell)."""
    rows = {}
    for p in sorted(Path(model_dir("HAR", "L1", "Log", "2Y", "Weekly", "RV")).parents[3].glob("*/*/*/*/forecasts.csv")):
        f = pd.read_csv(p, index_col=0, parse_dates=True).loc[EVAL_START:]
        rows[tuple(p.parts[-5:-1])] = pd.Series(qlike(f["sigma_true"], f["sigma_hat"]), index=f.index)
    return pd.DataFrame(rows)


def fig_measures_by_year(L):
    """GK/RV és Parkinson/RV QLIKE-arány évenként (az összes azonos beállítású pár átlaga)."""
    years = L.index.year
    meas = L.columns.get_level_values(3)
    rv = L.loc[:, meas == "RV"]
    fig, ax = plt.subplots(figsize=(6.0, 2.4))
    for m, c, lab in [("Garman_Klass", BLUE, "Garman–Klass / RV"), ("Parkinson", ORANGE, "Parkinson / RV")]:
        x = L.loc[:, meas == m]
        r = x.groupby(years).mean().mean(axis=1) / rv.groupby(years).mean().mean(axis=1)
        ax.plot(r.index, r.values, color=c, lw=2, marker="o", ms=4, label=lab)
    ax.axhline(1, color=MUTED, lw=1, ls="--")
    ax.set_ylabel("QLIKE-arány (1 alatt jobb)")
    ax.set_xticks(sorted(set(years)))
    ax.legend(loc="lower left", ncol=2)
    save(fig, "f3_measures_by_year.png")


def fig_levels():
    """Lépcsők és családok: QLIKE a HAR_L1-hez képest, teljes minta és a legviharosabb 5% nap."""
    m = pd.read_csv(TABLES / "metrics_all.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.7), sharey=True)
    for ax, sample, title in [(axes[0], "full", "teljes minta"), (axes[1], "top5pct_days", "a legviharosabb 5% nap")]:
        t = m[m["sample"] == sample].groupby(["family", "level"])["QLIKE"].mean()
        t = t / t.loc[("HAR", "L1")]
        x = np.arange(4)
        for k, (fam, c) in enumerate([("HAR", BLUE), ("LSTM", ORANGE)]):
            vals = t.loc[fam].values
            ax.bar(x + (k - 0.5) * 0.36, vals, width=0.34, color=c, label="HAR-X" if fam == "HAR" else "HAR-X + LSTM")
        ax.axhline(1, color=MUTED, lw=1, ls="--")
        ax.set_xticks(x, ["L1\nalap", "L2\n+makró", "L3\n+nyersanyag", "L4\n+hírek"], fontsize=8)
        ax.set_title(title, fontsize=9, color=INK)
        ax.set_ylim(0.7, 1.05)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("QLIKE / QLIKE(HAR_L1)")
    axes[0].legend(loc="upper right")
    save(fig, "f4_levels_families.png")


def fig_covid():
    """A COVID-időszak: realizált szórás és két HAR-X előrejelzés (Log, 3Y, heti, Garman–Klass)."""
    a, b = "2020-02-15", "2020-07-15"
    l1 = pd.read_csv(model_dir("HAR", "L1", "Log", "3Y", "Weekly", "Garman_Klass") / "forecasts.csv", index_col=0, parse_dates=True).loc[a:b]
    l2 = pd.read_csv(model_dir("HAR", "L2", "Log", "3Y", "Weekly", "Garman_Klass") / "forecasts.csv", index_col=0, parse_dates=True).loc[a:b]
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    ax.plot(l1.index, l1["sigma_true"] * 100, color=INK, lw=1.2, label="realizált szórás (holnap)")
    ax.plot(l1.index, l1["sigma_hat"] * 100, color=ORANGE, lw=2, label="HAR_L1 (csak saját múlt)")
    ax.plot(l2.index, l2["sigma_hat"] * 100, color=BLUE, lw=2, label="HAR_L2 (+ OVX, VIX, S&P, USD)")
    ax.set_yscale("log")
    plain_log_axis(ax, [2, 5, 10, 20, 50])
    ax.set_ylabel("napi szórás, % (log skála)")
    ax.legend(loc="upper right")
    save(fig, "f5_covid.png")


if __name__ == "__main__":
    fig_volatility()
    fig_main_effects()
    fig_measures_by_year(daily_losses())
    fig_levels()
    fig_covid()
    print("ábrák:", sorted(p.name for p in OUT.glob("*.png")))
