"""8. lépés: a dolgozat és a prezentáció ábrái -> results/figures/dolgozat/ (PNG 300 dpi + PDF).

Csak olvas: a kész eredményekből (data/, models/, results/tables/, extra/) rajzol, semmit nem számol újra
és semmit nem ír felül. Az ábrák a fejezetek sorrendjében számozottak; a leírásuk és az olvasatuk:
results/figures/dolgozat/README.md.

Stílus: egy ábrán legfeljebb 3–4 adatsor, a többi kis panelekre bontva; kék = a fő modell / a legjobb,
narancs = az összevetett alternatíva, türkiz = harmadik sor, szürke = a valóság vagy a háttér.
Futtatás:  python scripts/08_make_thesis_figures.py
           python scripts/08_make_thesis_figures.py --cim-nelkul   (a dolgozat szövegébe: a cím az ábra alá kerül
           a Word-dokumentumban, ezért az ábrába nem égetjük bele -> results/figures/dolgozat/szoveg/)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.ticker import FuncFormatter
import matplotlib.dates as mdates

from config import (FIGURES, INPUT, LEVELS, PROCESSED, ROOT, TABLES, WTI_PIPELINE, level_columns, model_dir)
from evaluation import CRISES, qlike

NOTITLE = "--cim-nelkul" in sys.argv
FONT_SCALE = 1.22                                          # a szöveges változat betűnagyítása
OUT = FIGURES / "dolgozat" / ("szoveg" if NOTITLE else "")
OUT.mkdir(parents=True, exist_ok=True)

# --- Színek és stílus ------------------------------------------------------------------------------
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"     # az első három, színvak-biztos kategória-szín
GRAY, LIGHT, INK, INK2, GRID = "#8a8985", "#d9d8d3", "#0b0b0b", "#52514e", "#e6e5e1"
RED = "#e34948"
SHADE = "#f1f0ec"                                          # válságidőszakok háttere
EVAL_START = pd.Timestamp("2018-08-30")
VAL_START, VAL_END = pd.Timestamp("2017-08-17"), pd.Timestamp("2018-08-29")
CFG = ("Log", "3Y", "Weekly", "Garman_Klass")               # a legjobb modell beállításai

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11.5, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.labelcolor": INK2, "axes.edgecolor": LIGHT, "axes.linewidth": 0.8,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "figure.dpi": 100,
    "savefig.bbox": "tight", "savefig.facecolor": "white", "axes.axisbelow": True,
})


def comma(dec=None):
    """Tizedesvessző a tengelyeken (magyar írásmód)."""
    return FuncFormatter(lambda x, _: (f"{x:.{dec}f}" if dec is not None else f"{x:g}").replace(".", ",").replace("-", "−"))


def pct(dec=0):
    return FuncFormatter(lambda x, _: f"{x * 100:.{dec}f}%".replace(".", ",").replace("-", "−"))


def hu(x, dec=2):
    return f"{x:.{dec}f}".replace(".", ",").replace("-", "−")


def logpct(axis, ticks=(1, 2, 3, 5, 10, 20, 30, 50)):
    """Log skála %-os jelölésekkel (a matplotlib 6×10⁰ alakú jelölései helyett)."""
    from matplotlib.ticker import FixedLocator, NullFormatter
    axis.set_major_locator(FixedLocator(ticks))
    axis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}%".replace(".", ",")))
    axis.set_minor_formatter(NullFormatter())


def shade_crises(ax, label=True):
    for name, (a, b) in CRISES.items():
        ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color=SHADE, zorder=0, lw=0)
        if label:
            ax.text(pd.Timestamp(a), 1.0, {"COVID": " COVID", "Ukraine_war": " háború"}[name], transform=ax.get_xaxis_transform(),
                    va="top", ha="left", fontsize=8, color=INK2)


def main_title(ax, text, **kw):
    """Az ábra fő címe – a dolgozatba szánt változatból kimarad (ott a Word-felirat a cím)."""
    if not NOTITLE:
        ax.set_title(text, **kw)


def sup(fig, text, **kw):
    if not NOTITLE:
        fig.suptitle(text, **kw)


def save(fig, name):
    if NOTITLE:                     # a dolgozatba szánt változat kicsinyítve kerül be: nagyobb betűk
        from matplotlib.text import Text
        for txt in fig.findobj(Text):
            txt.set_fontsize(txt.get_fontsize() * FONT_SCALE)
        for ax in fig.axes:         # a tengelyfeliratok rajzoláskor újra létrejönnek: a méretüket külön kell állítani
            for axis in (ax.xaxis, ax.yaxis):
                labs = axis.get_ticklabels()
                if labs:
                    axis.set_tick_params(labelsize=labs[0].get_fontsize())
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300 if ext == "png" else None)
    plt.close(fig)
    print("kész:", name)


def fc(family, level, *cfg):
    return pd.read_csv(model_dir(family, level, *(cfg or CFG)) / "forecasts.csv", index_col="date", parse_dates=True)


METRICS = pd.read_csv(TABLES / "metrics_all.csv")
FULL = METRICS[METRICS["sample"] == "full"]
KEY = ["family", "level", "transformation", "window", "frequency", "measure"]


# =====================================================================================================
# III. ADATOK
# =====================================================================================================
def f01_price_vol():
    w = pd.read_parquet(WTI_PIPELINE / "wti_daily.parquet")
    w = w[w["valid"]].loc["2015-06-25":"2026-07-10"]
    sig = np.sqrt(w["rv"]) * 100
    fig, ax = plt.subplots(2, 1, figsize=(10, 5.6), sharex=True, gridspec_kw={"height_ratios": [1, 1.25], "hspace": 0.12})
    for a in ax:
        shade_crises(a, label=(a is ax[0]))
        a.axvline(EVAL_START, color=INK2, lw=0.8, ls="--")
    ax[0].plot(w.index, w["close_last"], color=INK, lw=1)
    ax[0].set_ylabel("USD / hordó")
    main_title(ax[0], "A WTI határidős ára és napi volatilitása, 2015–2026")
    ax[1].plot(sig.index, sig, color=BLUE, lw=0.7)
    ax[1].plot(sig.index, sig.rolling(22).mean(), color=INK, lw=1.4)
    ax[1].set_yscale("log")
    logpct(ax[1].yaxis)
    ax[1].set_ylabel("napi szórás, √RV (log skála)")
    ax[1].text(sig.index[-1], sig.rolling(22).mean().iloc[-1], " 22 napos átlag", fontsize=8, color=INK, va="center")
    ax[1].text(EVAL_START, 0.02, " a kiértékelés kezdete", transform=ax[1].get_xaxis_transform(), fontsize=8, color=INK2)
    save(fig, "01_wti_ar_es_volatilitas")


def f02_jump_days():
    bars = pd.read_parquet(WTI_PIPELINE / "wti_5min_sessions.parquet")
    meas = pd.read_csv(PROCESSED / "measures_daily.csv", index_col=0, parse_dates=True)
    days = [("2019-06-12", "Egy átlagos nap"), ("2016-12-16", "Egy ugrásos nap")]
    fig, ax = plt.subplots(2, 2, figsize=(10, 5.2), sharex="col", gridspec_kw={"height_ratios": [1.3, 1], "hspace": 0.1, "wspace": 0.18})
    for j, (d, title) in enumerate(days):
        b = bars[bars["trading_day"] == d]
        t = (b["bar_start"] - b["bar_start"].iloc[0]).dt.total_seconds() / 3600
        r = b["r"] * 100
        m = meas.loc[d]
        ax[0, j].plot(t, b["close"], color=INK, lw=1)
        ax[0, j].set_title(f"{title} ({d})")
        ax[0, j].set_ylabel("ár (USD)" if j == 0 else "")
        ax[0, j].yaxis.set_major_formatter(comma(1))
        cols = [ORANGE if abs(v) == r.abs().max() and m["J"] > 0 else BLUE for v in r]
        ax[1, j].bar(t, r, width=0.07, color=cols)
        ax[1, j].set_ylabel("5 perces hozam (%)" if j == 0 else "")
        ax[1, j].set_xlabel("órák a kereskedési nap kezdete óta (18:00 ET)")
        ax[1, j].yaxis.set_major_formatter(comma())
        txt = (f"RV = {m['RV'] * 1e4:.2f}·10⁻⁴   BPV = {m['BPV'] * 1e4:.2f}·10⁻⁴\n"
               f"z = {m['jump_z']:.1f}  →  " + (f"ugrás: J = {m['J'] / m['RV']:.0%} az RV-ből" if m["J"] > 0 else "nincs szignifikáns ugrás"))
        ax[0, j].text(0.02, 0.04, txt.replace(".", ","), transform=ax[0, j].transAxes, fontsize=8.3, color=INK2,
                      bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=LIGHT))
    save(fig, "02_atlagos_es_ugrasos_nap")


def f03_persistence():
    m = pd.read_csv(PROCESSED / "measures_daily.csv", index_col=0, parse_dates=True).loc["2015-06-25":]
    lags = np.arange(1, 23)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for col, name, c in [("RV", "RV", ORANGE), ("Parkinson", "Parkinson", AQUA), ("Garman_Klass", "Garman–Klass", BLUE)]:
        s = np.log(np.sqrt(m[col]))
        acf = [s.autocorr(l) for l in lags]
        ax.plot(lags, acf, color=c, lw=2, marker="o", ms=3.5, label=name)
        ax.text(22.4, acf[-1] + {"RV": 0, "Parkinson": -0.007, "Garman_Klass": 0.007}[col], name, color=INK, fontsize=9, va="center")
    ax.set_xlim(0.5, 25.5)
    main_title(ax, "A mértékek tartóssága: a log-szórás autokorrelációja")
    ax.set_xlabel("késleltetés (kereskedési nap)")
    ax.set_ylabel("autokorreláció")
    ax.yaxis.set_major_formatter(comma(2))
    ax.legend(loc="lower left")
    ax.text(0.98, 0.95, "A range-alapú mértékek tartósabbak (kevesebb mérési zaj)\n→ jobban előrejelezhetők (H1)",
            transform=ax.transAxes, ha="right", va="top", fontsize=8.5, color=INK2)
    save(fig, "03_mertekek_tartossaga")


def f04_correlation():
    t = pd.read_csv(INPUT / "Log" / "Garman_Klass.csv", index_col="date", parse_dates=True)
    cols = level_columns("L4", "Garman_Klass")
    c = t[cols].corr()
    names = {"har_d": "HAR napi", "har_w": "HAR heti", "har_m": "HAR havi", "ovx": "OVX", "vix": "VIX", "sp500": "S&P 500",
             "usd": "USD-index", "copper": "réz", "gold": "arany", "natgas": "földgáz", "wheat": "búza", "corn": "kukorica",
             "soybeans": "szója", "news_abnormal": "hírmennyiség", "news_tone": "hírhangulat", "news_tone_disp": "hangulatszórás"}
    fig, ax = plt.subplots(figsize=(8.2, 7))
    im = ax.imshow(c.values, cmap="RdBu_r", vmin=-1, vmax=1)
    lab = [names[x] for x in cols]
    ax.set_xticks(range(len(cols)), lab, rotation=55, ha="right")
    ax.set_yticks(range(len(cols)), lab)
    ax.grid(False)
    for i in range(len(cols)):
        for j in range(len(cols)):
            v = c.values[i, j]
            if not NOTITLE:
                ax.text(j, i, hu(round(v, 1) + 0.0, 1), ha="center", va="center", fontsize=6.5, color="white" if abs(v) > 0.6 else INK)
    for k in (3, 7, 13):
        ax.axhline(k - 0.5, color="white", lw=2.5)
        ax.axvline(k - 0.5, color="white", lw=2.5)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, ticks=[-1, -0.5, 0, 0.5, 1])
    cb.ax.yaxis.set_major_formatter(comma(1))
    main_title(ax, "A 16 magyarázó változó korrelációja (log-szórás, 2015–2026)")
    save(fig, "04_valtozok_korrelacioja")


# =====================================================================================================
# IV. MÓDSZERTAN (sémák)
# =====================================================================================================
def box(ax, x, y, w, h, text, fc="white", ec=LIGHT, fs=9, color=INK, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.015", fc=fc, ec=ec, lw=1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=color, weight="bold" if bold else None)


def arrow(ax, a, b, color=INK2):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=12, color=color, lw=1.2))


def f05_timing():
    fig, ax = plt.subplots(figsize=(10, 3.9))
    ax.set_xlim(15.55, 18.45)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.plot([15.65, 18.4], [0.4, 0.4], color=INK2, lw=1.2)
    for h in np.arange(15.75, 18.3, 0.25):
        ax.plot([h, h], [0.385, 0.415], color=INK2, lw=1)
        ax.text(h, 0.35, f"{int(h)}:{int(round((h % 1) * 60)):02d}", ha="center", va="top", fontsize=8, color=INK2)
    # A t_D előtt lezárult adatok: bekerülnek a t napi előrejelzésbe
    for x, t, y in [(16.0, "S&P 500\n16:00", 0.78), (16.25, "VIX\n16:15", 0.62), (16.5, "OVX\n16:30", 0.78)]:
        ax.plot([x, x], [0.4, y - 0.02], color=BLUE, lw=1.4)
        ax.text(x, y, t, ha="center", va="bottom", fontsize=8.5, color=INK)
    # 17:00: a WTI kereskedési nap vége = az előrejelzés időpontja (t_D)
    ax.plot([17.0, 17.0], [0.4, 0.83], color=ORANGE, lw=2.2)
    ax.text(17.0, 0.85, "$t_D$ = 17:00: a WTI kereskedési napjának vége,\naz előrejelzés elkészül", ha="center", va="bottom",
            fontsize=8.5, color=INK, weight="bold")
    ax.text(17.04, 0.55, "USD-index és nyersanyagok\n(záróidő 17:00, nem $< t_D$)\n→ csak a holnapi előrejelzésbe", ha="left",
            va="center", fontsize=8, color=INK2)
    # A cél
    ax.add_patch(Rectangle((18.0, 0.42), 0.4, 0.1, color=AQUA, alpha=0.35, lw=0))
    ax.text(18.2, 0.57, "18:00-tól a t+1 kereskedési nap:\nennek a √RV-je a cél", ha="center", va="bottom", fontsize=8.5, color=INK)
    ax.text(15.65, 0.2, "BEKERÜL: minden adat, amelynek a záróideje szigorúan $t_D$ előtt van ($\\tau < t_D$)", fontsize=8.5,
            color=BLUE, weight="bold")
    ax.text(15.65, 0.1, "NEM KERÜL BE: ami $t_D$-kor vagy utána zár – az a következő kereskedési nap előrejelzésébe megy "
            "(legfeljebb 7 napos elévüléssel)", fontsize=8.5, color=INK2)
    main_title(ax, "Időzítés: mi kerülhet bele a t napi előrejelzésbe? (New York-i idő, normál kereskedési nap)")
    save(fig, "05_idozites_sema")


def f06_sample_timeline():
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.set_ylim(0, 1)
    d = pd.Timestamp
    ax.set_xlim(d("2015-01-01"), d("2026-12-31"))
    for s in ["left", "right", "top"]:
        ax.spines[s].set_visible(False)
    ax.set_yticks([])
    ax.grid(False)
    ax.add_patch(Rectangle((d("2015-06-25"), 0.62), d("2026-07-09") - d("2015-06-25"), 0.14, color=LIGHT, lw=0))
    ax.text(d("2015-06-25"), 0.8, "közös minta: 2591 nap (2015-06-25 – 2026-07-09)", fontsize=8.5, color=INK)
    ax.add_patch(Rectangle((VAL_START, 0.62), VAL_END - VAL_START, 0.14, color=ORANGE, lw=0))
    ax.text(VAL_START, 0.52, "LSTM-hangolás\n(validáció)", fontsize=8, color=INK, va="top")
    ax.add_patch(Rectangle((EVAL_START, 0.62), d("2026-07-09") - EVAL_START, 0.14, color=BLUE, lw=0))
    ax.text(d("2022-06-01"), 0.69, "kiértékelés: 1841 nap", fontsize=8.5, color="white", va="center", ha="center", weight="bold")
    # Gördülő ablak illusztráció
    ex = [d("2020-03-02"), d("2023-03-06")]
    for k, t in enumerate(ex):
        y = 0.28 - k * 0.13
        ax.add_patch(Rectangle((t - pd.Timedelta(days=1095), y), pd.Timedelta(days=1095), 0.07, color=AQUA, alpha=0.55, lw=0))
        ax.plot([t, t], [y, y + 0.07], color=ORANGE, lw=3)
        ax.text(t + pd.Timedelta(days=40), y + 0.035, f"előrejelzés {t.date()}", fontsize=8, va="center", color=INK)
    ax.text(d("2015-01-15"), 0.31, "gördülő 3 éves\ntanulóablak", fontsize=8, color=INK2, va="center")
    main_title(ax, "A minta felosztása: hangolás a tesztidőszak előtt, kiértékelés gördülő ablakkal")
    save(fig, "06_minta_idovonal")


def _canvas(w_units, h_units, width_in=10.0):
    """Egyenlő méretarányú rajzlap sémákhoz (a lekerekített sarkok így valóban kerekek)."""
    fig, ax = plt.subplots(figsize=(width_in, width_in * h_units / w_units))
    ax.set_xlim(0, w_units)
    ax.set_ylim(0, h_units)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1)
    return fig, ax


def _rbox(ax, x, y, w, h, fc="white", ec=LIGHT, lw=1.0, r=0.8):
    ax.add_patch(FancyBboxPatch((x + r, y + r), w - 2 * r, h - 2 * r, boxstyle=f"round,pad={r}",
                                fc=fc, ec=ec, lw=lw, zorder=2))


def _arrow(ax, a, b, color=INK2, lw=1.3):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=13, color=color, lw=lw,
                                 shrinkA=0, shrinkB=0, zorder=3))


def f07_framework():
    fig, ax = _canvas(100, 47)
    ax.text(1.5, 44.5, "Információs lépcsők", fontsize=11, weight="bold", color=INK, va="center")
    ax.text(30.5, 44.5, "kumulatívak: minden lépcső tartalmazza az előző összes változóját", fontsize=9, color=INK2,
            va="center")
    lv = [("L1", "Technikai", "a mérték napi, heti\nés havi HAR-tagja", "3 változó (RV-CJ: 6)", "#dbe8f8", INK),
          ("L2", "Piaci", "+ OVX, VIX, S&P 500,\nUSD-index", "7 változó", "#b7d3f6", INK),
          ("L3", "Nyersanyag", "+ réz, arany, földgáz,\nbúza, kukorica, szója", "13 változó", "#6da7ec", INK),
          ("L4", "Hírek", "+ hírmennyiség, hangulat,\nhangulatszórás", "16 változó", "#2a78d6", "white")]
    bw, gap, y0, bh = 22.0, 3.0, 24.5, 16.0
    for k, (code, name, desc, n, fc, tc) in enumerate(lv):
        x = 1.5 + k * (bw + gap)
        _rbox(ax, x, y0, bw, bh, fc=fc, ec=fc)
        ax.text(x + bw / 2, y0 + bh - 3.0, f"{code}  {name}", ha="center", va="center", fontsize=11, weight="bold", color=tc)
        ax.text(x + bw / 2, y0 + bh / 2 - 0.6, desc, ha="center", va="center", fontsize=9, color=tc, linespacing=1.35)
        ax.text(x + bw / 2, y0 + 2.4, n, ha="center", va="center", fontsize=8.5, color=tc, style="italic")
        if k:
            _arrow(ax, (x - gap + 0.5, y0 + bh / 2), (x - 0.5, y0 + bh / 2))
    ax.text(50, 20.8, "×   minden lépcső kombinálva az alábbi öt dimenzió összes szintjével   ×", ha="center",
            va="center", fontsize=9, color=INK2)
    dims = [("Modellcsalád", "HAR-X · hibrid LSTM"), ("Volatilitásmérték", "RV · RV-CJ · PK · GK"),
            ("Transzformáció", "log σ · √σ"), ("Tanulóablak", "2 év · 3 év"), ("Újrabecslés", "heti · havi")]
    dw, dg, dy, dh = 18.2, 1.5, 8.0, 10.0
    for k, (t, s) in enumerate(dims):
        x = 1.5 + k * (dw + dg)
        _rbox(ax, x, dy, dw, dh, fc="#f7f7f5", ec="#c8c7c2")
        ax.text(x + dw / 2, dy + 6.6, t, ha="center", va="center", fontsize=10, weight="bold", color=INK)
        ax.text(x + dw / 2, dy + 3.0, s, ha="center", va="center", fontsize=9, color=INK2)
    _rbox(ax, 1.5, 0.6, 97, 5.6, fc="#1f3a5f", ec="#1f3a5f", r=0.6)
    ax.text(50, 3.4, "2 × 4 × 4 × 2 × 2 × 2 = 256 előrejelző modell   ·   közös cél, minta és mérce: 1841 kiértékelési nap",
            ha="center", va="center", fontsize=10, color="white", weight="bold")
    save(fig, "07_kutatasi_keret")


def f08_hybrid():
    fig, ax = _canvas(110, 37)
    if not NOTITLE:
        ax.text(1, 35.5, "A hibrid LSTM: a HAR-X adja a szintet, az LSTM csak korrekciót", fontsize=11.5, weight="bold")
    cy = 17.0
    # Bemenet
    _rbox(ax, 1, cy - 6, 17, 12, fc="#f7f7f5", ec="#c8c7c2")
    ax.text(9.5, cy + 3.0, "Bemenet", ha="center", va="center", fontsize=10, weight="bold")
    ax.text(9.5, cy - 0.2, "a lépcső változói", ha="center", va="center", fontsize=8.5, color=INK2)
    ax.text(9.5, cy - 3.2, r"$z_{t-21},\,\dots,\,z_t$", ha="center", va="center", fontsize=10)
    # HAR-X és LSTM
    for yc, fc, ec, t1, t2, eq in [
            (cy + 9.5, "#e3eefb", BLUE, "HAR-X (OLS)", "lineáris rész: a szint követése",
             r"$y^{\mathrm{lin}}_{t+1}=\beta_0+\beta'x_t$"),
            (cy - 9.5, "#fde9df", ORANGE, "LSTM-korrekció", "az OLS-reziduumokon tanul",
             r"$y^{\mathrm{nl}}_{t+1}=\mathrm{LSTM}(z_{t-21},\dots,z_t)$")]:
        _rbox(ax, 27, yc - 6, 28, 12, fc=fc, ec=ec, lw=1.3)
        ax.text(41, yc + 3.3, t1, ha="center", va="center", fontsize=10, weight="bold")
        ax.text(41, yc + 0.3, t2, ha="center", va="center", fontsize=8.5, color=INK2)
        ax.text(41, yc - 3.0, eq, ha="center", va="center", fontsize=10)
    ax.text(41, cy, "kezdőértéke 0: ha a validáción nem javít,\na modell pontosan a HAR-X", ha="center", va="center",
            fontsize=8, color=INK2, style="italic", linespacing=1.3)
    # Előrejelzés és visszaváltás
    _rbox(ax, 64, cy - 6, 19, 12, fc="white", ec=INK, lw=1.4)
    ax.text(73.5, cy + 3.0, "Előrejelzés", ha="center", va="center", fontsize=10, weight="bold")
    ax.text(73.5, cy - 1.8, r"$\hat{y}_{t+1}=y^{\mathrm{lin}}_{t+1}+y^{\mathrm{nl}}_{t+1}$", ha="center", va="center",
            fontsize=10)
    _rbox(ax, 91, cy - 6, 18, 12, fc="#f7f7f5", ec="#c8c7c2")
    ax.text(100, cy + 3.0, "Visszaváltás", ha="center", va="center", fontsize=10, weight="bold")
    ax.text(100, cy - 1.8, r"$\hat{\sigma}_{t+1}=e^{\,\hat{y}_{t+1}+s^2/2}$", ha="center", va="center", fontsize=10)
    # Nyilak
    _arrow(ax, (18, cy + 2), (27, cy + 9.5))
    _arrow(ax, (18, cy - 2), (27, cy - 9.5))
    _arrow(ax, (55, cy + 9.5), (64, cy + 2))
    _arrow(ax, (55, cy - 9.5), (64, cy - 2))
    _arrow(ax, (83, cy), (91, cy))
    save(fig, "08_hibrid_modell_sema")


# =====================================================================================================
# V. EREDMÉNYEK
# =====================================================================================================
def naive_and_best():
    naive = pd.read_csv(ROOT / "extra" / "Naive" / "comparison.csv")
    return naive


def f09_modelling_gain():
    c = naive_and_best()
    full = c[c["sample"] == "full"].set_index("model")["QLIKE"]
    rows = [("Naiv: holnap = ma", full["Naiv (holnap = ma)"], GRAY)]
    for lv, lab in [("L1", "HAR_L1: technikai"), ("L2", "HAR_L2: + piaci"), ("L3", "HAR_L3: + nyersanyagok"), ("L4", "HAR_L4: + hírek")]:
        f = fc("HAR", lv)
        f = f[f.index >= EVAL_START]
        rows.append((lab, float(np.mean(qlike(f["sigma_true"], f["sigma_hat"]))), BLUE))
    fig, ax = plt.subplots(figsize=(8, 3.8))
    y = np.arange(len(rows))[::-1]
    for yi, (lab, v, col) in zip(y, rows):
        ax.barh(yi, v, color=col, height=0.6)
        ax.text(v, yi, f"  {hu(v, 4)}  ({hu(v / rows[-1][1], 2)}×)", va="center", fontsize=9, color=INK)
    ax.set_yticks(y, [r[0] for r in rows])
    ax.set_xlim(0, max(r[1] for r in rows) * 1.3)
    ax.xaxis.set_major_formatter(comma(3))
    ax.set_xlabel("QLIKE (kisebb = jobb), 1841 nap")
    ax.grid(axis="y", visible=False)
    main_title(ax, "Mennyit ér a modellezés? (Log, 3 év, heti, Garman–Klass)")
    save(fig, "09_modellezes_haszna")


def ratio_table(base_col, base_val, vals, sample="full"):
    m = METRICS[METRICS["sample"] == sample]
    rest = [k for k in KEY if k != base_col]
    piv = m.pivot_table(index=rest, columns=base_col, values="QLIKE")
    return {v: (piv[v] / piv[base_val]).dropna().to_numpy() for v in vals}


def strip(ax, data, labels, colors, ref=1.0):
    rng = np.random.default_rng(0)
    for k, (d, c) in enumerate(zip(data, colors)):
        y = len(data) - 1 - k
        ax.scatter(d, y + rng.uniform(-0.18, 0.18, len(d)), s=14, color=c, alpha=0.55, lw=0)
        med = np.median(d)
        ax.plot([med, med], [y - 0.3, y + 0.3], color=INK, lw=2)
        ax.text(med, y + 0.36, f"medián {hu(med, 3)}", ha="center", fontsize=8, color=INK)
    ax.axvline(ref, color=INK2, lw=1, ls="--")
    ax.set_yticks(range(len(data))[::-1], labels)
    ax.grid(axis="y", visible=False)
    ax.xaxis.set_major_formatter(comma(2))


def f10_h1_measures():
    r = ratio_table("measure", "RV", ["RV_CJ", "Parkinson", "Garman_Klass"])
    fig, ax = plt.subplots(figsize=(8, 3.8))
    strip(ax, [r["RV_CJ"], r["Parkinson"], r["Garman_Klass"]], ["RV-CJ", "Parkinson", "Garman–Klass"], [GRAY, AQUA, BLUE])
    ax.set_xlabel("QLIKE-arány az azonos beállítású RV-modellhez (< 1: jobb, mint az RV)")
    main_title(ax, "H1: a Parkinson és a Garman–Klass minden modellpárban jobb az RV-nél")
    ax.text(0.99, 0.02, "pont = egy modellpár (64 mértékenként)", transform=ax.transAxes, ha="right", fontsize=8, color=INK2)
    save(fig, "10_H1_mertekek")


def f11_h1_years():
    rows = []
    for fam in ("HAR", "LSTM"):
        for lv in LEVELS:
            for tr in ("Log", "QuarticRoot"):
                for w in ("2Y", "3Y"):
                    for fr in ("Weekly", "Monthly"):
                        q = {}
                        for me in ("RV", "Parkinson", "Garman_Klass"):
                            f = fc(fam, lv, tr, w, fr, me)
                            f = f[f.index >= EVAL_START]
                            q[me] = pd.Series(qlike(f["sigma_true"], f["sigma_hat"]), index=f.index).groupby(f.index.year).mean()
                        rows.append(pd.DataFrame({"PK": q["Parkinson"] / q["RV"], "GK": q["Garman_Klass"] / q["RV"]}))
    yr = pd.concat(rows).groupby(level=0)
    mean, lo, hi = yr.mean(), yr.quantile(0.1), yr.quantile(0.9)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.fill_between(mean.index, lo["GK"], hi["GK"], color=BLUE, alpha=0.10, lw=0)
    for col, lab, c in [("PK", "Parkinson / RV", AQUA), ("GK", "Garman–Klass / RV", BLUE)]:
        ax.plot(mean.index, mean[col], color=c, lw=2, marker="o", ms=5, label=lab)
    ax.axhline(1, color=INK2, lw=1, ls="--")
    ax.yaxis.set_major_formatter(comma(2))
    ax.set_xticks(mean.index)
    ax.set_ylabel("átlagos QLIKE-arány (< 1: jobb az RV-nél)")
    main_title(ax, "H1 évente: a range-alapú mértékek mind a 9 évben jobbak")
    ax.legend(loc="upper left")
    ax.text(0.99, 0.02, "sáv: a Garman–Klass 64 modellpárjának 10–90%-os tartománya", transform=ax.transAxes, ha="right", fontsize=8, color=INK2)
    save(fig, "11_H1_evente")


def f12_h2_levels():
    r = {}
    for fam in ("HAR", "LSTM"):
        m = FULL[FULL["family"] == fam]
        piv = m.pivot_table(index=["transformation", "window", "frequency", "measure"], columns="level", values="QLIKE")
        r[fam] = {lv: (piv[lv] / piv["L1"]).to_numpy() for lv in LEVELS}
    fig, ax = plt.subplots(figsize=(8, 4.2))
    rng = np.random.default_rng(1)
    for off, fam, c in [(-0.13, "HAR", BLUE), (0.13, "LSTM", ORANGE)]:
        means = []
        for k, lv in enumerate(LEVELS):
            d = r[fam][lv]
            ax.scatter(k + off + rng.uniform(-0.05, 0.05, len(d)), d, s=12, color=c, alpha=0.4, lw=0)
            means.append(d.mean())
        ax.plot(np.arange(4) + off, means, color=c, lw=2, marker="o", ms=6, label=f"{fam}-X" if fam == "HAR" else "hibrid (HAR-X + LSTM)")
        for k, v in enumerate(means):
            if k:
                ax.text(k + off + (0.09 if fam == "LSTM" else -0.09), v, hu(v, 3), fontsize=8, va="center",
                        ha="left" if fam == "LSTM" else "right", color=INK,
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
    ax.axhline(1, color=INK2, lw=1, ls="--")
    ax.set_xticks(range(4), ["L1\ntechnikai", "L2\n+ piaci", "L3\n+ nyersanyagok", "L4\n+ hírek"])
    ax.yaxis.set_major_formatter(comma(2))
    ax.set_ylabel("QLIKE / a saját L1-modell QLIKE-ja")
    main_title(ax, "H2: szinte a teljes javulás az L2 piaci lépcsőből jön")
    ax.legend(loc="lower left")
    ax.text(0.99, 0.02, "pont = egy beállítás (32 lépcsőnként)", transform=ax.transAxes, ha="right", fontsize=8, color=INK2)
    ax.grid(axis="x", visible=False)
    save(fig, "12_H2_lepcsok")


def f13_h3_lstm():
    m = FULL.pivot_table(index=[k for k in KEY if k != "family"], columns="family", values="QLIKE")
    ratio = (m["LSTM"] / m["HAR"]).dropna()
    zero = []
    for idx in m.index:
        lin = pd.read_csv(model_dir("LSTM", *idx) / "linear_part.csv", index_col="date", parse_dates=True)["y_lin"]
        f = fc("LSTM", *idx)
        f = f[f.index >= EVAL_START]
        zero.append(float(((f["y_hat"] - lin.loc[f.index]).abs() < 1e-12).mean()))
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw={"wspace": 0.28})
    ax[0].hist(ratio, bins=np.arange(0.99, 1.04, 0.0025), color=ORANGE, edgecolor="white", lw=1)
    ax[0].axvline(1, color=INK2, lw=1, ls="--")
    ax[0].xaxis.set_major_formatter(comma(3))
    ax[0].set_xlabel("QLIKE: HAR-X + LSTM / HAR-X  (> 1: az LSTM rosszabb)")
    ax[0].set_ylabel("modellpárok száma (128)")
    from matplotlib.ticker import MaxNLocator
    for a in ax:
        a.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax[0].set_title("QLIKE-arány modellpáronként")
    ax[0].text(0.97, 0.95, f"{(ratio > 1).mean():.0%} rosszabb (pontbecslés)\nátlag: {hu(ratio.mean(), 3)}", transform=ax[0].transAxes,
               ha="right", va="top", fontsize=8.5, color=INK)
    ax[1].hist(zero, bins=np.arange(0.4, 0.9, 0.025), color=AQUA, edgecolor="white", lw=1)
    ax[1].xaxis.set_major_formatter(pct())
    ax[1].set_xlabel("a napok aránya, ahol a korrekció pontosan 0")
    ax[1].set_title("A nulla korrekciójú napok aránya")
    ax[1].text(0.97, 0.95, f"medián: {np.median(zero):.0%}".replace(".", ","), transform=ax[1].transAxes, ha="right", va="top", fontsize=8.5)
    save(fig, "13_H3_lstm_vs_har")


def f14_h5_effects():
    e = pd.read_csv(TABLES / "main_effects.csv").sort_values("mean_relative_spread")
    lab = {"level": "információs lépcső (L1–L4)", "transformation": "transzformáció (log vs. √)", "measure": "mérték (RV, RV-CJ, PK, GK)",
           "frequency": "újrabecslés (heti vs. havi)", "window": "ablak (2 vs. 3 év)", "family": "modellcsalád (HAR-X vs. hibrid)"}
    core = {"level", "measure"}
    fig, ax = plt.subplots(figsize=(8, 3.6))
    y = np.arange(len(e))
    ax.barh(y, e["mean_relative_spread"], color=[BLUE if d in core else GRAY for d in e["dimension"]], height=0.6)
    for yi, v in zip(y, e["mean_relative_spread"]):
        ax.text(v, yi, f"  {v * 100:.1f}%".replace(".", ","), va="center", fontsize=9)
    ax.set_yticks(y, [lab[d] for d in e["dimension"]])
    ax.xaxis.set_major_formatter(pct())
    ax.set_xlim(0, e["mean_relative_spread"].max() * 1.2)
    ax.set_xlabel("a QLIKE átlagos relatív terjedelme a dimenzió szintjei között")
    ax.grid(axis="y", visible=False)
    main_title(ax, "H5: mi számít? A transzformáció többet számít a mértéknél")
    ax.text(0.99, 0.03, "kék: a H5 szerinti „lényegi” döntések", transform=ax.transAxes, ha="right", fontsize=8, color=INK2)
    save(fig, "14_H5_fohatasok")


def f15_crisis_heatmap():
    rows, names = [], []
    samples = ["full", "ex_2020", "COVID", "Ukraine_war", "top5pct_days"]
    for fam in ("HAR", "LSTM"):
        for lv in LEVELS:
            if fam == "HAR" and lv == "L1":
                continue
            vals = []
            for s in samples:
                m = METRICS[METRICS["sample"] == s]
                piv = m.pivot_table(index=["transformation", "window", "frequency", "measure"], columns=["family", "level"], values="QLIKE")
                vals.append(float((piv[(fam, lv)] / piv[("HAR", "L1")]).mean()))
            rows.append(vals)
            names.append({"HAR": "HAR-X", "LSTM": "hibrid"}[fam] + f" {lv}")
    a = np.array(rows)
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    lim = 0.3
    im = ax.imshow(a - 1, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            ax.text(j, i, hu(a[i, j], 3), ha="center", va="center", fontsize=9, color="white" if abs(a[i, j] - 1) > 0.18 else INK)
    ax.set_xticks(range(5), ["teljes", "2020 nélkül", "COVID", "ukrajnai\nháború", "legviharosabb\n5% nap"])
    ax.set_yticks(range(len(names)), names)
    ax.axhline(2.5, color="white", lw=3)
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("QLIKE / HAR_L1 − 1 (kék: jobb)")
    cb.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:+.0%}".replace(".", ",").replace("-", "−")))
    main_title(ax, "Kritikus időszakok: QLIKE a sima HAR-hoz (L1) viszonyítva")
    fig.text(0.01, -0.02, "Cellánként a 32 azonos beállítású modellpár (transzformáció × ablak × újrabecslés × mérték) átlaga.",
             fontsize=8, color=INK2)
    save(fig, "15_kritikus_idoszakok")


def f16_crisis_paths():
    h1, h4 = fc("HAR", "L1"), fc("HAR", "L4")
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8), sharey=False, gridspec_kw={"wspace": 0.18})
    for a, (name, (s, e)), title in zip(ax, CRISES.items(), ["COVID (2020. febr.–jún.)", "Ukrajnai háború (2022. febr.–jún.)"]):
        sl = slice(pd.Timestamp(s) - pd.Timedelta(days=20), pd.Timestamp(e))
        d = h4.loc[sl]
        a.plot(d.index, d["sigma_true"] * 100, color=GRAY, lw=1.1, label="valódi (holnapi √RV)")
        a.plot(d.index, h1.loc[sl, "sigma_hat"] * 100, color=ORANGE, lw=1.8, label="HAR_L1 (csak HAR-tagok)")
        a.plot(d.index, d["sigma_hat"] * 100, color=BLUE, lw=1.8, label="HAR_L4 (+ piaci, nyersanyag, hírek)")
        a.axvline(pd.Timestamp(s), color=INK2, lw=0.8, ls=":")
        a.set_title(title)
        a.set_yscale("log")
        logpct(a.yaxis)
        a.tick_params(axis="x", labelrotation=30)
    ax[0].set_ylabel("napi szórás (log skála)")
    ax[0].legend(loc="upper left", fontsize=8, frameon=True, framealpha=0.9, edgecolor="none")
    sup(fig, "Két válság: a COVID elején a külső információ gyorsabban követte a sokkot, a háborúban alig számított",
                 x=0.07, ha="left", y=1.02,
                 fontsize=11.5, weight="bold")
    save(fig, "16_covid_es_haboru")


def f17_cumulative():
    h1, h4, l4 = fc("HAR", "L1"), fc("HAR", "L4"), fc("LSTM", "L4")
    idx = h4.index[h4.index >= EVAL_START]
    q = lambda f: pd.Series(qlike(f.loc[idx, "sigma_true"], f.loc[idx, "sigma_hat"]), index=idx)
    gain_info = (q(h1) - q(h4)).cumsum()
    gain_lstm = (q(h4) - q(l4)).cumsum()
    fig, ax = plt.subplots(figsize=(10, 4))
    shade_crises(ax)
    ax.plot(gain_info.index, gain_info, color=BLUE, lw=2, label="HAR_L4 a HAR_L1-hez képest (külső információ)")
    ax.plot(gain_lstm.index, gain_lstm, color=ORANGE, lw=2, label="hibrid LSTM_L4 a HAR_L4-hez képest (nemlineáris korrekció)")
    ax.axhline(0, color=INK2, lw=0.8)
    ax.yaxis.set_major_formatter(comma(1))
    ax.set_ylabel("kumulált QLIKE-előny (felfelé: jobb)")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.88), fontsize=8.5)
    main_title(ax, "Mikor keletkezik az előny? Kumulált QLIKE-különbség, 2018–2026")
    ax.text(0.99, 0.03, "Log, 3 év, heti, Garman–Klass", transform=ax.transAxes, ha="right", fontsize=8, color=INK2)
    save(fig, "17_kumulalt_elony")


def f18_forecast_vs_real():
    f = fc("HAR", "L4")
    f = f[f.index >= EVAL_START]
    x, y = f["sigma_hat"] * 100, f["sigma_true"] * 100
    fig, ax = plt.subplots(figsize=(5.8, 5.4))
    ax.scatter(x, y, s=8, color=BLUE, alpha=0.3, lw=0)
    lo, hi = 0.5, 80
    ax.plot([lo, hi], [lo, hi], color=INK2, lw=1, ls="--")
    ax.text(40, 50, "tökéletes\nelőrejelzés", fontsize=8, color=INK2, ha="right")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    for a in (ax.xaxis, ax.yaxis):
        a.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}%".replace(".", ",")))
    ax.set_xlabel("előrejelzett napi szórás (HAR_L4)")
    ax.set_ylabel("valódi napi szórás (holnapi √RV)")
    r = np.corrcoef(np.log(x), np.log(y))[0, 1]
    main_title(ax, "Előrejelzés és valóság, 1841 nap")
    ax.text(0.03, 0.95, f"log-korreláció: {hu(r, 2)}", transform=ax.transAxes, fontsize=9, va="top")
    save(fig, "18_elorejelzes_vs_valosag")


def f19_mcs():
    mcs = pd.read_csv(TABLES / "mcs.csv")
    parts = mcs["Model name"].str.split("|", expand=True)
    parts.columns = KEY
    parts["in"] = mcs["Pvalue"] > 0.10
    dims = [("family", "modellcsalád", {"HAR": "HAR-X", "LSTM": "hibrid"}), ("level", "lépcső", None),
            ("measure", "mérték", {"RV": "RV", "RV_CJ": "RV-CJ", "Parkinson": "PK", "Garman_Klass": "GK"}),
            ("transformation", "transzf.", {"Log": "log", "QuarticRoot": "√"}),
            ("window", "ablak", {"2Y": "2 év", "3Y": "3 év"}), ("frequency", "újrabecslés", {"Weekly": "heti", "Monthly": "havi"})]
    fig, ax = plt.subplots(1, 6, figsize=(11, 3.4), sharey=True, gridspec_kw={"wspace": 0.12})
    for a, (col, title, lab) in zip(ax, dims):
        s = parts.groupby(col)["in"].mean()
        order = {"level": list(LEVELS), "measure": ["RV", "RV_CJ", "Parkinson", "Garman_Klass"],
                 "transformation": ["Log", "QuarticRoot"], "window": ["2Y", "3Y"], "frequency": ["Weekly", "Monthly"],
                 "family": ["HAR", "LSTM"]}
        s = s.reindex(order[col])
        a.bar(range(len(s)), s.values, color=BLUE, width=0.65)
        for k, v in enumerate(s.values):
            a.text(k, v + 0.02, f"{v:.0%}", ha="center", fontsize=8)
        a.set_xticks(range(len(s)), [lab[i] if lab else i for i in s.index], fontsize=8.5)
        a.set_title(title, fontsize=10)
        a.grid(axis="x", visible=False)
        a.set_ylim(0, 1.12)
    ax[0].yaxis.set_major_formatter(pct())
    ax[0].set_ylabel("a legjobbak halmazában (MCS, 10%)")
    sup(fig, f"Model Confidence Set: {int(parts['in'].sum())} modell marad a 256-ból – mely beállítások jutnak be?",
                 x=0.07, ha="left", y=1.04, fontsize=11.5, weight="bold")
    save(fig, "19_mcs")


def f20_coefficients():
    c = pd.read_csv(model_dir("HAR", "L4", *CFG) / "coefficients.csv", index_col="refit_date", parse_dates=True)
    show = [("har_d", "HAR napi"), ("har_w", "HAR heti"), ("har_m", "HAR havi"), ("ovx", "OVX")]
    fig, ax = plt.subplots(2, 2, figsize=(10, 5), sharex=True, gridspec_kw={"hspace": 0.3, "wspace": 0.15})
    for a, (col, title) in zip(ax.ravel(), show):
        shade_crises(a, label=False)
        a.plot(c.index, c[col], color=BLUE, lw=1.4)
        a.axhline(0, color=INK2, lw=0.8)
        a.set_title(title, fontsize=10)
        a.yaxis.set_major_formatter(comma(2))
        a.xaxis.set_major_locator(mdates.YearLocator(2))
        a.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    sup(fig, "A HAR_L4 fő együtthatói heti újrabecslésenként (3 éves ablak, log-szórás)", x=0.07, ha="left", y=1.0,
                 fontsize=11.5, weight="bold")
    save(fig, "20_egyutthatok_idoben")


# =====================================================================================================
# KIEGÉSZÍTÉS: LSTM-kísérletek (extra/)
# =====================================================================================================
def f21_lstm_experiments():
    c = naive_and_best()
    full = c[c["sample"] == "full"].set_index("model")["QLIKE"]
    rows = [("Naiv: holnap = ma", "Naiv (holnap = ma)", GRAY), ("HAR_L4 (legjobb)", "HAR_L4", BLUE),
            ("Hibrid: HAR-X + LSTM", "LSTM_L4 (hibrid)", BLUE), ("Tiszta LSTM", "LSTM_Pure (tiszta)", ORANGE),
            ("Maximalizált LSTM, horgony nélkül", "LSTM_Max Strict", ORANGE), ("Horgonyos LSTM (bővülő ablak)", "LSTM_Max Free", AQUA)]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    y = np.arange(len(rows))[::-1]
    base = full["HAR_L4"]
    for yi, (lab, key, col) in zip(y, rows):
        v = full[key] / base
        ax.barh(yi, v, color=col, height=0.6)
        ax.text(v, yi, f"  {hu(v, 2)}×", va="center", fontsize=9)
    ax.axvline(1, color=INK2, lw=1, ls="--")
    ax.set_yticks(y, [r[0] for r in rows])
    ax.set_xlim(0, 3.1)
    ax.xaxis.set_major_formatter(comma(1))
    ax.set_xlabel("QLIKE a HAR_L4-hez viszonyítva (1 = a legjobb modell; nagyobb = rosszabb)")
    ax.grid(axis="y", visible=False)
    main_title(ax, "Az LSTM horgony nélkül még a naiv tippnél is rosszabb")
    save(fig, "21_lstm_kiserletek")


def f22_saturation():
    h4 = fc("HAR", "L4")
    pure = pd.read_csv(ROOT / "extra" / "LSTM_Pure" / "forecasts.csv", index_col="date", parse_dates=True)
    free = pd.read_csv(ROOT / "extra" / "LSTM_Max" / "Free" / "forecasts.csv", index_col="date", parse_dates=True)
    sl = slice("2020-02-01", "2020-06-30")
    fig, ax = plt.subplots(figsize=(10, 4))
    d = h4.loc[sl]
    ax.plot(d.index, d["sigma_true"] * 100, color=GRAY, lw=1.1, label="valódi (holnapi √RV)")
    ax.plot(d.index, d["sigma_hat"] * 100, color=BLUE, lw=1.8, label="HAR_L4")
    ax.plot(d.index, pure.loc[sl, "sigma_hat"] * 100, color=ORANGE, lw=1.8, label="tiszta LSTM")
    ax.plot(d.index, free.loc[sl, "sigma_hat"] * 100, color=AQUA, lw=1.8, ls="--", label="horgonyos LSTM (bővülő ablak)")
    ax.set_yscale("log")
    logpct(ax.yaxis)
    ax.set_ylabel("napi szórás (log skála)")
    ax.legend(loc="upper left", fontsize=8.5)
    ax.annotate("a tiszta LSTM 2–4,5%-on „megáll”", (pd.Timestamp("2020-04-21"), pure.loc["2020-04-21", "sigma_hat"] * 100),
                (pd.Timestamp("2020-04-28"), 1.7), fontsize=8.5, arrowprops=dict(arrowstyle="-|>", color=ORANGE))
    main_title(ax, "Telítődés: horgony nélkül a háló nem tudja követni a COVID-sokkot")
    save(fig, "22_lstm_telitodes_covid")


if __name__ == "__main__":
    for f in [f01_price_vol, f02_jump_days, f03_persistence, f04_correlation, f05_timing, f06_sample_timeline,
              f07_framework, f08_hybrid, f09_modelling_gain, f10_h1_measures, f11_h1_years, f12_h2_levels,
              f13_h3_lstm, f14_h5_effects, f15_crisis_heatmap, f16_crisis_paths, f17_cumulative,
              f18_forecast_vs_real, f19_mcs, f20_coefficients, f21_lstm_experiments, f22_saturation]:
        f()
