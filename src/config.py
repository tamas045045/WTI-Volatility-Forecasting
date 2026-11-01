"""A projekt közös beállításai: útvonalak, a keret dimenziói és a lépcsők változólistái.

Minden más modul innen olvas. Ha a keret változik, csak itt kell módosítani
(és a docs/02_keretrendszer_vazlat.md-ben dokumentálni).
"""
from pathlib import Path

# --- Útvonalak ------------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW_MARKET = DATA / "raw" / "market"
RAW_CROSSASSET = DATA / "raw" / "crossasset"
WTI_PIPELINE = DATA / "wti_pipeline"          # a WTI-projekt auditált, kész táblái (lásd data/README.md)
PROCESSED = DATA / "processed"
INPUT = DATA / "input"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"      # összesítő CSV-k (06_evaluate)
FIGURES = RESULTS / "figures"    # ábrák (07_make_figures)
LOGS = RESULTS / "logs"          # futási naplók

NY_TZ = "America/New_York"

# --- A keret dimenziói (docs/02_keretrendszer_vazlat.md) ---------------------------------------
FAMILIES = ["HAR", "LSTM"]
TRANSFORMATIONS = ["Log", "QuarticRoot"]              # log(sigma), illetve sqrt(sigma) = variancia^(1/4)
MEASURES = ["RV", "RV_CJ", "Parkinson", "Garman_Klass"]
WINDOWS = {"2Y": 500, "3Y": 750}                       # tanulóablak érvényes kereskedési napokban
FREQUENCIES = {"Weekly": "W", "Monthly": "M"}         # újrabecslés: a naptári hét / hónap váltásakor

# A modellmappák neve: <család>_<lépcső>
LEVELS = {"L1": "Technical", "L2": "Macro", "L3": "CrossAsset", "L4": "News"}

# --- Változók lépcsőnként (kumulatív) ------------------------------------------------------
# L1: a HAR-tagok a mérték saját napi, heti (5) és havi (22) értékei. Az RV_CJ-nél a folytonos (C)
# és az ugrás (J) rész külön tagokat kap (Andersen, Bollerslev, Diebold 2007).
HAR_COLUMNS = {
    "RV": ["har_d", "har_w", "har_m"],
    "Parkinson": ["har_d", "har_w", "har_m"],
    "Garman_Klass": ["har_d", "har_w", "har_m"],
    "RV_CJ": ["har_c_d", "har_c_w", "har_c_m", "har_j_d", "har_j_w", "har_j_m"],
}
MACRO_COLUMNS = ["ovx", "vix", "sp500", "usd"]                                   # L2 (03_valtozolista)
CROSSASSET_COLUMNS = ["copper", "gold", "natgas", "wheat", "corn", "soybeans"]   # L3 (03_valtozolista)
NEWS_COLUMNS = ["news_abnormal", "news_tone", "news_tone_disp"]                  # L4 (04_hirvaltozok)


def level_columns(level: str, measure: str) -> list[str]:
    """Az adott lépcső összes magyarázó változója (a lépcsők egymásra épülnek)."""
    cols = list(HAR_COLUMNS[measure])
    if level in ("L2", "L3", "L4"):
        cols += MACRO_COLUMNS
    if level in ("L3", "L4"):
        cols += CROSSASSET_COLUMNS
    if level == "L4":
        cols += NEWS_COLUMNS
    return cols


def model_dir(family: str, level: str, transformation: str, window: str, frequency: str, measure: str) -> Path:
    """Egy modell levélmappája: models/<család>_<lépcső>/<transzformáció>/<ablak>_<gyakoriság>/<mérték>/"""
    return MODELS / f"{family}_{level}_{LEVELS[level]}" / transformation / f"{window}_{frequency}" / measure


def all_models():
    """Mind a 256 modell azonosítója (család, lépcső, transzformáció, ablak, gyakoriság, mérték)."""
    for family in FAMILIES:
        for level in LEVELS:
            for tr in TRANSFORMATIONS:
                for w in WINDOWS:
                    for f in FREQUENCIES:
                        for m in MEASURES:
                            yield family, level, tr, w, f, m

# --- LSTM-beállítások (docs/09_lstm_protokoll.md) ----------------------------------------------
# A seq_len, hidden, lr és epochs_* értékeit a tesztminta ELŐTTI validáció választja ki
# (scripts/04_tune_lstm.py, eredmény: results/tables/lstm_tuning.csv); utána mind a 128 LSTM-re fixek.
LSTM_PARAMS = {
    # A validáció nyertese (results/tables/lstm_tuning.csv, 2026-10-10): QLIKE 0,991 × a HAR-X-é a validációs időszakon
    "seq_len": 22,          # a szekvencia hossza (nap)                         <- validáció: {5, 22}
    "hidden": 32,           # az LSTM rejtett mérete                             <- validáció: {16, 32}
    "lr": 1e-3,             # tanulási ráta (AdamW)                              <- validáció: {1e-3, 3e-3}
    "epochs_cold": 300,     # legfeljebb ennyi lépés nulláról indulva (korai leállítással)
    "epochs_warm": 100,     # legfeljebb ennyi lépés folytatott tanításnál (korai leállítással)
    "holdout": 0.2,         # a tanulóablak utolsó 20%-a a korai leállítás validációs része
    "eval_every": 10,       # ennyi lépésenként mérjük a validációs hibát
    "patience": 5,          # ennyi javulás nélküli mérés után leállunk
    "weight_decay": 1e-4,   # súlycsökkentés a túlillesztés ellen
    "cold_every": {"Weekly": 13, "Monthly": 3},   # negyedévente teljes újratanítás
    "seeds": 5,
    "base_seed": 2026,
}
