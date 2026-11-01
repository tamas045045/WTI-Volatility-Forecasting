"""0. lépés: minden modell mappájába egy önállóan futtatható model.py (256 db, azonos felépítéssel).

A model.py-k egyetlen sablonból készülnek, így garantáltan egyformák; csak a beállításaik és a
leírásuk különbözik. Ha a sablon változik, ezt a szkriptet kell újra futtatni.
Futtatás:  python scripts/00_make_model_files.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import (CROSSASSET_COLUMNS, LEVELS, MACRO_COLUMNS, NEWS_COLUMNS, WINDOWS, all_models,
                    level_columns, model_dir)

FAMILY_TEXT = {
    "HAR": ("HAR-X: lineáris regresszió (OLS) gördülő tanulóablakon.",
            "y_hat = b0 + b' x_t"),
    "LSTM": ("HAR-X + LSTM-korrekció: a lineáris rész pontosan az azonos beállítású HAR-X modell, amelyhez\n"
             "egy LSTM-háló az utolsó 22 nap változóiból nemlineáris korrekciót tanul (docs/09_lstm_protokoll.md).",
             "y_hat = (b0 + b' x_t) + LSTM(x_(t-21), …, x_t)"),
}
LEVEL_TEXT = {
    "L1": "csak a mérték saját napi, heti és havi tagja (alapmodell)",
    "L2": "az L1 + 4 makróváltozó",
    "L3": "az L2 + 6 nyersanyag (cross-asset)",
    "L4": "az L3 + 3 hírváltozó (GDELT GKG)",
}
VAR_TEXT = {
    "har_d": "a mérték napi szórása (ma)", "har_w": "az elmúlt 5 nap átlagos varianciájának gyöke",
    "har_m": "az elmúlt 22 nap átlagos varianciájának gyöke",
    "har_c_d": "a folytonos rész (C) napi szórása", "har_c_w": "C, 5 napos", "har_c_m": "C, 22 napos",
    "har_j_d": "az ugrásrész (J) napi gyöke", "har_j_w": "J, 5 napos", "har_j_m": "J, 22 napos",
    "ovx": "OVX (olaj implikált vol.) napi szórásban", "vix": "VIX napi szórásban",
    "sp500": "S&P 500 napi range-szórása", "usd": "USD index napi range-szórása",
    "copper": "réz", "gold": "arany", "natgas": "földgáz", "wheat": "búza", "corn": "kukorica", "soybeans": "szója",
    "news_abnormal": "olajhírek szokatlan mennyisége (z-érték)", "news_tone": "olajhírek átlagos hangulata",
    "news_tone_disp": "a hírek hangulatának szórása (megosztottság)",
}
MEASURE_TEXT = {
    "RV": "realizált variancia (5 perces hozamokból)",
    "RV_CJ": "realizált variancia folytonos (C) és ugrás (J) részre bontva",
    "Parkinson": "Parkinson-variancia az 5 perces bárok terjedelméből",
    "Garman_Klass": "Garman–Klass-variancia az 5 perces bárok OHLC-jéből",
}
TRANSFORMATION_TEXT = {"Log": "log(szórás)", "QuarticRoot": "gyök(szórás) = variancia^(1/4)"}
FREQ_TEXT = {"Weekly": "hetente (minden naptári hét első napján)", "Monthly": "havonta (minden hónap első napján)"}

TEMPLATE = '''"""{family}_{level} | {transformation} | {window}_{frequency} | {measure}

MODELL: {family_text}
    {equation}

BEÁLLÍTÁSOK
  Mérték:           {measure} – {measure_text}
  Lépcső:           {level} – {level_text}
  Transzformáció:   {transformation} – {transformation_text}
  Tanulóablak:      {window} = az utolsó {rows} kereskedési nap
  Újrabecslés:      {frequency_text}
  Cél:              a holnapi napi szórás (sqrt RV), {transformation} transzformálva;
                    a kiértékelés szórás-skálán történik (forecasts.csv: sigma_true vs. sigma_hat)

MAGYARÁZÓ VÁLTOZÓK ({n_vars} db)
{variables}

MIBEN TÉR EL A SZOMSZÉDAITÓL
{differences}

EBBEN A MAPPÁBAN
  model.py         ez a fájl: a modell beállításai és futtatása
  forecasts.csv    előrejelzések: date, sigma_true, sigma_hat, y_true, y_hat, resid_var
  {extra_file}
  metrics.csv      hibamérők részmintánként (scripts/06_evaluate.py tölti ki)

FUTTATÁS (a projekt bármely mappájából):  python "{relpath}"
A számítás a közös, tesztelt motorban történik ({engine}); ez a fájl csak a beállításokat adja át,
így minden modell pontosan ugyanazzal a kóddal készül.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[5] / "src"))   # a projekt src/ mappája

from runner import run_model

# --- A modell beállításai -------------------------------------------------------------------
FAMILY = "{family}"
LEVEL = "{level}"
TRANSFORMATION = "{transformation}"
WINDOW = "{window}"
FREQUENCY = "{frequency}"
MEASURE = "{measure}"

if __name__ == "__main__":
    forecasts = run_model(FAMILY, LEVEL, TRANSFORMATION, WINDOW, FREQUENCY, MEASURE)
    print(f"{{FAMILY}}_{{LEVEL}} {{TRANSFORMATION}} {{WINDOW}}_{{FREQUENCY}} {{MEASURE}}: "
          f"{{len(forecasts)}} előrejelzés -> {{Path(__file__).parent / 'forecasts.csv'}}")
'''


def differences(family, level):
    lv = list(LEVELS)
    out = []
    i = lv.index(level)
    if i == 0:
        out.append("  - Ez az alapmodell (L1): ehhez mérjük a bővítéseket.")
    else:
        prev = lv[i - 1]
        added = {"L2": MACRO_COLUMNS, "L3": CROSSASSET_COLUMNS, "L4": NEWS_COLUMNS}[level]
        out.append(f"  - ELTÉRÉS a(z) {family}_{prev}-hez képest: + {', '.join(added)}")
    other = "LSTM" if family == "HAR" else "HAR"
    if family == "HAR":
        out.append(f"  - Az azonos beállítású {other}_{level} modell ugyanezt a lineáris részt használja, és egy LSTM-korrekciót ad hozzá.")
    else:
        out.append(f"  - ELTÉRÉS a(z) HAR_{level}-hez képest: + LSTM-korrekció (a lineáris rész azonos).")
    out.append("  - A többi mérték / transzformáció / ablak / gyakoriság modelljei ugyanígy épülnek fel, csak a fenti beállításokban térnek el.")
    return "\n".join(out)


def main():
    root = Path(__file__).resolve().parents[1]
    n = 0
    for fam, lv, tr, w, f, m in all_models():
        d = model_dir(fam, lv, tr, w, f, m)
        d.mkdir(parents=True, exist_ok=True)
        cols = level_columns(lv, m)
        text = TEMPLATE.format(
            family=fam, level=lv, transformation=tr, window=w, frequency=f, measure=m,
            family_text=FAMILY_TEXT[fam][0], equation=FAMILY_TEXT[fam][1],
            measure_text=MEASURE_TEXT[m], level_text=LEVEL_TEXT[lv], transformation_text=TRANSFORMATION_TEXT[tr],
            rows=WINDOWS[w], frequency_text=FREQ_TEXT[f], n_vars=len(cols),
            variables="\n".join(f"  {c:<15} {VAR_TEXT[c]}" for c in cols),
            differences=differences(fam, lv),
            extra_file="coefficients.csv a becsült együtthatók újrabecslésenként" if fam == "HAR"
            else "linear_part.csv  a lineáris rész (= HAR-X) előrejelzése, összevetéshez",
            relpath=(d / "model.py").relative_to(root).as_posix(),
            engine="src/har.py" if fam == "HAR" else "src/lstm.py",
        )
        (d / "model.py").write_text(text, encoding="utf-8")
        n += 1
    print(f"{n} model.py elkészült")


if __name__ == "__main__":
    main()
