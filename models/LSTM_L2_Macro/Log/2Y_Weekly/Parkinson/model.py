"""LSTM_L2 | Log | 2Y_Weekly | Parkinson

MODELL: HAR-X + LSTM-korrekció: a lineáris rész pontosan az azonos beállítású HAR-X modell, amelyhez
egy LSTM-háló az utolsó 22 nap változóiból nemlineáris korrekciót tanul (docs/09_lstm_protokoll.md).
    y_hat = (b0 + b' x_t) + LSTM(x_(t-21), …, x_t)

BEÁLLÍTÁSOK
  Mérték:           Parkinson – Parkinson-variancia az 5 perces bárok terjedelméből
  Lépcső:           L2 – az L1 + 4 makróváltozó
  Transzformáció:   Log – log(szórás)
  Tanulóablak:      2Y = az utolsó 500 kereskedési nap
  Újrabecslés:      hetente (minden naptári hét első napján)
  Cél:              a holnapi napi szórás (sqrt RV), Log transzformálva;
                    a kiértékelés szórás-skálán történik (forecasts.csv: sigma_true vs. sigma_hat)

MAGYARÁZÓ VÁLTOZÓK (7 db)
  har_d           a mérték napi szórása (ma)
  har_w           az elmúlt 5 nap átlagos varianciájának gyöke
  har_m           az elmúlt 22 nap átlagos varianciájának gyöke
  ovx             OVX (olaj implikált vol.) napi szórásban
  vix             VIX napi szórásban
  sp500           S&P 500 napi range-szórása
  usd             USD index napi range-szórása

MIBEN TÉR EL A SZOMSZÉDAITÓL
  - ELTÉRÉS a(z) LSTM_L1-hez képest: + ovx, vix, sp500, usd
  - ELTÉRÉS a(z) HAR_L2-hez képest: + LSTM-korrekció (a lineáris rész azonos).
  - A többi mérték / transzformáció / ablak / gyakoriság modelljei ugyanígy épülnek fel, csak a fenti beállításokban térnek el.

EBBEN A MAPPÁBAN
  model.py         ez a fájl: a modell beállításai és futtatása
  forecasts.csv    előrejelzések: date, sigma_true, sigma_hat, y_true, y_hat, resid_var
  linear_part.csv  a lineáris rész (= HAR-X) előrejelzése, összevetéshez
  metrics.csv      hibamérők részmintánként (scripts/06_evaluate.py tölti ki)

FUTTATÁS (a projekt bármely mappájából):  python "models/LSTM_L2_Macro/Log/2Y_Weekly/Parkinson/model.py"
A számítás a közös, tesztelt motorban történik (src/lstm.py); ez a fájl csak a beállításokat adja át,
így minden modell pontosan ugyanazzal a kóddal készül.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[5] / "src"))   # a projekt src/ mappája

from runner import run_model

# --- A modell beállításai -------------------------------------------------------------------
FAMILY = "LSTM"
LEVEL = "L2"
TRANSFORMATION = "Log"
WINDOW = "2Y"
FREQUENCY = "Weekly"
MEASURE = "Parkinson"

if __name__ == "__main__":
    forecasts = run_model(FAMILY, LEVEL, TRANSFORMATION, WINDOW, FREQUENCY, MEASURE)
    print(f"{FAMILY}_{LEVEL} {TRANSFORMATION} {WINDOW}_{FREQUENCY} {MEASURE}: "
          f"{len(forecasts)} előrejelzés -> {Path(__file__).parent / 'forecasts.csv'}")
