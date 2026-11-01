"""HAR_L2 | QuarticRoot | 3Y_Weekly | Garman_Klass

MODELL: HAR-X: lineáris regresszió (OLS) gördülő tanulóablakon.
    y_hat = b0 + b' x_t

BEÁLLÍTÁSOK
  Mérték:           Garman_Klass – Garman–Klass-variancia az 5 perces bárok OHLC-jéből
  Lépcső:           L2 – az L1 + 4 makróváltozó
  Transzformáció:   QuarticRoot – gyök(szórás) = variancia^(1/4)
  Tanulóablak:      3Y = az utolsó 750 kereskedési nap
  Újrabecslés:      hetente (minden naptári hét első napján)
  Cél:              a holnapi napi szórás (sqrt RV), QuarticRoot transzformálva;
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
  - ELTÉRÉS a(z) HAR_L1-hez képest: + ovx, vix, sp500, usd
  - Az azonos beállítású LSTM_L2 modell ugyanezt a lineáris részt használja, és egy LSTM-korrekciót ad hozzá.
  - A többi mérték / transzformáció / ablak / gyakoriság modelljei ugyanígy épülnek fel, csak a fenti beállításokban térnek el.

EBBEN A MAPPÁBAN
  model.py         ez a fájl: a modell beállításai és futtatása
  forecasts.csv    előrejelzések: date, sigma_true, sigma_hat, y_true, y_hat, resid_var
  coefficients.csv a becsült együtthatók újrabecslésenként
  metrics.csv      hibamérők részmintánként (scripts/06_evaluate.py tölti ki)

FUTTATÁS (a projekt bármely mappájából):  python "models/HAR_L2_Macro/QuarticRoot/3Y_Weekly/Garman_Klass/model.py"
A számítás a közös, tesztelt motorban történik (src/har.py); ez a fájl csak a beállításokat adja át,
így minden modell pontosan ugyanazzal a kóddal készül.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[5] / "src"))   # a projekt src/ mappája

from runner import run_model

# --- A modell beállításai -------------------------------------------------------------------
FAMILY = "HAR"
LEVEL = "L2"
TRANSFORMATION = "QuarticRoot"
WINDOW = "3Y"
FREQUENCY = "Weekly"
MEASURE = "Garman_Klass"

if __name__ == "__main__":
    forecasts = run_model(FAMILY, LEVEL, TRANSFORMATION, WINDOW, FREQUENCY, MEASURE)
    print(f"{FAMILY}_{LEVEL} {TRANSFORMATION} {WINDOW}_{FREQUENCY} {MEASURE}: "
          f"{len(forecasts)} előrejelzés -> {Path(__file__).parent / 'forecasts.csv'}")
