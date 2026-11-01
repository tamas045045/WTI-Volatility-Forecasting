"""HAR_L1 | QuarticRoot | 3Y_Monthly | RV_CJ

MODELL: HAR-X: lineáris regresszió (OLS) gördülő tanulóablakon.
    y_hat = b0 + b' x_t

BEÁLLÍTÁSOK
  Mérték:           RV_CJ – realizált variancia folytonos (C) és ugrás (J) részre bontva
  Lépcső:           L1 – csak a mérték saját napi, heti és havi tagja (alapmodell)
  Transzformáció:   QuarticRoot – gyök(szórás) = variancia^(1/4)
  Tanulóablak:      3Y = az utolsó 750 kereskedési nap
  Újrabecslés:      havonta (minden hónap első napján)
  Cél:              a holnapi napi szórás (sqrt RV), QuarticRoot transzformálva;
                    a kiértékelés szórás-skálán történik (forecasts.csv: sigma_true vs. sigma_hat)

MAGYARÁZÓ VÁLTOZÓK (6 db)
  har_c_d         a folytonos rész (C) napi szórása
  har_c_w         C, 5 napos
  har_c_m         C, 22 napos
  har_j_d         az ugrásrész (J) napi gyöke
  har_j_w         J, 5 napos
  har_j_m         J, 22 napos

MIBEN TÉR EL A SZOMSZÉDAITÓL
  - Ez az alapmodell (L1): ehhez mérjük a bővítéseket.
  - Az azonos beállítású LSTM_L1 modell ugyanezt a lineáris részt használja, és egy LSTM-korrekciót ad hozzá.
  - A többi mérték / transzformáció / ablak / gyakoriság modelljei ugyanígy épülnek fel, csak a fenti beállításokban térnek el.

EBBEN A MAPPÁBAN
  model.py         ez a fájl: a modell beállításai és futtatása
  forecasts.csv    előrejelzések: date, sigma_true, sigma_hat, y_true, y_hat, resid_var
  coefficients.csv a becsült együtthatók újrabecslésenként
  metrics.csv      hibamérők részmintánként (scripts/06_evaluate.py tölti ki)

FUTTATÁS (a projekt bármely mappájából):  python "models/HAR_L1_Technical/QuarticRoot/3Y_Monthly/RV_CJ/model.py"
A számítás a közös, tesztelt motorban történik (src/har.py); ez a fájl csak a beállításokat adja át,
így minden modell pontosan ugyanazzal a kóddal készül.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[5] / "src"))   # a projekt src/ mappája

from runner import run_model

# --- A modell beállításai -------------------------------------------------------------------
FAMILY = "HAR"
LEVEL = "L1"
TRANSFORMATION = "QuarticRoot"
WINDOW = "3Y"
FREQUENCY = "Monthly"
MEASURE = "RV_CJ"

if __name__ == "__main__":
    forecasts = run_model(FAMILY, LEVEL, TRANSFORMATION, WINDOW, FREQUENCY, MEASURE)
    print(f"{FAMILY}_{LEVEL} {TRANSFORMATION} {WINDOW}_{FREQUENCY} {MEASURE}: "
          f"{len(forecasts)} előrejelzés -> {Path(__file__).parent / 'forecasts.csv'}")
