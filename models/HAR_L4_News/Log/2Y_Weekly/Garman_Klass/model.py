"""HAR_L4 | Log | 2Y_Weekly | Garman_Klass

MODELL: HAR-X: lineáris regresszió (OLS) gördülő tanulóablakon.
    y_hat = b0 + b' x_t

BEÁLLÍTÁSOK
  Mérték:           Garman_Klass – Garman–Klass-variancia az 5 perces bárok OHLC-jéből
  Lépcső:           L4 – az L3 + 3 hírváltozó (GDELT GKG)
  Transzformáció:   Log – log(szórás)
  Tanulóablak:      2Y = az utolsó 500 kereskedési nap
  Újrabecslés:      hetente (minden naptári hét első napján)
  Cél:              a holnapi napi szórás (sqrt RV), Log transzformálva;
                    a kiértékelés szórás-skálán történik (forecasts.csv: sigma_true vs. sigma_hat)

MAGYARÁZÓ VÁLTOZÓK (16 db)
  har_d           a mérték napi szórása (ma)
  har_w           az elmúlt 5 nap átlagos varianciájának gyöke
  har_m           az elmúlt 22 nap átlagos varianciájának gyöke
  ovx             OVX (olaj implikált vol.) napi szórásban
  vix             VIX napi szórásban
  sp500           S&P 500 napi range-szórása
  usd             USD index napi range-szórása
  copper          réz
  gold            arany
  natgas          földgáz
  wheat           búza
  corn            kukorica
  soybeans        szója
  news_abnormal   olajhírek szokatlan mennyisége (z-érték)
  news_tone       olajhírek átlagos hangulata
  news_tone_disp  a hírek hangulatának szórása (megosztottság)

MIBEN TÉR EL A SZOMSZÉDAITÓL
  - ELTÉRÉS a(z) HAR_L3-hez képest: + news_abnormal, news_tone, news_tone_disp
  - Az azonos beállítású LSTM_L4 modell ugyanezt a lineáris részt használja, és egy LSTM-korrekciót ad hozzá.
  - A többi mérték / transzformáció / ablak / gyakoriság modelljei ugyanígy épülnek fel, csak a fenti beállításokban térnek el.

EBBEN A MAPPÁBAN
  model.py         ez a fájl: a modell beállításai és futtatása
  forecasts.csv    előrejelzések: date, sigma_true, sigma_hat, y_true, y_hat, resid_var
  coefficients.csv a becsült együtthatók újrabecslésenként
  metrics.csv      hibamérők részmintánként (scripts/06_evaluate.py tölti ki)

FUTTATÁS (a projekt bármely mappájából):  python "models/HAR_L4_News/Log/2Y_Weekly/Garman_Klass/model.py"
A számítás a közös, tesztelt motorban történik (src/har.py); ez a fájl csak a beállításokat adja át,
így minden modell pontosan ugyanazzal a kóddal készül.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[5] / "src"))   # a projekt src/ mappája

from runner import run_model

# --- A modell beállításai -------------------------------------------------------------------
FAMILY = "HAR"
LEVEL = "L4"
TRANSFORMATION = "Log"
WINDOW = "2Y"
FREQUENCY = "Weekly"
MEASURE = "Garman_Klass"

if __name__ == "__main__":
    forecasts = run_model(FAMILY, LEVEL, TRANSFORMATION, WINDOW, FREQUENCY, MEASURE)
    print(f"{FAMILY}_{LEVEL} {TRANSFORMATION} {WINDOW}_{FREQUENCY} {MEASURE}: "
          f"{len(forecasts)} előrejelzés -> {Path(__file__).parent / 'forecasts.csv'}")
