# A WTI nyersolaj volatilitásának előrejelzése – HAR-X és LSTM

TDK-dolgozat (Budapesti Corvinus Egyetem, 2026–27), György Tamás; témavezető: Dr. Bihary Zsolt. A projekt 256 előrejelző modellt hasonlít össze a WTI holnapi napi szórására.

**A keret:** 2 modellcsalád × 4 információs lépcső × 4 volatilitásmérték × 2 transzformáció × 2 tanulóablak × 2 újrabecslési gyakoriság.

## Hol mit találsz

| Mappa / fájl | Tartalom |
|---|---|
| `data/` | nyers és feldolgozott adatok, a közös input táblák (`data/README.md`) |
| `models/<Család>_<Lépcső>/<Transzformáció>/<Ablak>_<Gyakoriság>/<Mérték>/` | **egy mappa = egy modell**: `model.py` (a modell kódja és leírása), `forecasts.csv`, `metrics.csv` |
| `models/<Család>_<Lépcső>/run.py` | a lépcső mind a 32 modelljének futtatása |
| `src/` | a közös, tesztelt motor: `config.py`, `measures.py`, `external.py`, `inputs.py`, `forecasting.py`, `har.py`, `lstm.py`, `runner.py`, `evaluation.py` |
| `scripts/` | a teljes futtatás lépései, a futtatás sorrendjében számozva (00–08; a 08 a dolgozat ábráit készíti -> `results/figures/dolgozat/`) |
| `results/` | `tables/` (hibamérők, hipotézistesztek, MCS, kritikus időszakok), `figures/` (ábrák), `logs/` (futási naplók) |
| `notebooks/` | **kipróbálás és ellenőrzés**: 4 notebook, amelyek a kulcslépéseket kézzel újraszámolják és összevetik a projekt eredményeivel (lásd lent) |
| `tests/` | tesztek: képletek, időzítés (nincs jövőbeli adat), reprodukálhatóság |
| `extra/` | **kísérletek a 256 modell keretén kívül** (a meglévő kódhoz nem nyúlnak): `LSTM_Pure/` – tiszta LSTM, HAR-X nélkül; `LSTM_Max/` – maximalizált LSTM két változatban; `LSTM_Max/FreeRolling/` – lineáris horgonyú, gördülő ablakos LSTM; `HAR_OVX/` – csak OVX-szel bővített HAR; `Naive/` – naiv benchmark. |

## Futtatás

```
python -m venv venv
venv\Scripts\pip install -r requirements.txt

venv\Scripts\python scripts\00_make_model_files.py   # a 256 model.py legyártása (csak ha a sablon változik)
venv\Scripts\python scripts\01_build_data.py         # napi mértékek, külső adatok, hírek
venv\Scripts\python scripts\02_build_inputs.py       # a 8 közös input tábla
venv\Scripts\python scripts\03_run_har.py            # a 128 HAR-X modell (másodpercek)
venv\Scripts\python scripts\04_tune_lstm.py          # az LSTM-beállítások kiválasztása (a tesztminta előtt)
venv\Scripts\python scripts\05_run_lstm.py           # a 128 LSTM-modell (párhuzamosan, kb. 14,5 óra)
venv\Scripts\python scripts\06_evaluate.py           # kiértékelés, hipotézisek -> results/tables/
venv\Scripts\python scripts\07_make_figures.py       # ábrák -> results/figures/
venv\Scripts\python -m pytest tests                  # tesztek
```

**Egyetlen modell futtatása:** például `venv\Scripts\python models\HAR_L2_Macro\Log\3Y_Weekly\RV\model.py`

## Kipróbálás és ellenőrzés: `notebooks/`

A leggyorsabb belépési pont. Mindegyik notebook a közös motort (`src/`) hívja, a kulcslépéseket egyszerű képletekkel
**kézzel is újraszámolja**, és összeveti a projekt tárolt eredményével. A beállítások (nap, modell) szabadon átírhatók.
A kimenetek a notebookokban el vannak mentve, így futtatás nélkül is olvashatók.

| Notebook | Mit ellenőriz | Futási idő |
|---|---|---|
| `01_adatok_es_mertekek.ipynb` | egy nap RV, BPV, ugrásteszt, Parkinson, GK értéke kézzel; az input tábla HAR-tagjai és célja | ~5 mp |
| `02_egy_HAR_modell.ipynb` | egy HAR-X modell futtatása; egy nap előrejelzése kézi OLS-sel; szivárgásteszt (a jövő adatai nem hatnak); QLIKE kézzel | ~5 mp |
| `03_LSTM_korrekcio.ipynb` | egy LSTM-modell rövid szakaszának újraszámolása (alapból a 2020-as COVID-időszak), egyezés a tárolt előrejelzéssel | ~20 mp |
| `04_kiertekeles_es_hipotezisek.ipynb` | hibamérők és egy Giacomini–White-teszt kézzel; a hipotézisek, MCS, ábrák; a 21 automatikus teszt | ~20 mp |

**Megnyitás VS Code-ban:** nyisd meg a notebookot, jobb felül a *Select Kernel* gombbal válaszd a projekt `venv` környezetét,
majd *Run All* (vagy cellánként Shift+Enter).
