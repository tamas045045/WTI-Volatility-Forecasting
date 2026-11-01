# Data

## Availability (English)

| Data | Source | In this repository? |
|---|---|---|
| NYMEX WTI crude oil futures (CL), 5-minute OHLCV, 2009–2026 | [Kibot](https://www.kibot.com) (purchased, licensed) | **No** – licensed data, not redistributed |
| S&P 500, US Dollar Index, VIX daily OHLC | Investing.com export | **No** – third-party terms |
| Copper, gold, natural gas, wheat, corn, soybean futures daily OHLC | Investing.com export | **No** – third-party terms |
| OVX (CBOE Crude Oil Volatility Index) | [FRED, OVXCLS](https://fred.stlouisfed.org/series/OVXCLS) | Yes – `raw/market/OVXCLS.csv` |
| GDELT GKG news variables | [GDELT Project](https://www.gdeltproject.org) via BigQuery | Yes – `wti_pipeline/gkg_news_daily.parquet` |
| Derived daily tables (volatility measures, target, external and news features, model inputs) | computed by this project | Yes – `processed/`, `input/`, `wti_pipeline/wti_daily.parquet` |

To rebuild everything from scratch, obtain the licensed files and place them at the paths listed in `SHA256SUMS.txt`
(the checksums verify that your copy matches the one used in the paper), then run `scripts/01_build_data.py`.
Everything from `scripts/02_build_inputs.py` onwards runs on the included derived tables.

---

## Részletes leírás (magyarul)

Az ellenőrzőösszegek a `SHA256SUMS.txt`-ben vannak. A `raw/` és a `wti_pipeline/` fájljait **soha nem módosítjuk**.

## A WTI 5 perces adat forrása

- **Szolgáltató: Kibot** (kibot.com), megvásárolt adat (a felhasználó közlése, 2026-10-11). NYMEX WTI Crude Oil futures (CL), folytonos sorozat, 5 perces OHLCV, 2009-09-27 – 2026-07-13.
- **Roll-módszertan (a Kibot GYIK-ja szerint):** naptár-alapú, kiigazítás nélküli (unadjusted) roll, szerződésenként fix eltolással (0–10 kereskedési nappal a lejárat előtt). A roll-dátumokat a Kibot közzéteszi. Back-adjustment nincs.
- **Időzóna (Kibot):** US Eastern Time. A WTI-projekt auditja ezt megerősítette (óraátállítással; a vasárnapi nyitás télen és nyáron is 18:00).
- **Bárcímke:** a Kibot GYIK-ja nem rögzíti. Az adat alapján a címke a bár *kezdete* (az első bár 18:00, az utolsó 16:55; WTI-audit).
- **A mi roll-kezelésünk:** mivel a sorozat kiigazítatlan, a roll-napi ugrás a session-határon jelentkezik. A hozamok csak sessionön belül számolódnak, így a roll-ugrás egyetlen napi mértékbe sem kerül be.
  A WTI-audit szerint a roll 2011 szeptembere óta kb. 2 kereskedési nappal a lejárat előtt történik.
- **Még pontosítandó (a dolgozat Adat fejezetéhez):** a Kibot-fiókban közzétett pontos CL roll-eltolás, amivel az audit becslése ellenőrizhető.

## raw/ – nyers fájlok (másolat)

| Fájl | Forrás | Megjegyzés |
|---|---|---|
| `market/OVXCLS.csv` | FRED, OVXCLS (a WTI-projekt `data/raw/market`-ból) | napi záró, 2007-05 – 2026-09 |
| `market/Volatility_Index.csv`, `SP500.csv`, `USD_Index.csv` | Investing.com export (WTI-projekt `data/raw/market`) | `;`-vel tagolt, napi OHLC, 2009-10 – 2026-07 |
| `crossasset/*_Futures.csv` (réz, arany, földgáz, búza, kukorica, szója) | Investing.com export (NN-projekt `data/Features/Layer2_CrossAsset/Raw`) | ugyanaz a formátum |

## wti_pipeline/ – a WTI-projekt auditált, kész táblái

A WTI_News_Absorption projekt tesztelt pipeline-ja készítette őket (`src/wti_absorption/wti.py`, `gkg.py`).
A részletes leírás: a WTI-projekt `docs/timing_protocol.md` és `reports/audit/`, illetve `docs/05_GDELT_GKG_leiras.pdf`.

| Fájl | Tartalom |
|---|---|
| `wti_5min_sessions.parquet` | WTI 5 perces bárok kereskedési naphoz rendelve ([D−1 18:00, D 17:00) New York), sessionön belüli log-hozam `r` |
| `wti_daily.parquet` | naptár: minden kereskedési nap, `t_D` (előrejelzési időpont), `valid` (≥ 234 bár, nem rövidített nap), auditált `rv` |
| `gkg_news_daily.parquet` | GDELT GKG hírváltozók kereskedési naphoz rendelve (csak t_D előtti cikkek), `gkg_missing` jelölő |

Miért nem futtatjuk újra a WTI-pipeline-t? Azért, mert az adatfeldolgozás része (GDELT-letöltés, roll-azonosítás, audit)
már kész és ellenőrzött, a V3 kódja így a mi keretünkre szűkíthető. A V3 az RV-t újraszámolja, és ellenőrzi, hogy egyezik
a WTI-projektével (`scripts/01_build_data.py`).

## processed/ – a V3 napi táblái (`scripts/01_build_data.py`)

| Fájl | Tartalom |
|---|---|
| `measures_daily.csv` | RV, BPV, J, C, Parkinson, Garman_Klass (5 perces bárokból), érvényes napokra |
| `target_daily.csv` | `sigma_next` = √RV a következő kereskedési napon (ha az érvényes) |
| `external_daily.csv` | L2 + L3 változók napi szórása, és melyik napi megfigyelést használtuk (`*_obs_date`) |
| `news_daily.csv` | L4: `news_abnormal`, `news_tone`, `news_tone_disp` |

## input/ – a közös input táblák (`scripts/02_build_inputs.py`)

`input/<Transformation>/<Measure>.csv`, 2 × 4 = 8 tábla, mind ugyanazon a **közös mintán**: 2591 nap, 2015-06-25 – 2026-07-09.
Oszlopok:
- `sigma_next`, `y` (cél);
- HAR-tagok (`har_d/w/m`, illetve RV_CJ-nél `har_c_*`, `har_j_*`);
- L2: `ovx`, `vix`, `sp500`, `usd`;
- L3: `copper`, `gold`, `natgas`, `wheat`, `corn`, `soybeans`;
- L4: `news_abnormal`, `news_tone`, `news_tone_disp`.
