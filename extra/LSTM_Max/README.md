# Kísérlet: maximalizált LSTM (két változat)

Kérdés: mit lehet kihozni egy LSTM-ből, ha mindent rá állítunk, hogy a lehető legpontosabb legyen?
Ez **kísérlet a 256 modell keretén kívül**: nem szerepel a hipotézisekben, az MCS-ben és a `results/` tábláiban, és a
meglévő kódhoz (src/, scripts/, models/) nem nyúl – a közös építőelemeket csak beolvassa.

## A két változat

| | `Strict/` – szigorú maximum | `Free/` – szabad maximum |
|---|---|---|
| Mit tanul a háló | a holnapi szórás szintjét (mint a tiszta LSTM) | a validáció dönti el: a szintet, vagy a változást a mai szinthez képest („horgony”) |
| Tanulóadat | 3 éves gördülő ablak (mint minden modell) | bővülő ablak: mindig a közös minta elejétől (2015-06) |
| Regresszió (OLS/HAR) | nincs | nincs |

Mindkettő: a legjobb modell beállításai (L4 – mind a 16 változó, Log, Garman–Klass, heti újrabecslés), ugyanaz a cél
és ugyanaz a kiértékelés (1841 nap, QLIKE, MSE, R²_OOS), 10 seed átlaga.

## Hangolás – csak a tesztidőszak előtt

A közös minta 500–749. során (2017-08 – 2018-08), a modell saját beállításán, QLIKE alapján (`tune.py`):

    sorozathossz {5, 22} × rejtett méret {16, 32, 64} × rétegek {1, 2} × dropout {0; 0,2} × tanulási ráta {0,001; 0,003}
    horgony: nincs (Strict és Free) / a mai szint (csak Free)

Ez 48 + 48 kombináció. A bővülő ablakot a validáció nem tudja eldönteni (a minta elején a bővülő és a gördülő ablak
szinte ugyanannyi adatot lát), ezért ez a Free változat előre rögzített tervezési döntése.

## Ellenőrzések

- A motor (`max_lstm.py`) a tiszta LSTM beállításaival bitre ugyanazt adja, mint a már lefutott tiszta LSTM.
- Szivárgásteszt (a jövőbeli adatok elrontása) horgonnyal és bővülő ablakkal is: az eltérés 0.

## Futtatás és eredmények

```
venv\Scripts\python extra\LSTM_Max\tune.py    # hangolás (96 kombináció, párhuzamosan)
venv\Scripts\python extra\LSTM_Max\run.py     # a két végső modell és az összevetés
```

`Strict/` és `Free/`: `tuning.csv`, `forecasts.csv`, `metrics.csv`. `comparison.csv`: HAR_L4, hibrid LSTM_L4,
tiszta LSTM, Strict és Free egymás mellett. A futási napló: `log.txt`.
