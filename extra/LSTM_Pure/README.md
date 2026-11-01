# Kiegészítő összevetés: tiszta LSTM (egy modell)

Kérdés: mennyit ér a hibrid felépítés? Ehhez a legjobb modell beállításain (L4 | Log | 3Y_Weekly | Garman_Klass)
egy **HAR-X nélküli, tiszta LSTM** is készült. Ez **nem része a 256 modellnek**: nem szerepel a hipotézisekben,
az MCS-ben és a `results/` tábláiban, és a meglévő kódhoz (src/, scripts/, models/) nem nyúl.

## Ami azonos a hibrid LSTM_L4 modellel

- input tábla, a 16 változó, a cél (a holnapi szórás logja), 750 napos ablak, heti újrabecslés;
- korai leállítás, súlycsökkentés, 5 seed, negyedéves újraindítás, ugyanazok a seedek;
- kiértékelés: ugyanaz az 1841 nap, ugyanazok a részminták és hibamérők (QLIKE, MSE, R²_OOS a HAR_L1-hez).

## Az egyetlen különbség

| | Hibrid (LSTM_L4) | Tiszta LSTM |
|---|---|---|
| Előrejelzés | HAR-X + LSTM-korrekció | az ablak átlaga + LSTM |
| Mit tanul a háló | az OLS hibáit | a teljes célt |
| s² a visszaváltáshoz | az OLS-reziduumokból | a háló saját hibáiból a tanulóablakon |

## Korrekt hangolás

A hiperparamétereket (sorozathossz, rejtett méret, tanulási ráta) **ugyanaz az eljárás** választja ki, mint a
hibridnél (`scripts/04_tune_lstm.py`): ugyanaz a 8 kombinációs rács, ugyanaz a validációs beállítás és ugyanaz a
validációs időszak, a tesztidőszak előtt. Eredmény: `tuning.csv`.

## Futtatás

```
venv\Scripts\python extra\LSTM_Pure\tune.py     # hangolás (kb. 5–10 perc)
venv\Scripts\python extra\LSTM_Pure\model.py    # a modell és a kiértékelés (kb. 1–3 óra)
```

Eredmények: `forecasts.csv`, `metrics.csv`, `comparison.csv` (a három modell egymás mellett).
