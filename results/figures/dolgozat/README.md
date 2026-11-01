# A dolgozat és a prezentáció ábrái

Készíti: `scripts/08_make_thesis_figures.py` (csak a kész eredményekből rajzol, semmit nem számol újra).
Minden ábra PNG (300 dpi, a dolgozatba és diákra) és PDF (vektoros, nyomtatáshoz) formában.
Színek: kék = a fő/legjobb modell, narancs = az összevetett alternatíva, türkiz = harmadik sor, szürke = a valóság vagy háttér.

| # | Fájl | Fejezet | Mit mutat – mit kell leolvasni |
|---|---|---|---|
| 1 | `01_wti_ar_es_volatilitas` | III.1 | A WTI ára és napi szórása 2015–2026, a két válsággal és a kiértékelés kezdetével. A volatilitás hullámokban jön, és válságban tízszeresére is nő. |
| 2 | `02_atlagos_es_ugrasos_nap` | III.2 | Egy átlagos és egy ugrásos nap 5 perces hozamai. Ugrásos napon egyetlen bár adja a variancia nagy részét (z = 10,8; J = 57%). |
| 3 | `03_mertekek_tartossaga` | III.2, V.2 | A log-szórás autokorrelációja. A Parkinson és a GK minden késleltetésen tartósabb az RV-nél – a H1 eredményének magyarázata. |
| 4 | `04_valtozok_korrelacioja` | III.3–III.4 | A 16 változó korrelációja. A HAR-tagok és az OVX erősen együtt mozognak (0,9); a nyersanyagok és a hírek alig. |
| 5 | `05_idozites_sema` | III.5 | Mi kerülhet bele a t napi előrejelzésbe: csak a 17:00 (t_D) előtt lezárult adatok. A „nincs jövőbeli adat” bizonyítéka. |
| 6 | `06_minta_idovonal` | III.6, IV.5 | A közös minta, az LSTM-hangolás (tesztidőszak előtt) és az 1841 napos kiértékelés; a gördülő 3 éves ablak. |
| 7 | `07_kutatasi_keret` | IV.1 | A 256 modell kerete: 4 kumulatív lépcső × 5 kétszintű/négyszintű dimenzió. |
| 8 | `08_hibrid_modell_sema` | IV.4 | A hibrid LSTM felépítése: a HAR-X adja a szintet, az LSTM csak korrekciót. |
| 9 | `09_modellezes_haszna` | V.1 | Naiv tipp → L1 → L4: a modellezés 24%-kal csökkenti a hibát; a javulás kb. 60%-a a HAR-szerkezet, kb. 40%-a a külső információ. |
| 10 | `10_H1_mertekek` | V.2 | H1: a Parkinson és a GK mind a 64–64 modellpárban jobb az RV-nél; az RV-CJ nem. |
| 11 | `11_H1_evente` | V.2 | H1 évente: a range-alapú mértékek átlagosan mind a 9 évben jobbak. |
| 12 | `12_H2_lepcsok` | V.3 | H2: a javulás szinte teljes egészében az L2 makró-lépcsőből jön; L3 és L4 alig változtat. Mindkét családra. |
| 13 | `13_H3_lstm_vs_har` | V.4 | H3: az LSTM-korrekció a párok 98%-ában kissé ront; a napok mediánban 64%-án nem is korrigál. |
| 14 | `14_H5_fohatasok` | V.6 | H5: mi számít? Lépcső 11,7% > transzformáció 5,1% > mérték 4,7% > újrabecslés, ablak, család. |
| 15 | `15_kritikus_idoszakok` | V.7 | Hőtérkép: a COVID alatt a makró-blokk 27%-ot javít; a háborúban a nyersanyagok rontanak; a legviharosabb napokon az L4 a legjobb. |
| 16 | `16_covid_es_haboru` | V.7 | A két válság közelről: a COVID elején a HAR_L4 gyorsabban reagál a HAR_L1-nél; a háborúban a kettő szinte együtt mozog. |
| 17 | `17_kumulalt_elony` | V.7–V.8 | Mikor keletkezik az előny? A külső információ előnye ugrásszerűen a COVID-ban és 2026-ban gyűlik; az LSTM-korrekció kumulált hatása végig ~0, a végén negatív. |
| 18 | `18_elorejelzes_vs_valosag` | V.8 | A legjobb modell előrejelzése a valóság függvényében (log-korreláció 0,89); a pontok a 45°-os vonal körül szóródnak. |
| 19 | `19_mcs` | V.8 | MCS: 178 modell marad. Az RV-modellek csak 17%-a jut be, a PK/GK-modellek 95–97%-a; a log 80%, a √ 59%. |
| 20 | `20_egyutthatok_idoben` | V.8 (robusztusság) | A HAR_L4 fő együtthatói heti újrabecslésenként. Az OVX súlya tartósan a legnagyobb; a napi tag a COVID után megnőtt. |
| 21 | `21_lstm_kiserletek` | VI. / melléklet | Kiegészítő kísérlet: a tiszta LSTM még a „holnap = ma” tippnél is rosszabb (2,65×); horgonnyal 1,06×. |
| 22 | `22_lstm_telitodes_covid` | VI. / melléklet | A telítődés: a tiszta LSTM 3–4%-on „megáll”, miközben a valóság 30% fölé ugrik; horgonnyal követi a sokkot. |

**Prezentációhoz (10–12 dia):** 1, 5, 7, 9, 10, 12, 13, 14, 15, 22 – ezek egyenként egy-egy állítást mondanak el.
