# Algorytm PFM oparty o uczenie maszynowe

Ten projekt trenuje model ryzyka utraty plynnosci finansowej na danych PersonaLedger i zamienia wynik modelu na rekomendacje PFM dla uzytkownika dashboardu.

## Co robi algorytm

1. Czyta transakcje z plikow parquet bez modyfikowania danych zrodlowych.
2. Agreguje transakcje do profilu uzytkownika: przeplywy, wydatki wedlug kategorii, udzial wydatkow uznaniowych, czestotliwosc transakcji, platnosci online, wydatki weekendowe i podobne cechy.
3. Trenuje klasyfikator ryzyka niewyplacalnosci na etykietach z `labels.json`.
4. Wyznacza prog decyzyjny pod ostrzeganie o ryzyku, z naciskiem na recall/F2.
5. Generuje rekomendacje, np. ograniczenie konkretnej kategorii, budowa poduszki finansowej, lokata/obligacje dla stabilnej nadwyzki albo ostrzezenie o bliskiej utracie plynnosci.

## Instalacja

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## Trening modelu

```bash
.venv/bin/python -m pfm_ml.train \
  --data-dir PersonaLedger \
  --task insolvency_prediction_1months \
  --output-dir models/pfm_insolvency_1m \
  --progress
```

Wyniki treningu zostana zapisane w:

- `models/pfm_insolvency_1m/model.joblib` - model, konfiguracja cech i prog decyzyjny
- `models/pfm_insolvency_1m/metrics.json` - metryki jak ROC-AUC, average precision, precision/recall
- `models/pfm_insolvency_1m/test_scores.csv` - scoring uzytkownikow testowych
- `models/pfm_insolvency_1m/sample_recommendations.json` - przykladowe rekomendacje

## Predykcja i rekomendacje

Dla jednego uzytkownika:

```bash
.venv/bin/python -m pfm_ml.predict \
  --model-dir models/pfm_insolvency_1m \
  --transactions PersonaLedger/insolvency_prediction_1months/test.parquet \
  --user-id 0
```

Dla listy najbardziej ryzykownych uzytkownikow:

```bash
.venv/bin/python -m pfm_ml.predict \
  --model-dir models/pfm_insolvency_1m \
  --transactions PersonaLedger/insolvency_prediction_1months/test.parquet \
  --top-n 10
```

## Integracja z dashboardem

Dashboard moze wywolywac `pfm_ml.predict.score_transactions(...)` dla transakcji uzytkownika, a potem wyswietlac pola:

- `risk_probability` - prawdopodobienstwo utraty plynnosci
- `risk_label` - decyzja wedlug progu modelu
- `recommendations` - lista rekomendacji z priorytetem, tytulem, uzasadnieniem i sugerowana kwota

To jest pierwsza wersja algorytmu badawczego. W pracy magisterskiej warto opisac, ze model predykcyjny uczy sie z danych transakcyjnych, a rekomendacje sa warstwa hybrydowa: wynik ML plus reguly finansow osobistych.
