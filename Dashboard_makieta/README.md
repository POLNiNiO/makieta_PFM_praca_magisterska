# Dashboard makieta PFM

Makieta startuje z pustym profilem i zapisuje wyłącznie dane wpisane lokalnie przez użytkownika w:

```text
Dashboard_makieta/data/user_profile.json
```

Dashboard używa istniejącego modelu z `models/pfm_insolvency_1m/model.joblib` oraz modułów `pfm_ml`. Kod algorytmu nie jest modyfikowany.

## Uruchomienie

```bash
.venv/bin/python Dashboard_makieta/server.py --port 8015
```

Adres lokalny:

```text
http://127.0.0.1:8015
```

## Co można dodać

- przychody i wydatki z własnymi kategoriami,
- nowe kategorie wydatków, przychodów albo wspólne,
- aktywa finansowe,
- zobowiązania, raty i oprocentowanie.

Po dodaniu transakcji dashboard przelicza profil modelem ryzyka niewypłacalności i pokazuje Friendly Tips. Tipy mają charakter edukacyjny i informacyjny, nie są poradą inwestycyjną.
