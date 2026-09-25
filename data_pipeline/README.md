# Module 1 — Data Pipeline

Run:

```bash
python pipeline.py
```

The pipeline uses `requests` + `BeautifulSoup` against `books.toscrape.com`, scraping five listing pages until at least 60 books are collected. Price is converted using the required fixed baseline **1 GBP = 105.50 INR**.

Numeric parsing failures become `NaN` and are median-imputed. The normalized SQLite design uses `categories(category_id, category_name)` and `books(..., category_id)` with a foreign key.

`outputs/sql_outputs.txt` records the required SQL query strings and outputs, plus `pd.read_sql` and `pd.merge` verification.
