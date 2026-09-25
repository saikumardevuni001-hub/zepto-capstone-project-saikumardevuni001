"""
Zepto Capstone — Module 1: Data Pipeline
Scrape -> clean -> convert -> SQLite -> SQL/pandas verification.
"""
from pathlib import Path
import re
import sqlite3
import requests
import pandas as pd
from bs4 import BeautifulSoup

BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "data"
OUT_DIR = BASE / "outputs"
DB_PATH = DATA_DIR / "books.db"
DATA_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)

BASE_URL = "https://books.toscrape.com/"
GBP_TO_INR = 105.50
RATING_MAP = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}

def scrape_books(min_rows=60):
    session = requests.Session()
    rows = []
    page = 1
    while len(rows) < min_rows and page <= 5:
        url = BASE_URL if page == 1 else f"{BASE_URL}catalogue/page-{page}.html"
        response = session.get(url, timeout=20)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        for card in soup.select("article.product_pod"):
            title = card.h3.a.get("title", "").strip()
            price = card.select_one(".price_color").get_text(strip=True)
            rating_class = next(
                (c for c in card.select_one("p.star-rating").get("class", [])
                 if c in RATING_MAP), None
            )
            availability = card.select_one(".availability").get_text(" ", strip=True)
            category = "Unknown"
            detail_href = card.h3.a.get("href")
            if detail_href:
                detail_url = requests.compat.urljoin(url, detail_href)
                d = session.get(detail_url, timeout=20)
                d.raise_for_status()
                ds = BeautifulSoup(d.text, "html.parser")
                breadcrumb = ds.select("ul.breadcrumb li a")
                if len(breadcrumb) >= 3:
                    category = breadcrumb[-1].get_text(strip=True)
            rows.append({
                "title": title,
                "price": price,
                "star_rating": rating_class or "",
                "availability": availability,
                "category": category,
            })
        page += 1
    df = pd.DataFrame(rows)
    if len(df) < min_rows:
        raise RuntimeError(f"Only {len(df)} books were scraped; at least {min_rows} are required.")
    return df.drop_duplicates(subset=["title"]).reset_index(drop=True)

def clean_books(raw):
    df = raw.copy()
    # Numeric parse failure is represented as NaN and median-imputed.
    df["price_gbp"] = pd.to_numeric(
        df["price"].astype(str).str.replace(r"[^0-9.]", "", regex=True),
        errors="coerce"
    )
    df["rating"] = df["star_rating"].map(RATING_MAP)
    df["in_stock"] = df["availability"].str.contains("In stock", case=False, na=False)
    numeric_medians = {"price_gbp": df["price_gbp"].median(), "rating": df["rating"].median()}
    for col, med in numeric_medians.items():
        df[col] = df[col].fillna(med)
    # Category/title are essential identifiers; rows with missing values are dropped.
    df["category"] = df["category"].fillna("Unknown").astype(str).str.strip()
    df["title"] = df["title"].astype(str).str.strip()
    df["price_inr"] = df["price_gbp"] * GBP_TO_INR
    df["rating"] = df["rating"].round().astype(int).clip(1, 5)
    return df[["title","price_gbp","price_inr","rating","in_stock","category"]]

def create_database(df):
    if DB_PATH.exists():
        DB_PATH.unlink()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("PRAGMA foreign_keys = ON;")
    cur.executescript("""
    CREATE TABLE categories (
        category_id INTEGER PRIMARY KEY AUTOINCREMENT,
        category_name TEXT NOT NULL UNIQUE
    );
    CREATE TABLE books (
        book_id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        price_gbp REAL NOT NULL,
        price_inr REAL NOT NULL,
        rating INTEGER NOT NULL CHECK(rating BETWEEN 1 AND 5),
        in_stock INTEGER NOT NULL,
        category_id INTEGER NOT NULL,
        FOREIGN KEY(category_id) REFERENCES categories(category_id)
    );
    """)
    for category in sorted(df["category"].unique()):
        cur.execute("INSERT INTO categories(category_name) VALUES (?)", (category,))
    category_map = dict(cur.execute("SELECT category_name, category_id FROM categories"))
    records = [
        (r.title, float(r.price_gbp), float(r.price_inr), int(r.rating),
         int(bool(r.in_stock)), category_map[r.category])
        for r in df.itertuples(index=False)
    ]
    cur.executemany("""
        INSERT INTO books(title, price_gbp, price_inr, rating, in_stock, category_id)
        VALUES (?, ?, ?, ?, ?, ?)
    """, records)
    conn.commit()
    return conn

QUERIES = {
    "select_where": """
        SELECT title, price_inr, rating FROM books
        WHERE in_stock = 1 AND rating >= 4;
    """,
    "order_limit": """
        SELECT title, price_inr FROM books
        ORDER BY price_inr DESC LIMIT 10;
    """,
    "distinct": """
        SELECT DISTINCT c.category_name
        FROM books b JOIN categories c ON b.category_id = c.category_id
        ORDER BY c.category_name;
    """,
    "between": """
        SELECT title, price_gbp, rating FROM books
        WHERE price_gbp BETWEEN 10 AND 30
        ORDER BY price_gbp;
    """,
    "join": """
        SELECT c.category_name, b.title, b.rating, b.price_inr
        FROM books b
        JOIN categories c ON b.category_id = c.category_id
        ORDER BY b.rating DESC, b.price_inr DESC
        LIMIT 10;
    """,
}

def run_queries(conn, df):
    output_parts = []
    for name, query in QUERIES.items():
        result = pd.read_sql_query(query, conn)
        output_parts.append(f"\n### {name}\n{query.strip()}\n\n{result.to_string(index=False)}\n")
    # Required pandas read_sql for at least two results.
    q1_pd = pd.read_sql(QUERIES["select_where"], conn)
    q2_pd = pd.read_sql(QUERIES["order_limit"], conn)

    # Required in-memory merge equivalent to the SQL join.
    books_mem = df.copy()
    books_mem["category_id"] = books_mem["category"].astype("category").cat.codes + 1
    categories_mem = (
        books_mem[["category_id", "category"]].drop_duplicates()
        .rename(columns={"category": "category_name"})
    )
    merge_result = (
        books_mem.merge(categories_mem, on="category_id", how="inner")
        [["category_name","title","rating","price_inr"]]
        .sort_values(["rating","price_inr"], ascending=[False,False])
        .head(10)
        .reset_index(drop=True)
    )
    sql_join = pd.read_sql(QUERIES["join"], conn).reset_index(drop=True)

    output_parts.append(
        "\n### pandas pd.read_sql result 1\n" + q1_pd.to_string(index=False) +
        "\n\n### pandas pd.read_sql result 2\n" + q2_pd.to_string(index=False) +
        "\n\n### SQL JOIN vs pd.merge\nSQL JOIN:\n" + sql_join.to_string(index=False) +
        "\n\npd.merge:\n" + merge_result.to_string(index=False) +
        f"\n\nEquivalent: {sql_join.equals(merge_result)}\n"
    )
    (OUT_DIR / "sql_outputs.txt").write_text("\n".join(output_parts), encoding="utf-8")

def main():
    raw = scrape_books()
    raw.to_csv(DATA_DIR / "books_raw.csv", index=False)
    clean = clean_books(raw)
    clean.to_csv(DATA_DIR / "books_clean.csv", index=False)
    conn = create_database(clean)
    run_queries(conn, clean)
    conn.close()
    print(f"Completed. Rows: {len(clean)}")
    print(f"SQLite: {DB_PATH}")
    print(f"Outputs: {OUT_DIR / 'sql_outputs.txt'}")

if __name__ == "__main__":
    main()
