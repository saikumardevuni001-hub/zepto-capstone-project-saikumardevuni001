# Zepto Data & AI Platform — Capstone Project

This repository implements the three required modules for the Masai AI/ML Capstone:

- `data_pipeline/` — web scraping, cleaning, currency conversion, SQLite normalization and SQL/pandas analysis.
- `analytics/` — Titanic profiling, EDA, predictive modeling, imbalance comparison, tuning, regression and model persistence.
- `support_assistant/` — local RAG-style support assistant using Sentence Transformers, ChromaDB, LangGraph, Pydantic and FastAPI.

## Repository structure

```text
zepto_capstone_project/
├── README.md
├── requirements.txt
├── data_pipeline/
│   ├── pipeline.py
│   ├── README.md
│   ├── data/
│   └── outputs/
├── analytics/
│   ├── pipeline.py
│   ├── README.md
│   ├── titanic.csv
│   ├── artifacts/
│   └── outputs/
└── support_assistant/
    ├── main.py
    ├── ingest.py
    ├── rag.py
    ├── prompt.py
    ├── README.md
    ├── Dockerfile
    ├── requirements.txt
    ├── docs/
    └── chroma_db/
```

## Setup

One consolidated `requirements.txt` is provided at the repository root.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

## Run Module 1

```bash
python data_pipeline/pipeline.py
```

The script scrapes at least 5 catalogue pages (targeting 60+ books), cleans the fields, converts GBP to INR at the required fixed rate **1 GBP = 105.50 INR**, builds a normalized SQLite database and writes SQL query outputs.

## Run Module 2

```bash
python analytics/pipeline.py
```

The script loads `sns.load_dataset("titanic")` once when `analytics/titanic.csv` is not present, immediately saves the raw DataFrame to `titanic.csv`, then uses that same saved dataset for the complete EDA/modeling workflow. If the CSV already exists, it uses the offline fallback directly.

Important: the assignment requires the initial Seaborn load to happen once and the committed CSV to be the offline fallback. Run this module once on a machine with internet access if `titanic.csv` has not yet been generated.

Outputs include missing-value analysis, charts, survival breakdowns, correlation heatmap, standardized-column check, classifier metrics, imbalance comparison, GridSearchCV/OOB results, regression metrics, and a reloadable complete pipeline artifact.

## Run Module 3

From the repository root:

```bash
python support_assistant/ingest.py
uvicorn support_assistant.main:app --host 127.0.0.1 --port 7860
```

Then POST:

```json
{"query":"How long does delivery take?"}
```

The graded baseline leaves `MOCK_LLM` unset (equivalent to `MOCK_LLM=1`) and therefore uses deterministic local logic with no LLM API call.

## Docker

```bash
docker build -t zepto-support ./support_assistant
docker run --rm -p 7860:7860 zepto-support
```

## Git workflow required by the brief

The repository should contain at least one feature branch with two commits and a merge back into `main`. Example:

```bash
git checkout -b feature/capstone
git add .
git commit -m "Add capstone modules"
git add .
git commit -m "Complete analysis and support assistant"
git checkout main
git merge --no-ff feature/capstone -m "Merge capstone feature"
git log --graph --all
```

## Design decisions

### Data pipeline
A two-table normalized SQLite schema separates category attributes from book records. The required project-defined fixed currency rate is used rather than a live exchange-rate API so the graded result is deterministic.

### Analytics
Cleaning follows the specified missingness thresholds. Classification uses a stratified split followed by a training-only `ColumnTransformer` and `Pipeline`, preventing preprocessing leakage. Regression is treated separately because its metrics are not directly comparable with classification metrics.

### Support assistant
The policy documents are embedded locally with `all-MiniLM-L6-v2` and stored in ChromaDB. LangGraph routes policy questions to retrieval and general questions to a direct response. The `MOCK_LLM` switch controls generation only; retrieval always remains local.

## Final Validation

The project structure and module documentation have been reviewed for capstone submission.

## Submission Checklist

The repository contains the data pipeline, analytics, and support assistant modules required for submission.
