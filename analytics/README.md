# Module 2 — Analytics Pipeline

Run:

```bash
python pipeline.py
```

The script uses one initial `sns.load_dataset("titanic")` call only when `titanic.csv` does not already exist, then saves the raw dataset immediately as the required offline fallback.

Cleaning follows the assignment threshold rule. EDA includes missingness, IQR outliers, mean/median/mode, survival-rate breakdowns, the exact six-column correlation matrix, four interpreted charts, and an age/fare standardization check.

The modeling section uses a stratified split and training-only preprocessing through scikit-learn `Pipeline`/`ColumnTransformer`. It evaluates Logistic Regression, Decision Tree and Random Forest; compares baseline/class-weight/SMOTE imbalance handling; runs Random Forest `GridSearchCV` with OOB enabled; performs the fare regression side-task; and saves the complete fitted classifier pipeline with `joblib`.

### Important first run
Because the execution environment used to prepare this package did not have outbound internet access, `titanic.csv` could not be downloaded here. Run `python pipeline.py` once on a machine with internet access; it will create `analytics/titanic.csv` automatically. Commit that generated file before submitting.
