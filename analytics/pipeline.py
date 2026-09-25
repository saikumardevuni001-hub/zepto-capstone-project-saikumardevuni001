"""
Zepto Capstone — Module 2: Analytics Pipeline.
One cohesive Titanic workflow: load once -> save fallback -> clean -> EDA -> modeling -> regression.
"""

from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from imblearn.over_sampling import SMOTE


BASE = Path(__file__).resolve().parent
OUT = BASE / "outputs"
ART = BASE / "artifacts"

OUT.mkdir(exist_ok=True)
ART.mkdir(exist_ok=True)

CSV_PATH = BASE / "titanic.csv"


def load_once():
    if CSV_PATH.exists():
        return pd.read_csv(CSV_PATH), "offline_csv"

    # The assignment explicitly requires this single network/cache load.
    df = sns.load_dataset("titanic")
    df.to_csv(CSV_PATH, index=False)

    return df, "seaborn_load"


def profile(df):
    info_text = []

    df.info(buf=_Buffer(info_text))
    info = "\n".join(info_text)

    desc = df.describe(include="all").to_string()
    shape = str(df.shape)

    missing = (
        df.isna()
        .mean()
        .mul(100)
        .loc[lambda s: s > 0]
        .sort_values(ascending=False)
    )

    (OUT / "profile.txt").write_text(
        "df.info()\n"
        + info
        + "\n\ndf.describe()\n"
        + desc
        + f"\n\ndf.shape = {shape}\n\nMissing percentages:\n"
        + missing.to_string(),
        encoding="utf-8",
    )

    return missing


class _Buffer:
    def __init__(self, lines):
        self.lines = lines

    def write(self, x):
        self.lines.append(x)

    def flush(self):
        pass


def clean_for_eda(df, missing):
    work = df.copy()
    decisions = []

    for col, pct in missing.items():

        if pct < 5:
            work = work.dropna(subset=[col])

            decisions.append(
                f"{col}: {pct:.2f}% missing -> dropped affected rows (<5%)."
            )

        elif pct <= 30:

            if pd.api.types.is_numeric_dtype(work[col]):
                work[col] = work[col].fillna(work[col].median())

                decisions.append(
                    f"{col}: {pct:.2f}% missing -> median imputation (5%-30%)."
                )

            else:
                work[col] = work[col].fillna(work[col].mode().iloc[0])

                decisions.append(
                    f"{col}: {pct:.2f}% missing -> mode imputation (5%-30%)."
                )

        else:

            # Handle pandas Categorical columns correctly.
            # A new category must be added before fillna("Missing").
            if hasattr(work[col].dtype, "categories"):
                work[col] = (
                    work[col]
                    .cat.add_categories(["Missing"])
                    .fillna("Missing")
                )
            else:
                work[col] = work[col].fillna("Missing")

            # For the high-missing deck column, encode missing as its own category.
            decisions.append(
                f"{col}: {pct:.2f}% missing -> encoded as 'Missing' category (>30%)."
            )

    # `deck` is retained as a Missing category for EDA
    # but excluded from model features.
    (OUT / "missing_value_decisions.md").write_text(
        "# Missing-value decisions\n\n"
        + "\n".join("- " + x for x in decisions),
        encoding="utf-8",
    )

    return work


def iqr_count(s):
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1

    return int(
        ((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum()
    )


def eda(work):
    stats = []

    for col in ["age", "fare"]:

        fig, ax = plt.subplots(figsize=(7, 4))

        sns.histplot(
            work[col],
            kde=True,
            ax=ax
        )

        ax.set_title(f"{col.title()} histogram")

        fig.tight_layout()
        fig.savefig(OUT / f"{col}_hist.png")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 3))

        sns.boxplot(
            x=work[col],
            ax=ax
        )

        ax.set_title(f"{col.title()} box plot")

        fig.tight_layout()
        fig.savefig(OUT / f"{col}_box.png")
        plt.close(fig)

    fare = work["fare"]

    mode = fare.mode().iloc[0]
    mean = fare.mean()
    median = fare.median()

    skew = (
        "right-skewed"
        if mean > median and median >= mode
        else (
            "left-skewed"
            if mean < median and median <= mode
            else "approximately symmetric"
        )
    )

    by_sex = work.groupby("sex")["survived"].mean().mul(100)
    by_class = work.groupby("pclass")["survived"].mean().mul(100)
    by_both = (
        work.groupby(["sex", "pclass"])["survived"]
        .mean()
        .mul(100)
    )

    corr_cols = [
        "survived",
        "pclass",
        "age",
        "sibsp",
        "parch",
        "fare",
    ]

    corr = work[corr_cols].corr()

    pairs = []

    for i, a in enumerate(corr_cols):

        for b in corr_cols[i + 1:]:

            pairs.append(
                (
                    abs(corr.loc[a, b]),
                    a,
                    b,
                    corr.loc[a, b],
                )
            )

    top2 = sorted(pairs, reverse=True)[:2]

    plt.figure(figsize=(7, 6))

    sns.heatmap(
        corr,
        annot=True,
        cmap="coolwarm",
        center=0,
    )

    plt.title("Titanic numeric correlation matrix")

    plt.tight_layout()
    plt.savefig(OUT / "correlation_heatmap.png")
    plt.close()

    charts = []

    fig, ax = plt.subplots(figsize=(7, 4))

    sns.barplot(
        data=work,
        x="sex",
        y="survived",
        hue="pclass",
        errorbar=None,
        ax=ax,
    )

    ax.set_ylabel("Survival rate")
    ax.set_title("Survival rate by sex and class")

    fig.tight_layout()
    fig.savefig(OUT / "survival_sex_class.png")
    plt.close(fig)

    charts.append(
        "Women show higher survival proportions than men, while first-class "
        "groups generally show higher survival than lower classes. The grouped "
        "view makes the interaction between sex and passenger class visible "
        "rather than treating either variable in isolation."
    )

    fig, ax = plt.subplots(figsize=(7, 4))

    sns.boxplot(
        data=work,
        x="survived",
        y="fare",
        ax=ax,
    )

    ax.set_title("Fare distribution by survival outcome")

    fig.tight_layout()
    fig.savefig(OUT / "fare_survival_box.png")
    plt.close(fig)

    charts.append(
        "Survivors have a higher fare distribution than non-survivors in this "
        "dataset. This is consistent with fare acting partly as a proxy for "
        "socioeconomic status and passenger class."
    )

    fig, ax = plt.subplots(figsize=(7, 4))

    sns.scatterplot(
        data=work,
        x="age",
        y="fare",
        hue="survived",
        alpha=0.65,
        ax=ax,
    )

    ax.set_title("Age, fare and survival")

    fig.tight_layout()
    fig.savefig(OUT / "age_fare_survival.png")
    plt.close(fig)

    charts.append(
        "The scatter shows survival observations distributed across age and "
        "fare rather than being determined by one variable alone. Higher-fare "
        "observations are more concentrated in groups with higher survival "
        "proportions."
    )

    fig, ax = plt.subplots(figsize=(7, 4))

    sns.countplot(
        data=work,
        x="pclass",
        hue="survived",
        ax=ax,
    )

    ax.set_title("Passenger class and survival counts")

    fig.tight_layout()
    fig.savefig(OUT / "class_survival_counts.png")
    plt.close(fig)

    charts.append(
        "Passenger counts and survival counts vary substantially by class. "
        "The plot complements the rate-based analysis by showing the "
        "underlying group sizes."
    )

    # Standardization sanity check.
    before = work[["age", "fare"]].agg(["mean", "std"])

    z = work[["age", "fare"]].copy()
    z = (z - z.mean()) / z.std()

    after = z.agg(["mean", "std"])

    pd.concat(
        {
            "before": before,
            "after": after,
        },
        axis=1,
    ).to_csv(
        OUT / "standardization_check.csv"
    )

    report = f"""# EDA results

## IQR outliers
- age: {iqr_count(work['age'])}
- fare: {iqr_count(work['fare'])}

## Fare distribution
- mean: {mean:.4f}
- median: {median:.4f}
- mode: {mode:.4f}
- conclusion: {skew}

## Survival rate by sex
{by_sex.to_string()}

## Survival rate by pclass
{by_class.to_string()}

## Survival rate by sex and pclass
{by_both.to_string()}

## Two strongest absolute off-diagonal correlations
1. {top2[0][1]} vs {top2[0][2]}: {top2[0][3]:.4f}
2. {top2[1][1]} vs {top2[1][2]}: {top2[1][3]:.4f}

## Chart interpretations
""" + "\n\n".join(
        f"### Chart {i + 1}\n{x}"
        for i, x in enumerate(charts)
    )

    (OUT / "eda_report.md").write_text(
        report,
        encoding="utf-8",
    )


def make_preprocessor():
    numeric = [
        "pclass",
        "age",
        "sibsp",
        "parch",
        "fare",
    ]

    categorical = [
        "sex",
        "embarked",
    ]

    num = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    cat = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )

    return ColumnTransformer(
        [
            ("num", num, numeric),
            ("cat", cat, categorical),
        ]
    )


def classifier_metrics(model, X_test, y_test, name):
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    return {
        "model": name,
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(
            y_test,
            pred,
            zero_division=0,
        ),
        "recall": recall_score(
            y_test,
            pred,
            zero_division=0,
        ),
        "f1": f1_score(
            y_test,
            pred,
            zero_division=0,
        ),
        "auc": roc_auc_score(
            y_test,
            proba,
        ),
        "confusion_matrix": confusion_matrix(
            y_test,
            pred,
        ).tolist(),
    }


def modeling(work):

    model_cols = [
        "survived",
        "pclass",
        "age",
        "sibsp",
        "parch",
        "fare",
        "sex",
        "embarked",
    ]

    model_df = work[model_cols].copy()

    X = model_df.drop(
        columns="survived"
    )

    y = model_df["survived"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    pre = make_preprocessor()

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5,
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            oob_score=True,
        ),
    }

    fitted = {}
    rows = []

    for name, est in models.items():

        pipe = Pipeline(
            [
                (
                    "preprocess",
                    make_preprocessor(),
                ),
                (
                    "model",
                    est,
                ),
            ]
        )

        pipe.fit(
            X_train,
            y_train,
        )

        fitted[name] = pipe

        rows.append(
            classifier_metrics(
                pipe,
                X_test,
                y_test,
                name,
            )
        )

        if name == "Decision Tree":

            feature_names = (
                pipe
                .named_steps["preprocess"]
                .get_feature_names_out()
            )

            fig, ax = plt.subplots(
                figsize=(18, 10)
            )

            plot_tree(
                pipe.named_steps["model"],
                feature_names=feature_names,
                class_names=[
                    "not survived",
                    "survived",
                ],
                filled=True,
                ax=ax,
                max_depth=3,
            )

            fig.tight_layout()

            fig.savefig(
                OUT / "decision_tree.png"
            )

            plt.close(fig)

    # Comparison table
    metrics_df = pd.DataFrame(
        [
            {
                k: v
                for k, v in r.items()
                if k != "confusion_matrix"
            }
            for r in rows
        ]
    )

    metrics_df.to_csv(
        OUT / "classifier_metrics.csv",
        index=False,
    )

    # Confusion matrices and ROC curves
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 4),
    )

    for ax, (name, pipe) in zip(
        axes,
        fitted.items(),
    ):

        pred = pipe.predict(X_test)

        sns.heatmap(
            confusion_matrix(
                y_test,
                pred,
            ),
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax,
        )

        ax.set_title(name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    fig.tight_layout()

    fig.savefig(
        OUT / "confusion_matrices.png"
    )

    plt.close(fig)

    fig, ax = plt.subplots(
        figsize=(7, 5)
    )

    for name, pipe in fitted.items():

        prob = pipe.predict_proba(
            X_test
        )[:, 1]

        fpr, tpr, _ = roc_curve(
            y_test,
            prob,
        )

        ax.plot(
            fpr,
            tpr,
            label=f"{name} AUC={roc_auc_score(y_test, prob):.3f}",
        )

    ax.plot(
        [0, 1],
        [0, 1],
        "--",
    )

    ax.legend()

    ax.set_xlabel(
        "False positive rate"
    )

    ax.set_ylabel(
        "True positive rate"
    )

    ax.set_title(
        "ROC curves"
    )

    fig.tight_layout()

    fig.savefig(
        OUT / "roc_curves.png"
    )

    plt.close(fig)

    # Imbalance comparison with Logistic Regression.
    imbalance_rows = []

    baseline = fitted[
        "Logistic Regression"
    ]

    imbalance_rows.append(
        (
            "baseline",
            classifier_metrics(
                baseline,
                X_test,
                y_test,
                "baseline",
            ),
        )
    )

    balanced = Pipeline(
        [
            (
                "preprocess",
                make_preprocessor(),
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    balanced.fit(
        X_train,
        y_train,
    )

    imbalance_rows.append(
        (
            "class_weight_balanced",
            classifier_metrics(
                balanced,
                X_test,
                y_test,
                "balanced",
            ),
        )
    )

    # SMOTE must occur only after training preprocessing.
    pre_smote = make_preprocessor()

    Xt = pre_smote.fit_transform(
        X_train,
        y_train,
    )

    Xv = pre_smote.transform(
        X_test
    )

    sm = SMOTE(
        random_state=42
    )

    Xt_sm, yt_sm = sm.fit_resample(
        Xt,
        y_train,
    )

    sm_model = LogisticRegression(
        max_iter=2000
    )

    sm_model.fit(
        Xt_sm,
        yt_sm,
    )

    pred = sm_model.predict(
        Xv
    )

    prob = sm_model.predict_proba(
        Xv
    )[:, 1]

    sm_metrics = {
        "model": "SMOTE",
        "precision": precision_score(
            y_test,
            pred,
            zero_division=0,
        ),
        "recall": recall_score(
            y_test,
            pred,
            zero_division=0,
        ),
        "f1": f1_score(
            y_test,
            pred,
            zero_division=0,
        ),
    }

    imbalance_df = pd.DataFrame(
        [
            {
                "strategy": s,
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
            }
            for s, m in imbalance_rows
        ]
        + [
            {
                "strategy": "SMOTE",
                **{
                    k: sm_metrics[k]
                    for k in [
                        "precision",
                        "recall",
                        "f1",
                    ]
                },
            }
        ]
    )

    imbalance_df.to_csv(
        OUT / "imbalance_comparison.csv",
        index=False,
    )

    # RF grid search with OOB enabled.
    rf_pipe = Pipeline(
        [
            (
                "preprocess",
                make_preprocessor(),
            ),
            (
                "model",
                RandomForestClassifier(
                    random_state=42,
                    oob_score=True,
                ),
            ),
        ]
    )

    grid = GridSearchCV(
        rf_pipe,
        {
            "model__n_estimators": [
                100,
                200,
            ],
            "model__max_depth": [
                None,
                5,
                10,
            ],
            "model__max_features": [
                "sqrt",
                "log2",
            ],
        },
        cv=5,
        scoring="f1",
        n_jobs=-1,
    )

    grid.fit(
        X_train,
        y_train,
    )

    best_rf = grid.best_estimator_

    oob = best_rf.named_steps[
        "model"
    ].oob_score_

    (
        OUT / "gridsearch.txt"
    ).write_text(
        f"Best parameters: {grid.best_params_}\n"
        f"CV best F1: {grid.best_score_:.6f}\n"
        f"OOB score: {oob:.6f}",
        encoding="utf-8",
    )

    # Regression: predict fare from all other available features.
    reg_cols = [
        "pclass",
        "age",
        "sibsp",
        "parch",
        "sex",
        "embarked",
        "survived",
    ]

    reg_df = work[
        reg_cols + ["fare"]
    ].copy()

    Xr = reg_df.drop(
        columns="fare"
    )

    yr = reg_df["fare"]

    Xtr, Xte, ytr, yte = train_test_split(
        Xr,
        yr,
        test_size=0.2,
        random_state=42,
    )

    reg = Pipeline(
        [
            (
                "preprocess",
                make_preprocessor_reg(),
            ),
            (
                "model",
                LinearRegression(),
            ),
        ]
    )

    reg.fit(
        Xtr,
        ytr,
    )

    yp = reg.predict(
        Xte
    )

    mae = mean_absolute_error(
        yte,
        yp,
    )

    rmse = np.sqrt(
        mean_squared_error(
            yte,
            yp,
        )
    )

    r2 = r2_score(
        yte,
        yp,
    )

    n = len(yte)

    p = (
        reg
        .named_steps["preprocess"]
        .transform(Xte)
        .shape[1]
    )

    adj = (
        1 - (1 - r2) * (n - 1) / (n - p - 1)
        if n - p - 1 > 0
        else np.nan
    )

    fig, ax = plt.subplots(
        figsize=(7, 4)
    )

    residual = yte - yp

    ax.scatter(
        yp,
        residual,
        alpha=0.65,
    )

    ax.axhline(
        0,
        ls="--",
    )

    ax.set_xlabel(
        "Predicted fare"
    )

    ax.set_ylabel(
        "Residual"
    )

    ax.set_title(
        "Fare regression residuals"
    )

    fig.tight_layout()

    fig.savefig(
        OUT / "regression_residuals.png"
    )

    plt.close(fig)

    reg_report = (
        f"MAE: {mae:.6f}\n"
        f"RMSE: {rmse:.6f}\n"
        f"R2: {r2:.6f}\n"
        f"Adjusted R2: {adj:.6f}\n"
        "Heteroscedasticity: inspect the residual plot; "
        "a funnel/non-random spread indicates heteroscedasticity."
    )

    (
        OUT / "regression_metrics.txt"
    ).write_text(
        reg_report,
        encoding="utf-8",
    )

    # Pick a classifier by F1 for the saved complete pipeline.
    best_name = max(
        fitted,
        key=lambda k: f1_score(
            y_test,
            fitted[k].predict(X_test),
            zero_division=0,
        ),
    )

    full_pipeline = fitted[
        best_name
    ]

    joblib.dump(
        full_pipeline,
        ART / "best_classifier_pipeline.joblib",
    )

    reloaded = joblib.load(
        ART / "best_classifier_pipeline.joblib"
    )

    reload_pred = reloaded.predict(
        X_test.iloc[:5]
    )

    (
        OUT / "artifact_reload.txt"
    ).write_text(
        f"Selected pipeline: {best_name}\n"
        f"Raw-input prediction after joblib reload: "
        f"{reload_pred.tolist()}",
        encoding="utf-8",
    )

    # Separate metric groups: classification vs regression.
    summary = metrics_df.copy()

    summary["reg_MAE"] = mae
    summary["reg_RMSE"] = rmse
    summary["reg_R2"] = r2
    summary["reg_Adjusted_R2"] = adj

    summary.to_csv(
        OUT / "final_model_comparison.csv",
        index=False,
    )

    final_text = (
        f"Classifier selected for deployment: {best_name}, "
        "based on highest test-set F1 among the three evaluated classifiers. "
        "See classifier_metrics.csv for accuracy, precision, recall, F1 and AUC. "
        "The regression task is reported separately because MAE/RMSE/R2/"
        "Adjusted R2 are regression metrics and should not be compared "
        "numerically with classification metrics. "
        "The saved artifact is a complete preprocessing+estimator pipeline, "
        "so raw rows can be passed directly to predict()."
    )

    (
        OUT / "final_recommendation.md"
    ).write_text(
        final_text,
        encoding="utf-8",
    )


def make_preprocessor_reg():

    numeric = [
        "pclass",
        "age",
        "sibsp",
        "parch",
        "survived",
    ]

    categorical = [
        "sex",
        "embarked",
    ]

    return ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="median"
                            ),
                        ),
                        (
                            "scaler",
                            StandardScaler(),
                        ),
                    ]
                ),
                numeric,
            ),
            (
                "cat",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="most_frequent"
                            ),
                        ),
                        (
                            "encoder",
                            OneHotEncoder(
                                handle_unknown="ignore"
                            ),
                        ),
                    ]
                ),
                categorical,
            ),
        ]
    )


def main():

    df, source = load_once()

    missing = profile(df)

    work = clean_for_eda(
        df,
        missing,
    )

    eda(work)

    modeling(work)

    print(
        f"Dataset source: {source}; "
        f"cleaned shape: {work.shape}"
    )

    print(
        f"Outputs: {OUT}"
    )


if __name__ == "__main__":
    main()