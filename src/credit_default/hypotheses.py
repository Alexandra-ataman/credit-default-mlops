import argparse
import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from credit_default.data import BILLS, PAY, PAYMENTS, TARGET
from credit_default.features import NEW_NUMERIC, add_features
from credit_default.split import profile_groups


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["logreg", "boosting"], default="logreg")
    args = parser.parse_args()
    if args.model == "logreg":
        estimator = LogisticRegression(
            solver="liblinear", max_iter=1000, random_state=42
        )
        prefix = "hypothesis_cv"
    else:
        estimator = HistGradientBoostingClassifier(
            max_iter=150,
            max_leaf_nodes=15,
            learning_rate=0.08,
            min_samples_leaf=30,
            l2_regularization=1,
            early_stopping=False,
            random_state=42,
        )
        prefix = "hypothesis_boosting_cv"
    # Test не загружаем. Все варианты сравниваем на одних блоках train.
    train = pd.read_csv("data/processed/train.csv")
    df = add_features(train)
    demo = ["SEX", "EDUCATION", "MARRIAGE", "AGE", "AGE_BAND"]
    behavior = ["LIMIT_BAL"] + PAY + BILLS + PAYMENTS
    variants = {
        "Демография": demo,
        "Последний статус": ["PAY_0"],
        "Шесть статусов": PAY,
        "Поведение": behavior,
        "Поведение + демография": behavior + demo,
        "Поведение + новые признаки": behavior + NEW_NUMERIC,
        "Все признаки": behavior + demo + NEW_NUMERIC,
    }
    if args.model == "boosting":
        variants = {
            name: variants[name]
            for name in ["Поведение", "Поведение + новые признаки", "Все признаки"]
        }
    categorical = ["SEX", "EDUCATION", "MARRIAGE", "AGE_BAND"] + PAY
    folds = list(
        StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42).split(
            train, train[TARGET], profile_groups(train)
        )
    )
    results = []
    warnings.filterwarnings("error", category=ConvergenceWarning)
    for name, columns in variants.items():
        cat = [col for col in columns if col in categorical]
        num = [col for col in columns if col not in categorical]
        transforms = []
        if num:
            transforms.append(
                (
                    "num",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scaler", StandardScaler()),
                        ]
                    ),
                    num,
                )
            )
        if cat:
            transforms.append(
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    cat,
                )
            )
        pipeline = Pipeline(
            [
                ("prepare", ColumnTransformer(transforms)),
                ("model", estimator),
            ]
        )
        for fold, (fit, valid) in enumerate(folds, start=1):
            pipeline.fit(df.iloc[fit][columns], df.iloc[fit][TARGET])
            probability = pipeline.predict_proba(df.iloc[valid][columns])[:, 1]
            y = df.iloc[valid][TARGET]
            results.append(
                {
                    "Вариант": name,
                    "Блок": fold,
                    "ROC-AUC": roc_auc_score(y, probability),
                    "AP": average_precision_score(y, probability),
                    "Brier": brier_score_loss(y, probability),
                    "Доля дефолта": y.mean(),
                }
            )
        print(name, flush=True)
    report = pd.DataFrame(results)
    report.to_csv(f"reports/{prefix}_folds.csv", index=False)
    report.groupby("Вариант", sort=False)[["ROC-AUC", "AP", "Brier"]].agg(
        ["mean", "std"]
    ).to_csv(f"reports/{prefix}_summary.csv")
    metadata = {
        "source": "train.csv only",
        "seed": 42,
        "folds": 5,
        "grouping": "identical original features, excluding ID and target",
        "model": type(estimator).__name__,
        "parameters": estimator.get_params(),
        "sklearn_version": sklearn.__version__,
        "train_sha256": hashlib.sha256(
            Path("data/processed/train.csv").read_bytes()
        ).hexdigest(),
        "variants": variants,
    }
    Path(f"reports/{prefix}_settings.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"
    )
    assert np.isfinite(report[["ROC-AUC", "AP", "Brier"]].to_numpy()).all()


if __name__ == "__main__":
    main()
