import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

from credit_default.data import BILLS, PAY, PAYMENTS, TARGET
from credit_default.features import (
    MONTHLY_NUMERIC,
    NEW_NUMERIC,
    SEQUENCE_NUMERIC,
    add_history_features,
)
from credit_default.split import profile_groups

RAW = ["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE"] + PAY + BILLS + PAYMENTS
CATEGORICAL = ["SEX", "EDUCATION", "MARRIAGE", "AGE_BAND"] + PAY
FEATURE_SETS = {
    "raw": RAW,
    "base": RAW + ["AGE_BAND"] + NEW_NUMERIC,
    "sequence": RAW + ["AGE_BAND"] + NEW_NUMERIC + SEQUENCE_NUMERIC,
    "monthly": RAW + ["AGE_BAND"] + NEW_NUMERIC + MONTHLY_NUMERIC,
}
# Небольшой фиксированный список: отдельно меняем модель и набор признаков.
VARIANTS = {
    "hgb_base": ("hgb", "base", {}),
    "hgb_sequence": ("hgb", "sequence", {}),
    "hgb_monthly": ("hgb", "monthly", {}),
    "cat_raw": ("cat", "raw", {}),
    "cat_base": ("cat", "base", {}),
    "cat_sequence": ("cat", "sequence", {}),
    "cat_monthly": ("cat", "monthly", {}),
    "cat_shallow": ("cat", "base", {"depth": 4, "l2_leaf_reg": 10}),
    "xgb_base": ("xgb", "base", {}),
    "xgb_sequence": ("xgb", "sequence", {}),
    "xgb_monthly": ("xgb", "monthly", {}),
}


def native_categories(df):
    result = df.copy()
    for col in result.columns.intersection(CATEGORICAL):
        result[col] = result[col].astype(str)
    return result


def build_pipeline(model, feature_set, seed=42, parameters=None):
    columns = FEATURE_SETS[feature_set]
    cat = [c for c in columns if c in CATEGORICAL]
    num = [c for c in columns if c not in CATEGORICAL]
    if model == "cat":
        from catboost import CatBoostClassifier

        estimator = CatBoostClassifier(
            iterations=600,
            depth=6,
            learning_rate=0.04,
            l2_leaf_reg=5,
            loss_function="Logloss",
            random_seed=seed,
            thread_count=2,
            verbose=False,
            allow_writing_files=False,
            cat_features=cat,
        )
        prepare = FunctionTransformer(native_categories)
    else:
        prepare = ColumnTransformer(
            [
                ("num", SimpleImputer(strategy="median"), num),
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    cat,
                ),
            ]
        )
        if model == "hgb":
            estimator = HistGradientBoostingClassifier(
                max_iter=150,
                max_leaf_nodes=15,
                learning_rate=0.08,
                min_samples_leaf=30,
                l2_regularization=1,
                early_stopping=False,
                random_state=seed,
            )
        elif model == "xgb":
            from xgboost import XGBClassifier

            estimator = XGBClassifier(
                n_estimators=400,
                max_depth=3,
                learning_rate=0.04,
                min_child_weight=20,
                reg_lambda=5,
                subsample=0.8,
                colsample_bytree=0.9,
                tree_method="hist",
                eval_metric="logloss",
                random_state=seed,
                n_jobs=2,
            )
        else:
            raise ValueError(f"Неизвестная модель: {model}")
    estimator.set_params(**(parameters or {}))
    return Pipeline([("prepare", prepare), ("model", estimator)])


def review_metrics(y, probability, fraction):
    y, probability = np.asarray(y), np.asarray(probability)
    count = int(np.ceil(len(y) * fraction))
    threshold = np.sort(probability)[-count]
    above = probability > threshold
    tied = probability == threshold
    # При равных рисках усредняем выбор на границе, без сортировки по ответу.
    weight = (count - above.sum()) / tied.sum()
    caught = y[above].sum() + weight * y[tied].sum()
    return float(caught / y.sum()), float(caught / count)


def score(y, probability):
    recall, precision = review_metrics(y, probability, 0.2)
    return {
        "ROC-AUC": roc_auc_score(y, probability),
        "AP": average_precision_score(y, probability),
        "Brier": brier_score_loss(y, probability),
        "Recall@20%": recall,
        "Precision@20%": precision,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS)
    )
    args = parser.parse_args()
    path = Path("data/processed/train.csv")
    train = pd.read_csv(path)
    df = add_history_features(train)
    groups = profile_groups(train)
    folds = list(
        StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=args.seed).split(
            train, train[TARGET], groups
        )
    )
    output = Path(f"reports/experiments/seed_{args.seed}")
    output.mkdir(parents=True, exist_ok=True)
    assignment = np.zeros(len(train), dtype=int)
    for fold, (fit, valid) in enumerate(folds, 1):
        assert not set(groups.iloc[fit]) & set(groups.iloc[valid])
        assignment[valid] = fold
    pd.DataFrame({"ID": train.ID, "fold": assignment}).to_csv(
        output / "fold_ids.csv", index=False
    )
    oof = pd.DataFrame({"ID": train.ID, "fold": assignment, TARGET: train[TARGET]})
    results, settings = [], {}
    for name in args.variants:
        model, feature_set, parameters = VARIANTS[name]
        columns = FEATURE_SETS[feature_set]
        pipeline = build_pipeline(model, feature_set, args.seed, parameters)
        settings[name] = {
            "model": model,
            "columns": columns,
            "parameters": pipeline.named_steps["model"].get_params(),
        }
        probability = np.zeros(len(train))
        for fold, (fit, valid) in enumerate(folds, 1):
            pipeline.fit(df.iloc[fit][columns], train.iloc[fit][TARGET])
            probability[valid] = pipeline.predict_proba(df.iloc[valid][columns])[:, 1]
            results.append(
                {
                    "variant": name,
                    "fold": fold,
                    **score(train.iloc[valid][TARGET], probability[valid]),
                }
            )
        assert np.isfinite(probability).all()
        oof[name] = probability
        report = pd.DataFrame(results)
        report.to_csv(output / "fold_metrics.csv", index=False)
        report.drop(columns="fold").groupby("variant", sort=False).agg(
            ["mean", "std"]
        ).to_csv(output / "summary.csv")
        oof.to_csv(output / "oof.csv", index=False)
        print(name, json.dumps(score(train[TARGET], probability)), flush=True)
    metadata = {
        "source": "train.csv only; test not loaded",
        "train_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "seed": args.seed,
        "folds": 5,
        "grouping": "identical raw features, excluding ID and target",
        "python": platform.python_version(),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ["numpy", "pandas", "scikit-learn", "catboost", "xgboost"]
        },
        "code_sha256": {
            name: hashlib.sha256(
                Path(__file__).with_name(name).read_bytes()
            ).hexdigest()
            for name in ["experiments.py", "features.py", "split.py"]
        },
        "variants": settings,
    }
    (output / "settings.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
