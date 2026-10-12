import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from credit_default.data import TARGET


def profile_groups(df):
    # Одинаковые признаки получают одну группу. ID и ответ в неё не входят.
    features = df.drop(columns=["ID", TARGET])
    return features.groupby(list(features.columns), dropna=False).ngroup()


def split_data(df, seed=42):
    if df["ID"].isna().any() or not df["ID"].is_unique:
        raise ValueError("ID должны быть заполнены и уникальны")
    if df[TARGET].isna().any() or set(df[TARGET].unique()) != {0, 1}:
        raise ValueError("Нужны оба класса: 0 и 1, без пропусков")

    groups = profile_groups(df)
    if groups.nunique() < 5:
        raise ValueError("Нужно хотя бы пять разных профилей")
    if df[TARGET].value_counts().min() < 5:
        raise ValueError("Нужно хотя бы пять строк каждого класса")

    # Один из пяти блоков — test (~20%). Профили не разрываем между выборками.
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    train_index, test_index = next(splitter.split(df, df[TARGET], groups))
    train = df.iloc[train_index].copy()
    test = df.iloc[test_index].copy()
    if train[TARGET].nunique() != 2 or test[TARGET].nunique() != 2:
        raise ValueError("В каждую выборку должны попасть оба класса")
    return train, test


def split_report(df, train, test, seed=42):
    groups = profile_groups(df)
    labels = pd.Series(groups.to_numpy(), index=df["ID"])
    train_groups = set(labels.loc[train["ID"]])
    test_groups = set(labels.loc[test["ID"]])
    # Разные ответы при одинаковых признаках сохраняем, а не исправляем вручную.
    conflicting = df.groupby(groups)[TARGET].nunique().gt(1).sum()
    return {
        "seed": seed,
        "method": "StratifiedGroupKFold: first of 5 folds",
        "rows": len(df),
        "train_rows": len(train),
        "test_rows": len(test),
        "train_default_rate": float(train[TARGET].mean()),
        "test_default_rate": float(test[TARGET].mean()),
        "overlapping_ids": len(set(train["ID"]) & set(test["ID"])),
        "overlapping_profiles": len(train_groups & test_groups),
        "conflicting_profiles": int(conflicting),
    }


def main():
    from credit_default.validation import validate

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path, default=Path("data/processed/clients.csv")
    )
    parser.add_argument("--output", type=Path, default=Path("data/processed"))
    parser.add_argument("--reports", type=Path, default=Path("reports"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    if not validate(df)["success"]:
        raise ValueError("Входные данные не прошли валидацию")
    train, test = split_data(df, args.seed)
    report = split_report(df, train, test, args.seed)
    if report["overlapping_ids"] or report["overlapping_profiles"]:
        raise ValueError("Train и test пересекаются")

    args.output.mkdir(parents=True, exist_ok=True)
    train.to_csv(args.output / "train.csv", index=False)
    test.to_csv(args.output / "test.csv", index=False)
    args.reports.mkdir(parents=True, exist_ok=True)
    # Фиксируем состав выборок и хеш входного CSV для повторного запуска.
    df[["ID"]].assign(
        split=df["ID"].isin(test["ID"]).map({True: "test", False: "train"})
    ).to_csv(args.reports / "split_ids.csv", index=False)
    report["input_sha256"] = hashlib.sha256(args.input.read_bytes()).hexdigest()
    (args.reports / "split_summary.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
