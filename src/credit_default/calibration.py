import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedGroupKFold

from credit_default.data import TARGET
from credit_default.experiments import FEATURE_SETS, VARIANTS, build_pipeline, score
from credit_default.features import add_history_features
from credit_default.split import profile_groups


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=VARIANTS, default="cat_base")
    args = parser.parse_args()
    path = Path("data/processed/train.csv")
    train = pd.read_csv(path)
    df = add_history_features(train)
    groups = profile_groups(train)
    output = Path("reports/experiments/seed_42")
    saved = pd.read_csv(output / "fold_ids.csv")
    settings = json.loads((output / "settings.json").read_text())
    train_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    if settings["train_sha256"] != train_hash:
        raise ValueError("Train изменился: сначала повторите эксперименты")
    if train.ID.tolist() != saved.ID.tolist() or set(saved.fold) != {1, 2, 3, 4, 5}:
        raise ValueError("Состав блоков не соответствует train")
    model, feature_set, params = VARIANTS[args.variant]
    columns = FEATURE_SETS[feature_set]
    probability = np.zeros(len(train))
    rows, inner_ids = [], []
    for fold in range(1, 6):
        fit = np.flatnonzero(saved.fold.ne(fold))
        valid = np.flatnonzero(saved.fold.eq(fold))
        if set(groups.iloc[fit]) & set(groups.iloc[valid]):
            raise ValueError("Одинаковые профили попали в разные блоки")
        inner = list(
            StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42).split(
                train.iloc[fit], train.iloc[fit][TARGET], groups.iloc[fit]
            )
        )
        for block, (a, b) in enumerate(inner, 1):
            assert not set(groups.iloc[fit[a]]) & set(groups.iloc[fit[b]])
            inner_ids.extend(
                {"outer_fold": fold, "ID": int(client), "inner_fold": block}
                for client in train.iloc[fit[b]].ID
            )
        base = build_pipeline(model, feature_set, 42, params)
        # Калибратор видит только внутренние OOF; внешние ответы не используются.
        calibrated = CalibratedClassifierCV(
            base, method="sigmoid", cv=inner, ensemble=False, n_jobs=1
        )
        calibrated.fit(df.iloc[fit][columns], train.iloc[fit][TARGET])
        probability[valid] = calibrated.predict_proba(df.iloc[valid][columns])[:, 1]
        rows.append(
            {"fold": fold, **score(train.iloc[valid][TARGET], probability[valid])}
        )
        print("calibration", args.variant, fold, flush=True)
    pd.DataFrame(rows).to_csv(output / "sigmoid_metrics.csv", index=False)
    pd.DataFrame(inner_ids).to_csv(output / "sigmoid_inner_ids.csv", index=False)
    pd.DataFrame({"ID": train.ID, "risk": probability}).to_csv(
        output / "sigmoid_oof.csv", index=False
    )
    metadata = {
        "variant": args.variant,
        "method": "sigmoid",
        "ensemble": False,
        "outer_folds": 5,
        "inner_folds": 3,
        "seed": 42,
        "train_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (output / "sigmoid_settings.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(score(train[TARGET], probability))


if __name__ == "__main__":
    main()
