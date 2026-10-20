import argparse
import hashlib
import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from mlflow.models import infer_signature
from sklearn.metrics import RocCurveDisplay, precision_recall_fscore_support
from sklearn.model_selection import (
    ParameterGrid,
    ParameterSampler,
    RandomizedSearchCV,
    StratifiedGroupKFold,
)

from credit_default.data import TARGET
from credit_default.experiments import RAW, score
from credit_default.model import build_model
from credit_default.split import profile_groups, split_report


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def candidates(config):
    sampled = ParameterSampler(
        config["space"],
        min(config["trials"], len(ParameterGrid(config["space"]))),
        random_state=config["seed"],
    )
    # Берём запасной вариант на случай повтора контроля.
    choices = [config["control"]] + [p for p in sampled if p != config["control"]]
    return [
        {f"model__{k}": [v] for k, v in p.items()} for p in choices[: config["trials"]]
    ]


def check_split(train, test):
    report = split_report(pd.concat([train, test]), train, test)
    if report["overlapping_ids"] or report["overlapping_profiles"]:
        raise ValueError("Train и test пересекаются")


def evaluate(y, risk, threshold=0.5):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, np.asarray(risk) >= threshold, average="binary", zero_division=0
    )
    return {**score(y, risk), "Precision": precision, "Recall": recall, "F1": f1}


def save_roc(y, risk, path):
    figure = Figure()
    ax = figure.subplots()
    RocCurveDisplay.from_predictions(y, risk, ax=ax, name="Модель")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set(
        title="ROC на отложенной выборке",
        xlabel="Доля ложных тревог",
        ylabel="Доля найденных дефолтов",
    )
    figure.savefig(path, dpi=120, bbox_inches="tight")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("params.json"))
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--tune", action="store_true")
    modes.add_argument("--logreg", action="store_true")
    args = parser.parse_args()
    mode = "tune" if args.tune else "logreg" if args.logreg else "train"
    config = json.loads(args.config.read_text())[mode]
    train = pd.read_csv("data/processed/train.csv")
    train_hash = sha256("data/processed/train.csv")
    baseline = json.loads(Path("reports/baseline.json").read_text())
    if train_hash != baseline["train_sha256"]:
        raise ValueError("Train изменился относительно исходного сравнения")
    space = (
        candidates(config)
        if args.tune
        else {
            f"model__{k}": config[k]
            for k in (["C"] if args.logreg else ["depth", "l2_leaf_reg"])
        }
    )
    search = RandomizedSearchCV(
        build_model(config["seed"], kind="logreg" if args.logreg else "cat"),
        space,
        n_iter=len(ParameterGrid(space)),
        scoring="roc_auc",
        random_state=config["seed"],
        n_jobs=1,
        cv=StratifiedGroupKFold(5, shuffle=True, random_state=config["seed"]),
        error_score="raise",
        verbose=1,
    )
    output = (
        Path("reports")
        / {"train": "training", "tune": "tuning", "logreg": "logreg"}[mode]
    )
    output.mkdir(parents=True, exist_ok=True)
    Path("models").mkdir(exist_ok=True)
    mlflow.set_tracking_uri(Path("mlruns").resolve().as_uri())
    mlflow.set_experiment("credit-default")
    run_name = {
        "train": "base-search",
        "tune": "expanded-search",
        "logreg": "logreg-baseline",
    }[mode]
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(config)
        mlflow.log_param(
            "model_type", "LogisticRegression" if args.logreg else "CatBoost"
        )
        mlflow.set_tag("train_sha256", train_hash)
        search.fit(train[RAW], train[TARGET], groups=profile_groups(train))
        results = search.cv_results_
        table = pd.DataFrame(results["params"]).rename(
            columns=lambda c: c.removeprefix("model__")
        )
        parameters = list(table.columns)
        table["cv_auc"] = results["mean_test_score"]
        table["cv_auc_std"] = results["std_test_score"]
        for i in range(5):
            table[f"fold_{i + 1}_auc"] = results[f"split{i}_test_score"]
        for i, params in enumerate(results["params"]):
            with mlflow.start_run(run_name=f"candidate-{i + 1}", nested=True):
                mlflow.log_params(params)
                mlflow.log_metrics(table.iloc[i].drop(parameters).to_dict())
        columns = list(table) if args.tune else parameters + ["cv_auc", "cv_auc_std"]
        table[columns].to_csv(output / "search.csv", index=False)
        model = search.best_estimator_
        selected = {
            "parameters": search.best_params_,
            "cv_auc": search.best_score_,
            "train_sha256": train_hash,
        }
        name = {"train": "model", "tune": "candidate", "logreg": "logreg"}[mode]
        model_path = Path("models") / f"{name}.joblib"
        if args.tune:
            control = table.loc[
                table[parameters].eq(pd.Series(config["control"])).all(axis=1)
            ].iloc[0]
            selected.update(control_cv_auc=float(control.cv_auc), test_evaluated=False)
        else:
            # Test читаем только после выбора параметров.
            test = pd.read_csv("data/processed/test.csv")
            check_split(train, test)
            risk = model.predict_proba(test[RAW])[:, 1]
            metrics = evaluate(test[TARGET], risk, config["threshold"])
            save_roc(test[TARGET], risk, output / "roc.png")
            selected.update(
                threshold=config["threshold"],
                test=metrics,
                test_sha256=sha256("data/processed/test.csv"),
            )
            mlflow.log_metrics(
                {
                    k.lower().replace("-", "_").replace("@", "_at_").replace("%", ""): v
                    for k, v in metrics.items()
                }
            )
            model.threshold_ = config["threshold"]
        joblib.dump(model, model_path)
        if mode == "train":
            # Фон PSI рассчитываем по train и привязываем к версии модели.
            train_risk = model.predict_proba(train[RAW])[:, 1]
            edges = np.unique(np.quantile(train_risk, np.linspace(0, 1, 11)))
            edges[0], edges[-1] = -np.inf, np.inf
            counts = np.histogram(train_risk, edges)[0]
            write_json(
                "models/reference.json",
                {
                    "edges": [None, *edges[1:-1], None],
                    "counts": counts.tolist(),
                    "model_sha256": sha256(model_path),
                },
            )
        write_json(
            output / ("selection.json" if args.tune else "metrics.json"), selected
        )
        mlflow.log_params(
            {
                f"model__{k}": v
                for k, v in model.named_steps["model"].get_params().items()
            }
        )
        mlflow.log_metric("cv_auc", search.best_score_)
        mlflow.log_artifacts(str(output))
        example = train[RAW].head(3)
        mlflow.sklearn.log_model(
            model,
            "model",
            input_example=example,
            signature=infer_signature(example, model.predict(example)),
            code_paths=["src/credit_default"],
            pip_requirements=[
                "scikit-learn==1.5.2",
                "pandas==2.1.4",
                "numpy==1.26.4",
                "catboost==1.2.8",
            ],
        )
    print(json.dumps(selected, indent=2))


if __name__ == "__main__":
    main()
