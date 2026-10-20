import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from credit_default.api import create_app
from credit_default.data import PAY, TARGET
from credit_default.experiments import RAW
from credit_default.model import build_model
from credit_default.train import candidates, check_split, evaluate


def test_metrics_use_probability_and_fixed_threshold():
    metrics = evaluate([0, 1, 0, 1], [0.1, 0.4, 0.3, 0.9])
    assert metrics["ROC-AUC"] == 1
    assert metrics["Precision"] == 1
    assert metrics["Recall"] == 0.5
    assert metrics["F1"] == pytest.approx(2 / 3)


@pytest.mark.parametrize("kind", ["cat", "logreg"])
def test_saved_pipeline_accepts_raw_features(tmp_path, kind):
    frame = pd.DataFrame(dict.fromkeys(RAW, range(1, 21)))
    frame = frame.assign(AGE=30, LIMIT_BAL=10000, SEX=1, EDUCATION=2, MARRIAGE=1)
    frame[PAY] = 0
    model = build_model(kind=kind)
    if kind == "cat":
        model.set_params(model__iterations=3, model__depth=2)
    model.fit(frame, [0, 1] * 10)
    joblib.dump(model, tmp_path / "model.joblib")
    restored = joblib.load(tmp_path / "model.joblib")
    np.testing.assert_allclose(
        model.predict_proba(frame), restored.predict_proba(frame)
    )
    assert len(restored.predict(frame.head(1))) == 1
    with TestClient(create_app(tmp_path / "model.joblib")) as client:
        response = client.post(
            "/predict", json={"clients": frame.head(1).to_dict("records")}
        )
    assert response.status_code == 200
    assert response.json()["predictions"][0]["probability"] == pytest.approx(
        model.predict_proba(frame.head(1))[0, 1]
    )


def test_split_rejects_matching_profiles_and_ids():
    train = pd.DataFrame({"ID": [1], "AGE": [20], TARGET: [0]})
    check_split(train, train.assign(ID=2, AGE=30))
    for test in [train.assign(ID=2), train.assign(AGE=30)]:
        with pytest.raises(ValueError, match="пересекаются"):
            check_split(train, test)


def test_tuning_keeps_control_and_reproduces_candidates():
    config = json.loads(Path("params.json").read_text())["tune"]
    choices = candidates(config)
    assert choices == candidates(config)
    assert choices[0] == {f"model__{k}": [v] for k, v in config["control"].items()}
    assert len(choices) == config["trials"]


@pytest.mark.parametrize("trials", [1, 2, 3])
def test_tuning_replaces_repeated_control(trials):
    config = {
        "seed": 0,
        "trials": trials,
        "control": {"depth": 6},
        "space": {"depth": [4, 5, 6]},
    }
    choices = candidates(config)
    assert len(choices) == trials
    assert choices[0] == {"model__depth": [6]}
    assert len({p["model__depth"][0] for p in choices}) == trials
