import joblib
import numpy as np
import pytest
from fastapi.testclient import TestClient

from credit_default.api import create_app
from credit_default.experiments import RAW


class FixedModel:
    threshold_ = 0.9

    def predict_proba(self, frame):
        assert list(frame.columns) == RAW
        return np.tile([0.2, 0.8], (len(frame), 1))


@pytest.fixture
def request_data(tmp_path):
    path = tmp_path / "model.joblib"
    joblib.dump(FixedModel(), path)
    row = dict.fromkeys(RAW, 0)
    row.update(LIMIT_BAL=10000, AGE=30, SEX=1, EDUCATION=2, MARRIAGE=1)
    with TestClient(create_app(path)) as client:
        yield client, row


def test_prediction_uses_saved_threshold(request_data):
    client, row = request_data
    response = client.post("/predict", json={"clients": [row]})
    assert response.status_code == 200
    assert response.json() == {"predictions": [{"class": 0, "probability": 0.8}]}
    assert len(client.get("/health").json()["model_sha256"]) == 64


@pytest.mark.parametrize(
    "field,value",
    [
        ("AGE", 17),
        ("LIMIT_BAL", 0),
        ("SEX", 3),
        ("PAY_0", 10),
        ("PAY_AMT1", -1),
        ("AGE", "30"),
        ("AGE", 30.5),
        ("AGE", None),
    ],
)
def test_invalid_input_rejected(request_data, field, value):
    client, row = request_data
    row[field] = value
    assert client.post("/predict", json={"clients": [row]}).status_code == 422


def test_missing_unknown_fields_and_empty_batch(request_data):
    client, row = request_data
    row.pop("AGE")
    for batch in [[row], [dict(row, AGE=30, ID=1)], []]:
        assert client.post("/predict", json={"clients": batch}).status_code == 422
