import pandas as pd
import pytest

from credit_default.data import COLUMNS, TARGET, read_source, summarize
from credit_default.validation import validate


@pytest.fixture
def clients():
    frame = pd.DataFrame(0, index=range(2), columns=COLUMNS, dtype="int64")
    frame["ID"] = [1, 2]
    frame["LIMIT_BAL"] = [10000, 20000]
    frame["SEX"] = [1, 2]
    frame["AGE"] = [21, 40]
    frame[TARGET] = [0, 1]
    return frame


def test_source_header_is_normalized(monkeypatch):
    source = pd.DataFrame({"default payment next month": [1]})
    monkeypatch.setattr(pd, "read_excel", lambda path, header: source)
    assert read_source("unused.xls").columns.tolist() == [TARGET]


def test_summary(clients):
    result = summarize(clients)
    assert result["rows"] == 2
    assert result["features"] == 23
    assert result["default_rate"] == 0.5


def test_valid_data(clients):
    # Допустимые коды и отрицательный BILL_AMT.
    clients.loc[0, "EDUCATION"] = 6
    clients.loc[0, "PAY_0"] = -2
    clients.loc[0, "BILL_AMT1"] = -100
    assert validate(clients)["success"]


@pytest.mark.parametrize(
    "column,value",
    [
        ("AGE", 10),
        ("SEX", 3),
        (TARGET, 2),
        ("PAY_0", 10),
        ("PAY_AMT1", -1),
        ("LIMIT_BAL", 0),
        ("AGE", None),
        ("AGE", "old"),
    ],
)
def test_anomalies_are_rejected(clients, column, value):
    if isinstance(value, str):
        clients[column] = clients[column].astype(object)
    clients.loc[0, column] = value
    assert not validate(clients)["success"]


def test_duplicate_id_is_rejected(clients):
    clients["ID"] = 1
    assert not validate(clients)["success"]


def test_missing_column_is_rejected(clients):
    assert not validate(clients.drop(columns="AGE"))["success"]


def test_empty_data_is_rejected(clients):
    assert not validate(clients.iloc[:0])["success"]
