import pandas as pd
import pytest

from credit_default.data import TARGET
from credit_default.split import split_data, split_report


@pytest.fixture
def clients():
    # По две строки на профиль, часть ответов внутри профиля различается.
    return pd.DataFrame(
        {
            "ID": range(200),
            "AGE": [20 + i // 2 for i in range(200)],
            TARGET: [int(i % 4 == 0) for i in range(200)],
        }
    )


def test_no_rows_lost_or_duplicated(clients):
    train, test = split_data(clients)
    assert set(train.ID) | set(test.ID) == set(clients.ID)
    assert set(train.ID).isdisjoint(test.ID)
    assert len(train) + len(test) == len(clients)


def test_profiles_stay_together(clients):
    train, test = split_data(clients)
    assert set(train.AGE).isdisjoint(test.AGE)
    report = split_report(clients, train, test)
    assert report["overlapping_profiles"] == 0
    assert report["conflicting_profiles"] == 50


def test_split_is_reproducible(clients):
    first = split_data(clients, seed=42)
    second = split_data(clients, seed=42)
    for left, right in zip(first, second):
        pd.testing.assert_frame_equal(left, right)


def test_class_balance(clients):
    train, test = split_data(clients)
    assert 0.15 <= len(test) / len(clients) <= 0.25
    for part in [train, test]:
        assert set(part[TARGET]) == {0, 1}
        assert abs(part[TARGET].mean() - clients[TARGET].mean()) < 0.05


def test_values_are_not_changed(clients):
    original = clients.copy(deep=True)
    train, test = split_data(clients)
    restored = pd.concat([train, test]).sort_index()
    pd.testing.assert_frame_equal(restored, original)
    pd.testing.assert_frame_equal(clients, original)


def test_repeated_id_is_rejected(clients):
    clients.loc[1, "ID"] = clients.loc[0, "ID"]
    with pytest.raises(ValueError, match="ID"):
        split_data(clients)


def test_single_class_is_rejected(clients):
    clients[TARGET] = 0
    with pytest.raises(ValueError, match="оба класса"):
        split_data(clients)
