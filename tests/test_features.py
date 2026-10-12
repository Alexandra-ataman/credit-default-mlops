import pandas as pd
import pytest

from credit_default.data import BILLS, PAY, PAYMENTS, TARGET
from credit_default.features import NEW_NUMERIC, add_features


@pytest.fixture
def clients():
    df = pd.DataFrame(
        {"ID": [1, 2], "AGE": [29, 60], "LIMIT_BAL": [100, 200], TARGET: [0, 1]}
    )
    for col in PAY:
        df[col] = [-2, 0]
    df["PAY_0"] = [2, -1]
    df["PAY_2"] = [1, 3]
    for i, col in enumerate(BILLS):
        df[col] = [60 - i * 10, -10]
    for i, col in enumerate(PAYMENTS):
        df[col] = [10 if i < 3 else 0, 0]
    return df


def test_delay_codes_and_month_order(clients):
    result = add_features(clients)
    assert result.delay_months.tolist() == [2, 1]
    assert result.previous_delay_months.tolist() == [1, 1]
    assert result.max_delay.tolist() == [2, 3]
    assert result.severe_delay_months.tolist() == [1, 1]
    assert result.loc[0, "bill_change_to_limit"] == 0.5
    assert result.loc[0, "payment_change_to_limit"] == 0.1


def test_ratios_and_age_boundaries(clients):
    result = add_features(clients)
    assert result.loc[0, "payment_to_previous_bill"] == 0.2
    assert pd.isna(result.loc[1, "payment_to_previous_bill"])
    assert result.loc[1, "bill_to_limit"] == -0.05
    assert result.AGE_BAND.tolist() == ["До 30", "60+"]


def test_no_target_or_id_leakage(clients):
    original = clients.copy(deep=True)
    first = add_features(clients)
    clients[TARGET] = 1 - clients[TARGET]
    clients["ID"] = [999, 888]
    second = add_features(clients)
    pd.testing.assert_frame_equal(
        first[NEW_NUMERIC + ["AGE_BAND"]], second[NEW_NUMERIC + ["AGE_BAND"]]
    )
    pd.testing.assert_frame_equal(first[original.columns], original)


def test_features_do_not_depend_on_other_rows(clients):
    original = clients.copy(deep=True)
    full = add_features(clients)
    one = add_features(clients.iloc[:1])
    pd.testing.assert_frame_equal(full.iloc[:1], one)
    pd.testing.assert_frame_equal(clients, original)


def test_consecutive_delays_and_recency(clients):
    from credit_default.features import add_history_features

    patterns = [
        [2, 1, -1, 2, 2, 2],
        [-1, 2, -1, 2, -2, 0],
        [0, -2, -1, 0, -1, -2],
        [1, 1, 1, 1, 1, 1],
    ]
    frame = pd.concat([clients.iloc[:1]] * len(patterns), ignore_index=True)
    frame[PAY] = patterns
    result = add_history_features(frame)
    assert result.longest_delay_run.tolist() == [3, 1, 0, 6]
    assert result.current_delay_run.tolist() == [2, 0, 0, 6]
    assert result.months_since_delay.tolist() == [0, 1, 6, 0]
    assert result.delay_count_change.tolist() == [-1, 0, 0, 0]
    assert result.known_status_change.iloc[:2].tolist() == [1, -2]
    assert pd.isna(result.known_status_change.iloc[2])


def test_extended_features_are_row_local(clients):
    from credit_default.features import (
        MONTHLY_NUMERIC,
        SEQUENCE_NUMERIC,
        add_history_features,
    )

    cols = NEW_NUMERIC + SEQUENCE_NUMERIC + MONTHLY_NUMERIC
    expected = add_history_features(clients)[cols]
    changed = clients.copy()
    changed[TARGET] = 1 - changed[TARGET]
    changed["ID"] += 100
    pd.testing.assert_frame_equal(expected, add_history_features(changed)[cols])
    pd.testing.assert_frame_equal(
        expected.iloc[:1], add_history_features(clients.iloc[:1])[cols]
    )
