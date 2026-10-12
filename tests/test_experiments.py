import numpy as np
import pandas as pd
import pytest

from credit_default.experiments import build_pipeline, review_metrics


def test_review_metrics_do_not_use_target_to_break_ties():
    y = np.array([1, 0, 0, 1, 0])
    risk = np.full(5, 0.3)
    recall, precision = review_metrics(y, risk, 0.2)
    assert recall == pytest.approx(0.2)
    assert precision == pytest.approx(0.4)
    assert review_metrics(y[::-1], risk, 0.2) == pytest.approx((recall, precision))


def test_review_metrics_at_fixed_capacity():
    recall, precision = review_metrics([1, 0, 1, 0, 0], [0.9, 0.8, 0.7, 0.2, 0.1], 0.4)
    assert recall == 0.5
    assert precision == 0.5


def test_preprocessing_fits_training_rows_only():
    from credit_default.experiments import CATEGORICAL, FEATURE_SETS

    columns = FEATURE_SETS["base"]
    train = pd.DataFrame({c: [1.0, 3.0, np.nan, 5.0] for c in columns})
    for c in CATEGORICAL:
        train[c] = ["a", "b", "a", "b"]
    pipeline = build_pipeline("hgb", "base")
    pipeline.fit(train, [0, 1, 0, 1])
    imputer = pipeline.named_steps["prepare"].named_transformers_["num"]
    statistics = imputer.statistics_.copy()
    unseen = train.iloc[:1].copy()
    unseen["LIMIT_BAL"] = 1_000_000
    unseen["EDUCATION"] = "unseen"
    assert np.isfinite(pipeline.predict_proba(unseen)).all()
    np.testing.assert_array_equal(statistics, imputer.statistics_)
    assert statistics[0] == 3
