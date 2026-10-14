import pytest

from credit_default.monitor import population_stability


REFERENCE = {"edges": [None, 0.5, None], "counts": [100, 100]}


def test_psi_zero_for_equal_distributions():
    assert population_stability([0.1] * 100 + [0.8] * 100, REFERENCE) == 0


def test_psi_detects_shift_and_handles_empty_bin():
    assert population_stability([0.9] * 200, REFERENCE) > 1


@pytest.mark.parametrize("values", [[], [float("nan")], [-0.1], [1.1]])
def test_psi_rejects_invalid_probability(values):
    with pytest.raises(ValueError):
        population_stability(values, REFERENCE)
