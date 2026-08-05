from __future__ import annotations

import pytest

from evaluation.metrics.correlation import pearson_r


def test_pearson_r_perfect_positive_correlation() -> None:
    xs = [1.0, 2.0, 3.0, 4.0]
    ys = [2.0, 4.0, 6.0, 8.0]

    assert pearson_r(xs, ys) == pytest.approx(1.0)


def test_pearson_r_perfect_negative_correlation() -> None:
    xs = [1.0, 2.0, 3.0, 4.0]
    ys = [8.0, 6.0, 4.0, 2.0]

    assert pearson_r(xs, ys) == pytest.approx(-1.0)


def test_pearson_r_no_correlation_with_constant_series() -> None:
    xs = [1.0, 2.0, 3.0]
    ys = [5.0, 5.0, 5.0]

    assert pearson_r(xs, ys) == 0.0


def test_pearson_r_fewer_than_two_points_is_zero_not_a_crash() -> None:
    assert pearson_r([], []) == 0.0
    assert pearson_r([1.0], [2.0]) == 0.0


def test_pearson_r_binary_outcome_series() -> None:
    # Point-biserial correlation is mathematically Pearson's r with one series
    # 0/1-coded -- exactly the confidence-vs-correctness use case.
    confidence = [0.9, 0.8, 0.2, 0.1]
    correct = [1.0, 1.0, 0.0, 0.0]

    assert pearson_r(confidence, correct) > 0.9
