"""Pearson correlation, hand-rolled rather than adding a `scipy`/`numpy` runtime
dependency for one statistic used by a single one-off analysis
(`scripts/confidence_correlation.py`) -- see that script for what it's used for.
"""

from __future__ import annotations


def pearson_r(xs: list[float], ys: list[float]) -> float:
    """Pearson correlation coefficient between `xs` and `ys` (paired, same length).

    Returns 0.0 for fewer than 2 points or when either series has zero variance
    (correlation is undefined there; 0.0 signals "no measurable relationship"
    rather than raising, since callers report this alongside sample size).
    """
    n = len(xs)
    if n < 2:
        return 0.0
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True)) / n
    std_x = (sum((x - mean_x) ** 2 for x in xs) / n) ** 0.5
    std_y = (sum((y - mean_y) ** 2 for y in ys) / n) ** 0.5
    if std_x == 0.0 or std_y == 0.0:
        return 0.0
    return cov / (std_x * std_y)
