"""Forecast-evaluation statistics shared by the finance forecasting tasks."""
from __future__ import annotations

import math

import numpy as np

from .runtime import CandidateError


def normal_sf(z):
    """Upper-tail probability of the standard normal."""
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def oos_r2(actual, forecast, benchmark):
    """Campbell-Thompson out-of-sample R^2 (percent) of ``forecast`` against ``benchmark``."""
    sse = float(np.sum((actual - forecast) ** 2))
    sse_bench = float(np.sum((actual - benchmark) ** 2))
    return 100.0 * (1.0 - sse / sse_bench)


def newey_west_variance(x, lags):
    """Long-run variance of a mean-zero-tested series with Bartlett weights."""
    x = np.asarray(x, dtype=float) - np.mean(x)
    n = len(x)
    value = float(x @ x) / n
    for lag in range(1, lags + 1):
        value += 2.0 * (1.0 - lag / (lags + 1.0)) * float(x[lag:] @ x[:-lag]) / n
    return value


def clark_west(actual, forecast, benchmark, lags=0):
    """Clark and West (2007) MSFE-adjusted statistic; one-sided p-value for H1: forecast beats benchmark."""
    f = (actual - benchmark) ** 2 - ((actual - forecast) ** 2 - (benchmark - forecast) ** 2)
    variance = newey_west_variance(f, lags)
    if not variance > 0:
        return 0.0, 0.5
    stat = float(np.mean(f)) / math.sqrt(variance / len(f))
    return stat, normal_sf(stat)


def diebold_mariano(loss_a, loss_b, lags=0):
    """Diebold and Mariano (1995) statistic for mean(loss_a - loss_b); negative favours ``a``.

    Returns (statistic, one-sided p-value for H1: a has lower expected loss).
    """
    d = np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float)
    variance = newey_west_variance(d, lags)
    if not variance > 0:
        return 0.0, 0.5
    stat = float(np.mean(d)) / math.sqrt(variance / len(d))
    return stat, normal_sf(-stat)


def finite_scalar(value, name, bound=None):
    """Validate a candidate's scalar output."""
    if isinstance(value, (bool, np.bool_, str, bytes)):
        raise CandidateError(f"{name} must be a real number, got {type(value).__name__}")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):  # OverflowError: an integer beyond the float range
        raise CandidateError(f"{name} must be a real number, got {type(value).__name__}") from None
    if not math.isfinite(number):
        raise CandidateError(f"{name} must be finite")
    if bound is not None and abs(number) > bound:
        raise CandidateError(f"{name}={number} is outside [-{bound}, {bound}]")
    return number


def finite_vector(value, name, shape, bound=None):
    """Validate a candidate's array output of a fixed shape."""
    try:
        array = np.asarray(value)
        if array.dtype.kind in "bSU":  # booleans and text are not numbers, as in finite_scalar
            raise TypeError(array.dtype)
        array = np.asarray(array, dtype=float)
    except (TypeError, ValueError, OverflowError):
        raise CandidateError(f"{name} must be numeric") from None
    if array.shape != tuple(shape):
        raise CandidateError(f"{name} must have shape {tuple(shape)}, got {array.shape}")
    if not np.isfinite(array).all():
        raise CandidateError(f"{name} must be finite")
    if bound is not None and np.abs(array).max(initial=0.0) > bound:
        raise CandidateError(f"{name} has entries outside [-{bound}, {bound}]")
    return array
