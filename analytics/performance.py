"""
Performance metrics module for quantitative backtesting.

Operates on equity_curve (list of dicts with 'date' and 'equity')
and trades (list of dicts with 'profit_pct', 'hold_days', 'sell_date').

All return values are in percent (e.g., 5.2 means 5.2%, not 0.052).
"""

import math
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd

from .common import to_equity_df as _to_equity_df, safe_divide as _safe_divide


def total_return(equity_curve: list[dict], initial_capital: float) -> float:
    """
    Calculate total return as a percentage.

    Args:
        equity_curve: List of dicts with 'date' and 'equity' keys.
        initial_capital: Starting capital amount.

    Returns:
        Total return in percent (e.g., 15.3 means 15.3%).
    """
    if not equity_curve or initial_capital <= 0:
        return 0.0
    df = _to_equity_df(equity_curve)
    final_equity = float(df["equity"].iloc[-1])
    if np.isnan(final_equity) or np.isinf(final_equity):
        return 0.0
    return (final_equity - initial_capital) / initial_capital * 100.0


def annual_return(equity_curve: list[dict], initial_capital: float) -> float:
    """
    Calculate annualized return (CAGR) as a percentage.

    Args:
        equity_curve: List of dicts with 'date' and 'equity' keys.
        initial_capital: Starting capital amount.

    Returns:
        Annualized return in percent.
    """
    if not equity_curve or initial_capital <= 0:
        return 0.0
    df = _to_equity_df(equity_curve)
    if len(df) < 2:
        return 0.0

    final_equity = float(df["equity"].iloc[-1])
    if np.isnan(final_equity) or np.isinf(final_equity) or final_equity <= 0:
        return 0.0

    start_date = df.index[0]
    end_date = df.index[-1]
    years = (end_date - start_date).days / 365.25
    if years <= 0:
        return 0.0

    total_ret_ratio = final_equity / initial_capital
    if total_ret_ratio <= 0:
        return -100.0

    cagr = (total_ret_ratio ** (1.0 / years) - 1.0) * 100.0
    return cagr


def monthly_returns_table(equity_curve: list[dict]) -> dict[str, float]:
    """
    Calculate monthly returns as a dict mapping 'YYYY-MM' to return in percent.

    Args:
        equity_curve: List of dicts with 'date' and 'equity' keys.

    Returns:
        Dict mapping month string 'YYYY-MM' to return in percent.
    """
    if not equity_curve:
        return {}

    df = _to_equity_df(equity_curve)
    if len(df) < 2:
        return {}

    # Resample to month-end and compute monthly returns
    monthly = df["equity"].resample("ME").last()
    # Drop months with NaN (can happen with sparse data)
    monthly = monthly.dropna()
    if len(monthly) < 2:
        return {}

    returns = monthly.pct_change().dropna() * 100.0
    result: dict[str, float] = {}
    for dt, val in returns.items():
        key = dt.strftime("%Y-%m")
        result[key] = round(float(val), 4)
    return result


def yearly_returns_table(equity_curve: list[dict]) -> dict[int, float]:
    """
    Calculate yearly returns as a dict mapping year to return in percent.

    Args:
        equity_curve: List of dicts with 'date' and 'equity' keys.

    Returns:
        Dict mapping year (int) to return in percent.
    """
    if not equity_curve:
        return {}

    df = _to_equity_df(equity_curve)
    if len(df) < 2:
        return {}

    yearly = df["equity"].resample("YE").last()
    yearly = yearly.dropna()
    if len(yearly) < 2:
        return {}

    returns = yearly.pct_change().dropna() * 100.0
    result: dict[int, float] = {}
    for dt, val in returns.items():
        result[int(dt.year)] = round(float(val), 4)
    return result


def rolling_returns(
    equity_curve: list[dict], window_days: int = 252
) -> list[dict[str, Any]]:
    """
    Calculate rolling window returns.

    Args:
        equity_curve: List of dicts with 'date' and 'equity' keys.
        window_days: Rolling window size in trading days (default 252 = ~1 year).

    Returns:
        List of dicts with 'date' and 'return_pct' keys.
    """
    if not equity_curve:
        return []

    df = _to_equity_df(equity_curve)
    if len(df) < window_days:
        return []

    rolling = df["equity"].pct_change(periods=window_days) * 100.0
    rolling = rolling.dropna()

    result: list[dict[str, Any]] = []
    for dt, val in rolling.items():
        if not np.isnan(val) and not np.isinf(val):
            result.append({
                "date": dt.strftime("%Y-%m-%d"),
                "return_pct": round(float(val), 4),
            })
    return result


def cumulative_returns_series(
    equity_curve: list[dict], initial_capital: float
) -> list[dict[str, Any]]:
    """
    Calculate cumulative return series as a percentage of initial capital.

    Args:
        equity_curve: List of dicts with 'date' and 'equity' keys.
        initial_capital: Starting capital amount.

    Returns:
        List of dicts with 'date' and 'cum_return_pct' keys.
    """
    if not equity_curve or initial_capital <= 0:
        return []

    df = _to_equity_df(equity_curve)
    cum_returns = (df["equity"] / initial_capital - 1.0) * 100.0

    result: list[dict[str, Any]] = []
    for dt, val in cum_returns.items():
        val_f = float(val)
        if np.isnan(val_f) or np.isinf(val_f):
            val_f = 0.0
        result.append({
            "date": dt.strftime("%Y-%m-%d"),
            "cum_return_pct": round(val_f, 4),
        })
    return result
