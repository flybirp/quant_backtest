"""
Shared utilities for the analytics package.

Provides common data-conversion helpers used across performance,
risk, statistics, and benchmark sub-modules.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any


def to_equity_df(equity_curve: list[dict]) -> pd.DataFrame:
    """Convert equity_curve list-of-dicts to a sorted DataFrame with datetime index."""
    if not equity_curve:
        return pd.DataFrame(columns=["date", "equity"])
    df = pd.DataFrame(equity_curve)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df.set_index("date", inplace=True)
    return df


def extract_trade_returns(trades: list[dict]) -> np.ndarray:
    """Extract profit_pct values from trades as a numpy array."""
    if not trades:
        return np.array([])
    arr = np.array([t.get("profit_pct", 0.0) for t in trades], dtype=float)
    return arr[~np.isnan(arr)]


def compute_daily_returns(equity_curve: list[dict]) -> pd.Series:
    """Compute daily percentage returns from equity curve (in percent)."""
    df = to_equity_df(equity_curve)
    if len(df) < 2:
        return pd.Series(dtype=float)
    returns = df["equity"].pct_change().dropna() * 100.0
    return returns


def forward_fill_daily(equity_curve: list[dict]) -> pd.DataFrame:
    """
    Forward-fill a sparse equity curve to create a daily series.

    Signal-mode equity curves only have entries on trade sell dates.
    This fills gaps so that daily returns (pct_change) are meaningful.
    Handles duplicate dates by keeping the last value per date.
    """
    df = to_equity_df(equity_curve)
    if df.empty:
        return df
    # Remove duplicate dates: keep last equity value per day
    if df.index.duplicated().any():
        df = df[~df.index.duplicated(keep="last")]
    # Reindex to a continuous daily date range and forward-fill
    full_idx = pd.date_range(start=df.index[0], end=df.index[-1], freq="D")
    df = df.reindex(full_idx, method="ffill")
    return df


def safe_divide(a: float, b: float) -> float:
    """Divide a by b, returning 0.0 when b is 0."""
    if b == 0:
        return 0.0
    return a / b


def annualized_from_daily(daily_ratio_returns: np.ndarray, trading_days: int = 252) -> float:
    """Annualize a mean daily return ratio (e.g., 0.001 = 0.1%) to percent."""
    if len(daily_ratio_returns) < 2:
        return 0.0
    mean_daily = float(np.mean(daily_ratio_returns))
    # Compound: (1 + r)^252 - 1
    annual = ((1.0 + mean_daily) ** trading_days - 1.0) * 100.0
    return round(annual, 4)
