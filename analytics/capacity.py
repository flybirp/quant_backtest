"""
Capacity and turnover analysis for quantitative backtesting.

Estimates strategy turnover, trading frequency, and maximum
capacity constraints based on trade history and market liquidity.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd


def turnover_analysis(trades: list[dict], initial_capital: float) -> dict[str, Any]:
    """
    Analyse trade turnover characteristics from trade history.

    Args:
        trades: List of trade dicts with 'buy_date', 'sell_date',
                'buy_price', 'sell_price', 'shares', 'profit_pct'.
        initial_capital: Starting capital for portfolio mode, or
                         reference capital for signal mode.

    Returns:
        Dict with: total_trades, avg_trades_per_year, avg_trades_per_month,
        turnover_rate (annual, estimate), avg_hold_days, signal_density.
    """
    if not trades:
        return {
            "total_trades": 0,
            "avg_trades_per_year": 0.0,
            "avg_trades_per_month": 0.0,
            "turnover_rate_annual": 0.0,
            "avg_hold_days": 0.0,
            "signal_density_pct": 0.0,
            "unique_codes": 0,
        }

    # Temporal range
    all_dates = []
    hold_days_list = []

    for t in trades:
        try:
            all_dates.append(pd.Timestamp(t.get("sell_date", t.get("buy_date", ""))))
        except Exception:
            pass
        hd = t.get("hold_days", 0)
        if hd > 0:
            hold_days_list.append(hd)

    if not all_dates:
        return {
            "total_trades": len(trades),
            "avg_trades_per_year": 0.0,
            "avg_trades_per_month": 0.0,
            "turnover_rate_annual": 0.0,
            "avg_hold_days": 0.0,
            "signal_density_pct": 0.0,
            "unique_codes": 0,
        }

    min_date = min(all_dates)
    max_date = max(all_dates)
    total_days = max((max_date - min_date).days, 1)
    total_years = total_days / 365.25

    # Frequency
    avg_per_year = len(trades) / total_years if total_years > 0 else 0.0
    avg_per_month = avg_per_year / 12.0

    # Average hold days
    avg_hold = float(np.mean(hold_days_list)) if hold_days_list else 0.0

    # Turnover: for signal mode, estimate as trades * 100% / hold_days * years
    # For portfolio mode this would need actual capital data
    if avg_hold > 0:
        turnover_rate = (252.0 / avg_hold) * 100.0
    else:
        turnover_rate = avg_per_year * 100.0

    # Signal density: what fraction of days had a trade
    unique_trade_dates = set()
    for t in trades:
        d = t.get("sell_date", "")
        if d:
            unique_trade_dates.add(d[:10])
    signal_density = len(unique_trade_dates) / total_days * 100.0 if total_days > 0 else 0.0

    # Unique codes
    codes = set()
    for t in trades:
        c = t.get("code", "")
        if c:
            codes.add(c)

    return {
        "total_trades": len(trades),
        "avg_trades_per_year": round(avg_per_year, 1),
        "avg_trades_per_month": round(avg_per_month, 1),
        "turnover_rate_annual": round(turnover_rate, 1),
        "avg_hold_days": round(avg_hold, 1),
        "signal_density_pct": round(signal_density, 2),
        "unique_codes": len(codes),
    }


def capacity_estimate(
    trades: list[dict],
    daily_volume_data: dict[str, list[dict]] | None = None,
    participation_rate: float = 0.01,
) -> dict[str, Any]:
    """
    Estimate maximum strategy capacity based on trade volume participation.

    Without actual daily volume data, returns a conservative estimate
    based on trade frequency and average position turnover.

    Args:
        trades: List of trade dicts.
        daily_volume_data: Optional dict of code -> list of {date, volume, close}
                           for each traded stock.
        participation_rate: Maximum fraction of daily volume the strategy
                            can consume without impact (default 1%).

    Returns:
        Dict with estimated max capital and per-trade capacity.
    """
    if not trades:
        return {
            "max_capital_estimate": 0.0,
            "avg_trade_value_estimate": 0.0,
            "method": "insufficient_data",
        }

    # Without volume data: estimate from average trade size
    avg_shares = float(np.mean([t.get("shares", 100) for t in trades]))
    avg_price = float(np.mean([t.get("buy_price", 0) for t in trades if t.get("buy_price", 0) > 0]))

    if avg_shares > 0 and avg_price > 0:
        avg_trade_value = avg_shares * avg_price
    else:
        avg_trade_value = 0.0

    # Conservative: assume we can do 100x avg_trade_value across the portfolio
    max_capital = avg_trade_value * 100.0 if avg_trade_value > 0 else 0.0

    return {
        "max_capital_estimate": round(max_capital, 0),
        "avg_trade_value_estimate": round(avg_trade_value, 0),
        "avg_shares_per_trade": round(avg_shares, 0),
        "avg_price": round(avg_price, 2) if avg_price > 0 else 0.0,
        "target_participation_rate": participation_rate,
        "method": "average_trade_size (no volume data)",
    }


def monthly_trade_frequency(trades: list[dict]) -> dict[str, int]:
    """
    Number of trades per month (by sell_date).

    Args:
        trades: List of trade dicts with 'sell_date'.

    Returns:
        Dict mapping 'YYYY-MM' -> trade count.
    """
    monthly: dict[str, int] = defaultdict(int)
    for t in trades:
        sd = str(t.get("sell_date", ""))[:7]
        if sd:
            monthly[sd] += 1

    return dict(sorted(monthly.items()))
