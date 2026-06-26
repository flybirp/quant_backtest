"""
Return attribution module for quantitative backtesting.

Provides breakdowns of trade performance by year, sell reason,
position concentration, and monthly heatmap.
All return values are in percent (e.g., 5.2 means 5.2%, not 0.052).
"""

import math
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd


def yearly_attribution(trades: list[dict]) -> dict[int, dict[str, Any]]:
    """
    Attribute trade performance by year based on sell_date.

    Args:
        trades: List of trade dicts with 'profit_pct', 'sell_date' keys.

    Returns:
        Dict mapping year (int) to dict with:
        {'trades': count, 'win_rate': win rate in percent,
         'avg_return': average return in percent, 'total_return': sum return in percent}.
    """
    if not trades:
        return {}

    yearly: dict[int, list[float]] = defaultdict(list)

    for trade in trades:
        sell_date = trade.get("sell_date", "")
        profit = trade.get("profit_pct", 0.0)
        if np.isnan(profit):
            continue

        try:
            dt = pd.to_datetime(sell_date)
            year = int(dt.year)
        except Exception:
            continue

        yearly[year].append(profit)

    result: dict[int, dict[str, Any]] = {}
    for year in sorted(yearly.keys()):
        returns = yearly[year]
        n = len(returns)
        wins = sum(1 for r in returns if r > 0)
        result[year] = {
            "trades": n,
            "win_rate": round(wins / n * 100.0, 2) if n > 0 else 0.0,
            "avg_return": round(float(np.mean(returns)), 4),
            "total_return": round(float(np.sum(returns)), 4),
        }

    return result


def sell_reason_attribution(trades: list[dict]) -> dict[str, dict[str, Any]]:
    """
    Attribute trade performance by sell_reason.

    Args:
        trades: List of trade dicts with 'profit_pct', 'sell_reason' keys.

    Returns:
        Dict mapping reason (str) to dict with:
        {'count': number of trades, 'avg_return': average return in percent,
         'win_rate': win rate in percent}.
    """
    if not trades:
        return {}

    reason_map: dict[str, list[float]] = defaultdict(list)

    for trade in trades:
        reason = str(trade.get("sell_reason", "unknown"))
        profit = trade.get("profit_pct", 0.0)
        if np.isnan(profit):
            continue
        reason_map[reason].append(profit)

    result: dict[str, dict[str, Any]] = {}
    for reason, returns in reason_map.items():
        n = len(returns)
        wins = sum(1 for r in returns if r > 0)
        result[reason] = {
            "count": n,
            "avg_return": round(float(np.mean(returns)), 4),
            "win_rate": round(wins / n * 100.0, 2) if n > 0 else 0.0,
        }

    return result


def position_concentration(
    trades: list[dict], top_n: int = 10
) -> dict[str, list[dict[str, Any]]]:
    """
    Calculate position concentration - top N stocks by trade count and total return.

    Args:
        trades: List of trade dicts with 'code', 'profit_pct' keys.
        top_n: Number of top stocks to return (default 10).

    Returns:
        Dict with:
        - 'by_count': top N stocks by number of trades [{'code', 'count', 'total_return_pct'}].
        - 'by_return': top N stocks by total return [{'code', 'count', 'total_return_pct'}].
    """
    if not trades:
        return {"by_count": [], "by_return": []}

    stock_map: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"count": 0, "total_return": 0.0}
    )

    for trade in trades:
        code = str(trade.get("code", "unknown"))
        profit = trade.get("profit_pct", 0.0)
        if np.isnan(profit):
            continue
        stock_map[code]["count"] += 1
        stock_map[code]["total_return"] += profit

    # By count
    by_count = sorted(
        stock_map.items(), key=lambda x: x[1]["count"], reverse=True
    )[:top_n]
    by_count_list = [
        {
            "code": code,
            "count": info["count"],
            "total_return_pct": round(info["total_return"], 4),
        }
        for code, info in by_count
    ]

    # By return
    by_return = sorted(
        stock_map.items(), key=lambda x: x[1]["total_return"], reverse=True
    )[:top_n]
    by_return_list = [
        {
            "code": code,
            "count": info["count"],
            "total_return_pct": round(info["total_return"], 4),
        }
        for code, info in by_return
    ]

    return {"by_count": by_count_list, "by_return": by_return_list}


def monthly_heatmap(trades: list[dict]) -> dict[str, dict[str, float]]:
    """
    Create a month-year matrix of returns for heatmap visualization.

    Args:
        trades: List of trade dicts with 'profit_pct', 'sell_date' keys.

    Returns:
        Dict of the form {'YYYY': {'MM': return_pct, ...}, ...}.
        Returns are total returns for trades sold in that month.
    """
    if not trades:
        return {}

    monthly: dict[str, dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )

    for trade in trades:
        sell_date = trade.get("sell_date", "")
        profit = trade.get("profit_pct", 0.0)
        if np.isnan(profit):
            continue

        try:
            dt = pd.to_datetime(sell_date)
            year = str(dt.year)
            month = f"{dt.month:02d}"
        except Exception:
            continue

        monthly[year][month] += profit

    # Convert to regular dict and round
    result: dict[str, dict[str, float]] = {}
    for year in sorted(monthly.keys()):
        result[year] = {}
        for month in sorted(monthly[year].keys()):
            result[year][month] = round(monthly[year][month], 4)

    return result


# ── Sector Attribution ──────────────────────────────────────────


# A-share sector approximation by code prefix
_CODE_SECTOR_MAP: dict[str, str] = {}

def _init_sector_map() -> dict[str, str]:
    """Return a simple sector mapping based on stock code ranges."""
    base = {}
    for i in range(600, 604):
        base[str(i)] = "沪市主板"
    base["605"] = "沪市主板"
    base["000"] = "深市主板"
    base["001"] = "深市主板"
    base["002"] = "深市主板"
    base["300"] = "创业板"
    base["301"] = "创业板"
    base["688"] = "科创板"
    base["689"] = "科创板"
    return base


_CODE_SECTOR_MAP = _init_sector_map()


def _guess_sector(code: str) -> str:
    """Guess sector from stock code prefix."""
    code = str(code)
    for prefix in _CODE_SECTOR_MAP:
        if code.startswith(prefix):
            return _CODE_SECTOR_MAP[prefix]
    return "其他"


def sector_attribution(trades: list[dict]) -> dict[str, dict[str, Any]]:
    """Attribute trade performance by sector (inferred from stock code).

    Args:
        trades: List of trade dicts with 'code' and 'profit_pct'.

    Returns:
        Dict mapping sector name to
        {'count', 'avg_return', 'win_rate', 'total_return'}.
    """
    if not trades:
        return {}

    sector_map: dict[str, list[float]] = defaultdict(list)

    for t in trades:
        code = str(t.get("code", ""))
        profit = t.get("profit_pct", 0.0)
        if np.isnan(profit):
            continue
        sector = _guess_sector(code)
        sector_map[sector].append(profit)

    result = {}
    for sector, returns in sector_map.items():
        n = len(returns)
        wins = sum(1 for r in returns if r > 0) if n > 0 else 0
        result[sector] = {
            "count": n,
            "avg_return": round(float(np.mean(returns)), 4) if n > 0 else 0.0,
            "win_rate": round(wins / n * 100.0, 2) if n > 0 else 0.0,
            "total_return": round(float(np.sum(returns)), 4) if n > 0 else 0.0,
        }

    return result
