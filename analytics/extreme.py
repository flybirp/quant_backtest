"""
Extreme drawdown analysis — worst N-day rolling returns and week-level stress.

Answers: "What's the worst that can happen in 5 trading days?"
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from .common import to_equity_df as _to_equity_df


def rolling_max_drawdown(
    equity_curve: list[dict],
    window: int = 5,
    initial_capital: float = 100000,
) -> dict:
    """
    Compute worst-case rolling N-day returns on the equity curve.

    Uses forward-filled daily equity to ensure calendar-day windows,
    not trade-count windows.

    Args:
        equity_curve: List of dicts with 'date' and 'equity' keys.
        window: Rolling window in CALENDAR days (default 5 = 1 week).
        initial_capital: Starting capital for percentage calculation.

    Returns:
        {
            "worst_return_pct": float,
            "worst_start_date": str,
            "worst_end_date": str,
            "worst_return_money": float,
            "avg_worst_5_pct": float,
            "all_rolling_pct": [float, ...],
        }
    """
    from .common import forward_fill_daily

    # Forward-fill to get daily equity
    daily = forward_fill_daily(equity_curve)
    eq = daily["equity"].values

    if len(eq) <= window:
        return {
            "worst_return_pct": 0,
            "worst_start_date": "",
            "worst_end_date": "",
            "worst_return_money": 0,
            "avg_worst_5_pct": 0,
            "all_rolling_pct": [],
        }

    # N-day rolling returns on calendar days
    rolling_ret = (eq[window:] / eq[:-window] - 1) * 100

    worst_idx = np.argmin(rolling_ret)
    worst_pct = float(rolling_ret[worst_idx])
    dates = daily.index
    worst_start = str(dates[worst_idx])[:10]
    worst_end = str(dates[worst_idx + window])[:10]

    # Top 5 worst
    sorted_ret = sorted(rolling_ret)
    avg_worst_5 = float(np.mean(sorted_ret[:5])) if len(sorted_ret) >= 5 else float(np.min(sorted_ret))

    return {
        "worst_return_pct": worst_pct,
        "worst_start_date": worst_start,
        "worst_end_date": worst_end,
        "worst_return_money": float(initial_capital * worst_pct / 100),
        "avg_worst_5_pct": avg_worst_5,
        "all_rolling_pct": rolling_ret.tolist(),
    }


def weekly_stress_report(
    equity_curve: list[dict],
    initial_capital: float = 100000,
) -> dict:
    """
    Weekly-level stress test — worst 1-day, 3-day, 5-day, 10-day rolling returns.

    Returns dict with keys: worst_1d, worst_3d, worst_5d, worst_10d
    Each is a dict from rolling_max_drawdown().
    """
    windows = [1, 3, 5, 10]
    result = {}
    names = {1: "worst_1d", 3: "worst_3d", 5: "worst_5d", 10: "worst_10d"}

    for w in windows:
        result[names[w]] = rolling_max_drawdown(equity_curve, window=w, initial_capital=initial_capital)

    return result


def format_weekly_stress(stress: dict) -> str:
    """Format weekly stress report as a readable table."""
    lines = ["极端滚动亏损分析", "-" * 72, 
             f"{'窗口':>8}  {'最差回报%':>10}  {'起始日':>12}  {'结束日':>12}  {'平均最差5次%':>14}",
             "-" * 72]

    for window_name in ["worst_1d", "worst_3d", "worst_5d", "worst_10d"]:
        d = stress.get(window_name, {})
        w_tag = window_name.replace("worst_", "")
        lines.append(
            f"{w_tag:>8}  {d.get('worst_return_pct', 0):>10.2f}  "
            f"{d.get('worst_start_date', '-')[:10]:>12}  "
            f"{d.get('worst_end_date', '-')[:10]:>12}  "
            f"{d.get('avg_worst_5_pct', 0):>14.2f}"
        )

    # Survival check
    worst_5d = stress.get("worst_5d", {}).get("worst_return_pct", 0)
    if abs(worst_5d) > 25:
        lines.append(f"\n!! 危险: 最差5日回撤 {abs(worst_5d):.1f}%, 超过25%阈值. "
                     "实盘中可能爆仓!")
    elif abs(worst_5d) > 15:
        lines.append(f"\n⚠ 警告: 最差5日回撤 {abs(worst_5d):.1f}%, 超过15%阈值. "
                     "需要缩小仓位.")
    else:
        lines.append(f"\n✓ 安全: 最差5日回撤 {abs(worst_5d):.1f}%, 在可接受范围内.")

    return "\n".join(lines)
