"""
Risk metrics module for quantitative backtesting.

Provides drawdown analysis, Value at Risk (VaR), volatility metrics,
and trade-level risk statistics. All return values are in percent
(e.g., 5.2 means 5.2%, not 0.052).
"""

import math
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


from .common import (
    to_equity_df as _to_equity_df,
    compute_daily_returns as _compute_daily_returns,
    extract_trade_returns as _extract_trade_returns,
)


def max_drawdown(
    equity_curve: list[dict],
) -> tuple[float, str, str, int]:
    """
    Calculate the maximum drawdown from an equity curve.

    Args:
        equity_curve: List of dicts with 'date' and 'equity' keys.

    Returns:
        Tuple of (max_dd_pct, start_date, end_date, duration_days).
        max_dd_pct is positive (e.g., 15.3 means a 15.3% drawdown).
    """
    if not equity_curve:
        return (0.0, "", "", 0)

    df = _to_equity_df(equity_curve)
    if len(df) < 2:
        return (0.0, "", "", 0)

    equity = df["equity"].values
    dates = df.index

    peak = equity[0]
    max_dd = 0.0
    peak_idx = 0
    trough_idx = 0
    current_peak_idx = 0

    for i in range(1, len(equity)):
        if equity[i] > peak:
            peak = equity[i]
            current_peak_idx = i
        dd = (peak - equity[i]) / peak * 100.0
        if dd > max_dd:
            max_dd = dd
            peak_idx = current_peak_idx
            trough_idx = i

    if max_dd == 0.0:
        return (0.0, "", "", 0)

    start_date = dates[peak_idx].strftime("%Y-%m-%d")
    end_date = dates[trough_idx].strftime("%Y-%m-%d")
    duration = (dates[trough_idx] - dates[peak_idx]).days

    return (round(float(max_dd), 4), start_date, end_date, duration)


def drawdown_periods(
    equity_curve: list[dict], threshold: float = 10.0
) -> list[dict[str, Any]]:
    """
    Identify all drawdown periods exceeding a given threshold.

    Args:
        equity_curve: List of dicts with 'date' and 'equity' keys.
        threshold: Drawdown threshold in percent (default 10.0 means 10%).

    Returns:
        List of dicts, each describing a drawdown event:
        {'start_date', 'end_date', 'peak', 'trough', 'max_dd_pct', 'duration_days'}.
    """
    if not equity_curve:
        return []

    df = _to_equity_df(equity_curve)
    if len(df) < 2:
        return []

    equity = df["equity"].values
    dates = df.index

    events: list[dict[str, Any]] = []
    peak = equity[0]
    peak_idx = 0
    in_drawdown = False
    dd_start_idx = 0
    dd_max = 0.0
    dd_trough_idx = 0

    for i in range(1, len(equity)):
        if equity[i] > peak:
            # New high - close any open drawdown
            if in_drawdown and dd_max >= threshold:
                events.append({
                    "start_date": dates[dd_start_idx].strftime("%Y-%m-%d"),
                    "end_date": dates[dd_trough_idx].strftime("%Y-%m-%d"),
                    "peak": round(float(equity[dd_start_idx]), 2),
                    "trough": round(float(equity[dd_trough_idx]), 2),
                    "max_dd_pct": round(float(dd_max), 4),
                    "duration_days": (dates[dd_trough_idx] - dates[dd_start_idx]).days,
                })
            peak = equity[i]
            peak_idx = i
            in_drawdown = False
            dd_max = 0.0
        else:
            dd = (peak - equity[i]) / peak * 100.0
            if dd > dd_max:
                if not in_drawdown:
                    dd_start_idx = peak_idx
                    in_drawdown = True
                dd_max = dd
                dd_trough_idx = i

    # Close any remaining drawdown at end
    if in_drawdown and dd_max >= threshold:
        events.append({
            "start_date": dates[dd_start_idx].strftime("%Y-%m-%d"),
            "end_date": dates[dd_trough_idx].strftime("%Y-%m-%d"),
            "peak": round(float(equity[dd_start_idx]), 2),
            "trough": round(float(equity[dd_trough_idx]), 2),
            "max_dd_pct": round(float(dd_max), 4),
            "duration_days": (dates[dd_trough_idx] - dates[dd_start_idx]).days,
        })

    return events


def var_historical(trades: list[dict], confidence: float = 0.95) -> float:
    """
    Calculate historical Value at Risk (VaR).

    Args:
        trades: List of trade dicts with 'profit_pct' key.
        confidence: Confidence level (default 0.95).

    Returns:
        VaR in percent (positive means loss, e.g., 3.0 means 3% VaR).
    """
    returns = _extract_trade_returns(trades)
    if len(returns) == 0:
        return 0.0
    # VaR is the loss at the (1-confidence) percentile
    # We return positive for loss
    var_val = -np.percentile(returns, (1.0 - confidence) * 100.0)
    return round(float(var_val), 4)


def cvar_historical(trades: list[dict], confidence: float = 0.95) -> float:
    """
    Calculate Conditional Value at Risk (Expected Shortfall).

    Args:
        trades: List of trade dicts with 'profit_pct' key.
        confidence: Confidence level (default 0.95).

    Returns:
        CVaR in percent (positive means loss).
    """
    returns = _extract_trade_returns(trades)
    if len(returns) == 0:
        return 0.0
    var_threshold = np.percentile(returns, (1.0 - confidence) * 100.0)
    tail = returns[returns <= var_threshold]
    if len(tail) == 0:
        return 0.0
    cvar_val = -float(tail.mean())
    return round(cvar_val, 4)


def volatility_annualized(daily_returns: list[float]) -> float:
    """
    Calculate annualized volatility from daily returns.

    Args:
        daily_returns: List of daily return values (in percent).

    Returns:
        Annualized volatility in percent.
    """
    if not daily_returns:
        return 0.0
    arr = np.array(daily_returns, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) < 2:
        return 0.0
    return round(float(np.std(arr, ddof=1) * np.sqrt(252)), 4)


def downside_deviation(daily_returns: list[float], mar: float = 0.0) -> float:
    """
    Calculate downside deviation (semi-deviation below minimum acceptable return).

    Args:
        daily_returns: List of daily return values (in percent).
        mar: Minimum acceptable return in percent (default 0).

    Returns:
        Downside deviation in percent, annualized.
    """
    if not daily_returns:
        return 0.0
    arr = np.array(daily_returns, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return 0.0

    below = arr[arr < mar]
    if len(below) == 0:
        return 0.0

    # Population-style for downside (use full N)
    squared = (below - mar) ** 2
    daily_downside = np.sqrt(np.mean(squared))
    return round(float(daily_downside * np.sqrt(252)), 4)


def sortino_ratio(
    equity_curve: list[dict], initial_capital: float, mar: float = 0.0
) -> float:
    """
    Calculate Sortino ratio using annualized return and downside deviation.

    Args:
        equity_curve: List of dicts with 'date' and 'equity' keys.
        initial_capital: Starting capital amount.
        mar: Minimum acceptable return in percent (annualized, default 0).

    Returns:
        Sortino ratio (float). Higher is better.
    """
    if not equity_curve or initial_capital <= 0:
        return 0.0

    # Need annual return
    from .performance import annual_return

    ann_ret = annual_return(equity_curve, initial_capital)
    daily_rets = _compute_daily_returns(equity_curve).tolist()
    dd = downside_deviation(daily_rets, mar / 252.0)  # Convert annual MAR to daily

    if dd == 0.0:
        return 0.0 if ann_ret <= mar else float("inf")

    # Account for risk-free rate
    return round((ann_ret - mar) / dd, 4)


def calmar_ratio(equity_curve: list[dict], initial_capital: float) -> float:
    """
    Calculate Calmar ratio (annualized return / maximum drawdown).

    Args:
        equity_curve: List of dicts with 'date' and 'equity' keys.
        initial_capital: Starting capital amount.

    Returns:
        Calmar ratio. Higher is better.
    """
    if not equity_curve or initial_capital <= 0:
        return 0.0

    from .performance import annual_return

    ann_ret = annual_return(equity_curve, initial_capital)
    max_dd_pct, _, _, _ = max_drawdown(equity_curve)

    if max_dd_pct == 0.0:
        return 0.0 if ann_ret <= 0 else float("inf")

    return round(ann_ret / max_dd_pct, 4)


def max_consecutive_losses(trades: list[dict]) -> tuple[int, float]:
    """
    Calculate the maximum consecutive losing trades streak.

    Args:
        trades: List of trade dicts with 'profit_pct' key.

    Returns:
        Tuple of (max_streak, total_loss_in_percent).
        total_loss is the sum of profit_pct during the streak (negative).
    """
    returns = _extract_trade_returns(trades)
    if len(returns) == 0:
        return (0, 0.0)

    max_streak = 0
    max_total_loss = 0.0
    current_streak = 0
    current_loss = 0.0

    for r in returns:
        if r < 0:
            current_streak += 1
            current_loss += r
        else:
            if current_streak > max_streak:
                max_streak = current_streak
                max_total_loss = current_loss
            elif current_streak == max_streak and current_loss < max_total_loss:
                max_total_loss = current_loss
            current_streak = 0
            current_loss = 0.0

    # Check final streak
    if current_streak > max_streak:
        max_streak = current_streak
        max_total_loss = current_loss
    elif current_streak == max_streak and current_loss < max_total_loss:
        max_total_loss = current_loss

    return (max_streak, round(float(max_total_loss), 4))


def max_consecutive_wins(trades: list[dict]) -> tuple[int, float]:
    """
    Calculate the maximum consecutive winning trades streak.

    Args:
        trades: List of trade dicts with 'profit_pct' key.

    Returns:
        Tuple of (max_streak, total_profit_in_percent).
    """
    returns = _extract_trade_returns(trades)
    if len(returns) == 0:
        return (0, 0.0)

    max_streak = 0
    max_total_profit = 0.0
    current_streak = 0
    current_profit = 0.0

    for r in returns:
        if r > 0:
            current_streak += 1
            current_profit += r
        else:
            if current_streak > max_streak:
                max_streak = current_streak
                max_total_profit = current_profit
            elif current_streak == max_streak and current_profit > max_total_profit:
                max_total_profit = current_profit
            current_streak = 0
            current_profit = 0.0

    # Check final streak
    if current_streak > max_streak:
        max_streak = current_streak
        max_total_profit = current_profit
    elif current_streak == max_streak and current_profit > max_total_profit:
        max_total_profit = current_profit

    return (max_streak, round(float(max_total_profit), 4))


def profit_distribution_stats(trades: list[dict]) -> dict[str, Any]:
    """
    Calculate descriptive statistics of trade profit distribution.

    Args:
        trades: List of trade dicts with 'profit_pct' key.

    Returns:
        Dict with keys: mean, std, skewness, kurtosis, min, max,
        p25, p50 (median), p75, p90, p95, p99, count.
        All in percent.
    """
    returns = _extract_trade_returns(trades)
    if len(returns) == 0:
        return {
            "mean": 0.0, "std": 0.0, "skewness": 0.0, "kurtosis": 0.0,
            "min": 0.0, "max": 0.0,
            "p25": 0.0, "p50": 0.0, "p75": 0.0,
            "p90": 0.0, "p95": 0.0, "p99": 0.0,
            "count": 0,
        }

    result = {
        "mean": round(float(np.mean(returns)), 4),
        "std": round(float(np.std(returns, ddof=1)), 4),
        "skewness": round(float(scipy_stats.skew(returns)), 4),
        "kurtosis": round(float(scipy_stats.kurtosis(returns, fisher=True)), 4),
        "min": round(float(np.min(returns)), 4),
        "max": round(float(np.max(returns)), 4),
        "p25": round(float(np.percentile(returns, 25)), 4),
        "p50": round(float(np.percentile(returns, 50)), 4),
        "p75": round(float(np.percentile(returns, 75)), 4),
        "p90": round(float(np.percentile(returns, 90)), 4),
        "p95": round(float(np.percentile(returns, 95)), 4),
        "p99": round(float(np.percentile(returns, 99)), 4),
        "count": len(returns),
    }
    return result


def ulcer_index(equity_curve: list[dict]) -> float:
    """
    Calculate the Ulcer Index, which measures depth and duration of drawdowns.

    Args:
        equity_curve: List of dicts with 'date' and 'equity' keys.

    Returns:
        Ulcer Index in percent (lower is better).
    """
    if not equity_curve:
        return 0.0

    df = _to_equity_df(equity_curve)
    if len(df) < 2:
        return 0.0

    equity = df["equity"].values
    n = len(equity)

    # Running max (peak)
    running_max = np.maximum.accumulate(equity)
    # Percentage drawdown from peak
    dd_pct = (running_max - equity) / running_max * 100.0
    # Ulcer Index = sqrt(mean of squared drawdowns)
    ui = np.sqrt(np.mean(dd_pct ** 2))
    return round(float(ui), 4)


# ── Equity-level (portfolio) VaR / CVaR ─────────────────────────
# These operate on the daily equity curve, producing proper
# portfolio-level risk measures, not trade-level percentiles.


def var_daily(equity_curve: list[dict], confidence: float = 0.95) -> float:
    """Calculate historical Value at Risk from DAILY equity returns.

    This is the correct portfolio-level VaR.  The legacy
    ``var_historical`` operates on trade-level returns and
    should be used for per-trade analysis only.

    Args:
        equity_curve: List of dicts with 'date' and 'equity' keys.
        confidence: Confidence level (default 0.95).

    Returns:
        VaR in percent (positive means loss, e.g., 3.0 means 3% daily VaR).
    """
    from .common import forward_fill_daily

    df = forward_fill_daily(equity_curve)
    if len(df) < 2:
        return 0.0
    daily_ret = df["equity"].pct_change().dropna() * 100.0
    if len(daily_ret) == 0:
        return 0.0
    var_val = -float(np.percentile(daily_ret.values, (1.0 - confidence) * 100.0))
    return round(var_val, 4)


def cvar_daily(equity_curve: list[dict], confidence: float = 0.95) -> float:
    """Calculate Conditional Value at Risk (Expected Shortfall) from
    DAILY equity returns.

    This is the correct portfolio-level CVaR.

    Args:
        equity_curve: List of dicts with 'date' and 'equity' keys.
        confidence: Confidence level (default 0.95).

    Returns:
        CVaR in percent (positive means loss).
    """
    from .common import forward_fill_daily

    df = forward_fill_daily(equity_curve)
    if len(df) < 2:
        return 0.0
    daily_ret = df["equity"].pct_change().dropna() * 100.0
    if len(daily_ret) == 0:
        return 0.0
    ret_vals = daily_ret.values
    var_threshold = np.percentile(ret_vals, (1.0 - confidence) * 100.0)
    tail = ret_vals[ret_vals <= var_threshold]
    if len(tail) == 0:
        return 0.0
    return round(-float(tail.mean()), 4)


# ── Sharpe Ratio ────────────────────────────────────────────────


def sharpe_ratio(
    equity_curve: list[dict],
    initial_capital: float,
    risk_free_rate: float = 0.0,
) -> float:
    """Calculate annualized Sharpe ratio from forward-filled daily returns.

    Uses proper daily-equity forward-fill so signal-mode equity curves
    with sparse dates still produce meaningful daily returns.

    Args:
        equity_curve: List of dicts with 'date' and 'equity' keys.
        initial_capital: Starting capital (used only if curve is empty).
        risk_free_rate: Annual risk-free rate in percent (default 0.0).

    Returns:
        Sharpe ratio (float).
    """
    from .common import forward_fill_daily

    df = forward_fill_daily(equity_curve)
    if len(df) < 2:
        return 0.0

    daily_ret = df["equity"].pct_change().dropna()
    if len(daily_ret) < 2:
        return 0.0

    # Convert annual risk-free to daily ratio
    daily_rf = (1.0 + risk_free_rate / 100.0) ** (1.0 / 252.0) - 1.0
    excess = daily_ret.values - daily_rf

    mean_excess = float(np.mean(excess))
    std_excess = float(np.std(excess, ddof=1))

    if std_excess == 0:
        return 0.0

    return round(mean_excess / std_excess * np.sqrt(252), 4)


# ── Drawdown Recovery Analysis ──────────────────────────────────


def drawdown_recovery_stats(
    equity_curve: list[dict],
) -> dict[str, Any]:
    """Analyse drawdown recovery characteristics.

    Returns:
        Dict with:
        - 'max_dd_pct': maximum drawdown (percent, positive).
        - 'max_dd_days': duration from peak to trough (calendar days).
        - 'max_recovery_days': days from trough back to previous peak
          (-1 if never recovered).
        - 'avg_recovery_days': mean recovery time for completed drawdowns.
        - 'drawdown_count': number of distinct drawdown events.
        - 'underwater_ratio': fraction of time spent in drawdown.
        - 'avg_drawdown_pct': average peak-to-trough drawdown across events.
    """
    from .common import forward_fill_daily

    df = forward_fill_daily(equity_curve)
    if len(df) < 2:
        return {
            "max_dd_pct": 0.0, "max_dd_days": 0, "max_recovery_days": -1,
            "avg_recovery_days": 0.0, "drawdown_count": 0,
            "underwater_ratio": 0.0, "avg_drawdown_pct": 0.0,
        }

    equity = df["equity"].values
    n = len(equity)

    peak = np.maximum.accumulate(equity)
    dd_pct = (peak - equity) / peak * 100.0

    # Detect drawdown events
    in_dd = False
    dd_start = 0
    dd_max = 0.0
    dd_trough = 0
    events = []

    for i in range(n):
        if dd_pct[i] > 0:
            if not in_dd:
                in_dd = True
                dd_start = i
                dd_max = dd_pct[i]
                dd_trough = i
            if dd_pct[i] > dd_max:
                dd_max = dd_pct[i]
                dd_trough = i
        else:
            if in_dd:
                # Find recovery point
                recovery = i
                for j in range(i, n):
                    if equity[j] >= peak[dd_start]:
                        recovery = j
                        break
                events.append({
                    "start": dd_start,
                    "trough": dd_trough,
                    "end": recovery if recovery > dd_trough else n - 1,
                    "max_dd_pct": round(float(dd_max), 4),
                    "trough_days": dd_trough - dd_start,
                    "recovery_days": recovery - dd_trough if recovery > dd_trough else -1,
                })
                in_dd = False

    # Close open drawdown at end
    if in_dd:
        events.append({
            "start": dd_start,
            "trough": dd_trough,
            "end": n - 1,
            "max_dd_pct": round(float(dd_max), 4),
            "trough_days": dd_trough - dd_start,
            "recovery_days": -1,
        })

    if not events:
        return {
            "max_dd_pct": 0.0, "max_dd_days": 0, "max_recovery_days": -1,
            "avg_recovery_days": 0.0, "drawdown_count": 0,
            "underwater_ratio": 0.0, "avg_drawdown_pct": 0.0,
        }

    max_dd_event = max(events, key=lambda e: e["max_dd_pct"])
    completed_recoveries = [e["recovery_days"] for e in events if e["recovery_days"] > 0]
    underwater_days = sum(e["end"] - e["start"] for e in events)

    return {
        "max_dd_pct": max_dd_event["max_dd_pct"],
        "max_dd_days": max_dd_event["trough_days"],
        "max_recovery_days": max_dd_event["recovery_days"],
        "avg_recovery_days": round(float(np.mean(completed_recoveries)), 1) if completed_recoveries else -1.0,
        "drawdown_count": len(events),
        "underwater_ratio": round(underwater_days / n * 100.0, 2) if n > 0 else 0.0,
        "avg_drawdown_pct": round(float(np.mean([e["max_dd_pct"] for e in events])), 4),
    }
