"""
Benchmark comparison module for quantitative backtesting.

Compares strategy performance against a benchmark index.
Uses statsmodels for linear regression (alpha/beta calculation).
All return values are in percent (e.g., 5.2 means 5.2%, not 0.052).
"""

import math
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

try:
    import statsmodels.api as sm
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False


from .common import to_equity_df as _to_equity_df


def _align_returns(
    strategy_equity: list[dict],
    benchmark_equity: list[dict],
) -> tuple[pd.Series, pd.Series]:
    """
    Align strategy and benchmark daily returns by date.

    Returns:
        Tuple of (strategy_returns_pct, benchmark_returns_pct) as aligned Series.
    """
    s_df = _to_equity_df(strategy_equity)
    b_df = _to_equity_df(benchmark_equity)

    if s_df.empty or b_df.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    # Remove duplicate dates: take last equity value per day
    if s_df.index.duplicated().any():
        s_df = s_df[~s_df.index.duplicated(keep="last")]
    if b_df.index.duplicated().any():
        b_df = b_df[~b_df.index.duplicated(keep="last")]

    # Align on dates: use benchmark dates as reference, forward-fill strategy equity
    s_aligned = s_df["equity"].reindex(b_df.index, method="ffill").dropna()
    b_aligned = b_df["equity"].reindex(b_df.index).dropna()

    # Now both have same index (benchmark dates where strategy has data)
    common_idx = s_aligned.index.intersection(b_aligned.index)
    if len(common_idx) <= 1:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    s_ret = s_aligned.loc[common_idx].pct_change().dropna() * 100.0
    b_ret = b_aligned.loc[common_idx].pct_change().dropna() * 100.0

    # Final alignment after pct_change drops one row
    final_dates = s_ret.index.intersection(b_ret.index)
    if len(final_dates) == 0:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    return s_ret.loc[final_dates], b_ret.loc[final_dates]


def compute_benchmark_returns(
    benchmark_equity: list[dict], initial_value: float = 1.0
) -> dict[str, Any]:
    """
    Compute benchmark performance metrics.

    Args:
        benchmark_equity: List of dicts with 'date' and 'equity' keys.
        initial_value: Starting value of the benchmark (default 1.0).

    Returns:
        Dict with keys: total_return_pct, annual_return_pct, volatility_pct,
        max_drawdown_pct, sharpe_ratio (assuming 0 risk-free rate),
        start_date, end_date.
    """
    if not benchmark_equity:
        return {
            "total_return_pct": 0.0,
            "annual_return_pct": 0.0,
            "volatility_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "sharpe_ratio": 0.0,
            "start_date": "",
            "end_date": "",
        }

    df = _to_equity_df(benchmark_equity)
    if df.empty or len(df) < 2:
        return {
            "total_return_pct": 0.0,
            "annual_return_pct": 0.0,
            "volatility_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "sharpe_ratio": 0.0,
            "start_date": "",
            "end_date": "",
        }

    final_val = float(df["equity"].iloc[-1])
    total_ret = (final_val - initial_value) / initial_value * 100.0

    start_date = df.index[0]
    end_date = df.index[-1]
    years = max((end_date - start_date).days / 365.25, 0.001)
    cagr = (((final_val / initial_value) ** (1.0 / years)) - 1.0) * 100.0

    daily_ret = df["equity"].pct_change().dropna() * 100.0
    vol = float(daily_ret.std(ddof=1) * np.sqrt(252)) if len(daily_ret) > 1 else 0.0

    # Max drawdown
    equity = df["equity"].values
    peak = np.maximum.accumulate(equity)
    dd = (peak - equity) / peak * 100.0
    max_dd = float(dd.max())

    sharpe = cagr / vol if vol > 0 else 0.0

    return {
        "total_return_pct": round(total_ret, 4),
        "annual_return_pct": round(cagr, 4),
        "volatility_pct": round(vol, 4),
        "max_drawdown_pct": round(max_dd, 4),
        "sharpe_ratio": round(sharpe, 4),
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
    }


def alpha_beta(
    strategy_returns: list[float], benchmark_returns: list[float]
) -> tuple[float, float]:
    """
    Calculate alpha and beta using linear regression.

    Alpha = intercept (strategy excess return over benchmark).
    Beta = slope (strategy sensitivity to benchmark).

    Args:
        strategy_returns: List of strategy daily returns (in percent).
        benchmark_returns: List of benchmark daily returns (in percent).

    Returns:
        Tuple of (alpha, beta) in percent. Alpha is annualized.
    """
    s_arr = np.array(strategy_returns, dtype=float)
    b_arr = np.array(benchmark_returns, dtype=float)

    # Remove NaN pairs
    mask = ~(np.isnan(s_arr) | np.isnan(b_arr))
    s_arr = s_arr[mask]
    b_arr = b_arr[mask]

    if len(s_arr) < 2 or len(b_arr) < 2:
        return (0.0, 0.0)

    if HAS_STATSMODELS:
        X = sm.add_constant(b_arr)
        model = sm.OLS(s_arr, X).fit()
        alpha_daily = float(model.params[0])
        beta = float(model.params[1])
    else:
        # Fallback using numpy polyfit: s_arr = beta * b_arr + alpha_daily
        coeffs = np.polyfit(b_arr, s_arr, 1)
        beta = float(coeffs[0])
        alpha_daily = float(coeffs[1])

    # Annualize alpha: cap daily alpha ratio to prevent overflow
    # Floor at -99%, ceiling at +99% (extreme values come from sparse data)
    daily_alpha_ratio = max(-0.99, min(0.99, alpha_daily / 100.0))
    alpha_annual = ((1.0 + daily_alpha_ratio) ** 252.0 - 1.0) * 100.0

    return (round(alpha_annual, 4), round(beta, 4))


def information_ratio(
    strategy_returns: list[float], benchmark_returns: list[float]
) -> float:
    """
    Calculate Information Ratio (mean excess return / std of excess returns).

    Args:
        strategy_returns: List of strategy daily returns (in percent).
        benchmark_returns: List of benchmark daily returns (in percent).

    Returns:
        Information ratio (annualized).
    """
    excess = excess_returns(strategy_returns, benchmark_returns)
    if not excess:
        return 0.0

    arr = np.array(excess, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) < 2:
        return 0.0

    mean_excess = float(np.mean(arr))
    std_excess = float(np.std(arr, ddof=1))

    if std_excess == 0:
        return 0.0 if mean_excess <= 0 else float("inf")

    # Annualize
    ir = (mean_excess / std_excess) * np.sqrt(252)
    return round(ir, 4)


def tracking_error(
    strategy_returns: list[float], benchmark_returns: list[float]
) -> float:
    """
    Calculate tracking error (std deviation of excess returns, annualized).

    Args:
        strategy_returns: List of strategy daily returns (in percent).
        benchmark_returns: List of benchmark daily returns (in percent).

    Returns:
        Tracking error in percent (annualized).
    """
    excess = excess_returns(strategy_returns, benchmark_returns)
    if not excess:
        return 0.0

    arr = np.array(excess, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) < 2:
        return 0.0

    te = float(np.std(arr, ddof=1)) * np.sqrt(252)
    return round(te, 4)


def excess_returns(
    strategy_returns: list[float], benchmark_returns: list[float]
) -> list[float]:
    """
    Calculate excess returns (strategy - benchmark) for each period.

    Args:
        strategy_returns: List of strategy returns (in percent).
        benchmark_returns: List of benchmark returns (in percent).

    Returns:
        List of excess returns in percent.
    """
    if not strategy_returns or not benchmark_returns:
        return []

    n = min(len(strategy_returns), len(benchmark_returns))
    result = []
    for i in range(n):
        sr = strategy_returns[i]
        br = benchmark_returns[i]
        if np.isnan(sr) or np.isnan(br):
            result.append(0.0)
        else:
            result.append(round(float(sr) - float(br), 4))
    return result


def compare_to_benchmark(
    strategy_equity: list[dict],
    benchmark_equity: list[dict],
    initial_capital: float,
) -> dict[str, Any]:
    """
    Comprehensive comparison of strategy vs benchmark.

    Args:
        strategy_equity: List of dicts with 'date' and 'equity' keys.
        benchmark_equity: List of dicts with 'date' and 'equity' keys.
        initial_capital: Starting capital amount.

    Returns:
        Dict with comprehensive comparison metrics including:
        strategy_metrics, benchmark_metrics, alpha, beta, information_ratio,
        tracking_error, capture_ratios, correlation, r_squared.
    """
    from .performance import total_return, annual_return, monthly_returns_table
    from .risk import max_drawdown, volatility_annualized

    result: dict[str, Any] = {}

    # Strategy metrics
    result["strategy"] = {
        "total_return_pct": round(total_return(strategy_equity, initial_capital), 4),
        "annual_return_pct": round(annual_return(strategy_equity, initial_capital), 4),
    }
    max_dd_pct, dd_start, dd_end, dd_dur = max_drawdown(strategy_equity)
    result["strategy"]["max_drawdown_pct"] = max_dd_pct
    result["strategy"]["max_drawdown_duration"] = dd_dur

    # Benchmark metrics
    s_df = _to_equity_df(strategy_equity)
    b_df = _to_equity_df(benchmark_equity)
    if not b_df.empty and len(b_df) > 1:
        benchmark_start_val = float(b_df["equity"].iloc[0])
        bench_metrics = compute_benchmark_returns(benchmark_equity, benchmark_start_val if benchmark_start_val > 0 else 1.0)
        result["benchmark"] = bench_metrics
    else:
        result["benchmark"] = {
            "total_return_pct": 0.0, "annual_return_pct": 0.0,
            "volatility_pct": 0.0, "max_drawdown_pct": 0.0,
            "sharpe_ratio": 0.0, "start_date": "", "end_date": "",
        }

    # Aligned returns
    s_ret, b_ret = _align_returns(strategy_equity, benchmark_equity)
    if len(s_ret) > 1 and len(b_ret) > 1:
        s_list = s_ret.tolist()
        b_list = b_ret.tolist()

        alpha, beta = alpha_beta(s_list, b_list)
        result["alpha"] = alpha
        result["beta"] = beta

        result["information_ratio"] = information_ratio(s_list, b_list)
        result["tracking_error"] = tracking_error(s_list, b_list)

        # Correlation
        corr = float(np.corrcoef(s_list, b_list)[0, 1])
        if np.isnan(corr):
            corr = 0.0
        result["correlation"] = round(corr, 4)
        result["r_squared"] = round(corr ** 2, 4)

        # Upside/downside capture
        up_mask = np.array(b_list) > 0
        down_mask = np.array(b_list) < 0
        if up_mask.any():
            up_capture = float(np.mean(np.array(s_list)[up_mask]) / np.mean(np.array(b_list)[up_mask]) * 100.0)
            if np.isnan(up_capture):
                up_capture = 0.0
        else:
            up_capture = 0.0
        if down_mask.any():
            down_capture = float(np.mean(np.array(s_list)[down_mask]) / np.mean(np.array(b_list)[down_mask]) * 100.0)
            if np.isnan(down_capture):
                down_capture = 0.0
        else:
            down_capture = 0.0

        result["upside_capture_pct"] = round(up_capture, 4)
        result["downside_capture_pct"] = round(down_capture, 4)
    else:
        result["alpha"] = 0.0
        result["beta"] = 0.0
        result["information_ratio"] = 0.0
        result["tracking_error"] = 0.0
        result["correlation"] = 0.0
        result["r_squared"] = 0.0
        result["upside_capture_pct"] = 0.0
        result["downside_capture_pct"] = 0.0

    # Monthly returns comparison
    monthly_s = monthly_returns_table(strategy_equity)
    result["monthly_returns"] = {
        "strategy": monthly_s,
    }

    return result


def bull_bear_analysis(
    trades_by_date: list[dict],
    benchmark_index: list[dict],
) -> dict[str, Any]:
    """
    Split trades into bull and bear market periods based on benchmark performance.

    A period is considered 'bull' if the benchmark's cumulative return over the
    trade's holding period is positive, 'bear' otherwise.

    Args:
        trades_by_date: List of trade dicts with 'profit_pct', 'buy_date', 'sell_date'.
        benchmark_index: List of dicts with 'date' and 'equity' keys.

    Returns:
        Dict with 'bull' and 'bear' keys, each containing:
        {'trades', 'win_rate_pct', 'avg_return_pct', 'total_return_pct', 'count'}.
    """
    if not trades_by_date or not benchmark_index:
        return {
            "bull": {"trades": 0, "win_rate_pct": 0.0, "avg_return_pct": 0.0, "total_return_pct": 0.0, "count": 0},
            "bear": {"trades": 0, "win_rate_pct": 0.0, "avg_return_pct": 0.0, "total_return_pct": 0.0, "count": 0},
        }

    b_df = _to_equity_df(benchmark_index)
    if b_df.empty:
        return {
            "bull": {"trades": 0, "win_rate_pct": 0.0, "avg_return_pct": 0.0, "total_return_pct": 0.0, "count": 0},
            "bear": {"trades": 0, "win_rate_pct": 0.0, "avg_return_pct": 0.0, "total_return_pct": 0.0, "count": 0},
        }

    bull_returns = []
    bear_returns = []

    for trade in trades_by_date:
        buy_date = trade.get("buy_date", "")
        sell_date = trade.get("sell_date", "")
        profit = trade.get("profit_pct", 0.0)
        if np.isnan(profit):
            continue

        # Determine market regime during holding period
        try:
            buy_dt = pd.to_datetime(buy_date)
            sell_dt = pd.to_datetime(sell_date)

            # Find closest benchmark values
            bench_slice = b_df.loc[buy_dt:sell_dt]
            if len(bench_slice) >= 2:
                bench_ret = (bench_slice["equity"].iloc[-1] / bench_slice["equity"].iloc[0] - 1.0) * 100.0
            elif len(bench_slice) == 1:
                bench_ret = 0.0
            else:
                # No exact match, check broader
                before_mask = b_df.index <= sell_dt
                after_mask = b_df.index >= buy_dt
                if before_mask.any() and after_mask.any():
                    start_val = b_df.loc[after_mask, "equity"].iloc[0]
                    end_val = b_df.loc[before_mask, "equity"].iloc[-1]
                    bench_ret = (end_val / start_val - 1.0) * 100.0
                else:
                    bench_ret = 0.0
        except Exception:
            bench_ret = 0.0

        if bench_ret >= 0:
            bull_returns.append(profit)
        else:
            bear_returns.append(profit)

    def _summarize(returns: list[float]) -> dict:
        if not returns:
            return {"trades": 0, "win_rate_pct": 0.0, "avg_return_pct": 0.0, "total_return_pct": 0.0, "count": 0}
        wins = sum(1 for r in returns if r > 0)
        return {
            "trades": len(returns),
            "win_rate_pct": round(wins / len(returns) * 100.0, 2),
            "avg_return_pct": round(float(np.mean(returns)), 4),
            "total_return_pct": round(float(np.sum(returns)), 4),
            "count": len(returns),
        }

    return {
        "bull": _summarize(bull_returns),
        "bear": _summarize(bear_returns),
    }
