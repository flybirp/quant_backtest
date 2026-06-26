"""
Statistical tests and simulations module for quantitative backtesting.

Provides bootstrap confidence intervals, hypothesis tests,
normality checks, and Monte Carlo simulation.
All return values are in percent (e.g., 5.2 means 5.2%, not 0.052).
"""

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from .common import to_equity_df as _to_equity_df, extract_trade_returns as _extract_trade_returns


def bootstrap_confidence_interval(
    trades: list[dict],
    statistic: str = "mean",
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for a statistic of trade returns.

    Args:
        trades: List of trade dicts with 'profit_pct' key.
        statistic: 'mean' or 'median' (default 'mean').
        n_bootstrap: Number of bootstrap resamples (default 10000).
        confidence: Confidence level (default 0.95).

    Returns:
        Tuple of (lower_bound, upper_bound, point_estimate) in percent.
    """
    returns = _extract_trade_returns(trades)
    if len(returns) == 0:
        return (0.0, 0.0, 0.0)

    rng = np.random.RandomState(42)

    if statistic == "median":
        point_est = float(np.median(returns))
        stat_fn = np.median
    else:
        point_est = float(np.mean(returns))
        stat_fn = np.mean

    boot_stats = np.zeros(n_bootstrap)
    n = len(returns)
    for i in range(n_bootstrap):
        sample = returns[rng.randint(0, n, size=n)]
        boot_stats[i] = stat_fn(sample)

    alpha = (1.0 - confidence) / 2.0
    lower = float(np.percentile(boot_stats, alpha * 100.0))
    upper = float(np.percentile(boot_stats, (1.0 - alpha) * 100.0))

    return (round(lower, 4), round(upper, 4), round(point_est, 4))


def bootstrap_ev_ci(
    trades: list[dict], n_bootstrap: int = 10000
) -> tuple[float, float, float]:
    """
    Bootstrap confidence interval for expected value (mean) of trade returns.

    This is a convenience wrapper around bootstrap_confidence_interval with
    statistic='mean' and confidence=0.95.

    Args:
        trades: List of trade dicts with 'profit_pct' key.
        n_bootstrap: Number of bootstrap resamples (default 10000).

    Returns:
        Tuple of (lower_bound, upper_bound, mean) in percent.
    """
    return bootstrap_confidence_interval(
        trades, statistic="mean", n_bootstrap=n_bootstrap, confidence=0.95
    )


def t_test_mean(trades: list[dict]) -> tuple[float, float, bool]:
    """
    Perform a one-sample t-test to check if the mean trade return is
    significantly different from zero.

    Args:
        trades: List of trade dicts with 'profit_pct' key.

    Returns:
        Tuple of (t_statistic, p_value, significant_at_5pct).
    """
    returns = _extract_trade_returns(trades)
    if len(returns) < 2:
        return (0.0, 1.0, False)

    t_stat, p_value = scipy_stats.ttest_1samp(returns, 0.0)
    significant = p_value < 0.05

    return (
        round(float(t_stat), 4),
        round(float(p_value), 4),
        bool(significant),
    )


def normality_test(trades: list[dict]) -> tuple[float, float, bool]:
    """
    Test whether trade returns are normally distributed using
    the Shapiro-Wilk test (for n <= 5000) or D'Agostino-Pearson test.

    Args:
        trades: List of trade dicts with 'profit_pct' key.

    Returns:
        Tuple of (test_statistic, p_value, is_normal).
        is_normal is True if we fail to reject normality at alpha=0.05.
    """
    returns = _extract_trade_returns(trades)
    n = len(returns)

    if n < 3:
        return (0.0, 1.0, True)

    if n <= 5000:
        stat, p_value = scipy_stats.shapiro(returns)
    else:
        stat, p_value = scipy_stats.normaltest(returns)

    is_normal = p_value >= 0.05

    return (
        round(float(stat), 4),
        round(float(p_value), 4),
        bool(is_normal),
    )


def monte_carlo_simulation(
    equity_curve: list[dict],
    initial_capital: float,
    n_simulations: int = 1000,
    horizon_days: int = 252,
) -> dict[str, Any]:
    """
    Run Monte Carlo simulation of future equity paths based on historical
    daily returns (parametric -- assumes normal distribution).

    WARNING: This uses a normality assumption which underestimates tail risk.
    For a non-parametric approach that preserves the empirical return
    distribution (including skewness and fat tails), use
    ``validation.monte_carlo.equity_curve_simulation`` instead.

    Args:
        equity_curve: List of dicts with 'date' and 'equity' keys.
        initial_capital: Starting capital amount for simulation.
        n_simulations: Number of simulation paths (default 1000).
        horizon_days: Number of trading days to simulate (default 252).

    Returns:
        Dict with keys:
            - 'final_values': list of final equity values from each simulation.
            - 'mean_final': mean final equity.
            - 'median_final': median final equity.
            - 'ci_95_lower': 95% CI lower bound.
            - 'ci_95_upper': 95% CI upper bound.
            - 'mean_return_pct': mean return in percent.
            - 'median_return_pct': median return in percent.
            - 'paths': list of selected paths for visualization (max 100).
    """
    if not equity_curve or initial_capital <= 0:
        return {
            "final_values": [],
            "mean_final": initial_capital,
            "median_final": initial_capital,
            "ci_95_lower": initial_capital,
            "ci_95_upper": initial_capital,
            "mean_return_pct": 0.0,
            "median_return_pct": 0.0,
            "paths": [],
        }

    df = _to_equity_df(equity_curve)
    if len(df) < 2:
        return {
            "final_values": [initial_capital] * n_simulations,
            "mean_final": initial_capital,
            "median_final": initial_capital,
            "ci_95_lower": initial_capital,
            "ci_95_upper": initial_capital,
            "mean_return_pct": 0.0,
            "median_return_pct": 0.0,
            "paths": [],
        }

    daily_returns = df["equity"].pct_change().dropna().values
    if len(daily_returns) == 0:
        return {
            "final_values": [initial_capital] * n_simulations,
            "mean_final": initial_capital,
            "median_final": initial_capital,
            "ci_95_lower": initial_capital,
            "ci_95_upper": initial_capital,
            "mean_return_pct": 0.0,
            "median_return_pct": 0.0,
            "paths": [],
        }

    mean_ret = np.mean(daily_returns)
    std_ret = np.std(daily_returns, ddof=1)

    if std_ret == 0:
        final = initial_capital * (1.0 + mean_ret) ** horizon_days
        return {
            "final_values": [final] * n_simulations,
            "mean_final": final,
            "median_final": final,
            "ci_95_lower": final,
            "ci_95_upper": final,
            "mean_return_pct": round((final / initial_capital - 1.0) * 100.0, 4),
            "median_return_pct": round((final / initial_capital - 1.0) * 100.0, 4),
            "paths": [],
        }

    rng = np.random.RandomState(42)
    final_values = np.zeros(n_simulations)

    n_plot = min(n_simulations, 100)
    plot_paths = np.zeros((n_plot, horizon_days + 1))
    plot_paths[:, 0] = initial_capital

    for i in range(n_simulations):
        sim_returns = rng.normal(mean_ret, std_ret, horizon_days)
        path = initial_capital * np.cumprod(1.0 + sim_returns)
        final_values[i] = path[-1]

        if i < n_plot:
            plot_paths[i, 1:] = path

    mean_final = float(np.mean(final_values))
    median_final = float(np.median(final_values))
    ci_lower = float(np.percentile(final_values, 2.5))
    ci_upper = float(np.percentile(final_values, 97.5))

    paths_list = []
    for i in range(n_plot):
        paths_list.append([round(float(v), 2) for v in plot_paths[i]])

    return {
        "final_values": [round(float(v), 2) for v in final_values.tolist()],
        "mean_final": round(mean_final, 2),
        "median_final": round(median_final, 2),
        "ci_95_lower": round(ci_lower, 2),
        "ci_95_upper": round(ci_upper, 2),
        "mean_return_pct": round((mean_final / initial_capital - 1.0) * 100.0, 4),
        "median_return_pct": round((median_final / initial_capital - 1.0) * 100.0, 4),
        "paths": paths_list,
    }
