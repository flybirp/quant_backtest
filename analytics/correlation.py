"""
Inter-strategy correlation analysis.

Computes return correlations between multiple strategies to help
with portfolio construction and diversification assessment.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .common import forward_fill_daily


def correlation_matrix(
    strategy_curves: dict[str, list[dict]],
) -> dict[str, Any]:
    """
    Compute pairwise correlation matrix of strategy daily returns.

    Args:
        strategy_curves: Dict mapping strategy name to equity_curve
                         (list of dicts with 'date' and 'equity').

    Returns:
        Dict with:
        - 'matrix': nested dict {name1: {name2: corr}}.
        - 'mean_correlation': average pairwise correlation.
        - 'diversification_ratio': 1 - mean_correlation (crude).
        - 'names': ordered list of strategy names.
    """
    names = list(strategy_curves.keys())
    if len(names) < 2:
        return {
            "matrix": {},
            "mean_correlation": 0.0,
            "diversification_ratio": 0.0,
            "names": names,
        }

    # Build daily return DataFrames for each strategy
    ret_dfs = {}
    for name in names:
        df = forward_fill_daily(strategy_curves[name])
        if len(df) < 2:
            continue
        ret_dfs[name] = df["equity"].pct_change().dropna()

    if len(ret_dfs) < 2:
        return {
            "matrix": {},
            "mean_correlation": 0.0,
            "diversification_ratio": 0.0,
            "names": names,
        }

    # Align on common dates
    all_rets = pd.DataFrame(ret_dfs).dropna()
    if len(all_rets) < 30:
        return {
            "matrix": {},
            "mean_correlation": 0.0,
            "diversification_ratio": 0.0,
            "names": names,
        }

    corr = all_rets.corr()

    # Build nested dict
    matrix: dict[str, dict[str, float]] = {}
    for n1 in corr.columns:
        matrix[n1] = {}
        for n2 in corr.columns:
            val = corr.loc[n1, n2]
            matrix[n1][n2] = round(float(val) if not np.isnan(val) else 0.0, 4)

    # Mean off-diagonal correlation
    off_diag = []
    for i, n1 in enumerate(corr.columns):
        for j, n2 in enumerate(corr.columns):
            if i < j:
                val = corr.loc[n1, n2]
                if not np.isnan(val):
                    off_diag.append(val)

    mean_corr = float(np.mean(off_diag)) if off_diag else 0.0

    return {
        "matrix": matrix,
        "mean_correlation": round(mean_corr, 4),
        "diversification_ratio": round(1.0 - abs(mean_corr), 4),
        "names": list(corr.columns),
    }


def rolling_correlation(
    strategy_curve_a: list[dict],
    strategy_curve_b: list[dict],
    window_days: int = 60,
) -> list[dict[str, Any]]:
    """
    Compute rolling correlation between two strategies' daily returns.

    Args:
        strategy_curve_a: Equity curve for strategy A.
        strategy_curve_b: Equity curve for strategy B.
        window_days: Rolling window in calendar days.

    Returns:
        List of dicts with 'date' and 'correlation'.
    """
    a_df = forward_fill_daily(strategy_curve_a)
    b_df = forward_fill_daily(strategy_curve_b)

    a_ret = a_df["equity"].pct_change().dropna()
    b_ret = b_df["equity"].pct_change().dropna()

    common = a_ret.index.intersection(b_ret.index)
    if len(common) < window_days:
        return []

    a_ret = a_ret.loc[common]
    b_ret = b_ret.loc[common]

    rolling_corr = a_ret.rolling(window=window_days).corr(b_ret).dropna()

    result = []
    for dt, val in rolling_corr.items():
        if not np.isnan(val):
            result.append({
                "date": dt.strftime("%Y-%m-%d"),
                "correlation": round(float(val), 4),
            })

    return result
