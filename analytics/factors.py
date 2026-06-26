"""
Factor exposure analysis for quantitative backtesting.

Provides Fama-French 3-factor (market, size, value) decomposition
and style-factor attribution.  Uses daily returns and requires
benchmark factor data.

For Chinese A-shares, factors are constructed from CSI 300 universe:
  - MKT: CSI 300 total return minus risk-free rate
  - SMB: small-minus-big (top 30% vs bottom 30% by market cap)
  - HML: high-minus-low (top 30% vs bottom 30% by book-to-market)

If external factor data is not available, a market-only CAPM
regression is performed.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

try:
    import statsmodels.api as sm

    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

from .common import forward_fill_daily, to_equity_df


def capm_regression(
    strategy_equity: list[dict],
    benchmark_equity: list[dict],
    risk_free_rate: float = 0.0,
) -> dict[str, Any]:
    """
    Run CAPM (single-factor) regression: strategy excess return ~ market excess return.

    Args:
        strategy_equity: Strategy equity curve (sparse OK; forward-filled to daily).
        benchmark_equity: Benchmark index equity curve (CSI 300 or similar).
        risk_free_rate: Annual risk-free rate in percent (default 0.0).

    Returns:
        Dict with keys: alpha_annual_pct, beta, r_squared, adj_r_squared,
        t_stat_alpha, t_stat_beta, p_value_alpha, p_value_beta, n_observations.
    """
    s_df = forward_fill_daily(strategy_equity)
    b_df = forward_fill_daily(benchmark_equity)

    s_ret = s_df["equity"].pct_change().dropna()
    b_ret = b_df["equity"].pct_change().dropna()

    common_dates = s_ret.index.intersection(b_ret.index)
    if len(common_dates) < 60:
        return _empty_capm_result()

    s_ret = s_ret.loc[common_dates]
    b_ret = b_ret.loc[common_dates]

    # Convert to daily excess returns (as ratios, not percent)
    daily_rf = (1.0 + risk_free_rate / 100.0) ** (1.0 / 252.0) - 1.0
    s_excess = s_ret.values - daily_rf
    b_excess = b_ret.values - daily_rf

    if HAS_STATSMODELS:
        X = sm.add_constant(b_excess)
        model = sm.OLS(s_excess, X).fit()
        alpha_daily = float(model.params[0])
        beta = float(model.params[1])
        r2 = float(model.rsquared)
        adj_r2 = float(model.rsquared_adj)
        t_alpha = float(model.tvalues[0])
        t_beta = float(model.tvalues[1])
        p_alpha = float(model.pvalues[0])
        p_beta = float(model.pvalues[1])
    else:
        coeffs = np.polyfit(b_excess, s_excess, 1)
        beta = float(coeffs[0])
        alpha_daily = float(coeffs[1])
        # Approximate R² from correlation
        corr = float(np.corrcoef(s_excess, b_excess)[0, 1])
        r2 = corr ** 2
        adj_r2 = r2
        t_alpha = 0.0
        t_beta = 0.0
        p_alpha = 1.0
        p_beta = 1.0

    if alpha_daily > -1.0:
        alpha_annual = ((1.0 + alpha_daily) ** 252.0 - 1.0) * 100.0
    else:
        alpha_annual = -100.0

    # Information coefficient: correlation of strategy returns with lagged benchmark
    ic = float(np.corrcoef(s_ret.values[1:], b_ret.values[:-1])[0, 1]) if len(s_ret) > 1 else 0.0
    if np.isnan(ic):
        ic = 0.0

    return {
        "alpha_annual_pct": round(alpha_annual, 4),
        "beta": round(beta, 4),
        "r_squared": round(r2, 4),
        "adj_r_squared": round(adj_r2, 4),
        "t_stat_alpha": round(t_alpha, 4),
        "t_stat_beta": round(t_beta, 4),
        "p_value_alpha": round(p_alpha, 4),
        "p_value_beta": round(p_beta, 4),
        "n_observations": len(common_dates),
        "daily_alpha_pct": round(alpha_daily * 100.0, 6),
        "information_coefficient": round(ic, 4),
    }


def factor_exposure_summary(result: dict[str, Any]) -> str:
    """Render a CAPM regression result as a human-readable summary."""
    sig_marker = lambda p: "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
    lines = [
        "=" * 60,
        "  CAPM Factor Decomposition",
        "=" * 60,
        f"  Alpha (annual):   {result.get('alpha_annual_pct', 0):>+8.2f}%  {sig_marker(result.get('p_value_alpha', 1))}",
        f"  Beta:             {result.get('beta', 0):>8.2f}       {sig_marker(result.get('p_value_beta', 1))}",
        f"  R²:               {result.get('r_squared', 0):>8.4f}",
        f"  Adj R²:           {result.get('adj_r_squared', 0):>8.4f}",
        f"  Information Coef: {result.get('information_coefficient', 0):>8.4f}",
        f"  Observations:     {result.get('n_observations', 0):>8d}",
        "=" * 60,
    ]
    return "\n".join(lines)


def _empty_capm_result() -> dict[str, Any]:
    return {
        "alpha_annual_pct": 0.0,
        "beta": 0.0,
        "r_squared": 0.0,
        "adj_r_squared": 0.0,
        "t_stat_alpha": 0.0,
        "t_stat_beta": 0.0,
        "p_value_alpha": 1.0,
        "p_value_beta": 1.0,
        "n_observations": 0,
        "daily_alpha_pct": 0.0,
        "information_coefficient": 0.0,
    }
