"""
Quant Analytics Package

A comprehensive analytics library for quantitative backtesting systems.
Provides performance metrics, risk analysis, statistical tests,
benchmark comparison, return attribution, and pretty-printing formatters.

All return values are in percent (e.g., 5.2 means 5.2%, not 0.052) unless
otherwise noted.
"""

# --- Performance Metrics ---
from .performance import (
    annual_return,
    cumulative_returns_series,
    monthly_returns_table,
    rolling_returns,
    total_return,
    yearly_returns_table,
)

# --- Risk Metrics ---
from .risk import (
    calmar_ratio,
    cvar_daily,
    cvar_historical,
    downside_deviation,
    drawdown_periods,
    drawdown_recovery_stats,
    max_consecutive_losses,
    max_consecutive_wins,
    max_drawdown,
    profit_distribution_stats,
    sharpe_ratio,
    sortino_ratio,
    ulcer_index,
    var_daily,
    var_historical,
    volatility_annualized,
)

# --- Statistical Tests ---
from .statistics import (
    bootstrap_confidence_interval,
    bootstrap_ev_ci,
    monte_carlo_simulation,
    normality_test,
    t_test_mean,
)

# --- Benchmark Comparison ---
from .benchmark import (
    alpha_beta,
    bull_bear_analysis,
    compare_to_benchmark,
    compute_benchmark_returns,
    excess_returns,
    information_ratio,
    tracking_error,
)

# --- Return Attribution ---
from .attribution import (
    monthly_heatmap,
    position_concentration,
    sector_attribution,
    sell_reason_attribution,
    yearly_attribution,
)

# --- Factor Analysis ---
from .factors import (
    capm_regression,
    factor_exposure_summary,
)

# --- Capacity & Turnover ---
from .capacity import (
    capacity_estimate,
    monthly_trade_frequency,
    turnover_analysis,
)

# --- Transaction Cost Analysis ---
from .tca import (
    cost_attribution,
    estimated_cost_breakdown,
)

# --- Strategy Correlation ---
from .correlation import (
    correlation_matrix,
    rolling_correlation,
)

# --- Scenario Stress Testing ---
from .scenario import (
    A_SHARE_SCENARIOS,
    scenario_stress_test,
    scenario_summary_table,
)

# --- Pretty Printers ---
from .formatters import (
    format_attribution_table,
    format_summary_table,
    format_trade_distribution,
    format_validation_report,
)

# --- Shared Utilities ---
from .common import (
    compute_daily_returns,
    extract_trade_returns,
    forward_fill_daily,
    to_equity_df,
)

__all__ = [
    # Performance
    "total_return",
    "annual_return",
    "monthly_returns_table",
    "yearly_returns_table",
    "rolling_returns",
    "cumulative_returns_series",
    # Risk
    "max_drawdown",
    "drawdown_periods",
    "drawdown_recovery_stats",
    "var_historical",
    "var_daily",
    "cvar_historical",
    "cvar_daily",
    "volatility_annualized",
    "downside_deviation",
    "sharpe_ratio",
    "sortino_ratio",
    "calmar_ratio",
    "max_consecutive_losses",
    "max_consecutive_wins",
    "profit_distribution_stats",
    "ulcer_index",
    # Statistics
    "bootstrap_confidence_interval",
    "bootstrap_ev_ci",
    "t_test_mean",
    "normality_test",
    "monte_carlo_simulation",
    # Benchmark
    "compute_benchmark_returns",
    "alpha_beta",
    "information_ratio",
    "tracking_error",
    "excess_returns",
    "compare_to_benchmark",
    "bull_bear_analysis",
    # Attribution
    "yearly_attribution",
    "sell_reason_attribution",
    "position_concentration",
    "sector_attribution",
    "monthly_heatmap",
    # Factors
    "capm_regression",
    "factor_exposure_summary",
    # Capacity
    "turnover_analysis",
    "capacity_estimate",
    "monthly_trade_frequency",
    # TCA
    "cost_attribution",
    "estimated_cost_breakdown",
    # Correlation
    "correlation_matrix",
    "rolling_correlation",
    # Scenario
    "scenario_stress_test",
    "scenario_summary_table",
    "A_SHARE_SCENARIOS",
    # Formatters
    "format_summary_table",
    "format_trade_distribution",
    "format_attribution_table",
    "format_validation_report",
    # Common
    "to_equity_df",
    "extract_trade_returns",
    "compute_daily_returns",
    "forward_fill_daily",
]
