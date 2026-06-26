"""
quant_validation — Quantitative strategy validation framework.

Provides:
  - Walk-forward analysis
  - Parameter sensitivity & stability testing
  - Monte Carlo simulation
"""

from validation.walk_forward import (
    walk_forward_split,
    run_walk_forward,
    walk_forward_summary_table,
)

from validation.sensitivity import (
    parameter_sweep,
    parameter_heatmap,
    parameter_stability_score,
)

from validation.monte_carlo import (
    bootstrap_trades,
    equity_curve_simulation,
    ruin_probability,
)

from validation.sample_split import (
    temporal_split,
    rolling_cv_with_test,
    overfitting_report,
    optimize_on_train_val,
)

__all__ = [
    # walk_forward
    "walk_forward_split",
    "run_walk_forward",
    "walk_forward_summary_table",
    # sensitivity
    "parameter_sweep",
    "parameter_heatmap",
    "parameter_stability_score",
    # monte_carlo
    "bootstrap_trades",
    "equity_curve_simulation",
    "ruin_probability",
    # sample_split
    "temporal_split",
    "rolling_cv_with_test",
    "overfitting_report",
    "optimize_on_train_val",
]
