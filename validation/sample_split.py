"""
Train / Validation / Test split for time-series backtesting.

Prevents overfitting by enforcing strict temporal separation:
  - Train:   used for parameter optimization
  - Validation: used for strategy selection (pick best config)
  - Test:    truly unseen, evaluated only ONCE for final report

For time-series data, splits are temporal (not random) to avoid
look-ahead bias.
"""

from __future__ import annotations

import logging
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)


# ── Temporal Split ───────────────────────────────────────────────

def temporal_split(
    start_date: str,
    end_date: str,
    train_ratio: float = 0.60,
    val_ratio: float = 0.20,
) -> Tuple[Tuple[str, str], Tuple[str, str], Tuple[str, str]]:
    """
    Split a date range into train / validation / test periods.

    Args:
        start_date: Overall range start (e.g. "2015-01-01").
        end_date:   Overall range end (e.g. "2024-12-31").
        train_ratio: Fraction for training (default 0.60).
        val_ratio:   Fraction for validation (default 0.20).
                     Test gets the remainder (1.0 - train - val).

    Returns:
        ((train_start, train_end), (val_start, val_end), (test_start, test_end))
    """
    from datetime import datetime

    sd = datetime.strptime(start_date, "%Y-%m-%d")
    ed = datetime.strptime(end_date, "%Y-%m-%d")
    total_days = (ed - sd).days

    if total_days < 365:
        raise ValueError(f"Date range too short for split: {total_days} days")

    train_days = int(total_days * train_ratio)
    val_days = int(total_days * val_ratio)

    train_end = sd + __import__("datetime").timedelta(days=train_days)
    val_end = train_end + __import__("datetime").timedelta(days=val_days)

    from datetime import timedelta

    train_end = sd + timedelta(days=train_days)
    val_end = train_end + timedelta(days=val_days)

    train_period = (start_date, train_end.strftime("%Y-%m-%d"))
    val_period = (train_end.strftime("%Y-%m-%d"), val_end.strftime("%Y-%m-%d"))
    test_period = (val_end.strftime("%Y-%m-%d"), end_date)

    return train_period, val_period, test_period


# ── Rolling Cross-Validation with Holdout Test ────────────────────

def rolling_cv_with_test(
    start_date: str,
    end_date: str,
    train_years: int = 5,
    val_years: int = 1,
    test_years: int = 2,
    step_years: int = 1,
) -> tuple[List[Tuple[Tuple[str, str], Tuple[str, str]]], Tuple[str, str]]:
    """
    Generate rolling train/val pairs, with the final test period
    completely held out.

    The test period is carved out from the END of the date range
    and never appears in any train or validation window.

    Args:
        start_date: Overall range start.
        end_date:   Overall range end.
        train_years: Training window length.
        val_years:   Validation window length (immediately after train).
        test_years:  Final test period length (carved from end).
        step_years:  Roll-forward step size.

    Returns:
        List of ((train_start, train_end), (val_start, val_end)),
        plus the test period is stored separately.
        The caller should also retrieve the test period via the
        companion function.
    """
    from datetime import datetime, timedelta

    sd = datetime.strptime(start_date, "%Y-%m-%d")
    ed = datetime.strptime(end_date, "%Y-%m-%d")

    # Carve out test period from end
    test_start = ed - timedelta(days=test_years * 365)
    usable_end = test_start - timedelta(days=1)

    windows = []
    offset_years = 0

    while True:
        train_start = sd + timedelta(days=offset_years * 365)
        train_end = train_start + timedelta(days=train_years * 365)
        val_start = train_end + timedelta(days=1)
        val_end = val_start + timedelta(days=val_years * 365) - timedelta(days=1)

        if val_end > usable_end:
            break

        windows.append((
            (train_start.strftime("%Y-%m-%d"), train_end.strftime("%Y-%m-%d")),
            (val_start.strftime("%Y-%m-%d"), val_end.strftime("%Y-%m-%d")),
        ))

        offset_years += step_years

    test_period = (test_start.strftime("%Y-%m-%d"), end_date)

    return windows, test_period


def get_test_period(start_date: str, end_date: str, test_years: int = 2) -> Tuple[str, str]:
    """Return the holdout test period carved from the end of the range."""
    from datetime import datetime, timedelta
    ed = datetime.strptime(end_date, "%Y-%m-%d")
    test_start = ed - timedelta(days=test_years * 365)
    return (test_start.strftime("%Y-%m-%d"), end_date)


# ── Overfitting Detection ─────────────────────────────────────────

def overfitting_report(
    train_metrics: dict,
    val_metrics: dict,
    test_metrics: dict | None = None,
) -> dict:
    """
    Quantify the overfitting gap between train, validation, and test.

    Args:
        train_metrics:  Dict with 'ev', 'sharpe', 'win_rate' from train period.
        val_metrics:    Same from validation period.
        test_metrics:   Optional, same from holdout test period.

    Returns:
        Dict with:
        - 'ev_decay_train_to_val': EV drop from train to validation (%)
        - 'ev_decay_val_to_test': EV drop from validation to test (%)
        - 'sharpe_decay': Sharpe decay
        - 'overfitting_score': 0-100 (100 = no overfitting)
        - 'verdict': human-readable verdict
        - 'details': breakdown of each metric's decay
    """
    train_ev = train_metrics.get("ev", 0)
    val_ev = val_metrics.get("ev", 1e-9)
    test_ev = test_metrics.get("ev", val_ev) if test_metrics else val_ev

    # EV decay ratios
    if train_ev != 0:
        ev_decay_tv = (train_ev - val_ev) / abs(train_ev) * 100.0
    else:
        ev_decay_tv = 0.0

    if val_ev != 0:
        ev_decay_vt = (val_ev - test_ev) / abs(val_ev) * 100.0
    else:
        ev_decay_vt = 0.0

    # Sharpe decay
    train_sh = train_metrics.get("sharpe", 0)
    val_sh = val_metrics.get("sharpe", train_sh)
    if abs(train_sh) > 1e-9:
        sharpe_decay = (train_sh - val_sh) / abs(train_sh) * 100.0
    else:
        sharpe_decay = 0.0

    # Overfitting score: 100 = no decay, 0 = complete decay
    # Weighted: EV decay 60%, Sharpe decay 40%
    ev_score = max(0.0, 100.0 - abs(ev_decay_tv))
    sh_score = max(0.0, 100.0 - abs(sharpe_decay))
    score = round(ev_score * 0.6 + sh_score * 0.4, 1)

    # Verdict
    if score >= 85:
        verdict = "轻微过拟合 — 策略稳健"
    elif score >= 60:
        verdict = "中等过拟合 — 需要参数正则化或简化策略"
    elif score >= 30:
        verdict = "严重过拟合 — 建议减少参数数量或增加训练数据"
    else:
        verdict = "极端过拟合 — 策略不可用"

    return {
        "ev_decay_train_to_val_pct": round(ev_decay_tv, 1),
        "ev_decay_val_to_test_pct": round(ev_decay_vt, 1) if test_metrics else None,
        "sharpe_decay_pct": round(sharpe_decay, 1),
        "overfitting_score": score,
        "verdict": verdict,
        "details": {
            "train_ev": round(train_ev, 4),
            "val_ev": round(val_ev, 4),
            "test_ev": round(test_ev, 4) if test_metrics else None,
            "train_sharpe": round(train_sh, 2),
            "val_sharpe": round(val_sh, 2),
        },
    }


# ── Parameter Optimization with Train/Val/Test ────────────────────

def optimize_on_train_val(
    strategy_config_dict: dict,
    stock_pool: list,
    start_date: str,
    end_date: str,
    param_name: str,
    param_values: list,
    train_ratio: float = 0.60,
    val_ratio: float = 0.20,
    metric: str = "ev",
) -> dict:
    """
    Parameter optimization with proper train/val split.

    For each parameter value:
    1. Run backtest on TRAIN period
    2. Run backtest on VALIDATION period with the same config
    3. Select the parameter that maximizes the validation metric
    4. Report overfitting gap

    Args:
        strategy_config_dict: Strategy config as dict.
        stock_pool: Stock pool codes.
        start_date, end_date: Overall date range.
        param_name: Parameter to optimize.
        param_values: Candidate values.
        train_ratio, val_ratio: Split ratios.
        metric: 'ev' or 'sharpe' — metric to maximize on validation.

    Returns:
        Dict with:
        - 'best_param': best parameter value
        - 'train_metrics': metrics on train for best param
        - 'val_metrics': metrics on validation for best param
        - 'overfitting': overfitting_report dict
        - 'all_results': list of all (param, train_metrics, val_metrics)
        - 'train_period', 'val_period', 'test_period': date ranges
    """
    import copy
    from backend.main import _config_from_dict
    from backend.backtest_engine import run_backtest

    train_period, val_period, test_period = temporal_split(
        start_date, end_date, train_ratio, val_ratio
    )

    pool = list(stock_pool) if stock_pool else strategy_config_dict.get("stock_pool", [])

    all_results = []
    best_val_metric = float("-inf")
    best_config = {}
    best_train = {}
    best_val = {}

    for val in param_values:
        cfg = copy.deepcopy(strategy_config_dict)
        cfg[param_name] = val
        if stock_pool:
            cfg["stock_pool"] = pool

        try:
            config = _config_from_dict(cfg)
        except Exception as exc:
            logger.warning("Config build failed for %s=%s: %s", param_name, val, exc)
            continue

        # Train
        try:
            train_res = run_backtest(config, start_date=train_period[0], end_date=train_period[1])
        except Exception as exc:
            logger.warning("Train backtest failed: %s", exc)
            continue

        # Validation
        try:
            val_res = run_backtest(config, start_date=val_period[0], end_date=val_period[1])
        except Exception as exc:
            logger.warning("Val backtest failed: %s", exc)
            continue

        train_m = {"ev": train_res.expected_value, "sharpe": train_res.sharpe_ratio,
                    "win_rate": train_res.win_rate, "trades": train_res.total_trades}
        val_m = {"ev": val_res.expected_value, "sharpe": val_res.sharpe_ratio,
                  "win_rate": val_res.win_rate, "trades": val_res.total_trades}

        all_results.append({
            "param_value": val,
            "train": train_m,
            "val": val_m,
        })

        val_metric = val_m.get(metric, val_m.get("ev", 0))
        if val_metric > best_val_metric:
            best_val_metric = val_metric
            best_config = {"param_value": val}
            best_train = train_m
            best_val = val_m

    of_report = overfitting_report(best_train, best_val)

    return {
        "param_name": param_name,
        "best_param": best_config.get("param_value"),
        "train_metrics": best_train,
        "val_metrics": best_val,
        "overfitting": of_report,
        "all_results": all_results,
        "train_period": train_period,
        "val_period": val_period,
        "test_period": test_period,
    }
