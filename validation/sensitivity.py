"""
Parameter sensitivity analysis.

Sweep single parameters, build 2-parameter heatmaps, and quantify
how sensitive strategy performance is around the optimal configuration.
"""

from __future__ import annotations

import copy
import logging
from typing import Any, List, Optional

logger = logging.getLogger(__name__)


def parameter_sweep(
    base_config_dict: dict,
    param_name: str,
    param_values: list,
    stock_pool: Optional[list] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> dict:
    """Run a one-dimensional parameter sweep.

    For each value in *param_values*, the parameter is injected into a
    deep-copy of *base_config_dict* and a full backtest is executed.
    Collected metrics: expected value (EV), win rate, total trades, Sharpe.

    Args:
        base_config_dict: Strategy config dict (same shape as a YAML file).
        param_name: Top-level key to vary (e.g. ``"stop_loss_pct"``).
        param_values: Values to test for the parameter.
        stock_pool: Optional override stock pool list.
        start_date: Optional backtest start date.
        end_date: Optional backtest end date.

    Returns:
        Dict with:
        * ``param_name`` — the name of the swept parameter.
        * ``results`` — list of per-value result dicts, each containing
          ``param_value``, ``ev``, ``win_rate``, ``total_trades``, ``sharpe``.
        * ``optimal`` — dict with ``param_value`` and metrics for the
          value that maximised EV.
        * ``best_ev`` — the maximum EV found.
    """
    from backend.main import _config_from_dict
    from backend.backtest_engine import run_backtest

    pool = list(stock_pool) if stock_pool else base_config_dict.get("stock_pool", [])

    results: list[dict] = []
    best_ev = float("-inf")
    optimal: dict = {}

    total = len(param_values)
    for i, val in enumerate(param_values):
        logger.info("Sweep [%d/%d] %s = %s", i + 1, total, param_name, val)

        cfg_dict = copy.deepcopy(base_config_dict)
        cfg_dict[param_name] = val
        if stock_pool is not None:
            cfg_dict["stock_pool"] = pool

        try:
            config = _config_from_dict(cfg_dict)
        except Exception as exc:
            logger.warning("Config build failed for %s=%s: %s", param_name, val, exc)
            results.append({
                "param_value": val,
                "ev": None,
                "win_rate": None,
                "total_trades": 0,
                "sharpe": None,
                "error": str(exc),
            })
            continue

        try:
            bt = run_backtest(config, start_date=start_date, end_date=end_date)
        except Exception as exc:
            logger.warning("Backtest failed for %s=%s: %s", param_name, val, exc)
            results.append({
                "param_value": val,
                "ev": None,
                "win_rate": None,
                "total_trades": 0,
                "sharpe": None,
                "error": str(exc),
            })
            continue

        entry = {
            "param_value": val,
            "ev": bt.expected_value,
            "win_rate": bt.win_rate,
            "total_trades": bt.total_trades,
            "sharpe": bt.sharpe_ratio,
        }
        results.append(entry)

        if bt.expected_value > best_ev:
            best_ev = bt.expected_value
            optimal = dict(entry)

    return {
        "param_name": param_name,
        "results": results,
        "optimal": optimal,
        "best_ev": best_ev if best_ev != float("-inf") else None,
    }


def parameter_heatmap(
    config_dict: dict,
    param1_name: str,
    param1_values: list,
    param2_name: str,
    param2_values: list,
    stock_pool: Optional[list] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> dict:
    """Build a 2-D EV heatmap by sweeping two parameters together.

    Args:
        config_dict: Strategy config dict.
        param1_name: First parameter key (rows).
        param1_values: Values for the first parameter (rows).
        param2_name: Second parameter key (columns).
        param2_values: Values for the second parameter (columns).
        stock_pool: Optional stock pool override.
        start_date: Optional backtest start.
        end_date: Optional backtest end.

    Returns:
        Dict with:
        * ``param1_name``, ``param2_name``
        * ``param1_values``, ``param2_values``
        * ``grid`` — list of list of EV floats (row-major).
        * ``best_ev`` — highest EV found.
        * ``best_combo`` — (p1_val, p2_val) that produced best_ev.
    """
    from backend.main import _config_from_dict
    from backend.backtest_engine import run_backtest

    pool = list(stock_pool) if stock_pool else config_dict.get("stock_pool", [])

    grid: list[list] = []
    best_ev = float("-inf")
    best_combo = (None, None)
    total = len(param1_values) * len(param2_values)
    count = 0

    for v1 in param1_values:
        row: list = []
        for v2 in param2_values:
            count += 1
            logger.info("Heatmap [%d/%d] %s=%s, %s=%s",
                        count, total, param1_name, v1, param2_name, v2)

            cfg_dict = copy.deepcopy(config_dict)
            cfg_dict[param1_name] = v1
            cfg_dict[param2_name] = v2
            if stock_pool is not None:
                cfg_dict["stock_pool"] = pool

            try:
                config = _config_from_dict(cfg_dict)
            except Exception as exc:
                logger.warning("Config build failed: %s", exc)
                row.append(None)
                continue

            try:
                bt = run_backtest(config, start_date, end_date)
                ev = bt.expected_value
            except Exception as exc:
                logger.warning("Backtest failed: %s", exc)
                ev = None

            row.append(ev)
            if ev is not None and ev > best_ev:
                best_ev = ev
                best_combo = (v1, v2)

        grid.append(row)

    return {
        "param1_name": param1_name,
        "param2_name": param2_name,
        "param1_values": param1_values,
        "param2_values": param2_values,
        "grid": grid,
        "best_ev": best_ev if best_ev != float("-inf") else None,
        "best_combo": list(best_combo),
    }


def parameter_stability_score(
    results: dict,
    ev_key: str = "ev",
) -> float:
    """Compute a 0–100 score measuring how stable EV is around its optimum.

    The score is derived from the coefficient of variation of the top-N
    values (where N = max(3, 20% of results)).  A low CV yields a high
    stability score (the optimum isn't a fragile spike).

    Args:
        results: The dict returned by ``parameter_sweep``.
        ev_key: Key inside each result entry holding EV (default ``"ev"``).

    Returns:
        Float in [0, 100].  Higher = more stable / less fragile.
    """
    entries = results.get("results", [])
    if not entries:
        return 0.0

    # Collect valid EV values
    evs = [e[ev_key] for e in entries if e.get(ev_key) is not None]
    if len(evs) < 3:
        return 0.0

    # Sort descending and pick top-N
    evs_sorted = sorted(evs, reverse=True)
    n_top = max(3, int(len(evs) * 0.2))
    top_evs = evs_sorted[:n_top]

    mean = sum(top_evs) / len(top_evs)
    if mean == 0:
        return 0.0

    variance = sum((v - mean) ** 2 for v in top_evs) / len(top_evs)
    std = variance ** 0.5
    cv = std / abs(mean)  # coefficient of variation
    score = 100.0 * (1.0 - min(cv, 1.0))  # clamp CV at 1 → score floor at 0
    return round(score, 1)
