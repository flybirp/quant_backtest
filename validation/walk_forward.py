"""
Walk-forward analysis framework.

Splits a date range into sequential training/test windows, runs the backtest
engine on each test window, and aggregates results to evaluate strategy
robustness and parameter stability over time.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta
from typing import List, Tuple

logger = logging.getLogger(__name__)


def walk_forward_split(
    start_date: str,
    end_date: str,
    train_years: int = 5,
    test_years: int = 1,
    step_years: int = 1,
) -> List[Tuple[str, str, str, str]]:
    """Generate walk-forward window tuples: (train_start, train_end, test_start, test_end).

    Windows slide forward by ``step_years`` years. The first window uses
    ``start_date`` as the beginning of the train period; subsequent training
    windows grow by ``step_years`` on the left as well.

    Args:
        start_date: Overall range start, e.g. ``"2015-01-01"``.
        end_date:   Overall range end, e.g. ``"2024-12-31"``.
        train_years: Length of each training window in years (default 5).
        test_years:  Length of each test window in years (default 1).
        step_years:  Amount the window advances each step in years (default 1).

    Returns:
        List of (train_start, train_end, test_start, test_end) date strings.

    Raises:
        ValueError: If the range is too short for even one window.
    """
    sd = datetime.strptime(start_date, "%Y-%m-%d")
    ed = datetime.strptime(end_date, "%Y-%m-%d")

    if (ed - sd).days < 365 * (train_years + test_years):
        raise ValueError(
            f"Date range too short for walk-forward: {start_date} → {end_date} "
            f"(need at least {train_years + test_years} years)"
        )

    windows: List[Tuple[str, str, str, str]] = []
    offset_years = 0
    step_days = 365 * step_years

    while True:
        train_start = sd + timedelta(days=offset_years * 365)
        train_end = train_start + timedelta(days=train_years * 365)
        test_start = train_end + timedelta(days=1)
        test_end = test_start + timedelta(days=test_years * 365) - timedelta(days=1)

        # Clamp test_end to overall end_date
        if test_end > ed:
            if test_start >= ed:
                break
            test_end = ed

        windows.append((
            train_start.strftime("%Y-%m-%d"),
            train_end.strftime("%Y-%m-%d"),
            test_start.strftime("%Y-%m-%d"),
            test_end.strftime("%Y-%m-%d"),
        ))

        offset_years += step_years
        # Safety: break if test_start passes end_date
        if test_start >= ed:
            break

    if not windows:
        raise ValueError("Could not generate any walk-forward windows with the given parameters.")

    logger.info("Walk-forward split produced %d windows.", len(windows))
    return windows


def run_walk_forward(
    strategy_config_dict: dict,
    stock_pool: list,
    start_date: str,
    end_date: str,
    train_years: int = 5,
    test_years: int = 1,
) -> dict:
    """Execute walk-forward analysis by running one backtest per test window.

    The underlying ``run_backtest`` is called for each test window. The
    training portion of each window is provided for future parameter-
    optimisation hooks, but the current implementation simply passes the
    supplied config to each test run.

    Args:
        strategy_config_dict: Strategy config as a plain dict (same shape as a
            YAML strategy file).
        stock_pool: List of stock codes to include.
        start_date: Overall range start.
        end_date: Overall range end.
        train_years: Training window length in years.
        test_years: Test window length in years.

    Returns:
        Dict with keys:
        * ``windows`` — list of per-window results (each a dict with train &
          test date ranges, backtest summary, and trade count).
        * ``summary`` — aggregate stats across all windows: avg EV, avg
          win_rate, avg sharpe, total trades, consistency scores.
        * ``stability`` — walk-forward efficiency (WFE) and other stability
          metrics.

        Returns empty defaults on failure.
    """
    # --- Local imports to avoid circular dependencies ---
    from backend.main import _config_from_dict
    from backend.backtest_engine import run_backtest

    # --- Resolve pool ---
    pool = list(stock_pool) if stock_pool else []

    # --- Generate windows ---
    try:
        windows = walk_forward_split(start_date, end_date, train_years, test_years)
    except ValueError as exc:
        logger.error("walk_forward_split failed: %s", exc)
        return {
            "windows": [],
            "summary": {"error": str(exc)},
            "stability": {},
        }

    # --- Run backtests ---
    window_results: list[dict] = []
    for i, (train_s, train_e, test_s, test_e) in enumerate(windows):
        logger.info("Window %d/%d: train [%s → %s] test [%s → %s]",
                     i + 1, len(windows), train_s, train_e, test_s, test_e)

        cfg = _config_from_dict(strategy_config_dict)
        cfg.stock_pool = pool
        cfg.name = strategy_config_dict.get("name", "wf_strategy")

        try:
            result = run_backtest(cfg, start_date=test_s, end_date=test_e)
        except Exception as exc:
            logger.exception("Backtest failed for window %d: %s", i + 1, exc)
            window_results.append({
                "window_index": i,
                "train_start": train_s,
                "train_end": train_e,
                "test_start": test_s,
                "test_end": test_e,
                "error": str(exc),
                "ev": None,
                "win_rate": None,
                "total_trades": 0,
                "sharpe": None,
            })
            continue

        window_results.append({
            "window_index": i,
            "train_start": train_s,
            "train_end": train_e,
            "test_start": test_s,
            "test_end": test_e,
            "ev": result.expected_value,
            "win_rate": result.win_rate,
            "total_trades": result.total_trades,
            "sharpe": result.sharpe_ratio,
            "annual_return": result.annual_return_pct,
            "max_drawdown": result.max_drawdown_pct,
        })

    # --- Compute aggregate summary ---
    valid = [w for w in window_results if w.get("ev") is not None]
    if not valid:
        return {
            "windows": window_results,
            "summary": {"error": "All windows failed or produced no results."},
            "stability": {},
        }

    evs = [w["ev"] for w in valid]
    win_rates = [w["win_rate"] for w in valid]
    sharpes = [w["sharpe"] for w in valid if w["sharpe"] is not None]
    total_trades = sum(w["total_trades"] for w in valid)

    avg_ev = sum(evs) / len(evs)
    avg_wr = sum(win_rates) / len(win_rates)
    avg_sharpe = sum(sharpes) / len(sharpes) if sharpes else 0.0

    # Walk-Forward Efficiency: ratio of WF average EV to full-period EV
    # (optimistic baseline — full-period EV assumes perfect foreknowledge)
    wfe = None
    try:
        full_cfg = _config_from_dict(strategy_config_dict)
        full_cfg.stock_pool = pool
        full_cfg.name = strategy_config_dict.get("name", "wf_strategy")
        full_result = run_backtest(full_cfg, start_date=start_date, end_date=end_date)
        if full_result.expected_value != 0:
            wfe = round(avg_ev / full_result.expected_value * 100, 1)
    except Exception as exc:
        logger.warning("Could not compute full-period baseline for WFE: %s", exc)

    # EV consistency: 1 - CV (coefficient of variation)
    mean_ev = abs(avg_ev) if avg_ev != 0 else 1.0
    ev_std = _safe_std(evs)
    ev_consistency = max(0.0, (1.0 - ev_std / mean_ev)) * 100 if mean_ev else 0.0

    summary = {
        "windows_count": len(valid),
        "total_windows": len(window_results),
        "avg_ev": round(avg_ev, 4),
        "avg_win_rate": round(avg_wr, 2),
        "avg_sharpe": round(avg_sharpe, 2),
        "total_trades": total_trades,
        "ev_consistency_pct": round(ev_consistency, 1),
        "wfe_pct": wfe,
        "min_ev": round(min(evs), 4),
        "max_ev": round(max(evs), 4),
        "ev_std": round(ev_std, 4),
    }

    stability = {
        "wfe_pct": wfe,
        "ev_consistency_pct": round(ev_consistency, 1),
        "profitable_windows": sum(1 for e in evs if e > 0),
        "total_valid_windows": len(valid),
    }

    return {
        "windows": window_results,
        "summary": summary,
        "stability": stability,
    }


def walk_forward_summary_table(results: dict) -> str:
    """Render a walk-forward results dict as a human-readable formatted table.

    Args:
        results: The dict returned by ``run_walk_forward``.

    Returns:
        Multi-line string suitable for printing or logging.
    """
    lines: list[str] = []
    headers = [
        "Win", "Train Start", "Train End", "Test Start", "Test End",
        "EV%", "WinRate%", "Trades", "Sharpe",
    ]
    fmt = (
        "{win:>3}  {t_start:<12} {t_end:<12} {ts_start:<12} {ts_end:<12}"
        "  {ev:>8}  {wr:>8}  {tr:>6}  {sh:>7}"
    )

    lines.append("=" * 100)
    lines.append("Walk-Forward Summary")
    lines.append("=" * 100)
    lines.append(fmt.format(
        win="#", t_start="TrainStart", t_end="TrainEnd",
        ts_start="TestStart", ts_end="TestEnd",
        ev="EV%", wr="WinRate%", tr="#Trades", sh="Sharpe",
    ))
    lines.append("-" * 100)

    windows = results.get("windows", [])
    for w in windows:
        lines.append(fmt.format(
            win=w.get("window_index", "?"),
            t_start=w.get("train_start", ""),
            t_end=w.get("train_end", ""),
            ts_start=w.get("test_start", ""),
            ts_end=w.get("test_end", ""),
            ev=f"{w.get('ev', 0):+.2f}" if w.get("ev") is not None else "N/A",
            wr=f"{w.get('win_rate', 0):.1f}" if w.get("win_rate") is not None else "N/A",
            tr=w.get("total_trades", 0),
            sh=f"{w.get('sharpe', 0):.2f}" if w.get("sharpe") is not None else "N/A",
        ))

    lines.append("-" * 100)
    summary = results.get("summary", {})
    if "error" in summary:
        lines.append(f"  ERROR: {summary['error']}")
    else:
        lines.append(f"  Avg EV: {summary.get('avg_ev', 0):+.4f} | "
                      f"Avg WinRate: {summary.get('avg_win_rate', 0):.1f}% | "
                      f"Avg Sharpe: {summary.get('avg_sharpe', 0):.2f}")
        lines.append(f"  WFE: {summary.get('wfe_pct', 'N/A')}% | "
                      f"EV Consistency: {summary.get('ev_consistency_pct', 0):.1f}% | "
                      f"Profitable Windows: {results.get('stability', {}).get('profitable_windows', 0)}/"
                      f"{results.get('stability', {}).get('total_valid_windows', 0)}")
    lines.append("=" * 100)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_std(values: list) -> float:
    """Standard deviation safe for single-element lists."""
    if len(values) <= 1:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(variance)
