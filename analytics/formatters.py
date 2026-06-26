"""
Pretty-printing formatters module for quantitative backtesting analytics.

Provides functions to format and display analytics results in human-readable
text tables, histograms, and reports. All functions return formatted strings
and optionally print to stdout.
"""

import math
from collections import defaultdict
from typing import Any

import numpy as np


def _bar_char(value: float, max_val: float, width: int = 20) -> str:
    """Generate a text bar proportional to value/max_val."""
    if max_val == 0:
        return ""
    ratio = min(abs(value) / max_val, 1.0)
    filled = int(ratio * width)
    return "█" * filled


def format_summary_table(result: dict[str, Any]) -> str:
    """
    Format a comprehensive performance summary as a rich text table.

    Args:
        result: Dict with performance metrics. Expected keys include:
            total_return_pct, annual_return_pct, max_drawdown_pct,
            sharpe_ratio, sortino_ratio, calmar_ratio, win_rate_pct,
            total_trades, avg_return_pct, volatility_pct.

    Returns:
        Formatted string suitable for console output.
    """
    lines = []
    lines.append("=" * 60)
    lines.append("              PERFORMANCE SUMMARY")
    lines.append("=" * 60)

    metrics = [
        ("Total Return", result.get("total_return_pct", 0.0), "%", "{:.2f}"),
        ("Annual Return (CAGR)", result.get("annual_return_pct", 0.0), "%", "{:.2f}"),
        ("Max Drawdown", result.get("max_drawdown_pct", 0.0), "%", "{:.2f}"),
        ("Sharpe Ratio", result.get("sharpe_ratio", 0.0), "", "{:.2f}"),
        ("Sortino Ratio", result.get("sortino_ratio", 0.0), "", "{:.2f}"),
        ("Calmar Ratio", result.get("calmar_ratio", 0.0), "", "{:.2f}"),
        ("Volatility (Ann.)", result.get("volatility_pct", 0.0), "%", "{:.2f}"),
        ("Win Rate", result.get("win_rate_pct", 0.0), "%", "{:.1f}"),
        ("Total Trades", result.get("total_trades", 0), "", "{}"),
        ("Avg Return/Trade", result.get("avg_return_pct", 0.0), "%", "{:.2f}"),
        ("Profit Factor", result.get("profit_factor", 0.0), "", "{:.2f}"),
    ]

    for label, value, unit, fmt in metrics:
        formatted_val = fmt.format(value)
        if unit:
            formatted_val += unit
        lines.append(f"  {label:<25} {formatted_val:>15}")

    # Max drawdown details
    if "max_dd_start" in result:
        lines.append(f"  {'Max DD Start':<25} {result['max_dd_start']:>15}")
    if "max_dd_end" in result:
        lines.append(f"  {'Max DD End':<25} {result['max_dd_end']:>15}")
    if "max_dd_duration" in result:
        lines.append(f"  {'Max DD Duration':<25} {str(result['max_dd_duration']) + ' days':>15}")

    lines.append("=" * 60)
    return "\n".join(lines)


def format_trade_distribution(trades: list[dict], bins: int = 10) -> str:
    """
    Create a text-based histogram of trade return distribution.

    Args:
        trades: List of trade dicts with 'profit_pct' key.
        bins: Number of bins for the histogram (default 10).

    Returns:
        Formatted string with a text histogram.
    """
    if not trades:
        return "No trades to display."

    returns = np.array([t.get("profit_pct", 0.0) for t in trades], dtype=float)
    returns = returns[~np.isnan(returns)]

    if len(returns) == 0:
        return "No valid trade returns."

    lines = []
    lines.append("=" * 55)
    lines.append("          TRADE RETURN DISTRIBUTION")
    lines.append("=" * 55)

    # Stats
    lines.append(f"  Count: {len(returns)}")
    lines.append(f"  Mean:  {np.mean(returns):.2f}%")
    lines.append(f"  Std:   {np.std(returns, ddof=1):.2f}%")
    lines.append(f"  Min:   {np.min(returns):.2f}%")
    lines.append(f"  Max:   {np.max(returns):.2f}%")
    lines.append(f"  Skew:  {float(np.mean((returns - np.mean(returns))**3) / np.std(returns, ddof=1)**3) if np.std(returns, ddof=1) > 0 else 0.0:.2f}")
    lines.append("")

    # Build histogram
    hist, bin_edges = np.histogram(returns, bins=bins)
    max_count = int(np.max(hist)) if len(hist) > 0 else 1
    bar_width = 30

    for i in range(len(hist)):
        edge_left = bin_edges[i]
        edge_right = bin_edges[i + 1]
        count = int(hist[i])
        bar = "█" * int(count / max_count * bar_width) if max_count > 0 else ""
        pct = count / len(returns) * 100.0
        lines.append(
            f"  [{edge_left:>7.1f}% to {edge_right:>7.1f}%) "
            f"{bar:<{bar_width}} {count:>4d} ({pct:>5.1f}%)"
        )

    lines.append("=" * 55)
    return "\n".join(lines)


def format_attribution_table(attribution_data: dict[str, Any]) -> str:
    """
    Format attribution data (e.g., from yearly_attribution) as a table.

    Args:
        attribution_data: Dict where keys are categories and values are dicts
            with 'trades', 'win_rate', 'avg_return', 'total_return' keys.

    Returns:
        Formatted string with attribution table.
    """
    if not attribution_data:
        return "No attribution data available."

    lines = []
    lines.append("=" * 75)
    lines.append("                    ATTRIBUTION TABLE")
    lines.append("=" * 75)
    header = f"  {'Category':<12} {'Trades':>7} {'Win Rate':>9} {'Avg Ret':>9} {'Total Ret':>10}"
    lines.append(header)
    lines.append("  " + "-" * 65)

    # Sort by key
    sorted_keys = sorted(attribution_data.keys(), key=str)

    for key in sorted_keys:
        data = attribution_data[key]
        trades = data.get("trades", data.get("count", 0))
        win_rate = data.get("win_rate", 0.0)
        avg_return = data.get("avg_return", 0.0)
        total_return = data.get("total_return", 0.0)

        lines.append(
            f"  {str(key):<12} {trades:>7} {win_rate:>8.1f}% {avg_return:>8.2f}% {total_return:>9.2f}%"
        )

    lines.append("=" * 75)
    return "\n".join(lines)


def format_validation_report(walk_forward_results: dict[str, Any]) -> str:
    """
    Format a walk-forward validation report.

    Args:
        walk_forward_results: Dict with walk-forward validation data.
            Expected keys:
            - 'periods': list of period result dicts, each with
              'period', 'trades', 'return_pct', 'win_rate'.
            - 'summary': dict with aggregate metrics.

    Returns:
        Formatted string with validation report.
    """
    lines = []
    lines.append("=" * 70)
    lines.append("           WALK-FORWARD VALIDATION REPORT")
    lines.append("=" * 70)

    # Summary section
    summary = walk_forward_results.get("summary", {})
    if summary:
        lines.append("  SUMMARY:")
        lines.append(f"    Total Return:     {summary.get('total_return_pct', 0.0):.2f}%")
        lines.append(f"    Annual Return:    {summary.get('annual_return_pct', 0.0):.2f}%")
        lines.append(f"    Sharpe Ratio:     {summary.get('sharpe_ratio', 0.0):.2f}")
        lines.append(f"    Max Drawdown:     {summary.get('max_drawdown_pct', 0.0):.2f}%")
        lines.append(f"    Win Rate:         {summary.get('win_rate_pct', 0.0):.1f}%")
        lines.append(f"    Total Trades:     {summary.get('total_trades', 0)}")
        lines.append("")

    # Period details
    periods = walk_forward_results.get("periods", [])
    if periods:
        lines.append("  PERIOD DETAILS:")
        header = f"  {'Period':<12} {'Trades':>7} {'Return':>9} {'Win Rate':>9} {'Cum Ret':>9}"
        lines.append(header)
        lines.append("  " + "-" * 55)

        cumulative = 0.0
        for period in periods:
            name = str(period.get("period", ""))
            trades = period.get("trades", 0)
            ret = period.get("return_pct", 0.0)
            win_rate = period.get("win_rate", 0.0)
            cumulative += ret

            # Mark profitable/unprofitable
            marker = "✓" if ret > 0 else "✗"
            lines.append(
                f"  {name:<12} {trades:>7} {ret:>8.2f}% {win_rate:>8.1f}% {cumulative:>8.2f}%  {marker}"
            )

    # Consistency metrics
    if periods:
        period_returns = [p.get("return_pct", 0.0) for p in periods]
        profitable_periods = sum(1 for r in period_returns if r > 0)
        total_periods = len(period_returns)
        if total_periods > 0:
            lines.append("")
            lines.append("  CONSISTENCY:")
            lines.append(f"    Profitable Periods: {profitable_periods}/{total_periods} "
                         f"({profitable_periods/total_periods*100:.1f}%)")

    lines.append("=" * 70)
    return "\n".join(lines)
