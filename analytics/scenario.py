"""
Historical scenario stress testing.

Replays strategy performance through specific historical crisis
periods to estimate tail-risk behavior.

When benchmark_equity is provided, computes actual benchmark
returns during each scenario (instead of using hardcoded values).

Chinese A-share scenarios:
  - 2015-06-12 to 2015-08-26: 2015 stock market crash
  - 2016-01-04 to 2016-01-28: Circuit breaker / yuan devaluation
  - 2018-01-26 to 2018-12-28: Trade war / deleveraging
  - 2020-01-23 to 2020-03-23: COVID crash
  - 2021-02-18 to 2021-04-13: "Huddle trade" unwind
  - 2024-01-02 to 2024-02-05: Liquidity crunch / small-cap crash
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .common import forward_fill_daily, to_equity_df

# ── Predefined A-share crisis scenarios ─────────────────────────────

A_SHARE_SCENARIOS: dict[str, dict[str, Any]] = {
    "2015_crash": {
        "name": "2015 股灾 (Crash)",
        "start": "2015-06-12",
        "end": "2015-08-26",
        "description": "Leverage unwind, circuit breaker, CSI 300 -43%",
    },
    "2016_circuit_breaker": {
        "name": "2016 熔断 (Circuit Breaker)",
        "start": "2016-01-04",
        "end": "2016-01-28",
        "description": "Yuan devaluation, circuit breaker triggered twice",
    },
    "2018_trade_war": {
        "name": "2018 贸易战 (Trade War)",
        "start": "2018-01-26",
        "end": "2018-12-28",
        "description": "US-China trade war + deleveraging",
    },
    "2020_covid": {
        "name": "2020 新冠 (COVID)",
        "start": "2020-01-23",
        "end": "2020-03-23",
        "description": "COVID pandemic, global risk-off",
    },
    "2021_huddle_unwind": {
        "name": "2021 抱团瓦解 (Huddle Unwind)",
        "start": "2021-02-18",
        "end": "2021-04-13",
        "description": "Large-cap growth 'huddle trade' unwinds",
    },
    "2024_liquidity_crunch": {
        "name": "2024 流动性危机 (Liquidity Crunch)",
        "start": "2024-01-02",
        "end": "2024-02-05",
        "description": "Small-cap crash, margin calls, quant fund liquidations",
    },
}


def scenario_stress_test(
    equity_curve: list[dict],
    scenarios: dict[str, dict[str, Any]] | None = None,
    benchmark_equity: list[dict] | None = None,
    benchmark_name: str = "CSI 300",
) -> dict[str, Any]:
    """
    Compute strategy return during each historical crisis scenario.

    When benchmark_equity is provided, actual benchmark returns are
    computed from the index data and displayed alongside strategy returns.

    Args:
        equity_curve: Strategy equity curve (sparse OK).
        scenarios: Optional custom scenario dict. Defaults to A_SHARE_SCENARIOS.
        benchmark_equity: Optional benchmark equity curve for comparison.
        benchmark_name: Name for the benchmark column header.

    Returns:
        Dict mapping scenario key to scenario result dict.
    """
    if scenarios is None:
        scenarios = A_SHARE_SCENARIOS

    df = forward_fill_daily(equity_curve)
    if df.empty:
        return {}

    # Pre-process benchmark if provided
    bench_df = None
    if benchmark_equity:
        bench_df = forward_fill_daily(benchmark_equity)

    results = {}
    for key, sc in scenarios.items():
        try:
            start_dt = pd.Timestamp(sc["start"])
            end_dt = pd.Timestamp(sc["end"])
        except Exception:
            continue

        # Strategy performance in scenario
        period = df.loc[start_dt:end_dt]
        if len(period) < 2:
            results[key] = {
                "name": sc["name"],
                "start": sc["start"],
                "end": sc["end"],
                "strategy_return_pct": None,
                "max_dd_pct": None,
                "recovery_days": None,
                "benchmark_return_pct": None,
                "available": False,
            }
            continue

        start_val = float(period["equity"].iloc[0])
        end_val = float(period["equity"].iloc[-1])
        strat_ret = (end_val - start_val) / start_val * 100.0 if start_val > 0 else 0.0

        # Max drawdown within scenario
        equity_vals = period["equity"].values
        peak = np.maximum.accumulate(equity_vals)
        dd = (peak - equity_vals) / peak * 100.0
        max_dd = float(dd.max()) if len(dd) > 0 else 0.0

        # Recovery: days from scenario end until equity recovers to pre-crash peak
        pre_crash_peak = float(df.loc[:start_dt]["equity"].max()) if len(df.loc[:start_dt]) > 0 else start_val
        recovery_days = -1
        post_crash = df.loc[end_dt:]
        for dt, row in post_crash.iterrows():
            if float(row["equity"]) >= pre_crash_peak:
                recovery_days = (dt - end_dt).days
                break

        # Benchmark performance in scenario (if available)
        bench_ret = None
        if bench_df is not None:
            b_period = bench_df.loc[start_dt:end_dt]
            if len(b_period) >= 2:
                b_start = float(b_period["equity"].iloc[0])
                b_end = float(b_period["equity"].iloc[-1])
                if b_start > 0:
                    bench_ret = round((b_end - b_start) / b_start * 100.0, 1)

        results[key] = {
            "name": sc["name"],
            "start": sc["start"],
            "end": sc["end"],
            "strategy_return_pct": round(strat_ret, 2),
            "max_dd_pct": round(max_dd, 2),
            "recovery_days": recovery_days,
            "benchmark_return_pct": bench_ret,
            "available": True,
        }

    return results


def scenario_summary_table(
    results: dict[str, Any],
    benchmark_name: str = "Bench",
) -> str:
    """Render scenario stress test results as a table."""
    lines = [
        "=" * 90,
        "  Historical Scenario Stress Test",
        "=" * 90,
    ]
    header = f"  {'Scenario':<28} {'Return':>8} {'MaxDD':>7} {'Bench':>8} {'Recovery':>9}"
    lines.append(header)
    lines.append("  " + "-" * 70)

    # Compute excess for sorting
    scored = []
    for key, sc in results.items():
        s_ret = sc.get("strategy_return_pct", -999)
        b_ret = sc.get("benchmark_return_pct", -999)
        excess = s_ret - b_ret if (s_ret is not None and b_ret is not None) else -999
        scored.append((excess, key, sc))
    scored.sort(key=lambda x: x[0] if x[0] != -999 else -999, reverse=True)

    for _, key, sc in scored:
        name = sc.get("name", key)[:27]
        if not sc.get("available"):
            lines.append(f"  {name:<28} {'N/A':>8} {'N/A':>7} {'N/A':>8} {'N/A':>9}")
            continue

        strat_ret = sc.get("strategy_return_pct")
        max_dd = sc.get("max_dd_pct")
        bench_ret = sc.get("benchmark_return_pct")
        recovery = sc.get("recovery_days", -1)

        ret_str = f"{strat_ret:+.1f}%" if strat_ret is not None else "N/A"
        dd_str = f"{max_dd:.1f}%" if max_dd is not None else "N/A"
        bench_str = f"{bench_ret:+.1f}%" if bench_ret is not None else "N/A"
        rec_str = f"{recovery}d" if recovery >= 0 else "N/A"

        lines.append(f"  {name:<28} {ret_str:>8} {dd_str:>7} {bench_str:>8} {rec_str:>9}")

    lines.append("=" * 90)
    return "\n".join(lines)
