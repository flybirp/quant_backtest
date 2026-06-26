"""
Transaction Cost Analysis (TCA) for quantitative backtesting.

Reports the total cost drag from commissions, stamp tax, and slippage
so the user knows exactly how much of gross alpha was consumed by frictions.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def cost_attribution(trades: list[dict]) -> dict[str, Any]:
    """
    Estimate and attribute total transaction costs from trade history.

    Computes the actual costs embedded in each trade's profit_pct
    by reconstructing gross vs net profit.

    Args:
        trades: List of trade dicts with 'buy_price', 'sell_price',
                'shares', 'profit_pct'.

    Returns:
        Dict with total commission, stamp tax, slippage estimates,
        and cost-to-profit ratios.
    """
    if not trades:
        return _empty_cost_result()

    total_cost = 0.0
    total_turnover = 0.0  # buy + sell notional
    gross_profit_sum = 0.0
    net_profit_sum = 0.0
    win_trades = 0
    lose_trades = 0

    for t in trades:
        bp = t.get("buy_price", 0)
        sp = t.get("sell_price", 0)
        sh = t.get("shares", 0)
        profit = t.get("profit_pct", 0)

        if bp <= 0 or sp <= 0 or sh <= 0:
            continue

        # Notional amounts
        buy_notional = bp * sh
        sell_notional = sp * sh
        trade_turnover = buy_notional + sell_notional
        total_turnover += trade_turnover

        # Gross profit (before costs)
        gross_pct = (sp - bp) / bp * 100.0
        gross_amt = gross_pct / 100.0 * buy_notional

        # Net profit (after costs embedded in profit_pct)
        net_amt = profit / 100.0 * buy_notional

        # Cost = gross - net
        trade_cost = gross_amt - net_amt
        total_cost += trade_cost

        gross_profit_sum += gross_amt
        net_profit_sum += net_amt

        if profit > 0:
            win_trades += 1
        elif profit < 0:
            lose_trades += 1

    total_trades = win_trades + lose_trades
    if total_trades == 0:
        return _empty_cost_result()

    avg_cost_per_trade = total_cost / total_trades
    cost_bps = (total_cost / total_turnover * 10000.0) if total_turnover > 0 else 0.0

    # Cost impact on returns
    cost_to_gross_ratio = (abs(total_cost) / abs(gross_profit_sum) * 100.0) if abs(gross_profit_sum) > 0 else 0.0

    return {
        "total_cost": round(total_cost, 2),
        "total_turnover": round(total_turnover, 2),
        "cost_per_trade": round(avg_cost_per_trade, 2),
        "cost_bps_turnover": round(cost_bps, 2),
        "cost_to_gross_profit_pct": round(cost_to_gross_ratio, 2),
        "gross_profit_sum": round(gross_profit_sum, 2),
        "net_profit_sum": round(net_profit_sum, 2),
        "trades_analyzed": total_trades,
    }


def estimated_cost_breakdown(
    trades: list[dict],
    commission_rate: float = 0.0003,
    stamp_tax_rate: float = 0.001,
    slippage_pct: float = 0.001,
) -> dict[str, Any]:
    """
    Estimate cost breakdown by component using config rates.

    This is a model-based estimate (not computed from actual trade pnl)
    and gives the theoretical cost on each side.

    Args:
        trades: List of trade dicts.
        commission_rate: Per-side commission rate (default 0.03%).
        stamp_tax_rate: Sell-side stamp tax (default 0.10%, A-shares).
        slippage_pct: Per-side slippage (default 0.10%).

    Returns:
        Dict with commission, stamp_tax, slippage breakdowns.
    """
    if not trades:
        return _empty_breakdown_result()

    buy_notional = 0.0
    sell_notional = 0.0

    for t in trades:
        bp = t.get("buy_price", 0)
        sp = t.get("sell_price", 0)
        sh = t.get("shares", 0)
        if bp > 0:
            buy_notional += bp * sh
        if sp > 0:
            sell_notional += sp * sh

    total_notional = buy_notional + sell_notional

    commission = (buy_notional + sell_notional) * commission_rate
    stamp_tax = sell_notional * stamp_tax_rate
    slippage = total_notional * slippage_pct
    total = commission + stamp_tax + slippage

    return {
        "commission": round(commission, 2),
        "stamp_tax": round(stamp_tax, 2),
        "slippage_estimate": round(slippage, 2),
        "total_cost": round(total, 2),
        "total_notional": round(total_notional, 2),
        "cost_bps": round(total / total_notional * 10000.0, 2) if total_notional > 0 else 0.0,
    }


def _empty_cost_result() -> dict[str, Any]:
    return {
        "total_cost": 0.0,
        "total_turnover": 0.0,
        "cost_per_trade": 0.0,
        "cost_bps_turnover": 0.0,
        "cost_to_gross_profit_pct": 0.0,
        "gross_profit_sum": 0.0,
        "net_profit_sum": 0.0,
        "trades_analyzed": 0,
    }


def _empty_breakdown_result() -> dict[str, Any]:
    return {
        "commission": 0.0,
        "stamp_tax": 0.0,
        "slippage_estimate": 0.0,
        "total_cost": 0.0,
        "total_notional": 0.0,
        "cost_bps": 0.0,
    }
