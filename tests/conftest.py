"""
Shared test fixtures for the quant backtest system.

All fixtures produce clean, minimal data for deterministic testing.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


# ── Equity Curve fixtures ────────────────────────────────────────


@pytest.fixture
def empty_equity_curve():
    """Empty equity curve."""
    return []


@pytest.fixture
def single_point_equity():
    """Single-point equity curve."""
    return [{"date": "2020-01-02", "equity": 100000.0}]


@pytest.fixture
def flat_equity_curve():
    """Flat equity: 100 days at 100k each."""
    base = datetime(2020, 1, 2)
    return [
        {"date": (base + timedelta(days=i)).strftime("%Y-%m-%d"), "equity": 100000.0}
        for i in range(100)
    ]


@pytest.fixture
def growing_equity_curve():
    """Linearly growing equity: 100k → 200k over 100 days."""
    base = datetime(2020, 1, 2)
    curve = []
    for i in range(100):
        equity = 100000.0 + i * 1000.0  # +1k/day
        curve.append({
            "date": (base + timedelta(days=i)).strftime("%Y-%m-%d"),
            "equity": equity,
        })
    return curve


@pytest.fixture
def declining_equity_curve():
    """Linearly declining equity: 100k → 50k over 100 days."""
    base = datetime(2020, 1, 2)
    curve = []
    for i in range(100):
        equity = max(100000.0 - i * 500.0, 1.0)
        curve.append({
            "date": (base + timedelta(days=i)).strftime("%Y-%m-%d"),
            "equity": equity,
        })
    return curve


@pytest.fixture
def v_shaped_equity_curve():
    """V-shaped equity: drops 20% then recovers."""
    base = datetime(2020, 1, 2)
    curve = []
    for i in range(100):
        if i < 50:
            equity = 100000.0 * (1 - 0.004 * i)  # drop 0.4%/day → -20%
        else:
            equity = 80000.0 * (1 + 0.004 * (i - 50))  # recover
        curve.append({
            "date": (base + timedelta(days=i)).strftime("%Y-%m-%d"),
            "equity": round(equity, 2),
        })
    return curve


@pytest.fixture
def sparse_equity_curve():
    """Signal-mode equity (sparse dates — only on trade sell dates)."""
    return [
        {"date": "2020-01-15", "equity": 101000.0},
        {"date": "2020-02-20", "equity": 103500.0},
        {"date": "2020-03-10", "equity": 99500.0},
        {"date": "2020-05-01", "equity": 108000.0},
        {"date": "2020-06-15", "equity": 112000.0},
    ]


# ── Trades fixtures ──────────────────────────────────────────────


@pytest.fixture
def empty_trades():
    """Empty trade list."""
    return []


@pytest.fixture
def winning_trades():
    """10 winning trades, +5% each."""
    return [
        {"code": f"00000{i}", "buy_date": f"2020-0{i+1}-01", "buy_price": 10.0,
         "sell_date": f"2020-0{i+1}-10", "sell_price": 10.5,
         "profit_pct": 5.0, "hold_days": 9, "sell_reason": "止盈", "shares": 100}
        for i in range(1, 11)
    ]


@pytest.fixture
def losing_trades():
    """10 losing trades, -3% each."""
    return [
        {"code": f"00001{i}", "buy_date": f"2020-0{i+1}-01", "buy_price": 10.0,
         "sell_date": f"2020-0{i+1}-10", "sell_price": 9.7,
         "profit_pct": -3.0, "hold_days": 9, "sell_reason": "止损", "shares": 100}
        for i in range(1, 11)
    ]


@pytest.fixture
def mixed_trades(winning_trades, losing_trades):
    """20 mixed trades: 10 wins + 10 losses."""
    return winning_trades + losing_trades


@pytest.fixture
def single_trade():
    """A single winning trade."""
    return [{"code": "000001", "buy_date": "2020-01-02", "buy_price": 10.0,
             "sell_date": "2020-01-15", "sell_price": 11.0,
             "profit_pct": 10.0, "hold_days": 13, "sell_reason": "止盈", "shares": 100}]


@pytest.fixture
def trade_with_nan():
    """Trade list with a NaN profit value."""
    return [
        {"code": "000001", "buy_date": "2020-01-02", "profit_pct": 5.0, "hold_days": 5,
         "buy_price": 10.0, "sell_price": 10.5, "sell_date": "2020-01-07",
         "sell_reason": "止盈", "shares": 100},
        {"code": "000002", "buy_date": "2020-01-03", "profit_pct": float("nan"), "hold_days": 3,
         "buy_price": 20.0, "sell_price": 20.0, "sell_date": "2020-01-06",
         "sell_reason": "止损", "shares": 100},
        {"code": "000003", "buy_date": "2020-01-04", "profit_pct": -2.0, "hold_days": 7,
         "buy_price": 15.0, "sell_price": 14.7, "sell_date": "2020-01-11",
         "sell_reason": "止损", "shares": 100},
    ]


# ── Capital fixture ──────────────────────────────────────────────


@pytest.fixture
def initial_capital():
    return 100000.0
