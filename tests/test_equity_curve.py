"""Tests for report.py equity curve building logic.

Tests the signal-mode equity curve construction based on daily returns
(equally-weighted portfolio approach).
"""

import pytest
import pandas as pd
import numpy as np
import copy
from unittest.mock import patch, MagicMock
from pathlib import Path


# ── Fixtures ──────────────────────────────────────────────────────

@pytest.fixture
def sample_stock_data():
    """Sample stock daily close prices for testing."""
    dates = pd.date_range("2024-01-02", periods=20, freq="B")
    # Stock A: steady growth
    close_a = pd.Series(
        [10.0, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8, 10.9,
         11.0, 11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 11.7, 11.8, 11.9],
        index=dates, name="close"
    )
    # Stock B: volatile
    close_b = pd.Series(
        [20.0, 20.5, 19.5, 20.0, 21.0, 20.0, 19.0, 20.0, 21.0, 22.0,
         21.0, 20.0, 19.0, 20.0, 21.0, 22.0, 23.0, 22.0, 21.0, 22.0],
        index=dates, name="close"
    )
    return {"000001": close_a, "000002": close_b}


@pytest.fixture
def sample_trades():
    """Sample trades for testing."""
    return [
        {
            "code": "000001",
            "buy_date": "2024-01-02",
            "sell_date": "2024-01-10",
            "profit_pct": 9.0,
        },
        {
            "code": "000002",
            "buy_date": "2024-01-03",
            "sell_date": "2024-01-12",
            "profit_pct": 10.0,
        },
    ]


@pytest.fixture
def overlapping_trades():
    """Trades with overlapping holding periods."""
    return [
        {
            "code": "000001",
            "buy_date": "2024-01-02",
            "sell_date": "2024-01-10",
            "profit_pct": 9.0,
        },
        {
            "code": "000002",
            "buy_date": "2024-01-02",
            "sell_date": "2024-01-10",
            "profit_pct": 10.0,
        },
    ]


# ── Tests for _build_equity_by_return ─────────────────────────────

class TestBuildEquityByReturn:

    def test_empty_trades(self):
        """空交易列表应该返回空权益曲线"""
        from report import _build_equity_by_return
        result = _build_equity_by_return([], 100000)
        assert result == []

    def test_single_trade(self, sample_stock_data):
        """单笔交易应该正确计算权益曲线"""
        from report import _build_equity_by_return

        trades = [
            {
                "code": "000001",
                "buy_date": "2024-01-02",
                "sell_date": "2024-01-10",
                "profit_pct": 9.0,
            }
        ]

        with patch("report._load_stock_close") as mock_load:
            mock_load.side_effect = lambda code: sample_stock_data.get(code, pd.Series(dtype=float))

            result = _build_equity_by_return(trades, 100000)

        assert len(result) > 0
        # 权益应该大于初始资金（因为是正收益）
        assert result[-1]["equity"] > 100000
        # 日期应该有序
        dates = [e["date"] for e in result]
        assert dates == sorted(dates)

    def test_overlapping_trades_equal_weight(self, sample_stock_data, overlapping_trades):
        """重叠交易应该做等权平均"""
        from report import _build_equity_by_return

        with patch("report._load_stock_close") as mock_load:
            mock_load.side_effect = lambda code: sample_stock_data.get(code, pd.Series(dtype=float))

            result = _build_equity_by_return(overlapping_trades, 100000)

        assert len(result) > 0
        # 权益应该大于初始资金
        assert result[-1]["equity"] > 100000

    def test_no_data_for_trade(self):
        """如果股票数据不存在，应该跳过该交易"""
        from report import _build_equity_by_return

        trades = [
            {
                "code": "999999",  # 不存在的股票
                "buy_date": "2024-01-02",
                "sell_date": "2024-01-10",
                "profit_pct": 9.0,
            }
        ]

        with patch("report._load_stock_close") as mock_load:
            mock_load.return_value = pd.Series(dtype=float)

            result = _build_equity_by_return(trades, 100000)

        assert result == []

    def test_dates_ordered(self, sample_stock_data, sample_trades):
        """权益曲线的日期应该严格有序"""
        from report import _build_equity_by_return

        with patch("report._load_stock_close") as mock_load:
            mock_load.side_effect = lambda code: sample_stock_data.get(code, pd.Series(dtype=float))

            result = _build_equity_by_return(sample_trades, 100000)

        dates = [e["date"] for e in result]
        assert dates == sorted(dates)
        assert len(dates) == len(set(dates))  # 无重复日期

    def test_cum_return_pct_format(self, sample_stock_data, sample_trades):
        """cum_return_pct应该是正确的百分比格式"""
        from report import _build_equity_by_return

        with patch("report._load_stock_close") as mock_load:
            mock_load.side_effect = lambda code: sample_stock_data.get(code, pd.Series(dtype=float))

            result = _build_equity_by_return(sample_trades, 100000)

        # 检查格式
        for entry in result:
            assert "date" in entry
            assert "equity" in entry
            assert "cum_return_pct" in entry
            assert isinstance(entry["equity"], float)
            assert isinstance(entry["cum_return_pct"], float)

    def test_empty_holding_period(self, sample_stock_data):
        """如果持仓期间没有数据，应该跳过"""
        from report import _build_equity_by_return

        trades = [
            {
                "code": "000001",
                "buy_date": "2024-01-02",
                "sell_date": "2024-01-02",  # 同一天买卖
                "profit_pct": 0.0,
            }
        ]

        with patch("report._load_stock_close") as mock_load:
            mock_load.side_effect = lambda code: sample_stock_data.get(code, pd.Series(dtype=float))

            result = _build_equity_by_return(trades, 100000)

        # 同一天买卖没有持仓期间数据，应该返回空
        assert result == []


# ── Tests for _fix_signal_equity_curve ────────────────────────────

class TestFixSignalEquityCurve:

    def test_no_anomaly_no_fix(self):
        """没有异常时不应该修正"""
        from report import _fix_signal_equity_curve

        data = {
            "summary": {"initial_capital": 100000},
            "trades": [{"code": "000001", "buy_date": "2024-01-02", "sell_date": "2024-01-10"}],
            "equity_curve": [
                {"date": "2024-01-10", "equity": 110000, "cum_return_pct": 10.0},
                {"date": "2024-01-11", "equity": 105000, "cum_return_pct": 5.0},
            ],
        }

        result = _fix_signal_equity_curve(data)
        # 不应该修正
        assert result["equity_curve"] == data["equity_curve"]

    def test_anomaly_triggers_fix(self, sample_stock_data):
        """cum_return_pct < -100%应该触发修正"""
        from report import _fix_signal_equity_curve

        data = {
            "summary": {"initial_capital": 100000},
            "trades": [
                {
                    "code": "000001",
                    "buy_date": "2024-01-02",
                    "sell_date": "2024-01-10",
                    "profit_pct": -50.0,
                },
            ],
            "equity_curve": [
                {"date": "2024-01-10", "equity": 50000, "cum_return_pct": -50.0},
                {"date": "2024-01-11", "equity": 0.01, "cum_return_pct": -150.0},  # 异常
            ],
        }

        # 保存原始equity_curve的副本
        original_equity_curve = copy.deepcopy(data["equity_curve"])

        with patch("report._load_stock_close") as mock_load:
            mock_load.side_effect = lambda code: sample_stock_data.get(code, pd.Series(dtype=float))

            result = _fix_signal_equity_curve(data)

        # 应该修正（结果和原始不同）
        assert result["equity_curve"] != original_equity_curve
        # 修正后应该没有异常
        for entry in result["equity_curve"]:
            assert entry["cum_return_pct"] >= -100

    def test_duplicate_dates_trigger_fix(self, sample_stock_data):
        """同一天多个数据点应该触发修正"""
        from report import _fix_signal_equity_curve

        data = {
            "summary": {"initial_capital": 100000},
            "trades": [
                {
                    "code": "000001",
                    "buy_date": "2024-01-02",
                    "sell_date": "2024-01-10",
                    "profit_pct": 10.0,
                },
            ],
            "equity_curve": [
                {"date": "2024-01-10", "equity": 110000, "cum_return_pct": 10.0},
                {"date": "2024-01-10", "equity": 105000, "cum_return_pct": 5.0},  # 重复日期
            ],
        }

        # 保存原始equity_curve的副本
        original_equity_curve = copy.deepcopy(data["equity_curve"])

        with patch("report._load_stock_close") as mock_load:
            mock_load.side_effect = lambda code: sample_stock_data.get(code, pd.Series(dtype=float))

            result = _fix_signal_equity_curve(data)

        # 应该修正（结果和原始不同）
        assert result["equity_curve"] != original_equity_curve
        # 修正后应该没有重复日期
        dates = [e["date"] for e in result["equity_curve"]]
        assert len(dates) == len(set(dates))

    def test_no_trades_no_fix(self):
        """没有交易数据时不应该修正"""
        from report import _fix_signal_equity_curve

        data = {
            "summary": {"initial_capital": 100000},
            "trades": [],
            "equity_curve": [
                {"date": "2024-01-10", "equity": 0.01, "cum_return_pct": -150.0},
            ],
        }

        result = _fix_signal_equity_curve(data)
        # 没有交易，不应该修正
        assert result["equity_curve"] == data["equity_curve"]


# ── Integration test ──────────────────────────────────────────────

class TestIntegration:

    def test_full_pipeline(self, sample_stock_data, sample_trades):
        """完整流程测试：从交易数据构建权益曲线"""
        from report import _build_equity_by_return
        from analytics.risk import max_drawdown

        with patch("report._load_stock_close") as mock_load:
            mock_load.side_effect = lambda code: sample_stock_data.get(code, pd.Series(dtype=float))

            equity_curve = _build_equity_by_return(sample_trades, 100000)

        # 权益曲线应该有效
        assert len(equity_curve) > 0

        # 计算最大回撤
        dd_result = max_drawdown(equity_curve)
        max_dd = dd_result[0]

        # 最大回撤应该在合理范围内
        assert 0 <= max_dd <= 100

        # 总收益应该正确
        total_return = equity_curve[-1]["cum_return_pct"]
        assert total_return > 0  # 正收益

    def test_max_drawdown_no_anomaly(self, sample_stock_data, sample_trades):
        """最大回撤不应该出现异常值（如-150%）"""
        from report import _build_equity_by_return
        from analytics.risk import max_drawdown

        with patch("report._load_stock_close") as mock_load:
            mock_load.side_effect = lambda code: sample_stock_data.get(code, pd.Series(dtype=float))

            equity_curve = _build_equity_by_return(sample_trades, 100000)

        # 计算最大回撤
        dd_result = max_drawdown(equity_curve)
        max_dd = dd_result[0]

        # 最大回撤不应该超过100%
        assert max_dd <= 100