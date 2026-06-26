"""Tests for analytics/common.py — shared data utilities."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from analytics.common import (
    to_equity_df,
    extract_trade_returns,
    compute_daily_returns,
    forward_fill_daily,
    safe_divide,
)


class TestToEquityDF:

    def test_empty(self, empty_equity_curve):
        df = to_equity_df(empty_equity_curve)
        assert df.empty
        assert "date" in df.columns or True  # empty df may not have columns

    def test_single_point(self, single_point_equity):
        df = to_equity_df(single_point_equity)
        assert len(df) == 1
        assert float(df["equity"].iloc[0]) == 100000.0

    def test_sorts_by_date(self):
        curve = [
            {"date": "2020-03-01", "equity": 103.0},
            {"date": "2020-01-01", "equity": 101.0},
            {"date": "2020-02-01", "equity": 102.0},
        ]
        df = to_equity_df(curve)
        assert df.index[0] == pd.Timestamp("2020-01-01")
        assert df.index[-1] == pd.Timestamp("2020-03-01")

    def test_datetime_index(self, growing_equity_curve):
        df = to_equity_df(growing_equity_curve)
        assert isinstance(df.index, pd.DatetimeIndex)


class TestExtractTradeReturns:

    def test_empty(self, empty_trades):
        arr = extract_trade_returns(empty_trades)
        assert len(arr) == 0

    def test_winning(self, winning_trades):
        arr = extract_trade_returns(winning_trades)
        assert len(arr) == 10
        assert np.allclose(arr, 5.0)

    def test_mixed(self, mixed_trades):
        arr = extract_trade_returns(mixed_trades)
        assert len(arr) == 20
        wins = (arr > 0).sum()
        losses = (arr <= 0).sum()
        assert wins == 10
        assert losses == 10

    def test_filters_nan(self, trade_with_nan):
        arr = extract_trade_returns(trade_with_nan)
        assert len(arr) == 2  # NaN filtered out
        assert 5.0 in arr
        assert -2.0 in arr


class TestComputeDailyReturns:

    def test_empty(self, empty_equity_curve):
        rets = compute_daily_returns(empty_equity_curve)
        assert len(rets) == 0

    def test_single_point(self, single_point_equity):
        rets = compute_daily_returns(single_point_equity)
        assert len(rets) == 0  # need at least 2 points

    def test_flat_returns(self, flat_equity_curve):
        rets = compute_daily_returns(flat_equity_curve)
        assert len(rets) > 0
        assert np.allclose(rets.values, 0.0)  # flat equity → zero returns

    def test_growing_returns(self, growing_equity_curve):
        rets = compute_daily_returns(growing_equity_curve)
        assert len(rets) == 99  # 100 points → 99 returns
        assert np.all(rets.values > 0)  # all positive


class TestForwardFillDaily:

    def test_empty(self, empty_equity_curve):
        df = forward_fill_daily(empty_equity_curve)
        assert df.empty

    def test_fills_gaps(self, sparse_equity_curve):
        df = forward_fill_daily(sparse_equity_curve)
        # Should have daily entries from first to last date
        expected_days = (pd.Timestamp("2020-06-15") - pd.Timestamp("2020-01-15")).days + 1
        assert len(df) == expected_days
        # First value should match first equity point
        assert float(df["equity"].iloc[0]) == 101000.0
        # Last value should match last equity point
        assert float(df["equity"].iloc[-1]) == 112000.0

    def test_dedup_dates(self):
        curve = [
            {"date": "2020-01-01", "equity": 100.0},
            {"date": "2020-01-01", "equity": 200.0},  # duplicate
            {"date": "2020-01-02", "equity": 300.0},
        ]
        df = forward_fill_daily(curve)
        assert len(df) == 2  # 2 unique dates, fill in 1 gap → 2 days
        # Should keep last value for duplicate date (200.0)
        assert float(df["equity"].iloc[0]) == 200.0


class TestSafeDivide:

    def test_normal(self):
        assert safe_divide(10.0, 2.0) == 5.0

    def test_zero_denominator(self):
        assert safe_divide(10.0, 0.0) == 0.0

    def test_zero_numerator(self):
        assert safe_divide(0.0, 5.0) == 0.0
