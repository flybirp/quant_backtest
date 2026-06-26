"""Tests for analytics/performance.py."""

import pytest
import numpy as np

from analytics.performance import (
    total_return,
    annual_return,
    monthly_returns_table,
    yearly_returns_table,
)


class TestTotalReturn:

    def test_empty(self, empty_equity_curve, initial_capital):
        assert total_return(empty_equity_curve, initial_capital) == 0.0

    def test_no_capital(self, growing_equity_curve):
        assert total_return(growing_equity_curve, 0.0) == 0.0

    def test_growing(self, growing_equity_curve, initial_capital):
        # 100k → 199k = +99%
        ret = total_return(growing_equity_curve, initial_capital)
        assert ret == pytest.approx(99.0, rel=0.05)

    def test_declining(self, declining_equity_curve, initial_capital):
        ret = total_return(declining_equity_curve, initial_capital)
        assert ret < 0

    def test_flat(self, flat_equity_curve, initial_capital):
        assert total_return(flat_equity_curve, initial_capital) == 0.0


class TestAnnualReturn:

    def test_empty(self, empty_equity_curve, initial_capital):
        assert annual_return(empty_equity_curve, initial_capital) == 0.0

    def test_flat(self, flat_equity_curve, initial_capital):
        assert annual_return(flat_equity_curve, initial_capital) == 0.0

    def test_single_point(self, single_point_equity, initial_capital):
        assert annual_return(single_point_equity, initial_capital) == 0.0

    def test_growing(self, growing_equity_curve, initial_capital):
        # 100 days of growth, CAGR should be huge because annualized
        cagr = annual_return(growing_equity_curve, initial_capital)
        assert cagr > 0

    def test_negative_final(self):
        # final_equity <= 0 → returns 0.0 (CAGR undefined)
        curve = [
            {"date": "2020-01-01", "equity": 100000.0},
            {"date": "2020-12-31", "equity": -1.0},
        ]
        assert annual_return(curve, 100000.0) == 0.0


class TestMonthlyReturnsTable:

    def test_empty(self, empty_equity_curve):
        assert monthly_returns_table(empty_equity_curve) == {}

    def test_single_month(self):
        curve = [
            {"date": "2020-01-02", "equity": 100.0},
            {"date": "2020-01-31", "equity": 110.0},
        ]
        # Resample to month-end: single point → no pct_change possible
        tbl = monthly_returns_table(curve)
        # With pandas ME resample on a single month, we get 1 point.
        # pct_change of a single-element series is NaN, so dropped.
        assert isinstance(tbl, dict)


class TestYearlyReturnsTable:

    def test_empty(self, empty_equity_curve):
        assert yearly_returns_table(empty_equity_curve) == {}
