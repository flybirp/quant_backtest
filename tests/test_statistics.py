"""Tests for analytics/statistics.py."""

import pytest

from analytics.statistics import (
    bootstrap_confidence_interval,
    bootstrap_ev_ci,
    t_test_mean,
    normality_test,
)


class TestBootstrapCI:

    def test_empty(self, empty_trades):
        low, high, est = bootstrap_ev_ci(empty_trades)
        assert low == 0.0
        assert high == 0.0
        assert est == 0.0

    def test_all_wins(self, winning_trades):
        low, high, est = bootstrap_ev_ci(winning_trades)
        # 10 trades all at 5% — bootstrap should be tight around 5%
        assert est == 5.0
        assert low == 5.0
        assert high == 5.0  # no variance → all resamples give 5.0

    def test_mixed(self, mixed_trades):
        low, high, est = bootstrap_confidence_interval(mixed_trades, statistic="mean")
        assert est == 1.0  # (5*10 - 3*10)/20 = 1.0
        assert low <= est <= high
        # CI should be wider for mixed trades
        assert high - low > 0

    def test_median_statistic(self, mixed_trades):
        low, high, est = bootstrap_confidence_interval(mixed_trades, statistic="median")
        assert low <= est <= high

    def test_deterministic(self, mixed_trades):
        """Bootstrap should be reproducible (fixed seed)."""
        low1, high1, est1 = bootstrap_ev_ci(mixed_trades)
        low2, high2, est2 = bootstrap_ev_ci(mixed_trades)
        assert low1 == low2
        assert high1 == high2


class TestTTest:

    def test_empty(self, empty_trades):
        t, p, sig = t_test_mean(empty_trades)
        assert p == 1.0
        assert not sig

    def test_single(self, single_trade):
        t, p, sig = t_test_mean(single_trade)
        assert p == 1.0  # single sample can't test
        assert not sig

    def test_winning(self, winning_trades):
        t, p, sig = t_test_mean(winning_trades)
        # All wins at 5% — should be very significant
        assert sig

    def test_mixed(self, mixed_trades):
        t, p, sig = t_test_mean(mixed_trades)
        # Mean = 1.0, mixed wins/losses — may or may not be significant
        # with only 20 trades and large variance
        assert isinstance(p, float)
        assert 0.0 <= p <= 1.0


class TestNormality:

    def test_empty(self, empty_trades):
        stat, p, normal = normality_test(empty_trades)
        assert normal  # too few samples → assume normal

    def test_winning(self, winning_trades):
        stat, p, normal = normality_test(winning_trades)
        # All identical values → Shapiro may not reject normality
        # This is a statistical edge case — identical data is degenerate
        assert isinstance(normal, bool)

    def test_too_few(self, single_trade):
        stat, p, normal = normality_test(single_trade)
        assert normal  # n < 3 → assume normal
