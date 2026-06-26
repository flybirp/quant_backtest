"""Tests for analytics/risk.py."""

import pytest
import numpy as np

from analytics.risk import (
    max_drawdown,
    max_consecutive_losses,
    max_consecutive_wins,
    var_historical,
    cvar_historical,
    var_daily,
    cvar_daily,
    sharpe_ratio,
    ulcer_index,
    drawdown_recovery_stats,
    profit_distribution_stats,
)


class TestMaxDrawdown:

    def test_empty(self, empty_equity_curve):
        dd, start, end, dur = max_drawdown(empty_equity_curve)
        assert dd == 0.0
        assert start == ""

    def test_growing(self, growing_equity_curve):
        dd, start, end, dur = max_drawdown(growing_equity_curve)
        assert dd == 0.0  # never declines

    def test_declining(self, declining_equity_curve):
        dd, _, _, _ = max_drawdown(declining_equity_curve)
        assert dd > 0

    def test_v_shaped(self, v_shaped_equity_curve):
        dd, start, end, dur = max_drawdown(v_shaped_equity_curve)
        assert pytest.approx(dd, 0.1) == 20.0  # ~20% drawdown
        assert dur > 0

    def test_exact_drawdown(self):
        curve = [
            {"date": "2020-01-01", "equity": 100.0},
            {"date": "2020-01-02", "equity": 90.0},   # -10%
            {"date": "2020-01-03", "equity": 85.0},   # -15% from peak
            {"date": "2020-01-04", "equity": 100.0},   # recovered
        ]
        dd, _, _, _ = max_drawdown(curve)
        assert dd == 15.0


class TestConsecutiveStreaks:

    def test_all_wins(self, winning_trades):
        streak, total = max_consecutive_wins(winning_trades)
        assert streak == 10
        assert total == 50.0  # 10 × 5%

    def test_all_losses(self, losing_trades):
        streak, total = max_consecutive_losses(losing_trades)
        assert streak == 10
        assert total == -30.0

    def test_mixed(self, mixed_trades):
        # mixed_trades = 10 wins then 10 losses
        win_streak, _ = max_consecutive_wins(mixed_trades)
        loss_streak, _ = max_consecutive_losses(mixed_trades)
        assert win_streak == 10
        assert loss_streak == 10

    def test_empty(self, empty_trades):
        s, t = max_consecutive_losses(empty_trades)
        assert s == 0
        assert t == 0.0


class TestVaRCVaR:

    def test_var_historical(self, winning_trades):
        # All wins at 5% → returns = [5.0]*10
        # percentile(5) of uniform [5]*10 = 5.0
        # var = -5.0 (the "loss" is actually a negative of the smallest gain)
        var = var_historical(winning_trades, 0.95)
        assert var == -5.0

    def test_cvar_historical(self, losing_trades):
        cvar = cvar_historical(losing_trades, 0.95)
        assert cvar >= 0  # CVaR is positive for loss

    def test_var_daily_flat(self, flat_equity_curve):
        var = var_daily(flat_equity_curve, 0.95)
        assert var == 0.0  # no returns → no risk

    def test_var_daily_growing(self, growing_equity_curve):
        var = var_daily(growing_equity_curve, 0.95)
        # Growing equity: smallest daily returns may be near-zero positives
        # VaR returns positive for loss — with all gains, VaR is negative (no loss)
        assert var <= 0.0  # negative var means no tail loss

    def test_cvar_daily_v_shaped(self, v_shaped_equity_curve):
        cvar = cvar_daily(v_shaped_equity_curve, 0.95)
        assert cvar > 0  # declining phase creates tail losses


class TestSharpeRatio:

    def test_flat(self, flat_equity_curve, initial_capital):
        sh = sharpe_ratio(flat_equity_curve, initial_capital)
        assert sh == 0.0

    def test_growing(self, growing_equity_curve, initial_capital):
        sh = sharpe_ratio(growing_equity_curve, initial_capital)
        assert sh > 0

    def test_declining(self, declining_equity_curve, initial_capital):
        sh = sharpe_ratio(declining_equity_curve, initial_capital)
        assert sh < 0

    def test_sparse(self, sparse_equity_curve, initial_capital):
        # Forward-fill should make this work
        sh = sharpe_ratio(sparse_equity_curve, initial_capital)
        assert isinstance(sh, float)


class TestUlcerIndex:

    def test_flat(self, flat_equity_curve):
        assert ulcer_index(flat_equity_curve) == 0.0

    def test_growing(self, growing_equity_curve):
        assert ulcer_index(growing_equity_curve) == 0.0

    def test_declining(self, declining_equity_curve):
        ui = ulcer_index(declining_equity_curve)
        assert ui > 0  # declining → positive ulcer


class TestDrawdownRecovery:

    def test_empty(self, empty_equity_curve):
        rec = drawdown_recovery_stats(empty_equity_curve)
        assert rec["drawdown_count"] == 0

    def test_v_shaped(self, v_shaped_equity_curve):
        rec = drawdown_recovery_stats(v_shaped_equity_curve)
        assert rec["drawdown_count"] >= 1
        assert rec["max_dd_pct"] > 0
        # V-shaped: drops to 80k and recovers near 100k
        # Recovery may or may not reach the exact pre-crash peak
        assert isinstance(rec["max_recovery_days"], int)

    def test_growing_no_drawdown(self, growing_equity_curve):
        rec = drawdown_recovery_stats(growing_equity_curve)
        assert rec["drawdown_count"] == 0
        assert rec["underwater_ratio"] == 0.0


class TestProfitDistribution:

    def test_empty(self, empty_trades):
        dist = profit_distribution_stats(empty_trades)
        assert dist["count"] == 0

    def test_mixed(self, mixed_trades):
        dist = profit_distribution_stats(mixed_trades)
        assert dist["count"] == 20
        assert dist["mean"] == 1.0  # (10*5 + 10*-3)/20 = (50-30)/20 = 1.0
        assert dist["p50"] == 1.0  # median of [5]*10 + [-3]*10

    def test_nan_filtered(self, trade_with_nan):
        dist = profit_distribution_stats(trade_with_nan)
        assert dist["count"] == 2
