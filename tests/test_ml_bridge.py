"""Tests for backend/ml_bridge.py."""

import pytest
import tempfile
import os
from pathlib import Path

# Must add backend to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.ml_bridge import (
    load_ml_predictions,
    filter_trades_by_ml,
    MLComparisonResult,
    _compute_ev,
    _compute_win_rate,
    _build_equity_curve,
)


class TestLoadMLPredictions:

    def test_empty_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            preds = load_ml_predictions(tmpdir)
            assert preds == {}

    def test_single_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "pred_20200101.csv")
            with open(csv_path, "w") as f:
                f.write("date_str,code,ensemble_score,pred_a\n")
                f.write("20200101,000001,0.45,0.50\n")
                f.write("20200101,000002,0.30,0.35\n")
                f.write("20200102,000001,0.60,0.55\n")

            preds = load_ml_predictions(tmpdir, score_col="ensemble_score")
            assert len(preds) == 2  # 2 dates
            assert "2020-01-01" in preds
            assert preds["2020-01-01"]["000001"] == 0.45
            assert preds["2020-01-01"]["000002"] == 0.30
            assert preds["2020-01-02"]["000001"] == 0.60

    def test_date_normalization(self):
        """8-digit dates should be normalized to YYYY-MM-DD."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "pred.csv")
            with open(csv_path, "w") as f:
                f.write("date_str,code,ensemble_score\n")
                f.write("20200101,000001,0.45\n")

            preds = load_ml_predictions(tmpdir, score_col="ensemble_score")
            assert "2020-01-01" in preds

    def test_code_normalization(self):
        """Codes with .0 suffix should be cleaned."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "pred.csv")
            with open(csv_path, "w") as f:
                f.write("date_str,code,ensemble_score\n")
                f.write("20200101,000001.0,0.45\n")

            preds = load_ml_predictions(tmpdir, score_col="ensemble_score")
            assert preds["2020-01-01"]["000001"] == 0.45

    def test_multiple_csvs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(3):
                csv_path = os.path.join(tmpdir, f"pred_{i}.csv")
                with open(csv_path, "w") as f:
                    f.write("date_str,code,ensemble_score\n")
                    f.write(f"2020010{i+1},00000{i+1},0.{i+1}\n")

            preds = load_ml_predictions(tmpdir, score_col="ensemble_score")
            assert len(preds) == 3


class TestFilterTradesByML:

    def test_no_predictions(self, mixed_trades):
        passed, filtered = filter_trades_by_ml(mixed_trades, {}, 0.3)
        assert len(passed) == len(mixed_trades)
        assert len(filtered) == 0  # no predictions → all pass

    def test_filter_by_threshold(self):
        preds = {
            "2020-01-02": {"000001": 0.5},
            "2020-01-03": {"000002": 0.1},
        }
        trades = [
            {"code": "000001", "buy_date": "2020-01-02", "profit_pct": 5.0},
            {"code": "000002", "buy_date": "2020-01-03", "profit_pct": -3.0},
        ]
        passed, filtered = filter_trades_by_ml(trades, preds, 0.3)
        assert len(passed) == 1
        assert passed[0]["code"] == "000001"
        assert len(filtered) == 1
        assert filtered[0]["code"] == "000002"

    def test_missing_code_kept(self):
        preds = {"2020-01-02": {"000001": 0.5}}
        trades = [
            {"code": "000001", "buy_date": "2020-01-02", "profit_pct": 5.0},
            {"code": "000999", "buy_date": "2020-01-02", "profit_pct": 3.0},
        ]
        passed, filtered = filter_trades_by_ml(trades, preds, 0.3)
        assert len(passed) == 2  # unmatched trade kept
        assert len(filtered) == 0

    def test_missing_date_kept(self):
        preds = {"2020-01-02": {"000001": 0.5}}
        trades = [
            {"code": "000001", "buy_date": "2020-01-02", "profit_pct": 5.0},
            {"code": "000001", "buy_date": "2020-03-01", "profit_pct": 3.0},
        ]
        passed, filtered = filter_trades_by_ml(trades, preds, 0.3)
        assert len(passed) == 2


class TestHelpers:

    def test_compute_ev(self, mixed_trades):
        ev = _compute_ev(mixed_trades)
        assert ev == 1.0  # (5*10 - 3*10)/20

    def test_compute_ev_empty(self):
        assert _compute_ev([]) == 0.0

    def test_win_rate(self, mixed_trades):
        wr = _compute_win_rate(mixed_trades)
        assert wr == 50.0

    def test_win_rate_empty(self):
        assert _compute_win_rate([]) == 0.0

    def test_build_equity_curve(self, winning_trades):
        curve = _build_equity_curve(winning_trades, 100000.0)
        assert len(curve) == 10
        # 10 trades at +5% each → cumulative +50%
        final_equity = curve[-1]["equity"]
        assert final_equity == pytest.approx(150000.0, rel=0.01)


class TestMLComparisonResult:

    def test_ev_change(self):
        c = MLComparisonResult(raw_ev=4.0, filtered_ev=5.0)
        assert c.ev_change_pct == 25.0

    def test_sharpe_change_zero_denom(self):
        c = MLComparisonResult(raw_sharpe=0.0, filtered_sharpe=0.0)
        assert c.sharpe_change_pct == 0.0
