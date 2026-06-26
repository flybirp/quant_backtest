"""
指标计算回归测试：固化关键指标的输入输出，确保计算逻辑不变。

运行：pytest tests/test_indicators_regression.py -v
"""
import json
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.indicators import (
    sma, ema, macd, rsi, kdj, bollinger, atr,
    zhixing_fast, zhixing_slow, volume_rank_pct, price_position_pct,
    pocket_pivot_volume, ma_cross, bbi, volume_explosion_flag
)

FIXTURES_DIR = Path(__file__).parent


def _make_sample_data(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """生成标准化测试数据"""
    rng = np.random.RandomState(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    close = 10 + np.cumsum(rng.randn(n) * 0.5)
    high = close + rng.rand(n) * 0.5
    low = close - rng.rand(n) * 0.5
    open_ = close + rng.randn(n) * 0.2
    volume = rng.randint(1000, 10000, n).astype(float)
    return pd.DataFrame({
        "date": dates, "open": open_, "high": high,
        "low": low, "close": close, "volume": volume
    })


def _load_fixture(name: str) -> dict:
    """加载固化指标结果"""
    with open(FIXTURES_DIR / f"fixture_ind_{name}.json") as f:
        return json.load(f)


def _save_fixture(name: str, data: dict):
    """保存指标结果（用于首次生成）"""
    with open(FIXTURES_DIR / f"fixture_ind_{name}.json", "w") as f:
        json.dump(data, f, indent=2)


def _series_to_list(s: pd.Series, n: int = 10) -> list:
    """取序列的前n个非NaN值用于比较"""
    vals = s.dropna().head(n).tolist()
    return [round(v, 6) for v in vals]


# === 测试用例 ===

class TestSMARegression:
    """SMA 回归测试"""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.df = _make_sample_data()

    def test_sma_values(self):
        """SMA(20) 计算结果不变"""
        result = sma(self.df["close"], 20)
        expected = _series_to_list(result, 10)
        # 首次运行时保存: _save_fixture("sma_20", {"values": expected})
        fixture = _load_fixture("sma_20")
        assert expected == fixture["values"], f"SMA(20) 变化: {expected} != {fixture['values']}"


class TestEMARegression:
    """EMA 回归测试"""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.df = _make_sample_data()

    def test_ema_values(self):
        """EMA(12) 计算结果不变"""
        result = ema(self.df["close"], 12)
        expected = _series_to_list(result, 10)
        fixture = _load_fixture("ema_12")
        assert expected == fixture["values"]


class TestMACDRegression:
    """MACD 回归测试"""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.df = _make_sample_data()

    def test_macd_values(self):
        """MACD 计算结果不变"""
        result = macd(self.df["close"])
        expected = {
            "dif": _series_to_list(result["dif"], 10),
            "dea": _series_to_list(result["dea"], 10),
            "bar": _series_to_list(result["bar"], 10),
        }
        fixture = _load_fixture("macd")
        assert expected == fixture


class TestRSIRegression:
    """RSI 回归测试"""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.df = _make_sample_data()

    def test_rsi_values(self):
        """RSI(14) 计算结果不变"""
        result = rsi(self.df["close"], 14)
        expected = _series_to_list(result, 10)
        fixture = _load_fixture("rsi_14")
        assert expected == fixture["values"]


class TestKDJRegression:
    """KDJ 回归测试"""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.df = _make_sample_data()

    def test_kdj_values(self):
        """KDJ 计算结果不变"""
        result = kdj(self.df["high"], self.df["low"], self.df["close"])
        expected = {
            "k": _series_to_list(result["k"], 10),
            "d": _series_to_list(result["d"], 10),
            "j": _series_to_list(result["j"], 10),
        }
        fixture = _load_fixture("kdj")
        assert expected == fixture


class TestZhixingRegression:
    """执行指标回归测试"""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.df = _make_sample_data()

    def test_zhixing_fast(self):
        """执行快线计算结果不变"""
        result = zhixing_fast(self.df["close"])
        expected = _series_to_list(result, 10)
        fixture = _load_fixture("zhixing_fast")
        assert expected == fixture["values"]

    def test_zhixing_slow(self):
        """执行慢线计算结果不变"""
        result = zhixing_slow(self.df["close"])
        expected = _series_to_list(result, 10)
        fixture = _load_fixture("zhixing_slow")
        assert expected == fixture["values"]


class TestVolumeRankRegression:
    """量能排名回归测试"""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.df = _make_sample_data()

    def test_volume_rank_pct(self):
        """量能排名百分位计算结果不变"""
        result = volume_rank_pct(self.df["volume"], 120)
        expected = _series_to_list(result, 10)
        fixture = _load_fixture("vol_rank_pct")
        assert expected == fixture["values"]
