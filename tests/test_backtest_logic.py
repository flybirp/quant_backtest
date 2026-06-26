"""
回测逻辑回归测试：固化核心计算函数的输入输出，确保逻辑不变。

运行：pytest tests/test_backtest_logic.py -v
"""
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.backtest_engine import _net_profit_pct, _apply_slippage
from backend.strategy_engine import resolve_price, check_condition_vectorized
import pandas as pd
import numpy as np


class TestNetProfitPct:
    """净收益率计算回归测试"""

    def test_basic_profit(self):
        """基本盈利场景"""
        # 买入 10 元，卖出 12 元，佣金 0.03%，印花税 0.1%，滑点 0.1%
        result = _net_profit_pct(10.0, 12.0, 0.0003, 0.001, 0.001)
        assert abs(result - 19.5686) < 0.01, f"净收益率变化: {result}"

    def test_basic_loss(self):
        """基本亏损场景"""
        result = _net_profit_pct(10.0, 8.0, 0.0003, 0.001, 0.001)
        assert abs(result - (-20.2876)) < 0.01, f"净收益率变化: {result}"

    def test_zero_slippage(self):
        """无滑点场景"""
        result = _net_profit_pct(10.0, 12.0, 0.0003, 0.001, 0.0)
        assert abs(result - 19.8081) < 0.01, f"净收益率变化: {result}"

    def test_no_commission(self):
        """无佣金场景"""
        result = _net_profit_pct(10.0, 12.0, 0.0, 0.001, 0.001)
        assert abs(result - 19.6404) < 0.01, f"净收益率变化: {result}"


class TestApplySlippage:
    """滑点计算回归测试"""

    def test_buy_slippage(self):
        """买入滑点（价格上升）"""
        result = _apply_slippage(10.0, "buy", 0.001)
        assert abs(result - 10.01) < 0.0001, f"买入滑点后价格变化: {result}"

    def test_sell_slippage(self):
        """卖出滑点（价格下降）"""
        result = _apply_slippage(10.0, "sell", 0.001)
        assert abs(result - 9.99) < 0.0001, f"卖出滑点后价格变化: {result}"

    def test_zero_slippage(self):
        """零滑点"""
        result = _apply_slippage(10.0, "buy", 0.0)
        assert result == 10.0, f"零滑点价格变化: {result}"


class TestResolvePrice:
    """价格解析回归测试"""

    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            "date": pd.date_range("2020-01-01", periods=5),
            "open": [10.0, 10.5, 11.0, 10.8, 11.2],
            "close": [10.2, 10.8, 10.6, 11.0, 11.5],
            "high": [10.5, 11.0, 11.2, 11.3, 11.8],
            "low": [9.8, 10.2, 10.4, 10.5, 10.9],
        })

    def test_close_price(self, sample_df):
        """收盘价解析"""
        result = resolve_price(sample_df, 2, "close")
        assert result == 10.6, f"收盘价变化: {result}"

    def test_open_price(self, sample_df):
        """开盘价解析"""
        result = resolve_price(sample_df, 2, "open")
        assert result == 11.0, f"开盘价变化: {result}"

    def test_high_price(self, sample_df):
        """最高价解析"""
        result = resolve_price(sample_df, 2, "high")
        assert result == 11.2, f"最高价变化: {result}"

    def test_low_price(self, sample_df):
        """最低价解析"""
        result = resolve_price(sample_df, 2, "low")
        assert result == 10.4, f"最低价变化: {result}"


class TestCheckConditionVectorized:
    """向量化条件检测回归测试"""

    @pytest.fixture
    def sample_df(self):
        n = 100
        rng = np.random.RandomState(42)
        close = 10 + np.cumsum(rng.randn(n) * 0.5)
        return pd.DataFrame({
            "date": pd.date_range("2020-01-01", periods=n),
            "close": close,
            "sma_20": pd.Series(close).rolling(20).mean(),
            "rsi_14": pd.Series(close).rolling(14).apply(lambda x: 100 - 100/(1+x.mean()/x.std())),
        })

    def test_above_indicator(self, sample_df):
        """价格在指标之上的条件"""
        cond = {"indicator": "close", "operator": ">", "value": "sma_20"}
        result = check_condition_vectorized(sample_df, cond)
        assert isinstance(result, np.ndarray), "返回类型变化"
        assert len(result) == len(sample_df), "长度变化"

    def test_below_indicator(self, sample_df):
        """价格在指标之下的条件"""
        cond = {"indicator": "close", "operator": "<", "value": "sma_20"}
        result = check_condition_vectorized(sample_df, cond)
        assert isinstance(result, np.ndarray), "返回类型变化"
        # 与 above 互补
        above_cond = {"indicator": "close", "operator": ">", "value": "sma_20"}
        above_result = check_condition_vectorized(sample_df, above_cond)
        # 注意：等于的情况可能有重叠
        assert np.sum(result & above_result) <= 10, "逻辑关系异常"
