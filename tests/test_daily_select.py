"""Tests for daily_select_pipeline.py"""

import json
import pytest
import tempfile
from pathlib import Path
from datetime import datetime

# 添加项目根目录到 path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from daily_select_pipeline import (
    load_portfolio,
    load_strategy,
    _safe_float,
    _safe_int,
)


# ============================================================
# _safe_float / _safe_int
# ============================================================

class TestSafeConversion:
    """测试安全类型转换"""

    def test_safe_float_normal(self):
        assert _safe_float(10.5) == 10.5
        assert _safe_float(10) == 10.0
        assert _safe_float("10.5") == 10.5

    def test_safe_float_invalid(self):
        assert _safe_float("abc") == 0.0
        assert _safe_float(None) == 0.0
        assert _safe_float("") == 0.0

    def test_safe_float_default(self):
        assert _safe_float("abc", default=-1.0) == -1.0

    def test_safe_int_normal(self):
        assert _safe_int(10) == 10
        assert _safe_int(10.5) == 10
        assert _safe_int("10") == 10

    def test_safe_int_invalid(self):
        assert _safe_int("abc") == 0
        assert _safe_int(None) == 0
        assert _safe_int("") == 0

    def test_safe_int_default(self):
        assert _safe_int("abc", default=-1) == -1


# ============================================================
# load_portfolio
# ============================================================

class TestLoadPortfolio:
    """测试持仓文件加载"""

    def _write_temp(self, data: dict) -> str:
        """写入临时文件并返回路径"""
        f = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(data, f)
        f.close()
        return f.name

    def test_normal_trades_format(self):
        """正常 trades 格式"""
        path = self._write_temp({
            "trades": [
                {
                    "trade_id": "T001",
                    "code": "000001",
                    "buy_price": 10.50,
                    "shares": 1000,
                    "strategy": "RS10_A1",
                    "status": "holding"
                }
            ]
        })
        trades = load_portfolio(path, status="holding")
        assert len(trades) == 1
        assert trades[0]["code"] == "000001"
        assert trades[0]["buy_price"] == 10.50

    def test_normal_holdings_format(self):
        """兼容旧的 holdings 格式"""
        path = self._write_temp({
            "holdings": [
                {
                    "code": "000001",
                    "buy_price": 10.50,
                    "strategy": "RS10_A1"
                }
            ]
        })
        trades = load_portfolio(path, status="all")
        assert len(trades) == 1

    def test_direct_list_format(self):
        """直接列表格式"""
        path = self._write_temp([
            {
                "code": "000001",
                "buy_price": 10.50,
                "strategy": "RS10_A1"
            }
        ])
        trades = load_portfolio(path, status="all")
        assert len(trades) == 1

    def test_empty_trades(self):
        """空交易列表"""
        path = self._write_temp({"trades": []})
        trades = load_portfolio(path, status="all")
        assert len(trades) == 0

    def test_missing_code(self):
        """缺少 code 字段"""
        path = self._write_temp({
            "trades": [
                {"buy_price": 10.50, "strategy": "RS10_A1"}
            ]
        })
        trades = load_portfolio(path, status="all")
        assert len(trades) == 0

    def test_missing_strategy(self):
        """缺少 strategy 字段"""
        path = self._write_temp({
            "trades": [
                {"code": "000001", "buy_price": 10.50}
            ]
        })
        trades = load_portfolio(path, status="all")
        assert len(trades) == 0

    def test_invalid_buy_price_zero(self):
        """buy_price 为 0"""
        path = self._write_temp({
            "trades": [
                {"code": "000001", "buy_price": 0, "strategy": "RS10_A1"}
            ]
        })
        trades = load_portfolio(path, status="all")
        assert len(trades) == 0

    def test_invalid_buy_price_string(self):
        """buy_price 为非数字字符串"""
        path = self._write_temp({
            "trades": [
                {"code": "000001", "buy_price": "abc", "strategy": "RS10_A1"}
            ]
        })
        trades = load_portfolio(path, status="all")
        assert len(trades) == 0

    def test_invalid_buy_price_none(self):
        """buy_price 为 None"""
        path = self._write_temp({
            "trades": [
                {"code": "000001", "buy_price": None, "strategy": "RS10_A1"}
            ]
        })
        trades = load_portfolio(path, status="all")
        assert len(trades) == 0

    def test_status_filter_holding(self):
        """筛选持仓中"""
        path = self._write_temp({
            "trades": [
                {"code": "000001", "buy_price": 10.50, "strategy": "RS10_A1", "status": "holding"},
                {"code": "600036", "buy_price": 35.20, "strategy": "RS10_A1", "status": "sold"},
            ]
        })
        trades = load_portfolio(path, status="holding")
        assert len(trades) == 1
        assert trades[0]["code"] == "000001"

    def test_status_filter_sold(self):
        """筛选已卖出"""
        path = self._write_temp({
            "trades": [
                {"code": "000001", "buy_price": 10.50, "strategy": "RS10_A1", "status": "holding"},
                {"code": "600036", "buy_price": 35.20, "strategy": "RS10_A1", "status": "sold"},
            ]
        })
        trades = load_portfolio(path, status="sold")
        assert len(trades) == 1
        assert trades[0]["code"] == "600036"

    def test_status_filter_all(self):
        """筛选全部"""
        path = self._write_temp({
            "trades": [
                {"code": "000001", "buy_price": 10.50, "strategy": "RS10_A1", "status": "holding"},
                {"code": "600036", "buy_price": 35.20, "strategy": "RS10_A1", "status": "sold"},
            ]
        })
        trades = load_portfolio(path, status="all")
        assert len(trades) == 2

    def test_file_not_found(self):
        """文件不存在"""
        trades = load_portfolio("/tmp/nonexistent.json")
        assert len(trades) == 0

    def test_invalid_json(self):
        """无效 JSON"""
        f = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        f.write("not valid json")
        f.close()
        trades = load_portfolio(f.name)
        assert len(trades) == 0

    def test_default_values(self):
        """默认值填充"""
        path = self._write_temp({
            "trades": [
                {
                    "code": "000001",
                    "buy_price": 10.50,
                    "strategy": "RS10_A1",
                    "status": "holding"
                }
            ]
        })
        trades = load_portfolio(path, status="holding")
        assert len(trades) == 1
        assert trades[0]["trade_id"] == "T001"  # 自动生成
        assert trades[0]["name"] == ""  # 默认空
        assert trades[0]["shares"] == 0  # 默认0

    def test_multiple_trades_same_stock(self):
        """同一股票多笔交易（分批加仓）"""
        path = self._write_temp({
            "trades": [
                {"code": "000001", "buy_price": 10.50, "shares": 1000, "strategy": "RS10_A1", "status": "holding"},
                {"code": "000001", "buy_price": 10.20, "shares": 500, "strategy": "RS10_A1", "status": "holding"},
            ]
        })
        trades = load_portfolio(path, status="holding")
        assert len(trades) == 2
        assert trades[0]["buy_price"] == 10.50
        assert trades[1]["buy_price"] == 10.20


# ============================================================
# load_strategy
# ============================================================

class TestLoadStrategy:
    """测试策略加载"""

    def test_load_existing_strategy(self):
        """加载存在的策略"""
        cfg = load_strategy("RS10_A1")
        assert cfg["name"] == "RS10_A1"
        assert "buy_groups" in cfg
        assert "sell_groups" in cfg

    def test_load_nonexistent_strategy(self):
        """加载不存在的策略"""
        with pytest.raises(FileNotFoundError):
            load_strategy("NONEXISTENT_STRATEGY")


# ============================================================
# 策略一致性检查
# ============================================================

class TestStrategyConsistency:
    """测试策略一致性"""

    def _write_temp(self, data: dict) -> str:
        f = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(data, f)
        f.close()
        return f.name

    def test_strategy_mismatch_skipped(self):
        """策略不匹配时跳过"""
        path = self._write_temp({
            "trades": [
                {"code": "000001", "buy_price": 10.50, "strategy": "D1-知行金叉回踩", "status": "holding"},
                {"code": "600036", "buy_price": 35.20, "strategy": "RS10_A1", "status": "holding"},
            ]
        })
        trades = load_portfolio(path, status="holding")

        # 模拟策略一致性检查
        target_strategy = "RS10_A1"
        matched = [t for t in trades if t["strategy"] == target_strategy]
        assert len(matched) == 1
        assert matched[0]["code"] == "600036"
