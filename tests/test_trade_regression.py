"""
策略回归测试：固化交易详情，确保代码改动不改变回测结果。

每个策略固化约 50 笔代表性交易（覆盖不同退出原因和盈亏极端值）。
任何代码改动导致这些交易变化都会被捕获。

运行：pytest tests/test_trade_regression.py -v
"""
import json
import pytest
import yaml
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.main import _config_from_dict
from backend.backtest_engine import run_backtest

FIXTURES_DIR = Path(__file__).parent
PROJECT_DIR = FIXTURES_DIR.parent


def _load_fixture(name: str) -> list[dict]:
    """加载固化交易数据"""
    with open(FIXTURES_DIR / f"fixtures_{name}.json") as f:
        return json.load(f)


def _extract_fixture_scope(fixture: list[dict]) -> tuple[list[str], str, str]:
    """从 fixture 提取涉及的股票代码和日期范围"""
    codes = list(set(t["code"] for t in fixture))
    buy_dates = [t["buy_date"] for t in fixture]
    sell_dates = [t["sell_date"] for t in fixture]
    # 往前 90 天，往后 30 天 buffer
    from datetime import datetime, timedelta
    earliest = min(buy_dates)
    latest = max(sell_dates)
    start = (datetime.strptime(earliest, "%Y-%m-%d") - timedelta(days=90)).strftime("%Y-%m-%d")
    end = (datetime.strptime(latest, "%Y-%m-%d") + timedelta(days=30)).strftime("%Y-%m-%d")
    return codes, start, end


def _run_strategy(strategy_file: str, fixture: list[dict]) -> dict:
    """运行策略回测（只加载 fixture 涉及的股票和时间段），返回 {trade_key: trade_dict}"""
    yaml_path = PROJECT_DIR / "strategies" / "rule" / f"{strategy_file}.yaml"
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    codes, start_date, end_date = _extract_fixture_scope(fixture)
    cfg["stock_pool"] = codes
    config = _config_from_dict(cfg)

    result = run_backtest(config, start_date=start_date, end_date=end_date)

    # 构建 trade_key → trade 映射
    trades_map = {}
    for t in result.trades:
        code = str(t.get("code", ""))
        buy_date = str(t.get("buy_date", ""))[:10]
        key = f"{code}_{buy_date}"
        trades_map[key] = t
    return trades_map


def _trade_key(t: dict) -> str:
    """生成交易唯一键"""
    return f"{t['code']}_{t['buy_date']}"


# === 测试用例 ===

class TestDS6Regression:
    """DS6 策略回归测试（恐慌错杀-缩量+PB<3）"""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.fixture = _load_fixture("DS6")
        self.actual = _run_strategy("DS6", self.fixture)

    def test_all_fixture_trades_exist(self):
        """所有固化交易仍然存在"""
        for ft in self.fixture:
            key = _trade_key(ft)
            assert key in self.actual, f"DS6 缺失交易: {ft['code']} {ft['buy_date']}"

    def test_profit_match(self):
        """每笔交易利润不变（允许 0.5pp 浮点误差）"""
        for ft in self.fixture:
            key = _trade_key(ft)
            actual = self.actual[key]
            expected_profit = ft["profit_pct"]
            actual_profit = actual.get("profit_pct", 0)
            assert abs(actual_profit - expected_profit) < 0.5, \
                f"DS6 {ft['code']} {ft['buy_date']}: 利润变化 {expected_profit}→{actual_profit}"

    def test_sell_reason_match(self):
        """退出原因不变"""
        for ft in self.fixture:
            key = _trade_key(ft)
            actual = self.actual[key]
            expected_reason = ft["sell_reason"]
            actual_reason = actual.get("sell_reason", "")
            assert actual_reason == expected_reason, \
                f"DS6 {ft['code']} {ft['buy_date']}: 退出原因变化 '{expected_reason}'→'{actual_reason}'"


class TestFB1Regression:
    """FB1 策略回归测试（快速杀跌）"""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.fixture = _load_fixture("FB1")
        self.actual = _run_strategy("FB1", self.fixture)

    def test_all_fixture_trades_exist(self):
        for ft in self.fixture:
            key = _trade_key(ft)
            assert key in self.actual, f"FB1 缺失交易: {ft['code']} {ft['buy_date']}"

    def test_profit_match(self):
        for ft in self.fixture:
            key = _trade_key(ft)
            actual = self.actual[key]
            expected_profit = ft["profit_pct"]
            actual_profit = actual.get("profit_pct", 0)
            assert abs(actual_profit - expected_profit) < 0.5, \
                f"FB1 {ft['code']} {ft['buy_date']}: 利润变化 {expected_profit}→{actual_profit}"

    def test_sell_reason_match(self):
        for ft in self.fixture:
            key = _trade_key(ft)
            actual = self.actual[key]
            assert actual.get("sell_reason", "") == ft["sell_reason"], \
                f"FB1 {ft['code']} {ft['buy_date']}: 退出原因变化"


class TestZzh73Regression:
    """zzh7.3 策略回归测试（状态机模式）"""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.fixture = _load_fixture("zzh73")
        self.actual = _run_strategy("zzh7.3", self.fixture)

    def test_all_fixture_trades_exist(self):
        for ft in self.fixture:
            key = _trade_key(ft)
            assert key in self.actual, f"zzh7.3 缺失交易: {ft['code']} {ft['buy_date']}"

    def test_profit_match(self):
        for ft in self.fixture:
            key = _trade_key(ft)
            actual = self.actual[key]
            expected_profit = ft["profit_pct"]
            actual_profit = actual.get("profit_pct", 0)
            assert abs(actual_profit - expected_profit) < 0.5, \
                f"zzh7.3 {ft['code']} {ft['buy_date']}: 利润变化 {expected_profit}→{actual_profit}"

    def test_sell_reason_match(self):
        for ft in self.fixture:
            key = _trade_key(ft)
            actual = self.actual[key]
            assert actual.get("sell_reason", "") == ft["sell_reason"], \
                f"zzh7.3 {ft['code']} {ft['buy_date']}: 退出原因变化"
