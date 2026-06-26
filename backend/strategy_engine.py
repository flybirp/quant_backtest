"""策略引擎 — 配置驱动的买卖信号 + 持仓管理"""

from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Any, Optional
from dataclasses import dataclass, field

from backend.indicators import compute_all_indicators


@dataclass
class Signal:
    """单个买卖信号"""
    date: Any
    code: str
    signal_type: str  # "buy" | "sell" | "add" | "reduce" | "clear" | "stop_loss" | "take_profit"
    price: float
    reason: str = ""


@dataclass
class Trade:
    """一笔完整交易"""
    code: str
    buy_date: Any
    buy_price: float
    sell_date: Optional[Any] = None
    sell_price: Optional[float] = None
    sell_reason: str = ""
    shares: int = 0
    profit_pct: float = 0.0
    profit_amount: float = 0.0
    hold_days: int = 0
    action: str = "buy"  # "buy" | "add" | "reduce" | "clear"
    trade_id: str = ""   # 唯一ID：标识从建仓到清仓的完整交易


@dataclass
class StrategyConfig:
    """策略配置

    strategy_type:
        'rule' — 规则型策略（条件组 + 状态机），默认
        'ml'   — 多因子机器学习型策略
    """
    name: str = "default"
    strategy_type: str = "rule"  # 'rule' | 'ml'
    k_type: str = "daily"  # daily / weekly
    backtest_mode: str = "signal"  # signal(信号纯评估) / portfolio(资金组合)
    initial_capital: float = 100000

    # ── 规则型字段 ──
    # 买入条件：多个group之间OR，group内AND
    buy_groups: list[dict] = field(default_factory=list)
    # 卖出条件：同上（清仓）
    sell_groups: list[dict] = field(default_factory=list)
    # 加仓条件：已持仓时触发加仓
    add_groups: list[dict] = field(default_factory=list)
    # 减仓条件：部分卖出
    reduce_groups: list[dict] = field(default_factory=list)

    # 仓位管理
    position_pct: float = 1.0  # 单只股票仓位比例（总资金占比）
    max_positions: int = 5  # 最大同时持仓数
    add_threshold: float = 0.0  # 浮盈加仓阈值（%），0=不加仓
    add_pct: float = 0.0  # 加仓仓位比例
    reduce_pct: float = 0.5  # 减仓比例（卖出持仓的百分比）
    min_trade_unit: int = 10  # 最小交易单位（股），减仓数量向下取整到该倍数，剩余≤此值时清仓

    # 风控
    stop_loss_pct: float = 5.0  # 止损线 %
    take_profit_pct: float = 15.0  # 止盈线 %
    max_hold_days: int = 0  # 最大持仓天数，0=不限制
    trailing_stop_pct: float = 0.0  # 移动止损（从最高点回落%），0=不启用
    trailing_stop_tiers: dict[float, float] = field(default_factory=dict)  # {profit%: trail%} e.g. {0:25, 50:15}

    # 过滤条件
    min_volume_ratio: float = 0.0  # 最小成交量相对20日均量比
    stock_pool: list[str] = field(default_factory=list)  # 股票池，空=全部

    # 成交价策略
    buy_price_type: str = "close"   # open/high/low/close/avg/typical/vwap
    sell_price_type: str = "close"  # 同上
    buy_execution: str = "same_day"  # same_day / next_day（T+1执行）
    sell_execution: str = "same_day" # 同上

    # 状态机策略（优先级高于 buy/sell/add/reduce groups）
    state_machine: str = ""  # 空=使用传统条件组模式
    state_machine_params: dict = field(default_factory=dict)

    # 排他锁：同一标的已有持仓时是否阻止开新仓（默认开启）
    exclusive_lock: bool = True

    # 交易成本
    commission_rate: float = 0.0003   # 佣金费率（买卖双向），默认万三
    stamp_tax_rate: float = 0.001     # 印花税费率（仅卖出），默认千一
    slippage_pct: float = 0.001       # 滑点（买卖双向），默认千一，0=不启用
    limit_filter: bool = True         # 涨跌停过滤：涨停无法买入，跌停无法卖出

    # 分批建仓 (Entry Ladder)
    # 每层: {"trigger_pct": float, "weight": int}
    # trigger_pct: 从首笔买入价计算的浮盈百分比触发加仓，0=首笔买入
    # weight: 该层仓位权重（如50=50%仓位）
    # 例: [{"trigger_pct": 0, "weight": 50}, {"trigger_pct": 3, "weight": 30}, {"trigger_pct": 6, "weight": 20}]
    #   → 首笔50%，浮盈+3%加30%，浮盈+6%加20%
    entry_ladder: list[dict] = field(default_factory=list)

    # 分批止盈 (Exit Ladder)
    # 每层: {"profit_pct": float, "close_pct": float}
    # profit_pct: 浮盈达到该百分比时触发分批止盈
    # close_pct: 关闭该笔仓位剩余量的百分比（如50=卖出剩余的50%）
    # 例: [{"profit_pct": 5, "close_pct": 50}, {"profit_pct": 10, "close_pct": 50}]
    #   → 浮盈+5%时卖出50%，浮盈+10%时再卖出剩余的50%（即总仓位的25%）
    #   最后剩余25%由trailing_stop/止盈止损/卖出信号管理
    exit_ladder: list[dict] = field(default_factory=list)

    # ── ML 型字段 ──
    # 因子列表：每个因子包含 name, lookback_days, transform (zscore/rank/raw)
    factor_list: list[dict] = field(default_factory=list)
    # 模型类型：'linear', 'xgboost', 'lightgbm', 'mlp'
    model_type: str = ""
    # 模型文件路径（训练后保存）
    model_path: str = ""
    # 标签定义：'future_return_5d', 'future_return_20d', 'binary_up_down'
    label_type: str = "future_return_5d"
    # 标签前瞻天数
    label_horizon_days: int = 5
    # 特征回溯窗口天数
    lookback_days: int = 252
    # 调仓频率：'daily', 'weekly', 'monthly'
    rebalance_freq: str = "weekly"
    # 选股数量（top N by predicted score）
    top_n: int = 20
    # 训练参数（传给具体模型的超参）
    model_params: dict = field(default_factory=dict)


def resolve_price(df: pd.DataFrame, idx: int, price_type: str) -> float:
    """根据价格类型解析成交价

    支持: open, high, low, close, avg=(high+low)/2,
          typical=(high+low+close)/3, vwap=(open+high+low+close)/4
    """
    row = df.iloc[idx]
    if price_type == "open":
        return row["open"]
    elif price_type == "high":
        return row["high"]
    elif price_type == "low":
        return row["low"]
    elif price_type == "close":
        return row["close"]
    elif price_type == "avg":
        return (row["high"] + row["low"]) / 2
    elif price_type == "typical":
        return (row["high"] + row["low"] + row["close"]) / 3
    elif price_type == "vwap":
        return (row["open"] + row["high"] + row["low"] + row["close"]) / 4
    return row["close"]


def _find_double_bottom_support(df: pd.DataFrame, idx: int,
                                lookback: int = 20,
                                tolerance_pct: float = 3.0,
                                bounce_pct: float = 3.0) -> Optional[float]:
    """在当前idx位置查找双底支撑位，返回支撑价或None"""
    min_separation = 3
    if idx < lookback:
        return None
    low_arr = df["low"].values.astype(float)
    close_arr = df["close"].values.astype(float)
    start = idx - lookback
    window_low = low_arr[start:idx + 1]
    window_close = close_arr[start:idx + 1]

    # 找局部低点：比前后各2天都低
    local_mins = []
    for j in range(2, len(window_low) - 2):
        if (window_low[j] <= window_low[j - 1] and window_low[j] <= window_low[j - 2]
            and window_low[j] <= window_low[j + 1] and window_low[j] <= window_low[j + 2]):
            local_mins.append((j, window_low[j]))

    if len(local_mins) < 2:
        return None

    for k in range(len(local_mins) - 1, 0, -1):
        for m in range(k - 1, -1, -1):
            j2, low2 = local_mins[k]
            j1, low1 = local_mins[m]
            if j2 - j1 < min_separation:
                continue
            ref = max(low1, low2)
            if ref <= 0:
                continue
            if abs(low1 - low2) / ref * 100 > tolerance_pct:
                continue
            between_close = window_close[j1 + 1:j2]
            neck = max(low1, low2)
            if len(between_close) == 0:
                continue
            max_between = between_close.max()
            bounce = (max_between - neck) / neck * 100
            if bounce < bounce_pct:
                continue
            actual_j2 = start + j2
            if idx - actual_j2 > 5:
                continue
            return (low1 + low2) / 2
    return None


def check_condition(df: pd.DataFrame, idx: int, cond: dict) -> bool:
    """检查单个条件是否满足"""
    indicator = cond.get("indicator", "")
    params = cond.get("params", {})

    # 数据不够，不满足
    if idx < 60:
        return False

    try:
        # === 均线类 ===
        if indicator == "ma_above":
            # 短期均线在长期均线上方
            fast = params.get("fast", 5)
            slow = params.get("slow", 20)
            return df[f"ma{fast}"].iloc[idx] > df[f"ma{slow}"].iloc[idx]

        if indicator == "ma_below":
            fast = params.get("fast", 5)
            slow = params.get("slow", 20)
            return df[f"ma{fast}"].iloc[idx] < df[f"ma{slow}"].iloc[idx]

        if indicator == "ma_cross_up":
            # 金叉：当日
            fast = params.get("fast", 5)
            slow = params.get("slow", 20)
            cur = df[f"ma{fast}"].iloc[idx] > df[f"ma{slow}"].iloc[idx]
            prev = df[f"ma{fast}"].iloc[idx - 1] <= df[f"ma{slow}"].iloc[idx - 1]
            return cur and prev

        if indicator == "ma_cross_down":
            fast = params.get("fast", 5)
            slow = params.get("slow", 20)
            cur = df[f"ma{fast}"].iloc[idx] < df[f"ma{slow}"].iloc[idx]
            prev = df[f"ma{fast}"].iloc[idx - 1] >= df[f"ma{slow}"].iloc[idx - 1]
            return cur and prev

        if indicator == "price_above_ma":
            period = params.get("period", 20)
            return df["close"].iloc[idx] > df[f"ma{period}"].iloc[idx]

        if indicator == "price_below_ma":
            period = params.get("period", 20)
            return df["close"].iloc[idx] < df[f"ma{period}"].iloc[idx]

        # === MACD ===
        if indicator == "macd_golden_cross":
            cur = df["macd_dif"].iloc[idx] > df["macd_dea"].iloc[idx]
            prev = df["macd_dif"].iloc[idx - 1] <= df["macd_dea"].iloc[idx - 1]
            return cur and prev

        if indicator == "macd_dead_cross":
            cur = df["macd_dif"].iloc[idx] < df["macd_dea"].iloc[idx]
            prev = df["macd_dif"].iloc[idx - 1] >= df["macd_dea"].iloc[idx - 1]
            return cur and prev

        if indicator == "macd_above_zero":
            return df["macd_dif"].iloc[idx] > 0

        if indicator == "macd_below_zero":
            return df["macd_dif"].iloc[idx] < 0

        # === RSI ===
        if indicator == "rsi_oversold":
            period = params.get("period", 14)
            threshold = params.get("threshold", 30)
            return df[f"rsi{period}"].iloc[idx] < threshold

        if indicator == "rsi_overbought":
            period = params.get("period", 14)
            threshold = params.get("threshold", 70)
            return df[f"rsi{period}"].iloc[idx] > threshold

        # === KDJ ===
        if indicator == "kdj_golden_cross":
            cur = df["kdj_k"].iloc[idx] > df["kdj_d"].iloc[idx]
            prev = df["kdj_k"].iloc[idx - 1] <= df["kdj_d"].iloc[idx - 1]
            return cur and prev

        if indicator == "kdj_dead_cross":
            cur = df["kdj_k"].iloc[idx] < df["kdj_d"].iloc[idx]
            prev = df["kdj_k"].iloc[idx - 1] >= df["kdj_d"].iloc[idx - 1]
            return cur and prev

        if indicator == "kdj_j_oversold":
            threshold = params.get("threshold", 0)
            return df["kdj_j"].iloc[idx] < threshold

        if indicator == "kdj_j_overbought":
            threshold = params.get("threshold", 100)
            return df["kdj_j"].iloc[idx] > threshold

        # === 布林带 ===
        if indicator == "bb_lower_touch":
            return df["close"].iloc[idx] <= df["bb_lower"].iloc[idx] * 1.02

        if indicator == "bb_upper_touch":
            return df["close"].iloc[idx] >= df["bb_upper"].iloc[idx] * 0.98

        if indicator == "bb_mid_break":
            return df["close"].iloc[idx] > df["bb_mid"].iloc[idx] and df["close"].iloc[idx - 1] <= df["bb_mid"].iloc[idx - 1]

        # === 成交量 ===
        if indicator == "volume_breakout":
            ratio = params.get("ratio", 1.5)
            return df["volume"].iloc[idx] > df["vol_ma20"].iloc[idx] * ratio

        if indicator == "volume_shrink":
            ratio = params.get("ratio", 0.5)
            return df["volume"].iloc[idx] < df["vol_ma20"].iloc[idx] * ratio

        # === 趋势类 ===
        if indicator == "new_high":
            period = params.get("period", 20)
            return df["close"].iloc[idx] == df["close"].iloc[max(0, idx - period + 1):idx + 1].max()

        if indicator == "new_low":
            period = params.get("period", 20)
            return df["close"].iloc[idx] == df["close"].iloc[max(0, idx - period + 1):idx + 1].min()

        if indicator == "consecutive_up":
            n = params.get("n", 3)
            if idx < n:
                return False
            for i in range(1, n + 1):
                if df["close"].iloc[idx - i + 1] <= df["close"].iloc[idx - i]:
                    return False
            return True

        if indicator == "consecutive_down":
            n = params.get("n", 3)
            if idx < n:
                return False
            for i in range(1, n + 1):
                if df["close"].iloc[idx - i + 1] >= df["close"].iloc[idx - i]:
                    return False
            return True

        if indicator == "under_slow_days":
            """连续N天收盘价低于zhixing_slow"""
            n = params.get("days", 3)
            if idx < n:
                return False
            for i in range(n):
                if df["close"].iloc[idx - i] >= df["zhixing_slow"].iloc[idx - i]:
                    return False
            return True

        if indicator == "stabilized":
            """连续N天未创新低(收盘价不低于前N日最低)"""
            n = params.get("days", 3)
            if idx < n:
                return False
            recent_low = df["low"].iloc[idx - n + 1:idx + 1].min()
            return df["close"].iloc[idx] > recent_low * 1.01  # 高于近期低点1%以上

        # === 个股恐慌指标(discover.md) ===

        if indicator == "amplitude_gt":
            """当日振幅 > 阈值"""
            threshold = params.get("threshold", 6.0)
            amp = (df["high"].iloc[idx] - df["low"].iloc[idx]) / df["close"].iloc[idx] * 100
            return amp > threshold

        if indicator == "stock_deep_dd":
            """个股120天内最大跌幅 > 阈值"""
            lookback = params.get("lookback", 120)
            threshold = params.get("threshold", 30.0)
            if idx < lookback:
                return False
            peak = df["close"].iloc[idx - lookback + 1:idx + 1].max()
            dd = (df["close"].iloc[idx] - peak) / peak * 100
            return dd < -threshold

        if indicator == "dd_concentration":
            """最后30天跌幅占总跌幅>60%——杀跌集中在末期"""
            if idx < 120:
                return False
            total_dd_peak = df["close"].iloc[idx - 119:idx + 1].max()
            total_dd = (df["close"].iloc[idx] - total_dd_peak) / total_dd_peak * 100
            if total_dd >= 0:
                return False
            # 最后30天: 从30天前高点到现在
            peak_30 = df["close"].iloc[idx - 29:idx + 1].max()
            dd_30 = (df["close"].iloc[idx] - peak_30) / peak_30 * 100
            ratio = params.get("ratio", 60.0)
            return abs(dd_30) / abs(total_dd) * 100 > ratio

        if indicator == "volume_contracting":
            """前短窗口均量 < 前长窗口均量 × ratio"""
            short = params.get("short_window", 30)
            long = params.get("long_window", 60)
            ratio = params.get("ratio", 0.7)
            if idx < long + short:
                return False
            vol_short = df["volume"].iloc[idx - short:idx].mean()
            vol_long = df["volume"].iloc[idx - long - short:idx - short].mean()
            if vol_long == 0:
                return False
            return vol_short < vol_long * ratio

        # === 涨跌幅 ===
        if indicator == "pct_change_gt":
            pct = params.get("pct", 5)
            return df["pct_change"].iloc[idx] > pct

        if indicator == "pct_change_lt":
            pct = params.get("pct", -5)
            return df["pct_change"].iloc[idx] < pct

        # === 知行量化指标 ===
        if indicator == "zhixing_fast_above_slow":
            return df["zhixing_fast"].iloc[idx] > df["zhixing_slow"].iloc[idx]

        if indicator == "zhixing_fast_below_slow":
            return df["zhixing_fast"].iloc[idx] < df["zhixing_slow"].iloc[idx]

        if indicator == "zhixing_golden_cross":
            cur = df["zhixing_fast"].iloc[idx] > df["zhixing_slow"].iloc[idx]
            prev = df["zhixing_fast"].iloc[idx - 1] <= df["zhixing_slow"].iloc[idx - 1]
            return cur and prev

        if indicator == "zhixing_dead_cross":
            cur = df["zhixing_fast"].iloc[idx] < df["zhixing_slow"].iloc[idx]
            prev = df["zhixing_fast"].iloc[idx - 1] >= df["zhixing_slow"].iloc[idx - 1]
            return cur and prev

        if indicator == "zhixing_golden_cross_hold":
            # 金叉后连续N日站稳 fast > slow
            n_hold = params.get("days", 3)
            if idx < n_hold:
                return False
            for i in range(n_hold):
                if df["zhixing_fast"].iloc[idx - i] <= df["zhixing_slow"].iloc[idx - i]:
                    return False
            return True

        if indicator == "price_near_zhixing":
            # 价格回调至zhixing_fast/slow/ma60 ±3%范围内
            tolerance = params.get("tolerance", 3)
            targets = []
            targets.append(df["zhixing_fast"].iloc[idx])
            targets.append(df["zhixing_slow"].iloc[idx])
            if "ma60" in df.columns:
                targets.append(df["ma60"].iloc[idx])
            close_val = df["close"].iloc[idx]
            for t in targets:
                if t > 0 and abs((close_val - t) / t) * 100 <= tolerance:
                    return True
            return False

        if indicator == "zz_spread_gt":
            threshold = params.get("threshold", 60)
            return df["zz_spread"].iloc[idx] > threshold

        if indicator == "zz_short_lt":
            threshold = params.get("threshold", 30)
            return df["zz_short"].iloc[idx] < threshold

        # === 量能异动条件 ===
        if indicator == "pocket_pivot":
            # 口袋支点量能条件: 当日成交量 > 前N日下跌日最大成交量
            lookback = params.get("lookback", 10)
            if idx < lookback:
                return False
            vol = df["volume"].iloc[idx]
            window = df.iloc[idx - lookback:idx]
            down_mask = window["close"] < window["close"].shift(1)
            down_vols = window.loc[down_mask, "volume"]
            if len(down_vols) > 0:
                return vol > down_vols.max()
            else:
                return vol > 0  # 无下跌日，有量即通过

        if indicator == "low_position":
            # 低位判断：价格120日百分位<阈值 或 距MA60<阈值
            price_pct_threshold = params.get("price_pct", 40)
            ma60_dist_threshold = params.get("ma60_dist", 5)
            price_low = df["price_position_pct"].iloc[idx] < price_pct_threshold
            near_ma60 = df["dist_to_ma60"].iloc[idx] < ma60_dist_threshold
            return price_low or near_ma60

        if indicator == "volume_double":
            # 倍量柱：当日量 >= 前日2倍
            return df["is_double_vol"].iloc[idx] == 1

        if indicator == "volume_top5":
            # 历史极量：120日内top5%
            return df["is_vol_top5"].iloc[idx] == 1

        if indicator == "volume_anomaly":
            # 量能异动：倍量柱 或 top5 或 1.5x均量
            double = df["is_double_vol"].iloc[idx] == 1
            top5 = df["is_vol_top5"].iloc[idx] == 1
            ratio = params.get("ratio", 1.5)
            above_avg = df["volume"].iloc[idx] > df["vol_ma20"].iloc[idx] * ratio
            return double or top5 or above_avg

        if indicator == "volume_anomaly_strong":
            # 强异动：倍量柱 + top5 同时满足
            double = df["is_double_vol"].iloc[idx] == 1
            top5 = df["is_vol_top5"].iloc[idx] == 1
            return double and top5

        if indicator == "sideway_shrink":
            # 横盘缩量: N天振幅<N% + 每日量均<20日均量*ratio
            n_days = params.get("days", 3)
            amplitude_threshold = params.get("amplitude", 3.0)
            vol_ratio_threshold = params.get("vol_ratio", 0.4)
            if idx < n_days:
                return False
            # 横盘: 最近N天最高最低振幅<阈值
            highs = df["high"].iloc[idx - n_days + 1:idx + 1]
            lows = df["low"].iloc[idx - n_days + 1:idx + 1]
            period_low = lows.min()
            if period_low <= 0:
                return False
            amplitude = (highs.max() - period_low) / period_low * 100
            if amplitude >= amplitude_threshold:
                return False
            # 缩量: 每日量均<20日均量*ratio
            vol_ma20_val = df["vol_ma20"].iloc[idx]
            if vol_ma20_val <= 0:
                return False
            volumes = df["volume"].iloc[idx - n_days + 1:idx + 1]
            ratio_threshold = vol_ma20_val * vol_ratio_threshold
            return (volumes < ratio_threshold).all()

        if indicator == "not_distribution":
            # 排除出货: 非涨停倍量+局部高点无量异动
            # 涨停: 涨幅>9.5%
            if idx < 1:
                return True
            pct = df["pct_change"].iloc[idx]
            is_double = df["is_double_vol"].iloc[idx] == 1
            if pct > 9.5 and is_double:
                return False
            # 局部高位无量异动（简化: 120日位置>80%+倍量）
            if df["price_position_pct"].iloc[idx] > 80 and is_double:
                return False
            return True

        # === 多均线多头排列 ===
        if indicator == "ma_bullish_alignment":
            periods = params.get("periods", [5, 10, 20, 60])
            vals = [df[f"ma{p}"].iloc[idx] for p in periods]
            return all(vals[i] > vals[i + 1] for i in range(len(vals) - 1))

        if indicator == "ma_bearish_alignment":
            periods = params.get("periods", [5, 10, 20, 60])
            vals = [df[f"ma{p}"].iloc[idx] for p in periods]
            return all(vals[i] < vals[i + 1] for i in range(len(vals) - 1))

        # ================================================================
        # === 威科夫 (Wyckoff) 信号 ===
        # ================================================================

        # --- 1. Spring 弹簧效应（吸筹阶段C，最佳低吸买点）---
        # 前一日跌破近期支撑位，当日收回支撑位上方
        # 量能确认：跌破日缩量（弱卖压=洗盘）或反转日放量（主力入场）
        # spring_type: 1=Terminal Shakeout(>3%破位+放量), 2=Moderate(1-3%破位),
        #              3=Minor(<1%破位+缩量，最安全75-85%成功率)
        if indicator == "spring_shakeout":
            n = params.get("support_period", 20)      # N日低点作为支撑
            support_type = params.get("support_type", "low")  # low / ma60
            vol_confirm = params.get("vol_confirm", False)    # 是否需要量能确认
            spring_type = params.get("spring_type", 0)        # 0=任意, 1/2/3
            if idx < n + 1:
                return False
            # 支撑位：取前N日低点（不含前一日，避免前一日即为新低导致永远不触发）
            if support_type == "ma60":
                support = df["ma60"].iloc[idx - 1]
            else:
                support = df["low"].iloc[idx - n:idx - 1].min()
            if support <= 0:
                return False
            # 前一日跌破支撑
            prev_low = df["low"].iloc[idx - 1]
            broke_support = prev_low < support
            # 当日收回支撑上方
            recovered = df["close"].iloc[idx] > support
            if not (broke_support and recovered):
                return False
            # 破位幅度%
            penetration_pct = (support - prev_low) / support * 100
            # Spring类型过滤
            if spring_type == 1:
                # Terminal Shakeout: >3%破位, 高量(150%+均量), 宽幅
                if penetration_pct < 3:
                    return False
                if df["volume"].iloc[idx - 1] < df["vol_ma20"].iloc[idx - 1] * 1.5:
                    return False
            elif spring_type == 2:
                # Moderate Spring: 1-3%破位
                if penetration_pct < 1 or penetration_pct > 3:
                    return False
            elif spring_type == 3:
                # Minor Spring: <1%破位, 缩量(70%均量以下), 最安全
                if penetration_pct > 1:
                    return False
                if df["volume"].iloc[idx - 1] > df["vol_ma20"].iloc[idx - 1] * 0.7:
                    return False
            if vol_confirm:
                # 反转日放量（主力入场）
                return df["volume"].iloc[idx] > df["vol_ma20"].iloc[idx] * 1.2
            return True

        # --- 2. SOS 强势信号 / JOC 跳空突破（吸筹阶段D，放量突破阻力）---
        # 放量突破近期高点，涨幅显著
        if indicator == "sign_of_strength":
            n = params.get("breakout_period", 20)     # 突破N日高点
            vol_ratio = params.get("vol_ratio", 1.5)  # 量比阈值
            min_pct = params.get("min_pct", 2.0)      # 最低涨幅%
            wide_spread = params.get("wide_spread", False)  # 是否要求宽价差阳线
            if idx < n:
                return False
            recent_high = df["high"].iloc[idx - n:idx].max()
            breakout = df["close"].iloc[idx] > recent_high
            vol_break = df["volume"].iloc[idx] > df["vol_ma20"].iloc[idx] * vol_ratio
            strong_move = df["pct_change"].iloc[idx] > min_pct
            result = breakout and vol_break and strong_move
            if result and wide_spread:
                # 宽价差阳线：实体占全幅>60%
                body = abs(df["close"].iloc[idx] - df["open"].iloc[idx])
                full_range = df["high"].iloc[idx] - df["low"].iloc[idx]
                if full_range > 0:
                    result = body / full_range > 0.6
            return result

        # --- 3. LPS 最后支撑点（吸筹阶段D，突破后缩量回踩=加仓点）---
        # 价格在MA60上方，回调至均线附近，缩量
        if indicator == "last_point_support":
            pullback_to = params.get("pullback_to", "ma20")  # ma20 / ma10 / zhixing_fast
            vol_shrink = params.get("vol_shrink", 0.7)       # 回调缩量比
            tolerance = params.get("tolerance", 3)            # 距均线容差%
            higher_low = params.get("higher_low", False)      # 是否要求高于前低
            if idx < 20:
                return False
            # 价格在MA60上方（中期趋势向上）
            if df["close"].iloc[idx] < df["ma60"].iloc[idx]:
                return False
            # 回调至目标线
            if pullback_to == "ma10":
                target = df["ma10"].iloc[idx]
            elif pullback_to == "zhixing_fast":
                target = df["zhixing_fast"].iloc[idx]
            else:
                target = df["ma20"].iloc[idx]
            if target <= 0:
                return False
            near_target = abs((df["close"].iloc[idx] - target) / target * 100) <= tolerance
            # 缩量回调
            shrink = df["volume"].iloc[idx] < df["vol_ma20"].iloc[idx] * vol_shrink
            result = near_target and shrink
            if result and higher_low:
                # 要求当日低点高于前5日最低（形成更高低点）
                if idx >= 5:
                    prev_5_low = df["low"].iloc[idx - 5:idx].min()
                    result = df["low"].iloc[idx] > prev_5_low
            return result

        # --- 4. UTAD 上冲回落（派发阶段C，假突破=最佳卖点）---
        # 价格处于高位，冲破近期高点但收盘回落
        if indicator == "upthrust_distribution":
            n = params.get("resist_period", 20)       # N日高点作为阻力
            high_position = params.get("high_position", 70)  # 价格位置百分位阈值
            vol_confirm = params.get("vol_confirm", False)   # 是否需要放量
            if idx < n:
                return False
            # 处于高位
            if df["price_position_pct"].iloc[idx] < high_position:
                return False
            # 当日冲破阻力但收盘回落
            recent_high = df["high"].iloc[idx - n:idx].max()
            if recent_high <= 0:
                return False
            thrust_up = df["high"].iloc[idx] > recent_high
            close_below = df["close"].iloc[idx] < recent_high
            if not (thrust_up and close_below):
                return False
            if vol_confirm:
                return df["volume"].iloc[idx] > df["vol_ma20"].iloc[idx] * 1.2
            return True

        # --- 5. SOW 弱势信号（派发阶段D，放量跌破支撑）---
        # 放量跌破近期低点
        if indicator == "sign_of_weakness":
            n = params.get("support_period", 20)
            vol_ratio = params.get("vol_ratio", 1.5)
            min_drop = params.get("min_drop", -3.0)   # 最低跌幅%
            if idx < n:
                return False
            recent_low = df["low"].iloc[idx - n:idx].min()
            breakdown = df["close"].iloc[idx] < recent_low
            vol_break = df["volume"].iloc[idx] > df["vol_ma20"].iloc[idx] * vol_ratio
            sharp_drop = df["pct_change"].iloc[idx] < min_drop
            return breakdown and (vol_break or sharp_drop)

        # --- 6. LPSY 最后供给点（派发阶段D，下跌后缩量反弹至阻力=减仓点）---
        # 价格处于低位/MA60下方，反弹至均线附近，缩量
        if indicator == "last_point_supply":
            rally_to = params.get("rally_to", "ma20")  # ma20 / ma10
            vol_shrink = params.get("vol_shrink", 0.7)
            low_position = params.get("low_position", 40)  # 价格位置百分位阈值
            tolerance = params.get("tolerance", 3)
            if idx < 20:
                return False
            # 处于相对低位
            if df["price_position_pct"].iloc[idx] > low_position:
                return False
            # 反弹至均线附近
            if rally_to == "ma10":
                target = df["ma10"].iloc[idx]
            else:
                target = df["ma20"].iloc[idx]
            if target <= 0:
                return False
            near_target = abs((df["close"].iloc[idx] - target) / target * 100) <= tolerance
            # 缩量反弹（买盘不积极）
            shrink = df["volume"].iloc[idx] < df["vol_ma20"].iloc[idx] * vol_shrink
            return near_target and shrink

        # --- 7. 量价背离 (Effort vs Result) ---
        # type=bull_div: 放量大跌后缩量不跌=卖盘衰竭→看多
        # type=bear_div: 放量大涨后缩量不涨=买盘衰竭→看空
        if indicator == "effort_result_diverge":
            div_type = params.get("type", "bull_div")  # bull_div / bear_div
            n_days = params.get("days", 5)              # 回看天数
            vol_ratio = params.get("vol_ratio", 1.3)    # 放量阈值（放宽）
            vol_shrink = params.get("vol_shrink", 0.7)  # 缩量阈值（放宽）
            if idx < n_days:
                return False
            recent = df.iloc[idx - n_days + 1:idx + 1]
            mid = len(recent) // 2
            if mid < 2:
                return False
            first_half = recent.iloc[:mid]
            second_half = recent.iloc[mid:]
            if div_type == "bull_div":
                # 前段放量大跌 + 后段缩量企稳
                big_vol_drop = (first_half["volume"].mean() > df["vol_ma20"].iloc[idx] * vol_ratio
                                and first_half["close"].iloc[-1] < first_half["close"].iloc[0])
                shrink_stable = (second_half["volume"].mean() < df["vol_ma20"].iloc[idx] * vol_shrink
                                 and second_half["close"].iloc[-1] >= second_half["close"].iloc[0] * 0.98)
                return big_vol_drop and shrink_stable
            else:  # bear_div
                big_vol_rise = (first_half["volume"].mean() > df["vol_ma20"].iloc[idx] * vol_ratio
                                and first_half["close"].iloc[-1] > first_half["close"].iloc[0])
                shrink_stall = (second_half["volume"].mean() < df["vol_ma20"].iloc[idx] * vol_shrink
                                and second_half["close"].iloc[-1] <= second_half["close"].iloc[0] * 1.02)
                return big_vol_rise and shrink_stall

        # --- 8. SC 抛售高潮（吸筹阶段A，底部第一信号）---
        # 暴跌+巨量+长下影线=恐慌盘出清
        if indicator == "selling_climax":
            vol_ratio = params.get("vol_ratio", 2.0)      # 成交量>2x均量
            min_drop = params.get("min_drop", -5.0)        # 最低跌幅%
            wick_ratio = params.get("wick_ratio", 0.4)     # 下影线占全幅比
            if idx < 1:
                return False
            big_vol = df["volume"].iloc[idx] > df["vol_ma20"].iloc[idx] * vol_ratio
            sharp_drop = df["pct_change"].iloc[idx] < min_drop
            # 长下影线：下影线=(min(open,close)-low), 全幅=high-low
            full_range = df["high"].iloc[idx] - df["low"].iloc[idx]
            if full_range <= 0:
                return False
            lower_wick = min(df["open"].iloc[idx], df["close"].iloc[idx]) - df["low"].iloc[idx]
            long_wick = lower_wick / full_range > wick_ratio
            return big_vol and (sharp_drop or long_wick)

        # --- 9. BC 抢购高潮（派发阶段A，见顶第一信号）---
        # 加速大涨+天量+长上影线=买盘枯竭
        if indicator == "buying_climax":
            vol_ratio = params.get("vol_ratio", 2.0)      # 成交量>2x均量
            min_rise = params.get("min_rise", 5.0)         # 最低涨幅%
            wick_ratio = params.get("wick_ratio", 0.4)     # 上影线占全幅比
            high_position = params.get("high_position", 70) # 价格位置百分位
            if idx < 1:
                return False
            # 处于高位
            if df["price_position_pct"].iloc[idx] < high_position:
                return False
            big_vol = df["volume"].iloc[idx] > df["vol_ma20"].iloc[idx] * vol_ratio
            sharp_rise = df["pct_change"].iloc[idx] > min_rise
            # 长上影线
            full_range = df["high"].iloc[idx] - df["low"].iloc[idx]
            if full_range <= 0:
                return False
            upper_wick = df["high"].iloc[idx] - max(df["open"].iloc[idx], df["close"].iloc[idx])
            long_wick = upper_wick / full_range > wick_ratio
            return big_vol and (sharp_rise or long_wick)

        # --- 10. ST 二次测试（吸筹/派发阶段A确认）---
        # 重新测试SC/BC区域，但成交量显著缩小
        if indicator == "secondary_test":
            test_type = params.get("test_type", "support")  # support(测底)/resist(测顶)
            n = params.get("period", 20)                    # 回看N日找SC/BC
            vol_shrink = params.get("vol_shrink", 0.6)      # 缩量至SC/BC的60%以下
            if idx < n:
                return False
            if test_type == "support":
                # 找N日内最大量日作为SC
                sc_idx = df["volume"].iloc[idx - n:idx].idxmax()
                if sc_idx not in df.index:
                    return False
                sc_pos = df.index.get_loc(sc_idx)
                sc_low = df["low"].iloc[sc_pos]
                # 当前日在SC低点附近(±3%)且成交量远小于SC
                near_sc = abs(df["low"].iloc[idx] - sc_low) / sc_low * 100 < 3 if sc_low > 0 else False
                vol_shrunk = df["volume"].iloc[idx] < df["volume"].iloc[sc_pos] * vol_shrink
                return near_sc and vol_shrunk
            else:
                # 找N日内最大量日作为BC
                bc_idx = df["volume"].iloc[idx - n:idx].idxmax()
                if bc_idx not in df.index:
                    return False
                bc_pos = df.index.get_loc(bc_idx)
                bc_high = df["high"].iloc[bc_pos]
                near_bc = abs(df["high"].iloc[idx] - bc_high) / bc_high * 100 < 3 if bc_high > 0 else False
                vol_shrunk = df["volume"].iloc[idx] < df["volume"].iloc[bc_pos] * vol_shrink
                return near_bc and vol_shrunk

        # --- 11. Stopping Volume 停止成交量（下跌中供应被吸收）---
        # 下跌K线+高量+窄价差=主力在吸收供应
        if indicator == "stopping_volume":
            vol_ratio = params.get("vol_ratio", 1.5)     # 高量阈值
            max_spread = params.get("max_spread", 2.0)   # 最大价差%
            if idx < 1:
                return False
            high_vol = df["volume"].iloc[idx] > df["vol_ma20"].iloc[idx] * vol_ratio
            # 窄价差：全幅/收盘 < max_spread%
            close_val = df["close"].iloc[idx]
            if close_val <= 0:
                return False
            spread_pct = (df["high"].iloc[idx] - df["low"].iloc[idx]) / close_val * 100
            narrow = spread_pct < max_spread
            # 收盘在下半部（下跌或中性）
            mid = (df["high"].iloc[idx] + df["low"].iloc[idx]) / 2
            close_lower = close_val < mid
            return high_vol and narrow and close_lower

        # --- 12. No Demand 无需求（上涨乏力信号）---
        # 上涨K线+低量=没有买盘支撑，趋势不健康
        if indicator == "no_demand":
            vol_shrink = params.get("vol_shrink", 0.5)    # 缩量阈值
            min_rise = params.get("min_rise", 0.0)         # 最小涨幅
            max_rise = params.get("max_rise", 3.0)         # 最大涨幅
            if idx < 1:
                return False
            low_vol = df["volume"].iloc[idx] < df["vol_ma20"].iloc[idx] * vol_shrink
            is_up = df["pct_change"].iloc[idx] > min_rise
            small_up = df["pct_change"].iloc[idx] < max_rise
            return low_vol and is_up and small_up

        # --- 13. No Supply 无供应（卖盘枯竭信号=看多）---
        # 下跌K线+极低量=供应消失
        if indicator == "no_supply":
            vol_shrink = params.get("vol_shrink", 0.4)    # 极低量阈值
            max_drop = params.get("max_drop", -0.5)        # 最大跌幅（小跌即可）
            if idx < 1:
                return False
            very_low_vol = df["volume"].iloc[idx] < df["vol_ma20"].iloc[idx] * vol_shrink
            is_down = df["pct_change"].iloc[idx] < max_drop
            return very_low_vol and is_down

        # ================================================================
        # === 量能爆发底部反转策略 信号 ===
        # ================================================================

        # --- 长期下跌: zhixing_fast 连续N天低于 zhixing_slow ---
        if indicator == "zhixing_long_downtrend":
            n_days = params.get("days", 30)           # 至少N天
            check_window = params.get("window", 0)     # 在window天内至少有days天fast<slow, 0=连续
            if idx < n_days:
                return False
            if check_window > 0:
                # 宽松模式: 在最近window天内，至少days天fast<slow
                start = max(0, idx - check_window + 1)
                count = sum(1 for i in range(start, idx + 1)
                           if df["zhixing_fast"].iloc[i] < df["zhixing_slow"].iloc[i])
                return count >= n_days
            else:
                # 严格模式: 最近连续days天fast<slow
                for i in range(n_days):
                    if df["zhixing_fast"].iloc[idx - i] >= df["zhixing_slow"].iloc[idx - i]:
                        return False
                return True

        # --- 量能极端爆发（罕见天量）---
        # vol_rank_pct >= threshold OR volume >= vol_ma20 * ratio
        if indicator == "volume_extreme_explosion":
            vol_rank_threshold = params.get("vol_rank_threshold", 95)
            vol_ratio_threshold = params.get("vol_ratio_threshold", 3.0)
            by_rank = df["vol_rank_pct"].iloc[idx] >= vol_rank_threshold
            by_ratio = df["volume"].iloc[idx] >= df["vol_ma20"].iloc[idx] * vol_ratio_threshold
            return by_rank or by_ratio

        # --- 近期有量能爆发（回望N天内出现过天量）---
        if indicator == "recent_volume_explosion":
            lookback = params.get("lookback", 10)       # 回望N天
            vol_rank_threshold = params.get("vol_rank_threshold", 95)
            vol_ratio_threshold = params.get("vol_ratio_threshold", 3.0)
            if idx < lookback:
                return False
            for i in range(lookback):
                j = idx - i
                by_rank = df["vol_rank_pct"].iloc[j] >= vol_rank_threshold
                by_ratio = df["volume"].iloc[j] >= df["vol_ma20"].iloc[j] * vol_ratio_threshold
                if by_rank or by_ratio:
                    return True
            return False

        # --- 量能持续: 近期爆发后量能未快速萎缩 ---
        # 条件: 最近sustain_days天内平均量 >= vol_ma20 * sustain_ratio
        if indicator == "volume_sustained":
            sustain_days = params.get("sustain_days", 5)       # 检测天数
            sustain_ratio = params.get("sustain_ratio", 0.7)   # 量能维持比例
            lookback = params.get("explosion_lookback", 15)    # 爆发回望期
            vol_rank_threshold = params.get("vol_rank_threshold", 95)
            vol_ratio_threshold = params.get("vol_ratio_threshold", 3.0)
            if idx < lookback:
                return False
            # 先找到最近的爆发日
            explosion_idx = -1
            for i in range(min(lookback, idx - 60)):
                j = idx - i
                by_rank = df["vol_rank_pct"].iloc[j] >= vol_rank_threshold
                by_ratio = df["volume"].iloc[j] >= df["vol_ma20"].iloc[j] * vol_ratio_threshold
                if by_rank or by_ratio:
                    explosion_idx = j
                    break
            if explosion_idx < 0:
                return False
            # 从爆发日到当前的量能不能快速萎缩
            # 检查爆发后的天数（不含爆发日本身）
            days_since = idx - explosion_idx
            if days_since < 1:
                return True  # 刚爆发，算通过
            check_start = explosion_idx + 1
            check_end = min(idx + 1, explosion_idx + sustain_days + 1)
            if check_start >= check_end:
                return True
            recent_vols = df["volume"].iloc[check_start:check_end]
            avg_vol = recent_vols.mean()
            vol_ma = df["vol_ma20"].iloc[idx]
            if vol_ma <= 0:
                return False
            return avg_vol >= vol_ma * sustain_ratio

        # --- 价格站上zhixing_fast: close > zhixing_fast 且前一天 close <= zhixing_fast ---
        if indicator == "price_cross_above_zhixing_fast":
            if idx < 1:
                return False
            cur_above = df["close"].iloc[idx] > df["zhixing_fast"].iloc[idx]
            prev_below = df["close"].iloc[idx - 1] <= df["zhixing_fast"].iloc[idx - 1]
            return cur_above and prev_below

        # --- 价格站在zhixing_fast上方（持续N天）---
        if indicator == "price_above_zhixing_fast_hold":
            n_hold = params.get("days", 1)
            if idx < n_hold:
                return False
            for i in range(n_hold):
                if df["close"].iloc[idx - i] <= df["zhixing_fast"].iloc[idx - i]:
                    return False
            return True

        # --- 价格跌破zhixing_fast ---
        if indicator == "price_below_zhixing_fast":
            return df["close"].iloc[idx] < df["zhixing_fast"].iloc[idx]

        # --- 双底形态检测 ---
        if indicator == "double_bottom":
            lookback = params.get("lookback", 20)           # 回望期
            min_separation = params.get("min_separation", 3) # 两底最小间隔
            tolerance_pct = params.get("tolerance", 3.0)     # 两底价格容差%
            bounce_pct = params.get("bounce_pct", 3.0)       # 两底之间最小反弹幅度%
            if idx < lookback:
                return False
            low_arr = df["low"].values.astype(float)
            close_arr = df["close"].values.astype(float)
            # 在 [idx-lookback, idx] 窗口内找双底
            start = idx - lookback
            window_low = low_arr[start:idx + 1]
            window_close = close_arr[start:idx + 1]
            # 找局部低点：比前后各2天都低的点（更严格）
            local_mins = []
            for j in range(2, len(window_low) - 2):
                if (window_low[j] <= window_low[j - 1] and window_low[j] <= window_low[j - 2]
                    and window_low[j] <= window_low[j + 1] and window_low[j] <= window_low[j + 2]):
                    local_mins.append((j, window_low[j]))
            if len(local_mins) < 2:
                return False
            # 从最近往远找双底对
            for k in range(len(local_mins) - 1, 0, -1):
                for m in range(k - 1, -1, -1):
                    j2, low2 = local_mins[k]
                    j1, low1 = local_mins[m]
                    if j2 - j1 < min_separation:
                        continue
                    ref = max(low1, low2)
                    if ref <= 0:
                        continue
                    if abs(low1 - low2) / ref * 100 > tolerance_pct:
                        continue
                    # 两底之间有显著反弹：中间最高close比两底高至少bounce_pct%
                    between_close = window_close[j1 + 1:j2]
                    neck = max(low1, low2)
                    if len(between_close) == 0:
                        continue
                    max_between = between_close.max()
                    bounce = (max_between - neck) / neck * 100
                    if bounce < bounce_pct:
                        continue
                    # 第二个底必须在近期（距当前不超过5天）
                    actual_j2 = start + j2
                    if idx - actual_j2 > 5:
                        continue
                    return True
            return False

        # --- 双底形态破坏: close跌破双底支撑位 ---
        if indicator == "double_bottom_broken":
            lookback = params.get("lookback", 20)
            tolerance_pct = params.get("tolerance", 3.0)
            bounce_pct = params.get("bounce_pct", 3.0)
            break_pct = params.get("break_pct", 0.0)  # 跌破支撑位多少%算破坏
            if idx < lookback:
                return False
            # 重新计算双底支撑位
            support = _find_double_bottom_support(df, idx, lookback, tolerance_pct, bounce_pct)
            if support is None or support <= 0:
                return False
            return df["close"].iloc[idx] < support * (1 - break_pct / 100)

        # --- 连续N日低于BBI ---
        if indicator == "price_below_bbi_consecutive":
            n_days = params.get("days", 2)
            if idx < n_days:
                return False
            if "bbi" not in df.columns:
                return False
            for i in range(n_days):
                if df["close"].iloc[idx - i] >= df["bbi"].iloc[idx - i]:
                    return False
            return True

        # --- 连续N日低于zhixing_slow ---
        # 仅在近期曾经高于zhixing_slow后才触发（避免底部反转建仓时误触发）
        if indicator == "price_below_zhixing_slow_consecutive":
            n_days = params.get("days", 2)
            recent_above = params.get("recent_above_days", 30)  # 近N天内曾高于slow
            if idx < max(n_days, recent_above):
                return False
            # 检查近期是否曾高于zhixing_slow
            had_above = False
            for i in range(min(recent_above, idx - 60)):
                if df["close"].iloc[idx - i] > df["zhixing_slow"].iloc[idx - i]:
                    had_above = True
                    break
            if not had_above:
                return False
            # 检查连续N日低于zhixing_slow
            for i in range(n_days):
                if df["close"].iloc[idx - i] >= df["zhixing_slow"].iloc[idx - i]:
                    return False
            return True

        # ================================================================
        # === zzh1.0 新增指标 ===
        # ================================================================

        # --- Buying Climax (BC) 天量长上影派发 ---
        # 高位 + 极端天量 + 长上影线 = 主力派发/买盘枯竭
        if indicator == "buying_climax":
            vol_rank_threshold = params.get("vol_rank_threshold", 95)
            wick_ratio = params.get("wick_ratio", 0.35)
            position_threshold = params.get("position_threshold", 60)
            min_pct = params.get("min_pct", -2.0)
            if idx < 60:
                return False
            # 量不够大 → 不是BC
            if df["vol_rank_pct"].iloc[idx] < vol_rank_threshold:
                return False
            # 位置不够高 → 不是派发
            if df["price_position_pct"].iloc[idx] < position_threshold:
                return False
            # 暴跌日 → 不是BC（BC是高位诱多）
            if df["pct_change"].iloc[idx] < min_pct:
                return False
            # 计算上影线占比
            high_val = df["high"].iloc[idx]
            low_val = df["low"].iloc[idx]
            open_val = df["open"].iloc[idx]
            close_val = df["close"].iloc[idx]
            full_range = high_val - low_val
            if full_range <= 0:
                return False
            body_high = max(open_val, close_val)
            upper_wick = high_val - body_high
            return (upper_wick / full_range) >= wick_ratio

        # --- Repeated Slow Touch 反复测试slow支撑 ---
        # 短时间内多次回踩slow线 → 多头信心耗尽
        if indicator == "repeated_slow_touch":
            lookback = params.get("lookback", 30)
            tolerance = params.get("tolerance", 2.0)
            min_touches = params.get("min_touches", 4)
            if idx < lookback:
                return False
            touches = 0
            for j in range(idx - lookback + 1, idx + 1):
                slow_j = df["zhixing_slow"].iloc[j]
                if slow_j <= 0:
                    continue
                dist = abs((df["close"].iloc[j] - slow_j) / slow_j * 100)
                if dist < tolerance:
                    touches += 1
            return touches >= min_touches

        # === 基本面 ===
        if indicator == "pe_below":
            threshold = params.get("threshold", 20)
            return df["pe"].iloc[idx] < threshold

        if indicator == "pe_ttm_below":
            threshold = params.get("threshold", 20)
            return df["pe_ttm"].iloc[idx] < threshold

        if indicator == "pb_below":
            threshold = params.get("threshold", 1.5)
            return df["pb"].iloc[idx] < threshold

        if indicator == "pe_pct_low":
            lookback = params.get("lookback", 252)
            threshold = params.get("threshold", 20)
            if idx < lookback:
                return False
            pe_vals = df["pe"].iloc[idx - lookback + 1:idx + 1]
            pe_vals = pe_vals[pe_vals > 0]
            if len(pe_vals) < 60:
                return False
            rank = (pe_vals < df["pe"].iloc[idx]).sum() / len(pe_vals) * 100
            return rank < threshold

        if indicator == "pb_pct_low":
            lookback = params.get("lookback", 252)
            threshold = params.get("threshold", 20)
            if idx < lookback:
                return False
            pb_vals = df["pb"].iloc[idx - lookback + 1:idx + 1]
            pb_vals = pb_vals[pb_vals > 0]
            if len(pb_vals) < 60:
                return False
            rank = (pb_vals < df["pb"].iloc[idx]).sum() / len(pb_vals) * 100
            return rank < threshold

        if indicator == "total_mv_above":
            threshold = params.get("threshold", 100)  # 单位：亿
            return df["total_mv"].iloc[idx] > threshold

        if indicator == "total_mv_below":
            threshold = params.get("threshold", 500)
            return df["total_mv"].iloc[idx] < threshold

        if indicator == "turnover_rate_above":
            threshold = params.get("threshold", 3.0)  # 单位：%
            return df["turnover_rate"].iloc[idx] > threshold

        if indicator == "turnover_rate_below":
            threshold = params.get("threshold", 1.0)
            return df["turnover_rate"].iloc[idx] < threshold

        # === 市场状态 ===
        if indicator == "market_bull":
            index_name = params.get("index", "hs300")
            from backend.regime import get_regime_on
            date_str = str(df["date"].iloc[idx])[:10]
            return get_regime_on(date_str, index_name) == "bull"

        if indicator == "market_bear":
            index_name = params.get("index", "hs300")
            from backend.regime import get_regime_on
            date_str = str(df["date"].iloc[idx])[:10]
            return get_regime_on(date_str, index_name) == "bear"

        if indicator == "market_consolidation":
            index_name = params.get("index", "hs300")
            from backend.regime import get_regime_on
            date_str = str(df["date"].iloc[idx])[:10]
            return get_regime_on(date_str, index_name) == "consolidation"

        # === 相对抗跌（个股 vs 大盘） ===
        if indicator == "relative_strength":
            threshold = params.get("threshold", 0)
            col = f"relative_strength_{params.get('lookback', 60)}"
            if col not in df.columns:
                col = "relative_strength_60"
            return df[col].iloc[idx] > threshold

        if indicator == "market_crash_recent":
            col = f"market_crash_{params.get('lookback', 30)}d"
            if col not in df.columns:
                col = "market_crash_30d"
            return bool(df[col].iloc[idx])

        if indicator == "market_crash_fast":
            """10天内快速杀跌>5%"""
            if "market_crash_fast_10d" in df.columns:
                return bool(df["market_crash_fast_10d"].iloc[idx])
            return False

        if indicator == "zz1000_crash":
            """中证1000在N天内回撤超过X%"""
            days = params.get("days", 20)
            threshold = params.get("threshold", 10.0)
            from backend.indicators import _load_index_cached as _lic
            zz = _lic("zz1000")
            date_str = str(df["date"].iloc[idx])[:10]
            end = pd.Timestamp(date_str)
            start = end - pd.Timedelta(days=days + 5)
            window = zz[(zz.index >= start) & (zz.index <= end)]
            if len(window) < 5:
                return False
            peak = window["close"].max()
            dd = (window["close"].iloc[-1] - peak) / peak * 100
            return dd < -threshold

    except (KeyError, IndexError):
        return False

    return False


def check_condition_vectorized(df: pd.DataFrame, cond: dict) -> np.ndarray:
    """向量化条件检测：返回boolean数组，True=该行满足条件

    仅支持常用的条件类型，不支持的返回全False数组
    """
    indicator = cond.get("indicator", "")
    params = cond.get("params", {})
    n = len(df)
    result = np.zeros(n, dtype=bool)

    # 前60行不满足
    if n <= 60:
        return result

    try:
        if indicator == "zhixing_fast_above_slow":
            result[60:] = df["zhixing_fast"].values[60:] > df["zhixing_slow"].values[60:]

        elif indicator == "zhixing_fast_below_slow":
            result[60:] = df["zhixing_fast"].values[60:] < df["zhixing_slow"].values[60:]

        elif indicator == "zhixing_golden_cross":
            fast = df["zhixing_fast"].values
            slow = df["zhixing_slow"].values
            result[61:] = (fast[61:] > slow[61:]) & (fast[60:-1] <= slow[60:-1])

        elif indicator == "zhixing_dead_cross":
            fast = df["zhixing_fast"].values
            slow = df["zhixing_slow"].values
            result[61:] = (fast[61:] < slow[61:]) & (fast[60:-1] >= slow[60:-1])

        elif indicator == "zhixing_golden_cross_hold":
            n_hold = params.get("days", 3)
            fast = df["zhixing_fast"].values
            slow = df["zhixing_slow"].values
            above = fast > slow
            # 滑动窗口：连续n_hold天fast>slow
            for i in range(60 + n_hold - 1, n):
                if above[i - n_hold + 1:i + 1].all():
                    result[i] = True

        elif indicator == "price_near_zhixing":
            tolerance = params.get("tolerance", 3)
            close_arr = df["close"].values
            zf = df["zhixing_fast"].values
            zs = df["zhixing_slow"].values
            m60 = df["ma60"].values if "ma60" in df.columns else np.full(n, np.nan)
            for i in range(60, n):
                c = close_arr[i]
                targets = [zf[i], zs[i]]
                if not np.isnan(m60[i]):
                    targets.append(m60[i])
                for t in targets:
                    if t > 0 and abs((c - t) / t) * 100 <= tolerance:
                        result[i] = True
                        break

        elif indicator == "volume_anomaly":
            ratio = params.get("ratio", 1.5)
            is_double = df["is_double_vol"].values.astype(bool)
            is_top5 = df["is_vol_top5"].values.astype(bool)
            above_avg = df["volume"].values > df["vol_ma20"].values * ratio
            result[60:] = is_double[60:] | is_top5[60:] | above_avg[60:]

        elif indicator == "volume_double":
            result[60:] = df["is_double_vol"].values[60:].astype(bool)

        elif indicator == "volume_shrink":
            ratio = params.get("ratio", 0.5)
            result[60:] = df["volume"].values[60:] < df["vol_ma20"].values[60:] * ratio

        elif indicator == "low_position":
            price_pct_threshold = params.get("price_pct", 40)
            ma60_dist_threshold = params.get("ma60_dist", 5)
            price_low = df["price_position_pct"].values < price_pct_threshold
            near_ma60 = df["dist_to_ma60"].values < ma60_dist_threshold
            result[60:] = price_low[60:] | near_ma60[60:]

        elif indicator == "not_distribution":
            pct = df["pct_change"].values
            is_double = df["is_double_vol"].values.astype(bool)
            pp = df["price_position_pct"].values
            for i in range(60, n):
                if pct[i] > 9.5 and is_double[i]:
                    continue
                if pp[i] > 80 and is_double[i]:
                    continue
                result[i] = True

        elif indicator == "pocket_pivot":
            lookback = params.get("lookback", 10)
            vol = df["volume"].values.astype(float)
            close_arr = df["close"].values.astype(float)
            is_down = np.zeros(n, dtype=bool)
            is_down[1:] = close_arr[1:] < close_arr[:-1]
            for i in range(max(60, lookback), n):
                start = i - lookback
                down_mask = is_down[start:i]
                down_vols = vol[start:i][down_mask]
                if len(down_vols) > 0:
                    result[i] = vol[i] > down_vols.max()
                else:
                    result[i] = vol[i] > 0

        elif indicator == "sideway_shrink":
            n_days = params.get("days", 3)
            amp_threshold = params.get("amplitude", 3.0)
            vol_ratio_threshold = params.get("vol_ratio", 0.4)
            high_arr = df["high"].values.astype(float)
            low_arr = df["low"].values.astype(float)
            vol_arr = df["volume"].values.astype(float)
            vol_ma20_arr = df["vol_ma20"].values.astype(float)
            for i in range(max(60, n_days), n):
                start = i - n_days + 1
                period_low = low_arr[start:i + 1].min()
                if period_low <= 0:
                    continue
                amplitude = (high_arr[start:i + 1].max() - period_low) / period_low * 100
                if amplitude >= amp_threshold:
                    continue
                vma = vol_ma20_arr[i]
                if vma <= 0:
                    continue
                if (vol_arr[start:i + 1] < vma * vol_ratio_threshold).all():
                    result[i] = True

        elif indicator == "spring_shakeout":
            sp = params.get("support_period", 20)
            vol_confirm = params.get("vol_confirm", False)
            spring_type = params.get("spring_type", 0)
            low_arr = df["low"].values.astype(float)
            close_arr = df["close"].values.astype(float)
            vol_arr = df["volume"].values.astype(float)
            vol_ma20_arr = df["vol_ma20"].values.astype(float)
            for i in range(max(61, sp + 1), n):
                support = low_arr[i - sp:i - 1].min()
                if support <= 0:
                    continue
                prev_low = low_arr[i - 1]
                if prev_low >= support:
                    continue
                if close_arr[i] <= support:
                    continue
                pen_pct = (support - prev_low) / support * 100
                if spring_type == 1:
                    if pen_pct < 3 or vol_arr[i - 1] < vol_ma20_arr[i - 1] * 1.5:
                        continue
                elif spring_type == 2:
                    if pen_pct < 1 or pen_pct > 3:
                        continue
                elif spring_type == 3:
                    if pen_pct > 1 or vol_arr[i - 1] > vol_ma20_arr[i - 1] * 0.7:
                        continue
                if vol_confirm:
                    result[i] = vol_arr[i] > vol_ma20_arr[i] * 1.2
                else:
                    result[i] = True

        elif indicator == "no_supply":
            vol_shrink = params.get("vol_shrink", 0.4)
            max_drop = params.get("max_drop", -0.5)
            very_low_vol = df["volume"].values < df["vol_ma20"].values * vol_shrink
            is_down = df["pct_change"].values < max_drop
            result[61:] = very_low_vol[61:] & is_down[61:]

        elif indicator == "effort_result_diverge":
            div_type = params.get("type", "bull_div")
            n_days = params.get("days", 5)
            vol_ratio = params.get("vol_ratio", 1.3)
            vol_shrink = params.get("vol_shrink", 0.7)
            vol_arr = df["volume"].values.astype(float)
            close_arr = df["close"].values.astype(float)
            vol_ma20_arr = df["vol_ma20"].values.astype(float)
            for i in range(max(60, n_days), n):
                start = i - n_days + 1
                mid = n_days // 2
                first_vol = vol_arr[start:start + mid].mean()
                second_vol = vol_arr[start + mid:i + 1].mean()
                vma = vol_ma20_arr[i]
                if vma <= 0:
                    continue
                if div_type == "bull_div":
                    big_vol_drop = first_vol > vma * vol_ratio and close_arr[start + mid - 1] < close_arr[start]
                    shrink_stable = second_vol < vma * vol_shrink and close_arr[i] >= close_arr[start + mid] * 0.98
                    result[i] = big_vol_drop and shrink_stable
                else:
                    big_vol_rise = first_vol > vma * vol_ratio and close_arr[start + mid - 1] > close_arr[start]
                    shrink_stall = second_vol < vma * vol_shrink and close_arr[i] <= close_arr[start + mid] * 1.02
                    result[i] = big_vol_rise and shrink_stall

        # --- Buying Climax 向量化 ---
        elif indicator == "buying_climax":
            vol_rank_threshold = params.get("vol_rank_threshold", 95)
            wick_ratio = params.get("wick_ratio", 0.35)
            position_threshold = params.get("position_threshold", 60)
            min_pct = params.get("min_pct", -2.0)
            vol_rank_arr = df["vol_rank_pct"].values
            pos_arr = df["price_position_pct"].values
            pct_arr = df["pct_change"].values
            high_arr = df["high"].values
            low_arr = df["low"].values
            open_arr = df["open"].values
            close_arr = df["close"].values
            for i in range(60, n):
                if vol_rank_arr[i] < vol_rank_threshold:
                    continue
                if pos_arr[i] < position_threshold:
                    continue
                if pct_arr[i] < min_pct:
                    continue
                full_range = high_arr[i] - low_arr[i]
                if full_range <= 0:
                    continue
                body_high = max(open_arr[i], close_arr[i])
                upper_wick = high_arr[i] - body_high
                if (upper_wick / full_range) >= wick_ratio:
                    result[i] = True

        # --- Repeated Slow Touch 向量化 ---
        elif indicator == "repeated_slow_touch":
            lookback = params.get("lookback", 30)
            tolerance = params.get("tolerance", 2.0)
            min_touches = params.get("min_touches", 4)
            slow_arr = df["zhixing_slow"].values
            close_arr = df["close"].values
            for i in range(lookback, n):
                touches = 0
                for j in range(i - lookback + 1, i + 1):
                    slow_j = slow_arr[j]
                    if slow_j <= 0:
                        continue
                    dist = abs((close_arr[j] - slow_j) / slow_j * 100)
                    if dist < tolerance:
                        touches += 1
                if touches >= min_touches:
                    result[i] = True

        # 未向量化条件 → 回退到逐行检测
        else:
            for idx in range(60, n):
                if check_condition(df, idx, cond):
                    result[idx] = True

    except (KeyError, IndexError):
        # 回退到逐行检测
        for idx in range(60, n):
            if check_condition(df, idx, cond):
                result[idx] = True

    return result


def detect_signals_vectorized(df: pd.DataFrame, config: StrategyConfig) -> dict:
    """向量化信号检测：预计算各条件的boolean数组，组内AND

    返回: {
        "buy": set of indices,
        "sell": set of indices,
        "add": set of indices,
        "reduce": set of indices,
    }
    """
    n = len(df)
    signals = {"buy": set(), "sell": set(), "add": set(), "reduce": set()}

    # 每个条件组的向量化检测
    for group in config.buy_groups:
        conditions = group.get("conditions", [])
        if not conditions:
            continue
        group_result = np.ones(n, dtype=bool)
        group_result[:60] = False
        for cond in conditions:
            cond_result = check_condition_vectorized(df, cond)
            group_result &= cond_result
        signals["buy"].update(np.where(group_result)[0].tolist())

    for group in config.sell_groups:
        conditions = group.get("conditions", [])
        if not conditions:
            continue
        group_result = np.ones(n, dtype=bool)
        group_result[:60] = False
        for cond in conditions:
            cond_result = check_condition_vectorized(df, cond)
            group_result &= cond_result
        signals["sell"].update(np.where(group_result)[0].tolist())

    for group in config.add_groups:
        conditions = group.get("conditions", [])
        if not conditions:
            continue
        group_result = np.ones(n, dtype=bool)
        group_result[:60] = False
        for cond in conditions:
            cond_result = check_condition_vectorized(df, cond)
            group_result &= cond_result
        signals["add"].update(np.where(group_result)[0].tolist())

    for group in config.reduce_groups:
        conditions = group.get("conditions", [])
        if not conditions:
            continue
        group_result = np.ones(n, dtype=bool)
        group_result[:60] = False
        for cond in conditions:
            cond_result = check_condition_vectorized(df, cond)
            group_result &= cond_result
        signals["reduce"].update(np.where(group_result)[0].tolist())

    return signals


def check_group(df: pd.DataFrame, idx: int, group: dict) -> tuple[bool, str]:
    """检查一组条件（AND逻辑）"""
    conditions = group.get("conditions", [])
    if not conditions:
        return False, ""

    reasons = []
    for cond in conditions:
        if not check_condition(df, idx, cond):
            return False, ""
        indicator = cond.get("indicator", "")
        params = cond.get("params", {})
        reasons.append(f"{indicator}({params})")

    return True, " AND ".join(reasons)


def evaluate_signals(df: pd.DataFrame, config: StrategyConfig) -> list[Signal]:
    """评估所有信号"""
    df = compute_all_indicators(df)
    signals = []

    for idx in range(len(df)):
        date = df["date"].iloc[idx]
        price = df["close"].iloc[idx]

        # 检查买入信号
        for group in config.buy_groups:
            ok, reason = check_group(df, idx, group)
            if ok:
                signals.append(Signal(
                    date=date, code="",
                    signal_type="buy", price=price,
                    reason=reason
                ))
                break  # 一组满足即可

        # 检查卖出信号
        for group in config.sell_groups:
            ok, reason = check_group(df, idx, group)
            if ok:
                signals.append(Signal(
                    date=date, code="",
                    signal_type="sell", price=price,
                    reason=reason
                ))
                break

        # 检查加仓信号
        for group in config.add_groups:
            ok, reason = check_group(df, idx, group)
            if ok:
                signals.append(Signal(
                    date=date, code="",
                    signal_type="add", price=price,
                    reason=reason
                ))
                break

        # 检查减仓信号
        for group in config.reduce_groups:
            ok, reason = check_group(df, idx, group)
            if ok:
                signals.append(Signal(
                    date=date, code="",
                    signal_type="reduce", price=price,
                    reason=reason
                ))
                break

    return signals
