"""回测引擎 — 执行策略回测，生成交易记录和统计"""

from __future__ import annotations
import uuid
import pandas as pd
import numpy as np
from typing import Optional
from dataclasses import dataclass, field
from datetime import datetime

from backend.data_loader import load_stock, list_all_codes, load_stock_with_indicators, preload_indicator_cache
from backend.indicators import compute_all_indicators
from backend.strategy_engine import (
    StrategyConfig, Signal, Trade, check_group, resolve_price, detect_signals_vectorized
)
from backend.state_machine import ZZH10StateMachine


@dataclass
class Position:
    """持仓"""
    code: str
    buy_date: Any
    buy_price: float
    shares: int
    cost: float  # 总成本
    highest_price: float  # 持仓期间最高价（用于移动止损）


@dataclass
class BacktestResult:
    """回测结果"""
    config_name: str
    k_type: str
    backtest_mode: str  # signal / portfolio
    start_date: str
    end_date: str
    initial_capital: float  # portfolio模式有效
    final_capital: float    # portfolio模式有效
    total_return_pct: float
    annual_return_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float
    win_rate: float
    profit_loss_ratio: float
    expected_value: float  # 期望值 = 胜率×均盈 - 败率×均亏
    total_trades: int
    win_trades: int
    lose_trades: int
    avg_profit_pct: float
    avg_loss_pct: float
    max_profit_pct: float
    max_loss_pct: float
    avg_hold_days: float
    trades: list[dict] = field(default_factory=list)
    equity_curve: list[dict] = field(default_factory=list)
    annual_returns: list[dict] = field(default_factory=list)
    monthly_returns: list[dict] = field(default_factory=list)


def _resolve_stock_pool(config: StrategyConfig) -> list[str]:
    """确定股票池"""
    if config.stock_pool:
        return config.stock_pool
    return list_all_codes()


def _net_profit_pct(buy_price: float, sell_price: float,
                    commission_rate: float, stamp_tax_rate: float,
                    slippage_pct: float = 0.0) -> float:
    """计算扣除交易成本后的净收益率

    买入成本: buy_price * (1 + commission_rate + slippage_pct)
    卖出所得: sell_price * (1 - commission_rate - stamp_tax_rate - slippage_pct)
    净收益率 = (卖出所得 - 买入成本) / 买入成本 * 100
    """
    cost = buy_price * (1 + commission_rate + slippage_pct)
    proceeds = sell_price * (1 - commission_rate - stamp_tax_rate - slippage_pct)
    return (proceeds - cost) / cost * 100


def _apply_slippage(price: float, side: str, slippage_pct: float) -> float:
    """应用滑点

    Args:
        price: 原始价格
        side: 'buy' (买入成本更高) 或 'sell' (卖出收入更低)
        slippage_pct: 滑点比例（小数，如 0.001 = 0.1%）

    Returns:
        调整后的价格
    """
    if slippage_pct <= 0:
        return price
    if side == "buy":
        return price * (1 + slippage_pct)  # 买入成交价略高
    else:
        return price * (1 - slippage_pct)  # 卖出成交价略低


def _is_limit_up(pct_change: float, threshold: float = 9.5) -> bool:
    """判断是否涨停（A股 ±10%，ST ±5%，科创板/创业板 ±20%）

    保守处理：涨幅 >= 9.5% 判定为涨停
    """
    return pct_change >= threshold


def _is_limit_down(pct_change: float, threshold: float = -9.5) -> bool:
    """判断是否跌停"""
    return pct_change <= threshold


def _effective_trail(profit_pct: float, config) -> float:
    """计算动态移动止损：盈利越高，止损越紧。"""
    if not config.trailing_stop_tiers:
        return config.trailing_stop_pct
    tier = max((p for p in config.trailing_stop_tiers if p <= profit_pct), default=0)
    return config.trailing_stop_tiers[tier]


def _is_limit_locked(pct_change: float, turnover_rate: float,
                     threshold: float = 9.5, min_turnover: float = 0.5) -> bool:
    """涨停/跌停封板且无量 = 无法成交

    - 涨停板且换手率 < min_turnover% → 买不到
    - 跌停板且换手率 < min_turnover% → 卖不掉

    Args:
        pct_change: 当日涨跌幅%
        turnover_rate: 当日换手率%
        threshold: 涨跌停阈值
        min_turnover: 最低换手率（低于此值判定为无量封板）
    """
    is_limit = abs(pct_change) >= threshold
    is_locked = turnover_rate < min_turnover
    return is_limit and is_locked


def run_backtest(config: StrategyConfig,
                 start_date: Optional[str] = None,
                 end_date: Optional[str] = None,
                 progress_callback=None) -> BacktestResult:
    """执行回测（多股票轮动模式）"""
    codes = _resolve_stock_pool(config)

    # === Phase 0: 预热指标缓存 ===
    if progress_callback:
        progress_callback(0, len(codes))

    if config.backtest_mode == "signal":
        return _run_signal_mode_streaming(config, codes, start_date, end_date, progress_callback)

    # === Phase 1: 批量加载并计算信号（portfolio模式需要全量） ===
    all_signals: dict[str, list[Signal]] = {}
    all_dfs: dict[str, pd.DataFrame] = {}

    for i, code in enumerate(codes):
        try:
            df = load_stock_with_indicators(code, config.k_type)
        except FileNotFoundError:
            continue

        if len(df) < 60:
            continue

        # 过滤日期范围
        if start_date:
            df = df[df["date"] >= start_date]
        if end_date:
            df = df[df["date"] <= end_date]
        if len(df) < 60:
            continue

        all_dfs[code] = df

        # 生成信号
        signals = []
        for idx in range(len(df)):
            date = df["date"].iloc[idx]
            price = df["close"].iloc[idx]

            # 买入检查
            for g in config.buy_groups:
                ok, reason = check_group(df, idx, g)
                if ok:
                    signals.append(Signal(date=date, code=code,
                                          signal_type="buy", price=price,
                                          reason=reason))
                    break

            # 卖出检查（清仓）
            for g in config.sell_groups:
                ok, reason = check_group(df, idx, g)
                if ok:
                    signals.append(Signal(date=date, code=code,
                                          signal_type="sell", price=price,
                                          reason=reason))
                    break

            # 加仓检查
            for g in config.add_groups:
                ok, reason = check_group(df, idx, g)
                if ok:
                    signals.append(Signal(date=date, code=code,
                                          signal_type="add", price=price,
                                          reason=reason))
                    break

            # 减仓检查
            for g in config.reduce_groups:
                ok, reason = check_group(df, idx, g)
                if ok:
                    signals.append(Signal(date=date, code=code,
                                          signal_type="reduce", price=price,
                                          reason=reason))
                    break

        all_signals[code] = signals
        if progress_callback:
            progress_callback(i + 1, len(codes))

    # === Phase 2: 按回测模式执行 ===
    result = _run_portfolio_mode(config, all_signals, all_dfs)

    return result


def _run_signal_mode_streaming(config: StrategyConfig,
                                codes: list[str],
                                start_date: Optional[str],
                                end_date: Optional[str],
                                progress_callback=None) -> BacktestResult:
    """
    信号模式 - 流式处理优化版：
    - 每只股票独立处理，不需要all_dates全市场遍历
    - 信号查找表替代逐日列表遍历
    - dict(zip)替代df.iterrows
    - 跳过无信号+无持仓日期
    """
    all_trades: list[Trade] = []
    all_dates_set = set()
    processed = 0

    for code in codes:
        try:
            df = load_stock_with_indicators(code, config.k_type)
        except FileNotFoundError:
            continue
        if len(df) < 60:
            continue

        if start_date:
            df = df[df["date"] >= start_date]
        if end_date:
            df = df[df["date"] <= end_date]
        if len(df) < 60:
            continue
        df = df.reset_index(drop=True)

        n = len(df)
        date_strs = df["date"].astype(str).str[:10].values
        close_arr = df["close"].values
        open_arr = df["open"].values
        high_arr = df["high"].values
        low_arr = df["low"].values

        # date_to_pos: 快速查找
        date_to_pos = dict(zip(date_strs, range(n)))

        # === 信号检测（向量化） ===
        sig_map = detect_signals_vectorized(df, config)
        sig_buy = sig_map["buy"]
        sig_sell = sig_map["sell"]
        sig_add = sig_map["add"]
        sig_reduce = sig_map["reduce"]

        # === 交易执行 ===
        # open_lots: list of dict, 每个dict代表一笔持仓
        # {"buy_ds", "buy_price", "highest", "shares", "reason", "trade_id",
        #  "entry_layer", "exit_triggered": set()}
        open_lots: list[dict] = []

        # 分批建仓状态: trade_id → {"base_price": float, "entered_layers": set()}
        entry_ladder_state: dict[str, dict] = {}

        # zzh0.2: 交易级别状态追踪
        # add_triggered: 曾触发过 add_groups 的 trade_id（双底破坏清仓仅对这些trade_id生效）
        # reduce_triggered: 已触发过减仓的 trade_id（减仓仅触发一次）
        add_triggered: set[str] = set()
        reduce_triggered: set[str] = set()

        # 状态机模式
        use_sm = bool(config.state_machine)
        sm = None
        if use_sm:
            sm = ZZH10StateMachine(config.state_machine_params)
            sm.reset()
            sm.prepare(df)  # 预计算所有结构检测数组

        has_signals = bool(sig_buy or sig_sell or sig_add or sig_reduce) or use_sm

        for idx in range(n):
            ds = date_strs[idx]
            close_price = close_arr[idx]

            # 无信号且无持仓 → 跳过（状态机模式下也不跳过，但状态机已预计算）
            if not use_sm and not open_lots and idx not in sig_buy:
                all_dates_set.add(ds)
                continue

            # 计算执行价
            if config.buy_execution == "next_day" and idx + 1 < n:
                buy_exec_price = _resolve_price_arr(open_arr[idx+1], high_arr[idx+1],
                                                     low_arr[idx+1], close_arr[idx+1],
                                                     config.buy_price_type)
                buy_exec_ds = date_strs[idx + 1]
            else:
                buy_exec_price = _resolve_price_arr(open_arr[idx], high_arr[idx],
                                                     low_arr[idx], close_arr[idx],
                                                     config.buy_price_type)
                buy_exec_ds = ds

            if config.sell_execution == "next_day" and idx + 1 < n:
                sell_exec_price = _resolve_price_arr(open_arr[idx+1], high_arr[idx+1],
                                                      low_arr[idx+1], close_arr[idx+1],
                                                      config.sell_price_type)
                sell_exec_ds = date_strs[idx + 1]
            else:
                sell_exec_price = _resolve_price_arr(open_arr[idx], high_arr[idx],
                                                      low_arr[idx], close_arr[idx],
                                                      config.sell_price_type)
                sell_exec_ds = ds

            # 状态机评估
            sm_action = ""
            sm_reason = ""
            if use_sm and sm is not None:
                entry_p = open_lots[0]["buy_price"] if open_lots else 0.0
                entry_d = open_lots[0]["buy_ds"] if open_lots else ""
                sm_action, sm_reason = sm.evaluate(
                    df, idx,
                    has_position=bool(open_lots),
                    has_added=bool(add_triggered),
                    entry_price=entry_p,
                    entry_date=str(entry_d),
                )

            # 1. 清仓信号（双底破坏仅在加仓触发后生效）
            # 涨跌停过滤：跌停封板无量时跳过卖出
            sell_limit_blocked = False
            if config.limit_filter and sell_exec_price > 0:
                pct_chg_s = df["pct_change"].iloc[idx]
                tr_s = df.get("turnover_rate", pd.Series([0] * len(df))).iloc[idx]
                if _is_limit_locked(pct_chg_s, tr_s):
                    sell_limit_blocked = True

            if not sell_limit_blocked and (idx in sig_sell or sm_action == "sell") and open_lots and sell_exec_price > 0:
                current_tid = open_lots[0]["trade_id"] if open_lots else ""
                should_sell = False
                sell_reason_str = "sell_signal"

                if sm_action == "sell":
                    # 状态机卖点：直接使用状态机的理由
                    should_sell = True
                    sell_reason_str = sm_reason
                else:
                    # 重新评估哪个 sell_group 触发，区分"双底破坏"和"连续低于slow"
                    for sell_grp in config.sell_groups:
                        grp_name = sell_grp.get("name", "")
                        ok, _ = check_group(df, idx, sell_grp)
                        if not ok:
                            continue
                        # 双底破坏清仓: 仅在 add_groups 触发过才生效
                        if "双底" in grp_name and current_tid not in add_triggered:
                            continue
                        should_sell = True
                        sell_reason_str = grp_name
                        break

                if should_sell:
                    while open_lots:
                        lot = open_lots.pop(0)
                        profit_pct = _net_profit_pct(lot["buy_price"], sell_exec_price,
                                                      config.commission_rate, config.stamp_tax_rate)
                        hold_days = _calc_hold_days(lot["buy_ds"], sell_exec_ds)
                        all_trades.append(Trade(
                            code=code, buy_date=lot["buy_ds"], buy_price=lot["buy_price"],
                            sell_date=sell_exec_ds, sell_price=sell_exec_price,
                            sell_reason=sell_reason_str, shares=lot["shares"],
                            profit_pct=round(profit_pct, 2),
                            profit_amount=round(profit_pct, 2),
                            hold_days=hold_days, action="clear",
                            trade_id=lot["trade_id"],
                        ))
                    # 清理状态
                    entry_ladder_state.clear()
                    add_triggered.discard(current_tid)
                    reduce_triggered.discard(current_tid)

            # 2. 减仓信号（一次性触发：每个trade_id仅减一次，按总仓位比例）
            if (idx in sig_reduce or sm_action == "reduce") and open_lots and sell_exec_price > 0:
                current_tid = open_lots[0]["trade_id"] if open_lots else ""
                # zzh0.2: 减仓仅触发一次
                if current_tid not in reduce_triggered:
                    reduce_triggered.add(current_tid)
                    reduce_pct = config.reduce_pct
                    min_unit = config.min_trade_unit
                    # 按总仓位计算减仓量
                    total_shares = sum(lot["shares"] for lot in open_lots)
                    raw_reduce = int(total_shares * reduce_pct)
                    reduce_shares = (raw_reduce // min_unit) * min_unit
                    remaining_after = total_shares - reduce_shares

                    if remaining_after <= min_unit or reduce_shares == 0:
                        # 剩余太少 → 全部清仓
                        while open_lots:
                            lot = open_lots.pop(0)
                            profit_pct = _net_profit_pct(lot["buy_price"], sell_exec_price,
                                                          config.commission_rate, config.stamp_tax_rate)
                            hold_days = _calc_hold_days(lot["buy_ds"], sell_exec_ds)
                            all_trades.append(Trade(
                                code=code, buy_date=lot["buy_ds"], buy_price=lot["buy_price"],
                                sell_date=sell_exec_ds, sell_price=sell_exec_price,
                                sell_reason="reduce_clear", shares=lot["shares"],
                                profit_pct=round(profit_pct, 2),
                                profit_amount=round(profit_pct, 2),
                                hold_days=hold_days, action="clear",
                                trade_id=lot["trade_id"],
                            ))
                        entry_ladder_state.clear()
                        add_triggered.discard(current_tid)
                        reduce_triggered.discard(current_tid)
                    else:
                        # 部分减仓：从后往前扣减（LIFO），每个lot记录独立Trade
                        to_reduce = reduce_shares
                        while to_reduce > 0 and open_lots:
                            lot = open_lots[-1]
                            if lot["shares"] <= to_reduce:
                                # 整个lot被消耗
                                to_reduce -= lot["shares"]
                                open_lots.pop()
                                profit_pct = _net_profit_pct(lot["buy_price"], sell_exec_price,
                                                              config.commission_rate, config.stamp_tax_rate)
                                hold_days = _calc_hold_days(lot["buy_ds"], sell_exec_ds)
                                all_trades.append(Trade(
                                    code=code, buy_date=lot["buy_ds"], buy_price=lot["buy_price"],
                                    sell_date=sell_exec_ds, sell_price=sell_exec_price,
                                    sell_reason="reduce_signal", shares=lot["shares"],
                                    profit_pct=round(profit_pct, 2),
                                    profit_amount=round(profit_pct, 2),
                                    hold_days=hold_days, action="reduce",
                                    trade_id=lot["trade_id"],
                                ))
                            else:
                                # 部分消耗该lot
                                sold = to_reduce
                                lot["shares"] -= to_reduce
                                to_reduce = 0
                                profit_pct = _net_profit_pct(lot["buy_price"], sell_exec_price,
                                                              config.commission_rate, config.stamp_tax_rate)
                                hold_days = _calc_hold_days(lot["buy_ds"], sell_exec_ds)
                                all_trades.append(Trade(
                                    code=code, buy_date=lot["buy_ds"], buy_price=lot["buy_price"],
                                    sell_date=sell_exec_ds, sell_price=sell_exec_price,
                                    sell_reason="reduce_signal", shares=sold,
                                    profit_pct=round(profit_pct, 2),
                                    profit_amount=round(profit_pct, 2),
                                    hold_days=hold_days, action="reduce",
                                    trade_id=lot["trade_id"],
                                ))

            # 3. 分批止盈检查 (Exit Ladder)
            if config.exit_ladder and open_lots and sell_exec_price > 0:
                min_unit = config.min_trade_unit
                to_close_partial = []
                for lot in open_lots:
                    lot_profit = (close_price - lot["buy_price"]) / lot["buy_price"] * 100
                    # 计算相对最高价的盈利（用于 from_highest 模式）
                    highest_profit = (lot["highest"] - lot["buy_price"]) / lot["buy_price"] * 100 if lot["highest"] > lot["buy_price"] else 0
                    # 计算当前价格相对最高价的位置
                    from_highest_pct = (close_price - lot["highest"]) / lot["highest"] * 100 if lot["highest"] > 0 else 0

                    for level_idx, level in enumerate(config.exit_ladder):
                        if level_idx in lot["exit_triggered"]:
                            continue

                        # 判断是否触发止盈
                        should_trigger = False
                        trigger_reason = ""

                        if level.get("use_highest", False):
                            # 模式1：基于持仓最高价
                            # 当价格回到最高价时触发（from_highest_pct >= 0）
                            if from_highest_pct >= 0 and highest_profit > 0:
                                should_trigger = True
                                trigger_reason = f"回到前高+{highest_profit:.1f}%"
                        elif level.get("from_highest_pct", 0) > 0:
                            # 模式2：基于最高价的涨幅
                            # 当价格从最高价再涨X%时触发
                            if highest_profit > 0 and from_highest_pct >= level["from_highest_pct"]:
                                should_trigger = True
                                trigger_reason = f"前高+{level['from_highest_pct']}%"
                        else:
                            # 模式3：基于买入价（传统模式）
                            if lot_profit >= level["profit_pct"]:
                                should_trigger = True
                                trigger_reason = f"分批止盈+{level['profit_pct']}%"

                        if should_trigger:
                            # 触发该层止盈，向下取整到最小交易单位
                            raw_close = int(lot["shares"] * level["close_pct"] / 100)
                            close_shares = (raw_close // min_unit) * min_unit
                            remaining = lot["shares"] - close_shares
                            # 剩余 <= 最小交易单位 或 止盈量不足一个单位 → 清仓
                            if remaining <= min_unit or close_shares == 0:
                                close_shares = lot["shares"]
                            actual_close = min(close_shares, lot["shares"])
                            if actual_close > 0:
                                is_clear = (actual_close == lot["shares"])
                                action = "clear" if is_clear else "reduce"
                                reason = trigger_reason if not is_clear else f"{trigger_reason}清仓"
                                to_close_partial.append((lot, level_idx, actual_close, reason, action))

                # 直接使用 lot 对象引用，避免索引错位
                for lot, level_idx, close_shares, reason, action in to_close_partial:
                    profit_pct = _net_profit_pct(lot["buy_price"], sell_exec_price,
                                                  config.commission_rate, config.stamp_tax_rate)
                    hold_days = _calc_hold_days(lot["buy_ds"], sell_exec_ds)
                    actual_close = min(close_shares, lot["shares"])
                    all_trades.append(Trade(
                        code=code, buy_date=lot["buy_ds"], buy_price=lot["buy_price"],
                        sell_date=sell_exec_ds, sell_price=sell_exec_price,
                        sell_reason=reason, shares=actual_close,
                        profit_pct=round(profit_pct, 2),
                        profit_amount=round(profit_pct, 2),
                        hold_days=hold_days, action=action,
                        trade_id=lot["trade_id"],
                    ))
                    lot["exit_triggered"].add(level_idx)
                    lot["shares"] -= actual_close

                # 统一清理空仓位
                open_lots[:] = [lot for lot in open_lots if lot["shares"] > 0]

            # 4. 风控检查
            if open_lots and sell_exec_price > 0:
                to_close = []
                for i, lot in enumerate(open_lots):
                    loss_pct = (close_price - lot["buy_price"]) / lot["buy_price"] * 100
                    should_close = False
                    close_reason = ""

                    if config.stop_loss_pct > 0 and loss_pct <= -config.stop_loss_pct:
                        should_close = True
                        close_reason = f"止损{-config.stop_loss_pct}%"
                    elif config.take_profit_pct > 0 and loss_pct >= config.take_profit_pct:
                        should_close = True
                        close_reason = f"止盈+{config.take_profit_pct}%"
                    elif config.trailing_stop_tiers and lot["highest"] > lot["buy_price"]:
                        profit = (lot["highest"] - lot["buy_price"]) / lot["buy_price"] * 100
                        trail = _effective_trail(profit, config)
                        dd = (lot["highest"] - close_price) / lot["highest"] * 100
                        if dd >= trail:
                            close_reason = f"动态移动止损-{trail}%"
                    elif config.trailing_stop_pct > 0 and lot["highest"] > lot["buy_price"]:
                        dd = (lot["highest"] - close_price) / lot["highest"] * 100
                        if dd >= config.trailing_stop_pct:
                            should_close = True
                            close_reason = f"移动止损-{config.trailing_stop_pct}%"
                    elif config.max_hold_days > 0:
                        hd = _calc_hold_days(lot["buy_ds"], ds)
                        if hd >= config.max_hold_days:
                            should_close = True
                            close_reason = f"超期{config.max_hold_days}天"

                    if should_close:
                        to_close.append((i, close_reason))

                for i, reason in reversed(to_close):
                    lot = open_lots.pop(i)
                    profit_pct = _net_profit_pct(lot["buy_price"], sell_exec_price,
                                                  config.commission_rate, config.stamp_tax_rate)
                    hold_days = _calc_hold_days(lot["buy_ds"], sell_exec_ds)
                    all_trades.append(Trade(
                        code=code, buy_date=lot["buy_ds"], buy_price=lot["buy_price"],
                        sell_date=sell_exec_ds, sell_price=sell_exec_price,
                        sell_reason=reason, shares=lot["shares"],
                        profit_pct=round(profit_pct, 2),
                        profit_amount=round(profit_pct, 2),
                        hold_days=hold_days, action="clear",
                        trade_id=lot["trade_id"],
                    ))
                # 风控清仓后清理追踪状态
                if not open_lots:
                    add_triggered.clear()
                    reduce_triggered.clear()

            # 5. 更新持仓最高价
            for lot in open_lots:
                if close_price > lot["highest"]:
                    lot["highest"] = close_price

            # 6. 分批建仓检查 (Entry Ladder)
            if config.entry_ladder and open_lots and buy_exec_price > 0:
                # 检查每个trade_id是否还有未执行的entry ladder层级
                for tid, state in list(entry_ladder_state.items()):
                    base_price = state["base_price"]
                    entered = state["entered_layers"]
                    for layer_idx, layer in enumerate(config.entry_ladder):
                        if layer_idx in entered:
                            continue
                        if layer["trigger_pct"] == 0:
                            continue  # 第0层在买入时已处理
                        # 从首笔买入价计算浮盈
                        profit_from_base = (buy_exec_price - base_price) / base_price * 100
                        if profit_from_base >= layer["trigger_pct"]:
                            # 触发该层加仓
                            new_lot = {
                                "buy_ds": buy_exec_ds,
                                "buy_price": buy_exec_price,
                                "highest": buy_exec_price,
                                "shares": layer["weight"],
                                "reason": f"entry_ladder_L{layer_idx}",
                                "trade_id": tid,
                                "entry_layer": layer_idx,
                                "exit_triggered": set(),
                            }
                            open_lots.append(new_lot)
                            entered.add(layer_idx)

            # 7. 买入信号（排他锁：同一标的已有持仓时不开新仓，可通过 exclusive_lock 关闭）
            # 涨跌停过滤：涨停封板无量时跳过买入
            limit_blocked = False
            if config.limit_filter and buy_exec_price > 0:
                pct_chg = df["pct_change"].iloc[idx]
                tr = df.get("turnover_rate", pd.Series([0] * len(df))).iloc[idx]
                if _is_limit_locked(pct_chg, tr):
                    limit_blocked = True

            if not limit_blocked and (idx in sig_buy or sm_action == "buy") and buy_exec_price > 0 and (not open_lots or not config.exclusive_lock):
                new_tid = f"T{uuid.uuid4().hex[:8]}"
                buy_reason = sm_reason if sm_action == "buy" else "buy_signal"
                if config.entry_ladder:
                    # 分批建仓模式: 首笔按entry_ladder[0]的weight
                    first_weight = config.entry_ladder[0]["weight"] if config.entry_ladder else 100
                    new_lot = {
                        "buy_ds": buy_exec_ds,
                        "buy_price": buy_exec_price,
                        "highest": buy_exec_price,
                        "shares": first_weight,
                        "reason": buy_reason,
                        "trade_id": new_tid,
                        "entry_layer": 0,
                        "exit_triggered": set(),
                    }
                    open_lots.append(new_lot)
                    entry_ladder_state[new_tid] = {
                        "base_price": buy_exec_price,
                        "entered_layers": {0},
                    }
                else:
                    # 传统模式: 一次性满仓
                    new_lot = {
                        "buy_ds": buy_exec_ds,
                        "buy_price": buy_exec_price,
                        "highest": buy_exec_price,
                        "shares": 100,
                        "reason": buy_reason,
                        "trade_id": new_tid,
                        "entry_layer": 0,
                        "exit_triggered": set(),
                    }
                    open_lots.append(new_lot)

            # 8. 加仓信号（add_groups / 状态机，与entry_ladder独立）
            # zzh0.2: 加仓数量=entry_ladder[0].weight（与建仓等量），标记 add_triggered
            # zzh0.2: 每个 trade_id 仅加仓1次
            if (idx in sig_add or sm_action == "add") and open_lots and buy_exec_price > 0:
                current_tid = open_lots[0]["trade_id"]
                if current_tid in add_triggered:
                    pass  # 已加过仓，跳过
                else:
                    add_reason = sm_reason if sm_action == "add" else "add_signal"
                    # 加仓股数：优先取 entry_ladder[0].weight，否则默认100
                    add_shares = config.entry_ladder[0]["weight"] if config.entry_ladder else 100
                    if config.add_threshold > 0:
                        last_lot = open_lots[-1]
                        profit_from_last = (buy_exec_price - last_lot["buy_price"]) / last_lot["buy_price"] * 100
                        if profit_from_last >= config.add_threshold:
                            new_lot = {
                                "buy_ds": buy_exec_ds,
                                "buy_price": buy_exec_price,
                                "highest": buy_exec_price,
                                "shares": add_shares,
                                "reason": add_reason,
                                "trade_id": last_lot["trade_id"],
                                "entry_layer": last_lot.get("entry_layer", 0) + 1,
                                "exit_triggered": set(),
                            }
                            open_lots.append(new_lot)
                            add_triggered.add(last_lot["trade_id"])
                    else:
                        last_lot = open_lots[-1]
                        new_lot = {
                            "buy_ds": buy_exec_ds,
                            "buy_price": buy_exec_price,
                            "highest": buy_exec_price,
                            "shares": add_shares,
                            "reason": add_reason,
                            "trade_id": last_lot["trade_id"],
                            "entry_layer": last_lot.get("entry_layer", 0) + 1,
                            "exit_triggered": set(),
                        }
                        open_lots.append(new_lot)
                        add_triggered.add(last_lot["trade_id"])

            all_dates_set.add(ds)

        # 回测结束平仓
        if open_lots and n > 0:
            last_ds = date_strs[-1]
            last_price = close_arr[-1]
            for lot in open_lots:
                profit_pct = _net_profit_pct(lot["buy_price"], last_price,
                                              config.commission_rate, config.stamp_tax_rate)
                hold_days = _calc_hold_days(lot["buy_ds"], last_ds)
                all_trades.append(Trade(
                    code=code, buy_date=lot["buy_ds"], buy_price=lot["buy_price"],
                    sell_date=last_ds, sell_price=last_price,
                    sell_reason="回测结束平仓", shares=lot["shares"],
                    profit_pct=round(profit_pct, 2),
                    profit_amount=round(profit_pct, 2),
                    hold_days=hold_days, action="clear",
                    trade_id=lot["trade_id"],
                ))

        processed += 1
        if progress_callback and processed % 10 == 0:
            progress_callback(processed, len(codes))

    if progress_callback:
        progress_callback(len(codes), len(codes))

    all_dates = sorted(all_dates_set)
    # 转为pd.Timestamp列表以兼容_compute_statistics
    all_dates = [pd.Timestamp(d) for d in all_dates]

    # 构建简易权益曲线 — 信号模式用累计求和（每笔独立），避免复利爆炸
    equity_curve = []
    if all_trades:
        base = config.initial_capital
        cum_pnl = base
        cumulative_sum_pct = 0.0
        # 按卖出日期排序
        sorted_trades = sorted(all_trades, key=lambda t: str(t.sell_date))
        for t in sorted_trades:
            cumulative_sum_pct += t.profit_pct
            cum_pnl = base * (1 + cumulative_sum_pct / 100)
            equity_curve.append({
                "date": t.sell_date,
                "equity": round(max(cum_pnl, 0.01), 2),
                "cash": 0,
                "positions": 0,
                "cum_return_pct": round(cumulative_sum_pct, 2),
            })

    return _compute_statistics(config, all_trades, equity_curve, all_dates)


def _resolve_price_arr(open_p: float, high_p: float, low_p: float,
                        close_p: float, price_type: str) -> float:
    """从numpy数组值解析价格类型"""
    if price_type == "open":
        return open_p
    elif price_type == "high":
        return high_p
    elif price_type == "low":
        return low_p
    elif price_type == "close":
        return close_p
    elif price_type == "avg":
        return (high_p + low_p) / 2
    elif price_type == "typical":
        return (high_p + low_p + close_p) / 3
    elif price_type == "vwap":
        return (open_p + high_p + low_p + close_p) / 4
    return close_p


def _run_signal_mode(config: StrategyConfig,
                     all_signals: dict[str, list[Signal]],
                     all_dfs: dict[str, pd.DataFrame]) -> BacktestResult:
    """
    信号纯评估模式：
    - 每个买点独立执行，无资金占用概念
    - 同股票可多次加仓，支持减仓和清仓
    - 每笔操作记录 action: buy/add/reduce/clear
    - 目标：评估策略本身的胜率和盈亏比
    """
    all_dates = set()
    for df in all_dfs.values():
        all_dates.update(df["date"].tolist())
    all_dates = sorted(all_dates)

    trades: list[Trade] = []

    # 为每只股票建立价格索引
    for code in all_dfs:
        all_dfs[code]["_date_str"] = all_dfs[code]["date"].apply(lambda x: str(x)[:10])

    date_strs = [str(d)[:10] for d in all_dates]

    def _get_exec_price(signal_ds: str, price_type: str, execution: str,
                        date_to_pos: dict, df: pd.DataFrame) -> tuple[float, str]:
        """根据成交价策略和执行时机，返回 (成交价, 成交日期)

        same_day: 信号日当天用 price_type 取价
        next_day: 信号日在该股票的下一个交易日用 price_type 取价
        date_to_pos: 日期→DataFrame位置索引（用enumerate生成，确保iloc正确）
        """
        if signal_ds not in date_to_pos:
            return -1, signal_ds

        pos = date_to_pos[signal_ds]

        if execution == "next_day":
            # 该股票的下一行数据即为下一个交易日
            if pos + 1 >= len(df):
                # 信号日在该股票最后一天，无法次日执行
                return -1, signal_ds
            next_row = df.iloc[pos + 1]
            next_ds = next_row["_date_str"]
            if next_ds <= signal_ds:
                return -1, signal_ds
            return resolve_price(df, pos + 1, price_type), next_ds
        else:
            # same_day
            return resolve_price(df, pos, price_type), signal_ds

    # 按股票逐个处理
    for code, sigs in all_signals.items():
        if not sigs or code not in all_dfs:
            continue

        df = all_dfs[code]
        # 用 enumerate 生成位置索引，确保 iloc 正确
        date_to_pos: dict[str, int] = {}
        for pos, (i, row) in enumerate(df.iterrows()):
            date_to_pos[row["_date_str"]] = pos

        # 持仓层：(buy_date, buy_price, highest_since_buy, shares, reason, trade_id)
        # signal模式每层等量，减仓时从末尾减
        open_lots: list[tuple[str, float, float, int, str, str]] = []

        for ds in date_strs:
            if ds not in date_to_pos or code not in all_dfs:
                continue
            pos = date_to_pos[ds]
            # 风控检查和持仓最高价更新仍用 close（当日收盘价）
            close_price = all_dfs[code].iloc[pos]["close"]
            # 卖出执行价用 sell_price_type
            sell_exec_price, sell_exec_ds = _get_exec_price(
                ds, config.sell_price_type, config.sell_execution,
                date_to_pos, df)
            # 买入执行价用 buy_price_type
            buy_exec_price, buy_exec_ds = _get_exec_price(
                ds, config.buy_price_type, config.buy_execution,
                date_to_pos, df)

            # 当天信号
            day_sigs = [s for s in sigs if str(s.date)[:10] == ds]

            # === 1. 先处理清仓信号 (sell) ===
            day_sells = [s for s in day_sigs if s.signal_type == "sell"]
            if day_sells and open_lots and sell_exec_price > 0:
                for sell_sig in day_sells:
                    # 清仓：卖出所有持仓层
                    while open_lots:
                        bd, bp, hi, sh, reason, tid = open_lots.pop(0)
                        profit_pct = _net_profit_pct(bp, sell_exec_price,
                                                      config.commission_rate, config.stamp_tax_rate)
                        hold_days = _calc_hold_days(bd, sell_exec_ds)
                        trades.append(Trade(
                            code=code, buy_date=bd, buy_price=bp,
                            sell_date=sell_exec_ds, sell_price=sell_exec_price,
                            sell_reason=sell_sig.reason, shares=sh,
                            profit_pct=round(profit_pct, 2),
                            profit_amount=round(profit_pct, 2),
                            hold_days=hold_days,
                            action="clear",
                            trade_id=tid,
                        ))

            # === 2. 处理减仓信号 (reduce) ===
            day_reduces = [s for s in day_sigs if s.signal_type == "reduce"]
            if day_reduces and open_lots and sell_exec_price > 0:
                reduce_pct = config.reduce_pct
                min_unit = config.min_trade_unit
                for reduce_sig in day_reduces:
                    # 从最后一层开始减
                    if open_lots:
                        bd, bp, hi, sh, reason, tid = open_lots[-1]
                        raw_reduce = int(sh * reduce_pct)
                        # 向下取整到最小交易单位的倍数
                        reduce_shares = (raw_reduce // min_unit) * min_unit
                        remaining = sh - reduce_shares
                        # 剩余 <= 最小交易单位 或 减仓量不足一个单位 → 清仓
                        if remaining <= min_unit or reduce_shares == 0:
                            reduce_shares = sh
                        is_clear = (reduce_shares == sh)
                        if is_clear:
                            # 整层清掉
                            open_lots.pop()
                        else:
                            # 部分减仓
                            open_lots[-1] = (bd, bp, hi, remaining, reason, tid)
                        profit_pct = _net_profit_pct(bp, sell_exec_price,
                                                  config.commission_rate, config.stamp_tax_rate)
                        hold_days = _calc_hold_days(bd, sell_exec_ds)
                        trades.append(Trade(
                            code=code, buy_date=bd, buy_price=bp,
                            sell_date=sell_exec_ds, sell_price=sell_exec_price,
                            sell_reason="reduce_clear" if is_clear else reduce_sig.reason,
                            shares=reduce_shares,
                            profit_pct=round(profit_pct, 2),
                            profit_amount=round(profit_pct, 2),
                            hold_days=hold_days,
                            action="clear" if is_clear else "reduce",
                            trade_id=tid,
                        ))

            # === 3. 风控检查（止损/止盈/移动止损）— 用 close 评判，用 sell_exec_price 执行 ===
            to_close = []
            for i, (bd, bp, hi, sh, reason, tid) in enumerate(open_lots):
                loss_pct = (close_price - bp) / bp * 100
                should_close = False
                close_reason = ""

                if config.stop_loss_pct > 0 and loss_pct <= -config.stop_loss_pct:
                    should_close = True
                    close_reason = f"止损{-config.stop_loss_pct}%"
                elif config.take_profit_pct > 0 and loss_pct >= config.take_profit_pct:
                    should_close = True
                    close_reason = f"止盈+{config.take_profit_pct}%"
                elif config.trailing_stop_tiers and hi > bp:
                    profit = (hi - bp) / bp * 100
                    trail = _effective_trail(profit, config)
                    dd = (hi - close_price) / hi * 100
                    if dd >= trail:
                        should_close = True
                        close_reason = f"动态移动止损-{trail}%"
                elif config.trailing_stop_pct > 0 and hi > bp:
                    dd = (hi - close_price) / hi * 100
                    if dd >= config.trailing_stop_pct:
                        should_close = True
                        close_reason = f"移动止损-{config.trailing_stop_pct}%"
                elif config.max_hold_days > 0:
                    hd = _calc_hold_days(bd, ds)
                    if hd >= config.max_hold_days:
                        should_close = True
                        close_reason = f"超期{config.max_hold_days}天"

                if should_close:
                    to_close.append((i, close_reason))

            # 风控执行用 sell_exec_price
            if sell_exec_price > 0:
                for i, reason in reversed(to_close):
                    bd, bp, _, sh, _, tid = open_lots.pop(i)
                    profit_pct = _net_profit_pct(bp, sell_exec_price,
                                                  config.commission_rate, config.stamp_tax_rate)
                    hold_days = _calc_hold_days(bd, sell_exec_ds)
                    trades.append(Trade(
                        code=code, buy_date=bd, buy_price=bp,
                        sell_date=sell_exec_ds, sell_price=sell_exec_price,
                        sell_reason=reason, shares=sh,
                        profit_pct=round(profit_pct, 2),
                        profit_amount=round(profit_pct, 2),
                        hold_days=hold_days,
                        action="clear",
                        trade_id=tid,
                    ))

            # === 4. 更新持仓最高价（用 close） ===
            for i in range(len(open_lots)):
                bd, bp, hi, sh, reason, tid = open_lots[i]
                if close_price > hi:
                    open_lots[i] = (bd, bp, close_price, sh, reason, tid)

            # === 5. 处理买入信号（排他锁：同一标的已有持仓时不开新仓，可通过 exclusive_lock 关闭）===
            day_buys = [s for s in day_sigs if s.signal_type == "buy"]
            for buy_sig in day_buys:
                if buy_exec_price > 0 and (not open_lots or not config.exclusive_lock):
                    new_tid = f"T{uuid.uuid4().hex[:8]}"
                    open_lots.append((buy_exec_ds, buy_exec_price, buy_exec_price, 100, buy_sig.reason, new_tid))

            # === 6. 处理加仓信号 ===
            day_adds = [s for s in day_sigs if s.signal_type == "add"]
            for add_sig in day_adds:
                # 加仓：只在有持仓时生效
                if open_lots and buy_exec_price > 0:
                    # 加仓条件：满足浮盈阈值
                    if config.add_threshold > 0:
                        # 检查最新持仓是否满足浮盈
                        _, last_bp, _, _, _, last_tid = open_lots[-1]
                        profit_from_last = (buy_exec_price - last_bp) / last_bp * 100
                        if profit_from_last < config.add_threshold:
                            continue
                    else:
                        _, _, _, _, _, last_tid = open_lots[-1]
                    open_lots.append((buy_exec_ds, buy_exec_price, buy_exec_price, 100, add_sig.reason, last_tid))

        # 回测结束时平仓（用 sell_price_type 的 same_day 取最后一天的价）
        if all_dates:
            last_ds = date_strs[-1]
            for bd, bp, _, sh, _, tid in open_lots:
                if code in all_dfs and last_ds in date_to_pos:
                    last_pos = date_to_pos[last_ds]
                    last_price = resolve_price(all_dfs[code], last_pos, config.sell_price_type)
                else:
                    last_price = bp
                profit_pct = _net_profit_pct(bp, last_price,
                                              config.commission_rate, config.stamp_tax_rate)
                hold_days = _calc_hold_days(bd, last_ds)
                trades.append(Trade(
                    code=code, buy_date=bd, buy_price=bp,
                    sell_date=last_ds, sell_price=last_price,
                    sell_reason="回测结束平仓", shares=sh,
                    profit_pct=round(profit_pct, 2),
                    profit_amount=round(profit_pct, 2),
                    hold_days=hold_days,
                    action="clear",
                    trade_id=tid,
                ))

    # === 构建简易权益曲线（信号模式用累计求和，每笔独立） ===
    equity_curve = []
    if trades:
        base = config.initial_capital
        cum_pnl = base
        cumulative_sum_pct = 0.0
        sorted_trades = sorted(trades, key=lambda t: str(t.sell_date))
        for t in sorted_trades:
            cumulative_sum_pct += t.profit_pct
            cum_pnl = base * (1 + cumulative_sum_pct / 100)
            equity_curve.append({
                "date": t.sell_date,
                "equity": round(max(cum_pnl, 0.01), 2),
                "cash": 0,
                "positions": 0,
            })

    return _compute_statistics(config, trades, equity_curve, all_dates)


def _run_portfolio_mode(config: StrategyConfig,
                        all_signals: dict[str, list[Signal]],
                        all_dfs: dict[str, pd.DataFrame]) -> BacktestResult:
    """
    传统资金组合模式：
    - 有限初始资金，按仓位比例分配
    - 最大持仓数限制
    - 资金被占用时无法开新仓
    """
    all_dates = set()
    for df in all_dfs.values():
        all_dates.update(df["date"].tolist())
    all_dates = sorted(all_dates)
    date_strs = [str(d)[:10] for d in all_dates]

    # 建立日期→下一个交易日映射
    next_day_map: dict[str, str] = {}
    for i in range(len(date_strs) - 1):
        next_day_map[date_strs[i]] = date_strs[i + 1]

    trades: list[Trade] = []
    positions: dict[str, Position] = {}
    capital = config.initial_capital
    available_cash = capital
    equity_curve = []

    # 按信号日期索引
    buy_signals: dict[str, list[Signal]] = {}
    sell_signals: dict[str, list[Signal]] = {}
    for code, sigs in all_signals.items():
        for s in sigs:
            ds = str(s.date)[:10]
            if s.signal_type == "buy":
                buy_signals.setdefault(ds, []).append(s)
            else:
                sell_signals.setdefault(ds, []).append(s)

    for date in all_dates:
        ds = str(date)[:10]

        # 处理卖出
        codes_to_sell = []
        for code, pos in positions.items():
            if code not in all_dfs:
                continue
            df = all_dfs[code]
            row = df[df["date"] == date]
            if row.empty:
                continue
            # 风控评判用 close
            close_price = row["close"].iloc[0]

            if close_price > pos.highest_price:
                pos.highest_price = close_price

            sell_reason = ""
            should_sell = False

            if ds in sell_signals:
                for s in sell_signals[ds]:
                    if s.code == code:
                        should_sell = True
                        sell_reason = s.reason
                        break

            loss_pct = (close_price - pos.buy_price) / pos.buy_price * 100
            if config.stop_loss_pct > 0 and loss_pct <= -config.stop_loss_pct:
                should_sell = True
                sell_reason = f"止损{-config.stop_loss_pct}%"

            profit_pct = _net_profit_pct(pos.buy_price, close_price,
                                          config.commission_rate, config.stamp_tax_rate)
            if config.take_profit_pct > 0 and profit_pct >= config.take_profit_pct:
                should_sell = True
                sell_reason = f"止盈+{config.take_profit_pct}%"

            if config.trailing_stop_tiers and pos.highest_price > pos.buy_price:
                profit = (pos.highest_price - pos.buy_price) / pos.buy_price * 100
                trail = _effective_trail(profit, config)
                drawdown = (pos.highest_price - close_price) / pos.highest_price * 100
                if drawdown >= trail:
                    should_sell = True
                    sell_reason = f"动态移动止损-{trail}%"
            elif config.trailing_stop_pct > 0 and pos.highest_price > pos.buy_price:
                drawdown = (pos.highest_price - close_price) / pos.highest_price * 100
                if drawdown >= config.trailing_stop_pct:
                    should_sell = True
                    sell_reason = f"移动止损-{config.trailing_stop_pct}%"

            if should_sell:
                # 计算实际卖出价和日期
                if config.sell_execution == "next_day" and ds in next_day_map:
                    exec_ds = next_day_map[ds]
                    exec_df = all_dfs.get(code)
                    if exec_df is not None:
                        exec_row = exec_df[exec_df["_date_str"] == exec_ds] if "_date_str" in exec_df.columns else exec_df[exec_df["date"].astype(str).str[:10] == exec_ds]
                        if exec_row.empty:
                            exec_price = close_price
                            exec_ds = ds
                        else:
                            exec_idx = exec_row.index[0]
                            exec_price = resolve_price(exec_df, exec_idx, config.sell_price_type)
                    else:
                        exec_price = close_price
                        exec_ds = ds
                else:
                    exec_price = resolve_price(df, row.index[0], config.sell_price_type)
                    exec_ds = ds
                codes_to_sell.append((code, exec_price, sell_reason, exec_ds))

        for code, sell_price, reason, sell_ds in codes_to_sell:
            pos = positions.pop(code)
            profit = (sell_price - pos.buy_price) * pos.shares
            profit_pct = _net_profit_pct(pos.buy_price, sell_price,
                                          config.commission_rate, config.stamp_tax_rate)
            hold_days = _calc_hold_days(str(pos.buy_date)[:10], str(sell_ds)[:10])

            available_cash += sell_price * pos.shares

            trades.append(Trade(
                code=code,
                buy_date=pos.buy_date,
                buy_price=pos.buy_price,
                sell_date=sell_ds,
                sell_price=sell_price,
                sell_reason=reason,
                shares=pos.shares,
                profit_pct=round(profit_pct, 2),
                profit_amount=round(profit, 2),
                hold_days=hold_days,
            ))

        # 处理买入
        if ds in buy_signals and len(positions) < config.max_positions:
            candidates = buy_signals[ds]
            for sig in candidates:
                if config.exclusive_lock and sig.code in positions:
                    continue
                if len(positions) >= config.max_positions:
                    break
                if sig.code not in all_dfs:
                    continue

                # 计算实际买入价和日期
                sig_df = all_dfs[sig.code]
                if config.buy_execution == "next_day" and ds in next_day_map:
                    buy_ds = next_day_map[ds]
                    buy_row = sig_df[sig_df["_date_str"] == buy_ds] if "_date_str" in sig_df.columns else sig_df[sig_df["date"].astype(str).str[:10] == buy_ds]
                    if buy_row.empty:
                        continue  # 次日无数据，跳过
                    buy_price = resolve_price(sig_df, buy_row.index[0], config.buy_price_type)
                else:
                    sig_row = sig_df[sig_df["date"] == date]
                    if sig_row.empty:
                        continue
                    buy_price = resolve_price(sig_df, sig_row.index[0], config.buy_price_type)
                    buy_ds = ds

                position_cash = available_cash * config.position_pct
                if position_cash <= 0:
                    continue

                shares = int(position_cash / buy_price / 100) * 100
                if shares < 100:
                    continue

                cost = shares * buy_price
                if cost > available_cash:
                    continue

                available_cash -= cost
                positions[sig.code] = Position(
                    code=sig.code,
                    buy_date=date if config.buy_execution == "same_day" else buy_ds,
                    buy_price=buy_price,
                    shares=shares,
                    cost=cost,
                    highest_price=buy_price,
                )

        # 记录权益曲线
        total_equity = available_cash
        for code, pos in positions.items():
            if code in all_dfs:
                df = all_dfs[code]
                row = df[df["date"] == date]
                if not row.empty:
                    total_equity += row["close"].iloc[0] * pos.shares

        equity_curve.append({
            "date": ds,
            "equity": round(total_equity, 2),
            "cash": round(available_cash, 2),
            "positions": len(positions),
        })

    # 强制平仓
    if positions:
        last_date = all_dates[-1] if all_dates else None
        last_ds = str(last_date)[:10] if last_date else ""
        for code, pos in list(positions.items()):
            if code in all_dfs:
                df = all_dfs[code]
                last_price = resolve_price(df, len(df) - 1, config.sell_price_type)
                profit = (last_price - pos.buy_price) * pos.shares
                profit_pct = _net_profit_pct(pos.buy_price, last_price,
                                              config.commission_rate, config.stamp_tax_rate)
                hold_days = _calc_hold_days(str(pos.buy_date)[:10], last_ds) if last_date else 0

                available_cash += last_price * pos.shares

                trades.append(Trade(
                    code=code,
                    buy_date=pos.buy_date,
                    buy_price=pos.buy_price,
                    sell_date=last_date,
                    sell_price=last_price,
                    sell_reason="回测结束平仓",
                    shares=pos.shares,
                    profit_pct=round(profit_pct, 2),
                    profit_amount=round(profit, 2),
                    hold_days=hold_days,
                ))
        positions.clear()

    return _compute_statistics(config, trades, equity_curve, all_dates)


def _calc_hold_days(buy_date: str, sell_date: str) -> int:
    """计算持仓天数"""
    try:
        bd = pd.Timestamp(buy_date)
        sd = pd.Timestamp(sell_date)
        return (sd - bd).days
    except Exception:
        return 0


def _compute_statistics(config: StrategyConfig,
                         trades: list[Trade],
                         equity_curve: list[dict],
                         all_dates: list) -> BacktestResult:
    """计算统计指标（两种模式共用）

    signal 模式：基于交易盈亏累计的权益曲线计算所有指标（不再是僵尸指标）
    portfolio 模式：基于真实资金权益曲线计算所有指标
    """
    final_capital = config.initial_capital
    if equity_curve:
        final_capital = equity_curve[-1]["equity"]

    # 总收益：signal模式也能算（从交易累计权益曲线）
    total_return = (final_capital - config.initial_capital) / config.initial_capital * 100 if config.initial_capital > 0 else 0.0

    # 年化收益率
    if all_dates and len(all_dates) > 1:
        days = (all_dates[-1] - all_dates[0]).days
        if days > 0 and final_capital > 0:
            annual_return = ((final_capital / config.initial_capital) ** (365 / days) - 1) * 100
        else:
            annual_return = 0
    else:
        annual_return = 0

    # 最大回撤
    if config.backtest_mode == "signal" and equity_curve and len(equity_curve) > 1:
        # 信号模式：从累计收益率曲线算 MaxDD（避免兜底伪影）
        cum_returns = [0.0] + [e.get("cum_return_pct", (e["equity"] / config.initial_capital - 1) * 100)
                                for e in equity_curve]
        peak = cum_returns[0]
        max_dd = 0.0
        for r in cum_returns:
            if r > peak:
                peak = r
            dd = r - peak
            if dd < max_dd:
                max_dd = dd
        # max_dd 是累计收益率的百分点跌幅，转为相对峰值的百分比
        if peak > 0:
            max_dd = max_dd / (100 + peak) * 100
    elif equity_curve and len(equity_curve) > 1:
        equity_series = pd.Series([e["equity"] for e in equity_curve])
        rolling_max = equity_series.cummax()
        drawdown = (equity_series - rolling_max) / rolling_max * 100
        max_dd = drawdown.min()
    else:
        max_dd = 0

    # 胜率 & 盈亏比
    if trades:
        win_trades = [t for t in trades if t.profit_pct > 0]
        lose_trades = [t for t in trades if t.profit_pct <= 0]
        win_rate = len(win_trades) / len(trades) * 100

        avg_win = np.mean([t.profit_pct for t in win_trades]) if win_trades else 0
        avg_loss = abs(np.mean([t.profit_pct for t in lose_trades])) if lose_trades else 0
        profit_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0

        # 期望值 E = P_win × R_win - P_loss × R_loss
        p_win = win_rate / 100
        p_loss = 1 - p_win
        expected_value = p_win * avg_win - p_loss * avg_loss

        avg_profit = np.mean([t.profit_pct for t in trades])
        avg_hold = np.mean([t.hold_days for t in trades])
        max_profit = max(t.profit_pct for t in trades)
        max_loss = min(t.profit_pct for t in trades)
    else:
        win_trades = []
        win_rate = 0
        profit_loss_ratio = 0
        expected_value = 0
        avg_profit = 0
        avg_loss = 0
        avg_hold = 0
        max_profit = 0
        max_loss = 0

    # Sharpe ratio: use forward-filled daily returns for proper computation
    # even when equity_curve is sparse (signal mode)
    from analytics.common import forward_fill_daily
    if equity_curve and len(equity_curve) > 1:
        ff_df = forward_fill_daily(equity_curve)
        if len(ff_df) > 1:
            daily_ret = ff_df["equity"].pct_change().dropna()
            if len(daily_ret) > 1 and float(np.std(daily_ret)) > 0:
                sharpe = float(np.mean(daily_ret) / np.std(daily_ret) * np.sqrt(252))
            else:
                sharpe = 0.0
        else:
            sharpe = 0.0
    else:
        sharpe = 0.0

    annual_returns = _calc_period_returns(equity_curve, "Y") if equity_curve else []
    monthly_returns = _calc_period_returns(equity_curve, "M") if equity_curve else []

    return BacktestResult(
        config_name=config.name,
        k_type=config.k_type,
        backtest_mode=config.backtest_mode,
        start_date=str(all_dates[0])[:10] if all_dates else "",
        end_date=str(all_dates[-1])[:10] if all_dates else "",
        initial_capital=config.initial_capital,
        final_capital=round(final_capital, 2),
        total_return_pct=round(total_return, 2),
        annual_return_pct=round(annual_return, 2),
        max_drawdown_pct=round(max_dd, 2),
        sharpe_ratio=round(sharpe, 2),
        win_rate=round(win_rate, 2),
        profit_loss_ratio=round(profit_loss_ratio, 2),
        expected_value=round(expected_value, 2),
        total_trades=len(trades),
        win_trades=len(win_trades),
        lose_trades=len(trades) - len(win_trades),
        avg_profit_pct=round(avg_profit, 2),
        avg_loss_pct=round(avg_loss, 2),
        max_profit_pct=round(max_profit, 2),
        max_loss_pct=round(max_loss, 2),
        avg_hold_days=round(avg_hold, 1),
        trades=[_trade_to_dict(t) for t in trades],
        equity_curve=equity_curve,
        annual_returns=annual_returns,
        monthly_returns=monthly_returns,
    )


def _calc_period_returns(equity_curve: list[dict], freq: str) -> list[dict]:
    """计算期间收益率"""
    if not equity_curve:
        return []
    df = pd.DataFrame(equity_curve)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    if len(df) < 2:
        return []
    result = []
    freq_map = {"Y": "YE", "M": "ME"}
    grouped = df["equity"].resample(freq_map.get(freq, freq))
    for label, group in grouped:
        if len(group) < 1:
            continue
        start_val = group.iloc[0]
        end_val = group.iloc[-1]
        ret = (end_val - start_val) / start_val * 100 if start_val > 0 else 0
        label_str = str(label)[:7] if freq == "M" else str(label)[:4]
        result.append({"period": label_str, "return_pct": round(ret, 2)})
    return result


def _trade_to_dict(t: Trade) -> dict:
    return {
        "code": t.code,
        "buy_date": str(t.buy_date)[:10],
        "buy_price": t.buy_price,
        "sell_date": str(t.sell_date)[:10] if t.sell_date else "",
        "sell_price": round(t.sell_price, 2) if t.sell_price else 0,
        "sell_reason": t.sell_reason,
        "shares": t.shares,
        "profit_pct": t.profit_pct,
        "profit_amount": t.profit_amount,
        "hold_days": t.hold_days,
        "action": t.action,
        "trade_id": t.trade_id,
    }
