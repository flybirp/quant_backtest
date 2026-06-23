"""多进程并行回测引擎 — 全市场快速回测"""

from __future__ import annotations
import time
import multiprocessing as mp
from pathlib import Path
from typing import Optional
import pandas as pd

from backend.data_loader import load_stock_with_indicators, list_all_codes, preload_indicator_cache
from backend.strategy_engine import StrategyConfig, Signal, check_group, resolve_price
from backend.backtest_engine import BacktestResult, _run_signal_mode, _compute_statistics, _trade_to_dict, Trade


def _process_single_stock(args: tuple) -> list[dict]:
    """单股票信号检测（用于多进程Worker）

    返回: signals列表 (dict格式，可序列化)
    """
    code, cfg_dict, k_type, start_date, end_date = args

    from backend.main import _config_from_dict

    try:
        df = load_stock_with_indicators(code, k_type)
    except FileNotFoundError:
        return []
    if len(df) < 60:
        return []

    # 过滤日期
    if start_date:
        df = df[df["date"] >= start_date]
    if end_date:
        df = df[df["date"] <= end_date]
    if len(df) < 60:
        return []

    config = _config_from_dict(cfg_dict)

    # 生成信号
    signals = []
    for idx in range(len(df)):
        date = df["date"].iloc[idx]
        price = df["close"].iloc[idx]

        for g in config.buy_groups:
            ok, reason = check_group(df, idx, g)
            if ok:
                signals.append({
                    "date": str(date)[:10], "code": code,
                    "signal_type": "buy", "price": float(price),
                    "reason": reason,
                })
                break

        for g in config.sell_groups:
            ok, reason = check_group(df, idx, g)
            if ok:
                signals.append({
                    "date": str(date)[:10], "code": code,
                    "signal_type": "sell", "price": float(price),
                    "reason": reason,
                })
                break

        for g in config.add_groups:
            ok, reason = check_group(df, idx, g)
            if ok:
                signals.append({
                    "date": str(date)[:10], "code": code,
                    "signal_type": "add", "price": float(price),
                    "reason": reason,
                })
                break

        for g in config.reduce_groups:
            ok, reason = check_group(df, idx, g)
            if ok:
                signals.append({
                    "date": str(date)[:10], "code": code,
                    "signal_type": "reduce", "price": float(price),
                    "reason": reason,
                })
                break

    return signals


def run_backtest_parallel(config: StrategyConfig,
                          start_date: Optional[str] = None,
                          end_date: Optional[str] = None,
                          n_workers: int = 0,
                          batch_size: int = 50,
                          progress_callback=None) -> BacktestResult:
    """多进程并行回测

    Phase 1: 预热指标缓存（单进程顺序，写缓存）
    Phase 2: 并行信号检测（多进程，读缓存）
    Phase 3: 合并信号 + 执行回测逻辑（单进程）
    """
    from backend.main import _config_from_dict

    codes = list_all_codes() if not config.stock_pool else config.stock_pool
    if n_workers == 0:
        n_workers = min(mp.cpu_count() - 1, 8)

    # === Phase 0: 预热指标缓存 ===
    print(f"[Phase 0] 预热指标缓存 ({len(codes)}只股票)...")
    t0 = time.time()
    preload_indicator_cache(codes, config.k_type)
    print(f"[Phase 0] 缓存就绪, {time.time()-t0:.1f}s")

    # === Phase 1: 构造任务参数 ===
    cfg_dict = _config_to_dict(config)
    tasks = [(code, cfg_dict, config.k_type, start_date, end_date) for code in codes]

    # === Phase 2: 多进程信号检测 ===
    print(f"[Phase 1] 并行信号检测 ({n_workers}进程, {len(codes)}只股票)...")
    t0 = time.time()

    all_signal_dicts: list[dict] = []
    processed = 0

    with mp.Pool(processes=n_workers) as pool:
        # 分批提交，避免内存爆炸
        for i in range(0, len(tasks), batch_size * n_workers):
            batch = tasks[i:i + batch_size * n_workers]
            results = pool.map(_process_single_stock, batch, chunksize=batch_size)
            for sigs in results:
                all_signal_dicts.extend(sigs)
            processed += len(batch)
            if progress_callback:
                progress_callback(processed, len(codes))
            if processed % 500 == 0 or processed == len(codes):
                elapsed = time.time() - t0
                print(f"  [{processed}/{len(codes)}] {elapsed:.1f}s, {len(all_signal_dicts)}信号")

    print(f"[Phase 1] 完成, {time.time()-t0:.1f}s, 共{len(all_signal_dicts)}个信号")

    # === Phase 2.5: 重建Signal对象 + 加载所需股票数据 ===
    print("[Phase 2] 加载有信号的股票数据...")
    t0 = time.time()

    # 收集涉及的股票代码
    signal_codes = set(s["code"] for s in all_signal_dicts)
    print(f"  涉及{len(signal_codes)}只股票有信号")

    all_dfs: dict[str, pd.DataFrame] = {}
    all_signals: dict[str, list[Signal]] = {}

    for code in signal_codes:
        try:
            df = load_stock_with_indicators(code, config.k_type)
        except FileNotFoundError:
            continue
        if start_date:
            df = df[df["date"] >= start_date]
        if end_date:
            df = df[df["date"] <= end_date]
        if len(df) < 60:
            continue
        all_dfs[code] = df
        all_signals[code] = []

    # 重建Signal对象
    for s in all_signal_dicts:
        code = s["code"]
        if code not in all_dfs:
            continue
        all_signals[code].append(Signal(
            date=s["date"], code=code,
            signal_type=s["signal_type"], price=s["price"],
            reason=s["reason"],
        ))

    print(f"[Phase 2] 完成, {time.time()-t0:.1f}s, {len(all_dfs)}只股票有数据")

    # === Phase 3: 执行回测逻辑 ===
    print("[Phase 3] 执行回测...")
    t0 = time.time()

    if config.backtest_mode == "signal":
        result = _run_signal_mode(config, all_signals, all_dfs)
    else:
        from backend.backtest_engine import _run_portfolio_mode
        result = _run_portfolio_mode(config, all_signals, all_dfs)

    print(f"[Phase 3] 完成, {time.time()-t0:.1f}s")
    return result


def _config_to_dict(config: StrategyConfig) -> dict:
    """StrategyConfig转dict（用于多进程序列化）"""
    from dataclasses import asdict
    return asdict(config)
