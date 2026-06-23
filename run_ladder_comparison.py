#!/usr/bin/env python3
"""分批建仓/止盈 vs 原策略 对比回测"""

import json
import time
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import yaml
from backend.backtest_engine import run_backtest
from backend.strategy_engine import StrategyConfig
from backend.data_loader import preload_indicator_cache, load_stock_with_indicators, list_all_codes
import json


def load_stock_pool(name: str) -> list[str]:
    """加载标的池"""
    with open("stock_pools.json") as f:
        pools = json.load(f)
    pool = pools[name]
    if "codes" in pool and pool["codes"]:
        return pool["codes"]
    prefixes = pool.get("prefixes", [])
    all_codes = list_all_codes()
    if not prefixes:
        return all_codes
    return [c for c in all_codes if any(c.startswith(p) for p in prefixes)]


def load_strategy(yaml_path: str) -> dict:
    """加载策略YAML"""
    with open(yaml_path) as f:
        return yaml.safe_load(f)


def run_single_backtest(config: StrategyConfig, pool_name: str, pool_codes: list[str],
                        start_date: str, end_date: str) -> dict:
    """运行单次回测"""
    config.stock_pool = pool_codes
    result = run_backtest(config, start_date=start_date, end_date=end_date)
    return {
        "strategy": config.name,
        "pool": pool_name,
        "pool_size": len(pool_codes),
        "total_trades": result.total_trades,
        "win_rate": round(result.win_rate, 1),
        "profit_loss_ratio": round(result.profit_loss_ratio, 2),
        "expected_value": round(result.expected_value, 2),
        "avg_profit_pct": round(result.avg_profit_pct, 2),
        "avg_loss_pct": round(result.avg_loss_pct, 2),
        "max_profit_pct": round(result.max_profit_pct, 2),
        "max_loss_pct": round(result.max_loss_pct, 2),
        "avg_hold_days": round(result.avg_hold_days, 1),
        "win_trades": result.win_trades,
        "lose_trades": result.lose_trades,
    }


def config_from_yaml(yaml_path: str) -> StrategyConfig:
    """从YAML文件构建StrategyConfig"""
    d = load_strategy(yaml_path)
    return StrategyConfig(
        name=d.get("name", "default"),
        k_type=d.get("k_type", "daily"),
        backtest_mode=d.get("backtest_mode", "signal"),
        initial_capital=d.get("initial_capital", 100000),
        buy_groups=d.get("buy_groups", []),
        sell_groups=d.get("sell_groups", []),
        add_groups=d.get("add_groups", []),
        reduce_groups=d.get("reduce_groups", []),
        position_pct=d.get("position_pct", 1.0),
        max_positions=d.get("max_positions", 5),
        add_threshold=d.get("add_threshold", 0.0),
        add_pct=d.get("add_pct", 0.0),
        reduce_pct=d.get("reduce_pct", 0.5),
        stop_loss_pct=d.get("stop_loss_pct", 5.0),
        take_profit_pct=d.get("take_profit_pct", 15.0),
        max_hold_days=d.get("max_hold_days", 0),
        trailing_stop_pct=d.get("trailing_stop_pct", 0.0),
        min_volume_ratio=d.get("min_volume_ratio", 0.0),
        stock_pool=d.get("stock_pool", []),
        buy_price_type=d.get("buy_price_type", "close"),
        sell_price_type=d.get("sell_price_type", "close"),
        buy_execution=d.get("buy_execution", "same_day"),
        sell_execution=d.get("sell_execution", "same_day"),
        commission_rate=d.get("commission_rate", 0.0003),
        stamp_tax_rate=d.get("stamp_tax_rate", 0.001),
        entry_ladder=d.get("entry_ladder", []),
        exit_ladder=d.get("exit_ladder", []),
    )


if __name__ == "__main__":
    start_date = "20240101"
    end_date = "20260606"

    # 对比组合: 原策略 vs 分批策略, 在大蓝筹池上
    comparisons = [
        ("V1-吸筹确认-激进", "V1-吸筹确认-激进-分批"),
        ("A-口袋支点-激进", "A-口袋支点-激进-分批"),
        ("V6-弹簧确认-激进", "V6-弹簧确认-激进-分批"),
    ]

    print("=" * 70)
    print("  分批建仓/止盈 vs 原策略 对比回测（大蓝筹池）")
    print("=" * 70)

    # 预热缓存
    print("\n[1/3] 预热指标缓存...")
    all_codes = list_all_codes()
    preload_indicator_cache(all_codes, "daily")
    print("  缓存预热完成")

    # 加载大蓝筹池
    print("\n[2/3] 加载标的池...")
    dl_codes = load_stock_pool("大蓝筹")
    print(f"  大蓝筹: {len(dl_codes)}只")

    # 回测
    print(f"\n[3/3] 开始回测 ({len(comparisons)*2}组)...\n")
    results = []

    for orig_name, ladder_name in comparisons:
        for strategy_name in [orig_name, ladder_name]:
            yaml_path = f"strategies/{strategy_name}.yaml"
            if not os.path.exists(yaml_path):
                print(f"  [跳过] {yaml_path} 不存在")
                continue

            config = config_from_yaml(yaml_path)
            t0 = time.time()
            result = run_single_backtest(config, "大蓝筹", dl_codes, start_date, end_date)
            elapsed = time.time() - t0
            results.append(result)

            tag = "分批" if "分批" in strategy_name else "原版"
            ev_str = f"EV={result['expected_value']:+.2f}%" if result['expected_value'] != 0 else "EV=0.0%"
            print(f"  {strategy_name:25s} @ 大蓝筹  "
                  f"{ev_str}  胜率={result['win_rate']}%  "
                  f"盈亏比={result['profit_loss_ratio']}  "
                  f"交易={result['total_trades']}  ({elapsed:.1f}s)")

    # 保存结果
    output_path = "ladder_comparison_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 对比分析
    print("\n" + "=" * 70)
    print("  对比分析")
    print("=" * 70)
    for orig_name, ladder_name in comparisons:
        orig = next((r for r in results if r["strategy"] == orig_name), None)
        ladder = next((r for r in results if r["strategy"] == ladder_name), None)
        if not orig or not ladder:
            continue

        ev_diff = ladder["expected_value"] - orig["expected_value"]
        wr_diff = ladder["win_rate"] - orig["win_rate"]
        pl_diff = ladder["profit_loss_ratio"] - orig["profit_loss_ratio"]

        print(f"\n  {orig_name}:")
        print(f"    原版: EV={orig['expected_value']:+.2f}%  胜率={orig['win_rate']}%  "
              f"盈亏比={orig['profit_loss_ratio']}  交易={orig['total_trades']}")
        print(f"    分批: EV={ladder['expected_value']:+.2f}%  胜率={ladder['win_rate']}%  "
              f"盈亏比={ladder['profit_loss_ratio']}  交易={ladder['total_trades']}")
        print(f"    差异: EV{'↑' if ev_diff > 0 else '↓'}{abs(ev_diff):.2f}%  "
              f"胜率{'↑' if wr_diff > 0 else '↓'}{abs(wr_diff):.1f}%  "
              f"盈亏比{'↑' if pl_diff > 0 else '↓'}{abs(pl_diff):.2f}")

    print(f"\n结果已保存至: {output_path}")
