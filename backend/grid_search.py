"""网格搜索引擎 — 对止盈止损等参数做全参数网格扫描"""

from __future__ import annotations
import itertools
import time
from dataclasses import replace
from backend.strategy_engine import StrategyConfig
from backend.backtest_engine import run_backtest


def grid_search(
    base_config: StrategyConfig,
    param_grid: dict[str, list],
    start_date: str | None = None,
    end_date: str | None = None,
    metric: str = "profit_loss_ratio",  # 排序指标: win_rate / profit_loss_ratio / expected_value / avg_profit_pct
    top_k: int = 10,
) -> list[dict]:
    """
    对 base_config 的指定参数做全网格搜索。

    param_grid 示例:
    {
        "stop_loss_pct": [3.0, 5.0, 7.0, 10.0],
        "take_profit_pct": [10.0, 15.0, 20.0, 25.0, 30.0],
        "trailing_stop_pct": [0.0, 5.0, 8.0, 10.0, 15.0],
    }

    返回按 metric 降序排列的 top_k 结果，每个元素:
    {
        "params": {参数名: 值},
        "total_trades": int,
        "win_rate": float,
        "profit_loss_ratio": float,
        "avg_profit_pct": float,
        "avg_loss_pct": float,
    }
    """
    # 生成所有参数组合
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combos = list(itertools.product(*values))

    print(f"网格搜索: {len(combos)} 种参数组合, 排序指标={metric}, 取top_{top_k}")

    results = []
    t0 = time.time()

    for i, combo in enumerate(combos):
        overrides = dict(zip(keys, combo))
        config = replace(base_config, **overrides)
        config.name = f"{base_config.name}_" + "_".join(f"{k}={v}" for k, v in overrides.items())

        try:
            result = run_backtest(config, start_date=start_date, end_date=end_date)
        except Exception as e:
            print(f"  [{i+1}/{len(combos)}] {config.name} 失败: {e}")
            continue

        entry = {
            "params": overrides,
            "total_trades": result.total_trades,
            "win_rate": result.win_rate,
            "profit_loss_ratio": result.profit_loss_ratio,
            "expected_value": result.expected_value,
            "avg_profit_pct": result.avg_profit_pct,
            "avg_loss_pct": result.avg_loss_pct,
            "max_profit_pct": result.max_profit_pct,
            "max_loss_pct": result.max_loss_pct,
            "avg_hold_days": result.avg_hold_days,
        }
        results.append(entry)

        if (i + 1) % 10 == 0 or i + 1 == len(combos):
            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(combos)}] {elapsed:.1f}s "
                  f"胜率={result.win_rate:.1f}% 盈亏比={result.profit_loss_ratio:.2f} "
                  f"期望={result.expected_value:.2f}% params={overrides}")

    # 排序
    results.sort(key=lambda x: x.get(metric, 0), reverse=True)

    # 输出 top_k
    print(f"\n===== 网格搜索 Top {min(top_k, len(results))} (按 {metric} 排序) =====")
    print(f"{'排名':>4} {'参数':50s} {'交易数':>6} {'胜率':>8} {'盈亏比':>8} {'期望%':>8} {'均盈%':>8} {'均亏%':>8}")
    print("-" * 110)
    for rank, r in enumerate(results[:top_k], 1):
        param_str = ", ".join(f"{k}={v}" for k, v in r["params"].items())
        print(f"{rank:>4} {param_str:50s} {r['total_trades']:>6} "
              f"{r['win_rate']:>7.1f}% {r['profit_loss_ratio']:>7.2f} "
              f"{r['expected_value']:>7.2f}% "
              f"{r['avg_profit_pct']:>7.2f}% {r['avg_loss_pct']:>7.2f}%")

    return results


def batch_strategies(
    strategies: list[tuple[str, StrategyConfig]],
    start_date: str | None = None,
    end_date: str | None = None,
) -> list[dict]:
    """
    批量运行多个策略并对比结果。

    strategies: [(name, config), ...]
    返回按 profit_loss_ratio 降序排列的结果列表
    """
    results = []
    print(f"\n批量回测 {len(strategies)} 个策略")
    print(f"{'策略':<30} {'交易数':>6} {'胜率':>8} {'盈亏比':>8} {'期望%':>8} {'均盈%':>8} {'均亏%':>8} {'均持仓天':>8}")
    print("-" * 115)

    for name, config in strategies:
        t0 = time.time()
        config = replace(config, name=name)
        result = run_backtest(config, start_date=start_date, end_date=end_date)
        dt = time.time() - t0

        entry = {
            "name": name,
            "total_trades": result.total_trades,
            "win_rate": result.win_rate,
            "profit_loss_ratio": result.profit_loss_ratio,
            "expected_value": result.expected_value,
            "avg_profit_pct": result.avg_profit_pct,
            "avg_loss_pct": result.avg_loss_pct,
            "avg_hold_days": result.avg_hold_days,
            "max_profit_pct": result.max_profit_pct,
            "max_loss_pct": result.max_loss_pct,
        }
        results.append(entry)

        print(f"{name:<30} {result.total_trades:>6} {result.win_rate:>7.1f}% "
              f"{result.profit_loss_ratio:>7.2f} {result.expected_value:>7.2f}% "
              f"{result.avg_profit_pct:>7.2f}% "
              f"{result.avg_loss_pct:>7.2f}% {result.avg_hold_days:>7.1f}d  [{dt:.1f}s]")

        # 按卖出原因统计
        reasons = {}
        for tr in result.trades:
            r = tr["sell_reason"]
            reasons[r] = reasons.get(r, 0) + 1
        if reasons:
            reason_str = ", ".join(f"{k}:{v}" for k, v in sorted(reasons.items(), key=lambda x: -x[1]))
            print(f"     卖出原因: {reason_str}")

    results.sort(key=lambda x: x["profit_loss_ratio"], reverse=True)
    return results
