"""
9个正EV策略 × 2个标的池（大蓝筹 vs 创业板）对比回测
"""
import sys, time, json
sys.path.insert(0, '.')

from backend.backtest_engine import run_backtest
from backend.strategy_engine import StrategyConfig
from backend.data_loader import list_all_codes, preload_indicator_cache
import yaml
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# 9个正EV策略
STRATEGY_NAMES = [
    "V3-量能前置-激进",
    "V1-吸筹确认-激进",
    "A-口袋支点-激进",
    "V1-吸筹确认-保守",
    "D-均线口袋-激进",
    "A-口袋支点-保守",
    "V6-弹簧确认-激进",
    "B-弹簧口袋-激进",
    "V6-弹簧确认-保守",
]

# 标的池定义
def get_pool_codes(pool_name: str) -> list[str]:
    all_codes = list_all_codes()
    if pool_name == "大蓝筹":
        with open("stock_pools.json") as f:
            pools = json.load(f)
        valid = set(all_codes)
        return [c for c in pools["大蓝筹"]["codes"] if c in valid]
    elif pool_name == "创业板":
        return [c for c in all_codes if c.startswith(("300", "301", "302"))]
    elif pool_name == "全量":
        return all_codes
    return []


def load_strategy(name: str) -> StrategyConfig:
    fpath = Path("strategies") / f"{name}.yaml"
    with open(fpath) as f:
        raw = yaml.safe_load(f)
    return StrategyConfig(**raw), raw


def run_one(name: str, pool_name: str, codes: list[str]):
    """单个策略+池子回测"""
    config, raw = load_strategy(name)
    config.stock_pool = codes

    t0 = time.time()
    result = run_backtest(config, start_date="2014-01-01", end_date="2026-06-16")
    elapsed = time.time() - t0

    return {
        "strategy": name,
        "pool": pool_name,
        "pool_size": len(codes),
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
        "elapsed": round(elapsed, 1),
    }


def main():
    print("=" * 70)
    print("  9策略 × 2标的池 对比回测")
    print("=" * 70)

    # 预热缓存
    print("\n[1/3] 预热指标缓存...")
    t0 = time.time()
    all_codes = list_all_codes()
    preload_indicator_cache(all_codes)
    print(f"  缓存预热完成 ({time.time()-t0:.1f}s)")

    # 获取标的池
    print("\n[2/3] 准备标的池...")
    blue_chips = get_pool_codes("大蓝筹")
    chinext = get_pool_codes("创业板")
    print(f"  大蓝筹: {len(blue_chips)}只")
    print(f"  创业板: {len(chinext)}只")

    # 构建任务列表
    tasks = []
    for name in STRATEGY_NAMES:
        tasks.append((name, "大蓝筹", blue_chips))
        tasks.append((name, "创业板", chinext))

    # 运行回测
    print(f"\n[3/3] 开始回测 ({len(tasks)}组)...\n")
    results = []
    done = 0

    for name, pool_name, codes in tasks:
        done += 1
        print(f"  [{done:2d}/{len(tasks)}] {name} @ {pool_name} ({len(codes)}只)...", end=" ", flush=True)
        r = run_one(name, pool_name, codes)
        results.append(r)
        ev_str = f"+{r['expected_value']}%" if r['expected_value'] >= 0 else f"{r['expected_value']}%"
        print(f"EV={ev_str}  胜率={r['win_rate']}%  盈亏比={r['profit_loss_ratio']}  交易={r['total_trades']}  ({r['elapsed']}s)")

    # 保存结果
    output_path = "pool_comparison_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {output_path}")

    # 打印对比表
    print("\n" + "=" * 100)
    print("  策略 × 标的池 期望值(EV)对比")
    print("=" * 100)
    print(f"{'策略':<20s} | {'大蓝筹 EV':>10s} | {'大蓝筹 胜率':>10s} | {'大蓝筹 盈亏比':>10s} | {'大蓝筹 交易':>10s} | {'创业板 EV':>10s} | {'创业板 胜率':>10s} | {'创业板 盈亏比':>10s} | {'创业板 交易':>10s}")
    print("-" * 100)

    for name in STRATEGY_NAMES:
        r_b = next(r for r in results if r["strategy"] == name and r["pool"] == "大蓝筹")
        r_c = next(r for r in results if r["strategy"] == name and r["pool"] == "创业板")
        ev_b = f"+{r_b['expected_value']}%" if r_b['expected_value'] >= 0 else f"{r_b['expected_value']}%"
        ev_c = f"+{r_c['expected_value']}%" if r_c['expected_value'] >= 0 else f"{r_c['expected_value']}%"
        print(f"{name:<20s} | {ev_b:>10s} | {r_b['win_rate']:>9.1f}% | {r_b['profit_loss_ratio']:>10.2f} | {r_b['total_trades']:>10d} | {ev_c:>10s} | {r_c['win_rate']:>9.1f}% | {r_c['profit_loss_ratio']:>10.2f} | {r_c['total_trades']:>10d}")

    # 差异分析
    print("\n" + "=" * 60)
    print("  EV差异 (大蓝筹 - 创业板)")
    print("=" * 60)
    diffs = []
    for name in STRATEGY_NAMES:
        r_b = next(r for r in results if r["strategy"] == name and r["pool"] == "大蓝筹")
        r_c = next(r for r in results if r["strategy"] == name and r["pool"] == "创业板")
        diff = r_b["expected_value"] - r_c["expected_value"]
        diffs.append((name, diff, r_b["expected_value"], r_c["expected_value"]))

    diffs.sort(key=lambda x: x[1], reverse=True)
    for name, diff, ev_b, ev_c in diffs:
        arrow = "🔵大蓝筹优" if diff > 0 else "🔴创业板优" if diff < 0 else "⚖️持平"
        print(f"  {name:<20s}  差值={diff:+.2f}%  (蓝筹{ev_b:+.2f}% vs 创业板{ev_c:+.2f}%)  {arrow}")


if __name__ == "__main__":
    main()
