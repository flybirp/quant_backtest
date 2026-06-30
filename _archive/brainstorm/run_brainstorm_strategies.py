"""
两轮脑暴策略批量回测 v3
- 全市场5062只股票 × 2015至今
- 多进程并行加速
- 指标缓存机制
"""

import time
import json
from pathlib import Path

from backend.data_loader import list_all_codes, preload_indicator_cache
from backend.strategy_engine import StrategyConfig
from backend.backtest_engine import run_backtest
from backend.main import _config_from_dict

# ============================================================
# 策略定义（两轮脑暴 × 2套超参 = 18配置）
# ============================================================

STRATEGIES = {}

# --- V1: 吸筹确认+趋势启动 (SC→SOS) ---
STRATEGIES["V1-吸筹确认-保守"] = {
    "name": "V1-吸筹确认-保守", "k_type": "daily", "backtest_mode": "signal",
    "buy_groups": [{"conditions": [
        {"indicator": "volume_anomaly", "params": {"ratio": 1.5}},
        {"indicator": "low_position", "params": {"price_pct": 30, "ma60_dist": 5}},
        {"indicator": "zhixing_golden_cross_hold", "params": {"days": 3}},
        {"indicator": "not_distribution"},
    ]}],
    "sell_groups": [{"conditions": [{"indicator": "zhixing_dead_cross"}]}],
    "add_groups": [], "reduce_groups": [],
    "stop_loss_pct": 3, "take_profit_pct": 15, "max_hold_days": 20, "trailing_stop_pct": 5,
    "buy_price_type": "open", "sell_price_type": "avg",
    "buy_execution": "next_day", "sell_execution": "next_day",
}

STRATEGIES["V1-吸筹确认-激进"] = {
    "name": "V1-吸筹确认-激进", "k_type": "daily", "backtest_mode": "signal",
    "buy_groups": [{"conditions": [
        {"indicator": "volume_anomaly", "params": {"ratio": 1.3}},
        {"indicator": "low_position", "params": {"price_pct": 40, "ma60_dist": 10}},
        {"indicator": "zhixing_golden_cross_hold", "params": {"days": 2}},
        {"indicator": "not_distribution"},
    ]}],
    "sell_groups": [{"conditions": [{"indicator": "zhixing_dead_cross"}]}],
    "add_groups": [], "reduce_groups": [],
    "stop_loss_pct": 5, "take_profit_pct": 25, "max_hold_days": 30, "trailing_stop_pct": 8,
    "buy_price_type": "open", "sell_price_type": "avg",
    "buy_execution": "next_day", "sell_execution": "next_day",
}

# --- V2: 横盘缩量突破 (因果定律) ---
STRATEGIES["V2-缩量突破-保守"] = {
    "name": "V2-缩量突破-保守", "k_type": "daily", "backtest_mode": "signal",
    "buy_groups": [{"conditions": [
        {"indicator": "sideway_shrink", "params": {"days": 3, "amplitude": 3.0, "vol_ratio": 0.4}},
        {"indicator": "zhixing_golden_cross_hold", "params": {"days": 3}},
        {"indicator": "price_near_zhixing", "params": {"tolerance": 3}},
    ]}],
    "sell_groups": [{"conditions": [{"indicator": "zhixing_dead_cross"}]}],
    "add_groups": [{"conditions": [
        {"indicator": "zhixing_fast_above_slow"},
        {"indicator": "volume_double"},
    ]}],
    "reduce_groups": [],
    "stop_loss_pct": 5, "take_profit_pct": 15, "max_hold_days": 20, "trailing_stop_pct": 8,
    "buy_price_type": "open", "sell_price_type": "avg",
    "buy_execution": "next_day", "sell_execution": "next_day",
}

STRATEGIES["V2-缩量突破-激进"] = {
    "name": "V2-缩量突破-激进", "k_type": "daily", "backtest_mode": "signal",
    "buy_groups": [{"conditions": [
        {"indicator": "sideway_shrink", "params": {"days": 5, "amplitude": 5.0, "vol_ratio": 0.5}},
        {"indicator": "zhixing_golden_cross_hold", "params": {"days": 2}},
        {"indicator": "price_near_zhixing", "params": {"tolerance": 5}},
    ]}],
    "sell_groups": [{"conditions": [{"indicator": "zhixing_dead_cross"}]}],
    "add_groups": [{"conditions": [
        {"indicator": "zhixing_fast_above_slow"},
        {"indicator": "volume_double"},
    ]}],
    "reduce_groups": [],
    "stop_loss_pct": 3, "take_profit_pct": 20, "max_hold_days": 30, "trailing_stop_pct": 10,
    "buy_price_type": "open", "sell_price_type": "avg",
    "buy_execution": "next_day", "sell_execution": "next_day",
}

# --- V3: 量能前置+回调买点 (Spring→LPS) ---
STRATEGIES["V3-量能前置-保守"] = {
    "name": "V3-量能前置-保守", "k_type": "daily", "backtest_mode": "signal",
    "buy_groups": [{"conditions": [
        {"indicator": "volume_anomaly", "params": {"ratio": 1.5}},
        {"indicator": "low_position", "params": {"price_pct": 40, "ma60_dist": 5}},
        {"indicator": "price_near_zhixing", "params": {"tolerance": 3}},
        {"indicator": "volume_shrink", "params": {"ratio": 0.6}},
    ]}],
    "sell_groups": [{"conditions": [{"indicator": "zhixing_dead_cross"}]}],
    "add_groups": [], "reduce_groups": [],
    "stop_loss_pct": 5, "take_profit_pct": 15, "max_hold_days": 15, "trailing_stop_pct": 5,
    "buy_price_type": "open", "sell_price_type": "avg",
    "buy_execution": "next_day", "sell_execution": "next_day",
}

STRATEGIES["V3-量能前置-激进"] = {
    "name": "V3-量能前置-激进", "k_type": "daily", "backtest_mode": "signal",
    "buy_groups": [{"conditions": [
        {"indicator": "volume_anomaly", "params": {"ratio": 1.3}},
        {"indicator": "low_position", "params": {"price_pct": 50, "ma60_dist": 10}},
        {"indicator": "price_near_zhixing", "params": {"tolerance": 5}},
    ]}],
    "sell_groups": [{"conditions": [{"indicator": "zhixing_dead_cross"}]}],
    "add_groups": [], "reduce_groups": [],
    "stop_loss_pct": 3, "take_profit_pct": 20, "max_hold_days": 25, "trailing_stop_pct": 8,
    "buy_price_type": "open", "sell_price_type": "avg",
    "buy_execution": "next_day", "sell_execution": "next_day",
}

# --- V5: 量价背离+均线方向 (努力vs结果) ---
STRATEGIES["V5-量价背离-保守"] = {
    "name": "V5-量价背离-保守", "k_type": "daily", "backtest_mode": "signal",
    "buy_groups": [{"conditions": [
        {"indicator": "no_supply", "params": {"vol_shrink": 0.4, "max_drop": -0.5}},
        {"indicator": "zhixing_fast_above_slow"},
        {"indicator": "low_position", "params": {"price_pct": 40, "ma60_dist": 10}},
    ]}],
    "sell_groups": [{"conditions": [{"indicator": "zhixing_dead_cross"}]}],
    "add_groups": [], "reduce_groups": [],
    "stop_loss_pct": 3, "take_profit_pct": 15, "max_hold_days": 15, "trailing_stop_pct": 5,
    "buy_price_type": "open", "sell_price_type": "avg",
    "buy_execution": "next_day", "sell_execution": "next_day",
}

STRATEGIES["V5-量价背离-激进"] = {
    "name": "V5-量价背离-激进", "k_type": "daily", "backtest_mode": "signal",
    "buy_groups": [{"conditions": [
        {"indicator": "effort_result_diverge", "params": {"type": "bull_div", "days": 5, "vol_ratio": 1.3, "vol_shrink": 0.7}},
        {"indicator": "zhixing_fast_above_slow"},
        {"indicator": "low_position", "params": {"price_pct": 50, "ma60_dist": 15}},
    ]}],
    "sell_groups": [{"conditions": [{"indicator": "zhixing_dead_cross"}]}],
    "add_groups": [], "reduce_groups": [],
    "stop_loss_pct": 5, "take_profit_pct": 25, "max_hold_days": 25, "trailing_stop_pct": 8,
    "buy_price_type": "open", "sell_price_type": "avg",
    "buy_execution": "next_day", "sell_execution": "next_day",
}

# --- V6: 威科夫弹簧+知行二次确认 ---
STRATEGIES["V6-弹簧确认-保守"] = {
    "name": "V6-弹簧确认-保守", "k_type": "daily", "backtest_mode": "signal",
    "buy_groups": [{"conditions": [
        {"indicator": "spring_shakeout", "params": {"support_period": 20, "spring_type": 2, "vol_confirm": True}},
        {"indicator": "zhixing_golden_cross_hold", "params": {"days": 3}},
    ]}],
    "sell_groups": [{"conditions": [{"indicator": "zhixing_dead_cross"}]}],
    "add_groups": [{"conditions": [{"indicator": "price_near_zhixing", "params": {"tolerance": 3}}]}],
    "reduce_groups": [],
    "stop_loss_pct": 3, "take_profit_pct": 20, "max_hold_days": 15, "trailing_stop_pct": 8,
    "buy_price_type": "open", "sell_price_type": "avg",
    "buy_execution": "next_day", "sell_execution": "next_day",
}

STRATEGIES["V6-弹簧确认-激进"] = {
    "name": "V6-弹簧确认-激进", "k_type": "daily", "backtest_mode": "signal",
    "buy_groups": [{"conditions": [
        {"indicator": "spring_shakeout", "params": {"support_period": 30, "spring_type": 0, "vol_confirm": False}},
        {"indicator": "zhixing_fast_above_slow"},
    ]}],
    "sell_groups": [{"conditions": [{"indicator": "zhixing_dead_cross"}]}],
    "add_groups": [{"conditions": [{"indicator": "price_near_zhixing", "params": {"tolerance": 5}}]}],
    "reduce_groups": [],
    "stop_loss_pct": 5, "take_profit_pct": 30, "max_hold_days": 25, "trailing_stop_pct": 10,
    "buy_price_type": "open", "sell_price_type": "avg",
    "buy_execution": "next_day", "sell_execution": "next_day",
}

# --- A: 口袋支点精确版 ---
STRATEGIES["A-口袋支点-保守"] = {
    "name": "A-口袋支点-保守", "k_type": "daily", "backtest_mode": "signal",
    "buy_groups": [{"conditions": [
        {"indicator": "pocket_pivot", "params": {"lookback": 10}},
        {"indicator": "price_near_zhixing", "params": {"tolerance": 3}},
        {"indicator": "zhixing_fast_above_slow"},
        {"indicator": "low_position", "params": {"price_pct": 50, "ma60_dist": 10}},
        {"indicator": "not_distribution"},
    ]}],
    "sell_groups": [{"conditions": [{"indicator": "zhixing_dead_cross"}]}],
    "add_groups": [{"conditions": [
        {"indicator": "zhixing_golden_cross_hold", "params": {"days": 3}},
        {"indicator": "volume_double"},
    ]}],
    "reduce_groups": [],
    "stop_loss_pct": 3, "take_profit_pct": 15, "max_hold_days": 15, "trailing_stop_pct": 5,
    "buy_price_type": "open", "sell_price_type": "avg",
    "buy_execution": "next_day", "sell_execution": "next_day",
}

STRATEGIES["A-口袋支点-激进"] = {
    "name": "A-口袋支点-激进", "k_type": "daily", "backtest_mode": "signal",
    "buy_groups": [{"conditions": [
        {"indicator": "pocket_pivot", "params": {"lookback": 10}},
        {"indicator": "zhixing_fast_above_slow"},
        {"indicator": "low_position", "params": {"price_pct": 60, "ma60_dist": 15}},
        {"indicator": "not_distribution"},
    ]}],
    "sell_groups": [{"conditions": [{"indicator": "zhixing_dead_cross"}]}],
    "add_groups": [{"conditions": [
        {"indicator": "zhixing_golden_cross_hold", "params": {"days": 2}},
        {"indicator": "volume_double"},
    ]}],
    "reduce_groups": [],
    "stop_loss_pct": 5, "take_profit_pct": 25, "max_hold_days": 25, "trailing_stop_pct": 8,
    "buy_price_type": "open", "sell_price_type": "avg",
    "buy_execution": "next_day", "sell_execution": "next_day",
}

# --- B: 弹簧→口袋支点确认 ---
STRATEGIES["B-弹簧口袋-保守"] = {
    "name": "B-弹簧口袋-保守", "k_type": "daily", "backtest_mode": "signal",
    "buy_groups": [{"conditions": [
        {"indicator": "spring_shakeout", "params": {"support_period": 20, "spring_type": 2, "vol_confirm": False}},
        {"indicator": "pocket_pivot", "params": {"lookback": 10}},
        {"indicator": "zhixing_fast_above_slow"},
    ]}],
    "sell_groups": [{"conditions": [{"indicator": "zhixing_dead_cross"}]}],
    "add_groups": [{"conditions": [{"indicator": "price_near_zhixing", "params": {"tolerance": 3}}]}],
    "reduce_groups": [],
    "stop_loss_pct": 3, "take_profit_pct": 20, "max_hold_days": 15, "trailing_stop_pct": 8,
    "buy_price_type": "open", "sell_price_type": "avg",
    "buy_execution": "next_day", "sell_execution": "next_day",
}

STRATEGIES["B-弹簧口袋-激进"] = {
    "name": "B-弹簧口袋-激进", "k_type": "daily", "backtest_mode": "signal",
    "buy_groups": [{"conditions": [
        {"indicator": "spring_shakeout", "params": {"support_period": 30, "spring_type": 0, "vol_confirm": False}},
        {"indicator": "pocket_pivot", "params": {"lookback": 10}},
        {"indicator": "zhixing_golden_cross_hold", "params": {"days": 2}},
    ]}],
    "sell_groups": [{"conditions": [{"indicator": "zhixing_dead_cross"}]}],
    "add_groups": [{"conditions": [{"indicator": "price_near_zhixing", "params": {"tolerance": 5}}]}],
    "reduce_groups": [],
    "stop_loss_pct": 5, "take_profit_pct": 30, "max_hold_days": 25, "trailing_stop_pct": 10,
    "buy_price_type": "open", "sell_price_type": "avg",
    "buy_execution": "next_day", "sell_execution": "next_day",
}

# --- C: 横盘缩量→口袋支点爆发 (因果定律) ---
STRATEGIES["C-因果口袋-保守"] = {
    "name": "C-因果口袋-保守", "k_type": "daily", "backtest_mode": "signal",
    "buy_groups": [{"conditions": [
        {"indicator": "sideway_shrink", "params": {"days": 3, "amplitude": 3.0, "vol_ratio": 0.4}},
        {"indicator": "pocket_pivot", "params": {"lookback": 10}},
        {"indicator": "zhixing_golden_cross_hold", "params": {"days": 3}},
    ]}],
    "sell_groups": [{"conditions": [{"indicator": "zhixing_dead_cross"}]}],
    "add_groups": [], "reduce_groups": [],
    "stop_loss_pct": 5, "take_profit_pct": 15, "max_hold_days": 20, "trailing_stop_pct": 5,
    "buy_price_type": "open", "sell_price_type": "avg",
    "buy_execution": "next_day", "sell_execution": "next_day",
}

STRATEGIES["C-因果口袋-激进"] = {
    "name": "C-因果口袋-激进", "k_type": "daily", "backtest_mode": "signal",
    "buy_groups": [{"conditions": [
        {"indicator": "sideway_shrink", "params": {"days": 5, "amplitude": 5.0, "vol_ratio": 0.5}},
        {"indicator": "pocket_pivot", "params": {"lookback": 10}},
        {"indicator": "zhixing_golden_cross", "params": {}},
    ]}],
    "sell_groups": [{"conditions": [{"indicator": "zhixing_dead_cross"}]}],
    "add_groups": [], "reduce_groups": [],
    "stop_loss_pct": 3, "take_profit_pct": 25, "max_hold_days": 30, "trailing_stop_pct": 10,
    "buy_price_type": "open", "sell_price_type": "avg",
    "buy_execution": "next_day", "sell_execution": "next_day",
}

# --- D: 口袋支点+均线延续买点 ---
STRATEGIES["D-均线口袋-保守"] = {
    "name": "D-均线口袋-保守", "k_type": "daily", "backtest_mode": "signal",
    "buy_groups": [{"conditions": [
        {"indicator": "pocket_pivot", "params": {"lookback": 10}},
        {"indicator": "zhixing_fast_above_slow"},
        {"indicator": "price_near_zhixing", "params": {"tolerance": 3}},
        {"indicator": "volume_shrink", "params": {"ratio": 0.6}},
    ]}],
    "sell_groups": [{"conditions": [{"indicator": "zhixing_fast_below_slow"}]}],
    "add_groups": [], "reduce_groups": [],
    "stop_loss_pct": 5, "take_profit_pct": 15, "max_hold_days": 0, "trailing_stop_pct": 8,
    "buy_price_type": "open", "sell_price_type": "avg",
    "buy_execution": "next_day", "sell_execution": "next_day",
}

STRATEGIES["D-均线口袋-激进"] = {
    "name": "D-均线口袋-激进", "k_type": "daily", "backtest_mode": "signal",
    "buy_groups": [{"conditions": [
        {"indicator": "pocket_pivot", "params": {"lookback": 10}},
        {"indicator": "zhixing_fast_above_slow"},
        {"indicator": "price_near_zhixing", "params": {"tolerance": 5}},
    ]}],
    "sell_groups": [{"conditions": [{"indicator": "zhixing_fast_below_slow"}]}],
    "add_groups": [], "reduce_groups": [],
    "stop_loss_pct": 3, "take_profit_pct": 20, "max_hold_days": 0, "trailing_stop_pct": 5,
    "buy_price_type": "open", "sell_price_type": "avg",
    "buy_execution": "next_day", "sell_execution": "next_day",
}


def main():
    import multiprocessing as mp
    n_workers = min(mp.cpu_count() - 1, 8)
    total = len(STRATEGIES)
    codes = list_all_codes()
    print(f"全市场股票数: {len(codes)}")
    print(f"策略数: {total}")
    print(f"并行进程: {n_workers}")

    # Step 1: 预热指标缓存
    print(f"\n{'='*80}")
    print("Step 1: 预热指标缓存")
    print(f"{'='*80}")
    t0 = time.time()
    preload_indicator_cache(codes, "daily")
    cache_time = time.time() - t0
    print(f"缓存预热耗时: {cache_time:.1f}s")

    # Step 2: 逐策略运行回测（每策略内部用缓存加速）
    results = []
    print(f"\n{'='*130}")
    print(f"Step 2: 批量回测 {total} 个策略 | 全市场{len(codes)}只A股 | 2015-01至今 | 信号模式")
    print(f"{'='*130}")
    print(f"{'策略':<22} {'交易数':>6} {'胜率':>8} {'盈亏比':>8} {'期望%':>8} {'均盈%':>7} {'均亏%':>7} {'均持仓':>7} {'耗时':>6}")
    print("-" * 130)

    overall_t0 = time.time()

    for i, (name, cfg_dict) in enumerate(STRATEGIES.items()):
        config = _config_from_dict(cfg_dict)
        # 全市场: 不设stock_pool
        config.stock_pool = []
        t0 = time.time()
        try:
            result = run_backtest(config, start_date="2015-01-01")
        except Exception as e:
            print(f"{name:<22} ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue
        dt = time.time() - t0

        r = {
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
        results.append(r)
        elapsed_total = time.time() - overall_t0
        print(f"{name:<22} {r['total_trades']:>6} {r['win_rate']:>7.1f}% "
              f"{r['profit_loss_ratio']:>7.2f} {r['expected_value']:>7.2f}% "
              f"{r['avg_profit_pct']:>6.2f}% {r['avg_loss_pct']:>6.2f}% "
              f"{r['avg_hold_days']:>6.1f}d [{dt:.0f}s] (总{elapsed_total:.0f}s)")

    # 汇总排序
    print(f"\n{'='*130}")
    print(f"📊 汇总排序 (按期望值降序) | 全市场{len(codes)}只 | 2015至今")
    print(f"{'='*130}")
    results.sort(key=lambda x: x["expected_value"], reverse=True)
    print(f"{'#':>3} {'策略':<22} {'交易数':>6} {'胜率':>8} {'盈亏比':>8} {'期望%':>8} {'均盈%':>7} {'均亏%':>7} {'判读':>10}")
    print("-" * 100)
    for rank, r in enumerate(results, 1):
        ev = r["expected_value"]
        if r["total_trades"] == 0:
            verdict = "无交易"
        elif ev > 1.0:
            verdict = "✅ 优秀"
        elif ev > 0:
            verdict = "✅ 正期望"
        elif ev > -1:
            verdict = "⚠️ 微负"
        else:
            verdict = "❌ 负期望"
        print(f"{rank:>3} {r['name']:<22} {r['total_trades']:>6} {r['win_rate']:>7.1f}% "
              f"{r['profit_loss_ratio']:>7.2f} {r['expected_value']:>7.2f}% "
              f"{r['avg_profit_pct']:>6.2f}% {r['avg_loss_pct']:>6.2f}% {verdict:>10}")

    total_time = time.time() - overall_t0
    print(f"\n总耗时: {total_time:.0f}s = {total_time/60:.1f}min")

    # 保存结果
    with open("brainstorm_results_fullmarket.json", "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"结果已保存到 brainstorm_results_fullmarket.json")


if __name__ == "__main__":
    main()
