"""量能爆发底部反转策略 - 跨标的池横向对比
5个策略(TOP1-5) × 4个标的池(大蓝筹/科创板/创业板/全量) = 20组回测
"""
import sys, os, json, time
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backend.strategy_engine import StrategyConfig
from backend.backtest_engine import run_backtest
from backend.data_loader import list_all_codes

POOLS_FILE = Path(__file__).parent / 'stock_pools.json'
STRATEGIES_DIR = Path(__file__).parent / 'strategies'
START_DATE = '20240101'
END_DATE = '20260606'


def resolve_pool_codes(pool_name: str) -> list[str]:
    """根据池名解析股票代码列表"""
    with open(POOLS_FILE) as f:
        pools = json.load(f)
    pool = pools.get(pool_name, {})

    if 'codes' in pool and pool['codes']:
        all_codes = set(list_all_codes())
        return [c for c in pool['codes'] if c in all_codes]

    prefixes = pool.get('prefix', [])
    all_codes = list_all_codes()
    if not prefixes:
        return all_codes
    return [c for c in all_codes if any(c.startswith(p) for p in prefixes)]


def compute_position_ev(result):
    """计算持仓级别EV"""
    positions = defaultdict(list)
    for t in result.trades:
        positions[t['trade_id']].append(t)
    pos_returns = []
    for tid, trades in positions.items():
        sell_trades = [t for t in trades if t.get('sell_reason')]
        if not sell_trades:
            continue
        total_shares = sum(t['shares'] for t in sell_trades)
        if total_shares == 0:
            continue
        wr2 = sum(t['shares'] * t['profit_pct'] for t in sell_trades) / total_shares
        pos_returns.append(wr2)
    n = len(pos_returns)
    if n == 0:
        return {'ev': 0, 'win_rate': 0, 'pl_ratio': 0, 'positions': 0, 'trades': 0}
    wins = [r for r in pos_returns if r > 0]
    losses = [r for r in pos_returns if r <= 0]
    wr = len(wins) / n * 100
    aw = sum(wins) / len(wins) if wins else 0
    al = abs(sum(losses) / len(losses)) if losses else 0
    ev = wr / 100 * aw - (1 - wr / 100) * al
    plr = aw / al if al > 0 else 999
    return {'ev': round(ev, 2), 'win_rate': round(wr, 1), 'pl_ratio': round(plr, 2), 'positions': n, 'trades': result.total_trades}


# 5个策略的核心参数 (from TOP1-TOP5 YAML files)
STRATEGIES = {
    'TOP1': {'dt': 100, 'vr': 95, 'vratio': 3.0, 'sd': 3, 'sr': 0.7, 'dbl': 60, 'dbt': 3.0, 'bp': 3.0, 'ra': 10, 'el': 15,
             'note': 'EV=+6.07% dt=100 最优'},
    'TOP2': {'dt': 100, 'vr': 95, 'vratio': 3.0, 'sd': 3, 'sr': 0.7, 'dbl': 60, 'dbt': 3.0, 'bp': 3.0, 'ra': 15, 'el': 15,
             'note': 'EV=+6.06% ra=15 更保守退出'},
    'TOP3': {'dt': 100, 'vr': 95, 'vratio': 3.0, 'sd': 3, 'sr': 0.7, 'dbl': 60, 'dbt': 3.0, 'bp': 3.0, 'ra': 10, 'el': 20,
             'note': 'EV=+6.02% el=20 更多信号'},
    'TOP4': {'dt': 90,  'vr': 95, 'vratio': 3.0, 'sd': 3, 'sr': 0.7, 'dbl': 60, 'dbt': 2.0, 'bp': 3.0, 'ra': 15, 'el': 15,
             'note': 'EV=+5.82% dt=90 更宽松入场'},
    'TOP5': {'dt': 70,  'vr': 95, 'vratio': 3.0, 'sd': 3, 'sr': 0.7, 'dbl': 60, 'dbt': 2.0, 'bp': 3.0, 'ra': 15, 'el': 15,
             'note': 'EV=+4.06% dt=70 信号最多'},
}

POOL_NAMES = ['大蓝筹', '科创板', '创业板', '全量']


def build_config(sname: str, sp: dict, codes: list[str]) -> StrategyConfig:
    return StrategyConfig(
        name=f'{sname}',
        k_type='daily', backtest_mode='signal',
        buy_groups=[{'name': 'b', 'conditions': [
            {'indicator': 'zhixing_long_downtrend', 'params': {'days': sp['dt']}},
            {'indicator': 'recent_volume_explosion', 'params': {'lookback': sp['el'], 'vol_rank_threshold': sp['vr'], 'vol_ratio_threshold': sp['vratio']}},
            {'indicator': 'volume_sustained', 'params': {'sustain_days': sp['sd'], 'sustain_ratio': sp['sr'], 'explosion_lookback': sp['el'], 'vol_rank_threshold': sp['vr'], 'vol_ratio_threshold': sp['vratio']}},
            {'indicator': 'price_cross_above_zhixing_fast'},
        ]}],
        sell_groups=[
            {'name': 's1', 'conditions': [{'indicator': 'double_bottom_broken', 'params': {'lookback': sp['dbl'], 'tolerance': sp['dbt'], 'break_pct': sp['bp']}}]},
            {'name': 's2', 'conditions': [{'indicator': 'price_below_zhixing_slow_consecutive', 'params': {'days': 2, 'recent_above_days': sp['ra']}}]},
        ],
        add_groups=[{'name': 'a1', 'conditions': [
            {'indicator': 'price_below_zhixing_fast'},
            {'indicator': 'double_bottom', 'params': {'lookback': sp['dbl'], 'tolerance': sp['dbt']}},
        ]}],
        reduce_groups=[{'name': 'r1', 'conditions': [{'indicator': 'price_below_bbi_consecutive', 'params': {'days': 2}}]}],
        buy_price_type='open', sell_price_type='avg', buy_execution='next_day', sell_execution='next_day',
        stop_loss_pct=8, take_profit_pct=0, trailing_stop_pct=0, max_hold_days=60,
        stock_pool=codes, entry_ladder=[{'trigger_pct': 0, 'weight': 50}],
    )


def main():
    # 先解析所有标的池，报告大小
    pool_codes = {}
    print("=" * 60)
    print("量能爆发底部反转 - 跨标的池横向对比")
    print(f"回测区间: {START_DATE} ~ {END_DATE}")
    print("=" * 60)
    for pname in POOL_NAMES:
        codes = resolve_pool_codes(pname)
        pool_codes[pname] = codes
        print(f"  {pname}: {len(codes)} 只")

    results = {}
    total = len(STRATEGIES) * len(POOL_NAMES)
    done = 0
    t0 = time.time()

    for sname, sp in STRATEGIES.items():
        for pname in POOL_NAMES:
            done += 1
            codes = pool_codes[pname]
            print(f"\n[{done}/{total}] {sname} × {pname} ({len(codes)}只) ...", end=' ', flush=True)

            if len(codes) == 0:
                print("跳过(无标的)")
                results[(sname, pname)] = {'ev': None, 'win_rate': None, 'pl_ratio': None, 'positions': 0, 'trades': 0}
                continue

            config = build_config(sname, sp, codes)
            result = run_backtest(config, start_date=START_DATE, end_date=END_DATE)
            metrics = compute_position_ev(result)
            results[(sname, pname)] = metrics
            elapsed = time.time() - t0
            print(f"EV={metrics['ev']:+.2f}% 胜率={metrics['win_rate']:.1f}% 盈亏比={metrics['pl_ratio']:.2f} 持仓={metrics['positions']} ({elapsed:.0f}s)")

    # 保存原始结果
    output = []
    for (sname, pname), m in results.items():
        output.append({'strategy': sname, 'pool': pname, **m})
    with open('cross_pool_results.json', 'w') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    # 打印汇总表
    print("\n" + "=" * 90)
    print(f"{'':>6} | {'大蓝筹':>30} | {'科创板':>30} | {'创业板':>30} | {'全量':>30}")
    print("-" * 140)

    for metric_name, metric_key in [('EV(%)', 'ev'), ('胜率(%)', 'win_rate'), ('盈亏比', 'pl_ratio'), ('持仓数', 'positions')]:
        print(f"{metric_name:>6} |", end='')
        for pname in POOL_NAMES:
            pass
        # 按策略打印
        for sname in STRATEGIES:
            vals = []
            for pname in POOL_NAMES:
                v = results.get((sname, pname), {}).get(metric_key)
                if v is None:
                    vals.append('N/A')
                elif metric_key == 'ev':
                    vals.append(f'{v:+.2f}')
                elif metric_key == 'positions':
                    vals.append(f'{v}')
                else:
                    vals.append(f'{v:.1f}' if isinstance(v, float) else str(v))
            # Actually let's reformat this - print by metric×strategy
        break

    # Better format: strategy as rows, pools as columns
    print("\n\n====== EV(%) 汇总 ======")
    print(f"{'策略':<8}", end='')
    for pname in POOL_NAMES:
        print(f" | {pname:>10}", end='')
    print()
    print("-" * (8 + 14 * len(POOL_NAMES)))
    for sname in STRATEGIES:
        print(f"{sname:<8}", end='')
        for pname in POOL_NAMES:
            v = results.get((sname, pname), {}).get('ev')
            if v is None:
                print(f" | {'N/A':>10}", end='')
            else:
                print(f" | {v:>+10.2f}", end='')
        print()

    print("\n====== 胜率(%) 汇总 ======")
    print(f"{'策略':<8}", end='')
    for pname in POOL_NAMES:
        print(f" | {pname:>10}", end='')
    print()
    print("-" * (8 + 14 * len(POOL_NAMES)))
    for sname in STRATEGIES:
        print(f"{sname:<8}", end='')
        for pname in POOL_NAMES:
            v = results.get((sname, pname), {}).get('win_rate')
            if v is None:
                print(f" | {'N/A':>10}", end='')
            else:
                print(f" | {v:>10.1f}", end='')
        print()

    print("\n====== 盈亏比 汇总 ======")
    print(f"{'策略':<8}", end='')
    for pname in POOL_NAMES:
        print(f" | {pname:>10}", end='')
    print()
    print("-" * (8 + 14 * len(POOL_NAMES)))
    for sname in STRATEGIES:
        print(f"{sname:<8}", end='')
        for pname in POOL_NAMES:
            v = results.get((sname, pname), {}).get('pl_ratio')
            if v is None:
                print(f" | {'N/A':>10}", end='')
            else:
                print(f" | {v:>10.2f}", end='')
        print()

    print("\n====== 持仓数 汇总 ======")
    print(f"{'策略':<8}", end='')
    for pname in POOL_NAMES:
        print(f" | {pname:>10}", end='')
    print()
    print("-" * (8 + 14 * len(POOL_NAMES)))
    for sname in STRATEGIES:
        print(f"{sname:<8}", end='')
        for pname in POOL_NAMES:
            v = results.get((sname, pname), {}).get('positions')
            if v is None:
                print(f" | {'N/A':>10}", end='')
            else:
                print(f" | {v:>10}", end='')
        print()

    total_time = time.time() - t0
    print(f"\n总耗时: {total_time:.1f}s")
    print("结果已保存到 cross_pool_results.json")


if __name__ == '__main__':
    main()
