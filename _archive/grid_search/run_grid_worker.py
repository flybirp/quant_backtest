"""量能爆发底部反转策略 - 网格搜索
用法: python3 run_volume_reversal_grid.py [dt] [vr] [sd] [dbl]
单次运行一个参数组合，输出JSON到stdout
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backend.strategy_engine import StrategyConfig
from backend.backtest_engine import run_backtest
from collections import defaultdict


def compute_position_ev(result):
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
        return 0, 0, 0, 0, 0
    wins = [r for r in pos_returns if r > 0]
    losses = [r for r in pos_returns if r <= 0]
    wr = len(wins) / n * 100
    aw = sum(wins) / len(wins) if wins else 0
    al = abs(sum(losses) / len(losses)) if losses else 0
    ev = wr / 100 * aw - (1 - wr / 100) * al
    plr = aw / al if al > 0 else 999
    return ev, wr, plr, n, result.total_trades


if __name__ == '__main__':
    # 默认参数
    dt = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    vr = int(sys.argv[2]) if len(sys.argv) > 2 else 90
    sd = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    dbl = int(sys.argv[4]) if len(sys.argv) > 4 else 30

    with open('stock_pools.json') as f:
        pools = json.load(f)
    codes = pools['大蓝筹'].get('codes', [])

    config = StrategyConfig(
        name='grid', k_type='daily', backtest_mode='signal',
        buy_groups=[{'name': 'b', 'conditions': [
            {'indicator': 'zhixing_long_downtrend', 'params': {'days': dt}},
            {'indicator': 'recent_volume_explosion', 'params': {'lookback': 15, 'vol_rank_threshold': vr, 'vol_ratio_threshold': 3.0}},
            {'indicator': 'volume_sustained', 'params': {'sustain_days': sd, 'sustain_ratio': 0.7, 'explosion_lookback': 15, 'vol_rank_threshold': vr, 'vol_ratio_threshold': 3.0}},
            {'indicator': 'price_cross_above_zhixing_fast'},
        ]}],
        sell_groups=[
            {'name': 's1', 'conditions': [{'indicator': 'double_bottom_broken', 'params': {'lookback': dbl, 'tolerance': 3.0, 'break_pct': 1.0}}]},
            {'name': 's2', 'conditions': [{'indicator': 'price_below_zhixing_slow_consecutive', 'params': {'days': 2, 'recent_above_days': 20}}]},
        ],
        add_groups=[{'name': 'a1', 'conditions': [
            {'indicator': 'price_below_zhixing_fast'},
            {'indicator': 'double_bottom', 'params': {'lookback': dbl, 'tolerance': 3.0}},
        ]}],
        reduce_groups=[{'name': 'r1', 'conditions': [{'indicator': 'price_below_bbi_consecutive', 'params': {'days': 2}}]}],
        buy_price_type='open', sell_price_type='avg', buy_execution='next_day', sell_execution='next_day',
        stop_loss_pct=8, take_profit_pct=0, trailing_stop_pct=0, max_hold_days=60,
        stock_pool=codes, entry_ladder=[{'trigger_pct': 0, 'weight': 50}],
    )
    result = run_backtest(config, start_date='20240101', end_date='20260606')
    ev, wr, plr, npos, ntrades = compute_position_ev(result)
    out = {
        'params': {'downtrend_days': dt, 'vol_rank_threshold': vr, 'sustain_days': sd, 'double_bottom_lookback': dbl,
                   'vol_ratio_threshold': 3.0, 'sustain_ratio': 0.7, 'double_bottom_tolerance': 3.0, 'break_pct': 1.0, 'recent_above_days': 20},
        'position_ev': round(ev, 2), 'position_win_rate': round(wr, 1),
        'position_pl_ratio': round(plr, 2), 'position_count': npos, 'position_trades': ntrades,
    }
    print(json.dumps(out))
