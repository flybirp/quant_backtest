"""量能爆发底部反转策略 - 深层网格搜索 Worker
用法: python3 run_grid_worker_deep.py [dt] [vr] [vratio] [sd] [sr] [dbl] [dbt] [bp*10] [ra] [el]
所有参数通过命令行传入，输出JSON到stdout
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
    dt = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    vr = int(sys.argv[2]) if len(sys.argv) > 2 else 95
    vratio = float(sys.argv[3]) if len(sys.argv) > 3 else 3.0
    sd = int(sys.argv[4]) if len(sys.argv) > 4 else 3
    sr = float(sys.argv[5]) if len(sys.argv) > 5 else 0.7
    dbl = int(sys.argv[6]) if len(sys.argv) > 6 else 40
    dbt = float(sys.argv[7]) if len(sys.argv) > 7 else 2.0
    bp = float(sys.argv[8]) / 10.0 if len(sys.argv) > 8 else 2.0   # 传入整数×10，避免bash浮点问题
    ra = int(sys.argv[9]) if len(sys.argv) > 9 else 15
    el = int(sys.argv[10]) if len(sys.argv) > 10 else 15

    with open('stock_pools.json') as f:
        pools = json.load(f)
    codes = pools['大蓝筹'].get('codes', [])

    config = StrategyConfig(
        name='deep', k_type='daily', backtest_mode='signal',
        buy_groups=[{'name': 'b', 'conditions': [
            {'indicator': 'zhixing_long_downtrend', 'params': {'days': dt}},
            {'indicator': 'recent_volume_explosion', 'params': {'lookback': el, 'vol_rank_threshold': vr, 'vol_ratio_threshold': vratio}},
            {'indicator': 'volume_sustained', 'params': {'sustain_days': sd, 'sustain_ratio': sr, 'explosion_lookback': el, 'vol_rank_threshold': vr, 'vol_ratio_threshold': vratio}},
            {'indicator': 'price_cross_above_zhixing_fast'},
        ]}],
        sell_groups=[
            {'name': 's1', 'conditions': [{'indicator': 'double_bottom_broken', 'params': {'lookback': dbl, 'tolerance': dbt, 'break_pct': bp}}]},
            {'name': 's2', 'conditions': [{'indicator': 'price_below_zhixing_slow_consecutive', 'params': {'days': 2, 'recent_above_days': ra}}]},
        ],
        add_groups=[{'name': 'a1', 'conditions': [
            {'indicator': 'price_below_zhixing_fast'},
            {'indicator': 'double_bottom', 'params': {'lookback': dbl, 'tolerance': dbt}},
        ]}],
        reduce_groups=[{'name': 'r1', 'conditions': [{'indicator': 'price_below_bbi_consecutive', 'params': {'days': 2}}]}],
        buy_price_type='open', sell_price_type='avg', buy_execution='next_day', sell_execution='next_day',
        stop_loss_pct=8, take_profit_pct=0, trailing_stop_pct=0, max_hold_days=60,
        stock_pool=codes, entry_ladder=[{'trigger_pct': 0, 'weight': 50}],
    )
    result = run_backtest(config, start_date='20240101', end_date='20260606')
    ev, wr, plr, npos, ntrades = compute_position_ev(result)
    out = {
        'params': {'downtrend_days': dt, 'vol_rank_threshold': vr, 'vol_ratio_threshold': vratio,
                   'sustain_days': sd, 'sustain_ratio': sr, 'double_bottom_lookback': dbl,
                   'double_bottom_tolerance': dbt, 'break_pct': bp, 'recent_above_days': ra,
                   'explosion_lookback': el},
        'position_ev': round(ev, 2), 'position_win_rate': round(wr, 1),
        'position_pl_ratio': round(plr, 2), 'position_count': npos, 'position_trades': ntrades,
    }
    print(json.dumps(out))
