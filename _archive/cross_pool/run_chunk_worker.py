"""Chunked worker for 全量 pool - splits by first digit of stock code
用法: python3 run_chunk_worker.py [sname] [dt] [vr] [vratio*10] [sd] [sr*10] [dbl] [dbt*10] [bp*10] [ra] [el] [chunk_prefix]
"""
import sys, os, json
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backend.strategy_engine import StrategyConfig
from backend.backtest_engine import run_backtest
from backend.data_loader import list_all_codes


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
        return {'ev': 0, 'win_rate': 0, 'pl_ratio': 0, 'positions': 0, 'trades': 0}
    wins = [r for r in pos_returns if r > 0]
    losses = [r for r in pos_returns if r <= 0]
    wr = len(wins) / n * 100
    aw = sum(wins) / len(wins) if wins else 0
    al = abs(sum(losses) / len(losses)) if losses else 0
    ev = wr / 100 * aw - (1 - wr / 100) * al
    plr = aw / al if al > 0 else 999
    return {'ev': round(ev, 2), 'win_rate': round(wr, 1), 'pl_ratio': round(plr, 2), 'positions': n, 'trades': result.total_trades}


if __name__ == '__main__':
    sname = sys.argv[1]
    dt = int(sys.argv[2])
    vr = int(sys.argv[3])
    vratio = int(sys.argv[4]) / 10.0
    sd = int(sys.argv[5])
    sr = int(sys.argv[6]) / 10.0
    dbl = int(sys.argv[7])
    dbt = int(sys.argv[8]) / 10.0
    bp = int(sys.argv[9]) / 10.0
    ra = int(sys.argv[10])
    el = int(sys.argv[11])
    chunk_prefix = sys.argv[12]  # "0", "3", "6", "68"

    all_codes = list_all_codes()
    codes = [c for c in all_codes if c.startswith(chunk_prefix)]
    print(f"chunk={chunk_prefix} codes={len(codes)}", file=sys.stderr)

    config = StrategyConfig(
        name=sname, k_type='daily', backtest_mode='signal',
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
    metrics = compute_position_ev(result)
    out = {'strategy': sname, 'chunk': chunk_prefix, **metrics}
    print(json.dumps(out, ensure_ascii=False))
