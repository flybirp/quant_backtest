
import sys, os, json, gc
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backend.strategy_engine import StrategyConfig
from backend.backtest_engine import run_backtest
from collections import defaultdict

params = json.loads(sys.argv[1])
codes = json.loads(sys.argv[2])

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
        weighted_return = sum(t['shares'] * t['profit_pct'] for t in sell_trades) / total_shares
        pos_returns.append(weighted_return)
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

config = StrategyConfig(
    name="grid_worker", k_type='daily', backtest_mode='signal',
    buy_groups=[{"name": "量能爆发站上fast", "conditions": [
        {"indicator": "zhixing_long_downtrend", "params": {"days": params['downtrend_days']}},
        {"indicator": "recent_volume_explosion", "params": {"lookback": params.get('explosion_lookback', 15),
            "vol_rank_threshold": params['vol_rank_threshold'],
            "vol_ratio_threshold": params['vol_ratio_threshold']}},
        {"indicator": "volume_sustained", "params": {"sustain_days": params['sustain_days'],
            "sustain_ratio": params['sustain_ratio'],
            "explosion_lookback": params.get('explosion_lookback', 15),
            "vol_rank_threshold": params['vol_rank_threshold'],
            "vol_ratio_threshold": params['vol_ratio_threshold']}},
        {"indicator": "price_cross_above_zhixing_fast"},
    ]}],
    sell_groups=[
        {"name": "双底破坏清仓", "conditions": [
            {"indicator": "double_bottom_broken", "params": {"lookback": params['double_bottom_lookback'],
                "tolerance": params['double_bottom_tolerance'], "break_pct": params['break_pct']}}]},
        {"name": "连续低于slow清仓", "conditions": [
            {"indicator": "price_below_zhixing_slow_consecutive", "params": {"days": 2,
                "recent_above_days": params.get('recent_above_days', 20)}}]},
    ],
    add_groups=[{"name": "跌破fast后双底", "conditions": [
        {"indicator": "price_below_zhixing_fast"},
        {"indicator": "double_bottom", "params": {"lookback": params['double_bottom_lookback'],
            "tolerance": params['double_bottom_tolerance']}},
    ]}],
    reduce_groups=[{"name": "连续低于BBI减仓", "conditions": [
        {"indicator": "price_below_bbi_consecutive", "params": {"days": 2}}]}],
    buy_price_type='open', sell_price_type='avg', buy_execution='next_day', sell_execution='next_day',
    stop_loss_pct=8, take_profit_pct=0, trailing_stop_pct=0, max_hold_days=60,
    stock_pool=codes, entry_ladder=[{"trigger_pct": 0, "weight": 50}],
)

result = run_backtest(config, start_date='20240101', end_date='20260606')
ev, wr, plr, npos, ntrades = compute_position_ev(result)
output = {
    "position_ev": round(ev, 2), "position_win_rate": round(wr, 1),
    "position_pl_ratio": round(plr, 2), "position_count": npos, "total_trades": ntrades,
}
print(json.dumps(output))
