#!/bin/bash
# 细搜索 - 围绕TOP1 (dt=40, vr=95, sd=3, dbl=40) 微调
# 只搜3个维度: double_bottom_tolerance, break_pct, recent_above_days
# 其余固定为粗搜索最优值: vratio=3.0, sr=0.7

cd /Users/flybirp/Documents/quant_backtest
PY=/Users/flybirp/.workbuddy/binaries/python/envs/default/bin/python3
RESULTS=grid_search_volume_reversal_fine.jsonl
LOG=grid_search_fine.log

# 固定参数
DT=40
VR=95
SD=3
DBL=40
VRATIO=3.0
SR=0.7

echo "细搜索开始 $(date)" > $LOG
echo "围绕 TOP1: dt=$DT vr=$VR sd=$SD dbl=$DBL vratio=$VRATIO sr=$SR" >> $LOG

N=0
TOTAL=27  # 3*3*3

for dbt in 2.0 3.0 5.0; do
  for bp in 0.5 1.0 2.0; do
    for ra in 15 20 30; do
      N=$((N+1))
      echo "[$N/$TOTAL] dbt=$dbt bp=$bp ra=$ra ..." >> $LOG
      $PY -c "
import sys, os, json
sys.path.insert(0, '$PWD')
from backend.strategy_engine import StrategyConfig
from backend.backtest_engine import run_backtest
from collections import defaultdict

with open('stock_pools.json') as f: pools = json.load(f)
codes = pools['大蓝筹'].get('codes', [])

config = StrategyConfig(
    name='fine', k_type='daily', backtest_mode='signal',
    buy_groups=[{'name':'b','conditions':[
        {'indicator':'zhixing_long_downtrend','params':{'days':$DT}},
        {'indicator':'recent_volume_explosion','params':{'lookback':15,'vol_rank_threshold':$VR,'vol_ratio_threshold':$VRATIO}},
        {'indicator':'volume_sustained','params':{'sustain_days':$SD,'sustain_ratio':$SR,'explosion_lookback':15,'vol_rank_threshold':$VR,'vol_ratio_threshold':$VRATIO}},
        {'indicator':'price_cross_above_zhixing_fast'},
    ]}],
    sell_groups=[
        {'name':'s1','conditions':[{'indicator':'double_bottom_broken','params':{'lookback':$DBL,'tolerance':$dbt,'break_pct':$bp}}]},
        {'name':'s2','conditions':[{'indicator':'price_below_zhixing_slow_consecutive','params':{'days':2,'recent_above_days':$ra}}]},
    ],
    add_groups=[{'name':'a1','conditions':[
        {'indicator':'price_below_zhixing_fast'},
        {'indicator':'double_bottom','params':{'lookback':$DBL,'tolerance':$dbt}},
    ]}],
    reduce_groups=[{'name':'r1','conditions':[{'indicator':'price_below_bbi_consecutive','params':{'days':2}}]}],
    buy_price_type='open', sell_price_type='avg', buy_execution='next_day', sell_execution='next_day',
    stop_loss_pct=8, take_profit_pct=0, trailing_stop_pct=0, max_hold_days=60,
    stock_pool=codes, entry_ladder=[{'trigger_pct':0,'weight':50}],
)
result = run_backtest(config, start_date='20240101', end_date='20260606')

positions = defaultdict(list)
for t in result.trades:
    positions[t['trade_id']].append(t)
pos_returns = []
for tid, trades in positions.items():
    sell_trades = [t for t in trades if t.get('sell_reason')]
    if not sell_trades: continue
    total_shares = sum(t['shares'] for t in sell_trades)
    if total_shares == 0: continue
    wr2 = sum(t['shares'] * t['profit_pct'] for t in sell_trades) / total_shares
    pos_returns.append(wr2)
n = len(pos_returns)
if n == 0:
    print(json.dumps({'error': 'no positions'}))
else:
    wins = [r for r in pos_returns if r > 0]
    losses = [r for r in pos_returns if r <= 0]
    wr = len(wins)/n*100
    aw = sum(wins)/len(wins) if wins else 0
    al = abs(sum(losses)/len(losses)) if losses else 0
    ev = wr/100*aw - (1-wr/100)*al
    plr = aw/al if al > 0 else 999
    out = {
        'params': {'downtrend_days':$DT,'vol_rank_threshold':$VR,'sustain_days':$SD,'double_bottom_lookback':$DBL,
                   'vol_ratio_threshold':$VRATIO,'sustain_ratio':$SR,'double_bottom_tolerance':$dbt,'break_pct':$bp,'recent_above_days':$ra},
        'position_ev': round(ev,2), 'position_win_rate': round(wr,1),
        'position_pl_ratio': round(plr,2), 'position_count': n, 'position_trades': result.total_trades,
    }
    print(json.dumps(out))
" >> $RESULTS 2>>$LOG
      RC=$?
      if [ $RC -eq 0 ]; then
        LAST=$(tail -1 $RESULTS 2>/dev/null)
        echo "[$N/$TOTAL] OK $LAST" >> $LOG
      else
        echo "[$N/$TOTAL] FAIL rc=$RC" >> $LOG
      fi
    done
  done
done

echo "细搜索完成 $(date)" >> $LOG

# 汇总
$PY -c "
import json
results = []
with open('$RESULTS') as f:
    for line in f:
        line = line.strip()
        if line:
            try: results.append(json.loads(line))
            except: pass
valid = [r for r in results if 'position_ev' in r and r.get('position_count',0)>=5]
valid.sort(key=lambda x: x['position_ev'], reverse=True)
print(f'有效结果: {len(valid)}条')
print()
print(f'{\"排名\":>4s} {\"EV\":>8s} {\"胜率\":>6s} {\"盈亏比\":>6s} {\"持仓\":>6s} | dbt bp ra')
print('-'*80)
for i,r in enumerate(valid[:20],1):
    p=r['params']
    print(f'{i:4d} {r[\"position_ev\"]:+7.2f}% {r[\"position_win_rate\"]:5.1f}% {r[\"position_pl_ratio\"]:6.2f} {r[\"position_count\"]:6d} | {p[\"double_bottom_tolerance\"]} {p[\"break_pct\"]} {p[\"recent_above_days\"]}')

with open('grid_search_volume_reversal_fine.json','w') as f:
    json.dump({'results':results,'top20':valid[:20]},f,ensure_ascii=False,indent=2)
print(f'\n结果已保存到 grid_search_volume_reversal_fine.json')
" >> $LOG 2>&1

echo "全部完成 $(date)" >> $LOG
