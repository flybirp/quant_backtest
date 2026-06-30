#!/bin/bash
# 量能爆发底部反转 - 更深层网格搜索
# 4个分组共35组，每组~65秒，预计~38分钟
# 用法: nohup bash run_grid_search_deep.sh > /dev/null 2>&1 &

cd /Users/flybirp/Documents/quant_backtest
PY=/Users/flybirp/.workbuddy/binaries/python/envs/default/bin/python3
W=run_grid_worker_deep.py
R=grid_search_volume_reversal_deep.jsonl
LOG=grid_search_deep.log

echo "开始 $(date)" > $LOG

N=0
TOTAL=35

# Set 1: downtrend × vol_rank (9组)
# 固定: vratio=3.0 sd=3 sr=0.7 dbl=40 dbt=2.0 bp=2.0(ra=15 el=15
for dt in 40 50 60; do
  for vr in 95 97 98; do
    N=$((N+1))
    echo "[$N/$TOTAL] S1 dt=$dt vr=$vr" >> $LOG
    $PY $W $dt $vr 3.0 3 0.7 40 2.0 20 15 15 >> $R 2>>$LOG
    echo "[$N/$TOTAL] OK $(tail -1 $R)" >> $LOG
  done
done

# Set 2: dbl_lookback × break_pct (9组)
# 固定: dt=40 vr=95 vratio=3.0 sd=3 sr=0.7 dbt=2.0 ra=15 el=15
for dbl in 40 50 60; do
  for bp in 20 25 30; do  # bp×10: 20=2.0, 25=2.5, 30=3.0
    N=$((N+1))
    echo "[$N/$TOTAL] S2 dbl=$dbl bp=$(echo "scale=1; $bp/10" | bc)" >> $LOG
    $PY $W 40 95 3.0 3 0.7 $dbl 2.0 $bp 15 15 >> $R 2>>$LOG
    echo "[$N/$TOTAL] OK $(tail -1 $R)" >> $LOG
  done
done

# Set 3: recent_above_days × sustain_days (8组)
# 固定: dt=40 vr=95 vratio=3.0 sr=0.7 dbl=40 dbt=2.0 bp=2.0 el=15
for ra in 8 10 12 15; do
  for sd in 2 3; do
    N=$((N+1))
    echo "[$N/$TOTAL] S3 ra=$ra sd=$sd" >> $LOG
    $PY $W 40 95 3.0 $sd 0.7 40 2.0 20 $ra 15 >> $R 2>>$LOG
    echo "[$N/$TOTAL] OK $(tail -1 $R)" >> $LOG
  done
done

# Set 4: explosion_lookback × vol_ratio (9组)
# 固定: dt=40 vr=95 sd=3 sr=0.7 dbl=40 dbt=2.0 bp=2.0 ra=15
for el in 10 15 20; do
  for vratio in 2.5 3.0 3.5; do
    N=$((N+1))
    echo "[$N/$TOTAL] S4 el=$el vratio=$vratio" >> $LOG
    $PY $W 40 95 $vratio 3 0.7 40 2.0 20 15 $el >> $R 2>>$LOG
    echo "[$N/$TOTAL] OK $(tail -1 $R)" >> $LOG
  done
done

echo "搜索完成 $(date)" >> $LOG

# 汇总
$PY -c "
import json
results = []
with open('$R') as f:
    for line in f:
        line = line.strip()
        if line:
            try: results.append(json.loads(line))
            except: pass
valid = [r for r in results if 'position_ev' in r and r.get('position_count',0)>=5]
valid.sort(key=lambda x: x['position_ev'], reverse=True)
print(f'有效结果: {len(valid)}条', file=open('$LOG','a'))
print(f'', file=open('$LOG','a'))
print(f'{\"排名\":>4s} {\"EV\":>8s} {\"胜率\":>6s} {\"盈亏比\":>6s} {\"持仓\":>6s} | dt vr vratio sd sr dbl dbt bp ra el', file=open('$LOG','a'))
print('-'*110, file=open('$LOG','a'))
for i,r in enumerate(valid[:20],1):
    p=r['params']
    print(f'{i:4d} {r[\"position_ev\"]:+7.2f}% {r[\"position_win_rate\"]:5.1f}% {r[\"position_pl_ratio\"]:6.2f} {r[\"position_count\"]:6d} | '
          f'{p[\"downtrend_days\"]} {p[\"vol_rank_threshold\"]} {p[\"vol_ratio_threshold\"]} {p[\"sustain_days\"]} '
          f'{p[\"sustain_ratio\"]} {p[\"double_bottom_lookback\"]} {p[\"double_bottom_tolerance\"]} '
          f'{p[\"break_pct\"]} {p[\"recent_above_days\"]} {p[\"explosion_lookback\"]}', file=open('$LOG','a'))

with open('grid_search_volume_reversal_deep.json','w') as f:
    json.dump({'results':results,'top20':valid[:20]},f,ensure_ascii=False,indent=2)
print(f'\n已保存到 grid_search_volume_reversal_deep.json', file=open('$LOG','a'))
" >> $LOG 2>&1

echo "全部完成 $(date)" >> $LOG
