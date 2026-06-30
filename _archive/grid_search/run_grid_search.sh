#!/bin/bash
# 量能爆发底部反转 - 网格搜索
# 每组参数单独跑一个python进程，结果逐行追加到JSONL文件
# nohup 启动，不受超时限制

cd /Users/flybirp/Documents/quant_backtest
PY=/Users/flybirp/.workbuddy/binaries/python/envs/default/bin/python3
WORKER=run_grid_worker.py
RESULTS=grid_search_volume_reversal_results.jsonl
LOG=grid_search.log

echo "开始 $(date)" > $LOG

N=0
TOTAL=36  # 3*2*2*3

for dt in 20 30 40; do
  for vr in 90 95; do
    for sd in 3 5; do
      for dbl in 20 30 40; do
        N=$((N+1))
        echo "[$N/$TOTAL] dt=$dt vr=$vr sd=$sd dbl=$dbl ..." >> $LOG
        # 跑worker，输出一行JSON
        $PY $WORKER $dt $vr $sd $dbl >> $RESULTS 2>>$LOG
        RC=$?
        if [ $RC -eq 0 ]; then
          # 读取最后一行（刚写入的结果）
          LAST=$(tail -1 $RESULTS)
          echo "[$N/$TOTAL] OK $LAST" >> $LOG
        else
          echo "[$N/$TOTAL] FAIL rc=$RC" >> $LOG
        fi
      done
    done
  done
done

echo "粗搜索完成 $(date)" >> $LOG

# 汇总结果
$PY -c "
import json
results = []
with open('$RESULTS') as f:
    for line in f:
        line = line.strip()
        if line:
            try:
                results.append(json.loads(line))
            except:
                pass
valid = [r for r in results if 'position_ev' in r and r.get('position_count',0)>=5]
valid.sort(key=lambda x: x['position_ev'], reverse=True)
print(f'有效结果: {len(valid)}条')
print()
print(f'{\"排名\":>4s} {\"EV\":>8s} {\"胜率\":>6s} {\"盈亏比\":>6s} {\"持仓\":>6s} | dt vr sd dbl')
print('-'*70)
for i,r in enumerate(valid[:20],1):
    p=r['params']
    print(f'{i:4d} {r[\"position_ev\"]:+7.2f}% {r[\"position_win_rate\"]:5.1f}% {r[\"position_pl_ratio\"]:6.2f} {r[\"position_count\"]:6d} | {p[\"downtrend_days\"]} {p[\"vol_rank_threshold\"]} {p[\"sustain_days\"]} {p[\"double_bottom_lookback\"]}')

# 保存最终JSON
with open('grid_search_volume_reversal_results.json','w') as f:
    json.dump({'results':results,'top20':valid[:20]},f,ensure_ascii=False,indent=2)
print(f'\n结果已保存到 grid_search_volume_reversal_results.json')
" >> $LOG 2>&1

echo "全部完成 $(date)" >> $LOG
