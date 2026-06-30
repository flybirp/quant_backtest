#!/bin/bash
# 量能爆发底部反转 - 跨标的池横向对比 (逐个执行版)
# 每组单独进程，结果写入独立文件，最后合并

PY=/Users/flybirp/.workbuddy/binaries/python/envs/default/bin/python3
SCRIPT=/Users/flybirp/Documents/quant_backtest/run_cross_pool_worker.py
RESULT_DIR=/Users/flybirp/Documents/quant_backtest/cross_pool_tmp

rm -rf $RESULT_DIR
mkdir -p $RESULT_DIR

STRATEGIES=(
  "TOP1:100:95:30:3:7:60:30:30:10:15"
  "TOP2:100:95:30:3:7:60:30:30:15:15"
  "TOP3:100:95:30:3:7:60:30:30:10:20"
  "TOP4:90:95:30:3:7:60:20:30:15:15"
  "TOP5:70:95:30:3:7:60:20:30:15:15"
)

POOLS=("大蓝筹" "科创板" "创业板" "全量")

CNT=0
TOTAL=$((${#STRATEGIES[@]} * ${#POOLS[@]}))

for S in "${STRATEGIES[@]}"; do
  IFS=':' read -r SNAME DT VR VRATIO SD SR DBL DBT BP RA EL <<< "$S"

  for PNAME in "${POOLS[@]}"; do
    CNT=$((CNT + 1))
    OUTFILE="${RESULT_DIR}/${SNAME}_${PNAME}.json"
    echo -n "[$CNT/$TOTAL] $SNAME × $PNAME ... "
    $PY $SCRIPT "$SNAME" "$PNAME" "$DT" "$VR" "$VRATIO" "$SD" "$SR" "$DBL" "$DBT" "$BP" "$RA" "$EL" > "$OUTFILE" 2>"${OUTFILE}.err"
    if [ $? -eq 0 ] && [ -s "$OUTFILE" ]; then
      cat "$OUTFILE"
      echo ""
    else
      echo "FAILED (see ${OUTFILE}.err)"
      echo '{"strategy":"'"$SNAME"'","pool":"'"$PNAME"'","ev":null,"win_rate":null,"pl_ratio":null,"positions":0,"trades":0}' > "$OUTFILE"
    fi
  done
done

# Merge all results
echo "Merging results..."
OUT_FINAL=/Users/flybirp/Documents/quant_backtest/cross_pool_results.json
echo "[" > $OUT_FINAL
FIRST=true
for f in ${RESULT_DIR}/*.json; do
  [ ! -s "$f" ] && continue
  if [ "$FIRST" = true ]; then
    FIRST=false
  else
    echo "," >> $OUT_FINAL
  fi
  cat "$f" >> $OUT_FINAL
done
echo "" >> $OUT_FINAL
echo "]" >> $OUT_FINAL

echo "Done! Results saved to $OUT_FINAL"
