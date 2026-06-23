#!/bin/bash
# 量能爆发底部反转 - 跨标的池横向对比
# 5策略(TOP1-5) × 4标的池 = 20组回测
# 每组单独进程，避免OOM

PY=/Users/flybirp/.workbuddy/binaries/python/envs/default/bin/python3
SCRIPT=/Users/flybirp/Documents/quant_backtest/run_cross_pool_worker.py
OUTFILE=/Users/flybirp/Documents/quant_backtest/cross_pool_results.json

echo "Starting cross-pool backtest: 5 strategies × 4 pools = 20 combos"

# TOP1: dt=100 vr=95 vratio=30 sd=3 sr=7 dbl=60 dbt=30 bp=30 ra=10 el=15
# TOP2: dt=100 vr=95 vratio=30 sd=3 sr=7 dbl=60 dbt=30 bp=30 ra=15 el=15
# TOP3: dt=100 vr=95 vratio=30 sd=3 sr=7 dbl=60 dbt=30 bp=30 ra=10 el=20
# TOP4: dt=90  vr=95 vratio=30 sd=3 sr=7 dbl=60 dbt=20 bp=30 ra=15 el=15
# TOP5: dt=70  vr=95 vratio=30 sd=3 sr=7 dbl=60 dbt=20 bp=30 ra=15 el=15

STRATEGIES=(
  "TOP1:100:95:30:3:7:60:30:30:10:15"
  "TOP2:100:95:30:3:7:60:30:30:15:15"
  "TOP3:100:95:30:3:7:60:30:30:10:20"
  "TOP4:90:95:30:3:7:60:20:30:15:15"
  "TOP5:70:95:30:3:7:60:20:30:15:15"
)

POOLS=("大蓝筹" "科创板" "创业板")

rm -f $OUTFILE
echo "[" > $OUTFILE
FIRST=true

TOTAL=${#STRATEGIES[@]}
CNT=0

for S in "${STRATEGIES[@]}"; do
  IFS=':' read -r SNAME DT VR VRATIO SD SR DBL DBT BP RA EL <<< "$S"
  CNT=$((CNT + 1))

  for PNAME in "${POOLS[@]}"; do
    echo -n "[$CNT/$TOTAL] $SNAME × $PNAME ... "
    RESULT=$($PY $SCRIPT "$SNAME" "$PNAME" "$DT" "$VR" "$VRATIO" "$SD" "$SR" "$DBL" "$DBT" "$BP" "$RA" "$EL" 2>/dev/null)
    if [ $? -eq 0 ]; then
      echo "$RESULT"
      if [ "$FIRST" = true ]; then
        FIRST=false
      else
        echo "," >> $OUTFILE
      fi
      echo "$RESULT" >> $OUTFILE
    else
      echo "FAILED"
    fi
  done
done

echo "]" >> $OUTFILE
echo "Done! Results saved to $OUTFILE"
