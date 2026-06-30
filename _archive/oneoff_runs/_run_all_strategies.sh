#!/bin/bash
# 批量运行所有策略的pipeline

cd /Users/flybirp/Documents/quant_backtest

# 获取所有策略文件
strategies=$(ls strategies/rule/*.yaml | xargs -I {} basename {} .yaml)

total=$(echo "$strategies" | wc -l)
current=0

echo "=========================================="
echo "开始批量运行 $total 个策略"
echo "=========================================="

for strategy in $strategies; do
    current=$((current + 1))
    echo ""
    echo "[$current/$total] 运行策略: $strategy"
    echo "------------------------------------------"
    python3 run_pipeline.py "$strategy" --pool 大蓝筹 --skip-walk-forward --skip-monte-carlo --skip-scenario --skip-survival
done

echo ""
echo "=========================================="
echo "全部完成！"
echo "=========================================="
