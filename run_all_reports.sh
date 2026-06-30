#!/bin/bash
# 逐个执行 report.py --quick，对 results/ 下所有 JSON 生成独立报告
# 用法: bash run_all_reports.sh

cd /Users/flybirp/Documents/quant_backtest

files=(results/*.json)
total=${#files[@]}
current=0

echo "=========================================="
echo "开始生成 $total 份策略报告"
echo "输出目录: reports/"
echo "=========================================="

for f in "${files[@]}"; do
    current=$((current + 1))
    name=$(basename "$f" .json)
    echo ""
    echo "[$current/$total] $name"
    echo "------------------------------------------"
    python3 report.py "$f" --quick --skip-scenario
done

echo ""
echo "=========================================="
echo "全部完成！共生成 $current 份报告"
echo "=========================================="
