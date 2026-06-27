#!/bin/bash
# 运行剩余策略

cd /Users/flybirp/Documents/quant_backtest

# 剩余策略列表
strategies=(
  "A-口袋支点-激进-分批"
  "V1-吸筹确认-激进-分批"
  "V6-弹簧确认-激进-分批"
  "zzh4.5"
  "zzh7.3"
  "zzh7.3_agent_r1"
  "zzh7.3_agent_r2"
  "zzh7.3_agent_r3"
  "zzh7.3_agent_r4"
  "zzh7.3_value_test"
  "zzhX.0"
  "zzhY.3"
  "zzhY.4"
  "双均线MACD策略"
  "恐慌错杀-狙击"
  "恐慌错杀-狙击_r1"
  "恐慌错杀-狙击_r2"
  "恐慌错杀-狙击_r3"
  "恐慌错杀-狙击_r4"
  "恐慌错杀-狙击_r5"
  "恐慌错杀-狙击_r6"
  "恐慌错杀-狙击_r7"
  "恐慌错杀-狙击_r8"
  "恐慌错杀-狙击_r9"
  "恐慌错杀-狙击_r10"
  "恐慌错杀-狙击_r11"
  "恐慌错杀-狙击_r12"
  "恐慌错杀-狙击_r13"
  "恐慌错杀-狙击_r14"
  "恐慌错杀-狙击_r15"
  "恐慌错杀-狙击_r16"
  "知行量化-金叉回调"
  "威科夫吸筹-弹簧效应"
  "威科夫派发-上冲回落"
  "周线趋势策略"
  "小池测试策略"
  "横盘缩量突破"
  "量能异动前置-金叉确认"
  "威科夫量价法则"
  "量能爆发底部反转"
  "量能爆发底部反转-TOP1"
  "量能爆发底部反转-TOP2"
  "量能爆发底部反转-TOP3"
  "量能爆发底部反转-TOP4"
  "量能爆发底部反转-TOP5"
)

total=${#strategies[@]}
current=0

echo "=========================================="
echo "开始运行剩余 $total 个策略"
echo "=========================================="

for strategy in "${strategies[@]}"; do
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
