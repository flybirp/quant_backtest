# 量化回测框架使用指南

## 项目结构

```
quant_backtest/
├── strategies/
│   ├── rule/           # 规则型策略 (45个YAML)
│   └── ml/             # ML型策略 (_template.yaml)
├── backend/
│   ├── backtest_engine.py   # 回测引擎
│   ├── strategy_engine.py   # 策略配置 + 条件引擎
│   ├── state_machine.py     # 状态机策略 (zzh1.0)
│   ├── ml_engine.py         # ML策略引擎 (骨架)
│   ├── ml_bridge.py         # ML信号过滤桥接 ← 连接 quant_practical
│   └── main.py              # FastAPI 服务
├── analytics/          # 评价体系 (15个模块)
├── validation/         # 验证体系 (WF / 敏感性 / Monte Carlo / Train-Val-Test)
├── report.py           # 综合分析报告 (13节)
└── results/            # 回测结果JSON
```

---

## 快速开始: 规则型策略回测

```bash
cd ~/Documents/quant_backtest

# 基础报告
python report.py results/zzh7.3_大蓝筹.json

# 完整报告 (基准对比 + 情景压力 + Walk-Forward)
python report.py results/zzh7.3_大蓝筹.json \
    --benchmark hs300 \
    --scenario \
    --walk-forward --strategy zzh7.3 --pool 大蓝筹
```

---

## 报告章节说明

| 章节 | 内容 | 触发参数 |
|------|------|----------|
| 一、核心指标 | 收益/胜率/EV/夏普/最大回撤 | 默认 |
| 二、风险分析 | VaR/CVaR/Sortino/Calmar/连续亏损 | 默认 |
| 二B、回撤恢复 | 恢复天数/水下占比/回撤频率 | 默认 |
| 三、统计检验 | Bootstrap CI / t-test / 偏度峰度 | 默认 |
| 四、基准对比 | Alpha/Beta/CAPM分解/牛熊分析 | `--benchmark hs300` |
| 五、交易归因 | 年/板块/卖出原因/持仓集中度 | 默认 |
| 六、换手率容量 | 年换手率/均持仓天/信号密度 | 默认 |
| 七、交易成本 | TCA 实际成本 + 理论分解 | 默认 |
| 八、情景压力 | 6个A股历史危机回放 | `--scenario` |
| 九、策略相关性 | 多策略收益相关性矩阵 | 默认(多文件) |
| 十、Walk-Forward | 滚动样本外验证 | `--walk-forward` |
| 十一、参数敏感性 | 参数扫描 + 稳定性评分 | `--sensitivity` |
| 十二、过拟合检测 | Train/Val/Test EV衰减 | `--overfit` |
| 十三、ML信号过滤 | 全量 vs ML过滤对比 | `--ml-filter` |

---

## 一、规则型策略评估

### 1.1 单策略深度评估

```bash
python report.py results/zzh7.3_大蓝筹.json \
    --benchmark hs300 \
    --scenario \
    --walk-forward --strategy zzh7.3 --pool 大蓝筹 \
    --risk-free-rate 2.5
```

### 1.2 参数敏感性分析

```bash
python report.py results/zzh7.3_大蓝筹.json \
    --sensitivity stop_loss_pct \
    --values "5,10,15,20,25,30" \
    --strategy zzh7.3 --pool 大蓝筹
```

### 1.3 过拟合检测 (Train/Val 分离)

```bash
python report.py results/zzh7.3_大蓝筹.json \
    --overfit stop_loss_pct \
    --param-values "10,15,20" \
    --strategy zzh7.3 --pool 大蓝筹
```

### 1.4 多策略横向对比

```bash
python report.py \
    results/zzh7.3_大蓝筹.json \
    results/zzhY.3_大蓝筹.json \
    results/zzhX.0_大蓝筹.json \
    --benchmark hs300 --scenario
```

---

## 二、策略评估决策树

按以下顺序逐层淘汰，任一不通过即不合格：

### 第一层: 统计显著性

| 指标 | 阈值 | 来源章节 |
|------|------|----------|
| Bootstrap EV 95% CI 下限 | **> 0** | 三、统计检验 |
| t-test p-value | **< 0.05** | 三、统计检验 |
| 交易笔数 | **> 200** | 一、核心指标 |

### 第二层: 稳健性

| 指标 | 阈值 | 来源章节 |
|------|------|----------|
| Walk-Forward WFE | **> 60%** | 十、Walk-Forward |
| WF 盈利窗口占比 | **> 60%** | 十、Walk-Forward |
| 参数稳定性评分 | **> 70** | 十一、参数敏感性 |

### 第三层: 风险质量

| 指标 | 阈值 | 来源章节 |
|------|------|----------|
| 最大回撤 | **< 25%** | 一、核心指标 |
| 平均恢复天数 | **< 90天** | 二B、回撤恢复 |
| 水下占比 | **< 40%** | 二B、回撤恢复 |
| 连续亏损笔数 | **< 15** | 二、风险分析 |

### 第四层: Alpha 纯度

| 指标 | 阈值 | 来源章节 |
|------|------|----------|
| CAPM Alpha 年化 | **> 10% 且 p<0.05** | 四、基准对比 |
| Beta | **绝对值 < 0.3** | 四、基准对比 |
| R² | **< 0.2** | 四、基准对比 |

### 第五层: 场景韧性

| 指标 | 阈值 | 来源章节 |
|------|------|----------|
| 2015股灾回撤 | **< 基准回撤** | 八、情景压力 |
| 熊市胜率 | **> 40%** | 四、基准对比(牛熊分析) |

### 第六层: 实用性

| 指标 | 阈值 | 来源章节 |
|------|------|----------|
| 年换手率 | **< 500%** | 六、换手率容量 |
| 成本/毛利比 | **< 30%** | 七、交易成本 |
| 信号密度 | **> 5%** | 六、换手率容量 |

---

## 三、ML 信号过滤 (连接 quant_practical)

### 3.1 工作原理

```
quant_practical                    quant_backtest
     │                                    │
     ├─ T1: 规则信号 + Triple Barrier     │
     ├─ T2: LightGBM 训练                 │
     ├─ X0-X2: 全时段预测                 │
     │                                    │
     └─→ infer_review_csvs/*.csv ────→ ml_bridge.py
                                          │
                                   ┌──────┴──────┐
                                   │ 全量信号回测  │
                                   │ ML过滤后回测  │
                                   └──────┬──────┘
                                          │
                                   第十三节: 对比表
```

### 3.2 Step 1: 生成全时段 ML 预测

在 `~/Documents/quant_practical` 中，对每个需要的年份区间运行:

```bash
cd ~/Documents/quant_practical

# 批量生成 2024-2026 预测 (示例)
for year in 2024 2025; do
    python X0_select_top_of_today.py \
        --start ${year}-01-01 \
        --end ${year}-06-30 \
        --strategy standard_b1_v2
    python X0_select_top_of_today.py \
        --start ${year}-07-01 \
        --end ${year}-12-31 \
        --strategy standard_b1_v2
done
```

生成的 CSV 文件在 `infer_review_csvs/` 目录下。

### 3.3 Step 2: 对比回测

```bash
cd ~/Documents/quant_backtest

# 基础对比
python report.py results/zzh7.3_大蓝筹.json \
    --ml-filter ~/Documents/quant_practical/infer_review_csvs \
    --ml-threshold 0.3 \
    --ml-score-col ensemble_score \
    --strategy zzh7.3 --pool 大蓝筹

# 扫描最优阈值
for thresh in 0.2 0.3 0.4 0.5; do
    echo "=== Threshold: $thresh ==="
    python report.py results/zzh7.3_大蓝筹.json --quick \
        --ml-filter ~/Documents/quant_practical/infer_review_csvs \
        --ml-threshold $thresh \
        --strategy zzh7.3 --pool 大蓝筹 \
        2>&1 | grep -A8 "ML 信号"
done
```

### 3.4 ML 过滤判定标准

| 条件 | 判定 |
|------|------|
| EV 提升 > 10% 且 Sharpe 提升 > 10% | ML 过滤显著有效 |
| EV 提升 > 0 | 有正面效果 |
| EV 变化 < ±10% | 无明显效果 |
| EV 下降 > 10% | 反而降低收益 |

---

## 四、ML型策略开发 (未来)

当引入 ML 模型直接选股时，使用 Train/Val/Test 三层验证:

```bash
# 过拟合检测 (替代 Walk-Forward)
python report.py results/ml_demo_大蓝筹.json \
    --overfit stop_loss_pct \
    --param-values "5,10,15" \
    --strategy ml_demo --pool 大蓝筹
```

ML 策略的 Train/Val/Test 逻辑在 `validation/sample_split.py`:

```
全时段: |────────── Train 60% ──────────|── Val 20% ──|─ Test 20% ─|
        参数优化在此                      选最优配置在此   最终报告只看这个
```

---

## 五、策略目录约定

| 目录 | 策略类型 | 验证方式 | 示例 |
|------|----------|----------|------|
| `strategies/rule/` | 规则型 | Walk-Forward | `zzh7.3.yaml` |
| `strategies/ml/` | ML型 | Train/Val/Test | `_template.yaml` |

新建策略时放到对应目录，`_find_strategy_yaml()` 自动查找。

ML 策略模板: `strategies/ml/_template.yaml`

---

## 六、常用命令速查

```bash
# 最简报告
python report.py results/zzh7.3_大蓝筹.json

# 快速浏览 (跳过 Bootstrap 和 WF)
python report.py results/zzh7.3_大蓝筹.json --quick

# 完整深度评估
python report.py results/zzh7.3_大蓝筹.json \
    --benchmark hs300 --scenario --walk-forward \
    --strategy zzh7.3 --pool 大蓝筹

# 参数调优
python report.py results/zzh7.3_大蓝筹.json \
    --sensitivity stop_loss_pct --values "5,10,15,20,25,30" \
    --strategy zzh7.3 --pool 大蓝筹

# ML 过滤对比
python report.py results/zzh7.3_大蓝筹.json \
    --ml-filter ~/Documents/quant_practical/infer_review_csvs \
    --ml-threshold 0.3 --strategy zzh7.3 --pool 大蓝筹
```
