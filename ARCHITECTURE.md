# Architecture

## Data Flow

```
stock data (CSV)                    strategy YAML
      │                                  │
      ▼                                  ▼
 data_loader.py ◄────────────── strategy_engine.py
      │                                  │
      ├── indicators.py (MA, MACD, RSI, KDJ, zz_short/long...)
      │                                  │
      ▼                                  ▼
 backtest_engine.py ◄────────── state_machine.py (zzh1.0)
      │
      ├── signal mode (流式, 每笔独立, 无资金约束)
      └── portfolio mode (有限资金, 最大持仓数)
      │
      ▼
 trades + equity_curve (list[dict])
      │
      ├─────────────────────────────────────────────┐
      ▼                                             ▼
 analytics/                                  validation/
 ├── performance.py  (总收益, 年化, 月/年收益)     ├── walk_forward.py
 ├── risk.py         (回撤, VaR, Sharpe, 恢复)     ├── sensitivity.py
 ├── statistics.py   (Bootstrap, t-test, MC)       ├── monte_carlo.py
 ├── benchmark.py    (Alpha/Beta, CAPM, 牛熊)      └── sample_split.py
 ├── attribution.py  (年/板块/原因/集中度)
 ├── factors.py      (CAPM分解, 信息系数)
 ├── capacity.py     (换手率, 容量估算)
 ├── tca.py          (交易成本归因)
 ├── correlation.py  (策略间相关性)
 ├── scenario.py     (历史情景压力测试)
 ├── formatters.py   (文本表格)
 └── common.py       (共享工具: to_equity_df, forward_fill_daily)
      │
      ▼
 report.py (13节综合报告)
      │
      ▼
 reports/YYYYMMDD_HHMMSS_策略_池.txt
```

## Module Responsibilities

### `backend/` — 引擎层

| Module | Responsibility | Key Classes |
|--------|---------------|-------------|
| `strategy_engine.py` | StrategyConfig, Signal, Trade, 条件检查, 向量化信号检测 | `StrategyConfig`, `check_condition`, `check_group`, `detect_signals_vectorized` |
| `backtest_engine.py` | 回测执行 (signal/portfolio模式), 交易模拟, 统计计算 | `run_backtest`, `_compute_statistics`, `BacktestResult` |
| `state_machine.py` | 状态机策略 zzh1.0 (买/卖/加/减仓决策) | `ZZH10StateMachine` |
| `indicators.py` | 技术指标计算 (MA, MACD, RSI, KDJ, BB, zz_short/long...) | `compute_all_indicators` |
| `data_loader.py` | 股票数据加载 (CSV → DataFrame + 指标缓存) | `load_stock_with_indicators`, `preload_indicator_cache` |
| `ml_engine.py` | ML因子计算 + 模型训练 + 选股 (骨架) | `FactorCalculator`, `train_model`, `generate_signals` |
| `ml_bridge.py` | 连接 quant_practical ML 预测 → 交易过滤 | `load_ml_predictions`, `filter_trades_by_ml`, `run_ml_filtered_backtest` |
| `parallel_engine.py` | 多进程并行信号检测 | `run_backtest_parallel` |
| `main.py` | FastAPI 服务 + 策略路径解析 | `_config_from_dict`, `_resolve_strategy_path` |
| `benchmark_data.py` | 基准指数加载 (hs300, cyb, zz1000...) | `load_benchmark_equity_curve` |

### `analytics/` — 评价层

All functions take `equity_curve` (list of dicts) and/or `trades` (list of dicts) as input, return metrics. No side effects, no I/O.

### `validation/` — 验证层

| Module | Purpose | When to use |
|--------|---------|-------------|
| `walk_forward.py` | 滚动样本外验证 | 规则型策略日常验证 |
| `sensitivity.py` | 单参数扫描 + 热力图 + 稳定性评分 | 调参优化 |
| `monte_carlo.py` | Bootstrap重采样 + 破产概率 | 尾部风险估计 |
| `sample_split.py` | Train/Val/Test时序分离 + 过拟合检测 | ML型策略调参 |

### `strategies/` — 策略配置

```
strategies/
├── rule/    # 规则型: YAML → StrategyConfig → backtest_engine
└── ml/      # ML型: YAML → StrategyConfig → ml_engine (未来)
```

## Key Design Decisions

### 1. Signal mode equity curve uses cumulative sum, not product

信号模式下每笔交易独立，复利乘法会导致天文数字的总收益。改用 `cumulative_sum_pct` 累加:

```python
cumulative_sum_pct += t.profit_pct
cum_pnl = base * (1 + cumulative_sum_pct / 100)
```

### 2. Forward-fill for daily metrics

信号模式的 equity curve 只有交易卖出日有数据点。计算日 VaR/Sharpe 时先用 `forward_fill_daily()` 补全每日权益，再做 `pct_change()`。

### 3. `_to_equity_df` is the shared entry point

所有 analytics 函数都通过 `to_equity_df()` 把 `list[dict]` 转为 `pd.DataFrame`，统一处理排序、去重、日期索引。

### 4. Trade-level vs Portfolio-level VaR

`var_historical` / `cvar_historical` 基于单笔交易收益率 — 适用 trade-level 分析。
`var_daily` / `cvar_daily` 基于每日权益曲线 — 适用 portfolio-level 风险度量。

### 5. Strategy paths are resolved, not hardcoded

`_find_strategy_yaml()` / `_resolve_strategy_path()` 按 `strategies/rule/` → `strategies/ml/` → `strategies/` 顺序查找，向后兼容。

## Adding a New Analytics Module

1. Create `analytics/new_module.py`
2. Import shared tools: `from .common import to_equity_df as _to_equity_df`
3. Functions take `equity_curve: list[dict]` or `trades: list[dict]`
4. Register in `analytics/__init__.py`
5. Add display function to `report.py`

## Adding a New Validation Method

1. Create `validation/new_method.py`
2. Import `run_backtest` from `backend.backtest_engine`
3. Register in `validation/__init__.py`
4. Add CLI flag + section to `report.py`

## Running Tests

```bash
make test          # Run all 86 tests
make test-watch    # Auto-run on file changes
make lint          # Ruff linter
make typecheck     # Mypy
make check         # lint + typecheck + test (all three)
```
