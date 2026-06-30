# CLAUDE.md — 项目开发规范

## 项目概述

A股量化回测系统，用于测试和验证股票交易策略的历史表现。

**架构：**
- 后端：FastAPI (Python)，端口 8200
- 前端：React + TypeScript + Recharts，端口 3000
- 数据：/Users/flybirp/Documents/mainland_data (5062只A股日K CSV)

---

## 核心原则

### 1. 数据完整性
- **绝不修改原始数据文件** — CSV文件是只读的
- 所有数据处理必须在内存中完成
- 指标计算必须使用缓存机制（`_indicator_cache/`）

### 2. 策略配置一致性
- 策略YAML文件必须遵循 `strategies/` 目录结构
- 策略字段必须与 `StrategyConfig` 数据类匹配
- 新增指标必须在 `indicators.py` 中注册

### 3. 回测引擎稳定性
- **信号模式（signal）**：每笔交易独立，累计收益用 cumulative_sum_pct
- **组合模式（portfolio）**：有限资金，最大持仓数限制
- 两种模式的 equity_curve 计算逻辑不同，**禁止混用**

### 4. 测试覆盖
- 新增功能必须有对应测试
- 指标计算必须有回归测试
- 策略回测必须有交易回归测试

### 5. 策略一致性（核心原则）
- **买点和卖点必须使用同一个策略！**
- 不能用A策略买入，用B策略卖出
- 每个持仓必须记录买入时使用的策略
- 扫描卖点时必须校验策略是否匹配
- 违反此原则会导致策略失效，无法正确评估表现

**原因：**
- 不同策略有不同的止盈/止损逻辑
- 混用策略会导致回测结果无法复现
- 实盘中混用策略会破坏策略的期望值

**实现：**
- `portfolio.json` 中每个持仓必须有 `strategy` 字段
- `daily_select_pipeline.py` 扫描卖点时强制校验策略匹配
- 策略不匹配的持仓会被跳过并警告

### 6. 活着赚钱视角（策略评估标准）

**评估策略必须使用"活着赚钱视角"，而非单一指标：**

| 维度 | 权重 | 指标 | 说明 |
|------|------|------|------|
| **Calmar比率** | 30分 | CAGR / MaxDD | 每承受1%回撤的年化收益，>1满分 |
| **期望结构** | 25分 | EV / MaxDD | 每承受1%回撤的期望收益，>0.5满分 |
| **夏普比率** | 25分 | Sharpe | 风险调整收益，>1.5满分 |
| **可执行性** | 20分 | 月均交易数 | 散户月均≤2笔最佳 |

**为什么不用单一指标？**
- EV高但回撤大 → 散户扛不住
- 胜率高但盈亏比低 → 赚小亏大
- 夏普高但交易频繁 → 上班族没时间

**Calmar比率是核心：**
- Calmar = CAGR / MaxDD
- 每承受1%回撤，能获得多少年化收益
- Calmar > 1 说明策略值得承受回撤

**可执行性常被忽略：**
- 月均交易 > 5笔：上班族没时间
- 持仓天数 < 7天：需要频繁盯盘
- 止损 > 20%：散户心态会崩

**评分公式：**
```python
score = 0
# Calmar: 30分
score += min(calmar / 15 * 30, 30)
# EV/MaxDD: 25分
score += min(max(ev_dd, 0) / 1 * 25, 25)
# 胜率×盈亏比: 10分
score += min(win_pl / 1 * 10, 10)
# 月均交易: 10分（越少越好）
if monthly <= 20: score += 10
elif monthly <= 50: score += 7
else: score += 3
# 夏普: 25分
score += min(max(sharpe, 0) / 1.5 * 25, 25)
```

---

## 代码规范

### Python 代码风格
```python
# 使用类型注解
def calculate_return(trades: list[dict]) -> float:
    ...

# 使用 dataclass 定义数据结构
@dataclass
class Trade:
    entry_date: str
    exit_date: str
    profit_pct: float
    ...

# 使用常量定义配置
INITIAL_CAPITAL = 100000
COMMISSION_RATE = 0.0003
```

### 命名规范
- **模块文件**：小写下划线（`backtest_engine.py`）
- **类名**：大驼峰（`StrategyConfig`, `BacktestResult`）
- **函数名**：小写下划线（`run_backtest`, `check_condition`）
- **常量**：全大写（`INITIAL_CAPITAL`, `MAX_POSITIONS`）
- **策略文件**：中文或英文描述（`DS2.yaml`, `恐慌错杀-狙击.yaml`）

### 注释规范
```python
# 模块级文档
"""回测引擎 — 执行策略回测，生成交易记录和权益曲线"""

# 函数级文档
def run_backtest(config: StrategyConfig, progress_callback=None) -> BacktestResult:
    """
    运行策略回测

    Args:
        config: 策略配置
        progress_callback: 进度回调函数 (done, total)

    Returns:
        BacktestResult: 回测结果，包含交易记录和权益曲线
    """
```

---

## 模块职责边界

### `backend/` — 引擎层
| 模块 | 职责 | 禁止事项 |
|------|------|----------|
| `backtest_engine.py` | 回测执行，交易模拟 | 禁止修改策略配置 |
| `strategy_engine.py` | 条件检查，信号检测 | 禁止执行回测逻辑 |
| `indicators.py` | 技术指标计算 | 禁止修改原始数据 |
| `data_loader.py` | 数据加载，指标缓存 | 禁止执行策略逻辑 |
| `main.py` | FastAPI 路由 | 禁止包含业务逻辑 |

### `analytics/` — 评价层
- **所有函数必须是纯函数** — 无副作用，无 I/O
- 输入：`equity_curve: list[dict]` 或 `trades: list[dict]`
- 输出：指标值或 DataFrame
- 共享工具：`common.py` 中的 `to_equity_df()`

### `validation/` — 验证层
- 必须使用 `run_backtest` 运行回测
- 必须返回验证结果（不修改原始数据）
- 新增验证方法必须在 `__init__.py` 中注册

### `strategies/` — 策略配置
- `rule/` — 规则型策略（YAML）
- `ml/` — ML型策略（未来）
- 策略路径解析：`strategies/rule/` → `strategies/ml/` → `strategies/`

---

## 关键设计决策

### 1. 信号模式权益曲线用累加，不用复利
```python
# ✅ 正确：累加
cumulative_sum_pct += t.profit_pct
cum_pnl = base * (1 + cumulative_sum_pct / 100)

# ❌ 错误：复利（会导致天文数字）
cum_pnl *= (1 + t.profit_pct / 100)
```

### 2. 每日指标用前向填充
```python
# 信号模式的 equity curve 只有交易卖出日有数据点
# 计算日 VaR/Sharpe 时先用 forward_fill_daily() 补全每日权益
equity_df = forward_fill_daily(equity_curve)
daily_returns = equity_df['equity'].pct_change()
```

### 3. 所有 analytics 函数共享 `to_equity_df()`
```python
# 统一入口，处理排序、去重、日期索引
equity_df = to_equity_df(equity_curve)
```

### 4. 策略路径解析，不硬编码
```python
# 按顺序查找：strategies/rule/ → strategies/ml/ → strategies/
def _resolve_strategy_path(name: str) -> Path:
    ...
```

---

## 测试要求

### 单元测试
```bash
make test          # 运行全部测试
make test-watch    # 文件变更时自动运行
make lint          # Ruff linter
make typecheck     # Mypy 类型检查
make check         # lint + typecheck + test
```

### 测试覆盖率
- `tests/test_indicators_*.py` — 指标计算回归测试
- `tests/test_backtest_*.py` — 回测逻辑测试
- `tests/test_performance.py` — 收益计算测试
- `tests/test_risk.py` — 风险指标测试
- `tests/test_statistics.py` — 统计检验测试

### 测试数据
- 使用 `fixtures/` 目录中的固定数据
- 禁止使用实时市场数据运行测试
- 测试必须可重复（固定随机种子）

---

## 性能约束

### 回测引擎
- 单只股票回测：< 1 秒
- 276只股票（大蓝筹池）：< 5 分钟
- 5062只股票（全市场）：< 30 分钟

### 指标计算
- 指标缓存必须启用（`_indicator_cache/`）
- 预加载机制：`preload_indicator_cache()`
- 禁止重复计算已缓存指标

### 内存管理
- 单只股票数据：< 100 MB
- 全市场数据：< 8 GB
- 使用生成器处理大数据集

---

## 已知问题

### 分批策略 Bug
**问题：** `backtest_engine.py` 第 499 行 `open_lots[i]` 索引越界
**影响：** 以下策略无法运行
- A-口袋支点-激进-分批
- V1-吸筹确认-激进-分批
- V6-弹簧确认-激进-分批

**修复方向：** 检查分批买入/卖出逻辑中的索引管理

---

## 禁止事项

### 绝对禁止
- ❌ 修改原始 CSV 数据文件
- ❌ 在 analytics 函数中执行 I/O 操作
- ❌ 在 backtest_engine 中修改策略配置
- ❌ 使用复利计算信号模式权益曲线
- ❌ 跳过测试直接提交代码

### 需要确认
- ⚠️ 修改回测引擎核心逻辑（需要运行全部测试）
- ⚠️ 新增指标类型（需要更新策略配置 schema）
- ⚠️ 修改权益曲线计算逻辑（需要验证信号/组合模式）

---

## 提交规范

### 提交信息格式
```
<type>(<scope>): <subject>

# type: feat, fix, refactor, test, docs, chore, perf, style
# scope: engine, analytics, validation, report, strategy, ml

# 示例：
feat(analytics): add drawdown recovery analysis
fix(engine): correct signal mode Sharpe with forward-fill
refactor(analytics): extract shared to_equity_df to common.py
test(risk): add VaR and drawdown recovery unit tests
docs: add CLAUDE.md with development guidelines
```

### 提交前检查
```bash
make check         # lint + typecheck + test
```

---

## 开发流程

### 新增指标
1. 在 `indicators.py` 中实现计算函数
2. 在 `compute_all_indicators()` 中注册
3. 在 `strategy_engine.py` 中添加条件检查
4. 添加回归测试 `tests/test_indicators_*.py`
5. 更新 README.md 的指标列表

### 新增分析模块
1. 创建 `analytics/new_module.py`
2. 导入共享工具：`from .common import to_equity_df`
3. 函数签名：`def metric(equity_curve: list[dict]) -> float`
4. 在 `analytics/__init__.py` 中注册
5. 在 `report.py` 中添加显示函数
6. 添加单元测试

### 新增验证方法
1. 创建 `validation/new_method.py`
2. 导入回测引擎：`from backend.backtest_engine import run_backtest`
3. 在 `validation/__init__.py` 中注册
4. 在 `report.py` 中添加 CLI 标志
5. 添加集成测试

### 新增策略
1. 在 `strategies/rule/` 中创建 YAML 文件
2. 遵循现有策略的字段结构
3. 运行 `python run_pipeline.py <strategy> --pool 大蓝筹` 验证
4. 检查报告输出是否正常

---

## 文档要求

### 必须文档
- `README.md` — 项目概述、快速启动、API 文档
- `ARCHITECTURE.md` — 架构设计、数据流、模块职责
- `CLAUDE.md` — 开发规范、代码风格、禁止事项

### 可选文档
- `STRATEGY_GUIDE.md` — 策略开发指南
- `USAGE.md` — 详细使用说明
- `RESEARCH_AGENT_UPGRADE.md` — 研究代理升级记录

### 报告文件
- `reports/YYYYMMDD_HHMMSS_策略_池.txt` — 策略回测报告
- `results/策略_池.json` — 策略回测结果（JSON）
- `top_strategy.md` — 策略排名汇总

---

## 环境配置

### Python 版本
- Python 3.10+

### 依赖管理
```bash
pip install -r requirements.txt  # 或使用 pyproject.toml
```

### 数据路径
```python
# 环境变量或配置文件
DATA_DIR = /Users/flybirp/Documents/mainland_data
```

### 缓存目录
```
_indicator_cache/  # 指标缓存（自动生成，可删除）
results/           # 回测结果
reports/           # 回测报告
```

---

## 监控与告警

### 性能监控
- 回测耗时 > 10 分钟：检查指标计算
- 内存使用 > 8 GB：检查数据加载
- 测试失败：禁止提交代码

### 数据质量
- 退市股票：检查生存偏差
- 缺失数据：检查数据完整性
- 异常值：检查指标计算

---

## 紧急回滚

### 代码回滚
```bash
git revert <commit-hash>
git push
```

### 数据回滚
```bash
# 恢复策略配置
git checkout HEAD~1 -- strategies/

# 恢复回测结果
git checkout HEAD~1 -- results/
```

---

## 联系方式

- 项目地址：https://github.com/flybirp/quant_backtest
- 问题反馈：GitHub Issues
