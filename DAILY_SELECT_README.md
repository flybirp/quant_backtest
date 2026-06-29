# 每日选股/卖点扫描系统

## 功能

1. **下载数据**：使用tushare下载最新A股数据
2. **扫描买点**：扫描全市场符合策略条件的买点信号
3. **扫描卖点**：扫描已持仓股票的卖点信号（止损/止盈/技术卖点）

## 快速开始

### 1. 只下载数据

```bash
python3 daily_select_pipeline.py download
```

### 2. 扫描买点

```bash
python3 daily_select_pipeline.py buy --strategy RS10_A1 --pool 大蓝筹
```

### 3. 扫描卖点

```bash
# 使用默认持仓文件 portfolio.json
python3 daily_select_pipeline.py sell --strategy RS10_A1

# 使用自定义持仓文件
python3 daily_select_pipeline.py sell --strategy RS10_A1 --portfolio my_portfolio.json
```

### 4. 同时扫描买点和卖点（推荐）

```bash
python3 daily_select_pipeline.py scan --strategy RS10_A1 --pool 大蓝筹
```

## 持仓文件格式

`portfolio.json` 文件格式（按交易记录）：

```json
{
  "trades": [
    {
      "trade_id": "T001",
      "code": "000001",
      "name": "平安银行",
      "buy_date": "2024-01-15",
      "buy_price": 10.50,
      "shares": 1000,
      "strategy": "RS10_A1",
      "status": "holding",
      "sell_date": null,
      "sell_price": null,
      "sell_reason": null,
      "profit_pct": null,
      "notes": "首笔买入"
    },
    {
      "trade_id": "T002",
      "code": "000001",
      "name": "平安银行",
      "buy_date": "2024-02-20",
      "buy_price": 10.20,
      "shares": 500,
      "strategy": "RS10_A1",
      "status": "holding",
      "sell_date": null,
      "sell_price": null,
      "sell_reason": null,
      "profit_pct": null,
      "notes": "加仓"
    }
  ]
}
```

**字段说明：**
- `trade_id`: 必填，唯一交易ID
- `code`: 必填，股票代码
- `buy_price`: 必填，买入价格
- `shares`: 必填，持仓数量
- `strategy`: 必填，买入策略（卖点扫描时必须匹配）
- `status`: 必填，holding=持仓中，sold=已卖出

**支持分批加仓/减仓：**
- 同一股票可以有多笔交易记录
- 每笔交易独立计算止损/止盈
- 每笔交易有不同的成本价

## 卖点检测逻辑

### 1. 基于策略条件的卖点（不需要成本价）

- zhixing_dead_cross（知行死叉）
- rsi_overbought（RSI超买）
- volume_anomaly（放量出货）

### 2. 基于盈亏的卖点（需要成本价）

- 止损：亏损达到止损线（如-25%）
- 止盈：盈利达到止盈线（如+80%）
- 移动止损：从最高点回撤达到阈值（如-35%）

## 输出示例

```
📊 买点信号: 2 个
============================================================
股票代码     买入日期       买入价 原因
------------------------------------------------------------
000001     2024-03-01      10.50 恐慌后狙击
600036     2024-03-01      35.20 恐慌后狙击

📊 卖点信号: 1 个
================================================================================
股票代码     买入日期       买入价     卖出价     盈亏%  持仓天 原因
--------------------------------------------------------------------------------
000001     2024-01-15      10.50      12.80   🟢+21.90%     45 止盈+80%
```

## 信号文件

每次扫描会保存信号到 `daily_signals/` 目录：

```
daily_signals/
├── 20240301_RS10_A1.json
├── 20240302_RS10_A1.json
└── ...
```

## 每日工作流

### 上班族推荐流程

1. **早上9:00**：运行扫描
   ```bash
   python3 daily_select_pipeline.py scan --strategy RS10_A1 --pool 大蓝筹
   ```

2. **查看信号**：
   - 买点信号：考虑是否买入
   - 卖点信号：考虑是否卖出

3. **更新持仓**：
   - 买入后更新 `portfolio.json`
   - 卖出后从 `portfolio.json` 删除

### 自动化（可选）

使用cron定时运行：

```bash
# 每天早上9:30运行
30 9 * * 1-5 cd /Users/flybirp/Documents/quant_backtest && python3 daily_select_pipeline.py scan --strategy RS10_A1 --pool 大蓝筹
```

## 支持的策略

所有 `strategies/rule/` 目录下的策略都支持，例如：

- RS10_A1
- RS10_WF
- DS2
- D1-知行金叉回踩-稳健
- E5-绝地逢生-宽松版

## 注意事项

1. **数据更新**：首次运行需要下载完整数据（约30分钟）
2. **增量更新**：后续运行只下载最近30天数据（约2分钟）
3. **持仓维护**：需要手动维护 `portfolio.json` 文件
4. **信号确认**：扫描结果仅供参考，需人工确认后执行
