# 量化回测系统

## 架构
- **后端**: FastAPI (Python)，端口 8200
- **前端**: React + TypeScript + Recharts，端口 3000
- **数据**: /Users/flybirp/Documents/mainland_data (5062只A股日K CSV)

## 快速启动

### 一键启动（前后端同时）
```bash
bash start.sh
```

### 分别启动
```bash
# 后端
bash start.sh backend

# 前端
bash start.sh frontend
```

## 策略配置

策略以 JSON/YAML 格式存储在 `strategies/` 目录。核心字段：

| 字段 | 说明 | 默认值 |
|------|------|--------|
| name | 策略名称 | "default" |
| k_type | K线类型: daily/weekly | daily |
| initial_capital | 初始资金 | 100000 |
| buy_groups | 买入条件组（组间OR，组内AND） | [] |
| sell_groups | 卖出条件组 | [] |
| position_pct | 单只仓位比例 | 1.0 |
| max_positions | 最大持仓数 | 5 |
| stop_loss_pct | 止损线 % | 5 |
| take_profit_pct | 止盈线 % | 15 |
| trailing_stop_pct | 移动止损 % | 0 |
| stock_pool | 股票池（空=全市场） | [] |

### 可用指标

**均线类**: ma_above, ma_below, ma_cross_up, ma_cross_down, price_above_ma, price_below_ma, ma_bullish_alignment, ma_bearish_alignment

**MACD**: macd_golden_cross, macd_dead_cross, macd_above_zero, macd_below_zero

**RSI**: rsi_oversold, rsi_overbought

**KDJ**: kdj_golden_cross, kdj_dead_cross, kdj_j_oversold, kdj_j_overbought

**布林带**: bb_lower_touch, bb_upper_touch, bb_mid_break

**成交量**: volume_breakout, volume_shrink

**趋势**: new_high, new_low, consecutive_up, consecutive_down

**涨跌幅**: pct_change_gt, pct_change_lt

### 策略示例

```json
{
  "name": "双均线MACD策略",
  "k_type": "daily",
  "initial_capital": 100000,
  "buy_groups": [
    {
      "conditions": [
        {"indicator": "ma_cross_up", "params": {"fast": 5, "slow": 20}},
        {"indicator": "macd_above_zero"}
      ]
    }
  ],
  "sell_groups": [
    {
      "conditions": [
        {"indicator": "ma_cross_down", "params": {"fast": 5, "slow": 20}}
      ]
    }
  ],
  "position_pct": 0.3,
  "max_positions": 5,
  "stop_loss_pct": 5,
  "take_profit_pct": 15,
  "trailing_stop_pct": 8
}
```

## API

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | /api/stocks | 列出所有股票 |
| GET | /api/strategies | 列出所有策略 |
| GET | /api/strategies/{name} | 获取策略详情 |
| POST | /api/strategies | 保存策略 |
| DELETE | /api/strategies/{name} | 删除策略 |
| POST | /api/backtest | 启动回测 |
| GET | /api/backtest/{task_id} | 获取回测结果 |
| GET | /api/results | 列出所有回测结果 |
