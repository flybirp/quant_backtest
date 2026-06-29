"""
Research Agent 策略优化器
结合 quant_discover 的数据发现，自动优化策略参数

用法:
    python3 run_research_optimize.py D1-知行金叉回踩-稳健 --rounds 5
"""

import json
import sys
import time
import yaml
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from backend.main import _config_from_dict, _resolve_pool_codes
from backend.backtest_engine import run_backtest
from backend.research_agent import (
    SOTATracker,
    build_action_selection_context,
    build_hypothesis_context,
    ACTION_SELECTION_SYSTEM,
    HYPOTHESIS_SYSTEM_PROMPT,
)

BASE = Path(__file__).parent
RESULTS_DIR = BASE / "results"
STRATEGIES_DIR = BASE / "strategies" / "rule"

# ============================================================
# quant_discover 的关键发现
# ============================================================

QUANT_DISCOVER_INSIGHTS = """
## 来自 quant_discover 项目的关键发现

### 发现一：主升浪波段的回撤-涨幅反向关系
- 回撤 0%-5% 的波段：涨幅中位 378%，96.9% 涨幅超100%
- 回撤 45%-50% 的波段：涨幅中位仅 112%，胜率仅 57%
- **结论："安静的上涨"优于"刺激的上涨"**

### 发现二：超大波段启动前的量价共性
- 启动前最后30天跌幅占总跌幅的68%（末期杀跌）
- 启动前全程缩量，超大波段的绝对量能始终最低
- **结论：波段内回撤要小，但波段启动前的杀跌要深**

### 发现三：超大波段启动时的大盘环境
- 中证1000在20天内跌超15%时，是超大波段批量出现的环境
- **结论：恐慌越深、越集中，后续走出超大波段的概率越高**

### 发现六：知行金叉波段内的回踩规律
- 第1次回踩 zhixing_slow 后回升率 93.7%，创新高率 79.8%
- 金叉后20天内回踩是最佳买点：回升率 96-97%
- 浅破 zhixing_slow（跌破5%以内）后回升的创新高率反而更高（99.3%）
- **结论：第1次回踩 zhixing_slow 是最佳买点，跌破5%以内是安全区**

### 发现七：横盘震荡后的突破方向
- PE<15 的向下突破概率仅 6.3%，PE>60 的高达 28.3%
- 最佳组合：PE<25 + 量比>1.2，向上突破概率是向下的 6.5 倍
- **结论：低估值横盘后期放量是向上突破的强信号**

## 散户心理约束
- 止损超过 20% 心态会崩
- 最佳止损点：12%（夏普最高）
- 持仓时间越长，收益和胜率越高
"""

# ============================================================
# 优化配置
# ============================================================

OPTIMIZATION_CONFIG = {
    "D1-知行金叉回踩": {
        "base_strategy": "D1-知行金叉回踩-稳健",
        "focus": "知行金叉回踩策略",
        "key_params": ["stop_loss_pct", "take_profit_pct", "trailing_stop_pct", "max_hold_days"],
        "insights": [
            "第1次回踩 zhixing_slow 是最佳买点",
            "跌破5%以内创新高率99.3%",
            "止损12%是最佳平衡点",
        ],
    },
    "D2-恐慌底部反转": {
        "base_strategy": "D2-恐慌底部反转-稳健",
        "focus": "恐慌底部反转策略",
        "key_params": ["stop_loss_pct", "take_profit_pct", "trailing_stop_pct", "max_hold_days"],
        "insights": [
            "中证1000暴跌时是超大波段批量出现的环境",
            "恐慌越深、越集中，后续走出超大波段的概率越高",
            "止损15%是散户心理承受上限",
        ],
    },
    "D3-安静上涨": {
        "base_strategy": "D3-安静上涨-稳健",
        "focus": "安静上涨策略",
        "key_params": ["stop_loss_pct", "take_profit_pct", "trailing_stop_pct", "max_hold_days"],
        "insights": [
            "回撤0%-5%的波段涨幅中位378%",
            "安静的上涨优于刺激的上涨",
            "止损12%是最佳平衡点",
        ],
    },
}


def run_backtest_with_metrics(strategy_name: str, pool_name: str = "大蓝筹") -> dict:
    """运行回测并返回指标"""
    ypath = STRATEGIES_DIR / f"{strategy_name}.yaml"
    if not ypath.exists():
        return {"error": f"策略文件不存在: {ypath}"}

    with open(ypath) as f:
        cfg = yaml.safe_load(f)

    pool = _resolve_pool_codes(pool_name)
    cfg["stock_pool"] = pool

    r = run_backtest(_config_from_dict(cfg))

    return {
        "ev": r.expected_value,
        "win_rate": r.win_rate,
        "max_dd": r.max_drawdown_pct,
        "sharpe": r.sharpe_ratio,
        "trades": r.total_trades,
        "annual": r.annual_return_pct,
        "profit_loss_ratio": r.profit_loss_ratio,
        "avg_hold_days": r.avg_hold_days,
    }


def generate_optimization_prompt(
    strategy_name: str,
    current_metrics: dict,
    sota: SOTATracker,
    past_rounds: list,
    direction: str,
) -> str:
    """生成优化提示词"""

    config = OPTIMIZATION_CONFIG.get(strategy_name, {})
    insights = config.get("insights", [])

    prompt = f"""
# 策略优化任务

## 策略名称
{strategy_name}

## 当前指标
- EV: {current_metrics.get('ev', 'N/A')}%
- 胜率: {current_metrics.get('win_rate', 'N/A')}%
- 最大回撤: {current_metrics.get('max_dd', 'N/A')}%
- 夏普比率: {current_metrics.get('sharpe', 'N/A')}
- 交易数: {current_metrics.get('trades', 'N/A')}
- 年化收益: {current_metrics.get('annual', 'N/A')}%
- 盈亏比: {current_metrics.get('profit_loss_ratio', 'N/A')}
- 平均持仓天数: {current_metrics.get('avg_hold_days', 'N/A')}

## SOTA（最佳记录）
- EV: {sota.best_ev:.2f}%
- 轮次: {sota.round}
- 当前方向: {sota.current_direction or '无'}
- 连续失败: {sota.consecutive_failures}

## 优化方向
{direction}

## quant_discover 关键发现
{chr(10).join(f'- {insight}' for insight in insights)}

## 散户约束
- 止损 ≤ 15%（心理承受上限）
- 最佳止损点：12%（夏普最高）
- 持仓时间越长，收益和胜率越高

## 任务
请根据以上信息，生成下一步的参数调整建议。

要求：
1. 止损必须 ≤ 15%
2. 优先调整止损和止盈参数
3. 考虑 quant_discover 的发现
4. 给出具体的参数值

输出JSON格式：
{{
  "hypothesis": "如果我们将<X>从<旧值>改为<新值>，<预期效果>",
  "reason": "为什么这个调整能解决当前问题",
  "parameter_changes": {{
    "param_name": new_value
  }},
  "expected_effects": {{
    "ev_change": "+X% to +Y%",
    "max_dd_change": "+X% to -Y%",
    "win_rate_change": "+X% to -Y%"
  }},
  "confidence": "high | medium | low"
}}
"""
    return prompt


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Research Agent 策略优化器")
    ap.add_argument("strategy", help="策略名称（如 D1-知行金叉回踩）")
    ap.add_argument("--rounds", type=int, default=5, help="优化轮次")
    ap.add_argument("--pool", default="大蓝筹", help="股票池")
    args = ap.parse_args()

    strategy_name = args.strategy
    config = OPTIMIZATION_CONFIG.get(strategy_name)

    if not config:
        print(f"❌ 未找到策略配置: {strategy_name}")
        print(f"可用策略: {', '.join(OPTIMIZATION_CONFIG.keys())}")
        return

    base_strategy = config["base_strategy"]
    print(f"\n{'='*60}")
    print(f"  Research Agent 策略优化")
    print(f"  策略: {strategy_name}")
    print(f"  基础策略: {base_strategy}")
    print(f"  优化轮次: {args.rounds}")
    print(f"{'='*60}\n")

    # 初始化 SOTA tracker
    sota = SOTATracker()
    past_rounds = []

    # 运行基础策略
    print(f"[1/{args.rounds+1}] 运行基础策略: {base_strategy}")
    current_metrics = run_backtest_with_metrics(base_strategy, args.pool)

    if "error" in current_metrics:
        print(f"❌ 错误: {current_metrics['error']}")
        return

    print(f"  EV={current_metrics['ev']:+.2f}%  Win={current_metrics['win_rate']:.1f}%  "
          f"DD={current_metrics['max_dd']:.1f}%  Sharpe={current_metrics['sharpe']:.2f}")

    # 优化循环
    for round_num in range(1, args.rounds + 1):
        print(f"\n[{round_num+1}/{args.rounds+1}] 优化轮次 {round_num}")

        # 确定优化方向
        if sota.should_switch_direction or not sota.current_direction:
            direction = "explore_new_params"
            sota.current_direction = direction
        else:
            direction = sota.current_direction

        print(f"  优化方向: {direction}")

        # 生成优化提示词
        prompt = generate_optimization_prompt(
            strategy_name,
            current_metrics,
            sota,
            past_rounds,
            direction,
        )

        # 这里应该调用 LLM 生成参数调整
        # 由于没有 LLM API，我们使用基于规则的优化
        parameter_changes = generate_rule_based_changes(
            strategy_name, current_metrics, sota, direction
        )

        if not parameter_changes:
            print("  ⚠️ 无法生成有效的参数调整，跳过本轮")
            past_rounds.append({
                "round": round_num,
                "direction": direction,
                "decision": False,
                "ev": current_metrics["ev"],
            })
            sota.consecutive_failures += 1
            continue

        # 应用参数变化
        new_strategy_name = f"{base_strategy}-r{round_num}"
        print(f"  参数调整: {parameter_changes}")

        # 创建新策略文件
        create_strategy_variant(base_strategy, new_strategy_name, parameter_changes)

        # 运行新策略
        new_metrics = run_backtest_with_metrics(new_strategy_name, args.pool)

        if "error" in new_metrics:
            print(f"  ❌ 错误: {new_metrics['error']}")
            past_rounds.append({
                "round": round_num,
                "direction": direction,
                "decision": False,
                "ev": current_metrics["ev"],
            })
            sota.consecutive_failures += 1
            continue

        print(f"  新指标: EV={new_metrics['ev']:+.2f}%  Win={new_metrics['win_rate']:.1f}%  "
              f"DD={new_metrics['max_dd']:.1f}%  Sharpe={new_metrics['sharpe']:.2f}")

        # 更新 SOTA
        accepted, reason = sota.update(
            ev=new_metrics["ev"],
            max_dd=new_metrics["max_dd"],
            ci_low=new_metrics["ev"] - 2,  # 简化：使用 EV ± 2 作为 CI
            config=parameter_changes,
            metrics=new_metrics,
            direction=direction,
            previous_metrics=current_metrics,
        )

        print(f"  决策: {reason}")

        if accepted:
            current_metrics = new_metrics
            print(f"  ✅ 接受新策略")

        past_rounds.append({
            "round": round_num,
            "direction": direction,
            "decision": accepted,
            "ev": new_metrics["ev"],
        })

    # 输出最终结果
    print(f"\n{'='*60}")
    print(f"  优化完成")
    print(f"{'='*60}")
    print(f"  最终 EV: {current_metrics['ev']:+.2f}%")
    print(f"  最终胜率: {current_metrics['win_rate']:.1f}%")
    print(f"  最终回撤: {current_metrics['max_dd']:.1f}%")
    print(f"  最终夏普: {current_metrics['sharpe']:.2f}")
    print(f"  SOTA EV: {sota.best_ev:+.2f}%")
    print(f"  总轮次: {sota.round}")


def generate_rule_based_changes(
    strategy_name: str,
    current_metrics: dict,
    sota: SOTATracker,
    direction: str,
) -> dict:
    """基于规则生成参数调整（替代 LLM）"""

    changes = {}
    ev = current_metrics.get("ev", 0)
    win_rate = current_metrics.get("win_rate", 0)
    max_dd = current_metrics.get("max_dd", 50)
    sharpe = current_metrics.get("sharpe", 0)

    # 读取当前策略配置
    config = OPTIMIZATION_CONFIG.get(strategy_name, {})
    base_strategy = config.get("base_strategy", "")
    ypath = STRATEGIES_DIR / f"{base_strategy}.yaml"

    if not ypath.exists():
        return {}

    with open(ypath) as f:
        cfg = yaml.safe_load(f)

    current_stop_loss = cfg.get("stop_loss_pct", 12)
    current_take_profit = cfg.get("take_profit_pct", 25)
    current_trailing = cfg.get("trailing_stop_pct", 18)
    current_max_hold = cfg.get("max_hold_days", 180)

    # 基于方向的参数调整
    if direction == "reduce_drawdown":
        # 减少回撤：收紧止损
        if current_stop_loss > 8:
            changes["stop_loss_pct"] = max(8, current_stop_loss - 2)
        if current_trailing > 12:
            changes["trailing_stop_pct"] = max(12, current_trailing - 3)

    elif direction == "improve_win_rate":
        # 提高胜率：降低止盈，更快锁定利润
        if current_take_profit > 15:
            changes["take_profit_pct"] = max(15, current_take_profit - 5)
        if current_trailing > 12:
            changes["trailing_stop_pct"] = max(12, current_trailing - 2)

    elif direction == "increase_signals":
        # 增加信号：放宽条件
        # 这个方向不适合简单参数调整
        pass

    elif direction == "reduce_underwater":
        # 减少水下时间：加速退出
        if current_max_hold > 90:
            changes["max_hold_days"] = max(90, current_max_hold - 30)
        if current_trailing > 12:
            changes["trailing_stop_pct"] = max(12, current_trailing - 2)

    elif direction == "lower_turnover":
        # 降低换手率：延长持仓
        if current_max_hold < 365:
            changes["max_hold_days"] = min(365, current_max_hold + 30)

    elif direction == "explore_new_params":
        # 探索新参数：随机调整
        import random
        param = random.choice(["stop_loss_pct", "take_profit_pct", "trailing_stop_pct"])
        if param == "stop_loss_pct":
            changes["stop_loss_pct"] = max(8, min(15, current_stop_loss + random.randint(-2, 2)))
        elif param == "take_profit_pct":
            changes["take_profit_pct"] = max(15, min(50, current_take_profit + random.randint(-5, 5)))
        elif param == "trailing_stop_pct":
            changes["trailing_stop_pct"] = max(12, min(25, current_trailing + random.randint(-3, 3)))

    # 确保止损不超过15%
    if "stop_loss_pct" in changes:
        changes["stop_loss_pct"] = min(15, changes["stop_loss_pct"])

    return changes


def create_strategy_variant(base_name: str, new_name: str, changes: dict):
    """创建策略变体"""
    base_path = STRATEGIES_DIR / f"{base_name}.yaml"
    new_path = STRATEGIES_DIR / f"{new_name}.yaml"

    with open(base_path) as f:
        cfg = yaml.safe_load(f)

    # 应用参数变化
    for key, value in changes.items():
        if key in cfg:
            cfg[key] = value

    # 更新名称
    cfg["name"] = new_name

    with open(new_path, "w") as f:
        yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False)


if __name__ == "__main__":
    main()
