#!/usr/bin/env python3
"""分析新增策略结果并排序（使用修正后的report.py逻辑）"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from analytics.risk import max_drawdown, calmar_ratio, sharpe_ratio
from analytics.performance import total_return, annual_return
from analytics.common import to_equity_df
from report import _build_equity_by_return

def _fix_signal_equity_curve(data: dict) -> dict:
    """修正signal模式下的权益曲线数据（基于日收益率重新计算）"""
    trades = data.get("trades", [])
    equity_curve = data.get("equity_curve", [])
    
    # 检测是否需要修正
    needs_fix = False
    
    # 1. 检查是否有cum_return_pct < -100%的异常
    has_anomaly = any(
        e.get("cum_return_pct", 0) < -100
        for e in equity_curve
    )
    if has_anomaly:
        needs_fix = True
    
    # 2. 检查是否有同一天多个数据点的情况
    if not needs_fix and equity_curve:
        dates = [e.get("date", "") for e in equity_curve]
        if len(dates) != len(set(dates)):
            needs_fix = True
    
    # 不需要修正
    if not needs_fix or not trades:
        return data
    
    # 使用基于日收益率的方法重新计算权益曲线
    summary = data.get("summary", {})
    initial_capital = summary.get("initial_capital", 100000)
    new_equity_curve = _build_equity_by_return(trades, initial_capital)
    
    # 如果计算成功，更新数据
    if new_equity_curve:
        data["equity_curve"] = new_equity_curve
    
    return data

def calculate_score(summary, equity, trades):
    """根据"活着赚钱视角"计算策略评分"""
    
    initial_capital = summary.get("initial_capital", 100000)
    
    # 使用analytics函数计算修正后的指标
    total_ret = total_return(equity, initial_capital)
    annual_ret = annual_return(equity, initial_capital)
    dd_result = max_drawdown(equity) if equity else (0, "", "", 0)
    max_dd = dd_result[0]
    shp = sharpe_ratio(equity, initial_capital)
    calmar = calmar_ratio(equity, initial_capital)
    
    # 其他指标
    total_trades = summary.get("total_trades", 0)
    win_rate = summary.get("win_rate", 0)
    profit_loss_ratio = summary.get("profit_loss_ratio", 0)
    expected_value = summary.get("expected_value", 0)
    avg_hold_days = summary.get("avg_hold_days", 0)
    
    # 计算EV/MaxDD
    ev_dd = expected_value / max_dd if max_dd > 0 else 0
    
    # 计算月均交易数 (12年数据)
    years = 12
    monthly_trades = total_trades / (years * 12) if total_trades > 0 else 0
    
    # 计算胜率×盈亏比
    win_pl = win_rate * profit_loss_ratio / 100 if profit_loss_ratio > 0 else 0
    
    # 评分公式
    score = 0
    
    # Calmar: 30分
    score += min(calmar / 15 * 30, 30)
    
    # EV/MaxDD: 25分
    score += min(max(ev_dd, 0) / 1 * 25, 25)
    
    # 胜率×盈亏比: 10分
    score += min(win_pl / 1 * 10, 10)
    
    # 月均交易: 10分（越少越好）
    if monthly_trades <= 20:
        score += 10
    elif monthly_trades <= 50:
        score += 7
    else:
        score += 3
    
    # 夏普: 25分
    score += min(max(shp, 0) / 1.5 * 25, 25)
    
    return {
        'score': round(score, 2),
        'total_return_pct': round(total_ret, 2),
        'annual_return_pct': round(annual_ret, 2),
        'max_drawdown_pct': round(max_dd, 2),
        'sharpe_ratio': round(shp, 2),
        'calmar': round(calmar, 4),
        'ev_dd': round(ev_dd, 4),
        'monthly_trades': round(monthly_trades, 2),
        'win_pl': round(win_pl, 4),
        'total_trades': total_trades,
        'win_rate': win_rate,
        'profit_loss_ratio': profit_loss_ratio,
        'expected_value': expected_value,
        'avg_hold_days': avg_hold_days
    }

def main():
    results_dir = Path("results")
    
    # 新增的策略文件
    new_strategies = [
        "E2-深跌反弹_大蓝筹.json",
        "E3-绝地金叉_大蓝筹.json",
        "E4-绝地逢生-完整版_大蓝筹.json",
        "E5-绝地逢生-宽松版_大蓝筹.json",
        "F1-金叉回踩分批-v2_大蓝筹.json",
        "F1-金叉回踩分批_大蓝筹.json",
        "F1-金叉回踩分批升级_大蓝筹.json",
        "F2-金叉回踩前高止盈_大蓝筹.json",
        "F3-确定性回踩-r12_大蓝筹.json",
        "F3-确定性回踩-v2_大蓝筹.json",
        "F3-确定性回踩_大蓝筹.json",
        "G1-MA金叉回踩分批_大蓝筹.json",
        "G2-MA金叉回踩前高止盈-v2_大蓝筹.json",
        "G2-MA金叉回踩前高止盈_大蓝筹.json",
        "G3-MA确定性回踩-最佳_大蓝筹.json",
        "H1-zhixing快线MA60回踩分批_大蓝筹.json",
        "H2-zhixing快线MA60回踩前高止盈-v2_大蓝筹.json",
        "H2-zhixing快线MA60回踩前高止盈_大蓝筹.json",
        "H3-zhixing快线MA60确定性回踩-最佳_大蓝筹.json"
    ]
    
    strategies = []
    
    for filename in new_strategies:
        filepath = results_dir / filename
        if filepath.exists():
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                # 修正权益曲线
                data = _fix_signal_equity_curve(data)
                
                summary = data.get("summary", {})
                equity = data.get("equity_curve", [])
                trades = data.get("trades", [])
                
                # 计算评分（使用修正后的数据）
                scores = calculate_score(summary, equity, trades)
                
                strategies.append({
                    "filename": filename,
                    "strategy": summary.get("strategy", ""),
                    **scores
                })
            except Exception as e:
                print(f"Error reading {filename}: {e}")
    
    # 按评分排序
    strategies.sort(key=lambda x: x["score"], reverse=True)
    
    # 输出结果
    print("\n" + "=" * 140)
    print("新增策略效果排序 (修正后，按照'活着赚钱视角'评分)")
    print("=" * 140)
    print(f"{'排名':<4} {'策略名称':<35} {'评分':<8} {'总收益%':<10} {'年化%':<8} {'最大回撤%':<10} {'夏普':<8} {'Calmar':<8} {'交易数':<6} {'胜率%':<7} {'盈亏比':<7} {'月均交易':<8}")
    print("-" * 140)
    
    for i, s in enumerate(strategies, 1):
        print(f"{i:<4} {s['strategy']:<35} {s['score']:<8} {s['total_return_pct']:<10} {s['annual_return_pct']:<8} {s['max_drawdown_pct']:<10} {s['sharpe_ratio']:<8} {s['calmar']:<8} {s['total_trades']:<6} {s['win_rate']:<6.1f}% {s['profit_loss_ratio']:<6.2f} {s['monthly_trades']:<8}")
    
    print("\n" + "=" * 140)
    print("评分维度说明:")
    print("- Calmar比率 (30分): CAGR / MaxDD, 每承受1%回撤的年化收益")
    print("- 期望结构 (25分): EV / MaxDD, 每承受1%回撤的期望收益")
    print("- 夏普比率 (25分): 风险调整收益")
    print("- 可执行性 (20分): 月均交易数, 散户月均≤2笔最佳")
    print("=" * 140)

if __name__ == "__main__":
    main()