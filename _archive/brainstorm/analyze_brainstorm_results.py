"""分析全市场脑暴策略回测结果 — 生成可视化HTML报告"""

import json
from pathlib import Path

def load_results(path="brainstorm_results_fullmarket.json"):
    with open(path) as f:
        return json.load(f)

def generate_report(results):
    # Sort by expected_value descending
    results.sort(key=lambda x: x["expected_value"], reverse=True)

    # Classification
    for r in results:
        ev = r["expected_value"]
        trades = r["total_trades"]
        if trades == 0:
            r["verdict"] = "无交易"
            r["verdict_class"] = "none"
        elif ev > 1.0:
            r["verdict"] = "✅ 优秀"
            r["verdict_class"] = "excellent"
        elif ev > 0:
            r["verdict"] = "✅ 正期望"
            r["verdict_class"] = "positive"
        elif ev > -1:
            r["verdict"] = "⚠️ 微负"
            r["verdict_class"] = "marginal"
        else:
            r["verdict"] = "❌ 负期望"
            r["verdict_class"] = "negative"

    rows_html = ""
    for rank, r in enumerate(results, 1):
        cls = r["verdict_class"]
        rows_html += f"""
        <tr class="{cls}">
            <td>{rank}</td>
            <td><strong>{r['name']}</strong></td>
            <td>{r['total_trades']}</td>
            <td>{r['win_rate']:.1f}%</td>
            <td>{r['profit_loss_ratio']:.2f}</td>
            <td class="ev">{r['expected_value']:+.2f}%</td>
            <td>{r['avg_profit_pct']:.2f}%</td>
            <td>{r['avg_loss_pct']:.2f}%</td>
            <td>{r['avg_hold_days']:.1f}d</td>
            <td>{r['max_profit_pct']:.2f}%</td>
            <td>{r['max_loss_pct']:.2f}%</td>
            <td>{r['verdict']}</td>
        </tr>"""

    # Summary stats
    positive_ev = [r for r in results if r["expected_value"] > 0 and r["total_trades"] > 0]
    negative_ev = [r for r in results if r["expected_value"] <= 0 and r["total_trades"] > 0]
    no_trades = [r for r in results if r["total_trades"] == 0]

    # Best strategy
    best = results[0] if results else None

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>全市场脑暴策略回测报告</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 20px; background: #0d1117; color: #c9d1d9; }}
h1 {{ color: #58a6ff; }}
h2 {{ color: #8b949e; margin-top: 30px; }}
.summary {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin: 20px 0; }}
.card {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 15px; text-align: center; }}
.card .value {{ font-size: 28px; font-weight: bold; margin: 5px 0; }}
.card .label {{ color: #8b949e; font-size: 13px; }}
.card.green .value {{ color: #3fb950; }}
.card.red .value {{ color: #f85149; }}
.card.blue .value {{ color: #58a6ff; }}
.card.yellow .value {{ color: #d29922; }}
table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
th {{ background: #161b22; color: #58a6ff; padding: 10px 8px; text-align: center; font-size: 13px; border-bottom: 2px solid #30363d; }}
td {{ padding: 8px; text-align: center; border-bottom: 1px solid #21262d; font-size: 13px; }}
tr:hover {{ background: #161b22; }}
tr.excellent {{ background: rgba(63,185,80,0.1); }}
tr.positive {{ background: rgba(63,185,80,0.05); }}
tr.marginal {{ background: rgba(210,153,34,0.05); }}
tr.negative {{ background: rgba(248,81,73,0.05); }}
tr.none {{ opacity: 0.5; }}
td.ev {{ font-weight: bold; font-size: 14px; }}
tr.excellent td.ev {{ color: #3fb950; }}
tr.positive td.ev {{ color: #56d364; }}
tr.marginal td.ev {{ color: #d29922; }}
tr.negative td.ev {{ color: #f85149; }}
.best-box {{ background: #161b22; border: 2px solid #3fb950; border-radius: 10px; padding: 20px; margin: 20px 0; }}
.best-box h3 {{ color: #3fb950; margin-top: 0; }}
.insight {{ background: #161b22; border-left: 3px solid #58a6ff; padding: 12px 15px; margin: 10px 0; font-size: 14px; }}
</style>
</head>
<body>
<h1>📊 全市场脑暴策略回测报告</h1>
<p style="color:#8b949e;">5062只A股 | 2015-01至今 | 18个策略 | 信号模式</p>

<div class="summary">
    <div class="card green">
        <div class="value">{len(positive_ev)}</div>
        <div class="label">正期望策略</div>
    </div>
    <div class="card red">
        <div class="value">{len(negative_ev)}</div>
        <div class="label">负期望策略</div>
    </div>
    <div class="card yellow">
        <div class="value">{len(no_trades)}</div>
        <div class="label">无交易策略</div>
    </div>
    <div class="card blue">
        <div class="value">{best['expected_value']:+.2f}%</div>
        <div class="label">最佳期望值</div>
    </div>
</div>

{''.join(f'''<div class="insight">💡 {insight}</div>''' for insight in _generate_insights(results))}

<div class="best-box">
    <h3>🏆 最佳策略: {best['name']}</h3>
    <p>交易数: {best['total_trades']} | 胜率: {best['win_rate']:.1f}% | 盈亏比: {best['profit_loss_ratio']:.2f} | 期望值: {best['expected_value']:+.2f}%</p>
    <p>均盈: {best['avg_profit_pct']:.2f}% | 均亏: {best['avg_loss_pct']:.2f}% | 均持仓: {best['avg_hold_days']:.1f}天</p>
</div>

<h2>完整排名（按期望值降序）</h2>
<table>
<tr>
    <th>#</th><th>策略</th><th>交易数</th><th>胜率</th><th>盈亏比</th>
    <th>期望%</th><th>均盈%</th><th>均亏%</th><th>均持仓</th>
    <th>最大盈%</th><th>最大亏%</th><th>判读</th>
</tr>
{rows_html}
</table>

<h2>策略分类对比</h2>
{_category_comparison_html(results)}

</body>
</html>"""
    return html


def _generate_insights(results):
    insights = []
    positive = [r for r in results if r["expected_value"] > 0 and r["total_trades"] > 0]
    negative = [r for r in results if r["expected_value"] <= 0 and r["total_trades"] > 0]

    if positive:
        avg_wr = sum(r["win_rate"] for r in positive) / len(positive)
        insights.append(f"正期望策略共{len(positive)}个，平均胜率{avg_wr:.1f}%")
        # Conservative vs Aggressive
        cons = [r for r in positive if "保守" in r["name"]]
        aggr = [r for r in positive if "激进" in r["name"]]
        if cons and aggr:
            cons_ev = sum(r["expected_value"] for r in cons) / len(cons)
            aggr_ev = sum(r["expected_value"] for r in aggr) / len(aggr)
            better = "保守" if cons_ev > aggr_ev else "激进"
            insights.append(f"保守型平均期望{cons_ev:+.2f}% vs 激进型{aggr_ev:+.2f}% → {better}型更优")

    # Wyckoff vs Pocket Pivot
    wyckoff = [r for r in results if r["name"].startswith("V") and r["total_trades"] > 0]
    pocket = [r for r in results if r["name"][0] in "ABCD" and r["total_trades"] > 0]
    if wyckoff and pocket:
        w_ev = sum(r["expected_value"] for r in wyckoff) / len(wyckoff)
        p_ev = sum(r["expected_value"] for r in pocket) / len(pocket)
        better = "Wyckoff" if w_ev > p_ev else "口袋支点"
        insights.append(f"Wyckoff系平均期望{w_ev:+.2f}% vs 口袋支点系{p_ev:+.2f}% → {better}系更优")

    # High win rate insight
    high_wr = [r for r in results if r["win_rate"] > 55 and r["total_trades"] > 20]
    if high_wr:
        best_wr = max(high_wr, key=lambda r: r["win_rate"])
        insights.append(f"最高胜率策略: {best_wr['name']} ({best_wr['win_rate']:.1f}%)，交易{best_wr['total_trades']}笔")

    # Low trade count warning
    low_trades = [r for r in results if 0 < r["total_trades"] < 30]
    if low_trades:
        insights.append(f"⚠️ {len(low_trades)}个策略交易不足30笔，统计意义有限")

    return insights


def _category_comparison_html(results):
    # Group by category
    categories = {
        "V1-吸筹确认": [r for r in results if r["name"].startswith("V1")],
        "V2-缩量突破": [r for r in results if r["name"].startswith("V2")],
        "V3-量能前置": [r for r in results if r["name"].startswith("V3")],
        "V5-量价背离": [r for r in results if r["name"].startswith("V5")],
        "V6-弹簧确认": [r for r in results if r["name"].startswith("V6")],
        "A-口袋支点": [r for r in results if r["name"].startswith("A-")],
        "B-弹簧口袋": [r for r in results if r["name"].startswith("B-")],
        "C-因果口袋": [r for r in results if r["name"].startswith("C-")],
        "D-均线口袋": [r for r in results if r["name"].startswith("D-")],
    }

    rows = ""
    for cat, items in categories.items():
        if not items:
            continue
        avg_ev = sum(r["expected_value"] for r in items) / len(items)
        avg_wr = sum(r["win_rate"] for r in items) / len(items)
        total_trades = sum(r["total_trades"] for r in items)
        cls = "positive" if avg_ev > 0 else "negative"
        rows += f"""
        <tr class="{cls}">
            <td><strong>{cat}</strong></td>
            <td>{len(items)}</td>
            <td>{total_trades}</td>
            <td>{avg_wr:.1f}%</td>
            <td>{avg_ev:+.2f}%</td>
            <td>{items[0]['name']}</td>
        </tr>"""

    return f"""
    <table>
    <tr><th>分类</th><th>策略数</th><th>总交易数</th><th>平均胜率</th><th>平均期望%</th><th>代表策略</th></tr>
    {rows}
    </table>"""


if __name__ == "__main__":
    results = load_results()
    html = generate_report(results)
    out_path = Path("brainstorm_report_fullmarket.html")
    out_path.write_text(html, encoding="utf-8")
    print(f"报告已生成: {out_path.absolute()}")
