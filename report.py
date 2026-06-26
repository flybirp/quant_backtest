"""量化回测综合分析报告生成器

用法:
    python report.py results/zzh7.3_大蓝筹.json [results/zzh7.3_创业板.json ...]

    python report.py results/zzh7.3_大蓝筹.json --benchmark hs300
    python report.py results/zzh7.3_大蓝筹.json --walk-forward --strategy zzh7.3 --pool 大蓝筹
    python report.py results/zzh7.3_大蓝筹.json --scenario
    python report.py results/zzh7.3_大蓝筹.json --risk-free-rate 2.5
"""

import sys, json
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import numpy as np

BASE = Path(__file__).parent
RESULTS_DIR = BASE / "results"
REPORTS_DIR = BASE / "reports"
REPORTS_DIR.mkdir(exist_ok=True)
sys.path.insert(0, str(BASE))


class Tee:
    """Write to both stdout and a file simultaneously."""
    def __init__(self, filepath):
        self.file = open(filepath, "w", encoding="utf-8")
        self.stdout = sys.stdout
    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)
    def flush(self):
        self.stdout.flush()
        self.file.flush()
    def close(self):
        self.file.close()


def _find_strategy_yaml(name: str) -> Path | None:
    """查找策略 YAML 文件 (rule/ or ml/ subdirectories, or legacy flat)."""
    base = BASE / "strategies"
    for p in [
        base / "rule" / f"{name}.yaml",
        base / "ml" / f"{name}.yaml",
        base / f"{name}.yaml",
    ]:
        if p.exists():
            return p
    return None

# ── Analytics imports ──────────────────────────────────────────────
from analytics.performance import (
    total_return, annual_return, yearly_returns_table, monthly_returns_table,
    rolling_returns, cumulative_returns_series,
)
from analytics.risk import (
    max_drawdown, drawdown_periods, var_historical, var_daily,
    cvar_historical, cvar_daily, volatility_annualized, downside_deviation,
    sortino_ratio, calmar_ratio, max_consecutive_losses, max_consecutive_wins,
    profit_distribution_stats, ulcer_index, sharpe_ratio, drawdown_recovery_stats,
)
from analytics.statistics import (
    bootstrap_confidence_interval, bootstrap_ev_ci, t_test_mean, normality_test,
)
from analytics.benchmark import (
    compute_benchmark_returns, alpha_beta, information_ratio,
    tracking_error, excess_returns, compare_to_benchmark, bull_bear_analysis,
)
from analytics.attribution import (
    yearly_attribution, sell_reason_attribution, position_concentration,
    monthly_heatmap, sector_attribution,
)
from analytics.formatters import (
    format_summary_table, format_trade_distribution,
    format_attribution_table, format_validation_report,
)
from analytics.factors import capm_regression, factor_exposure_summary
from analytics.capacity import turnover_analysis, capacity_estimate
from analytics.tca import cost_attribution, estimated_cost_breakdown
from analytics.correlation import correlation_matrix
from analytics.scenario import scenario_stress_test, scenario_summary_table
from validation.monte_carlo import ruin_probability, equity_curve_simulation
from backend.survival_check import check_survivorship, format_survivorship_report

# ── Data loading ──────────────────────────────────────────────────

def load_results(*filenames):
    """加载一个或多个结果文件"""
    all_data = []
    for fn in filenames:
        path = Path(fn)
        if not path.exists() and not path.is_absolute():
            path = RESULTS_DIR / fn
        if not path.exists():
            print(f"  [WARN] 文件不存在: {path}")
            continue
        with open(path) as f:
            data = json.load(f)
        label = data["summary"]["strategy"]
        all_data.append((label, data))
    return all_data


# ── Report sections ───────────────────────────────────────────────

SEP = "=" * 100
SEP2 = "-" * 100


def print_header(strategy_name="", pool_name=""):
    print(f"\n{SEP}")
    print(f"  量化策略综合分析报告")
    if strategy_name:
        print(f"  策略: {strategy_name}  池: {pool_name}")
    print(f"  生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{SEP}")


# ── Section 0: 生存偏差检查 ─────────────────────────────────────

def print_survivorship_check():
    print(f"\n{'─' * 90} 零、生存偏差检查 {'─' * 90}\n")
    try:
        result = check_survivorship()
        print(format_survivorship_report(result))
    except Exception as e:
        print(f"  [WARN] 生存偏差检查失败: {e}")


# ── Section 12: Monte Carlo 验证 ─────────────────────────────────

def print_monte_carlo(all_data):
    print(f"\n{'─' * 90} 十二、Monte Carlo 验证 {'─' * 90}\n")

    for label, d in all_data:
        trades = d.get("trades", [])
        if not trades or len(trades) < 10:
            print(f"  {label}: 交易数不足 ({len(trades)}笔)，跳过 Monte Carlo")
            continue

        init_cap = d.get("summary", {}).get("initial_capital", 100000)
        print(f"\n  {label} ({len(trades)}笔交易, 初始资金{init_cap:,.0f}):")

        # 破产概率：回撤50%算破产
        ruin_50 = ruin_probability(trades, capital=init_cap, ruin_threshold_pct=50,
                                   n_simulations=5000, seed=42)
        # 破产概率：回撤30%算破产
        ruin_30 = ruin_probability(trades, capital=init_cap, ruin_threshold_pct=30,
                                   n_simulations=5000, seed=42)

        print(f"    破产概率 (回撤>50%): {ruin_50*100:.1f}%")
        print(f"    破产概率 (回撤>30%): {ruin_30*100:.1f}%")

        # 蒙特卡洛模拟统计
        curves = equity_curve_simulation(trades, n_simulations=1000,
                                         initial_capital=init_cap, seed=42)
        final_values = [c[-1] for c in curves]
        print(f"    1000次模拟终值: 中位数 {np.median(final_values):,.0f}")
        print(f"                    5%分位 {np.percentile(final_values, 5):,.0f}")
        print(f"                    95%分位 {np.percentile(final_values, 95):,.0f}")


# ── Section 0.5: 策略配置快照 ──────────────────────────────────

def print_config_snapshot(all_data):
    for label, d in all_data:
        cfg = d.get("summary", {}).get("config")
        if not cfg:
            continue
        print(f"\n{'─' * 90} 配置快照: {label} {'─' * 90}\n")
        print(f"  止损: {cfg.get('stop_loss_pct', '?')}%  "
              f"止盈: {cfg.get('take_profit_pct', '?')}%  "
              f"移动止损: {cfg.get('trailing_stop_pct', '?')}%  "
              f"最大持仓天数: {cfg.get('max_hold_days', '?')}")
        print(f"  买入价: {cfg.get('buy_price_type', '?')}  "
              f"卖出价: {cfg.get('sell_price_type', '?')}  "
              f"仓位: {cfg.get('position_pct', '?')}  "
              f"最大持仓: {cfg.get('max_positions', '?')}")
        print(f"  佣金: {cfg.get('commission_rate', '?')}  "
              f"印花税: {cfg.get('stamp_tax_rate', '?')}  "
              f"滑点: {cfg.get('slippage_pct', '?')}")
        sm = cfg.get("state_machine", "")
        if sm:
            print(f"  状态机: {sm}  参数: {cfg.get('state_machine_params', {})}")
        buy_g = cfg.get("buy_groups", [])
        sell_g = cfg.get("sell_groups", [])
        print(f"  买入条件组: {len(buy_g)}组  卖出条件组: {len(sell_g)}组")
        for i, g in enumerate(buy_g):
            conds = g.get("conditions", [])
            cond_strs = [c.get("indicator", "") for c in conds]
            print(f"    买入组{i+1}: {' AND '.join(cond_strs)}")
        for i, g in enumerate(sell_g):
            conds = g.get("conditions", [])
            cond_strs = [c.get("indicator", "") for c in conds]
            print(f"    卖出组{i+1}: {' AND '.join(cond_strs)}")


# ── Section 1: 核心指标摘要 ─────────────────────────────────────

def print_core_summary(all_data, risk_free_rate=0.0):
    print(f"\n{'─' * 90} 一、核心指标摘要 {'─' * 90}\n")

    header = (f"{'策略':<15} {'池':<10} {'交易':>6} {'胜率':>7} {'盈亏比':>7} "
              f"{'EV':>8} {'总收益':>10} {'年化':>8} {'最大回撤':>8} {'夏普':>7} {'日均':>7}")
    print(header)
    print("-" * len(header))

    for label, d in all_data:
        s = d["summary"]
        trades = d["trades"]
        equity = d.get("equity_curve", [])

        total_ret = s.get("total_return_pct", total_return(equity, s.get("initial_capital", 100000)))
        annual_ret = s.get("annual_return_pct", annual_return(equity, s.get("initial_capital", 100000)))
        max_dd = s.get("max_drawdown_pct", 0)
        # Use forward-filled Sharpe from analytics for accuracy
        shp = sharpe_ratio(equity, s.get("initial_capital", 100000), risk_free_rate=risk_free_rate)
        if shp == 0.0:
            shp = s.get("sharpe_ratio", 0)

        avg_daily = s["total_trades"] / max(1, len(equity)) if equity else 0

        print(f"{label:<15} {s['pool']:<10} {s['total_trades']:>6} "
              f"{s['win_rate']:>6.1f}% {s['profit_loss_ratio']:>6.2f} "
              f"{s['expected_value']:>+7.2f}% {total_ret:>+9.2f}% {annual_ret:>7.2f}% "
              f"{max_dd:>7.1f}% {shp:>6.2f} {avg_daily:>6.2f}")


# ── Section 2: 风险分析 ──────────────────────────────────────────

def print_risk_analysis(all_data, risk_free_rate=0.0):
    print(f"\n{'─' * 90} 二、风险分析 {'─' * 90}\n")

    header = (f"{'策略':<15} {'最大回撤':>8} {'回撤天数':>8} {'VaR日':>8} "
              f"{'CVaR日':>8} {'Sortino':>8} {'Calmar':>8} {'连续亏损':>10} {'波动率':>8}")
    print(header)
    print("-" * len(header))

    for label, d in all_data:
        trades = d["trades"]
        equity = d.get("equity_curve", [])
        init_cap = d["summary"].get("initial_capital", 100000)

        dd_pct, dd_start, dd_end, dd_days = max_drawdown(equity) if equity else (0, "", "", 0)
        # Use DAILY VaR/CVaR (correct portfolio-level)
        var95 = var_daily(equity, 0.95)
        cvar95 = cvar_daily(equity, 0.95)
        sortino = sortino_ratio(equity, init_cap)
        calmar = calmar_ratio(equity, init_cap)
        max_cl, total_cl = max_consecutive_losses(trades)
        vol = volatility_annualized(
            [t.get("profit_pct", 0) for t in trades]
        )

        print(f"{label:<15} {dd_pct:>7.1f}% {dd_days:>7}d {var95:>+7.2f}% "
              f"{cvar95:>+7.2f}% {sortino:>7.2f} {calmar:>7.2f} "
              f"{max_cl:>4}笔{total_cl:>+5.1f}% {vol:>7.2f}%")


# ── Section 2b: 回撤恢复分析 ────────────────────────────────────

def print_drawdown_recovery(all_data):
    print(f"\n{'─' * 90} 二B、回撤恢复分析 {'─' * 90}\n")

    header = (f"{'策略':<15} {'最大回撤':>8} {'峰值到谷底':>10} "
              f"{'恢复天数':>8} {'均恢复':>8} {'回撤次数':>8} {'水下占比':>8} {'均回撤':>8}")
    print(header)
    print("-" * len(header))

    for label, d in all_data:
        equity = d.get("equity_curve", [])
        if not equity:
            continue
        rec = drawdown_recovery_stats(equity)
        rec_str = f"{rec['max_recovery_days']}d" if rec['max_recovery_days'] >= 0 else "未恢复"
        avg_rec = f"{rec['avg_recovery_days']:.0f}d" if rec['avg_recovery_days'] >= 0 else "N/A"

        print(f"{label:<15} {rec['max_dd_pct']:>7.1f}% {rec['max_dd_days']:>7}d "
              f"{rec_str:>8} {avg_rec:>8} {rec['drawdown_count']:>8} "
              f"{rec['underwater_ratio']:>7.1f}% {rec['avg_drawdown_pct']:>7.1f}%")


# ── Section 3: 统计检验 ──────────────────────────────────────────

def print_statistical_tests(all_data):
    print(f"\n{'─' * 90} 三、统计检验 {'─' * 90}\n")

    header = (f"{'策略':<15} {'EV':>8} {'Bootstrap CI(95%)':>22} {'t值':>7} {'p值':>8} "
              f"{'显著':>6} {'偏度':>7} {'峰度':>7}")
    print(header)
    print("-" * len(header))

    for label, d in all_data:
        trades = d["trades"]
        ev = d["summary"]["expected_value"]
        ci_low, ci_high, ci_mean = bootstrap_ev_ci(trades)
        t_stat, p_val, sig = t_test_mean(trades)
        dist = profit_distribution_stats(trades)
        skew = dist.get("skewness", 0)
        kurt = dist.get("kurtosis", 0)

        ci_str = f"[{ci_low:+.2f}%, {ci_high:+.2f}%]"
        sig_str = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else ""))

        print(f"{label:<15} {ev:>+7.2f}% {ci_str:>22} {t_stat:>+6.2f} {p_val:>7.4f} "
              f"{sig_str:>6} {skew:>+6.2f} {kurt:>+6.2f}")


# ── Section 4: 基准对比 ──────────────────────────────────────────

def print_benchmark_comparison(all_data, benchmark_name):
    print(f"\n{'─' * 90} 四、基准对比 (vs {benchmark_name}) {'─' * 90}\n")

    from backend.benchmark_data import load_benchmark_equity_curve

    try:
        bench_equity = load_benchmark_equity_curve(benchmark_name)
    except FileNotFoundError as e:
        print(f"  [WARN] 无法加载基准: {e}")
        return

    for label, d in all_data:
        equity = d.get("equity_curve", [])
        init_cap = d["summary"].get("initial_capital", 100000)
        if not equity:
            continue

        comparison = compare_to_benchmark(equity, bench_equity, init_cap)

        print(f"\n  {label}:")
        print(f"    Alpha (年化):     {comparison.get('alpha', 0):>+8.2f}%")
        print(f"    Beta:              {comparison.get('beta', 0):>8.2f}")
        print(f"    信息比率:          {comparison.get('information_ratio', 0):>8.2f}")
        print(f"    跟踪误差 (年化):   {comparison.get('tracking_error', 0):>8.2f}%")
        print(f"    R²:                {comparison.get('r_squared', 0):>8.4f}")

        # CAPM因子分解
        try:
            capm = capm_regression(equity, bench_equity)
            print(f"\n    CAPM分解:")
            print(f"      Alpha (年化):   {capm.get('alpha_annual_pct', 0):>+8.2f}% "
                  f"(t={capm.get('t_stat_alpha', 0):.2f}, p={capm.get('p_value_alpha', 1):.4f})")
            print(f"      Beta:           {capm.get('beta', 0):>8.2f}     "
                  f"(t={capm.get('t_stat_beta', 0):.2f}, p={capm.get('p_value_beta', 1):.4f})")
            print(f"      Info Coef:      {capm.get('information_coefficient', 0):>8.4f}")
        except Exception:
            pass

        # 牛熊分析
        bull_bear = bull_bear_analysis(d["trades"], bench_equity)
        if bull_bear:
            print(f"\n    牛熊分解:")
            bull = bull_bear.get("bull", {})
            bear = bull_bear.get("bear", {})
            if bull:
                print(f"      牛市: {bull.get('count', 0)}笔  收益 {bull.get('avg_return', 0):>+6.2f}%  "
                      f"胜率 {bull.get('win_rate', 0):>5.1f}%")
            if bear:
                print(f"      熊市: {bear.get('count', 0)}笔  收益 {bear.get('avg_return', 0):>+6.2f}%  "
                      f"胜率 {bear.get('win_rate', 0):>5.1f}%")


# ── Section 5: 交易归因 ──────────────────────────────────────────

def print_trade_attribution(all_data):
    print(f"\n{'─' * 90} 五、交易归因分析 {'─' * 90}\n")

    for label, d in all_data:
        trades = d["trades"]
        print(f"\n  【{label}】\n")

        # 年度归因
        year_attr = yearly_attribution(trades)
        if year_attr:
            print(f"  {'年份':<8} {'笔数':>6} {'胜率':>7} {'均收益':>9} {'总收益':>10}")
            print(f"  {'-' * 45}")
            for year, info in sorted(year_attr.items()):
                print(f"  {year:<8} {info.get('trades', 0):>6} "
                      f"{info.get('win_rate', 0):>6.1f}% {info.get('avg_return', 0):>+8.2f}% "
                      f"{info.get('total_return', 0):>+9.2f}%")

        # 板块归因
        sector_attr = sector_attribution(trades)
        if sector_attr:
            print(f"\n  板块分布:")
            print(f"  {'板块':<12} {'笔数':>6} {'胜率':>7} {'均收益':>9} {'总收益':>10}")
            print(f"  {'-' * 50}")
            for sec, info in sorted(sector_attr.items(), key=lambda x: -x[1].get("count", 0)):
                print(f"  {sec:<12} {info.get('count', 0):>6} "
                      f"{info.get('win_rate', 0):>6.1f}% {info.get('avg_return', 0):>+8.2f}% "
                      f"{info.get('total_return', 0):>+9.2f}%")

        # 卖出原因
        reason_attr = sell_reason_attribution(trades)
        if reason_attr:
            print(f"\n  卖出原因分布:")
            print(f"  {'原因':<22} {'笔数':>6} {'均收益':>9} {'胜率':>7}")
            print(f"  {'-' * 50}")
            for reason, info in sorted(reason_attr.items(), key=lambda x: -x[1].get("count", 0)):
                print(f"  {reason:<22} {info.get('count', 0):>6} "
                      f"{info.get('avg_return', 0):>+8.2f}% {info.get('win_rate', 0):>6.1f}%")

        # 持仓集中度
        conc = position_concentration(trades, top_n=5)
        if conc:
            top_stocks = conc.get("by_count", [])[:5]
            if top_stocks:
                print(f"\n  交易最活跃股票 TOP5:")
                for s in top_stocks:
                    print(f"    {s.get('code', ''):<10} {s.get('count', 0):>4}笔  "
                          f"总收益 {s.get('total_return', 0):>+8.2f}%")


# ── Section 6: 换手率与容量 ─────────────────────────────────────

def print_turnover_capacity(all_data):
    print(f"\n{'─' * 90} 六、换手率与容量分析 {'─' * 90}\n")

    header = (f"{'策略':<15} {'年均交易':>8} {'月均交易':>8} {'年换手率':>9} "
              f"{'均持仓天':>8} {'信号密度':>8} {'标的数':>7} {'估算容量':>10}")
    print(header)
    print("-" * len(header))

    for label, d in all_data:
        trades = d["trades"]
        init_cap = d["summary"].get("initial_capital", 100000)
        to = turnover_analysis(trades, init_cap)
        cap = capacity_estimate(trades)

        print(f"{label:<15} {to['avg_trades_per_year']:>7.1f} {to['avg_trades_per_month']:>7.1f} "
              f"{to['turnover_rate_annual']:>8.0f}% {to['avg_hold_days']:>7.1f}d "
              f"{to['signal_density_pct']:>7.1f}% {to['unique_codes']:>7} "
              f"{cap['max_capital_estimate']:>9,.0f}")


# ── Section 7: 交易成本归因 ─────────────────────────────────────

def print_tca(all_data):
    print(f"\n{'─' * 90} 七、交易成本归因 (TCA) {'─' * 90}\n")

    for label, d in all_data:
        trades = d["trades"]
        s = d["summary"]

        # Actual cost from trade pnl
        cost = cost_attribution(trades)
        # Theoretical breakdown from config rates
        breakdown = estimated_cost_breakdown(
            trades,
            commission_rate=s.get("commission_rate", 0.0003),
            stamp_tax_rate=s.get("stamp_tax_rate", 0.001),
            slippage_pct=s.get("slippage_pct", 0.001),
        )

        print(f"\n  {label}:")
        print(f"    实际成本:         {cost['total_cost']:>10,.0f} 元")
        print(f"    总成交额:         {cost['total_turnover']:>10,.0f} 元")
        print(f"    成本/bp:          {cost['cost_bps_turnover']:>10.1f} bp")
        print(f"    成本/毛利:        {cost['cost_to_gross_profit_pct']:>10.1f}%")
        print(f"\n    理论成本分解:")
        print(f"      佣金 (双向万三): {breakdown['commission']:>10,.0f} 元")
        print(f"      印花税 (卖出千1):{breakdown['stamp_tax']:>10,.0f} 元")
        print(f"      滑点估算 (双向千1):{breakdown['slippage_estimate']:>10,.0f} 元")
        print(f"      合计:            {breakdown['total_cost']:>10,.0f} 元 ({breakdown['cost_bps']:.1f} bp)")


# ── Section 8: 风险情景压力测试 ─────────────────────────────────

def print_scenario_stress(all_data, benchmark_equity=None, benchmark_name="CSI 300"):
    print(f"\n{'─' * 90} 八、历史情景压力测试 {'─' * 90}\n")

    for label, d in all_data:
        equity = d.get("equity_curve", [])
        if not equity:
            continue
        results = scenario_stress_test(equity, benchmark_equity=benchmark_equity)
        if results:
            print(f"\n  {label}:")
            print(scenario_summary_table(results))


# ── Section 9: 策略间相关性 ─────────────────────────────────────

def print_correlation(all_data):
    if len(all_data) < 2:
        return
    print(f"\n{'─' * 90} 九、策略间相关性 {'─' * 90}\n")

    curves = {}
    for label, d in all_data:
        eq = d.get("equity_curve", [])
        if eq:
            curves[label] = eq

    if len(curves) < 2:
        print("  需要至少2个策略才能计算相关性")
        return

    corr = correlation_matrix(curves)
    if not corr.get("names"):
        print("  无法计算相关性（数据不足）")
        return

    names = corr["names"]
    matrix = corr["matrix"]

    # Print matrix header
    print(f"  {'':<15}", end="")
    for n in names:
        print(f"{n:<12}", end="")
    print()
    print("  " + "-" * (15 + 12 * len(names)))

    for n1 in names:
        print(f"  {n1:<15}", end="")
        for n2 in names:
            val = matrix.get(n1, {}).get(n2, 0)
            print(f"{val:>11.4f} ", end="")
        print()

    print(f"\n  平均相关系数: {corr['mean_correlation']:.4f}")
    print(f"  分散化比率:   {corr['diversification_ratio']:.4f}")


# ── Section 9B: 极端周亏损分析 ──────────────────────────────

def print_extreme_drawdown(all_data):
    print(f"\n{'─' * 90} 九B、极端滚动亏损分析 {'─' * 90}\n")
    from analytics.extreme import weekly_stress_report, format_weekly_stress
    for name, d in all_data:
        equity = d.get("equity_curve", [])
        if not equity:
            continue
        init_cap = d.get("summary", {}).get("initial_capital", 100000)
        stress = weekly_stress_report(equity, initial_capital=init_cap)
        print(f"\n策略: {name}")
        print(format_weekly_stress(stress))


# ── Section 10: Walk-Forward 验证 ────────────────────────────────

def print_walk_forward(strategy_name, pool_name, all_data):
    print(f"\n{'─' * 90} 十、Walk-Forward 验证 (样本外) {'─' * 90}\n")

    import yaml
    from backend.main import _resolve_pool_codes
    from validation.walk_forward import run_walk_forward as wf_run, walk_forward_summary_table

    strategy_path = _find_strategy_yaml(strategy_name)
    if not strategy_path:
        print(f"  [WARN] 策略文件不存在: {strategy_path}")
        return

    with open(strategy_path) as f:
        cfg = yaml.safe_load(f)

    pool = _resolve_pool_codes(pool_name)
    cfg["stock_pool"] = pool

    # Get date range from loaded data
    start_date = "2015-01-01"
    end_date = "2025-12-31"
    if all_data:
        first = all_data[0][1]
        s = first.get("summary", {})
        start_date = s.get("start_date", start_date)
        end_date = s.get("end_date", end_date)

    print(f"  运行 {strategy_name} / {pool_name} ({len(pool)}只) Walk-Forward...")
    print(f"  训练窗口: 5年 / 测试窗口: 1年 / 步长: 1年")
    print(f"  日期范围: {start_date} → {end_date}\n")

    try:
        results = wf_run(cfg, pool, start_date=start_date, end_date=end_date)
        table = walk_forward_summary_table(results)
        print(table)
    except Exception as e:
        print(f"  [ERROR] Walk-Forward 失败: {e}")
        import traceback
        traceback.print_exc()


# ── Section 11: 参数敏感性 ──────────────────────────────────────

def print_parameter_sensitivity(strategy_name, pool_name, param_name, param_values):
    print(f"\n{'─' * 90} 十一、参数敏感性 ({param_name}) {'─' * 90}\n")

    import yaml
    from backend.main import _config_from_dict, _resolve_pool_codes
    from validation.sensitivity import parameter_sweep, parameter_stability_score

    strategy_path = _find_strategy_yaml(strategy_name)
    if not strategy_path:
        print(f"  [WARN] 策略文件不存在: {strategy_path}")
        return

    with open(strategy_path) as f:
        base_cfg = yaml.safe_load(f)

    pool = _resolve_pool_codes(pool_name)

    print(f"  扫描 {param_name} = {param_values}\n")

    try:
        sweep = parameter_sweep(base_cfg, param_name, param_values, pool)
        for r in sweep.get("results", []):
            val = r.get("param_value")
            ev = r.get("ev", 0)
            wr = r.get("win_rate", 0)
            trades = r.get("total_trades", 0)
            if ev is None:
                print(f"  {param_name}={val:<8} ERROR")
            else:
                print(f"  {param_name}={val:<8} EV={ev:>+7.2f}%  胜率={wr:>5.1f}%  交易={trades:>5}")

        optimal = sweep.get("optimal", {})
        if optimal and optimal.get("ev") is not None:
            print(f"\n  最优: {param_name}={optimal.get('param_value')}  EV={optimal.get('ev'):+.2f}%")

        stability = parameter_stability_score(sweep)
        if stability >= 0:
            print(f"  稳定性评分: {stability:.0f}/100")
    except Exception as e:
        print(f"  [ERROR] 参数扫描失败: {e}")
        import traceback
        traceback.print_exc()


# ── Section 12: 过拟合检测 (Train/Val/Test) ──────────────────────

def print_overfitting_check(strategy_name, pool_name, param_name, param_values, all_data):
    print(f"\n{'─' * 90} 十二、过拟合检测 (Train/Val 分离) {'─' * 90}\n")

    import yaml
    from backend.main import _resolve_pool_codes
    from validation.sample_split import optimize_on_train_val

    strategy_path = _find_strategy_yaml(strategy_name)
    if not strategy_path:
        print(f"  [WARN] 策略文件不存在: {strategy_path}")
        return

    with open(strategy_path) as f:
        cfg = yaml.safe_load(f)

    pool = _resolve_pool_codes(pool_name)
    start_date = "2015-01-01"
    end_date = "2025-12-31"
    if all_data:
        first = all_data[0][1]
        s = first.get("summary", {})
        start_date = s.get("start_date", start_date)
        end_date = s.get("end_date", end_date)

    print(f"  策略: {strategy_name} / {pool_name} ({len(pool)}只)")
    print(f"  参数: {param_name} = {param_values}")
    print(f"  Train: 60% / Val: 20% / Test: 20% (holdout)\n")

    try:
        result = optimize_on_train_val(
            cfg, pool, start_date, end_date,
            param_name, param_values,
            train_ratio=0.60, val_ratio=0.20, metric="ev",
        )
        of = result["overfitting"]
        print(f"  最优参数:       {param_name} = {result['best_param']}")
        print(f"  Train 区间:     {result['train_period'][0]} -> {result['train_period'][1]}")
        print(f"  Val   区间:     {result['val_period'][0]} -> {result['val_period'][1]}")
        print(f"  Test  区间:     {result['test_period'][0]} -> {result['test_period'][1]} (holdout)")
        print(f"")
        print(f"  Train EV:       {of['details']['train_ev']:>+8.2f}%")
        print(f"  Val   EV:       {of['details']['val_ev']:>+8.2f}%")
        print(f"  EV 衰减:        {of['ev_decay_train_to_val_pct']:>+7.1f}%")
        print(f"  Sharpe 衰减:    {of['sharpe_decay_pct']:>+7.1f}%")
        print(f"  过拟合评分:     {of['overfitting_score']:>7.0f}/100")
        print(f"  判定:           {of['verdict']}")

        print(f"\n  各参数 Train vs Val 对比:")
        print(f"  {'参数值':>10} {'Train EV':>10} {'Val EV':>10} {'衰减%':>8} {'Val Sharpe':>11}")
        print("  " + "-" * 55)
        for r in result.get("all_results", []):
            tv = r["train"].get("ev", 0)
            vv = r["val"].get("ev", 0)
            vs = r["val"].get("sharpe", 0)
            decay = (tv - vv) / abs(tv) * 100 if tv != 0 else 0
            print(f"  {r['param_value']:>10} {tv:>+9.2f}% {vv:>+9.2f}% {decay:>+7.1f}% {vs:>10.2f}")

    except Exception as e:
        print(f"  [ERROR] 过拟合检测失败: {e}")
        import traceback
        traceback.print_exc()


# ── Section 13: ML 信号过滤对比 ──────────────────────────────────

def print_ml_comparison(strategy_name, pool_name, ml_dir, threshold, score_col, all_data):
    print(f"\n{'─' * 90} 十三、ML 信号过滤对比 {'─' * 90}\n")

    import yaml
    from backend.main import _config_from_dict, _resolve_pool_codes
    from backend.ml_bridge import run_ml_filtered_backtest, format_ml_comparison_table

    strategy_path = _find_strategy_yaml(strategy_name)
    if not strategy_path:
        print(f"  [WARN] 策略文件不存在: {strategy_name}")
        return

    with open(strategy_path) as f:
        cfg = yaml.safe_load(f)

    pool = _resolve_pool_codes(pool_name)
    cfg["stock_pool"] = pool

    start_date = "2015-01-01"
    end_date = "2025-12-31"
    if all_data:
        first = all_data[0][1]
        s = first.get("summary", {})
        start_date = s.get("start_date", start_date)
        end_date = s.get("end_date", end_date)

    print(f"  策略: {strategy_name} / {pool_name} ({len(pool)}只)")
    print(f"  ML 数据: {ml_dir}")
    print(f"  分数列: {score_col}  |  阈值: {threshold}")
    print(f"  日期范围: {start_date} → {end_date}\n")

    try:
        config = _config_from_dict(cfg)
        result = run_ml_filtered_backtest(
            config,
            predictions_dir=ml_dir,
            score_threshold=threshold,
            score_col=score_col,
            start_date=start_date,
            end_date=end_date,
        )

        comparison = result.get("comparison")
        if comparison:
            print(format_ml_comparison_table(comparison))
        else:
            print(f"  [WARN] 未加载到 ML 预测数据。请先运行 quant_practical 的预测生成。")

        print(f"\n  预测数据: {result.get('predictions_loaded', 0)} 条, "
              f"{result.get('predictions_dates', 0)} 个日期")

    except Exception as e:
        print(f"  [ERROR] ML 过滤失败: {e}")
        import traceback
        traceback.print_exc()


# ── Main ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="量化回测综合分析报告生成器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python report.py results/zzh7.3_大蓝筹.json
  python report.py results/zzh7.3_大蓝筹.json results/zzhY.3_大蓝筹.json --benchmark hs300
  python report.py results/zzh7.3_大蓝筹.json --walk-forward --strategy zzh7.3 --pool 大蓝筹
  python report.py --sensitivity broken_days --values 15,20,25,30,35,40 --strategy zzh7.3 --pool 大蓝筹
  python report.py results/zzh7.3_大蓝筹.json --risk-free-rate 2.5 --scenario
        """,
    )
    ap.add_argument("files", nargs="*", help="结果JSON文件路径")
    ap.add_argument("--benchmark", default=None, help="基准指数名称 (hs300, cyb, sh, sz)")
    ap.add_argument("--walk-forward", action="store_true", help="运行 Walk-Forward 验证")
    ap.add_argument("--strategy", default=None, help="策略名称 (用于验证/敏感性)")
    ap.add_argument("--pool", default="大蓝筹", help="标的池名称")
    ap.add_argument("--sensitivity", default=None, help="参数敏感性分析 - 参数名")
    ap.add_argument("--values", default="", help="参数值列表，逗号分隔")
    ap.add_argument("--quick", action="store_true", help="快速模式 (跳过耗时模块)")
    ap.add_argument("--risk-free-rate", type=float, default=0.0,
                    help="无风险利率 (年化, 如 2.5)")
    ap.add_argument("--scenario", action="store_true", help="运行历史情景压力测试")
    ap.add_argument("--skip-scenario", action="store_true", help="跳过情景压力测试")
    ap.add_argument("--skip-walk-forward", action="store_true", help="跳过 Walk-Forward 验证")
    ap.add_argument("--skip-monte-carlo", action="store_true", help="跳过 Monte Carlo 验证")
    ap.add_argument("--skip-survival", action="store_true", help="跳过生存偏差检查")
    ap.add_argument("--overfit", default=None, help="过拟合检测: 参数名 (如 stop_loss_pct)")
    ap.add_argument("--param-values", default="5,10,15,20,25", help="过拟合检测的参数值，逗号分隔")
    ap.add_argument("--ml-filter", default=None, help="ML信号过滤: 预测CSV目录路径")
    ap.add_argument("--ml-threshold", type=float, default=0.3,
                    help="ML分数阈值 (默认0.3)")
    ap.add_argument("--ml-score-col", default="ensemble_score",
                    help="ML分数列名 (默认 ensemble_score)")
    ap.add_argument("--output", default=None, help="报告输出路径 (默认自动保存到 reports/)")
    args = ap.parse_args()

    # Setup output file with timestamp
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output:
        outpath = Path(args.output)
    else:
        strategy_tag = args.strategy or (args.files[0].replace(".json","").replace("results/","") if args.files else "report")
        outpath = REPORTS_DIR / f"{ts}_{strategy_tag}.txt"
    tee = Tee(str(outpath))
    sys.stdout = tee
    rfr = args.risk_free_rate

    print_header(args.strategy, args.pool)

    # 生存偏差检查（默认运行）
    if not args.skip_survival and not args.quick:
        print_survivorship_check()

    # 加载结果
    if args.files:
        data = load_results(*args.files)
    else:
        json_files = sorted(RESULTS_DIR.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True)
        if json_files:
            data = load_results(*[f.name for f in json_files[:5]])
        else:
            print("  没有找到结果文件")
            sys.exit(1)

    if not data:
        print("  没有加载到任何数据")
        sys.exit(1)

    # 各节
    print_config_snapshot(data)
    print_core_summary(data, risk_free_rate=rfr)
    print_risk_analysis(data)
    print_drawdown_recovery(data)

    if not args.quick:
        print(f"\n  [计算中] Bootstrap 置信区间...")
    print_statistical_tests(data)

    if args.benchmark:
        print_benchmark_comparison(data, args.benchmark)

    print_trade_attribution(data)
    print_turnover_capacity(data)
    print_tca(data)

    if args.scenario or not args.skip_scenario:
        # Load benchmark for actual scenario comparison
        bench_eq = None
        if args.benchmark:
            try:
                from backend.benchmark_data import load_benchmark_equity_curve
                bench_eq = load_benchmark_equity_curve(args.benchmark)
            except Exception:
                pass
        print_scenario_stress(data, benchmark_equity=bench_eq)

    print_correlation(data)

    # 极端周亏损分析
    print_extreme_drawdown(data)

    # Monte Carlo 验证（默认运行）
    if not args.skip_monte_carlo and not args.quick:
        print_monte_carlo(data)

    if not args.skip_walk_forward and args.strategy:
        if not args.quick:
            print_walk_forward(args.strategy, args.pool, data)

    if args.sensitivity and args.strategy:
        values = [float(v.strip()) for v in args.values.split(",")] if args.values else []
        if values and not args.quick:
            print_parameter_sensitivity(args.strategy, args.pool, args.sensitivity, values)

    if args.overfit and args.strategy:
        values = [float(v.strip()) for v in args.param_values.split(",")] if args.param_values else []
        if values and not args.quick:
            print_overfitting_check(args.strategy, args.pool, args.overfit, values, data)

    if args.ml_filter and args.strategy:
        print_ml_comparison(args.strategy, args.pool, args.ml_filter,
                           args.ml_threshold, args.ml_score_col, data)

    print(f"\n{SEP}")
    print(f"  报告完成")
    print(f"{SEP}\n")

    sys.stdout = tee.stdout
    tee.close()
