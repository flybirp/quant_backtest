"""
ML Bridge — 连接 quant_practical 的 ML 信号过滤管线 到 quant_backtest 的回测引擎。

职责：
  1. 加载 ML 预测 CSV
  2. 根据 ML score 过滤回测交易
  3. 从过滤后的交易重新计算统计指标
  4. 输出"全量信号 vs ML过滤后"对比

用法：
  from backend.ml_bridge import run_ml_filtered_backtest

  result = run_ml_filtered_backtest(
      config, predictions_dir, score_threshold=0.3, score_col="ensemble_score"
  )
  # result.raw    — 全量信号的回测结果
  # result.filtered — ML 过滤后的回测结果
  # result.comparison — 对比表
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── 数据加载 ──────────────────────────────────────────────────────

def load_ml_predictions(
    sources: str | Path | list[str | Path],
    score_col: str = "ensemble_score",
) -> dict[str, dict[str, float]]:
    """
    加载 ML 预测数据，构建日期 → {股票代码 → 预测分数} 的快速查找表。

    Args:
        sources: 单个 CSV 路径、CSV 目录、或 CSV 路径列表。
        score_col: 用作过滤分数的列名（'ensemble_score', 'pred_a', 'pred_b', 'pred_c'）。

    Returns:
        {date_str: {code: score}}，date_str 格式为 'YYYYMMDD' 或 'YYYY-MM-DD'。
    """
    if isinstance(sources, (str, Path)):
        sources = [sources]

    all_paths: list[Path] = []
    for src in sources:
        p = Path(src)
        if p.is_dir():
            all_paths.extend(sorted(p.glob("*.csv")))
        elif p.is_file() and p.suffix == ".csv":
            all_paths.append(p)

    if not all_paths:
        logger.warning("No prediction CSV files found in %s", sources)
        return {}

    lookup: dict[str, dict[str, float]] = {}
    loaded = 0

    for csv_path in all_paths:
        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:
            logger.warning("Failed to read %s: %s", csv_path, exc)
            continue

        required = ["date_str", "code", score_col]
        missing = [c for c in required if c not in df.columns]
        if missing:
            # Try with _with_pred_score suffix variant
            if "pred_a" in df.columns:
                score_col_actual = score_col if score_col in df.columns else "pred_a"
            else:
                logger.warning("%s missing columns: %s", csv_path.name, missing)
                continue
        else:
            score_col_actual = score_col

        for _, row in df.iterrows():
            date_str = str(row.get("date_str", ""))
            code = str(row.get("code", ""))
            # Handle pandas reading "000001" as integer 1
            if code.isdigit():
                code = code.zfill(6)
            score = row.get(score_col_actual, 0)

            if not date_str or not code:
                continue
            if pd.isna(score):
                continue

            # Normalize date format
            date_str = date_str.strip()
            # Handle pandas reading dates as float (20200101.0)
            if date_str.endswith('.0') and len(date_str) > 2:
                date_str = date_str[:-2]
            # '20200101' → '2020-01-01'
            if len(date_str) == 8 and date_str.isdigit():
                date_str = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"

            # Normalize code: remove '.0' suffix, zero-pad
            code = code.strip()
            if code.endswith('.0') and len(code) > 2:
                code = code[:-2]
            if code.isdigit() and len(code) < 6:
                code = code.zfill(6)

            if date_str not in lookup:
                lookup[date_str] = {}
            lookup[date_str][code] = float(score)
            loaded += 1

    logger.info("Loaded %d predictions across %d dates from %d files.",
                loaded, len(lookup), len(all_paths))
    return lookup


# ── 交易过滤 ──────────────────────────────────────────────────────

def filter_trades_by_ml(
    trades: list[dict],
    predictions: dict[str, dict[str, float]],
    threshold: float = 0.3,
    score_col_hint: str = "ensemble_score",
) -> tuple[list[dict], list[dict]]:
    """
    根据 ML 预测分数过滤交易。

    匹配规则：trade['buy_date'][:10] == prediction_date  AND  trade['code'] == prediction_code

    Args:
        trades: 原始交易列表 (每条 dict 含 'buy_date', 'code', 'profit_pct' 等)。
        predictions: load_ml_predictions 的输出。
        threshold: 分数阈值，只保留 score >= threshold 的交易。
        score_col_hint: 日志用，列名提示。

    Returns:
        (passed_trades, filtered_out_trades)
    """
    if not predictions:
        logger.warning("No ML predictions loaded — returning all trades unfiltered.")
        return trades, []

    passed = []
    filtered_out = []
    matched = 0
    unmatched = 0

    for t in trades:
        code = str(t.get("code", ""))
        buy_date = str(t.get("buy_date", ""))[:10]

        # Lookup prediction
        date_scores = predictions.get(buy_date, {})
        score = date_scores.get(code)

        if score is None:
            # No prediction for this trade — keep it (don't filter)
            passed.append(t)
            unmatched += 1
        elif score >= threshold:
            passed.append(t)
            matched += 1
        else:
            filtered_out.append(t)
            matched += 1

    logger.info(
        "ML filter (threshold=%.2f): %d passed, %d filtered out, %d unmatched (kept).",
        threshold, len(passed), len(filtered_out), unmatched,
    )
    return passed, filtered_out


# ── 回测结果重算 ──────────────────────────────────────────────────

@dataclass
class MLComparisonResult:
    """全量信号 vs ML过滤 的对比结果"""
    raw_trades: int = 0
    filtered_trades: int = 0
    raw_ev: float = 0.0
    filtered_ev: float = 0.0
    raw_sharpe: float = 0.0
    filtered_sharpe: float = 0.0
    raw_win_rate: float = 0.0
    filtered_win_rate: float = 0.0
    raw_total_return: float = 0.0
    filtered_total_return: float = 0.0
    raw_max_dd: float = 0.0
    filtered_max_dd: float = 0.0
    threshold: float = 0.3
    kept_ratio_pct: float = 0.0

    @property
    def ev_change_pct(self) -> float:
        if self.raw_ev == 0:
            return 0.0
        return (self.filtered_ev - self.raw_ev) / abs(self.raw_ev) * 100.0

    @property
    def sharpe_change_pct(self) -> float:
        if self.raw_sharpe == 0:
            return 0.0
        return (self.filtered_sharpe - self.raw_sharpe) / abs(self.raw_sharpe) * 100.0


def compute_comparison(
    raw_trades: list[dict],
    filtered_trades: list[dict],
    equity_curve: list[dict],
    initial_capital: float,
    threshold: float,
) -> MLComparisonResult:
    """根据原始和过滤后的交易列表计算对比指标。"""
    from analytics.performance import total_return, annual_return
    from analytics.risk import max_drawdown, sharpe_ratio

    # Raw metrics
    raw_ev = _compute_ev(raw_trades)
    raw_wr = _compute_win_rate(raw_trades)
    raw_tr = total_return(equity_curve, initial_capital)
    raw_md, _, _, _ = max_drawdown(equity_curve)
    raw_sh = sharpe_ratio(equity_curve, initial_capital)

    # Build filtered equity curve
    filtered_equity = _build_equity_curve(filtered_trades, initial_capital)

    # Filtered metrics
    filt_ev = _compute_ev(filtered_trades)
    filt_wr = _compute_win_rate(filtered_trades)
    filt_tr = total_return(filtered_equity, initial_capital)
    filt_md, _, _, _ = max_drawdown(filtered_equity)
    filt_sh = sharpe_ratio(filtered_equity, initial_capital)

    return MLComparisonResult(
        raw_trades=len(raw_trades),
        filtered_trades=len(filtered_trades),
        raw_ev=round(raw_ev, 2),
        filtered_ev=round(filt_ev, 2),
        raw_sharpe=round(raw_sh, 2),
        filtered_sharpe=round(filt_sh, 2),
        raw_win_rate=round(raw_wr, 1),
        filtered_win_rate=round(filt_wr, 1),
        raw_total_return=round(raw_tr, 2),
        filtered_total_return=round(filt_tr, 2),
        raw_max_dd=round(raw_md, 2),
        filtered_max_dd=round(filt_md, 2),
        threshold=threshold,
        kept_ratio_pct=round(len(filtered_trades) / len(raw_trades) * 100.0, 1) if raw_trades else 0.0,
    )


# ── 一键运行 ──────────────────────────────────────────────────────

def run_ml_filtered_backtest(
    config,  # StrategyConfig
    predictions_dir: str | Path,
    score_threshold: float = 0.3,
    score_col: str = "ensemble_score",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    progress_callback=None,
) -> dict[str, Any]:
    """
    完整流水线：回测 → ML过滤 → 重算对比。

    Args:
        config: StrategyConfig
        predictions_dir: ML 预测 CSV 目录
        score_threshold: 分数阈值
        score_col: 分数列名
        start_date, end_date: 回测日期范围

    Returns:
        {
            'raw_result': BacktestResult,
            'filtered_trades': list[dict],
            'comparison': MLComparisonResult,
            'predictions_loaded': int,
            'predictions_dates': int,
        }
    """
    from backend.backtest_engine import run_backtest

    # Step 1: Run raw backtest
    raw_result = run_backtest(config, start_date=start_date, end_date=end_date,
                              progress_callback=progress_callback)

    trades = raw_result.trades
    if not trades:
        logger.warning("No trades generated — skipping ML filter.")
        return {
            "raw_result": raw_result,
            "filtered_trades": [],
            "comparison": None,
            "predictions_loaded": 0,
            "predictions_dates": 0,
        }

    # Step 2: Load ML predictions
    predictions = load_ml_predictions(predictions_dir, score_col=score_col)
    if not predictions:
        return {
            "raw_result": raw_result,
            "filtered_trades": trades,
            "comparison": None,
            "predictions_loaded": 0,
            "predictions_dates": 0,
        }

    # Step 3: Filter trades
    passed, filtered_out = filter_trades_by_ml(trades, predictions, score_threshold, score_col)

    # Step 4: Compute comparison
    comparison = compute_comparison(
        trades, passed, raw_result.equity_curve,
        config.initial_capital, score_threshold,
    )

    return {
        "raw_result": raw_result,
        "filtered_trades": passed,
        "filtered_out_trades": filtered_out,
        "comparison": comparison,
        "predictions_loaded": sum(len(v) for v in predictions.values()),
        "predictions_dates": len(predictions),
    }


# ── 辅助函数 ──────────────────────────────────────────────────────

def _compute_ev(trades: list[dict]) -> float:
    if not trades:
        return 0.0
    profits = [t.get("profit_pct", 0) for t in trades]
    return float(np.mean(profits))


def _compute_win_rate(trades: list[dict]) -> float:
    if not trades:
        return 0.0
    wins = sum(1 for t in trades if t.get("profit_pct", 0) > 0)
    return wins / len(trades) * 100.0


def _build_equity_curve(trades: list[dict], initial_capital: float) -> list[dict]:
    """从交易列表构建权益曲线（信号模式用累积求和）。"""
    if not trades:
        return [{"date": "2000-01-01", "equity": initial_capital}]

    sorted_trades = sorted(trades, key=lambda t: str(t.get("sell_date", "")))
    curve = []
    base = initial_capital
    cum_pnl = base
    cumulative_sum_pct = 0.0

    for t in sorted_trades:
        cumulative_sum_pct += t.get("profit_pct", 0)
        cum_pnl = base * (1 + cumulative_sum_pct / 100)
        curve.append({
            "date": str(t.get("sell_date", ""))[:10],
            "equity": round(max(cum_pnl, 0.01), 2),
        })

    return curve


def format_ml_comparison_table(comparison: MLComparisonResult) -> str:
    """生成 ML 过滤对比的格式化文本表格。"""
    if comparison is None:
        return "  ML 过滤数据不可用。"

    arrow_ev = "↑" if comparison.ev_change_pct > 0 else "↓"
    arrow_sh = "↑" if comparison.sharpe_change_pct > 0 else "↓"

    lines = [
        "=" * 80,
        "  ML 信号过滤对比分析",
        "=" * 80,
        f"  分数阈值: {comparison.threshold:.2f}  |  交易保留率: {comparison.kept_ratio_pct:.1f}%",
        "",
        f"  {'指标':<20} {'全量信号':>12} {'ML过滤后':>12} {'变化':>12}",
        "  " + "-" * 56,
        f"  {'交易笔数':<20} {comparison.raw_trades:>12} {comparison.filtered_trades:>12} "
        f"{comparison.filtered_trades - comparison.raw_trades:>+12}",
        f"  {'胜率':<20} {comparison.raw_win_rate:>11.1f}% {comparison.filtered_win_rate:>11.1f}% "
        f"{comparison.filtered_win_rate - comparison.raw_win_rate:>+11.1f}%",
        f"  {'EV':<20} {comparison.raw_ev:>+11.2f}% {comparison.filtered_ev:>+11.2f}% "
        f"{comparison.ev_change_pct:>+11.1f}% {arrow_ev}",
        f"  {'Sharpe':<20} {comparison.raw_sharpe:>12.2f} {comparison.filtered_sharpe:>12.2f} "
        f"{comparison.sharpe_change_pct:>+11.1f}% {arrow_sh}",
        f"  {'最大回撤':<20} {comparison.raw_max_dd:>11.1f}% {comparison.filtered_max_dd:>11.1f}% ",
        f"  {'总收益':<20} {comparison.raw_total_return:>+11.2f}% {comparison.filtered_total_return:>+11.2f}%",
        "=" * 80,
    ]

    # Verdict
    if comparison.ev_change_pct > 10 and comparison.sharpe_change_pct > 10:
        lines.append("  判定: ML 过滤显著有效 ✓")
    elif comparison.ev_change_pct > 0:
        lines.append("  判定: ML 过滤有正面效果（轻微）")
    elif comparison.ev_change_pct > -10:
        lines.append("  判定: ML 过滤无明显效果")
    else:
        lines.append("  判定: ML 过滤反而降低收益 ✗ — 检查阈值或模型")

    return "\n".join(lines)
