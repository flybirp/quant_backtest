"""
Strategy Pipeline — 原子化执行：保存YAML → 回测 → 报告 → 更新top_strategy.md

用法:
    python run_pipeline.py DS6 --pool 大蓝筹
    python run_pipeline.py --name DS11 --pool 大蓝筹 --config '{"buy_groups":[...]}'
    python run_pipeline.py DS6 --pool 大蓝筹 --skip-walk-forward
"""
import sys, json, yaml, time, argparse
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))
from backend.main import _config_from_dict, _resolve_pool_codes
from backend.backtest_engine import run_backtest

BASE = Path(__file__).parent
RESULTS_DIR = BASE / "results"
RESULTS_DIR.mkdir(exist_ok=True)
STRATEGIES_DIR = BASE / "strategies" / "rule"


class ProgressPrinter:
    """回测进度打印：每N只票报一次，带ETA"""
    def __init__(self, total: int, interval: int = 10):
        self.total = total
        self.interval = interval
        self.start = time.time()
        self.last_print = 0

    def __call__(self, done: int, total: int):
        if done == 0:
            print(f"  ▶ 开始: {total}只股票", flush=True)
            return
        if done == total:
            elapsed = time.time() - self.start
            print(f"  ✅ 完成: {done}/{total} ({elapsed:.1f}s)", flush=True)
            return
        now = time.time()
        if done - self.last_print >= self.interval or (now - self.start) > 5:
            elapsed = now - self.start
            per_stock = elapsed / done if done > 0 else 0
            remaining = per_stock * (total - done)
            bar_len = 20
            filled = int(bar_len * done / total)
            bar = "█" * filled + "░" * (bar_len - filled)
            print(f"  [{bar}] {done}/{total} ({elapsed:.0f}s, ETA {remaining:.0f}s)", flush=True)
            self.last_print = done


def run_pipeline(strategy_name: str, pool_name: str, yaml_path: str = None,
                 skip_scenario: bool = False, skip_walk_forward: bool = False,
                 skip_monte_carlo: bool = False, skip_survival: bool = False) -> dict:
    """
    完整pipeline：YAML → 回测 → 结果JSON → 报告 → top_strategy.md

    Returns: {success, ev, win_rate, trades, uw, ci, elapsed, result_path, report_path}
    """
    result = {"success": False}
    t0 = time.time()

    # === Step 1: 加载或创建 YAML ===
    if yaml_path:
        ypath = Path(yaml_path)
    else:
        ypath = STRATEGIES_DIR / f"{strategy_name}.yaml"

    if not ypath.exists():
        result["error"] = f"策略文件不存在: {ypath}"
        print(f"  [FAIL] {result['error']}", flush=True)
        return result

    with open(ypath) as f:
        cfg = yaml.safe_load(f)

    # 确保 _pool 字段存在
    if "_pool" not in cfg:
        cfg["_pool"] = pool_name
        with open(ypath, "w") as f:
            yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False)
        print(f"  📝 更新 {ypath.name} 加入 _pool={pool_name}", flush=True)

    # === Step 2: 解析池 ===
    pool = _resolve_pool_codes(pool_name)
    cfg["stock_pool"] = pool

    print(f"\n{'='*60}", flush=True)
    print(f"  Pipeline: {strategy_name} | {pool_name} ({len(pool)}只)", flush=True)
    print(f"{'='*60}", flush=True)

    # === Step 3: 回测 ===
    print(f"  [1/4] 回测...", flush=True)
    progress = ProgressPrinter(len(pool))
    r = run_backtest(_config_from_dict(cfg), progress_callback=progress)
    elapsed = time.time() - t0

    # === Step 4: 保存结果 ===
    print(f"  [2/4] 保存结果...", flush=True)
    trades_data = [
        {k: str(v)[:10] if k.endswith("_date") and v else v for k, v in t.items()}
        for t in r.trades
    ]

    summary = {
        "strategy": strategy_name,
        "pool": pool_name,
        "pool_count": len(pool),
        "initial_capital": r.initial_capital,
        "total_return_pct": r.total_return_pct,
        "annual_return_pct": r.annual_return_pct,
        "max_drawdown_pct": r.max_drawdown_pct,
        "sharpe_ratio": r.sharpe_ratio,
        "total_trades": r.total_trades,
        "win_rate": r.win_rate,
        "profit_loss_ratio": r.profit_loss_ratio,
        "expected_value": r.expected_value,
        "avg_profit_pct": r.avg_profit_pct,
        "avg_loss_pct": r.avg_loss_pct,
        "max_profit_pct": r.max_profit_pct,
        "max_loss_pct": r.max_loss_pct,
        "avg_hold_days": r.avg_hold_days,
        "start_date": r.start_date,
        "end_date": r.end_date,
        "win_trades": r.win_trades,
        "lose_trades": r.lose_trades,
        "elapsed_seconds": round(elapsed, 1),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        # ── 策略配置快照（确保 JSON 自包含） ──
        "config": {
            "stop_loss_pct": cfg.get("stop_loss_pct", 5.0),
            "take_profit_pct": cfg.get("take_profit_pct", 15.0),
            "trailing_stop_pct": cfg.get("trailing_stop_pct", 0.0),
            "max_hold_days": cfg.get("max_hold_days", 0),
            "buy_price_type": cfg.get("buy_price_type", "close"),
            "sell_price_type": cfg.get("sell_price_type", "close"),
            "position_pct": cfg.get("position_pct", 1.0),
            "max_positions": cfg.get("max_positions", 5),
            "commission_rate": cfg.get("commission_rate", 0.0003),
            "stamp_tax_rate": cfg.get("stamp_tax_rate", 0.001),
            "slippage_pct": cfg.get("slippage_pct", 0.001),
            "state_machine": cfg.get("state_machine", ""),
            "state_machine_params": cfg.get("state_machine_params", {}),
            "buy_groups": cfg.get("buy_groups", []),
            "sell_groups": cfg.get("sell_groups", []),
            "add_groups": cfg.get("add_groups", []),
            "reduce_groups": cfg.get("reduce_groups", []),
        },
    }

    fname = f"{strategy_name}_{pool_name}.json"
    outpath = RESULTS_DIR / fname
    with open(outpath, "w") as f:
        json.dump({"summary": summary, "trades": trades_data, "equity_curve": r.equity_curve}, f, ensure_ascii=False)

    # === Step 5: 生成报告 ===
    print(f"  [3/4] 生成报告...", flush=True)
    report_path = _generate_report(strategy_name, pool_name, str(outpath),
                                   skip_scenario, skip_walk_forward,
                                   skip_monte_carlo, skip_survival)

    # === Step 6: 更新 top_strategy.md ===
    print(f"  [4/4] 更新 top_strategy.md...", flush=True)
    _append_top_strategy(strategy_name, pool_name, summary)

    # === 完成 ===
    result = {
        "success": True,
        "strategy": strategy_name,
        "pool": pool_name,
        "ev": r.expected_value,
        "win_rate": r.win_rate,
        "trades": r.total_trades,
        "sharpe": r.sharpe_ratio,
        "elapsed": round(elapsed, 1),
        "result_path": str(outpath),
        "report_path": report_path,
    }

    print(f"\n{'='*60}", flush=True)
    print(f"  ✅ {strategy_name} | {pool_name}", flush=True)
    print(f"  EV={r.expected_value:+.2f}%  Win={r.win_rate:.1f}%  "
          f"Trade={r.total_trades}  Sharpe={r.sharpe_ratio:.2f}  "
          f"Time={elapsed:.0f}s", flush=True)
    print(f"  → {outpath}", flush=True)
    if report_path:
        print(f"  → {report_path}", flush=True)
    print(f"{'='*60}\n", flush=True)

    return result


def _generate_report(strategy_name: str, pool_name: str, result_path: str,
                     skip_scenario: bool = False, skip_walk_forward: bool = False,
                     skip_monte_carlo: bool = False, skip_survival: bool = False) -> str:
    """调用 report.py 生成报告，返回报告路径"""
    import subprocess
    from datetime import datetime

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = BASE / "reports"
    report_dir.mkdir(exist_ok=True)
    report_path = report_dir / f"{ts}_{strategy_name}_{pool_name}.txt"

    cmd = [
        sys.executable, str(BASE / "report.py"),
        result_path,
        "--strategy", strategy_name,
        "--pool", pool_name,
        "--output", str(report_path),
    ]

    if skip_scenario:
        cmd.append("--skip-scenario")
    if skip_walk_forward:
        cmd.append("--skip-walk-forward")
    if skip_monte_carlo:
        cmd.append("--skip-monte-carlo")
    if skip_survival:
        cmd.append("--skip-survival")

    try:
        subprocess.run(cmd, check=True)
        return str(report_path)
    except subprocess.CalledProcessError as e:
        print(f"  [WARN] 报告生成失败", flush=True)
        return ""


def _append_top_strategy(name: str, pool: str, summary: dict):
    """追加一行到 top_strategy.md（不覆盖已有内容）"""
    top_file = BASE / "top_strategy.md"
    if not top_file.exists():
        return

    ev = summary.get("expected_value", 0)
    wr = summary.get("win_rate", 0)
    trades = summary.get("total_trades", 0)
    sharpe = summary.get("sharpe_ratio", 0)
    dd = summary.get("max_drawdown_pct", 0)
    ts = summary.get("timestamp", "")

    line = f"| {name} | {pool} | {ev:+.2f}% | {wr:.1f}% | {trades} | {dd:.1f}% | {sharpe:.2f} | {ts} |\n"

    with open(top_file, "a", encoding="utf-8") as f:
        f.write(line)


def main():
    ap = argparse.ArgumentParser(description="策略Pipeline: YAML→回测→报告→top_strategy.md")
    ap.add_argument("strategy", help="策略文件名(不含.yaml)")
    ap.add_argument("--pool", default="大蓝筹", help="标的池名称")
    ap.add_argument("--skip-scenario", action="store_true", help="跳过情景压力测试")
    ap.add_argument("--skip-walk-forward", action="store_true", help="跳过 Walk-Forward 验证")
    ap.add_argument("--skip-monte-carlo", action="store_true", help="跳过 Monte Carlo 验证")
    ap.add_argument("--skip-survival", action="store_true", help="跳过生存偏差检查")
    args = ap.parse_args()

    run_pipeline(args.strategy, args.pool,
                 skip_scenario=args.skip_scenario,
                 skip_walk_forward=args.skip_walk_forward,
                 skip_monte_carlo=args.skip_monte_carlo,
                 skip_survival=args.skip_survival)


if __name__ == "__main__":
    main()
