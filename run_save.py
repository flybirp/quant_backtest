"""运行回测并保存全量交易到 results/ 目录"""
import sys, json, yaml, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from backend.main import _config_from_dict, _resolve_pool_codes
from backend.backtest_engine import run_backtest

BASE = Path(__file__).parent
RESULTS_DIR = BASE / "results"
RESULTS_DIR.mkdir(exist_ok=True)

def run_and_save(strategy_file, pool_name):
    """运行回测，保存交易数据和汇总到 JSON"""
    # Try strategies/rule/ first, then strategies/ flat
    yaml_path = BASE / "strategies" / "rule" / f"{strategy_file}.yaml"
    if not yaml_path.exists():
        yaml_path = BASE / "strategies" / f"{strategy_file}.yaml"
    if not yaml_path.exists():
        print(f"  [ERROR] 策略文件不存在: {strategy_file}")
        return None
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    pool = _resolve_pool_codes(pool_name)
    cfg["stock_pool"] = pool

    print(f"\n▶ {strategy_file} | {pool_name} {len(pool)}只", flush=True)
    t0 = time.time()
    r = run_backtest(_config_from_dict(cfg))
    elapsed = time.time() - t0

    # 保存全量交易（直接存，_trade_to_dict 已返回 dict）
    trades_data = [
        {k: str(v)[:10] if k.endswith("_date") and v else v for k, v in t.items()}
        for t in r.trades
    ]

    # 汇总统计
    summary = {
        "strategy": strategy_file,
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
    }

    out = {"summary": summary, "trades": trades_data, "equity_curve": r.equity_curve}
    fname = f"{strategy_file}_{pool_name}.json"
    outpath = RESULTS_DIR / fname
    with open(outpath, "w") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"  ✅ [{elapsed:.0f}s] 交易:{r.total_trades} EV:{r.expected_value:+.2f}% → saved {fname}")

    return r


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("strategy", help="策略文件名(不含.yaml)")
    ap.add_argument("--pool", default="大蓝筹", help="标的池名称")
    args = ap.parse_args()
    run_and_save(args.strategy, args.pool)
