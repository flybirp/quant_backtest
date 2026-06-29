"""
Daily Select Pipeline — 每日选股/卖点扫描

功能：
1. 下载最新数据（tushare）
2. 扫描全市场买点信号
3. 扫描已持仓股票卖点信号

用法：
    # 扫描买点
    python3 daily_select_pipeline.py buy --strategy RS10_A1 --pool 大蓝筹

    # 扫描卖点（需要持仓文件）
    python3 daily_select_pipeline.py sell --strategy RS10_A1 --portfolio portfolio.json

    # 同时扫描买点和卖点
    python3 daily_select_pipeline.py scan --strategy RS10_A1 --pool 大蓝筹 --portfolio portfolio.json

    # 只下载数据
    python3 daily_select_pipeline.py download
"""

import sys
import json
import yaml
import time
import argparse
import subprocess
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent))

BASE = Path(__file__).parent
STRATEGIES_DIR = BASE / "strategies" / "rule"
DATA_DIR = Path("/Users/flybirp/Documents/mainland_data_2014")
PORTFOLIO_FILE = BASE / "portfolio.json"

# Tushare token
TUSHARE_TOKEN = "02f99406780174c10ce0d29ec35ecb79913a5c908a72fe7f2b80f9bd"


def download_data(start_date: str = None, end_date: str = None):
    """下载最新数据"""
    if not start_date:
        # 默认下载最近30天数据（增量更新）
        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y%m%d")
    if not end_date:
        end_date = datetime.now().strftime("%Y%m%d")

    print(f"📥 下载数据: {start_date} → {end_date}")

    # 1. 下载个股数据
    print("  [1/2] 下载个股数据...")
    stocklist = Path("/Users/flybirp/Documents/StockTradebyZ/stocklist.csv")
    if stocklist.exists():
        cmd = [
            "python3.10", "/Users/flybirp/Documents/StockTradebyZ/fetch_kline.py",
            "--start", start_date,
            "--end", end_date,
            "--stocklist", str(stocklist),
            "--exclude-boards", "bj",
            "--out", str(DATA_DIR),
            "--workers", "6"
        ]
        try:
            subprocess.run(cmd, check=True, env={**__import__('os').environ, "TUSHARE_TOKEN": TUSHARE_TOKEN})
            print("  ✅ 个股数据下载完成")
        except subprocess.CalledProcessError as e:
            print(f"  ❌ 个股数据下载失败: {e}")
            return False
    else:
        print(f"  ⚠️ 股票列表文件不存在: {stocklist}")

    # 2. 下载指数数据
    print("  [2/2] 下载指数数据...")
    index_dir = Path.home() / "Documents" / "mainland_index_data_2014"
    cmd = [
        "python3.10", "quant_utils/get_index_util.py",
        "--start_date", start_date,
        "--end_date", end_date,
        "--token", TUSHARE_TOKEN,
        "--save_dir", str(index_dir)
    ]
    try:
        subprocess.run(cmd, check=True)
        print("  ✅ 指数数据下载完成")
    except subprocess.CalledProcessError as e:
        print(f"  ❌ 指数数据下载失败: {e}")
        return False

    print("✅ 数据下载完成")
    return True


def load_strategy(strategy_name: str) -> dict:
    """加载策略配置"""
    ypath = STRATEGIES_DIR / f"{strategy_name}.yaml"
    if not ypath.exists():
        raise FileNotFoundError(f"策略文件不存在: {ypath}")

    with open(ypath) as f:
        cfg = yaml.safe_load(f)

    return cfg


def load_portfolio(portfolio_file: str = None) -> list:
    """加载持仓信息"""
    if portfolio_file:
        pfile = Path(portfolio_file)
    else:
        pfile = PORTFOLIO_FILE

    if not pfile.exists():
        print(f"  ⚠️ 持仓文件不存在: {pfile}")
        return []

    try:
        with open(pfile) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"  ❌ 持仓文件格式错误: {e}")
        return []

    # 兼容两种格式：直接列表 或 {holdings: [...]}
    if isinstance(data, list):
        portfolio = data
    elif isinstance(data, dict):
        portfolio = data.get('holdings', [])
    else:
        print(f"  ❌ 持仓文件格式不正确")
        return []

    # 校验和清理每条记录
    valid_portfolio = []
    for i, holding in enumerate(portfolio):
        # 必须有股票代码
        code = holding.get('code', '').strip()
        if not code:
            print(f"  ⚠️ 第{i+1}条记录缺少 'code' 字段，跳过")
            continue

        # 默认值处理
        valid_holding = {
            'code': code,
            'name': holding.get('name', ''),
            'buy_date': holding.get('buy_date', ''),
            'buy_price': _safe_float(holding.get('buy_price', 0)),
            'shares': _safe_int(holding.get('shares', 0)),
            'strategy': holding.get('strategy', ''),
        }

        # 校验买入价格
        if valid_holding['buy_price'] <= 0:
            print(f"  ⚠️ {code} 的 'buy_price' 无效 ({holding.get('buy_price')})，跳过")
            continue

        valid_portfolio.append(valid_holding)

    return valid_portfolio


def _safe_float(value, default: float = 0.0) -> float:
    """安全转换为float"""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value, default: int = 0) -> int:
    """安全转换为int"""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def scan_buy_signals(strategy_name: str, pool_name: str = "大蓝筹") -> list:
    """扫描买点信号"""
    from backend.main import _config_from_dict, _resolve_pool_codes
    from backend.backtest_engine import run_backtest

    print(f"\n🔍 扫描买点信号: {strategy_name} | {pool_name}")

    # 加载策略
    cfg = load_strategy(strategy_name)
    pool = _resolve_pool_codes(pool_name)
    cfg["stock_pool"] = pool

    # 运行回测（只取最后一天的信号）
    print(f"  扫描 {len(pool)} 只股票...")
    r = run_backtest(_config_from_dict(cfg))

    # 提取最近的买入信号
    buy_signals = []
    today = datetime.now().strftime("%Y-%m-%d")

    for trade in r.trades:
        if trade.get('buy_date') == today:
            buy_signals.append({
                'code': trade['code'],
                'buy_date': trade['buy_date'],
                'buy_price': trade['buy_price'],
                'reason': trade.get('buy_reason', ''),
                'strategy': strategy_name,
            })

    print(f"  ✅ 发现 {len(buy_signals)} 个买点信号")
    return buy_signals


def scan_sell_signals(strategy_name: str, portfolio: list) -> list:
    """扫描卖点信号

    ⚠️ 重要原则：买点和卖点必须使用同一个策略！
    不能用A策略买入，用B策略卖出。
    """
    from backend.main import _config_from_dict
    from backend.backtest_engine import run_backtest

    print(f"\n🔍 扫描卖点信号: {strategy_name}")

    if not portfolio:
        print("  ⚠️ 无持仓信息，跳过卖点扫描")
        return []

    # 加载策略
    cfg = load_strategy(strategy_name)

    sell_signals = []
    today = datetime.now().strftime("%Y-%m-%d")
    skipped_count = 0

    for holding in portfolio:
        code = holding.get('code')
        buy_price = holding.get('buy_price', 0)
        buy_date = holding.get('buy_date', '')
        holding_strategy = holding.get('strategy', '')

        if not code or not buy_price:
            continue

        # ⚠️ 强制限制：买点和卖点必须使用同一个策略
        if holding_strategy and holding_strategy != strategy_name:
            print(f"  ⚠️ 跳过 {code}: 买入策略={holding_strategy} ≠ 当前策略={strategy_name}")
            skipped_count += 1
            continue

        # 为单只股票运行回测
        cfg['stock_pool'] = [code]
        r = run_backtest(_config_from_dict(cfg))

        # 检查是否有卖出信号
        for trade in r.trades:
            if trade.get('sell_date') == today:
                # 计算盈亏
                profit_pct = (trade['sell_price'] - buy_price) / buy_price * 100

                # 判断卖出原因
                sell_reason = trade.get('sell_reason', '')
                is_stop_loss = profit_pct <= -cfg.get('stop_loss_pct', 25)
                is_take_profit = profit_pct >= cfg.get('take_profit_pct', 80)

                sell_signals.append({
                    'code': code,
                    'buy_date': buy_date,
                    'buy_price': buy_price,
                    'sell_date': trade['sell_date'],
                    'sell_price': trade['sell_price'],
                    'profit_pct': round(profit_pct, 2),
                    'sell_reason': sell_reason,
                    'is_stop_loss': is_stop_loss,
                    'is_take_profit': is_take_profit,
                    'hold_days': (datetime.strptime(trade['sell_date'], '%Y-%m-%d') -
                                  datetime.strptime(buy_date, '%Y-%m-%d')).days,
                    'strategy': strategy_name,
                })

    if skipped_count > 0:
        print(f"  ⚠️ 跳过 {skipped_count} 只股票（策略不匹配）")

    print(f"  ✅ 发现 {len(sell_signals)} 个卖点信号")
    return sell_signals


def print_buy_signals(signals: list):
    """打印买点信号"""
    if not signals:
        print("\n📊 买点信号: 无")
        return

    print(f"\n📊 买点信号: {len(signals)} 个")
    print("=" * 60)
    print(f"{'股票代码':<10} {'买入日期':<12} {'买入价':>10} {'原因'}")
    print("-" * 60)

    for s in signals:
        print(f"{s['code']:<10} {s['buy_date']:<12} {s['buy_price']:>10.2f} {s['reason']}")


def print_sell_signals(signals: list):
    """打印卖点信号"""
    if not signals:
        print("\n📊 卖点信号: 无")
        return

    print(f"\n📊 卖点信号: {len(signals)} 个")
    print("=" * 80)
    print(f"{'股票代码':<10} {'买入日期':<12} {'买入价':>10} {'卖出价':>10} {'盈亏%':>8} {'持仓天':>6} {'原因'}")
    print("-" * 80)

    for s in signals:
        profit_str = f"{s['profit_pct']:+.2f}%"
        if s['is_stop_loss']:
            profit_str = f"🔴{profit_str}"
        elif s['is_take_profit']:
            profit_str = f"🟢{profit_str}"

        print(f"{s['code']:<10} {s['buy_date']:<12} {s['buy_price']:>10.2f} "
              f"{s['sell_price']:>10.2f} {profit_str:>8} {s['hold_days']:>6} {s['sell_reason']}")


def save_signals(buy_signals: list, sell_signals: list, strategy_name: str):
    """保存信号到文件"""
    today = datetime.now().strftime("%Y%m%d")
    output_dir = BASE / "daily_signals"
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / f"{today}_{strategy_name}.json"

    data = {
        'date': today,
        'strategy': strategy_name,
        'buy_signals': buy_signals,
        'sell_signals': sell_signals,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(output_file, 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\n💾 信号已保存: {output_file}")
    return str(output_file)


def main():
    ap = argparse.ArgumentParser(description="每日选股/卖点扫描")
    subparsers = ap.add_subparsers(dest='command', help='命令')

    # download 命令
    download_parser = subparsers.add_parser('download', help='下载数据')
    download_parser.add_argument('--start', help='开始日期 (YYYYMMDD)')
    download_parser.add_argument('--end', help='结束日期 (YYYYMMDD)')

    # buy 命令
    buy_parser = subparsers.add_parser('buy', help='扫描买点')
    buy_parser.add_argument('--strategy', required=True, help='策略名称')
    buy_parser.add_argument('--pool', default='大蓝筹', help='股票池')

    # sell 命令
    sell_parser = subparsers.add_parser('sell', help='扫描卖点')
    sell_parser.add_argument('--strategy', required=True, help='策略名称')
    sell_parser.add_argument('--portfolio', help='持仓文件 (JSON)')

    # scan 命令
    scan_parser = subparsers.add_parser('scan', help='同时扫描买点和卖点')
    scan_parser.add_argument('--strategy', required=True, help='策略名称')
    scan_parser.add_argument('--pool', default='大蓝筹', help='股票池')
    scan_parser.add_argument('--portfolio', help='持仓文件 (JSON)')

    args = ap.parse_args()

    if not args.command:
        ap.print_help()
        return

    if args.command == 'download':
        download_data(args.start, args.end)

    elif args.command == 'buy':
        buy_signals = scan_buy_signals(args.strategy, args.pool)
        print_buy_signals(buy_signals)
        save_signals(buy_signals, [], args.strategy)

    elif args.command == 'sell':
        portfolio = load_portfolio(args.portfolio)
        sell_signals = scan_sell_signals(args.strategy, portfolio)
        print_sell_signals(sell_signals)
        save_signals([], sell_signals, args.strategy)

    elif args.command == 'scan':
        # 先下载最新数据
        download_data()

        # 扫描买点
        buy_signals = scan_buy_signals(args.strategy, args.pool)
        print_buy_signals(buy_signals)

        # 扫描卖点
        portfolio = load_portfolio(args.portfolio)
        sell_signals = scan_sell_signals(args.strategy, portfolio)
        print_sell_signals(sell_signals)

        # 保存信号
        save_signals(buy_signals, sell_signals, args.strategy)


if __name__ == "__main__":
    main()
