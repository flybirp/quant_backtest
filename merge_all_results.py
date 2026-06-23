"""合并所有分块结果，生成完整的跨标的池对比报告"""
import json
from pathlib import Path
from collections import defaultdict

RDIR = Path(__file__).parent / 'cross_pool_tmp'

STRATEGIES = ['TOP1', 'TOP2', 'TOP3', 'TOP4', 'TOP5']
POOLS = ['大蓝筹', '科创板', '创业板', '全量']

# 全量分块定义
CHUNK_MAP = {
    'TOP1': ['0', '3', '60', '68'],
    'TOP2': ['0', '3', '60', '68'],
    'TOP3': ['0', '3', '60', '68'],
    'TOP4': ['0', '3', '60', '68'],
    'TOP5': ['0', '3', '600', '601', '603', '605', '68'],  # TOP5的60块需要更细
}


def load_pool_result(sname, pname):
    """加载某策略×某标的池的结果"""
    fpath = RDIR / f'{sname}_{pname}.json'
    if fpath.exists() and fpath.stat().st_size > 0:
        with open(fpath) as f:
            return json.load(f)
    return None


def merge_full_market(sname):
    """合并全量分块结果"""
    chunks = CHUNK_MAP[sname]
    chunks_data = []
    for c in chunks:
        fpath = RDIR / f'{sname}_全量_{c}.json'
        if not fpath.exists() or fpath.stat().st_size == 0:
            print(f"  Warning: {fpath} missing")
            continue
        with open(fpath) as f:
            data = json.load(f)
        chunks_data.append(data)

    if not chunks_data:
        return None

    total_positions = sum(d['positions'] for d in chunks_data)
    total_trades = sum(d['trades'] for d in chunks_data)
    if total_positions == 0:
        return {'strategy': sname, 'pool': '全量', 'ev': 0, 'win_rate': 0, 'pl_ratio': 0, 'positions': 0, 'trades': 0}

    weighted_ev = sum(d['ev'] * d['positions'] for d in chunks_data) / total_positions
    weighted_wr = sum(d['win_rate'] * d['positions'] for d in chunks_data) / total_positions
    weighted_plr = sum(d['pl_ratio'] * d['positions'] for d in chunks_data) / total_positions

    return {
        'strategy': sname, 'pool': '全量',
        'ev': round(weighted_ev, 2),
        'win_rate': round(weighted_wr, 1),
        'pl_ratio': round(weighted_plr, 2),
        'positions': total_positions,
        'trades': total_trades,
    }


def main():
    results = {}

    for sname in STRATEGIES:
        # 3个小池子直接读
        for pname in ['大蓝筹', '科创板', '创业板']:
            r = load_pool_result(sname, pname)
            if r:
                results[(sname, pname)] = r
            else:
                results[(sname, pname)] = {'strategy': sname, 'pool': pname, 'ev': None, 'win_rate': None, 'pl_ratio': None, 'positions': 0, 'trades': 0}

        # 全量合并
        r = merge_full_market(sname)
        if r:
            results[(sname, '全量')] = r
        else:
            results[(sname, '全量')] = {'strategy': sname, 'pool': '全量', 'ev': None, 'win_rate': None, 'pl_ratio': None, 'positions': 0, 'trades': 0}

    # 保存完整JSON
    output = []
    for (sname, pname), m in results.items():
        output.append(m)
    with open('cross_pool_results.json', 'w') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    # 打印表格
    print("\n" + "=" * 100)
    print("量能爆发底部反转 - 跨标的池横向对比")
    print("回测区间: 20240101 ~ 20260606")
    print("=" * 100)

    for metric_name, metric_key, fmt in [
        ('EV(%)', 'ev', '{:+.2f}'),
        ('胜率(%)', 'win_rate', '{:.1f}'),
        ('盈亏比', 'pl_ratio', '{:.2f}'),
        ('持仓数', 'positions', '{:.0f}'),
    ]:
        print(f"\n{'─' * 80}")
        print(f"  {metric_name}")
        print(f"{'─' * 80}")
        print(f"{'策略':<8}", end='')
        for pname in POOLS:
            print(f" │ {pname:>12}", end='')
        print()
        print(f"{'':─<8}", end='')
        for _ in POOLS:
            print(f"─┼─────────────", end='')
        print()

        for sname in STRATEGIES:
            print(f"{sname:<8}", end='')
            for pname in POOLS:
                v = results.get((sname, pname), {}).get(metric_key)
                if v is None:
                    print(f" │ {'N/A':>12}", end='')
                else:
                    print(f" │ {fmt.format(v):>12}", end='')
            print()

    # 找出每个池子的最佳策略
    print(f"\n{'=' * 80}")
    print("  各标的池最佳策略")
    print(f"{'=' * 80}")
    for pname in POOLS:
        best_s = None
        best_ev = -999
        for sname in STRATEGIES:
            v = results.get((sname, pname), {}).get('ev')
            if v is not None and v > best_ev:
                best_ev = v
                best_s = sname
        if best_s:
            r = results[(best_s, pname)]
            print(f"  {pname:>6}: {best_s} (EV={r['ev']:+.2f}%, 胜率={r['win_rate']:.1f}%, 盈亏比={r['pl_ratio']:.2f}, 持仓={r['positions']})")

    # 标的池对比洞察
    print(f"\n{'=' * 80}")
    print("  标的池横向洞察 (以TOP1为例)")
    print(f"{'=' * 80}")
    for pname in POOLS:
        r = results.get(('TOP1', pname), {})
        print(f"  {pname:>6}: EV={r.get('ev', 'N/A'):>+6.2f}%  胜率={r.get('win_rate', 0):>5.1f}%  盈亏比={r.get('pl_ratio', 0):>5.2f}  持仓={r.get('positions', 0):>5}")

    print(f"\n结果已保存到 cross_pool_results.json")


if __name__ == '__main__':
    main()
