"""合并分块回测结果，计算全量池的综合指标
加权EV = Σ(EV_i * N_i) / Σ(N_i)
加权胜率 = Σ(WR_i * N_i) / Σ(N_i)
加权盈亏比 = Σ(PLR_i * N_i) / Σ(N_i)
总持仓数 = Σ(N_i)
"""
import json, sys
from pathlib import Path

RDIR = Path(__file__).parent / 'cross_pool_tmp'

# 策略参数
STRATEGIES = {
    'TOP1': {'dt': 100, 'vr': 95, 'vratio': 30, 'sd': 3, 'sr': 7, 'dbl': 60, 'dbt': 30, 'bp': 30, 'ra': 10, 'el': 15},
    'TOP2': {'dt': 100, 'vr': 95, 'vratio': 30, 'sd': 3, 'sr': 7, 'dbl': 60, 'dbt': 30, 'bp': 30, 'ra': 15, 'el': 15},
    'TOP3': {'dt': 100, 'vr': 95, 'vratio': 30, 'sd': 3, 'sr': 7, 'dbl': 60, 'dbt': 30, 'bp': 30, 'ra': 10, 'el': 20},
    'TOP4': {'dt': 90,  'vr': 95, 'vratio': 30, 'sd': 3, 'sr': 7, 'dbl': 60, 'dbt': 20, 'bp': 30, 'ra': 15, 'el': 15},
    'TOP5': {'dt': 70,  'vr': 95, 'vratio': 30, 'sd': 3, 'sr': 7, 'dbl': 60, 'dbt': 20, 'bp': 30, 'ra': 15, 'el': 15},
}

CHUNKS = ['0', '3', '60', '68']  # 全量拆分的块


def merge_chunks(sname):
    """合并某策略的全量分块结果"""
    chunks_data = []
    for c in CHUNKS:
        fpath = RDIR / f'{sname}_全量_{c}.json'
        if not fpath.exists() or fpath.stat().st_size == 0:
            print(f"  Warning: {fpath} missing, skipping")
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

    # 加权EV: 各块的持仓收益按持仓数加权
    weighted_ev = sum(d['ev'] * d['positions'] for d in chunks_data) / total_positions
    weighted_wr = sum(d['win_rate'] * d['positions'] for d in chunks_data) / total_positions
    weighted_plr = sum(d['pl_ratio'] * d['positions'] for d in chunks_data) / total_positions

    return {
        'strategy': sname,
        'pool': '全量',
        'ev': round(weighted_ev, 2),
        'win_rate': round(weighted_wr, 1),
        'pl_ratio': round(weighted_plr, 2),
        'positions': total_positions,
        'trades': total_trades,
    }


if __name__ == '__main__':
    # 先合并TOP1的全量结果
    result = merge_chunks('TOP1')
    print(json.dumps(result, ensure_ascii=False))

    # 保存到文件
    with open(RDIR / 'TOP1_全量.json', 'w') as f:
        json.dump(result, f, ensure_ascii=False)
