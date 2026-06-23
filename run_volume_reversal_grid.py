"""量能爆发底部反转策略 - 网格搜索
用 subprocess.run 逐个执行 worker，每组独立进程。
直接前台运行，不依赖 nohup。
"""
import subprocess
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PY = sys.executable
WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'run_grid_worker.py')
RES_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'grid_search_volume_reversal_results.json')

# 粗搜索参数
DT = [20, 30, 40]
VR = [90, 95]
SD = [3, 5]
DBL = [20, 30, 40]

TOTAL = len(DT) * len(VR) * len(SD) * len(DBL)
results = []
N = 0

print(f"\n量能爆发底部反转策略 - 网格搜索（subprocess版）", flush=True)
print(f"共 {TOTAL} 组参数", flush=True)
print(f"每组约65秒，预计 {TOTAL * 65 // 60} 分钟\n", flush=True)

t0 = time.time()

for dt in DT:
    for vr in VR:
        for sd in SD:
            for dbl in DBL:
                N += 1
                try:
                    proc = subprocess.run(
                        [PY, WORKER, str(dt), str(vr), str(sd), str(dbl)],
                        capture_output=True, text=True, timeout=300,
                    )
                    if proc.returncode == 0 and proc.stdout.strip():
                        out = json.loads(proc.stdout.strip())
                        results.append(out)
                        ev = out['position_ev']
                        wr = out['position_win_rate']
                        marker = "✓" if ev > 0 else "✗"
                        print(f"  [{N}/{TOTAL}] {marker} EV={ev:+.2f}% WR={wr:.0f}% | dt={dt} vr={vr} sd={sd} dbl={dbl}", flush=True)
                    else:
                        results.append({"params": {"downtrend_days": dt, "vol_rank_threshold": vr, "sustain_days": sd, "double_bottom_lookback": dbl}, "error": proc.stderr[:100] if proc.stderr else "unknown"})
                        print(f"  [{N}/{TOTAL}] ✗ FAIL | dt={dt} vr={vr} sd={sd} dbl={dbl}", flush=True)
                except Exception as e:
                    results.append({"params": {"downtrend_days": dt, "vol_rank_threshold": vr, "sustain_days": sd, "double_bottom_lookback": dbl}, "error": str(e)})
                    print(f"  [{N}/{TOTAL}] ✗ ERROR: {e}", flush=True)

                # 即时保存
                with open(RES_FILE, 'w') as f:
                    json.dump({"results": results, "completed": N, "total": TOTAL}, f, ensure_ascii=False, indent=2)

elapsed = time.time() - t0
print(f"\n粗搜索完成！耗时 {elapsed/60:.1f} 分钟", flush=True)

# 排序输出
valid = [r for r in results if "error" not in r and r.get("position_count", 0) >= 5]
valid.sort(key=lambda x: x["position_ev"], reverse=True)

print(f"\nTOP 20:", flush=True)
print(f"{'排名':>4s} {'EV':>8s} {'胜率':>6s} {'盈亏比':>6s} {'持仓数':>6s} | dt vr sd dbl", flush=True)
print("-" * 70, flush=True)
for i, r in enumerate(valid[:20], 1):
    p = r['params']
    print(f"{i:4d} {r['position_ev']:+7.2f}% {r['position_win_rate']:5.1f}% {r['position_pl_ratio']:6.2f} "
          f"{r['position_count']:6d} | {p['downtrend_days']} {p['vol_rank_threshold']} {p['sustain_days']} {p['double_bottom_lookback']}", flush=True)

# 保存最终结果
with open(RES_FILE, 'w') as f:
    json.dump({"results": results, "top20": valid[:20]}, f, ensure_ascii=False, indent=2)
print(f"\n结果已保存到 {RES_FILE}", flush=True)
