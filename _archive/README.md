# _archive — 归档文件说明

> 创建时间：2026-06-30
>
> 以下文件均为项目开发过程中的一次性脚本、临时实验和已完成任务的产物，功能已被核心模块覆盖或任务已结束，保留备查。

---

## `oneoff_runs/` — 一次性全量回测脚本

用于针对特定策略版本跑全市场回测的临时脚本，硬编码了策略名和参数，任务完成后不再复用。

| 文件 | 功能 |
|---|---|
| `_run_full_pool.py` | R6 版恐慌错杀策略跑全量（2014-2025） |
| `_run_r9_full.py` | R9 版恐慌错杀策略跑全量 |
| `_run_rs10_full.py` | RS10 变体策略跑全量（修改了相对强度和比率参数） |
| `_run_all_strategies.sh` | 批量运行 `strategies/rule/` 下所有策略的大蓝筹回测 |
| `_run_missing.sh` | 补跑之前因故未完成的策略列表（含 30+ 个策略） |
| `_run_remaining.sh` | 跑剩余未执行的策略（与 `_run_missing.sh` 类似） |
| `_run_missing.log` | 补跑任务的运行日志（557KB） |
| `_grid_worker.py` | 早期 grid search worker，后被 `run_grid_worker.py` 替代 |

---

## `grid_search/` — 量能爆发底部反转策略参数搜索

针对"量能爆发底部反转"策略的三轮网格调优全套文件，已找到最优参数组合并固化到策略 YAML 中。

| 文件 | 功能 |
|---|---|
| `run_volume_reversal_grid.py` | 粗搜索主脚本（subprocess 调 worker） |
| `run_grid_worker.py` | 粗搜索 worker（dt/vr/sd/dbl 4 维 × 36 组） |
| `run_grid_search.sh` | 粗搜索 shell 脚本（nohup 后台运行） |
| `grid_search_volume_reversal_results.json/.jsonl` | 粗搜索结果 |
| `grid_search.log` | 粗搜索运行日志 |
| `run_grid_worker_deep.py` | 深层搜索 worker（10 维参数空间） |
| `run_grid_search_deep.sh` | 深层搜索 shell 脚本（35 组参数） |
| `grid_search_volume_reversal_deep.json/.jsonl` | 深层搜索结果 |
| `grid_search_deep.log` | 深层搜索运行日志 |
| `run_grid_search_fine.sh` | 细搜索 shell 脚本（围绕 TOP1 最优微调） |
| `grid_search_volume_reversal_fine.json/.jsonl` | 细搜索结果 |
| `grid_search_fine.log` | 细搜索运行日志 |

---

## `cross_pool/` — 跨标的池横向对比

对 TOP1-5 五个最优参数组合在大蓝筹/科创板/创业板/全量四个池上的横向对比回测。全量池因数据量大，拆分为按股票代码首数字分块并行执行后合并。

| 文件 | 功能 |
|---|---|
| `run_cross_pool.py` | 跨池对比主脚本（串行版） |
| `run_cross_pool.sh` | 跨池对比 shell 脚本（v1，直接调 worker） |
| `run_cross_pool_v2.sh` | 跨池对比 shell 脚本（v2，分块并行版） |
| `run_cross_pool_worker.py` | 单组回测 worker（接收策略名+池名+参数） |
| `run_chunk_worker.py` | 分块 worker（按股票代码首数字拆分全量池） |
| `merge_chunks.py` | 合并分块结果，计算加权 EV/胜率/盈亏比 |
| `merge_all_results.py` | 合并所有策略×池的结果，生成完整对比报告 |
| `cross_pool_report.html` | 生成的可视化 HTML 对比报告 |
| `cross_pool_results.json` | 跨池对比最终结果（JSON） |
| `cross_pool_tmp/` | 分块并行执行的临时结果目录（52 个中间文件） |

---

## `pool_comparison/` — 标的池/分批阶梯对比

| 文件 | 功能 |
|---|---|
| `run_pool_comparison.py` | 9 个正 EV 策略 × 2 个标的池（大蓝筹 vs 创业板）对比回测 |
| `pool_comparison_results.json` | 池对比回测结果 |
| `run_ladder_comparison.py` | 分批建仓 vs 原策略的阶梯对比回测（验证分批是否更优） |
| `ladder_comparison_results.json` | 阶梯对比回测结果 |

---

## `brainstorm/` — 脑暴策略批量回测

项目早期的策略探索阶段，通过"脑暴"方式生成了大量候选策略，在全市场 5062 只股票上批量验证。

| 文件 | 功能 |
|---|---|
| `run_brainstorm_strategies.py` | 两轮脑暴策略批量回测（18 个配置，多进程并行） |
| `analyze_brainstorm_results.py` | 分析脑暴结果，生成可视化 HTML 报告 |
| `brainstorm_results_fullmarket.json` | 全市场脑暴回测结果数据 |

---

## `analysis/` — 其他一次性分析

| 文件 | 功能 |
|---|---|
| `analyze_new_strategies_v3.py` | 新增策略结果分析脚本（v3），修正 signal 模式权益曲线异常后重排评分 |
| `run_save.py` | 批量运行回测并保存全量交易数据到 `results/` 目录 |
| `latest_report.txt` | 2026-06-24 生成的量化策略综合分析报告快照（36KB） |
