# Research Agent 升级总结

基于 RD-Agent 源码 (commit: main, 2025) 的实际架构，完整复刻了核心逻辑。

---

## 已实现

### 1. 两段式 LLM 调用

```
make research
  │
  ├── Stage 1: Action Selection
  │   输入: 当前指标 + SOTA + 历史方向
  │   输出: {"direction": "reduce_underwater", "reason": "水下占比76.2%, 远超50%阈值"}
  │
  └── Stage 2: Hypothesis Generation
      输入: 方向 + 当前参数值 + 渐进式阶段提示
      输出: {"parameter_changes": {...}, "expected_effects": {...}}
```

### 2. 渐进式提示词（Phase 1/2/3）

- Phase 1 (round 1-3): 只调 stop_loss_pct, take_profit_pct, ±20%
- Phase 2 (round 4-6): 增加 state_machine_params 核心阈值
- Phase 3 (round 7+): 全部参数可用，包括 entry_ladder/exit_ladder

### 3. 约束优先的 SOTA 决策

```
MaxDD > 20% → 直接拒绝，EV再高也不换
Bootstrap CI跨0 → 直接拒绝
通过约束 + EV提升 → 接受为新SOTA
通过约束 + EV持平但 MaxDD 改善 >5pp → 接受
其他 → 拒绝
```

### 4. 方向切换逻辑

- 同一方向连续 3 轮无提升 → 自动切到 explore_new_params
- 总计 9 轮失败 → 强制 explore
- LLM 在 Action Selection 阶段读取历史失败记录

### 5. 参数知识嵌入

LLM 拿到每个参数的含义、取值范围、调参方向说明，不需要猜。

---

## 文件清单

| 文件 | 内容 |
|------|------|
| `backend/research_agent.py` | SOTATracker + 两阶段提示词 + 上下文构建器 |
| `backend/research_bridge.py` | 指标提取 + prompt生成 + LLM响应处理 + YAML自动修改 |
| `RESEARCH_AGENT_UPGRADE.md` | 升级方案讨论文档 |
| `Makefile` | `make research` 命令 |

---

## 使用方式

```bash
# 生成研究prompt
make research

# 输出会保存到 /tmp/research_prompt.txt
# 包含两段完整prompt，喂给任何LLM即可
```

LLM 返回的 JSON 会被 `apply_llm_response()` 解析，自动：
1. 修改 YAML 参数并保存为新文件 (`zzh7.3_r1_20260624.yaml`)
2. 更新 SOTA tracker
3. 返回 `run_backtest` 或 `try_again` 指令

---

## 与 RD-Agent 的对照

| RD-Agent 做法 | 我们的实现 | 位置 |
|---------------|-----------|------|
| `QlibQuantHypothesisGen.prepare_context()` | `build_action_selection_context()` + `build_hypothesis_context()` | `research_agent.py` |
| `factor_hypothesis_specification` (渐进难度) | Phase 1/2/3 条件分支 | `HYPOTHESIS_SYSTEM_PROMPT` |
| `factor_feedback_generation` (SOTA对比) | `SOTATracker.update()` 约束优先决策 | `research_agent.py:156` |
| `QlibFactorExperiment2Feedback` (JSON输出) | `hypothesis_output_format` schema | `HYPOTHESIS_SYSTEM_PROMPT` |
| `EnvController` + `LinearThompsonTwoArm` (MAB) | 不需要 — 规则策略参数空间小，LLM 直接选方向更合适 | — |
| `process_results()` (Current vs SOTA 对比) | `build_action_selection_context()` 里的 Current State + SOTA 对比 | `research_agent.py:260` |
| `get_scenario_all_desc()` (场景描述) | `HYPOTHESIS_SYSTEM_PROMPT` 里的 Parameter Knowledge | `research_agent.py:120-160` |
