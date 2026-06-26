"""
Research Bridge — feeds backtest data into the Research Agent.

Takes result JSONs, extracts metrics, builds two-stage prompts
(action selection → hypothesis generation), tracks SOTA.

Usage:
  python -m backend.research_bridge results/zzh*.json --sota-config strategies/rule/zzh7.3.yaml
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE))

from backend.research_agent import (
    ACTION_SELECTION_SYSTEM,
    HYPOTHESIS_SYSTEM_PROMPT,
    SOTATracker,
    build_action_selection_context,
    build_hypothesis_context,
)


def load_strategy_config(yaml_path: str) -> dict:
    """Load strategy YAML as dict."""
    import yaml
    with open(yaml_path) as f:
        return yaml.safe_load(f)


def extract_metrics(json_path: str) -> dict[str, Any]:
    """Extract summary metrics from a backtest result JSON."""
    with open(json_path) as f:
        data = json.load(f)

    s = data.get("summary", {})
    equity = data.get("equity_curve", [])
    trades = data.get("trades", [])

    metrics = {
        "ev": s.get("expected_value", 0),
        "win_rate": s.get("win_rate", 0),
        "sharpe": s.get("sharpe_ratio", 0),
        "max_dd": abs(s.get("max_drawdown_pct", 0)),
        "trades": s.get("total_trades", 0),
        "cagr": s.get("annual_return_pct", 0),
    }

    # Compute additional metrics
    if equity:
        from analytics.risk import drawdown_recovery_stats
        from analytics.statistics import bootstrap_ev_ci
        from analytics.capacity import turnover_analysis

        rec = drawdown_recovery_stats(equity)
        metrics["underwater"] = rec.get("underwater_ratio", 50)

        ci_low, ci_high, _ = bootstrap_ev_ci(trades)
        metrics["ci_low"] = ci_low
        metrics["ci_high"] = ci_high

        turnover = turnover_analysis(trades, s.get("initial_capital", 100000))
        metrics["turnover"] = turnover.get("turnover_rate_annual", 0)

    return metrics


def generate_research_round(
    result_json_path: str,
    strategy_yaml_path: str,
    sota: SOTATracker,
    past_rounds: list[dict],
) -> dict[str, str]:
    """
    Generate prompts for one full research round.

    Returns:
        {
            "stage1_prompt": str,   # Action selection prompt
            "stage2_prompt": str,   # Hypothesis generation prompt
            "context": dict,        # Structured context for LLM
            "sota_state": dict,     # Current SOTA state
        }
    """
    current_metrics = extract_metrics(result_json_path)
    current_config = load_strategy_config(strategy_yaml_path)

    # Stage 1: Action Selection
    action_context = build_action_selection_context(current_metrics, sota, past_rounds)
    stage1 = f"{ACTION_SELECTION_SYSTEM}\n\n---\n\n{action_context}"

    # Default direction if this is the first round
    direction = sota.current_direction or _infer_direction(current_metrics)

    # Stage 2: Hypothesis Generation
    hypo_context = build_hypothesis_context(current_config, current_metrics, sota, direction)

    # Render the system prompt with Jinja2-style template (simple string replace)
    stage2 = _render_template(HYPOTHESIS_SYSTEM_PROMPT, hypo_context)

    return {
        "stage1_prompt": stage1,
        "stage2_prompt": stage2,
        "context": hypo_context,
        "sota_state": {
            "best_ev": sota.best_ev,
            "round": sota.round + 1,
            "direction": direction,
            "consecutive_failures": sota.consecutive_failures,
        },
    }


def _infer_direction(metrics: dict) -> str:
    """Infer initial direction from metrics."""
    if metrics.get("max_dd", 0) > 20:
        return "reduce_drawdown"
    if metrics.get("win_rate", 100) < 60:
        return "improve_win_rate"
    if metrics.get("trades", 0) < 500:
        return "increase_signals"
    if metrics.get("underwater", 100) > 50:
        return "reduce_underwater"
    return "fine_tune"


def _render_template(template: str, context: dict) -> str:
    """Simple Jinja2-like template rendering ({{ var }} replacement)."""
    import re

    def replace(match):
        key = match.group(1).strip()
        # Handle if/else blocks
        if "if " in key:
            parts = key.split("if ", 1)
            condition_body = parts[1] if len(parts) > 1 else ""
            # Parse: "round <= 3" from context
            if "round" in condition_body:
                val = context.get("round", 0)
                comparator = condition_body.strip()
                result = eval(comparator.replace("round", str(val)).replace("<=", "<="))
                return "% if block" if result else "% else block"
            return ""
        return str(context.get(key, ""))

    # Simple variable substitution
    for key, value in context.items():
        template = template.replace("{{ " + str(key) + " }}", str(value))
        template = template.replace("{{" + str(key) + "}}", str(value))

    # Strip Jinja2 control blocks ({% ... %}) — the caller handles phase logic
    import re as re_mod
    template = re_mod.sub(r'{%[^%]*%}', '', template)

    return template


def apply_llm_response(
    response_json: dict,
    strategy_yaml_path: str,
    sota: SOTATracker,
    result_json_path: str,
) -> dict:
    """
    Apply the LLM's hypothesis response:
    1. Extract parameter changes
    2. Update strategy YAML
    3. Update SOTA tracker
    4. Return next-action instructions

    Args:
        response_json: Parsed JSON from LLM (hypothesis output format)
        strategy_yaml_path: Path to strategy YAML to modify
        sota: SOTATracker instance
        result_json_path: Path to the result JSON for this round

    Returns:
        {"action": "run_backtest" | "accept_final" | "switch_direction",
         "parameter_changes": {...},
         "yaml_path": modified YAML path,
         "sota_updated": bool}
    """
    import yaml
    import copy
    from datetime import datetime

    current_config = load_strategy_config(strategy_yaml_path)
    current_metrics = extract_metrics(result_json_path)

    param_changes = response_json.get("parameter_changes", {})
    direction = response_json.get("direction", sota.current_direction)

    # Apply parameter changes to config
    new_config = copy.deepcopy(current_config)
    for param, value in param_changes.items():
        # Handle nested params (state_machine_params.xxx)
        if param.startswith("state_machine_params."):
            key = param.split(".", 1)[1]
            new_config.setdefault("state_machine_params", {})[key] = value
        else:
            new_config[param] = value

    # Save modified YAML
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = Path(strategy_yaml_path).stem
    new_yaml_path = Path(strategy_yaml_path).parent / f"{base_name}_r{sota.round+1}_{ts}.yaml"
    with open(new_yaml_path, "w") as f:
        yaml.dump(new_config, f, allow_unicode=True, default_flow_style=False)

    # Update SOTA
    ev = current_metrics.get("ev", 0)
    max_dd = current_metrics.get("max_dd", 0)
    ci_low = current_metrics.get("ci_low", 1)

    accepted, reason = sota.update(ev, max_dd, ci_low, new_config, current_metrics, direction)

    return {
        "action": "run_backtest" if accepted else "try_again",
        "sota_updated": accepted,
        "reason": reason,
        "parameter_changes": param_changes,
        "yaml_path": str(new_yaml_path),
        "next_direction": "explore_new_params" if sota.should_switch_direction else direction,
        "consecutive_failures": sota.consecutive_failures,
    }


if __name__ == "__main__":
    print("Research Bridge — use via 'make research' or import programmatically.")
    print("See RESEARCH_AGENT_UPGRADE.md for usage.")
