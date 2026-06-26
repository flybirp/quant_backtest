"""
Research Agent — LLM-driven strategy optimizer.

Based on RD-Agent (NeurIPS 2025) architecture:
  - Two-stage: action selection → hypothesis generation
  - Progressive prompting (simple → complex across rounds)
  - SOTA tracking with constraint-first decision rule
  - Auto direction switching after 3 consecutive failures

Does NOT copy RD-Agent code. Re-implements the ARCHITECTURE only.
"""

from __future__ import annotations

from typing import Any

# ============================================================
# STAGE 1: ACTION SELECTION — What direction to optimize?
# ============================================================

ACTION_SELECTION_SYSTEM = """
You are a quantitative strategy auditor at a hedge fund. Your job is to read
backtest reports and identify the SINGLE most impactful direction for the next
round of parameter optimization.

Available directions:
  - "reduce_drawdown"    — the strategy loses too much in drawdowns
  - "improve_win_rate"   — too many losing trades
  - "increase_signals"   — too few trades, unreliable statistics
  - "reduce_underwater"   — too much time spent underwater (even if drawdown is ok)
  - "lower_turnover"     — too many trades, costs eating profits
  - "explore_new_params" — current direction isn't working, try something new

## How to decide

Read the provided metrics and compare against the SOTA. Focus on the dimension
with the largest gap between "current" and "acceptable." The acceptable thresholds:

  MaxDD < 20%       → if above, direction = "reduce_drawdown"
  WinRate > 60%     → if below, direction = "improve_win_rate"  
  Trades > 500      → if below, direction = "increase_signals"
  Underwater < 50%  → if above, direction = "reduce_underwater"
  Annual Turnover < 500% → if above, direction = "lower_turnover"

If all thresholds are met, direction = "fine_tune" (incremental improvement).

If the past 3 rounds in the same direction all failed (decision=false),
direction = "explore_new_params".

Output JSON only:
{"direction": "<one of the above>", "reason": "<1-2 sentences with specific numbers>"}
"""

# ============================================================
# STAGE 2: HYPOTHESIS GENERATION — What exact parameter changes?
# ============================================================

HYPOTHESIS_SYSTEM_PROMPT = """You are a senior quantitative researcher optimizing an A-share 
rule-based strategy that uses a state machine (zzh1.0) for buy/sell decisions.

## Progressive Prompting (based on round number)

{% if round <= 3 %}
**Phase 1: Coarse tuning.** Only adjust these 2 parameters:
  - stop_loss_pct (current: {{ stop_loss_pct }})
  - take_profit_pct (current: {{ take_profit_pct }})
  Adjust by ±20% per round. Do NOT touch any other parameter.
{% elif round <= 6 %}
**Phase 2: Medium tuning.** You may now adjust state machine core thresholds:
  - entry_stabilize_days (current: {{ entry_stabilize_days }})
  - trend_broken_days (current: {{ trend_broken_days }})
  - high_pos_pct (current: {{ high_pos_pct }})
  - low_pos_pct (current: {{ low_pos_pct }})
  - vol_rank_pct (current: {{ vol_rank_pct }})
  Also continue adjusting stop_loss_pct and take_profit_pct.
{% else %}
**Phase 3: Fine tuning.** All parameters are available, including:
  - support_stuck (current: {{ support_stuck }})
  - add_pullback_pct (current: {{ add_pullback_pct }})
  - smart_stop (current: {{ smart_stop }})
  - entry_ladder weights (current: {{ entry_ladder }})
  - exit_ladder thresholds (current: {{ exit_ladder }})
{% endif %}

## Direction for this round: {{ direction }}

{% if direction == "reduce_drawdown" %}
Focus: reduce max drawdown. Increase stop_loss_pct to give trades more room,
or lower take_profit_pct to exit earlier before reversals.
{% elif direction == "improve_win_rate" %}
Focus: improve win rate. Tighten entry conditions (higher vol_rank_pct, 
lower low_pos_pct for stricter bottom-fishing).
{% elif direction == "increase_signals" %}
Focus: generate more trades. Loosen entry conditions (lower vol_rank_pct,
higher low_pos_pct for more bottom candidates).
{% elif direction == "reduce_underwater" %}
Focus: reduce time underwater. Accelerate exit conditions (lower 
trend_broken_days, tighter support_stuck).
{% elif direction == "lower_turnover" %}
Focus: reduce trade frequency. Lengthen hold time (higher max_hold_days,
higher take_profit_pct for bigger wins).
{% elif direction == "fine_tune" %}
Focus: incremental improvement. Small adjustments (±10%) to parameters
that are already in a good range.
{% else %}
Focus: explore new parameter combinations. You've been stuck in local optima.
Try a parameter range you haven't tested before.
{% endif %}

## Parameter Knowledge

stop_loss_pct: Fixed percentage loss trigger. Higher = wider stop = fewer false
  stops but deeper individual losses. Range: 5-30.

take_profit_pct: Fixed percentage gain trigger. Higher = hold for bigger wins
  but more reversals. Range: 10-40.

entry_stabilize_days: Days price must stay above zhixing_slow before buying.
  Higher = stricter, fewer signals. Range: 3-15.

trend_broken_days: Consecutive days below zhixing_slow before forced sell.
  Higher = more tolerant of dips. Range: 5-60.

high_pos_pct: Percentile threshold for "overbought." Higher = rarer sell signals.
  Range: 70-99.

low_pos_pct: Percentile threshold for "oversold." Lower = rarer buy signals.
  Range: 1-99. NOTE: this is inverted — 99 means "price is below 99% of its
  historical range" (extremely oversold). 50 means "below median" (mildly oversold).

vol_rank_pct: Volume percentile for "high volume." Higher = stricter.
  Range: 50-99.

support_stuck: Days near slow MA indicating "support exhaustion."
  Higher = more tolerant of ranging. Range: 3-20.

add_pullback_pct: Minimum pullback % from entry to allow adding.
  Higher = fewer adds. Range: 1-10.

## SOTA Constraint Rule

Any parameter change is REJECTED if it causes:
  - MaxDD > 20% (from current {{ max_dd }}%)
  - Bootstrap CI lower bound crosses zero (from current [{{ ci_low }}%, {{ ci_high }}%])

If both constraints hold, the change with the highest EV improvement is accepted
as the new SOTA.

## Output Format

```json
{
  "hypothesis": "If we change <param> from <old> to <new>, <expected effect>. Limit 2 sentences.",
  "reason": "Why this change addresses the current direction. Reference specific numbers from the results. Limit 2 sentences.",
  "parameter_changes": {
    "param_name": new_value,
    ...
  },
  "expected_effects": {
    "ev_change": "+0.X% to +0.Y%",
    "max_dd_change": "+Z% to -W%",
    "win_rate_change": "+A% to -B%",
    "trade_count_change": "+/- N trades"
  },
  "confidence": "high | medium | low"
}
```
"""

# ============================================================
# SOTA TRACKING
# ============================================================

class SOTATracker:
    """Tracks the best-so-far strategy configuration and its metrics."""

    def __init__(self):
        self.best_ev = float("-inf")
        self.best_config: dict[str, Any] = {}
        self.best_metrics: dict[str, Any] = {}
        self.round = 0
        self.consecutive_failures = 0
        self.current_direction = ""
        self.direction_failure_count: dict[str, int] = {}

    def update(
        self,
        ev: float,
        max_dd: float,
        ci_low: float,
        config: dict,
        metrics: dict,
        direction: str,
        previous_metrics: dict | None = None,
    ) -> tuple[bool, str]:
        """
        Returns (accepted, reason).
        Decision rule: all constraints + direction target must improve.
        """
        self.round += 1

        # Hard constraints
        if max_dd > 20:
            self.consecutive_failures += 1
            return False, f"REJECTED: MaxDD={max_dd:.1f}% exceeds 20% constraint"

        if ci_low <= 0:
            self.consecutive_failures += 1
            return False, f"REJECTED: Bootstrap CI lower bound ({ci_low:.2f}%) ≤ 0"

        # Direction-specific target check
        if previous_metrics:
            direction_target = {
                "reduce_drawdown": "max_dd",
                "improve_win_rate": "win_rate",
                "increase_signals": "trades",
                "reduce_underwater": "underwater",
                "lower_turnover": "turnover",
            }
            target_key = direction_target.get(direction)
            if target_key:
                old_val = previous_metrics.get(target_key, 0)
                new_val = metrics.get(target_key, 0)
                if target_key in ("max_dd", "underwater", "turnover"):
                    # Lower is better for these
                    if new_val >= old_val:
                        self.consecutive_failures += 1
                        self._check_target_trend(new_val, old_val, direction)
                        return False, f"REJECTED: {target_key}={new_val:.2f}% not improved from {old_val:.2f}% (direction={direction})"
                else:
                    if new_val <= old_val:
                        self.consecutive_failures += 1
                        return False, f"REJECTED: {target_key}={new_val:.2f}% not improved from {old_val:.2f}% (direction={direction})"

        previous_max_dd = previous_metrics.get("max_dd", 99) if previous_metrics else 99

        # EV-based decision
        if ev > self.best_ev:
            self.best_ev = ev
            self.best_config = config
            self.best_metrics = metrics
            self.consecutive_failures = 0
            self.direction_failure_count[direction] = 0
            return True, f"ACCEPTED: EV={ev:.2f}% > SOTA EV={self.best_ev:.2f}%"

        elif max_dd < previous_max_dd - 5:
            self.best_ev = ev
            self.best_config = config
            self.best_metrics = metrics
            self.consecutive_failures = 0
            return True, f"ACCEPTED: EV stable ({ev:.2f}%) but MaxDD improved by >5pp"

        else:
            self.consecutive_failures += 1
            self.direction_failure_count[direction] = self.direction_failure_count.get(direction, 0) + 1
            return False, f"REJECTED: EV={ev:.2f}% ≤ SOTA EV={self.best_ev:.2f}%"

    @property
    def should_switch_direction(self) -> bool:
        """Switch direction after 3 consecutive failures, OR 2 failures where target worsened."""
        if self.consecutive_failures >= 3 and self.current_direction != "explore_new_params":
            return True
        if self.consecutive_failures >= 2 and self._target_worsened:
            return True
        return False

    @property
    def should_escalate_phase(self) -> bool:
        """Escalate to next phase if current phase levers are exhausted.
        This happens when 2 consecutive failures in the same phase with
        target metric worsening each time."""
        return self.consecutive_failures >= 2 and self._target_worsened

    _target_worsened: bool = False

    def _check_target_trend(self, new_val: float, old_val: float, direction: str) -> None:
        """Track whether direction target is worsening."""
        self._target_worsened = (new_val >= old_val)  # for reduce_* directions

    @property
    def should_explore(self) -> bool:
        """Force explore after 9 total failures."""
        return self.consecutive_failures >= 9

    def get_phase(self) -> int:
        """Progressive phase: 1 = coarse (round 1-3), 2 = medium (4-6), 3 = fine (7+)"""
        if self.round <= 3:
            return 1
        elif self.round <= 6:
            return 2
        else:
            return 3


# ============================================================
# CONTEXT BUILDER
# ============================================================

def build_action_selection_context(
    current_metrics: dict[str, Any],
    sota: SOTATracker,
    past_rounds: list[dict],
) -> str:
    """Build the context for Stage 1: Action Selection."""

    lines = [
        "## Current Strategy State",
        f"- EV: {current_metrics.get('ev', 'N/A')}%",
        f"- Win Rate: {current_metrics.get('win_rate', 'N/A')}%",
        f"- Max DD: {current_metrics.get('max_dd', 'N/A')}%",
        f"- Sharpe: {current_metrics.get('sharpe', 'N/A')}",
        f"- Trades: {current_metrics.get('trades', 'N/A')}",
        f"- Underwater: {current_metrics.get('underwater', 'N/A')}%",
        f"- Annual Turnover: {current_metrics.get('turnover', 'N/A')}%",
        f"- Bootstrap CI: [{current_metrics.get('ci_low', 'N/A')}%, {current_metrics.get('ci_high', 'N/A')}%]",
        "",
        "## SOTA (Best So Far)",
        f"- EV: {sota.best_ev:.2f}%" if sota.best_ev > float("-inf") else "- No SOTA yet",
        f"- Round: {sota.round}",
        f"- Current direction: {sota.current_direction or 'none'}",
        f"- Consecutive failures: {sota.consecutive_failures}",
    ]

    if past_rounds:
        lines.append("")
        lines.append("## Past Rounds")
        for r in past_rounds[-5:]:  # last 5 rounds
            lines.append(f"- Round {r['round']}: direction={r['direction']}, "
                         f"decision={r['decision']}, EV={r.get('ev', 'N/A')}%")

    return "\n".join(lines)


def build_hypothesis_context(
    current_config: dict[str, Any],
    current_metrics: dict[str, Any],
    sota: SOTATracker,
    direction: str,
) -> dict[str, Any]:
    """Build the context for Stage 2: Hypothesis Generation."""

    phase = sota.get_phase()

    return {
        "round": sota.round + 1,
        "direction": direction,
        # Current parameter values
        "stop_loss_pct": current_config.get("stop_loss_pct", 15),
        "take_profit_pct": current_config.get("take_profit_pct", 15),
        "entry_stabilize_days": current_config.get("state_machine_params", {}).get("entry_stabilize_days", 7),
        "trend_broken_days": current_config.get("state_machine_params", {}).get("trend_broken_days", 30),
        "high_pos_pct": current_config.get("state_machine_params", {}).get("high_pos_pct", 95),
        "low_pos_pct": current_config.get("state_machine_params", {}).get("low_pos_pct", 99),
        "vol_rank_pct": current_config.get("state_machine_params", {}).get("vol_rank_pct", 99),
        "support_stuck": current_config.get("state_machine_params", {}).get("support_stuck", 5),
        "add_pullback_pct": current_config.get("state_machine_params", {}).get("add_pullback_pct", 3.0),
        "smart_stop": current_config.get("state_machine_params", {}).get("smart_stop", False),
        "entry_ladder": current_config.get("entry_ladder", []),
        "exit_ladder": current_config.get("exit_ladder", []),
        # Current metrics
        "max_dd": current_metrics.get("max_dd", 15),
        "ci_low": current_metrics.get("ci_low", 1.0),
        "ci_high": current_metrics.get("ci_high", 5.0),
        "ev": current_metrics.get("ev", 0),
        "win_rate": current_metrics.get("win_rate", 50),
        "sharpe": current_metrics.get("sharpe", 0),
        "trades": current_metrics.get("trades", 0),
        "underwater": current_metrics.get("underwater", 50),
        # SOTA
        "sota_ev": sota.best_ev if sota.best_ev > float("-inf") else current_metrics.get("ev", 0),
        "sota_max_dd": sota.best_metrics.get("max_dd", 15),
    }
