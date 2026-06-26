"""
Monte Carlo validation.

Resample trades via bootstrap, simulate equity curves, and estimate
risk-of-ruin to stress-test a strategy's robustness.
"""

from __future__ import annotations

import logging
import math
import random
from typing import Any, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def bootstrap_trades(
    trades: list,
    n_bootstrap: int = 1000,
    seed: Optional[int] = None,
) -> List[List[dict]]:
    """Generate bootstrap resamples of a trade list.

    Each resample is drawn **with replacement** from the original trades
    and has the same length.

    Args:
        trades: List of trade dicts (must have a ``profit_pct`` key).
        n_bootstrap: Number of resample sets to generate (default 1000).
        seed: Optional random seed for reproducibility.

    Returns:
        List of ``n_bootstrap`` resampled trade lists.

        Returns a single list containing the original trades if the input
        has fewer than 3 trades (insufficient data).
    """
    if len(trades) < 3:
        logger.warning("bootstrap_trades: fewer than 3 trades, returning copy of input.")
        return [[dict(t) for t in trades]]

    rng = random.Random(seed)
    n = len(trades)
    resamples: List[List[dict]] = []

    for i in range(n_bootstrap):
        sample = [trades[rng.randint(0, n - 1)] for _ in range(n)]
        resamples.append(sample)

    logger.info("Generated %d bootstrap resamples from %d trades.", n_bootstrap, n)
    return resamples


# ---------------------------------------------------------------------------
# Equity curve simulation
# ---------------------------------------------------------------------------

def equity_curve_simulation(
    trades: list,
    n_simulations: int = 1000,
    horizon_trades: Optional[int] = None,
    initial_capital: float = 100000.0,
    seed: Optional[int] = None,
) -> List[List[float]]:
    """Monte-Carlo equity-curve simulation by randomly sequencing trade returns.

    Each simulation shuffles the trade returns and computes a cumulative
    equity curve.  This captures **sequence-of-returns risk** — even a
    positive-EV strategy can have bad runs if losses cluster.

    Args:
        trades: List of trade dicts (must have ``profit_pct``).
        n_simulations: Number of simulations to run (default 1000).
        horizon_trades: Number of trades to simulate per run. Defaults to
            the length of *trades*.
        initial_capital: Starting capital for each simulation.
        seed: Optional random seed.

    Returns:
        List of ``n_simulations`` equity curves, where each curve is a list
        of floats (capital after each trade, including initial capital as
        first element).

        Returns a single curve if fewer than 2 trades available.
    """
    if len(trades) < 2:
        logger.warning("equity_curve_simulation: fewer than 2 trades, returning trivial curve.")
        return [[initial_capital]]

    returns = [t.get("profit_pct", 0.0) for t in trades]
    horizon = horizon_trades if horizon_trades else len(returns)
    if horizon < 1:
        horizon = len(returns)

    rng = random.Random(seed)
    curves: List[List[float]] = []

    for sim_idx in range(n_simulations):
        # Shuffle returns to simulate different sequences
        shuffled = returns[:]
        rng.shuffle(shuffled)

        curve = [initial_capital]
        capital = initial_capital
        for i in range(horizon):
            ret = shuffled[i % len(shuffled)]
            capital *= (1.0 + ret / 100.0)
            curve.append(round(capital, 2))

        curves.append(curve)

    logger.info("Generated %d equity curve simulations (horizon=%d trades).",
                n_simulations, horizon)
    return curves


# ---------------------------------------------------------------------------
# Ruin probability
# ---------------------------------------------------------------------------

def ruin_probability(
    trades: list,
    capital: float,
    ruin_threshold_pct: float,
    n_simulations: int = 5000,
    seed: Optional[int] = None,
) -> float:
    """Estimate the probability of ruin via Monte Carlo simulation.

    **Ruin** is defined as the equity falling below
    ``capital * (1 - ruin_threshold_pct / 100)`` at any point during the
    simulation.

    Args:
        trades: List of trade dicts (must have ``profit_pct``).
        capital: Starting capital.
        ruin_threshold_pct: Drawdown percentage that defines ruin, e.g.
            ``50`` means ruin occurs when equity drops 50% below initial
            capital.
        n_simulations: Number of simulation runs (default 5000).
        seed: Optional random seed.

    Returns:
        Float in [0, 1] — proportion of simulations that hit ruin.

        Returns ``1.0`` if fewer than 2 trades available (insufficient data
        to simulate).
    """
    if len(trades) < 2:
        logger.warning("ruin_probability: fewer than 2 trades, returning 1.0 (unknown risk).")
        return 1.0

    ruin_level = capital * (1.0 - ruin_threshold_pct / 100.0)
    returns = [t.get("profit_pct", 0.0) for t in trades]

    rng = random.Random(seed)
    ruinations = 0
    horizon = len(returns)  # use full trade history as horizon

    for sim_idx in range(n_simulations):
        shuffled = returns[:]
        rng.shuffle(shuffled)

        equity = capital
        ruined = False
        for i in range(horizon):
            ret = shuffled[i % len(shuffled)]
            equity *= (1.0 + ret / 100.0)
            if equity <= ruin_level:
                ruined = True
                break

        if ruined:
            ruinations += 1

    prob = ruinations / n_simulations
    logger.info("Ruin probability: %.4f (%d/%d sims hit %.0f%% drawdown)",
                prob, ruinations, n_simulations, ruin_threshold_pct)
    return round(prob, 6)
