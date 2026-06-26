"""
Market regime detection for A-share strategies.

Detects bull/bear/sideways regimes from index data (hs300 by default).
Strategies can use `indicator: market_bull` or `indicator: market_consolidation`
as buy/sell conditions to adapt to market state.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path

INDEX_DIR = Path("/Users/flybirp/Documents/mainland_index_data_2014")
_index_cache: dict[str, pd.DataFrame] = {}


def load_index(index_name: str = "hs300") -> pd.DataFrame:
    """Load index CSV (cached at module level)."""
    if index_name not in _index_cache:
        fpath = INDEX_DIR / f"{index_name}.csv"
        if not fpath.exists():
            raise FileNotFoundError(f"Index data not found: {fpath}")
        df = pd.read_csv(fpath, parse_dates=["date"])
        df = df.sort_values("date").reset_index(drop=True)
        _index_cache[index_name] = df
    return _index_cache[index_name]


def compute_regime(
    index_name: str = "hs300",
    ma_short: int = 60,
    ma_long: int = 200,
    drawdown_bear_threshold: float = 15.0,
) -> pd.DataFrame:
    """
    Compute market regime for each trading day.

    Regime types:
      - "bull":   close > MA_long  AND  close > MA_short AND drawdown from 252d peak < 10%
      - "bear":   close < MA_long  OR drawdown > threshold
      - "consolidation": neither bull nor bear

    Returns DataFrame with columns:
      date, close, regime, drawdown_252d, ma_short, ma_long
    """
    df = load_index(index_name)
    df["ma_short"] = df["close"].rolling(ma_short).mean()
    df["ma_long"] = df["close"].rolling(ma_long).mean()

    # 252-day rolling peak drawdown
    df["peak_252"] = df["close"].rolling(252).max()
    df["drawdown_252d"] = (df["close"] - df["peak_252"]) / df["peak_252"] * 100

    regimes = []
    for _, row in df.iterrows():
        if pd.isna(row["ma_long"]):
            regimes.append("unknown")
        elif row["drawdown_252d"] < -drawdown_bear_threshold:
            regimes.append("bear")
        elif row["close"] > row["ma_long"] and row["close"] > row.get("ma_short", 0):
            regimes.append("bull")
        elif row["close"] < row["ma_long"]:
            regimes.append("bear")
        else:
            regimes.append("consolidation")

    df["regime"] = regimes
    return df[["date", "close", "regime", "drawdown_252d", "ma_short", "ma_long"]]


def get_regime_on(date_str: str, index_name: str = "hs300") -> str:
    """Get regime for a specific date. Returns 'bull', 'bear', 'consolidation', or 'unknown'."""
    df = compute_regime(index_name)
    date = pd.Timestamp(date_str)
    match = df[df["date"] == date]
    if match.empty:
        # Find nearest date
        match = df[df["date"] <= date].tail(1)
    return match["regime"].iloc[0] if not match.empty else "unknown"


def regime_summary(index_name: str = "hs300") -> str:
    """Print regime distribution summary."""
    df = compute_regime(index_name)
    counts = df["regime"].value_counts()
    total = len(df[df["regime"] != "unknown"])

    lines = [
        f"市场状态分布 ({index_name})",
        "-" * 40,
    ]
    for regime in ["bull", "bear", "consolidation"]:
        cnt = counts.get(regime, 0)
        pct = cnt / total * 100 if total > 0 else 0
        lines.append(f"  {regime:<14}: {cnt:>5}天 ({pct:>5.1f}%)")

    # Current regime
    latest = df[df["regime"] != "unknown"].iloc[-1]
    lines.append(f"\n  当前状态: {latest['regime']}  ({latest['date'].strftime('%Y-%m-%d')})")
    lines.append(f"  当前位置: close={latest['close']:.0f}, "
                 f"MA200={latest['ma_long']:.0f}, 回撤={latest['drawdown_252d']:.1f}%")

    return "\n".join(lines)
