"""
状态机策略引擎 — 参数化版本
核心逻辑固定，参数控制敏感度
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional
from enum import Enum


class State(Enum):
    WAITING = "waiting"
    WATCHING = "watching"
    ENTERED_HALF = "half"
    ENTERED_FULL = "full"


class ZZH10StateMachine:
    """
    参数化状态机 — 核心逻辑：
      买点：低位带量 → 等价格在 slow 上站稳 → 半仓
           回踩 slow + 双底 → 加仓
      卖点：高位带量(派发) → 减仓
           反复回踩 slow(衰竭) → 清仓
           跌破 slow(趋势破坏) → 清仓
    """
    name = "zzh1.0"

    def __init__(self, params: dict = None):
        p = params or {}
        # 买入敏感度
        self._entry_stabilize_days = p.get("entry_stabilize_days", 3)  # 站稳天数
        self._vol_rank_pct = p.get("vol_rank_pct", 90)                 # 带量阈值
        self._low_pos_pct = p.get("low_pos_pct", 50)                   # 低位阈值
        # 卖出敏感度
        self._high_pos_pct = p.get("high_pos_pct", 80)                 # 高位阈值
        self._trend_broken_days = p.get("trend_broken_days", 3)        # 连续低于slow天数
        self._support_stuck = p.get("support_stuck", 5)                # 支撑衰竭：近10天贴slow天数
        # 加仓
        self._add_pullback_pct = p.get("add_pullback_pct", 3.0)        # 最小回调幅度%
        # 智能止损
        self._smart_stop = p.get("smart_stop", False)                  # 亏损+跌破slow提前走
        # 状态
        self.state = State.WAITING
        self._entry_idx = -1
        self._add_idx = -1
        self._vol_day_idx = -1
        self._vol_day_low = 0.0
        # 预计算
        self._prepared = False
        self._n = 0
        self._is_low_arr = None
        self._is_high_arr = None
        self._has_vol_arr = None
        self._above_slow_arr = None
        self._consec_above = None
        self._near_slow_arr = None
        self._has_db_arr = None
        self._long_wick_arr = None

    def reset(self):
        self.state = State.WAITING
        self._entry_idx = -1
        self._add_idx = -1
        self._vol_day_idx = -1
        self._vol_day_low = 0.0
        self._prepared = False

    # ================================================================
    # 预计算
    # ================================================================
    def prepare(self, df: pd.DataFrame):
        n = len(df)
        self._n = n
        self._prepared = True
        close = df["close"].values
        low = df["low"].values
        high = df["high"].values
        open_ = df["open"].values
        vol = df["volume"].values
        vol_rank = df["vol_rank_pct"].values
        vol_ma20 = df["vol_ma20"].values
        pos = df["price_position_pct"].values
        slow = df["zhixing_slow"].values

        vp = self._vol_rank_pct
        lp = self._low_pos_pct
        hp = self._high_pos_pct

        # 1. 低位/高位
        self._is_low_arr = np.zeros(n, dtype=bool)
        self._is_high_arr = np.zeros(n, dtype=bool)
        for i in range(60, n):
            if not np.isnan(pos[i]):
                self._is_low_arr[i] = pos[i] < lp
                self._is_high_arr[i] = pos[i] > hp

        # 2. 带量日
        self._has_vol_arr = np.zeros(n, dtype=bool)
        for i in range(60, n):
            if np.isnan(vol_rank[i]):
                continue
            if vol_rank[i] >= vp:
                self._has_vol_arr[i] = True
            elif not np.isnan(vol_ma20[i]) and vol_ma20[i] > 0 and vol[i] >= vol_ma20[i] * 2.5:
                self._has_vol_arr[i] = True

        # 3. 站上 slow / 连续天数
        self._above_slow_arr = np.zeros(n, dtype=bool)
        self._consec_above = np.zeros(n, dtype=int)
        cnt = 0
        for i in range(n):
            if slow[i] > 0 and close[i] > slow[i]:
                cnt += 1
            else:
                cnt = 0
            self._above_slow_arr[i] = (slow[i] > 0 and close[i] > slow[i])
            self._consec_above[i] = cnt

        # 4. 靠近 slow
        self._near_slow_arr = np.zeros(n, dtype=bool)
        for i in range(60, n):
            if slow[i] > 0:
                self._near_slow_arr[i] = abs((close[i] - slow[i]) / slow[i] * 100) < 2.0

        # 5. 双底
        self._has_db_arr = np.zeros(n, dtype=bool)
        for i in range(60, n):
            lookback = min(60, i - 30)
            if lookback < 20:
                continue
            start = i - lookback
            w_low = low[start:i + 1]
            w_close = close[start:i + 1]
            local_mins = []
            for j in range(2, len(w_low) - 2):
                if (w_low[j] <= w_low[j-1] and w_low[j] <= w_low[j-2]
                    and w_low[j] <= w_low[j+1] and w_low[j] <= w_low[j+2]):
                    local_mins.append((j, w_low[j]))
            if len(local_mins) < 2:
                continue
            for k in range(len(local_mins) - 1, 0, -1):
                j2, low2 = local_mins[k]
                j1, low1 = local_mins[k - 1]
                if j2 - j1 < 3:
                    continue
                ref = max(low1, low2)
                if ref <= 0 or abs(low1 - low2) / ref * 100 > 3.0:
                    continue
                between = w_close[j1 + 1:j2]
                if len(between) == 0:
                    continue
                if (between.max() - ref) / ref * 100 < 3.0:
                    continue
                if i - (start + j2) <= 5:
                    self._has_db_arr[i] = True
                    break

        # 6. 长上影
        self._long_wick_arr = np.zeros(n, dtype=bool)
        for i in range(60, n):
            full_range = high[i] - low[i]
            if full_range <= 0:
                continue
            body_high = max(open_[i], close[i])
            self._long_wick_arr[i] = (high[i] - body_high) / full_range >= 0.35

    # ================================================================
    # O(1) 查表
    # ================================================================
    def _arr(self, arr, idx):
        return bool(arr[idx]) if self._prepared and 0 <= idx < self._n else False

    def _is_low(self, idx):     return self._arr(self._is_low_arr, idx)
    def _is_high(self, idx):    return self._arr(self._is_high_arr, idx)
    def _has_vol(self, idx):    return self._arr(self._has_vol_arr, idx)
    def _above_slow(self, idx): return self._arr(self._above_slow_arr, idx)
    def _near_slow(self, idx):  return self._arr(self._near_slow_arr, idx)
    def _has_db(self, idx):     return self._arr(self._has_db_arr, idx)
    def _long_wick(self, idx):  return self._arr(self._long_wick_arr, idx)

    def _consec_above_slow(self, idx):
        return int(self._consec_above[idx]) if self._prepared and 0 <= idx < self._n else 0

    # ================================================================
    # 主评估
    # ================================================================
    def evaluate(self, df: pd.DataFrame, idx: int,
                 has_position: bool, has_added: bool,
                 entry_price: float = 0.0,
                 entry_date: str = "") -> tuple[Optional[str], str]:
        if idx < 60:
            return None, ""

        stab = self._entry_stabilize_days
        tbd = self._trend_broken_days

        # 跟踪带量日
        if self._has_vol(idx) and self._vol_day_idx < idx:
            self._vol_day_idx = idx
            self._vol_day_low = df["low"].iloc[idx]

        # ---- WAITING ----
        if self.state == State.WAITING:
            # 新入场：低位 + 带量 → 进入观察
            if self._is_low(idx) and self._has_vol(idx):
                self.state = State.WATCHING
                self._vol_day_idx = idx
                self._vol_day_low = df["low"].iloc[idx]
            # 重新入场：曾有过持仓 + 回踩 slow 后重新站稳 → 直接半仓
            elif (self._entry_idx >= 0 and not has_position and
                  self._consec_above_slow(idx) >= stab and
                  self._recently_near_slow(idx)):
                self.state = State.ENTERED_HALF
                self._entry_idx = idx
                return "buy", "回调站稳再入场"

        # ---- WATCHING ----
        elif self.state == State.WATCHING:
            # 陷阱：带量日后创新低
            if self._vol_day_idx >= 0 and self._vol_day_low > 0:
                if df["low"].iloc[idx] < self._vol_day_low * 0.97:
                    self.state = State.WAITING
                    self._vol_day_idx = -1
                    return None, ""
            # 更新带量日
            if self._has_vol(idx) and (self._vol_day_idx < 0 or
                                        df["low"].iloc[idx] >= self._vol_day_low * 0.97):
                self._vol_day_idx = idx
                self._vol_day_low = df["low"].iloc[idx]
            # 站稳 → 买入
            if (self._consec_above_slow(idx) >= stab and
                self._is_low(idx) and not has_position):
                self.state = State.ENTERED_HALF
                self._entry_idx = idx
                return "buy", "低位带量+站稳均线"

        # ---- ENTERED_HALF ----
        elif self.state == State.ENTERED_HALF:
            # 智能止损（仅当参数启用时）：连续3天低于slow + 浮亏>5% → 提前走
            if self._smart_stop and entry_price > 0:
                loss = (float(df["close"].iloc[idx]) - entry_price) / entry_price * 100
                below_3days = (not self._above_slow(idx) and
                               not self._above_slow(idx-1) and
                               not self._above_slow(idx-2))
                if loss < -5 and below_3days:
                    self.state = State.WAITING
                    return "sell", f"智能止损({loss:.1f}%)"
                if loss < -12:  # 硬止损兜底
                    self.state = State.WAITING
                    return "sell", f"硬止损({loss:.1f}%)"
            # 高位+带量+长上影 → 清仓
            if self._is_high(idx) and self._has_vol(idx) and self._long_wick(idx):
                self.state = State.WAITING
                return "sell", "极端BC清仓"
            # 高位+带量 → 减仓
            if self._is_high(idx) and self._has_vol(idx):
                return "reduce", "高位带量派发"
            # 支撑衰竭
            if self._support_exhausted(idx):
                self.state = State.WAITING
                return "sell", "反复回调均线衰竭"
            # 趋势破坏
            if self._trend_broken(idx, tbd):
                self.state = State.WAITING
                return "sell", "趋势破坏"
            # 加仓：回踩 slow + 双底 + 回调幅度够
            if (not has_added and
                df["close"].iloc[idx] < df["zhixing_fast"].iloc[idx] and
                self._near_slow(idx) and self._has_db(idx)):
                if self._entry_idx >= 0:
                    peak = df["close"].iloc[self._entry_idx:idx + 1].max()
                    if (peak - df["close"].iloc[idx]) / peak * 100 >= self._add_pullback_pct:
                        self.state = State.ENTERED_FULL
                        self._add_idx = idx
                        return "add", "回踩均线双底加仓"

        # ---- ENTERED_FULL ----
        elif self.state == State.ENTERED_FULL:
            # 智能止损
            if self._smart_stop and entry_price > 0:
                loss = (float(df["close"].iloc[idx]) - entry_price) / entry_price * 100
                below_3days = (not self._above_slow(idx) and
                               not self._above_slow(idx-1) and
                               not self._above_slow(idx-2))
                if loss < -5 and below_3days:
                    self.state = State.WAITING
                    return "sell", f"智能止损({loss:.1f}%)"
                if loss < -12:
                    self.state = State.WAITING
                    return "sell", f"硬止损({loss:.1f}%)"
            if self._is_high(idx) and self._has_vol(idx) and self._long_wick(idx):
                self.state = State.WAITING
                return "sell", "极端BC清仓"
            if self._is_high(idx) and self._has_vol(idx):
                self.state = State.ENTERED_HALF
                self._add_idx = -1
                return "reduce", "高位带量派发"
            if self._support_exhausted(idx):
                self.state = State.WAITING
                return "sell", "支撑衰竭清仓"
            if self._trend_broken(idx, tbd):
                self.state = State.WAITING
                return "sell", "趋势破坏清仓"

        return None, ""

    def _support_exhausted(self, idx):
        if not self._prepared or idx < 60:
            return False
        stuck = sum(1 for j in range(max(idx - 10, 0), idx + 1) if self._near_slow(j))
        return stuck >= self._support_stuck

    def _trend_broken(self, idx, days):
        if not self._prepared or idx < 60 + days - 1:
            return False
        for d in range(days):
            if self._above_slow(idx - d):
                return False
        for j in range(max(0, idx - 15), idx - days + 1):
            if self._above_slow(j):
                return True
        return False

    def _recently_near_slow(self, idx):
        """最近10天内是否曾靠近slow（确认发生过回调）"""
        if not self._prepared:
            return False
        for j in range(max(0, idx - 10), idx + 1):
            if self._near_slow(j):
                return True
        return False
