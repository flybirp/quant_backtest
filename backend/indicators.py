"""技术指标计算模块"""

import pandas as pd
import numpy as np
from typing import Optional


def sma(series: pd.Series, period: int) -> pd.Series:
    """简单移动均线"""
    return series.rolling(window=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    """指数移动均线"""
    return series.ewm(span=period, adjust=False).mean()


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """MACD"""
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    dif = ema_fast - ema_slow
    dea = ema(dif, signal)
    bar = 2 * (dif - dea)
    return pd.DataFrame({"dif": dif, "dea": dea, "bar": bar})


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """RSI相对强弱指标"""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def kdj(high: pd.Series, low: pd.Series, close: pd.Series,
        n: int = 9, m1: int = 3, m2: int = 3) -> pd.DataFrame:
    """KDJ指标"""
    low_n = low.rolling(window=n).min()
    high_n = high.rolling(window=n).max()
    rsv = (close - low_n) / (high_n - low_n) * 100
    k = rsv.ewm(com=m1 - 1, adjust=False).mean()
    d = k.ewm(com=m2 - 1, adjust=False).mean()
    j = 3 * k - 2 * d
    return pd.DataFrame({"k": k, "d": d, "j": j})


def bollinger(close: pd.Series, period: int = 20, nbdev: float = 2.0) -> pd.DataFrame:
    """布林带"""
    mid = sma(close, period)
    std = close.rolling(window=period).std()
    upper = mid + nbdev * std
    lower = mid - nbdev * std
    return pd.DataFrame({"upper": upper, "mid": mid, "lower": lower})


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """ATR 平均真实波幅"""
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def volume_ma(volume: pd.Series, period: int = 20) -> pd.Series:
    """成交量均线"""
    return volume.rolling(window=period).mean()


# ============================================================
# 知行量化指标
# ============================================================

def zhixing_fast(close: pd.Series) -> pd.Series:
    """双重平滑10日EMA: EMA(EMA(close, 10), 10)"""
    ema1 = ema(close, 10)
    return ema(ema1, 10)


def zhixing_slow(close: pd.Series) -> pd.Series:
    """四条MA均值: (MA14 + MA28 + MA57 + MA114) / 4"""
    ma14 = sma(close, 14)
    ma28 = sma(close, 28)
    ma57 = sma(close, 57)
    ma114 = sma(close, 114)
    return (ma14 + ma28 + ma57 + ma114) / 4


def zz_short(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """3日价格位置百分位: 100 * (close - MIN(low,3)) / (MAX(high,3) - MIN(low,3))"""
    low_3 = low.rolling(window=3).min()
    high_3 = high.rolling(window=3).max()
    return 100 * (close - low_3) / (high_3 - low_3 + 1e-10)


def zz_long(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """21日价格位置百分位: 100 * (close - MIN(low,21)) / (MAX(high,21) - MIN(low,21))"""
    low_21 = low.rolling(window=21).min()
    high_21 = high.rolling(window=21).max()
    return 100 * (close - low_21) / (high_21 - low_21 + 1e-10)


# ============================================================
# 量能异动监控指标
# ============================================================

def volume_rank_pct(volume: pd.Series, period: int = 120) -> pd.Series:
    """成交量在近N日内的排名百分位（0~100，越高越放量）

    向量化实现：用rolling rank替代逐行apply
    """
    n = len(volume)
    result = np.full(n, np.nan)

    vol_arr = volume.values
    for i in range(period - 1, n):
        window = vol_arr[i - period + 1:i + 1]
        # 当前值在窗口中的排名百分位
        rank = np.sum(window <= vol_arr[i]) / period * 100
        result[i] = rank

    return pd.Series(result, index=volume.index)


def price_position_pct(close: pd.Series, period: int = 120) -> pd.Series:
    """股价在近N日内的百分位（0~100，0=最低，100=最高）

    向量化实现：用rolling rank替代逐行apply
    """
    n = len(close)
    result = np.full(n, np.nan)

    close_arr = close.values
    for i in range(period - 1, n):
        window = close_arr[i - period + 1:i + 1]
        rank = np.sum(window <= close_arr[i]) / period * 100
        result[i] = rank

    return pd.Series(result, index=close.index)


def dist_to_line(price: pd.Series, line: pd.Series) -> pd.Series:
    """price 距离 line 的百分比: (price - line) / line * 100"""
    return (price - line) / line.replace(0, np.nan) * 100


def is_double_volume_signal(volume: pd.Series) -> pd.Series:
    """倍量柱信号: 当日量 >= 前一日2倍，满足为1"""
    prev_vol = volume.shift(1)
    return ((volume >= prev_vol * 2) & (prev_vol > 0)).astype(int)


def is_volume_top5(volume: pd.Series, period: int = 120) -> pd.Series:
    """历史极量信号: 当日量在近N日内排前5%（top5%）"""
    rank = volume_rank_pct(volume, period)
    return (rank >= 95).astype(int)


# ============================================================
# 口袋支点指标 (Pocket Pivot)
# ============================================================

def pocket_pivot_volume(volume: pd.Series, close: pd.Series, lookback: int = 10) -> pd.Series:
    """口袋支点量能条件: 当日成交量 > 前N日中下跌日最大成交量

    核心逻辑: 一天的买入力量(成交量)超过了近期任何一天的卖出力量(下跌日成交量)，
    这是需求压倒供应的证明，即威科夫SOS的精确量化版本。

    向量化实现，避免逐行Python循环。
    """
    n = len(volume)
    result = np.zeros(n, dtype=int)

    vol_arr = volume.values.astype(float)
    close_arr = close.values.astype(float)

    # 预计算下跌日mask
    is_down = np.zeros(n, dtype=bool)
    is_down[1:] = close_arr[1:] < close_arr[:-1]

    for i in range(lookback, n):
        # 窗口 [i-lookback, i)
        start = i - lookback
        down_mask = is_down[start:i]
        down_vols = vol_arr[start:i][down_mask]

        if len(down_vols) > 0:
            max_down_vol = down_vols.max()
            result[i] = int(vol_arr[i] > max_down_vol)
        else:
            # 没有下跌日，只要当日有量就算通过
            result[i] = int(vol_arr[i] > 0)

    return pd.Series(result, index=volume.index)


# ============================================================


def ma_cross(close: pd.Series, fast: int, slow: int) -> pd.Series:
    """均线交叉信号。1=金叉, -1=死叉, 0=无"""
    ma_fast = sma(close, fast)
    ma_slow = sma(close, slow)
    cross = pd.Series(0, index=close.index)
    cross[(ma_fast > ma_slow) & (ma_fast.shift(1) <= ma_slow.shift(1))] = 1
    cross[(ma_fast < ma_slow) & (ma_fast.shift(1) >= ma_slow.shift(1))] = -1
    return cross


def highest(series: pd.Series, period: int) -> pd.Series:
    """N周期最高值"""
    return series.rolling(window=period).max()


def lowest(series: pd.Series, period: int) -> pd.Series:
    """N周期最低值"""
    return series.rolling(window=period).min()


def bbi(close: pd.Series, periods: tuple = (3, 6, 12, 24)) -> pd.Series:
    """BBI 多空指标 = (MA3 + MA6 + MA12 + MA24) / 4"""
    ma_sum = sum(sma(close, p) for p in periods)
    return ma_sum / len(periods)


def volume_explosion_flag(volume: pd.Series, close: pd.Series,
                          vol_rank_pct: pd.Series,
                          vol_ma20: pd.Series,
                          vol_rank_threshold: float = 95,
                          vol_ratio_threshold: float = 3.0) -> pd.Series:
    """量能爆发标记: vol_rank_pct >= threshold OR volume >= vol_ma20 * ratio"""
    flag_rank = vol_rank_pct >= vol_rank_threshold
    flag_ratio = volume >= vol_ma20 * vol_ratio_threshold
    return (flag_rank | flag_ratio).astype(int)


def double_bottom_level(low: pd.Series, close: pd.Series,
                        lookback: int = 20,
                        min_separation: int = 3,
                        tolerance_pct: float = 3.0) -> pd.Series:
    """双底形态检测，返回双底支撑位（两个底部低点的均值）

    如果当前日存在双底形态，返回支撑位价格；否则返回NaN

    双底定义（回望lookback日内）：
    1. 找到两个局部低点
    2. 两个低点价格差在tolerance_pct%以内
    3. 两个低点间隔至少min_separation天
    4. 两低点之间有反弹（中间某日close高于两个低点）
    5. 第二个低点在近期（距当前不超过lookback/2天）
    """
    n = len(low)
    result = np.full(n, np.nan)

    low_arr = low.values.astype(float)
    close_arr = close.values.astype(float)

    for i in range(lookback, n):
        window = low_arr[i - lookback:i + 1]
        window_close = close_arr[i - lookback:i + 1]

        # 找局部低点：比前后各1天都低的点
        local_mins = []
        for j in range(1, len(window) - 1):
            if window[j] <= window[j - 1] and window[j] <= window[j + 1]:
                local_mins.append((j, window[j]))

        if len(local_mins) < 2:
            continue

        # 从最近往远找，找最近的一对双底
        found = False
        for k in range(len(local_mins) - 1, 0, -1):
            for m in range(k - 1, -1, -1):
                idx2, low2 = local_mins[k]
                idx1, low1 = local_mins[m]

                # 间隔至少min_separation天
                if idx2 - idx1 < min_separation:
                    continue

                # 两个低点价格差在tolerance内
                ref = max(low1, low2)
                if ref <= 0:
                    continue
                price_diff = abs(low1 - low2) / ref * 100
                if price_diff > tolerance_pct:
                    continue

                # 两个低点之间有反弹
                between_close = window_close[idx1 + 1:idx2]
                between_low = window[idx1 + 1:idx2]
                if len(between_close) == 0:
                    continue
                # 反弹：中间至少有一天close高于两个低点中的较高者
                neck = max(low1, low2)
                if not (between_close > neck).any():
                    continue

                # 第二个低点不能太远（在lookback/2内）
                actual_idx2 = i - lookback + idx2
                if i - actual_idx2 > lookback // 2:
                    continue

                # 找到双底，记录支撑位
                support = (low1 + low2) / 2
                result[i] = support
                found = True
                break
            if found:
                break

    return pd.Series(result, index=low.index)


def compute_all_indicators(df: pd.DataFrame, required_cols: set[str] | None = None) -> pd.DataFrame:
    """
    给DataFrame附加指标，返回增强后的DataFrame。

    Args:
        df: 原始OHLCV数据
        required_cols: 需要的列名集合。None=全部计算，set=只算需要的。
    """
    d = df.copy()
    close = d["close"]
    high = d["high"]
    low = d["low"]
    volume = d["volume"]

    need = required_cols  # None = 全算

    def _need(col: str) -> bool:
        return need is None or col in need

    # 均线
    for p in [5, 10, 20, 30, 60, 120, 250]:
        if _need(f"ma{p}"):
            d[f"ma{p}"] = sma(close, p)

    # MACD
    if _need("macd_dif") or _need("macd_dea") or _need("macd_bar"):
        macd_df = macd(close)
        if _need("macd_dif"):
            d["macd_dif"] = macd_df["dif"]
        if _need("macd_dea"):
            d["macd_dea"] = macd_df["dea"]
        if _need("macd_bar"):
            d["macd_bar"] = macd_df["bar"]

    # RSI
    if _need("rsi14"):
        d["rsi14"] = rsi(close, 14)
    if _need("rsi6"):
        d["rsi6"] = rsi(close, 6)

    # KDJ
    if _need("kdj_k") or _need("kdj_d") or _need("kdj_j"):
        kdj_df = kdj(high, low, close)
        if _need("kdj_k"):
            d["kdj_k"] = kdj_df["k"]
        if _need("kdj_d"):
            d["kdj_d"] = kdj_df["d"]
        if _need("kdj_j"):
            d["kdj_j"] = kdj_df["j"]

    # 布林带
    if _need("bb_upper") or _need("bb_mid") or _need("bb_lower"):
        bb_df = bollinger(close)
        if _need("bb_upper"):
            d["bb_upper"] = bb_df["upper"]
        if _need("bb_mid"):
            d["bb_mid"] = bb_df["mid"]
        if _need("bb_lower"):
            d["bb_lower"] = bb_df["lower"]

    # ATR
    if _need("atr14"):
        d["atr14"] = atr(high, low, close, 14)

    # 成交量均线
    if _need("vol_ma20"):
        d["vol_ma20"] = volume_ma(volume, 20)

    # 涨跌幅
    if _need("pct_change"):
        d["pct_change"] = close.pct_change() * 100

    # 知行量化指标
    if _need("zhixing_fast"):
        d["zhixing_fast"] = zhixing_fast(close)
    if _need("zhixing_slow"):
        d["zhixing_slow"] = zhixing_slow(close)
    if _need("zz_short") or _need("zz_long") or _need("zz_spread"):
        d["zz_short"] = zz_short(high, low, close)
        d["zz_long"] = zz_long(high, low, close)
        d["zz_spread"] = d["zz_long"] - d["zz_short"]

    # 量能异动指标
    if _need("vol_rank_pct"):
        d["vol_rank_pct"] = volume_rank_pct(volume, 120)
    if _need("price_position_pct"):
        d["price_position_pct"] = price_position_pct(close, 120)
    if _need("dist_to_ma60") and "ma60" in d:
        d["dist_to_ma60"] = dist_to_line(close, d["ma60"])
    if _need("dist_to_zhixing_fast") and "zhixing_fast" in d:
        d["dist_to_zhixing_fast"] = dist_to_line(close, d["zhixing_fast"])
    if _need("dist_to_zhixing_slow") and "zhixing_slow" in d:
        d["dist_to_zhixing_slow"] = dist_to_line(close, d["zhixing_slow"])
    if _need("is_double_vol"):
        d["is_double_vol"] = is_double_volume_signal(volume)
    if _need("is_vol_top5"):
        d["is_vol_top5"] = is_volume_top5(volume, 120)
    if _need("pocket_pivot_vol"):
        d["pocket_pivot_vol"] = pocket_pivot_volume(volume, close, 10)

    # BBI 多空指标
    if _need("bbi"):
        d["bbi"] = bbi(close)

    # 双底形态支撑位
    if _need("double_bottom_support"):
        d["double_bottom_support"] = double_bottom_level(low, close, lookback=20, min_separation=3, tolerance_pct=3.0)

    # 量能爆发标记
    if _need("vol_explosion"):
        d["vol_explosion"] = volume_explosion_flag(volume, close,
                                                    d.get("vol_rank_pct", volume_rank_pct(volume, 120)),
                                                    d.get("vol_ma20", volume_ma(volume, 20)),
                                                    vol_rank_threshold=95, vol_ratio_threshold=3.0)

    # 相对大盘强度
    for lb in [5, 10, 20, 60]:
        col = f"relative_strength_{lb}"
        if _need(col):
            d[col] = _compute_relative_strength(d, lookback=lb)

    # 大盘恐慌
    if _need("market_crash_30d"):
        d["market_crash_30d"] = _compute_market_crash_flag(d, lookback=30)
    if _need("market_crash_fast_10d"):
        d["market_crash_fast_10d"] = _compute_fast_crash_flag(d)

    return d


def _compute_fast_crash_flag(df: pd.DataFrame) -> "pd.Series":
    """10天内hs300是否快速杀跌超过5%。向量化+日期对齐。"""
    try:
        idx_df = _load_index_cached("hs300")
        dd10 = (idx_df["close"] / idx_df["close"].rolling(10).max() - 1) * 100
        crash_any = (dd10 < -5.0).rolling(10, min_periods=1).max().fillna(0).astype(bool)
        # 按日期对齐（不同股票上市日期不同，不能用位置）
        aligned = crash_any.reindex(df["date"], method="ffill").fillna(False)
        aligned.index = df.index
        return aligned
    except Exception:
        return pd.Series([False] * len(df), index=df.index)


# Module-level cache for hs300 index data
_idx_cache: dict = {}


def _load_index_cached(name: str = "hs300") -> "pd.DataFrame":
    """Load index once, cache across all calls."""
    if name not in _idx_cache:
        from pathlib import Path
        import pandas as pd
        idx_path = Path(f"/Users/flybirp/Documents/mainland_index_data_2014/{name}.csv")
        _idx_cache[name] = pd.read_csv(idx_path, parse_dates=["date"]).set_index("date").sort_index()
    return _idx_cache[name]


def _compute_relative_strength(df: pd.DataFrame, lookback: int = 60) -> pd.Series:
    """个股相对于hs300的超额收益。正数=跑赢大盘。按位置对齐。"""
    try:
        idx_df = _load_index_cached("hs300")

        # Stock rolling return (按行位置)
        stock_ret = (df["close"] / df["close"].shift(lookback) - 1) * 100

        # Index rolling return: 提前算好，按行数对齐（不用reindex）
        idx_close = idx_df["close"]
        idx_ret_full = (idx_close / idx_close.shift(lookback) - 1) * 100
        # 对齐：取前 len(df) 行（所有股票交易日历相同）
        idx_aligned = idx_ret_full.iloc[:len(df)].values

        rs = pd.Series(stock_ret.values - idx_aligned, index=df.index).fillna(0.0)
        return rs
    except Exception:
        return pd.Series([0.0] * len(df), index=df.index)


def _compute_market_crash_flag(df: pd.DataFrame, lookback: int = 30) -> pd.Series:
    """近N天内hs300最大回撤是否超10%。按位置对齐。"""
    try:
        idx_df = _load_index_cached("hs300")

        # 提前算好指数回撤
        peak252 = idx_df["close"].rolling(252).max()
        dd252 = (idx_df["close"] - peak252) / peak252 * 100

        # 按行数对齐
        aligned = dd252.iloc[:len(df)].values
        result = pd.Series(aligned < -10.0, index=df.index)
        return result
    except Exception:
        return pd.Series([False] * len(df), index=df.index)
        result.index = df.index  # align to caller's index
        return result
    except Exception:
        return pd.Series([False] * len(df), index=df.index)
