"""数据加载模块 — 日K和周K统一接口，支持Parquet缓存加速"""

import pandas as pd
from pathlib import Path
from typing import Optional, Literal
import time

DATA_DIR = Path("/Users/flybirp/Documents/mainland_data_2014")

# Parquet缓存目录（放在项目目录下，避免数据目录权限问题）
CACHE_DIR = Path(__file__).parent.parent / "_indicator_cache"
CACHE_DIR.mkdir(exist_ok=True)


def load_stock(code: str, ktype: Literal["daily", "weekly"] = "daily") -> pd.DataFrame:
    """加载单只股票数据，支持日K/周K"""
    fpath = DATA_DIR / f"{code}.csv"
    if not fpath.exists():
        raise FileNotFoundError(f"股票 {code} 数据文件不存在: {fpath}")

    df = pd.read_csv(fpath, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    if ktype == "weekly":
        df = _to_weekly(df)
    return df


def _to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """日K降采样为周K"""
    df = df.set_index("date")
    weekly = df.resample("W").agg({
        "open": "first",
        "close": "last",
        "high": "max",
        "low": "min",
        "volume": "sum",
    }).dropna()
    weekly = weekly.reset_index()
    return weekly


def list_all_codes() -> list[str]:
    """列出所有股票代码"""
    codes = []
    for f in DATA_DIR.glob("*.csv"):
        codes.append(f.stem)
    return sorted(codes)


def load_all_stocks(ktype: Literal["daily", "weekly"] = "daily",
                    codes: Optional[list[str]] = None,
                    start_date: Optional[str] = None,
                    end_date: Optional[str] = None) -> dict[str, pd.DataFrame]:
    """批量加载多只股票数据"""
    if codes is None:
        codes = list_all_codes()

    result = {}
    for code in codes:
        try:
            df = load_stock(code, ktype)
            if start_date:
                df = df[df["date"] >= start_date]
            if end_date:
                df = df[df["date"] <= end_date]
            if len(df) > 0:
                result[code] = df
        except FileNotFoundError:
            continue
    return result


# ============================================================
# 指标缓存机制
# ============================================================

def _cache_path(code: str, ktype: str) -> Path:
    """获取缓存文件路径"""
    return CACHE_DIR / f"{code}_{ktype}.parquet"


def _csv_mtime(code: str) -> float:
    """获取CSV文件修改时间"""
    fpath = DATA_DIR / f"{code}.csv"
    return fpath.stat().st_mtime if fpath.exists() else 0


def load_stock_with_indicators(code: str, ktype: str = "daily",
                               force_recompute: bool = False,
                               required_cols: set[str] | None = None) -> pd.DataFrame:
    """加载股票并计算指标，优先使用Parquet缓存

    如果缓存存在且比CSV新，直接读缓存；否则重新计算并写入缓存。

    Args:
        required_cols: 需要的预计算列。None=全部计算。
            注意：缓存始终存全量列，required_cols 仅加速首次计算。
    """
    from backend.indicators import compute_all_indicators

    cache_fpath = _cache_path(code, ktype)

    # 缓存命中检查
    if not force_recompute and cache_fpath.exists():
        csv_time = _csv_mtime(code)
        cache_time = cache_fpath.stat().st_mtime
        if cache_time > csv_time:
            try:
                df = pd.read_parquet(cache_fpath)
                # 检查缓存是否有足够列（防止版本升级后缺列）
                check_cols = {"zhixing_fast", "zhixing_slow", "vol_rank_pct",
                              "price_position_pct", "pocket_pivot_vol"}
                if check_cols.issubset(set(df.columns)):
                    return df
            except Exception:
                pass  # 缓存损坏，重新计算

    # 未命中：加载原始数据 + 计算指标（始终全量，保证缓存完整）
    df = load_stock(code, ktype)
    if len(df) < 60:
        return df

    df = compute_all_indicators(df, required_cols=None)  # 始终全量，缓存完整

    # 写入缓存
    try:
        df.to_parquet(cache_fpath, index=False)
    except Exception as e:
        print(f"  [缓存写入失败] {code}: {e}")

    return df


def preload_indicator_cache(codes: list[str], ktype: str = "daily",
                            force_recompute: bool = False) -> None:
    """预加载/预热指标缓存

    对给定股票代码列表，确保所有指标缓存文件存在。
    """
    t0 = time.time()
    cached = 0
    computed = 0

    for i, code in enumerate(codes):
        cache_fpath = _cache_path(code, ktype)

        # 缓存是否有效
        if not force_recompute and cache_fpath.exists():
            csv_time = _csv_mtime(code)
            cache_time = cache_fpath.stat().st_mtime
            if cache_time > csv_time:
                cached += 1
                continue

        # 需要计算
        try:
            load_stock_with_indicators(code, ktype, force_recompute)
            computed += 1
        except Exception:
            continue

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(codes)}] {elapsed:.1f}s "
                  f"(缓存命中={cached}, 新计算={computed})")

    elapsed = time.time() - t0
    print(f"  指标缓存就绪: {cached}命中 + {computed}新计算 = {cached+computed}股, {elapsed:.1f}s")
