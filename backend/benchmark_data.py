"""基准指数数据加载 — CSI 300, 创业板, etc."""

from __future__ import annotations
import pandas as pd
from pathlib import Path
from typing import Optional, Literal

# 两份数据：旧版(全历史) / 2014起(更新的)
INDEX_DATA_PATHS = [
    Path("/Users/flybirp/Documents/mainland_index_data_2014"),
    Path("/Users/flybirp/Documents/mainland_index_data"),
]

# 基准指数映射
BENCHMARK_MAP = {
    "hs300": "沪深300",
    "csi300": "沪深300",
    "cyb": "创业板指",
    "gem": "创业板指",
    "kcb": "科创板指",
    "star": "科创板指",
    "sh": "上证指数",
    "sse": "上证指数",
    "sz": "深证成指",
    "szse": "深证成指",
    "zz1000": "中证1000",
    "csi1000": "中证1000",
    "zz500": "中证500",
}

# 文件到简称
FILE_MAP = {
    "hs300.csv": "沪深300",
    "cyb.csv": "创业板指",
    "kcb.csv": "科创板指",
    "sh.csv": "上证指数",
    "sz.csv": "深证成指",
    "zz1000.csv": "中证1000",
}


def list_benchmarks() -> dict[str, str]:
    """列出所有可用的基准指数

    Returns:
        {简称: 中文名}
    """
    available = {}
    for base in INDEX_DATA_PATHS:
        if not base.exists():
            continue
        for fname, cname in FILE_MAP.items():
            if (base / fname).exists():
                available[fname.replace(".csv", "")] = cname
    return available


def load_benchmark(name: str,
                   start_date: Optional[str] = None,
                   end_date: Optional[str] = None) -> pd.DataFrame:
    """加载基准指数数据

    Args:
        name: 指数简称，如 'hs300', 'cyb', 'sh', 'sz', 'kcb', 'zz1000'
        start_date: 起始日期 'YYYY-MM-DD'
        end_date: 截止日期 'YYYY-MM-DD'

    Returns:
        DataFrame with columns: date, close, open, high, low, volume, pct_change
    """
    fname = f"{name}.csv"
    found = None
    for base in INDEX_DATA_PATHS:
        candidate = base / fname
        if candidate.exists():
            found = candidate
            break

    if not found:
        raise FileNotFoundError(
            f"基准指数 '{name}' 不存在。可用: {list(list_benchmarks().keys())}"
        )

    df = pd.read_csv(found)
    # 统一日期列：优先用 date，否则用 trade_date；避免重命名导致的重复列
    if "date" not in df.columns and "trade_date" in df.columns:
        df["date"] = pd.to_datetime(df["trade_date"])
    elif "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # 标准化列名
    col_map = {"vol": "volume", "amount": "amount"}
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    # 确保 pct_change 列存在（优先用原列，否则从 pct_chg/close 推算）
    if "pct_change" not in df.columns:
        if "pct_chg" in df.columns:
            df["pct_change"] = df["pct_chg"] / 100.0
        else:
            df["pct_change"] = df["close"].pct_change()

    if start_date:
        df = df[df["date"] >= start_date]
    if end_date:
        df = df[df["date"] <= end_date]

    return df


def load_benchmark_returns(name: str,
                           start_date: Optional[str] = None,
                           end_date: Optional[str] = None) -> pd.Series:
    """加载基准指数日收益率序列

    Returns:
        pd.Series with date index and daily returns (decimal, not percent)
    """
    df = load_benchmark(name, start_date, end_date)
    df = df.set_index("date")
    # pct_change 列中已经是小数（如 0.02 = 2%），直接返回
    returns = df["pct_change"].astype(float)
    return returns


def load_benchmark_equity_curve(name: str,
                                initial_value: float = 1.0,
                                start_date: Optional[str] = None,
                                end_date: Optional[str] = None) -> list[dict]:
    """加载基准指数权益曲线（用于与策略对比）

    Returns:
        list of {date, equity} dicts
    """
    df = load_benchmark(name, start_date, end_date)
    cum = float(initial_value)
    curve = []
    for _, row in df.iterrows():
        ret = float(row["pct_change"])  # 已经是小数形式
        cum *= (1 + ret)
        curve.append({
            "date": str(row["date"])[:10],
            "equity": round(float(cum), 4),
        })
    return curve


# ============================================================
# 行业分类数据
# ============================================================

_INDUSTRY_DIR = Path("/Users/flybirp/Documents/mainland_industry_index_data")


def load_industry_map() -> dict[str, str]:
    """加载股票→申万一级行业映射

    从 mainland_industry_index_data 目录下的行业分类CSV读取。

    Returns:
        {stock_code: industry_name}
    """
    industry_map = {}

    if not _INDUSTRY_DIR.exists():
        return industry_map

    # 优先简单分类
    simple_dir = _INDUSTRY_DIR / "simple"
    target = simple_dir if simple_dir.exists() else _INDUSTRY_DIR

    for fpath in target.glob("*.csv"):
        try:
            df = pd.read_csv(fpath)
            industry_name = fpath.stem
            # 尝试多种列名
            for col in ["code", "stock_code", "ts_code", "symbol"]:
                if col in df.columns:
                    for code in df[col].dropna():
                        code_str = str(code).strip()
                        if "." in code_str:
                            code_str = code_str.split(".")[0]
                        industry_map[code_str] = industry_name
                    break
        except Exception:
            continue

    return industry_map


# 缓存行业映射
_industry_map_cache: Optional[dict[str, str]] = None


def get_stock_industry(code: str) -> str:
    """获取单只股票的行业分类"""
    global _industry_map_cache
    if _industry_map_cache is None:
        _industry_map_cache = load_industry_map()
    return _industry_map_cache.get(str(code), "其他")


def get_industry_distribution(codes: list[str]) -> dict[str, int]:
    """获取股票池的行业分布"""
    industry_map = load_industry_map()
    dist = {}
    for code in codes:
        ind = industry_map.get(str(code), "其他")
        dist[ind] = dist.get(ind, 0) + 1
    return dict(sorted(dist.items(), key=lambda x: -x[1]))
