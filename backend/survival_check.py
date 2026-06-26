"""Survivorship bias checker."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

DEFAULT_DATA_DIR = "/Users/flybirp/Documents/mainland_data_2014"
DEFAULT_POOLS_PATH = "/Users/flybirp/Documents/quant_backtest/stock_pools.json"


def check_survivorship(
    data_dir: str = DEFAULT_DATA_DIR,
    pools_path: str = DEFAULT_POOLS_PATH,
    tushare_token: str | None = None,
) -> dict:
    result = {"total_in_data": 0, "delisted_in_data": [],
              "delisted_missing": 0, "pools_at_risk": {}, "status": "clean"}
    dp = Path(data_dir)
    if not dp.exists():
        result["status"] = "error"; return result
    data_codes = set(f.stem for f in dp.glob("*.csv"))
    result["total_in_data"] = len(data_codes)
    delisted_codes: set[str] = set()
    if tushare_token:
        try:
            import tushare as ts
            ts.set_token(tushare_token)
            pro = ts.pro_api()
            df = pro.stock_basic(exchange="", list_status="D",
                                 fields="ts_code,symbol,name,delist_date")
            delisted_codes = set(df["symbol"].tolist())
        except Exception:
            pass
    result["delisted_in_data"] = sorted(delisted_codes & data_codes)
    result["delisted_missing"] = len(delisted_codes - data_codes)
    pp = Path(pools_path)
    if pp.exists():
        with open(pp) as f:
            pools = json.load(f)
        for pn, codes in pools.items():
            cs = set(str(c).zfill(6) for c in codes)
            at_risk = cs & delisted_codes
            if at_risk:
                result["pools_at_risk"][pn] = sorted(at_risk)
    if result["pools_at_risk"]:
        result["status"] = "critical"
    elif result["delisted_in_data"]:
        result["status"] = "warning"
    return result


def format_survivorship_report(result: dict) -> str:
    lines = ["生存偏差检查", "-" * 50,
             f"数据总股票: {result['total_in_data']}",
             f"退市但在数据中: {len(result['delisted_in_data'])}",
             f"退市且缺失: {result['delisted_missing']}"]
    if result["delisted_in_data"]:
        lines.append(f"\n⚠ 数据中退市股(影响小): {result['delisted_in_data'][:10]}")
    if result["pools_at_risk"]:
        lines.append("\n!! 股票池含退市股:")
        for p, c in result["pools_at_risk"].items():
            lines.append(f"  {p}: {c}")
    lines.append(f"\n状态: {result['status']}")
    if result["status"] == "clean":
        lines.append("当前池无退市股，无生存偏差。")
    elif result["status"] == "warning":
        lines.append("数据有退市股但不在池中，影响有限。")
    else:
        lines.append("!! 池含退市股，收益被高估!")
    return "\n".join(lines)
