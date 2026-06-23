"""FastAPI 回测服务"""

from __future__ import annotations
import yaml
import uuid
import threading
import time
from pathlib import Path
from typing import Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import pandas as pd

from backend.backtest_engine import run_backtest, BacktestResult
from backend.strategy_engine import StrategyConfig
from backend.data_loader import list_all_codes

app = FastAPI(title="Quant Backtest System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 回测任务存储
_tasks: dict[str, dict] = {}
_resuts: dict[str, BacktestResult] = {}

STRATEGIES_DIR = Path(__file__).parent.parent / "strategies"


class StrategySaveRequest(BaseModel):
    name: str
    config: dict


class BacktestRequest(BaseModel):
    strategy_name: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    stock_pool_name: Optional[str] = None  # 标的池名称，覆盖YAML中的stock_pool


class StrategyConfigModel(BaseModel):
    name: str = "default"
    k_type: str = "daily"
    initial_capital: float = 100000
    buy_groups: list[dict] = Field(default_factory=list)
    sell_groups: list[dict] = Field(default_factory=list)
    position_pct: float = 1.0
    max_positions: int = 5
    add_threshold: float = 0.0
    add_pct: float = 0.0
    stop_loss_pct: float = 5.0
    take_profit_pct: float = 15.0
    max_hold_days: int = 0
    trailing_stop_pct: float = 0.0
    min_volume_ratio: float = 0.0
    stock_pool: list[str] = Field(default_factory=list)
    commission_rate: float = 0.0003
    stamp_tax_rate: float = 0.001
    entry_ladder: list[dict] = Field(default_factory=list)
    exit_ladder: list[dict] = Field(default_factory=list)


def _config_from_dict(d: dict) -> StrategyConfig:
    """从字典构建 StrategyConfig"""
    return StrategyConfig(
        name=d.get("name", "default"),
        k_type=d.get("k_type", "daily"),
        backtest_mode=d.get("backtest_mode", "signal"),
        initial_capital=d.get("initial_capital", 100000),
        buy_groups=d.get("buy_groups", []),
        sell_groups=d.get("sell_groups", []),
        add_groups=d.get("add_groups", []),
        reduce_groups=d.get("reduce_groups", []),
        position_pct=d.get("position_pct", 1.0),
        max_positions=d.get("max_positions", 5),
        add_threshold=d.get("add_threshold", 0.0),
        add_pct=d.get("add_pct", 0.0),
        reduce_pct=d.get("reduce_pct", 0.5),
        stop_loss_pct=d.get("stop_loss_pct", 5.0),
        take_profit_pct=d.get("take_profit_pct", 15.0),
        max_hold_days=d.get("max_hold_days", 0),
        trailing_stop_pct=d.get("trailing_stop_pct", 0.0),
        min_volume_ratio=d.get("min_volume_ratio", 0.0),
        stock_pool=d.get("stock_pool", []),
        buy_price_type=d.get("buy_price_type", "close"),
        sell_price_type=d.get("sell_price_type", "close"),
        buy_execution=d.get("buy_execution", "same_day"),
        sell_execution=d.get("sell_execution", "same_day"),
        commission_rate=d.get("commission_rate", 0.0003),
        stamp_tax_rate=d.get("stamp_tax_rate", 0.001),
        entry_ladder=d.get("entry_ladder", []),
        exit_ladder=d.get("exit_ladder", []),
    )


# ===================== 标的池 =====================

import json as _json

_POOLS_FILE = Path(__file__).parent.parent / "stock_pools.json"


def _resolve_pool_codes(pool_name: str) -> list[str]:
    """根据池名解析股票代码列表"""
    if not _POOLS_FILE.exists():
        return []
    with open(_POOLS_FILE) as f:
        pools = _json.load(f)
    pool = pools.get(pool_name)
    if not pool:
        return []
    # 如果有明确的codes列表，直接返回
    if "codes" in pool:
        # 只保留数据目录中实际存在的股票
        all_codes = set(list_all_codes())
        return [c for c in pool["codes"] if c in all_codes]
    # 如果是按前缀匹配
    prefixes = pool.get("prefix", [])
    all_codes = list_all_codes()
    if not prefixes:
        # 空prefix = 全量
        return all_codes
    return [c for c in all_codes if any(c.startswith(p) for p in prefixes)]


@app.get("/api/stock-pools")
def api_list_stock_pools():
    """列出所有标的池"""
    if not _POOLS_FILE.exists():
        return {"pools": []}
    with open(_POOLS_FILE) as f:
        pools = _json.load(f)
    result = []
    for name, info in pools.items():
        codes = _resolve_pool_codes(name)
        result.append({
            "name": name,
            "description": info.get("description", ""),
            "count": len(codes),
        })
    return {"pools": result}


@app.get("/api/stock-pools/{pool_name}")
def api_get_stock_pool(pool_name: str):
    """获取标的池详情（含股票代码列表）"""
    codes = _resolve_pool_codes(pool_name)
    if not codes:
        pools_data = {}
        if _POOLS_FILE.exists():
            with open(_POOLS_FILE) as f:
                pools_data = _json.load(f)
        if pool_name not in pools_data:
            raise HTTPException(404, f"标的池 {pool_name} 不存在")
    return {"name": pool_name, "codes": codes, "count": len(codes)}


# ===================== API =====================

@app.get("/api/stocks")
def api_list_stocks():
    """列出所有可回测股票"""
    codes = list_all_codes()
    return {"count": len(codes), "codes": codes}


@app.get("/api/strategies")
def api_list_strategies():
    """列出所有已保存策略"""
    strategies = []
    for f in STRATEGIES_DIR.glob("*.yaml"):
        try:
            with open(f) as fp:
                cfg = yaml.safe_load(fp)
            strategies.append({"name": f.stem, "config": cfg})
        except Exception:
            continue
    return {"strategies": strategies}


@app.get("/api/strategies/{name}")
def api_get_strategy(name: str):
    """获取单个策略"""
    fpath = STRATEGIES_DIR / f"{name}.yaml"
    if not fpath.exists():
        raise HTTPException(404, "策略不存在")
    with open(fpath) as f:
        cfg = yaml.safe_load(f)
    return {"name": name, "config": cfg}


@app.post("/api/strategies")
def api_save_strategy(req: StrategySaveRequest):
    """保存策略"""
    fpath = STRATEGIES_DIR / f"{req.name}.yaml"
    with open(fpath, "w") as f:
        yaml.dump(req.config, f, allow_unicode=True, default_flow_style=False)
    return {"status": "ok", "name": req.name}


@app.delete("/api/strategies/{name}")
def api_delete_strategy(name: str):
    """删除策略"""
    fpath = STRATEGIES_DIR / f"{name}.yaml"
    if fpath.exists():
        fpath.unlink()
    return {"status": "ok"}


@app.post("/api/backtest")
def api_start_backtest(req: BacktestRequest):
    """启动回测"""
    fpath = STRATEGIES_DIR / f"{req.strategy_name}.yaml"
    if not fpath.exists():
        raise HTTPException(404, f"策略 {req.strategy_name} 不存在")

    with open(fpath) as f:
        cfg_dict = yaml.safe_load(f)

    # 标的池覆盖：如果指定了stock_pool_name，覆盖YAML中的stock_pool
    if req.stock_pool_name:
        pool_codes = _resolve_pool_codes(req.stock_pool_name)
        if pool_codes:
            cfg_dict["stock_pool"] = pool_codes

    config = _config_from_dict(cfg_dict)

    task_id = str(uuid.uuid4())[:8]
    _tasks[task_id] = {
        "status": "running",
        "progress": 0,
        "total": 0,
        "started_at": datetime.now().isoformat(),
    }

    def _run():
        try:
            result = run_backtest(config, req.start_date, req.end_date)
            _resuts[task_id] = result
            _tasks[task_id]["status"] = "completed"
            _tasks[task_id]["progress"] = _tasks[task_id]["total"]
        except Exception as e:
            _tasks[task_id]["status"] = "failed"
            _tasks[task_id]["error"] = str(e)

    threading.Thread(target=_run, daemon=True).start()
    return {"task_id": task_id, "status": "running"}


@app.get("/api/backtest/{task_id}")
def api_get_backtest_result(task_id: str):
    """获取回测进度/结果"""
    task = _tasks.get(task_id)
    if not task:
        raise HTTPException(404, "任务不存在")

    if task["status"] == "running":
        return {"task_id": task_id, "status": "running", "progress": task.get("progress", 0), "total": task.get("total", 0)}

    if task["status"] == "failed":
        return {"task_id": task_id, "status": "failed", "error": task.get("error", "")}

    result = _resuts.get(task_id)
    if not result:
        raise HTTPException(404, "结果不存在")

    return {
        "task_id": task_id,
        "status": "completed",
        "summary": {
            "config_name": result.config_name,
            "k_type": result.k_type,
            "backtest_mode": result.backtest_mode,
            "start_date": result.start_date,
            "end_date": result.end_date,
            "initial_capital": result.initial_capital,
            "final_capital": result.final_capital,
            "total_return_pct": result.total_return_pct,
            "annual_return_pct": result.annual_return_pct,
            "max_drawdown_pct": result.max_drawdown_pct,
            "sharpe_ratio": result.sharpe_ratio,
            "win_rate": result.win_rate,
            "profit_loss_ratio": result.profit_loss_ratio,
            "expected_value": result.expected_value,
            "total_trades": result.total_trades,
            "win_trades": result.win_trades,
            "lose_trades": result.lose_trades,
            "avg_profit_pct": result.avg_profit_pct,
            "avg_loss_pct": result.avg_loss_pct,
            "max_profit_pct": result.max_profit_pct,
            "max_loss_pct": result.max_loss_pct,
            "avg_hold_days": result.avg_hold_days,
        },
        "trades": result.trades,
        "equity_curve": result.equity_curve,
        "annual_returns": result.annual_returns,
        "monthly_returns": result.monthly_returns,
    }


@app.get("/api/kline/{code}")
def api_get_kline(code: str, k_type: str = Query("daily"), start_date: Optional[str] = None, end_date: Optional[str] = None, task_id: Optional[str] = None):
    """获取单只股票K线数据 + 交易标注"""
    from backend.data_loader import load_stock

    try:
        df = load_stock(code, k_type)
    except FileNotFoundError:
        raise HTTPException(404, f"股票 {code} 不存在")

    if start_date:
        df = df[df["date"] >= start_date]
    if end_date:
        df = df[df["date"] <= end_date]

    # K线数据
    kline = []
    for _, row in df.iterrows():
        kline.append({
            "date": str(row["date"])[:10],
            "open": round(row["open"], 2),
            "high": round(row["high"], 2),
            "low": round(row["low"], 2),
            "close": round(row["close"], 2),
            "volume": int(row["volume"]) if not pd.isna(row["volume"]) else 0,
        })

    # 交易标注：从回测结果中提取该股票的交易
    # 如果指定task_id则只看该次回测，否则看全部
    annotations = []
    results_to_check = {}
    if task_id:
        if task_id in _resuts:
            results_to_check[task_id] = _resuts[task_id]
    else:
        results_to_check = _resuts

    for tid, result in results_to_check.items():
        for t in result.trades:
            if t["code"] != code:
                continue
            trade_id = t.get("trade_id", "")
            # 买入/加仓标注
            buy_action = t.get("action", "buy")
            annotations.append({
                "date": t["buy_date"],
                "type": buy_action,  # "buy" or "add"
                "price": t["buy_price"],
                "task_id": tid,
                "strategy": result.config_name,
                "trade_id": trade_id,
            })
            # 卖出标注
            if t["sell_date"]:
                sell_type = t.get("action", "clear")
                # action字段在卖出端映射: reduce->reduce, clear/buy/add->clear
                if sell_type in ("reduce",):
                    ann_type = "reduce"
                else:
                    ann_type = "clear"
                annotations.append({
                    "date": t["sell_date"],
                    "type": ann_type,
                    "price": t["sell_price"],
                    "reason": t["sell_reason"],
                    "profit_pct": t["profit_pct"],
                    "task_id": tid,
                    "strategy": result.config_name,
                    "trade_id": trade_id,
                })

    return {"code": code, "k_type": k_type, "kline": kline, "annotations": annotations}


@app.get("/api/results")
def api_list_results():
    """列出所有回测结果"""
    items = []
    for task_id, task in _tasks.items():
        if task["status"] == "completed":
            r = _resuts.get(task_id)
            if r:
                items.append({
                    "task_id": task_id,
                    "config_name": r.config_name,
                    "k_type": r.k_type,
                    "total_return_pct": r.total_return_pct,
                    "win_rate": r.win_rate,
                    "total_trades": r.total_trades,
                    "started_at": task.get("started_at", ""),
                })
    return {"results": items}


# 静态文件（前端 build 产物）
frontend_dist = Path(__file__).parent.parent / "frontend" / "dist"
if frontend_dist.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8200)
