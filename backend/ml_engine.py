"""
ML 策略引擎 — 多因子 + 机器学习

职责：
  1. 从日K数据计算因子值
  2. 构造训练样本 (X: 因子, y: 未来收益)
  3. 训练/加载模型
  4. 生成选股信号

当前为骨架实现。因子计算和模型训练的具体实现
在后续开发中填充。
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── 因子计算器 ────────────────────────────────────────────────────

class FactorCalculator:
    """从日K DataFrame 计算因子值"""

    @staticmethod
    def momentum(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
        """动量因子：过去N日收益率"""
        return df["close"].pct_change(periods=lookback)

    @staticmethod
    def volatility(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
        """波动率因子：过去N日日收益率标准差"""
        return df["close"].pct_change().rolling(lookback).std()

    @staticmethod
    def volume_ratio(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
        """量比因子：当日成交量 / 过去N日均量"""
        return df["volume"] / df["volume"].rolling(lookback).mean()

    @staticmethod
    def rsi(df: pd.DataFrame, lookback: int = 14) -> pd.Series:
        """RSI 因子"""
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0.0).rolling(lookback).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(lookback).mean()
        rs = gain / loss.replace(0, np.nan)
        return 100.0 - (100.0 / (1.0 + rs))

    @staticmethod
    def price_position(df: pd.DataFrame, lookback: int = 60) -> pd.Series:
        """价格位置因子：(close - low_N) / (high_N - low_N)"""
        low_n = df["low"].rolling(lookback).min()
        high_n = df["high"].rolling(lookback).max()
        denom = high_n - low_n
        denom = denom.replace(0, np.nan)
        return (df["close"] - low_n) / denom * 100.0

    # 因子名 → 计算函数 映射
    FACTOR_FUNCTIONS = {
        "momentum": momentum,
        "volatility": volatility,
        "volume_ratio": volume_ratio,
        "rsi": rsi,
        "price_position": price_position,
    }

    @classmethod
    def compute_factor(
        cls, df: pd.DataFrame, factor_name: str, lookback_days: int
    ) -> pd.Series:
        """计算单个因子"""
        fn = cls.FACTOR_FUNCTIONS.get(factor_name)
        if fn is None:
            logger.warning("Unknown factor: %s, returning zeros.", factor_name)
            return pd.Series(0.0, index=df.index)
        return fn.__func__(df, lookback_days)

    @classmethod
    def compute_all_factors(
        cls, df: pd.DataFrame, factor_list: list[dict]
    ) -> pd.DataFrame:
        """
        计算所有因子，返回 DataFrame。

        Args:
            df: 日K DataFrame (需含 close, high, low, volume)
            factor_list: [{"name": "momentum", "lookback_days": 20, "transform": "zscore"}, ...]

        Returns:
            DataFrame, columns = factor names, index = df.index
        """
        result = pd.DataFrame(index=df.index)
        for f in factor_list:
            name = f.get("name", "")
            lookback = f.get("lookback_days", 20)
            transform = f.get("transform", "raw")

            raw = cls.compute_factor(df, name, lookback)

            if transform == "zscore":
                raw = (raw - raw.rolling(252).mean()) / raw.rolling(252).std()
            elif transform == "rank":
                raw = raw.rolling(252).rank(pct=True)

            col_name = f"{name}_{lookback}d"
            result[col_name] = raw

        return result


# ── 训练样本构造 ──────────────────────────────────────────────────

def build_training_data(
    stock_dfs: dict[str, pd.DataFrame],
    factor_list: list[dict],
    label_horizon_days: int = 5,
    min_samples: int = 252,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    从多只股票构造训练数据集。

    Args:
        stock_dfs: {code: DataFrame} - 每只股票的日K数据
        factor_list: 因子定义列表
        label_horizon_days: 标签前瞻天数
        min_samples: 每只股票最少需要的数据行数

    Returns:
        (X, y) where X is factor values, y is future return
    """
    all_X = []
    all_y = []

    for code, df in stock_dfs.items():
        if len(df) < min_samples:
            continue

        factors = FactorCalculator.compute_all_factors(df, factor_list)

        # 标签：未来N日收益率
        future_close = df["close"].shift(-label_horizon_days)
        y_series = (future_close - df["close"]) / df["close"]

        # 合并 X 和 y，去掉 NaN
        combined = pd.concat([factors, y_series.rename("target")], axis=1)
        combined = combined.dropna()

        if len(combined) < 50:
            continue

        all_X.append(combined.drop(columns=["target"]))
        all_y.append(combined["target"])

    if not all_X:
        return pd.DataFrame(), pd.Series(dtype=float)

    X = pd.concat(all_X, axis=0)
    y = pd.concat(all_y, axis=0)

    return X, y


# ── 模型训练接口 ──────────────────────────────────────────────────

def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    model_type: str = "xgboost",
    model_params: dict | None = None,
) -> Any:
    """
    训练 ML 模型。

    Args:
        X: 因子特征矩阵
        y: 目标变量 (未来收益)
        model_type: 'linear', 'xgboost', 'lightgbm', 'mlp'
        model_params: 模型超参

    Returns:
        训练好的模型对象 (sklearn-compatible)
    """
    params = model_params or {}

    if model_type == "linear":
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(X, y)
        return model

    elif model_type == "xgboost":
        try:
            import xgboost as xgb
        except ImportError:
            logger.warning("xgboost not installed, falling back to linear")
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
            model.fit(X, y)
            return model

        model = xgb.XGBRegressor(
            max_depth=params.get("max_depth", 5),
            learning_rate=params.get("learning_rate", 0.05),
            n_estimators=params.get("n_estimators", 200),
            subsample=params.get("subsample", 0.8),
            random_state=42,
        )
        model.fit(X, y)
        return model

    elif model_type == "lightgbm":
        try:
            import lightgbm as lgb
        except ImportError:
            logger.warning("lightgbm not installed, falling back to linear")
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
            model.fit(X, y)
            return model

        model = lgb.LGBMRegressor(
            max_depth=params.get("max_depth", 5),
            learning_rate=params.get("learning_rate", 0.05),
            n_estimators=params.get("n_estimators", 200),
            subsample=params.get("subsample", 0.8),
            random_state=42,
            verbose=-1,
        )
        model.fit(X, y)
        return model

    elif model_type == "mlp":
        try:
            from sklearn.neural_network import MLPRegressor
        except ImportError:
            logger.warning("sklearn not available, returning None")
            return None

        model = MLPRegressor(
            hidden_layer_sizes=params.get("hidden_layers", (64, 32)),
            max_iter=params.get("max_iter", 500),
            random_state=42,
        )
        model.fit(X, y)
        return model

    else:
        logger.warning("Unknown model_type: %s", model_type)
        return None


# ── 选股信号生成 ──────────────────────────────────────────────────

def generate_signals(
    model: Any,
    stock_dfs: dict[str, pd.DataFrame],
    factor_list: list[dict],
    date: pd.Timestamp,
    top_n: int = 20,
) -> list[dict]:
    """
    在给定日期，用模型对所有股票打分，选出 top N。

    Args:
        model: 训练好的模型
        stock_dfs: 所有候选股票的 DataFrame
        factor_list: 因子定义
        date: 当前交易日
        top_n: 选股数量

    Returns:
        [{code, score, date}, ...] 按 score 降序排列
    """
    scores = []

    for code, df in stock_dfs.items():
        # 找到当前日期所在行
        df_dates = pd.to_datetime(df["date"] if "date" in df.columns else df.index)
        mask = df_dates == date
        if not mask.any():
            continue

        idx = df.index[mask][0] if not df.index.equals(df_dates) else mask.idxmax() if hasattr(mask, 'idxmax') else None
        if idx is None:
            continue

        # 提取该日期的因子值
        factors = FactorCalculator.compute_all_factors(df, factor_list)
        if idx not in factors.index:
            continue

        X_single = factors.loc[[idx]]
        if X_single.isna().any().any():
            continue

        try:
            score = float(model.predict(X_single)[0])
        except Exception:
            continue

        scores.append({"code": code, "score": round(score, 6), "date": str(date)[:10]})

    scores.sort(key=lambda x: x["score"], reverse=True)
    return scores[:top_n]


# ── ML 回测主流程 ─────────────────────────────────────────────────

def run_ml_backtest(
    config,  # StrategyConfig
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    progress_callback=None,
):
    """
    ML 策略回测主入口。

    Pipeline:
      1. 加载所有股票数据
      2. 按日期拆分 train/val/test
      3. 在 train 上计算因子 + 训练模型
      4. 在每个调仓日生成信号
      5. 执行交易 → 输出 BacktestResult
    """
    from backend.data_loader import load_stock_with_indicators
    from backend.backtest_engine import BacktestResult

    codes = config.stock_pool if config.stock_pool else []
    if not codes:
        logger.error("ML backtest requires stock_pool")
        return BacktestResult(
            config_name=config.name,
            k_type=config.k_type,
            backtest_mode=config.backtest_mode,
            start_date=start_date or "",
            end_date=end_date or "",
            initial_capital=config.initial_capital,
            final_capital=config.initial_capital,
            total_return_pct=0.0,
            annual_return_pct=0.0,
            max_drawdown_pct=0.0,
            sharpe_ratio=0.0,
            win_rate=0.0,
            profit_loss_ratio=0.0,
            expected_value=0.0,
            total_trades=0,
            win_trades=0,
            lose_trades=0,
            avg_profit_pct=0.0,
            avg_loss_pct=0.0,
            max_profit_pct=0.0,
            max_loss_pct=0.0,
            avg_hold_days=0.0,
        )

    # TODO: 实现完整的 ML 回测流程
    # 当前返回空结果，提醒用户这是骨架
    logger.warning(
        "ML backtest engine is a skeleton. "
        "Implement train/val/test split, model training, and signal generation."
    )

    return BacktestResult(
        config_name=config.name,
        k_type=config.k_type,
        backtest_mode=config.backtest_mode,
        start_date=start_date or "",
        end_date=end_date or "",
        initial_capital=config.initial_capital,
        final_capital=config.initial_capital,
        total_return_pct=0.0,
        annual_return_pct=0.0,
        max_drawdown_pct=0.0,
        sharpe_ratio=0.0,
        win_rate=0.0,
        profit_loss_ratio=0.0,
        expected_value=0.0,
        total_trades=0,
        win_trades=0,
        lose_trades=0,
        avg_profit_pct=0.0,
        avg_loss_pct=0.0,
        max_profit_pct=0.0,
        max_loss_pct=0.0,
        avg_hold_days=0.0,
    )
