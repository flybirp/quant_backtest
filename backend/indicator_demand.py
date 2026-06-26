"""
按需计算指标的映射和辅助函数。

每个条件(indicator name)映射到它需要的预计算列名。
compute_all_indicators 只算需要的列，跳过其余。
"""

# 条件名 → 需要的预计算列集合
CONDITION_TO_COLS: dict[str, set[str]] = {
    # 均线类
    "ma_above": {"ma5", "ma10", "ma20", "ma30", "ma60", "ma120", "ma250"},
    "ma_bullish_alignment": {"ma5", "ma10", "ma20", "ma60"},
    "price_near_ma": {"ma5", "ma10", "ma20", "ma60"},
    "price_near_zhixing": {"zhixing_fast", "zhixing_slow", "dist_to_zhixing_fast", "dist_to_zhixing_slow"},

    # 知行量化
    "zhixing_golden_cross": {"zhixing_fast", "zhixing_slow"},
    "zhixing_golden_cross_hold": {"zhixing_fast", "zhixing_slow"},
    "zhixing_fast_below_slow": {"zhixing_fast", "zhixing_slow"},
    "zhixing_trend_broken": {"zhixing_fast", "zhixing_slow"},
    "under_slow_days": {"zhixing_slow"},

    # RSI
    "rsi_overbought": {"rsi14", "rsi6"},
    "rsi_oversold": {"rsi14", "rsi6"},

    # KDJ
    "kdj_golden_cross": {"kdj_k", "kdj_d", "kdj_j"},
    "kdj_oversold": {"kdj_k", "kdj_d", "kdj_j"},

    # MACD
    "macd_golden_cross": {"macd_dif", "macd_dea", "macd_bar"},
    "macd_histogram_positive": {"macd_bar"},

    # 布林带
    "bb_lower_touch": {"bb_upper", "bb_mid", "bb_lower"},
    "bb_squeeze": {"bb_upper", "bb_mid", "bb_lower"},

    # 量能
    "volume_anomaly": {"is_double_vol", "is_vol_top5", "vol_ma20"},
    "volume_anomaly_strong": {"is_double_vol", "is_vol_top5"},
    "volume_explosion": {"vol_explosion"},
    "volume_double": {"is_double_vol"},
    "vol_rank_high": {"vol_rank_pct"},
    "vol_shrink": {"vol_ma20"},

    # 位置
    "low_position": {"price_position_pct", "dist_to_ma60"},
    "high_position": {"price_position_pct"},
    "not_distribution": {"pct_change", "is_double_vol", "price_position_pct"},

    # 口袋支点
    "pocket_pivot": {"pocket_pivot_vol"},

    # 双底
    "double_bottom": {"double_bottom_support"},

    # 涨跌幅
    "pct_change_gt": {"pct_change"},
    "pct_change_lt": {"pct_change"},

    # 相对抗跌
    "relative_strength": {"relative_strength_60"},  # 默认，会被动态扩展

    # 大盘恐慌
    "market_crash_recent": {"market_crash_30d"},
    "market_crash_fast": {"market_crash_fast_10d"},
    "market_bull": set(),       # 用 regime.py 实时算，不需要预计算列
    "market_bear": set(),
    "market_consolidation": set(),
    "zz1000_crash": set(),      # 用指数数据实时算

    # 基本面（来自CSV原始列，不需要预计算）
    "pe_below": set(),
    "pe_ttm_below": set(),
    "pb_below": set(),
    "pe_pct_low": set(),
    "total_mv_above": set(),
    "total_mv_below": set(),
    "turnover_rate_above": set(),
    "turnover_rate_below": set(),

    # 个股恐慌（用close实时算）
    "stock_deep_dd": set(),
    "dd_concentration": set(),
    "amplitude_gt": set(),
    "stabilized": set(),
    "volume_contracting": set(),  # 用volume实时算
}

# 指标列 → 计算所需的原始列（用于判断是否需要计算）
# None = 需要全部原始列（close/high/low/volume）
_COL_DEPENDENCIES: dict[str, set[str] | None] = {
    # 均线
    "ma5": {"close"}, "ma10": {"close"}, "ma20": {"close"}, "ma30": {"close"},
    "ma60": {"close"}, "ma120": {"close"}, "ma250": {"close"},
    # MACD
    "macd_dif": {"close"}, "macd_dea": {"close"}, "macd_bar": {"close"},
    # RSI
    "rsi14": {"close"}, "rsi6": {"close"},
    # KDJ
    "kdj_k": {"high", "low", "close"}, "kdj_d": {"high", "low", "close"}, "kdj_j": {"high", "low", "close"},
    # 布林带
    "bb_upper": {"close"}, "bb_mid": {"close"}, "bb_lower": {"close"},
    # ATR
    "atr14": {"high", "low", "close"},
    # 量能
    "vol_ma20": {"volume"}, "pct_change": {"close"},
    # 知行
    "zhixing_fast": {"close"}, "zhixing_slow": {"close"},
    "zz_short": {"high", "low", "close"}, "zz_long": {"high", "low", "close"}, "zz_spread": {"high", "low", "close"},
    # 衍生
    "vol_rank_pct": {"volume"}, "price_position_pct": {"close"},
    "dist_to_ma60": {"close", "ma60"}, "dist_to_zhixing_fast": {"close", "zhixing_fast"},
    "dist_to_zhixing_slow": {"close", "zhixing_slow"},
    "is_double_vol": {"volume"}, "is_vol_top5": {"volume"},
    "pocket_pivot_vol": {"volume", "close"},
    "bbi": {"close"},
    "double_bottom_support": {"low", "close"},
    "vol_explosion": {"volume", "close", "vol_rank_pct", "vol_ma20"},
    # 大盘
    "relative_strength_5": None, "relative_strength_10": None,
    "relative_strength_20": None, "relative_strength_60": None,
    "market_crash_30d": None, "market_crash_fast_10d": None,
}


def extract_required_cols(indicator_names: list[str]) -> set[str]:
    """从策略条件名列表提取所有需要的预计算列名。"""
    cols: set[str] = set()
    for name in indicator_names:
        if name in CONDITION_TO_COLS:
            cols.update(CONDITION_TO_COLS[name])
    # 递归展开依赖（比如 vol_explosion 需要 vol_rank_pct 和 vol_ma20）
    expanded = set(cols)
    for col in cols:
        deps = _COL_DEPENDENCIES.get(col)
        if deps:
            expanded.update(deps)
    return expanded


def extract_indicator_names(strategy_config) -> list[str]:
    """从 StrategyConfig 提取所有 indicator 名称。"""
    names = []
    for g in strategy_config.buy_groups:
        for c in g.conditions:
            names.append(c.indicator if hasattr(c, 'indicator') else c.get('indicator', ''))
    for g in strategy_config.sell_groups:
        for c in g.conditions:
            names.append(c.indicator if hasattr(c, 'indicator') else c.get('indicator', ''))
    # reduce_groups
    for g in getattr(strategy_config, 'reduce_groups', []) or []:
        for c in g.conditions:
            names.append(c.indicator if hasattr(c, 'indicator') else c.get('indicator', ''))
    return [n for n in names if n]
