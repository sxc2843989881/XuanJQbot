"""阶段2：多因子模块

包含3个核心因子（知识库提取）：
1. 回归动量：年化收益×R²（替换阶段1的简单ROC）
2. RSRS阻力支撑相对强度：High/Low回归斜率×R²
3. 双均线趋势过滤：价>MA20且MA20>MA60

知识库依据：
- 04_因子库/01_动量因子.md：回归动量公式
- 03_技术指标库/01_RSRS阻力支撑相对强度.md：RSRS完整实现
- 03_技术指标库/03_双均线趋势过滤.md：双均线过滤
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


# ============================================================
# 因子1：回归动量（年化收益×R²）
# ============================================================

def calculate_regression_momentum(close: pd.DataFrame, window: int = 25) -> pd.DataFrame:
    """回归动量：年化收益率 × R²

    知识库02方法二：
        ln(P_t) = α + β·t
        年化收益率 = e^(250·β) - 1
        综合得分 = 年化收益率 × R²

    R²意义：稳步上涨R²接近1，暴涨暴跌R²很低，用于过滤"妖基"

    Args:
        close: 收盘价 DataFrame
        window: 回归窗口（25日为同花顺默认）

    Returns:
        DataFrame: 回归动量得分
    """
    result = pd.DataFrame(index=close.index, columns=close.columns, dtype=float)

    for code in close.columns:
        prices = close[code].values
        scores = np.full(len(prices), np.nan)

        for t in range(window, len(prices)):
            y = np.log(prices[t - window : t])
            if np.any(np.isnan(y)) or np.any(y <= 0):
                continue
            x = np.arange(window).reshape(-1, 1)
            model = LinearRegression()
            model.fit(x, y)
            slope = model.coef_[0]
            annual_return = np.exp(slope * 250) - 1
            r_squared = model.score(x, y)
            scores[t] = annual_return * r_squared

        result[code] = scores

    return result


# ============================================================
# 因子2：RSRS阻力支撑相对强度
# ============================================================

def calculate_rsrs(high: pd.Series, low: pd.Series, window: int = 18) -> pd.Series:
    """RSRS修正版 = 斜率 × R²

    知识库03_RSRS：
        High_t = α + β·Low_t
        β = RSRS值，表示支撑强度相对阻力强度
        修正版 = β × R²（R²过滤噪声）

    Args:
        high: 最高价 Series
        low: 最低价 Series
        window: 回归窗口（18日为光大默认）

    Returns:
        Series: RSRS修正值
    """
    rsrs_values = pd.Series(index=high.index, dtype=float)

    for i in range(window, len(high)):
        y = high.iloc[i - window : i].values
        x = low.iloc[i - window : i].values.reshape(-1, 1)

        if np.any(np.isnan(y)) or np.any(np.isnan(x)):
            continue

        model = LinearRegression()
        model.fit(x, y)
        rsrs_values.iloc[i] = model.coef_[0] * model.score(x, y)

    return rsrs_values


def calculate_rsrs_panel(ohlcv_dict: dict, window: int = 18) -> pd.DataFrame:
    """计算所有ETF的RSRS

    Args:
        ohlcv_dict: {code: DataFrame[open,high,low,close,volume]}
        window: RSRS窗口

    Returns:
        DataFrame: RSRS值, index=date, columns=ETF代码
    """
    rsrs_data = {}
    for code, df in ohlcv_dict.items():
        rsrs_data[code] = calculate_rsrs(df["high"], df["low"], window=window)

    result = pd.DataFrame(rsrs_data)
    return result


# ============================================================
# 因子3：双均线趋势过滤
# ============================================================

def calculate_dual_ma_filter(close: pd.DataFrame, ma_short: int = 20, ma_long: int = 60) -> pd.DataFrame:
    """双均线趋势过滤

    知识库03_双均线：
        仅当 收盘价 > MA20 且 MA20 > MA60 时为True（趋势向上）

    Args:
        close: 收盘价
        ma_short: 短期均线窗口
        ma_long: 长期均线窗口

    Returns:
        DataFrame: True=趋势向上可入场, False=趋势向下不入场
    """
    ma_s = close.rolling(ma_short).mean()
    ma_l = close.rolling(ma_long).mean()
    return (close > ma_s) & (ma_s > ma_l)


# ============================================================
# 趋势反转风控（zfs1策略代码）
# ============================================================

def calculate_reversal_filter(close: pd.DataFrame, drop_threshold: float = 0.05) -> pd.DataFrame:
    """趋势反转风控：过滤近期跌幅过大的ETF

    知识库07_zfs1策略：
        con1: 三天内有一天跌超5%
        con2: 三天内每天都跌，总共跌超5%
        任一触发则该ETF得分为0（不入选）

    Args:
        close: 收盘价
        drop_threshold: 跌幅阈值（5%）

    Returns:
        DataFrame: True=安全可入选, False=近期暴跌不入选
    """
    safe = pd.DataFrame(True, index=close.index, columns=close.columns)

    for code in close.columns:
        prices = close[code].values
        for t in range(3, len(prices)):
            if np.any(np.isnan(prices[t-3:t+1])):
                continue
            # con1: 三天内有一天跌超阈值
            r1 = prices[t] / prices[t-1] - 1
            r2 = prices[t-1] / prices[t-2] - 1
            r3 = prices[t-2] / prices[t-3] - 1
            con1 = min(r1, r2, r3) < -drop_threshold
            # con2: 三天连跌且总跌幅超阈值
            con2 = (r1 < 0) and (r2 < 0) and (r3 < 0) and (prices[t] / prices[t-3] - 1 < -drop_threshold)
            if con1 or con2:
                safe.iloc[t, safe.columns.get_loc(code)] = False

    return safe


# ============================================================
# 综合评分：动量 × 质量
# ============================================================

def calculate_composite_score(
    close: pd.DataFrame,
    ohlcv_dict: dict,
    momentum_window: int = 25,
    rsrs_window: int = 18,
    ma_short: int = 20,
    ma_long: int = 60,
    use_rsrs: bool = True,
    use_ma_filter: bool = True,
    use_reversal_filter: bool = True,
) -> tuple:
    """计算综合评分：动量×质量

    知识库04_质量因子：
        综合 = 动量得分 × 质量得分
        质量因子 = RSRS（衡量支撑强度）

    风控过滤：
        - 双均线过滤：趋势向下时得分为0
        - 趋势反转过滤：近期暴跌时得分为0

    Args:
        close: 收盘价
        ohlcv_dict: OHLCV数据（用于RSRS）
        momentum_window: 回归动量窗口
        rsrs_window: RSRS窗口
        ma_short/ma_long: 双均线参数
        use_*: 各模块开关（用于消融分析）

    Returns:
        (score, momentum, rsrs, ma_filter, reversal_filter)
    """
    # 1. 回归动量
    momentum = calculate_regression_momentum(close, window=momentum_window)

    # 2. RSRS质量因子
    if use_rsrs:
        rsrs = calculate_rsrs_panel(ohlcv_dict, window=rsrs_window)
        # RSRS标准化到0-1区间（正值加分，负值减分）
        rsrs_score = rsrs.clip(-2, 2) / 4 + 0.5  # 映射到[0, 1]
    else:
        rsrs_score = pd.DataFrame(1.0, index=close.index, columns=close.columns)
        rsrs = rsrs_score.copy()

    # 3. 综合得分 = 动量 × 质量
    score = momentum * rsrs_score

    # 4. 双均线过滤
    if use_ma_filter:
        ma_filter = calculate_dual_ma_filter(close, ma_short, ma_long)
        score = score.where(ma_filter, 0)
    else:
        ma_filter = pd.DataFrame(True, index=close.index, columns=close.columns)

    # 5. 趋势反转过滤
    if use_reversal_filter:
        reversal_filter = calculate_reversal_filter(close)
        score = score.where(reversal_filter, 0)
    else:
        reversal_filter = pd.DataFrame(True, index=close.index, columns=close.columns)

    # 6. 负得分设为0（不做空）
    score = score.where(score > 0, 0)

    return score, momentum, rsrs, ma_filter, reversal_filter


# ============================================================
# 信号生成
# ============================================================

def generate_signal_v2(
    close: pd.DataFrame,
    score: pd.DataFrame,
    rebalance_freq: str = "M",
    hold_count: int = 2,
    empty_threshold: float = 0.0,
) -> pd.DataFrame:
    """生成轮动信号（阶段2版）

    规则：
    1. 按指定频率调仓
    2. 选评分最高的N只ETF等权持有
    3. 若最高分<阈值，空仓
    4. 信号shift(1)避免未来函数

    Args:
        close: 收盘价
        score: 综合评分
        rebalance_freq: 调仓频率
        hold_count: 持仓数量
        empty_threshold: 空仓阈值（最高分<此值则空仓）

    Returns:
        DataFrame: 持仓权重信号
    """
    # 调仓日
    if rebalance_freq == "M":
        rebalance_idx = close.groupby(close.index.to_period("M")).apply(lambda x: x.index[-1])
    elif rebalance_freq == "W":
        rebalance_idx = close.groupby(close.index.to_period("W")).apply(lambda x: x.index[-1])
    else:
        raise ValueError(f"不支持的频率: {rebalance_freq}")

    # 每个调仓日决定持仓
    holdings = {}  # date -> list of etf codes or None
    for rebalance_date in rebalance_idx:
        if rebalance_date not in score.index:
            continue
        score_today = score.loc[rebalance_date]
        if score_today.isna().all():
            holdings[rebalance_date] = None
            continue

        # 选评分最高的N只（评分>0）
        valid_scores = score_today[score_today > 0].sort_values(ascending=False)
        if len(valid_scores) == 0 or valid_scores.iloc[0] < empty_threshold:
            holdings[rebalance_date] = None  # 空仓
        else:
            top_n = valid_scores.head(hold_count).index.tolist()
            holdings[rebalance_date] = top_n

    # 逐日前向填充
    signal = pd.DataFrame(0, index=close.index, columns=close.columns, dtype=float)
    current_holding = None
    for date in close.index:
        if date in holdings:
            current_holding = holdings[date]
        if current_holding is not None:
            weight = 1.0 / len(current_holding)
            for code in current_holding:
                signal.loc[date, code] = weight

    # shift(1)避免未来函数
    signal = signal.shift(1).fillna(0)

    return signal


if __name__ == "__main__":
    # 快速自测
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from data_generator_v2 import generate_simulation_data_v2

    close_df, ohlcv_dict, _ = generate_simulation_data_v2(n_years=3, seed=42)

    print("计算综合评分...")
    score, momentum, rsrs, ma_filter, reversal_filter = calculate_composite_score(
        close_df, ohlcv_dict, use_rsrs=True, use_ma_filter=True, use_reversal_filter=True,
    )

    print(f"评分形状: {score.shape}")
    print(f"\n最后5日评分:")
    print(score.tail().round(3))

    signal = generate_signal_v2(close_df, score, rebalance_freq="M", hold_count=2)
    print(f"\n各ETF持仓天数占比:")
    for col in signal.columns:
        hold_pct = (signal[col] > 0).sum() / len(signal) * 100
        print(f"  {col}: {hold_pct:.1f}%")
