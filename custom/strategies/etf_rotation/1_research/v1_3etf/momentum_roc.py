"""阶段1：ROC动量因子

简单动量（ROC, Rate of Change）：
    M_N = (P_t - P_{t-N}) / P_{t-N}

知识库依据：01_ETF轮动策略/02_动量因子计算方法.md 方法一
- 优点：计算简单，适合入门验证
- 缺点：仅比较首尾，忽略中间路径，易选到上蹿下跳标的
- 改进方向：阶段2升级为回归动量（年化×R²）
"""
import pandas as pd
import numpy as np


def calculate_roc_momentum(close: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """计算N日ROC动量

    Args:
        close: 收盘价 DataFrame, index=date, columns=ETF代码
        window: 动量计算窗口（交易日）

    Returns:
        DataFrame: 动量值，同close形状
    """
    return close.pct_change(periods=window)


def generate_signal(close: pd.DataFrame, window: int = 60, rebalance_freq: str = "M") -> pd.DataFrame:
    """生成轮动信号

    规则（知识库01入门版）：
    1. 计算每个ETF的N日动量
    2. 选择动量最强的1个ETF持有
    3. 若最强动量<0（全池下跌），则空仓
    4. 按指定频率调仓（月频/周频）
    5. 信号 shift(1) 后移一天，避免未来函数

    Args:
        close: 收盘价 DataFrame
        window: 动量窗口
        rebalance_freq: 调仓频率 'W'=周频, 'M'=月频, 'Q'=季频

    Returns:
        DataFrame: 持仓信号，1=持有, 0=空仓，同close形状
    """
    # 1. 计算动量
    momentum = calculate_roc_momentum(close, window)

    # 2. 确定调仓日（月末/周末/季末）
    if rebalance_freq == "M":
        rebalance_idx = close.groupby(close.index.to_period("M")).apply(lambda x: x.index[-1])
    elif rebalance_freq == "W":
        rebalance_idx = close.groupby(close.index.to_period("W")).apply(lambda x: x.index[-1])
    elif rebalance_freq == "Q":
        rebalance_idx = close.groupby(close.index.to_period("Q")).apply(lambda x: x.index[-1])
    else:
        raise ValueError(f"不支持的频率: {rebalance_freq}")

    # 3. 在每个调仓日决定持仓标的（只持有1只或空仓）
    holdings = {}  # date -> etf_code 或 None(空仓)
    for rebalance_date in rebalance_idx:
        if rebalance_date not in momentum.index:
            continue
        mom_today = momentum.loc[rebalance_date]
        if mom_today.isna().all():
            continue

        best_etf = mom_today.idxmax()
        best_mom = mom_today.max()

        if pd.notna(best_mom) and best_mom > 0:
            holdings[rebalance_date] = best_etf
        else:
            holdings[rebalance_date] = None  # 全池动量<0，空仓

    # 4. 逐日前向填充持仓（调仓日之间保持持仓不变）
    signal = pd.DataFrame(0, index=close.index, columns=close.columns)
    current_holding = None
    for date in close.index:
        if date in holdings:
            current_holding = holdings[date]
        if current_holding is not None:
            signal.loc[date, current_holding] = 1

    # 5. 信号 shift(1) 后移一天，避免未来函数
    # 当日收盘计算信号，次日开盘执行
    signal = signal.shift(1).fillna(0)

    return signal


def get_rebalance_dates(close: pd.DataFrame, freq: str = "M") -> list:
    """获取调仓日期列表（用于报告展示）"""
    if freq == "M":
        dates = close.groupby(close.index.to_period("M")).apply(lambda x: x.index[-1])
    elif freq == "W":
        dates = close.groupby(close.index.to_period("W")).apply(lambda x: x.index[-1])
    elif freq == "Q":
        dates = close.groupby(close.index.to_period("Q")).apply(lambda x: x.index[-1])
    else:
        raise ValueError(f"不支持的频率: {freq}")
    return list(dates)


if __name__ == "__main__":
    # 快速自测
    from data_generator import generate_simulation_data

    close_df, _, _ = generate_simulation_data(n_years=3, seed=42)
    print(f"数据: {close_df.shape}")

    momentum = calculate_roc_momentum(close_df, window=60)
    print(f"\n60日动量（最后5行）:")
    print(momentum.tail())

    signal = generate_signal(close_df, window=60, rebalance_freq="M")
    print(f"\n持仓信号（最后5行）:")
    print(signal.tail())

    # 信号统计
    print(f"\n各ETF持仓天数占比:")
    for col in signal.columns:
        hold_pct = (signal[col] == 1).sum() / len(signal) * 100
        print(f"  {col}: {hold_pct:.1f}%")
