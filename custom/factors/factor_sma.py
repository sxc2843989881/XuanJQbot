"""SMA 因子 - 简单移动平均

因子逻辑独立抽离，策略只调用因子，便于复用与回测。
"""
import pandas as pd


def calc_sma(series: pd.Series, period: int) -> pd.Series:
    """计算简单移动平均"""
    return series.rolling(window=period, min_periods=1).mean()


def sma_cross_signal(close: pd.Series, fast: int = 10, slow: int = 30) -> pd.Series:
    """SMA 交叉信号

    Args:
        close: 收盘价序列
        fast: 快线周期
        slow: 慢线周期

    Returns:
        信号序列：1=多头（fast 上穿 slow），-1=空头，0=无
    """
    sma_fast = calc_sma(close, fast)
    sma_slow = calc_sma(close, slow)
    diff = sma_fast - sma_slow
    signal = pd.Series(0, index=close.index)
    signal[diff > 0] = 1
    signal[diff < 0] = -1
    return signal
