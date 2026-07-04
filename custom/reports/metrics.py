"""custom.reports.metrics - 通用回测指标计算

所有指标计算与策略解耦，输入为 pandas Series（资产净值/收益率序列）。
回撤定义遵循用户偏好：完整周期（从上一轮历史高点到下一轮低点），不按年切分。
"""
from typing import Optional, Tuple

import numpy as np
import pandas as pd


def calc_max_drawdown(equity: pd.Series) -> float:
    """计算完整周期最大回撤

    定义：从上一轮历史高点到下一轮低点的完整周期最大跌幅。
    不按年度切分，反映整个回测期内的最坏情况。

    Args:
        equity: 资产净值序列（正数，index 为日期）

    Returns:
        最大回撤（负数小数，如 -0.12 表示 -12%）。若全期无回撤返回 0.0。
    """
    if equity.empty:
        return 0.0
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    return float(drawdown.min())


def calc_max_drawdown_period(equity: pd.Series) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp], int]:
    """计算最大回撤的起止日期和持续天数

    Returns:
        (peak_date, trough_date, duration_days)
        - peak_date: 回撤起点（前期高点日期）
        - trough_date: 回撤谷底日期
        - duration_days: 从高点到谷底的天数
    """
    if equity.empty:
        return None, None, 0
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    trough_date = drawdown.idxmin()
    # 在 trough_date 之前找最高点
    peak_date = equity.loc[:trough_date].idxmax()
    duration = (trough_date - peak_date).days if (peak_date and trough_date) else 0
    return peak_date, trough_date, duration


def calc_drawdown_series(equity: pd.Series) -> pd.Series:
    """计算回撤序列（每个时刻相对历史高点的回撤比例）"""
    if equity.empty:
        return pd.Series(dtype=float)
    running_max = equity.cummax()
    return (equity - running_max) / running_max


def calc_max_drawdown_duration(equity: pd.Series) -> int:
    """计算最大回撤持续天数（从高点到恢复到高点的天数）"""
    if equity.empty:
        return 0
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    # 最长连续回撤期
    in_dd = drawdown < 0
    if not in_dd.any():
        return 0
    # 找最长连续 True 段
    max_dur = 0
    cur_dur = 0
    for v in in_dd:
        if v:
            cur_dur += 1
            max_dur = max(max_dur, cur_dur)
        else:
            cur_dur = 0
    return int(max_dur)


def calc_sharpe(returns: pd.Series, periods_per_year: int = 252, rf: float = 0.03) -> Optional[float]:
    """计算 Sharpe Ratio（年化）

    Args:
        returns: 日收益率序列
        periods_per_year: 年化周期数（默认 252 交易日）
        rf: 无风险利率（年化，默认 3%）

    Returns:
        Sharpe Ratio，若收益方差为 0 返回 None
    """
    if returns.empty or returns.std() == 0:
        return None
    excess = returns - rf / periods_per_year
    return float(excess.mean() / excess.std() * np.sqrt(periods_per_year))


def calc_sortino(returns: pd.Series, periods_per_year: int = 252, rf: float = 0.03) -> Optional[float]:
    """计算 Sortino Ratio（仅用下行波动率）"""
    if returns.empty:
        return None
    excess = returns - rf / periods_per_year
    downside = excess[excess < 0]
    if downside.empty or downside.std() == 0:
        return None
    return float(excess.mean() / downside.std() * np.sqrt(periods_per_year))


def calc_volatility(returns: pd.Series, periods_per_year: int = 252) -> Optional[float]:
    """计算年化波动率"""
    if returns.empty or returns.std() == 0:
        return None
    return float(returns.std() * np.sqrt(periods_per_year))


def calc_rolling_sharpe(returns: pd.Series, window: int = 252,
                        periods_per_year: int = 252, rf: float = 0.03) -> pd.Series:
    """计算滚动 Sharpe Ratio

    Args:
        returns: 日收益率序列
        window: 滚动窗口（默认 252 个交易日 ≈ 1 年）
    """
    if returns.empty:
        return pd.Series(dtype=float)
    excess = returns - rf / periods_per_year
    rolling_mean = excess.rolling(window=window, min_periods=window // 2).mean()
    rolling_std = excess.rolling(window=window, min_periods=window // 2).std()
    return (rolling_mean / rolling_std) * np.sqrt(periods_per_year)


def calc_rolling_volatility(returns: pd.Series, window: int = 252,
                            periods_per_year: int = 252) -> pd.Series:
    """计算滚动年化波动率"""
    if returns.empty:
        return pd.Series(dtype=float)
    return returns.rolling(window=window, min_periods=window // 2).std() * np.sqrt(periods_per_year)


def calc_monthly_returns(returns: pd.Series) -> pd.Series:
    """计算月度收益率序列

    Returns:
        Series, index 为月末日期，value 为月收益率（小数）
    """
    if returns.empty:
        return pd.Series(dtype=float)
    equity = (1 + returns).cumprod()
    monthly = equity.resample('M').last()
    return monthly.pct_change().dropna()


def calc_yearly_returns(returns: pd.Series) -> pd.Series:
    """计算年度收益率序列"""
    if returns.empty:
        return pd.Series(dtype=float)
    equity = (1 + returns).cumprod()
    yearly = equity.resample('Y').last()
    return yearly.pct_change().dropna()


def calc_monthly_returns_heatmap(returns: pd.Series) -> pd.DataFrame:
    """计算月度收益率热力图矩阵

    Returns:
        DataFrame, index=年（int），columns=月（1-12），value=月收益率（小数）
    """
    monthly = calc_monthly_returns(returns)
    if monthly.empty:
        return pd.DataFrame()
    df = pd.DataFrame({
        'year': monthly.index.year,
        'month': monthly.index.month,
        'return': monthly.values,
    })
    return df.pivot_table(index='year', columns='month', values='return')


def calc_metrics_from_equity(equity: pd.Series, cash: float,
                             start_date, end_date,
                             periods_per_year: int = 252,
                             rf: float = 0.03) -> dict:
    """从资产净值序列计算全套指标

    Args:
        equity: 资产净值序列（每日）
        cash: 初始资金
        start_date: 起始日期
        end_date: 结束日期
        periods_per_year: 年化周期数
        rf: 无风险利率

    Returns:
        dict 包含 total_return / annual_return / max_drawdown / sharpe / sortino /
              volatility / calmar / max_dd_duration 等字段
    """
    if equity.empty:
        return {}

    final_value = float(equity.iloc[-1])
    total_return = (final_value - cash) / cash
    actual_days = (end_date - start_date).days if hasattr(end_date, '__sub__') else 0
    annual_return = (1 + total_return) ** (365 / actual_days) - 1 if actual_days > 0 else 0.0

    # 日收益率
    returns = equity.pct_change().dropna()

    # 回撤
    max_dd = calc_max_drawdown(equity)
    max_dd_duration = calc_max_drawdown_duration(equity)

    # 风险指标
    sharpe = calc_sharpe(returns, periods_per_year, rf)
    sortino = calc_sortino(returns, periods_per_year, rf)
    volatility = calc_volatility(returns, periods_per_year)
    calmar = annual_return / abs(max_dd) if max_dd < 0 else 0.0

    return {
        'final_value': final_value,
        'total_return': total_return,
        'annual_return': annual_return,
        'max_drawdown': max_dd,
        'max_dd_duration': max_dd_duration,
        'sharpe': sharpe,
        'sortino': sortino,
        'volatility': volatility,
        'calmar': calmar,
        'days': actual_days,
    }
