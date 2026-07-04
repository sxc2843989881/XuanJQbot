"""
bt_engine.py  —  bt 事件驱动回测引擎（通用模块）
---------------------------------------------------
  【稳定模块，不随策略改动】
  职责：接收标准化的价格数据 + 持仓权重序列 → 输出回测结果
  特性：
    1. 事件驱动，逐 Bar 处理，支持多资产轮动
    2. 接收原始信号，内部自动处理 T+1 执行延迟
    3. 佣金 + 冲击滑点 + 跳空滑点追踪
    4. 输出包含各项指标 + 每日持仓 + 调仓记录
---------------------------------------------------
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Callable


# ============================================================
#  数据结构
# ============================================================

@dataclass
class BTBacktestInput:
    """bt 引擎输入 — 所有策略共用。

    Args:
        price_df: 各资产收盘价，列名=资产名
        weight_df: 每日持仓权重（同列名），范围[0,1]，引擎内部处理 T+1 延迟
        initial_capital: 初始资金
    """
    price_df: pd.DataFrame
    weight_df: pd.DataFrame
    initial_capital: float = 1_000_000


@dataclass
class BTBacktestTrade:
    """单笔调仓记录。"""
    trade_date: str           # 调仓执行日期
    signal_date: str          # 信号产生日期（前一日）
    action: str               # 'enter_growth' / 'enter_value' / 'exit'
    weight: float             # 目标权重
    cost: float               # 佣金+冲击成本


@dataclass
class BTBacktestResult:
    """bt 引擎输出 — 标准化回测结果。"""
    nav: np.ndarray                # 每日净值
    dates: np.ndarray              # 日期
    position: np.ndarray           # 每日持仓（'growth'/'value'/'cash'）
    trades: List[BTBacktestTrade]  # 调仓记录
    total_return: float            # 总收益率
    cagr: float                    # 年化收益率（基于交易日数）
    max_drawdown: float            # 最大回撤
    sharpe: float                  # 夏普比
    calmar: float                  # 卡尔玛比
    n_days: int                    # 交易日数
    cost_summary: Dict = field(default_factory=dict)


# ============================================================
#  自定义 Algo — 遵循 bt 设计模式
# ============================================================

class WeighFromSignal:
    """从 additional_data 中读取信号权重，注入 target.temp['weights']。

    遵循 bt 最佳实践：通过数据名称引用，而非直接存储 DataFrame。
    使用方式：
        bt.Backtest(s, data, additional_data={'my_signal': weight_df})
    """
    def __init__(self, signal_name: str = 'weights'):
        self.signal_name = signal_name

    def __call__(self, target):
        signal = target.get_data(self.signal_name)
        if signal is None:
            return True
        now = pd.Timestamp(target.now).normalize()
        if now not in signal.index:
            return True
        w = signal.loc[now]
        if isinstance(w, pd.Series):
            w = w.dropna()
        if len(w) > 0:
            target.temp['weights'] = w
        return True


# ============================================================
#  辅助函数
# ============================================================

def apply_t1_delay(weight_df: pd.DataFrame) -> pd.DataFrame:
    """对 weight_df 施加 T+1 延迟：shift(1) + 首日用第一天信号填充。

    轻量化引擎约定：T 日收盘产生信号 → T+1 日开盘执行。
    bt 事件驱动引擎需同样的约定才能保持持仓一致。
    本函数实现了这一转换。

    Args:
        weight_df: 原始权重（T 日信号决定 T+1 日起的持仓）

    Returns:
        处理后的权重（第 T 行的值决定 T 日的持仓，已延迟一天）
    """
    df = weight_df.copy()
    df = df.shift(1)
    # 首日：用第一个信号填充（跳过 NaN）
    if len(df) > 0:
        df.iloc[0] = weight_df.iloc[0].values if len(weight_df) > 0 else 0.0
    return df


def compute_gap_costs(
    price_df: pd.DataFrame,
    raw_weight_df: pd.DataFrame,
) -> Tuple[float, int]:
    """从外部计算跳空滑点总成本（不涉及 bt 内部结构）。

    跳空 = 调仓时，卖出资产从 close_prev 到 open 的缺口。
    由于 open=close，缺口 = |close/close_prev - 1|。

    Args:
        price_df: 收盘价
        raw_weight_df: 原始权重（T+1 延迟前）

    Returns:
        (总 gap 比例, 调仓次数)
    """
    # 推导实际持仓序列
    pos = pd.Series(index=price_df.index, dtype=str)
    for i, d in enumerate(price_df.index):
        w = raw_weight_df.loc[d]
        # 取权重最大的资产作为持仓
        max_asset = w.idxmax() if w.max() > 0 else 'cash'
        pos.iloc[i] = max_asset

    gap_log = []
    prev_pos = None
    for i, d in enumerate(price_df.index):
        curr = pos.iloc[i]
        if prev_pos is not None and curr != prev_pos and prev_pos != 'cash':
            # T+1 延迟意味着调仓执行日 = i
            # 被卖出的资产 = prev_pos
            # 跳空 = |close[i]/close[i-1] - 1|
            if i > 0 and prev_pos in price_df.columns:
                gap = abs(price_df[prev_pos].iloc[i] / price_df[prev_pos].iloc[i-1] - 1)
                gap_log.append(gap)
        prev_pos = curr

    total_gap = sum(gap_log)
    n_switches = len(gap_log)
    return total_gap, n_switches


def calc_metrics_from_nav(
    nav: np.ndarray,
    n_days: int,
) -> Dict[str, float]:
    """从净值序列计算各项指标。

    Args:
        nav: 净值数组
        n_days: 交易日数

    Returns:
        {ann, dd, sharpe, calmar}
    """
    total_ret = nav[-1] / nav[0] - 1
    cagr = (1 + total_ret) ** (252 / n_days) - 1 if n_days > 0 else 0.0

    eq = pd.Series(nav / nav[0])
    dd = (eq / eq.cummax() - 1).min()
    dret = eq.pct_change().fillna(0)
    rf_daily = 0.025 / 252
    sharpe = np.sqrt(252) * (dret.mean() - rf_daily) / dret.std() if dret.std() > 0 else 0
    calmar = cagr / abs(dd) if dd < 0 else 0

    return {'ann': cagr, 'dd': dd, 'sharpe': sharpe, 'calmar': calmar}


# ============================================================
#  核心运行函数
# ============================================================

def run_bt_backtest(
    bt_input: BTBacktestInput,
    commission_rate: float = 0.0001,
    impact_rate: float = 0.0005,
    t1_delay: bool = True,
) -> BTBacktestResult:
    """运行 bt 事件驱动回测。

    Args:
        bt_input: 标准输入（价格 + 权重）
        commission_rate: 佣金比例（默认 0.01%）
        impact_rate: 冲击滑点比例（默认 0.05%）
        t1_delay: 是否施加 T+1 执行延迟（默认 True，与轻量化引擎一致）

    Returns:
        BTBacktestResult: 回测结果
    """
    import bt

    price_df = bt_input.price_df
    raw_weight = bt_input.weight_df

    # ─── 输入校验 ───
    if not isinstance(price_df, pd.DataFrame) or not isinstance(raw_weight, pd.DataFrame):
        raise ValueError("price_df 和 weight_df 必须为 pd.DataFrame")
    if price_df.shape[1] != raw_weight.shape[1]:
        raise ValueError(f"price_df 列数({price_df.shape[1]})与 weight_df 列数({raw_weight.shape[1]})不一致")
    if not set(price_df.columns) == set(raw_weight.columns):
        raise ValueError(f"price_df 列名{list(price_df.columns)}与 weight_df 列名{list(raw_weight.columns)}不一致")
    if price_df.isnull().any().any():
        raise ValueError("price_df 包含 NaN")

    # ─── 自动处理 T+1 延迟 ───
    weight_df = apply_t1_delay(raw_weight) if t1_delay else raw_weight.copy()

    # ─── 预计算跳空成本（外部，不依赖 bt 内部结构） ───
    total_gap_rate, gap_count = compute_gap_costs(price_df, raw_weight)

    # ─── 构建 bt 策略链 ───
    strategy = bt.Strategy('strategy', [
        bt.algos.RunDaily(),
        bt.algos.SelectAll(),
        WeighFromSignal(signal_name='bt_weights'),
        bt.algos.Rebalance(),
    ])

    # ─── 运行回测（通过 additional_data 传递权重） ───
    total_rate = commission_rate + impact_rate
    res = bt.run(bt.Backtest(
        strategy, price_df,
        initial_capital=bt_input.initial_capital,
        commissions=lambda q, p: abs(q) * p * total_rate,
        additional_data={'bt_weights': weight_df},
    ))

    # ─── 提取净值（按日期对齐） ───
    idx_dt = pd.DatetimeIndex(price_df.index)
    bt_nav = res.prices['strategy']
    aligned = bt_nav[bt_nav.index.isin(idx_dt)]
    nav = aligned.values.astype(np.float64)
    n_days = len(nav)
    dates = np.array(aligned.index.strftime('%Y-%m-%d'), dtype=str)

    # ─── 提取每日持仓 ───
    # 从 weight_df（已延迟）反推每日实际持仓
    position = np.array(['cash'] * n_days, dtype=object)
    for i, d in enumerate(aligned.index):
        if d in weight_df.index:
            w = weight_df.loc[d]
            max_asset = w.idxmax() if w.max() > 0 else 'cash'
            if w.max() > 0:
                position[i] = max_asset

    # ─── 提取调仓记录 ───
    # 从原始信号/权重推导调仓次数（与轻量化引擎一致）
    raw_pos = pd.Series(index=price_df.index, dtype=str)
    for d in price_df.index:
        w = raw_weight.loc[d]
        raw_pos.loc[d] = w.idxmax() if w.max() > 0 else 'cash'

    n_trades_total = 1  # 首日建仓
    for i in range(1, len(price_df.index)):
        sig_changed = raw_pos.iloc[i] != raw_pos.iloc[i-1]
        wt_changed = abs(raw_weight.iloc[i, :].values - raw_weight.iloc[i-1, :].values).max() > 1e-10
        if sig_changed or wt_changed:
            n_trades_total += 1

    # 调仓记录（仅信号切换，供查看）
    trades = []
    t1_shift = 1 if t1_delay else 0
    for i, d in enumerate(price_df.index):
        if i == 0: continue
        curr_sig = raw_pos.iloc[i]
        prev_sig = raw_pos.iloc[i-1]
        if curr_sig != prev_sig and curr_sig != 'cash':
            exec_idx = i + t1_shift
            if exec_idx < len(aligned.index):
                exec_d = aligned.index[exec_idx]
                t = BTBacktestTrade(
                    trade_date=exec_d.strftime('%Y-%m-%d'),
                    signal_date=d.strftime('%Y-%m-%d'),
                    action=f'enter_{curr_sig}',
                    weight=float(raw_weight.loc[d, curr_sig]),
                    cost=total_rate,
                )
                trades.append(t)

    # ─── 计算指标 ───
    metrics = calc_metrics_from_nav(nav, n_days)

    return BTBacktestResult(
        nav=nav,
        dates=dates,
        position=position,
        trades=trades,
        total_return=nav[-1] / nav[0] - 1,
        cagr=metrics['ann'],
        max_drawdown=metrics['dd'],
        sharpe=metrics['sharpe'],
        calmar=metrics['calmar'],
        n_days=n_days,
        cost_summary={
            'gap_count': gap_count,
            'total_gap_rate': round(total_gap_rate, 6),
            'trade_count': n_trades_total,
        },
    )


def calc_metrics(result: BTBacktestResult) -> Dict[str, float]:
    """提取指标字典（与轻量化引擎 calc_metrics 对齐）。"""
    return {
        'ann': result.cagr,
        'dd': result.max_drawdown,
        'sharpe': result.sharpe,
        'calmar': result.calmar,
        'n_days': result.n_days,
        'n_trades': result.cost_summary.get('trade_count', 0),
    }
