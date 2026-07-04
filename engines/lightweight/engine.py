"""
backtest_engine.py  —  固定回测引擎（轻量化，移植版）
---------------------------------------------------
  【稳定模块，不随策略改动】
  职责：接收标准化的价格数据 + 信号序列 → 输出回测结果
  核心特性：
    1. T日信号变化 → T+1日开盘价调仓（防止未来函数）
    2. 跳空滑点：prev_close 到 next_open 的缺口真实计入
    3. 冲击滑点：可调常数，模拟买卖价差/市场冲击
    4. 手续费：双边扣除
---------------------------------------------------
"""
from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Union

# ============================================================
#  数据结构 — 标准化输入/输出
# ============================================================

@dataclass
class BacktestInput:
    """回测输入 — 所有策略共用这一套标准格式。

    列名约定（与 DataFetcher 输出一致）：
        dates:          日期数组，'YYYY-MM-DD' 或 datetime
        value_open/close: 价值指数开/收
        growth_open/close: 成长指数开/收
        signal:         每日信号，'value' 或 'growth'

    注：signal 是 "T日收盘后应持有的仓位"，
        实际调仓发生在 T+1 日开盘价。
    """
    dates: np.ndarray
    value_open: np.ndarray
    value_close: np.ndarray
    growth_open: np.ndarray
    growth_close: np.ndarray
    signal: np.ndarray

    def __post_init__(self):
        n = len(self.dates)
        for name in ('value_open', 'value_close', 'growth_open', 'growth_close', 'signal'):
            if len(getattr(self, name)) != n:
                raise ValueError(f"{name} 长度与 dates 不一致")
        valid_signals = {'value', 'growth', 'cash'}
        if not set(pd.unique(self.signal)).issubset(valid_signals):
            raise ValueError(f"signal 只能含 'value'/'growth'/'cash'，有: {set(pd.unique(self.signal))}")

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> "BacktestInput":
        """从标准 DataFrame 构造"""
        required = ['date', 'value_open', 'value_close', 'growth_open', 'growth_close', 'signal']
        for c in required:
            if c not in df.columns:
                raise ValueError(f"缺少列: {c}")
        return cls(
            dates=df['date'].astype(str).values,
            value_open=df['value_open'].values.astype(np.float64),
            value_close=df['value_close'].values.astype(np.float64),
            growth_open=df['growth_open'].values.astype(np.float64),
            growth_close=df['growth_close'].values.astype(np.float64),
            signal=df['signal'].values.astype(str),
        )


@dataclass
class BacktestConfig:
    """回测参数配置 — 区分策略参数与交易成本参数"""
    start_cash: float = 1_000_000.0
    commission: float = 0.0001        # 手续费率（双边），默认 0.01%
    impact_slippage: float = 0.0       # 冲击滑点率（单边），默认0，可调
    apply_gap_slippage: bool = True    # 是否启用跳空滑点（prev_close → next_open 的真实缺口）

    def describe(self) -> str:
        return (f"手续费 {self.commission*100:.3f}% | "
                f"冲击滑点 {self.impact_slippage*100:.3f}% | "
                f"跳空滑点: {'启用' if self.apply_gap_slippage else '关闭'}")


@dataclass
class Trade:
    """单笔交易记录"""
    trade_date: str            # 实际执行日（T+1开盘）
    signal_date: str           # 信号触发日（T日收盘）
    position: str              # 'value' or 'growth'
    trade_price: float         # 实际成交价（已含冲击滑点）
    nominal_price: float       # 名义开盘价（无滑点）
    amount: float              # 交易金额
    commission_cost: float     # 手续费损失
    impact_cost: float         # 冲击滑点损失
    gap_cost: float            # 跳空滑点绝对值（|open - prev_close| × shares）
    gap_cost_unfavorable: float  # 不利跳空（低开损失，open < prev_close）
    gap_cost_favorable: float   # 有利跳空（高开收益，open > prev_close）
    total_slippage_cost: float # 本笔总摩擦成本


@dataclass
class BacktestResult:
    """回测输出 — 所有策略输出格式一致"""
    dates: np.ndarray
    nav: np.ndarray
    signal: np.ndarray
    position: np.ndarray           # 当日实际持仓（与 signal 可能差1天，因为调仓在下一日开盘）
    trades: List[Trade] = field(default_factory=list)
    daily_pnl: np.ndarray = None   # 每日盈亏
    daily_ret: np.ndarray = None   # 每日收益率
    config: BacktestConfig = None

    # 性能指标（延迟计算）
    _metrics: Optional[Dict[str, float]] = None

    # ============================================================
    # 指标计算 — 固定公式
    # ============================================================
    @property
    def metrics(self) -> Dict[str, float]:
        if self._metrics is None:
            self._metrics = self._calc_metrics()
        return self._metrics

    def _calc_metrics(self) -> Dict[str, float]:
        nav = self.nav
        start_cash = nav[0]
        n = len(nav)

        total_ret = nav[-1] / start_cash - 1.0

        # 年化（按 (n-1)/252 个交易年计算）
        if n < 2:
            annual_ret = 0.0
        else:
            annual_ret = float((nav[-1] / nav[0]) ** (252.0 / (n - 1)) - 1.0)

        # 日收益率（排除第0日的0收益）
        rets = nav[1:] / nav[:-1] - 1.0

        # 夏普（日收益年化，样本标准差 ddof=1）
        annual_vol = float(np.std(rets, ddof=1) * np.sqrt(252.0))
        sharpe = float(annual_ret / annual_vol) if annual_vol > 0 else 0.0

        # 最大回撤
        peak = np.maximum.accumulate(nav)
        dd = (nav - peak) / peak
        max_dd = float(np.min(dd))

        # Calmar
        calmar = -annual_ret / max_dd if max_dd < -1e-8 else 0.0

        # 胜率（日收益率正的比例）
        win_rate = float(np.mean(rets > 0)) if len(rets) > 0 else 0.0

        # 交易摩擦成本汇总
        total_commission = sum(t.commission_cost for t in self.trades)
        total_impact = sum(t.impact_cost for t in self.trades)
        total_gap = sum(t.gap_cost for t in self.trades)
        total_gap_unfavorable = sum(t.gap_cost_unfavorable for t in self.trades)
        total_gap_favorable = sum(t.gap_cost_favorable for t in self.trades)
        total_slippage = sum(t.total_slippage_cost for t in self.trades)

        return {
            'total_ret': total_ret,
            'annual_ret': annual_ret,
            'sharpe': sharpe,
            'max_dd': max_dd,
            'calmar': calmar,
            'win_rate': win_rate,
            'final_multiple': nav[-1] / start_cash,
            'final_nav': nav[-1],
            'start_cash': start_cash,
            'num_trades': len(self.trades),
            'num_days': n,
            'total_commission_cost': total_commission,
            'total_impact_cost': total_impact,
            'total_gap_cost': total_gap,
            'total_gap_cost_unfavorable': total_gap_unfavorable,
            'total_gap_cost_favorable': total_gap_favorable,
            'total_slippage_cost': total_slippage,
        }

    # ============================================================
    # 便捷输出
    # ============================================================
    def summary(self, label: str = "") -> str:
        m = self.metrics
        lines = [
            f"{'─'*70}",
            f"  {label}",
            f"{'─'*70}",
            f"  总收益:    {m['total_ret']*100:8.2f}%    年化: {m['annual_ret']*100:8.2f}%",
            f"  夏普:      {m['sharpe']:8.3f}    最大回撤: {m['max_dd']*100:8.2f}%",
            f"  Calmar:    {m['calmar']:8.3f}    胜率:   {m['win_rate']*100:8.1f}%",
            f"  交易次数:  {m['num_trades']:>5d}    最终净值: {m['final_nav']:>12,.0f}",
            f"  倍率:      {m['final_multiple']:>6.2f}x",
            f"{'─'*70}",
            f"  摩擦成本分析:",
            f"    手续费:   {m['total_commission_cost']:>12,.0f}  ({m['total_commission_cost']/m['start_cash']*100:.3f}% 初始资金)",
            f"    冲击滑点: {m['total_impact_cost']:>12,.0f}  ({m['total_impact_cost']/m['start_cash']*100:.3f}% 初始资金)",
            f"    跳空滑点: {m['total_gap_cost']:>12,.0f}  ({m['total_gap_cost']/m['start_cash']*100:.3f}% 初始资金)",
            f"    合计:     {m['total_slippage_cost']:>12,.0f}  ({m['total_slippage_cost']/m['start_cash']*100:.3f}% 初始资金)",
            f"{'─'*70}",
        ]
        return "\n".join(lines)

    def to_dataframe(self) -> pd.DataFrame:
        """输出逐日明细 — 方便进一步分析/绘图"""
        df = pd.DataFrame({
            'date': self.dates,
            'signal': self.signal,
            'position': self.position,
            'nav': self.nav,
            'daily_ret': self.daily_ret if self.daily_ret is not None else 0.0,
        })
        return df

    def trades_to_dataframe(self) -> pd.DataFrame:
        if not self.trades:
            return pd.DataFrame()
        rows = []
        for t in self.trades:
            rows.append({
                'trade_date': t.trade_date,
                'signal_date': t.signal_date,
                'position': t.position,
                'trade_price': t.trade_price,
                'nominal_price': t.nominal_price,
                'amount': t.amount,
                'commission_cost': t.commission_cost,
                'impact_cost': t.impact_cost,
                'gap_cost': t.gap_cost,
                'total_slippage_cost': t.total_slippage_cost,
            })
        return pd.DataFrame(rows)


# ============================================================
#  核心回测引擎 — 这是固定模块的心脏
# ============================================================

def run_backtest_engine(
    bt_input: BacktestInput,
    config: BacktestConfig = None,
) -> BacktestResult:
    """
    标准回测引擎 —【稳定，不随策略改动】

    交易规则（严谨定义）:
        1. signal[i] 代表 "第i日收盘后决定的仓位"
        2. 若 signal[i] != signal[i-1] → 触发调仓，调仓在第i+1日开盘价执行
        3. 第0日：以 signal[0] 决定的方向在第0日开盘价建仓（首仓特例）
        4. 每日净值 = 持仓数量 × 当日收盘价
        5. 滑点模型：
             - 手续费：双边 commission
             - 冲击滑点：buy_price = open * (1 + impact)，sell_price = open * (1 - impact)
             - 跳空滑点（真实）：卖出时用 open[T+1]，而非 close[T]，
               跳空缺口 = |open[T+1] - close[T]| × shares，真实计入交易结果
    """
    if config is None:
        config = BacktestConfig()

    dates = bt_input.dates
    vo = bt_input.value_open.astype(np.float64)
    vc = bt_input.value_close.astype(np.float64)
    go = bt_input.growth_open.astype(np.float64)
    gc = bt_input.growth_close.astype(np.float64)
    sig = bt_input.signal

    n = len(dates)
    nav = np.zeros(n, dtype=np.float64)
    position_arr = np.array([''] * n, dtype=object)

    cash = float(config.start_cash)
    shares = 0.0
    current_pos = None
    trades: List[Trade] = []

    def price_open(pos: str, i: int) -> float:
        return vo[i] if pos == 'value' else go[i]

    def price_close(pos: str, i: int) -> float:
        return vc[i] if pos == 'value' else gc[i]

    pending_rebalance = False
    rebalance_pos = None

    for i in range(n):
        if i == 0:
            pos = sig[0]

            # 处理空仓信号
            if pos == 'cash':
                current_pos = 'cash'
                shares = 0.0
                position_arr[i] = 'cash'
                nav[i] = cash  # 空仓时净值等于现金
                continue

            nom_price = price_open(pos, 0)
            trade_price = nom_price * (1.0 + config.impact_slippage)
            amount_pre = cash
            commission = amount_pre * config.commission
            shares = (amount_pre - commission) / trade_price
            impact_cost = amount_pre - shares * nom_price
            cash = 0.0
            current_pos = pos
            position_arr[i] = pos
            nav[i] = shares * price_close(pos, i)
            trades.append(Trade(
                trade_date=str(dates[i]),
                signal_date=str(dates[i]),
                position=pos,
                trade_price=trade_price,
                nominal_price=nom_price,
                amount=amount_pre,
                commission_cost=commission,
                impact_cost=impact_cost,
                gap_cost=0.0,
                gap_cost_unfavorable=0.0,
                gap_cost_favorable=0.0,
                total_slippage_cost=commission + impact_cost,
            ))
            continue

        # ===== 第i日（i>0）核心逻辑 =====

        # 1. 先执行调仓（如果有待执行的调仓）
        #    这是在第i日开盘价执行的调仓，对应第i-1日收盘后的信号变化
        if pending_rebalance:
            old_pos = current_pos

            # 如果当前有持仓，先卖出
            if old_pos != 'cash':
                old_nom_price = price_open(old_pos, i)
                old_sell_price = old_nom_price * (1.0 - config.impact_slippage)

                sell_amount = shares * old_sell_price
                commission_sell = sell_amount * config.commission
                cash_after_sell = sell_amount - commission_sell

                prev_close_price = price_close(old_pos, i - 1)
                gap_diff = (old_nom_price - prev_close_price) * shares
                gap_cost = abs(gap_diff)
                # 区分有利/不利跳空
                if gap_diff < 0:
                    gap_cost_unfavorable = abs(gap_diff)  # 低开：损失
                    gap_cost_favorable = 0.0
                else:
                    gap_cost_unfavorable = 0.0
                    gap_cost_favorable = gap_diff  # 高开：收益

                impact_cost_sell = (old_nom_price - old_sell_price) * shares
            else:
                # 当前是空仓，没有卖出操作
                cash_after_sell = cash
                commission_sell = 0.0
                gap_cost = 0.0
                gap_cost_unfavorable = 0.0
                gap_cost_favorable = 0.0
                impact_cost_sell = 0.0

            new_pos = rebalance_pos

            # 如果新目标不是空仓，买入
            if new_pos != 'cash':
                new_nom_price = price_open(new_pos, i)
                new_trade_price = new_nom_price * (1.0 + config.impact_slippage)
                commission_buy = cash_after_sell * config.commission
                shares_new = (cash_after_sell - commission_buy) / new_trade_price
                impact_cost_buy = (new_trade_price - new_nom_price) * shares_new

                shares = shares_new
                cash = 0.0
                current_pos = new_pos

                trades.append(Trade(
                    trade_date=str(dates[i]),
                    signal_date=str(dates[i - 1]),
                    position=new_pos,
                    trade_price=new_trade_price,
                    nominal_price=new_nom_price,
                    amount=cash_after_sell,
                    commission_cost=commission_sell + commission_buy,
                    impact_cost=impact_cost_sell + impact_cost_buy,
                    gap_cost=gap_cost,
                    gap_cost_unfavorable=gap_cost_unfavorable,
                    gap_cost_favorable=gap_cost_favorable,
                    total_slippage_cost=commission_sell + commission_buy + impact_cost_sell + impact_cost_buy,
                ))
            else:
                # 新目标是空仓，保持现金
                cash = cash_after_sell
                shares = 0.0
                current_pos = 'cash'

                if old_pos != 'cash':
                    # 记录卖出交易
                    trades.append(Trade(
                        trade_date=str(dates[i]),
                        signal_date=str(dates[i - 1]),
                        position='cash',
                        trade_price=old_nom_price,
                        nominal_price=old_nom_price,
                        amount=sell_amount,
                        commission_cost=commission_sell,
                        impact_cost=impact_cost_sell,
                        gap_cost=gap_cost,
                        gap_cost_unfavorable=gap_cost_unfavorable,
                        gap_cost_favorable=gap_cost_favorable,
                        total_slippage_cost=commission_sell + impact_cost_sell,
                    ))

            pending_rebalance = False
            rebalance_pos = None

        # 2. 计算当日净值（基于调仓后的持仓 × 当日收盘价）
        pos_today = current_pos
        if pos_today == 'cash':
            nav[i] = cash  # 空仓时净值等于现金
        else:
            nav[i] = shares * price_close(pos_today, i)
        position_arr[i] = pos_today

        # 3. 检查是否需要在第i+1日调仓
        #    signal[i] 是第i日收盘后决定的仓位
        #    如果 signal[i] != signal[i-1]，说明信号变了，需要在第i+1日调仓
        if sig[i] != sig[i - 1]:
            pending_rebalance = True
            rebalance_pos = sig[i]

    daily_ret = np.zeros(n)
    daily_ret[1:] = nav[1:] / nav[:-1] - 1.0

    return BacktestResult(
        dates=dates,
        nav=nav,
        signal=sig,
        position=position_arr,
        trades=trades,
        daily_ret=daily_ret,
        config=config,
    )


# ============================================================
#  带仓位权重的回测引擎 — 支持部分仓位（如10%降仓）
# ============================================================

def run_backtest_engine_weighted(
    bt_input: BacktestInput,
    config: BacktestConfig = None,
    position_weight: Optional[np.ndarray] = None,
) -> BacktestResult:
    """
    带仓位权重的回测引擎 — 在 run_backtest_engine 基础上增加 position_weight 维度

    position_weight[i] 代表第i日收盘后决定的仓位权重（0.0~1.0）
      - None 或全 1.0 时等价于 run_backtest_engine
      - 0.1 表示10%仓位（90%现金），用于 MA75 择时降仓场景

    调仓触发条件（满足任一）：
      1. signal[i] != signal[i-1]            （方向变化）
      2. |position_weight[i] - position_weight[i-1]| > 1e-10  （权重变化）

    交易规则与 run_backtest_engine 完全一致：
      - T日信号变化 → T+1日开盘价调仓
      - 跳空滑点、冲击滑点、手续费（双边）
      - 每日净值 = 持仓市值 + 现金
    """
    if config is None:
        config = BacktestConfig()

    if position_weight is None:
        return run_backtest_engine(bt_input, config)

    position_weight = np.asarray(position_weight, dtype=np.float64)
    if len(position_weight) != len(bt_input.dates):
        raise ValueError(
            f"position_weight 长度({len(position_weight)})与 dates 长度({len(bt_input.dates)})不一致"
        )

    dates = bt_input.dates
    vo = bt_input.value_open.astype(np.float64)
    vc = bt_input.value_close.astype(np.float64)
    go = bt_input.growth_open.astype(np.float64)
    gc = bt_input.growth_close.astype(np.float64)
    sig = bt_input.signal
    w = position_weight

    n = len(dates)
    nav = np.zeros(n, dtype=np.float64)
    position_arr = np.array([''] * n, dtype=object)

    cash = float(config.start_cash)
    shares = 0.0
    current_pos = None
    current_weight = 1.0
    trades: List[Trade] = []

    def price_open(pos: str, i: int) -> float:
        return vo[i] if pos == 'value' else go[i]

    def price_close(pos: str, i: int) -> float:
        return vc[i] if pos == 'value' else gc[i]

    pending_rebalance = False
    rebalance_pos = None
    rebalance_weight = 1.0

    for i in range(n):
        if i == 0:
            pos = sig[0]
            w_i = float(w[0])

            if pos == 'cash' or w_i <= 0.0:
                current_pos = 'cash'
                current_weight = 0.0
                shares = 0.0
                cash = float(config.start_cash)
                position_arr[i] = 'cash'
                nav[i] = cash
                continue

            nom_price = price_open(pos, 0)
            trade_price = nom_price * (1.0 + config.impact_slippage)
            invest_amount = float(config.start_cash) * w_i
            commission = invest_amount * config.commission
            shares = (invest_amount - commission) / trade_price
            impact_cost = invest_amount - shares * nom_price
            cash = float(config.start_cash) - invest_amount
            current_pos = pos
            current_weight = w_i
            position_arr[i] = pos
            nav[i] = shares * price_close(pos, i) + cash
            trades.append(Trade(
                trade_date=str(dates[i]),
                signal_date=str(dates[i]),
                position=pos,
                trade_price=trade_price,
                nominal_price=nom_price,
                amount=invest_amount,
                commission_cost=commission,
                impact_cost=impact_cost,
                gap_cost=0.0,
                gap_cost_unfavorable=0.0,
                gap_cost_favorable=0.0,
                total_slippage_cost=commission + impact_cost,
            ))
            continue

        # ===== 第i日（i>0）核心逻辑 =====

        # 1. 执行待处理的调仓（T+1开盘价）
        if pending_rebalance:
            old_pos = current_pos
            old_nom_price = None
            sell_amount = 0.0

            if old_pos is not None and old_pos != 'cash' and shares > 0:
                old_nom_price = price_open(old_pos, i)
                old_sell_price = old_nom_price * (1.0 - config.impact_slippage)
                sell_amount = shares * old_sell_price
                commission_sell = sell_amount * config.commission
                cash = cash + sell_amount - commission_sell

                prev_close_price = price_close(old_pos, i - 1)
                gap_diff = (old_nom_price - prev_close_price) * shares
                gap_cost = abs(gap_diff)
                if gap_diff < 0:
                    gap_cost_unfavorable = abs(gap_diff)
                    gap_cost_favorable = 0.0
                else:
                    gap_cost_unfavorable = 0.0
                    gap_cost_favorable = gap_diff

                impact_cost_sell = (old_nom_price - old_sell_price) * shares
            else:
                commission_sell = 0.0
                gap_cost = 0.0
                gap_cost_unfavorable = 0.0
                gap_cost_favorable = 0.0
                impact_cost_sell = 0.0

            new_pos = rebalance_pos
            new_weight = rebalance_weight

            if new_pos != 'cash' and new_weight > 0:
                new_nom_price = price_open(new_pos, i)
                new_trade_price = new_nom_price * (1.0 + config.impact_slippage)
                invest_amount = cash * new_weight
                commission_buy = invest_amount * config.commission
                shares_new = (invest_amount - commission_buy) / new_trade_price
                impact_cost_buy = (new_trade_price - new_nom_price) * shares_new
                cash = cash - invest_amount
                shares = shares_new
                current_pos = new_pos
                current_weight = new_weight

                trades.append(Trade(
                    trade_date=str(dates[i]),
                    signal_date=str(dates[i - 1]),
                    position=new_pos,
                    trade_price=new_trade_price,
                    nominal_price=new_nom_price,
                    amount=invest_amount,
                    commission_cost=commission_sell + commission_buy,
                    impact_cost=impact_cost_sell + impact_cost_buy,
                    gap_cost=gap_cost,
                    gap_cost_unfavorable=gap_cost_unfavorable,
                    gap_cost_favorable=gap_cost_favorable,
                    total_slippage_cost=commission_sell + commission_buy + impact_cost_sell + impact_cost_buy,
                ))
            else:
                shares = 0.0
                current_pos = 'cash'
                current_weight = 0.0
                if old_pos is not None and old_pos != 'cash':
                    trades.append(Trade(
                        trade_date=str(dates[i]),
                        signal_date=str(dates[i - 1]),
                        position='cash',
                        trade_price=old_nom_price,
                        nominal_price=old_nom_price,
                        amount=sell_amount,
                        commission_cost=commission_sell,
                        impact_cost=impact_cost_sell,
                        gap_cost=gap_cost,
                        gap_cost_unfavorable=gap_cost_unfavorable,
                        gap_cost_favorable=gap_cost_favorable,
                        total_slippage_cost=commission_sell + impact_cost_sell,
                    ))

            pending_rebalance = False
            rebalance_pos = None
            rebalance_weight = 1.0

        # 2. 计算当日净值
        pos_today = current_pos
        if pos_today == 'cash' or pos_today is None:
            nav[i] = cash
        else:
            nav[i] = shares * price_close(pos_today, i) + cash
        position_arr[i] = pos_today if pos_today is not None else 'cash'

        # 3. 检查是否需要调仓（方向或权重变化）
        sig_changed = sig[i] != sig[i - 1]
        weight_changed = abs(w[i] - w[i - 1]) > 1e-10
        if sig_changed or weight_changed:
            pending_rebalance = True
            rebalance_pos = sig[i]
            rebalance_weight = float(w[i])

    daily_ret = np.zeros(n)
    daily_ret[1:] = nav[1:] / nav[:-1] - 1.0

    return BacktestResult(
        dates=dates,
        nav=nav,
        signal=sig,
        position=position_arr,
        trades=trades,
        daily_ret=daily_ret,
        config=config,
    )


# ============================================================
#  滑点敏感性测试 — 固定模块，方便比较不同成本下的表现
# ============================================================

def run_slippage_sensitivity(
    bt_input: BacktestInput,
    impact_slippage_grid: List[float] = None,
    commission_grid: List[float] = None,
) -> pd.DataFrame:
    """
    测试不同滑点/手续费假设下的策略表现。
    """
    if impact_slippage_grid is None:
        impact_slippage_grid = [0.0, 0.001, 0.002, 0.003, 0.005]
    if commission_grid is None:
        commission_grid = [0.0001]

    rows = []
    for comm in commission_grid:
        for sl in impact_slippage_grid:
            cfg = BacktestConfig(
                commission=comm,
                impact_slippage=sl,
                apply_gap_slippage=True,
            )
            res = run_backtest_engine(bt_input, cfg)
            m = res.metrics
            rows.append({
                'commission_%': comm * 100,
                'impact_slippage_%': sl * 100,
                'annual_ret_%': m['annual_ret'] * 100,
                'total_ret_%': m['total_ret'] * 100,
                'sharpe': m['sharpe'],
                'max_dd_%': m['max_dd'] * 100,
                'calmar': m['calmar'],
                'num_trades': m['num_trades'],
                'final_multiple': m['final_multiple'],
            })
    return pd.DataFrame(rows)


# ============================================================
#  与旧版兼容的薄封装（方便逐步迁移）
# ============================================================

def run_backtest_compat(
    signal_arr, value_open, value_close, growth_open, growth_close,
    dates, commission=0.0001, start_cash=1_000_000,
):
    """
    与旧 realrate_factor.run_backtest 接口一致的薄封装。
    行为差异：
      - 调仓在 T+1 日开盘（旧版在 T 日开盘，存在未来函数风险）
      - 默认启用跳空滑点 + 手续费
    """
    bt_in = BacktestInput(
        dates=np.array(dates),
        value_open=np.asarray(value_open, dtype=np.float64),
        value_close=np.asarray(value_close, dtype=np.float64),
        growth_open=np.asarray(growth_open, dtype=np.float64),
        growth_close=np.asarray(growth_close, dtype=np.float64),
        signal=np.asarray(signal_arr, dtype=str),
    )
    cfg = BacktestConfig(
        start_cash=start_cash,
        commission=commission,
        impact_slippage=0.0,
        apply_gap_slippage=True,
    )
    result = run_backtest_engine(bt_in, cfg)
    # 按旧格式返回，方便迁移时对比
    return {
        'nav': result.nav,
        'trades': [(t.trade_date, t.position, t.trade_price, t.amount) for t in result.trades],
        'final_nav': result.nav[-1],
        'total_ret': result.metrics['total_ret'],
        'num_trades': len(result.trades),
        'start_cash': start_cash,
    }


def calc_metrics_compat(nav_arr, dates_arr, label=""):
    """与旧版 calc_metrics 兼容"""
    res = BacktestResult(
        dates=np.asarray(dates_arr),
        nav=np.asarray(nav_arr, dtype=np.float64),
        signal=np.array(['growth'] * len(nav_arr)),
        position=np.array(['growth'] * len(nav_arr)),
    )
    return res.metrics


# ============================================================
#  自测：作为独立脚本运行时验证
# ============================================================

if __name__ == '__main__':
    # 生成简单模拟数据进行自测
    np.random.seed(42)
    n = 500
    dates = pd.date_range('2020-01-01', periods=n, freq='B').strftime('%Y-%m-%d').values
    value_close = 100 * np.exp(np.cumsum(np.random.normal(0.0005, 0.01, n)))
    growth_close = 100 * np.exp(np.cumsum(np.random.normal(0.0008, 0.015, n)))
    value_open = value_close * (1 + np.random.uniform(-0.005, 0.005, n))
    growth_open = growth_close * (1 + np.random.uniform(-0.005, 0.005, n))
    signal = np.array(['growth'] * n)

    bt_in = BacktestInput(
        dates=dates,
        value_open=value_open,
        value_close=value_close,
        growth_open=growth_open,
        growth_close=growth_close,
        signal=signal,
    )
    cfg = BacktestConfig(commission=0.0001, impact_slippage=0.0, apply_gap_slippage=True)
    res = run_backtest_engine(bt_in, cfg)
    print(res.summary("[自测] 纯成长持有（模拟数据）"))
    print()

    sens = run_slippage_sensitivity(bt_in)
    print("[自测] 滑点敏感性：")
    print(sens.to_string(index=False))
