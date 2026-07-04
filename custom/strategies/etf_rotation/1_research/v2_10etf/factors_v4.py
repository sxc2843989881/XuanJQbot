"""ETF轮动策略 v4：上涨区间识别 + 目标收益止盈

基于用户思路 + A/B 双角色共识：
1. 日频扫描所有ETF，识别"上涨区间"（价格>MA20 + MA20>MA60 + 动量>0）
2. 识别"上涨初期"（最近5日首次突破三重确认）
3. 触发式调仓（3条件：止损/止盈/动量衰减）
4. 目标收益10%条件止盈 + 25%硬上限 + 动量跌出Top4换仓
5. 候选排序 = 0.6*动量分位 + 0.4*突破新鲜度
6. 止盈2日确认/止损即时执行

知识库依据：
- 03_技术指标库：双均线趋势判断
- 01_策略基础原理.md：动量效应、让利润奔跑
- 04_风险控制机制.md：止损机制
"""
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
from factors_v2 import calculate_composite_score, calculate_regression_momentum


def identify_uptrend(close: pd.DataFrame, ma_short=20, ma_long=60,
                     momentum_window=25) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """识别上涨区间和上涨初期

    上涨区间 = 价格>MA20 + MA20>MA60 + 动量>0
    上涨初期 = 最近5日内首次满足上涨区间条件

    Returns:
        uptrend: DataFrame[bool] 上涨区间标记
        fresh_breakthrough: DataFrame[bool] 首次突破日
        early_stage: DataFrame[bool] 上涨初期（最近5日有首次突破）
    """
    ma_s = close.rolling(ma_short).mean()
    ma_l = close.rolling(ma_long).mean()
    momentum = close.pct_change(periods=momentum_window)

    # 上涨区间
    uptrend = (close > ma_s) & (ma_s > ma_l) & (momentum > 0)

    # 首次突破：前一天不满足、今天满足
    fresh_breakthrough = uptrend & ~uptrend.shift(1).fillna(False)

    # 上涨初期：最近5日有首次突破
    early_stage = fresh_breakthrough.rolling(5).sum() > 0

    return uptrend, fresh_breakthrough, early_stage


def calculate_candidate_score(close: pd.DataFrame, uptrend: pd.DataFrame,
                              early_stage: pd.DataFrame, momentum_window=25) -> pd.DataFrame:
    """计算候选得分 = 0.6*动量分位 + 0.4*突破新鲜度

    只对处于上涨区间的ETF计算得分，非上涨区间ETF得分为0
    """
    momentum = close.pct_change(periods=momentum_window)

    # 动量分位排名（横截面，每日）
    mom_rank = momentum.rank(axis=1, pct=True)  # 0-1

    # 突破新鲜度：最近5日内首次突破的天数权重（越近越高）
    fresh_weight = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    for lag in range(5):
        weight = (5 - lag) / 15.0  # 5/15, 4/15, 3/15, 2/15, 1/15
        # 检查lag天前是否首次突破
        shifted = early_stage.shift(lag).fillna(False)
        fresh_weight = fresh_weight + shifted.astype(float) * weight

    # 复合得分
    score = 0.6 * mom_rank + 0.4 * fresh_weight

    # 只保留上涨区间的ETF
    score = score.where(uptrend, 0)

    return score


def generate_signal_v4(
    close: pd.DataFrame,
    uptrend: pd.DataFrame,
    early_stage: pd.DataFrame,
    candidate_score: pd.DataFrame,
    momentum: pd.DataFrame,
    hold_count: int = 4,
    take_profit_threshold: float = 0.10,
    hard_take_profit: float = 0.25,
    momentum_rank_threshold: int = 6,
    momentum_decay_confirm_days: int = 3,
    take_profit_confirm_days: int = 2,
    cost_stop_loss: float = -0.12,
    trailing_stop_loss: float = -0.10,
    use_defense: bool = True,
    defensive_etfs: Optional[List[str]] = None,
    drawdown_threshold: float = -0.15,
    rebalance_weekly: bool = True,
    cost: float = 0.0015,
) -> Tuple[pd.DataFrame, dict]:
    """v4.1信号生成：日频扫描 + 触发式调仓（修正版）

    v4.1修正（基于v4实证换手485次过高的教训）：
    - 去掉"跌出上涨区间"止损条件（过于敏感）
    - 动量衰减从Top4放宽到Top6 + 连续3日确认
    - 补仓改为周频（rebalance_weekly=True）

    3种触发条件：
    1. 止损：成本止损-12% 或 跟踪止损-10%（即时执行）
    2. 止盈：持仓收益>=10% 且 有上涨初期候选（2日确认）
    3. 动量衰减：持仓ETF动量跌出Top6 + 连续3日确认
    """
    # 默认防御ETF：波动率最低的2只
    if defensive_etfs is None:
        daily_ret = close.pct_change().fillna(0)
        vols = daily_ret.std() * np.sqrt(252)
        defensive_etfs = vols.nsmallest(2).index.tolist()

    signal = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    current_holding = []
    entry_prices = {}
    highest_prices = {}
    take_profit_pending = {}  # etf -> 待确认天数
    momentum_decay_pending = {}  # etf -> 待确认天数
    blacklist = set()
    current_month = None
    last_rebalance_week = None  # 周频补仓控制

    # 状态统计
    n_scan = 0
    n_stop_loss = 0
    n_take_profit = 0
    n_momentum_decay = 0
    n_defense = 0
    n_switches = 0

    # 组合equity
    portfolio_equity = pd.Series(1.0, index=close.index)
    running_max = 1.0

    for i, date in enumerate(close.index):
        n_scan += 1
        # 更新组合equity
        if i > 0 and current_holding:
            prev_date = close.index[i-1]
            daily_ret = 0.0
            for etf in current_holding:
                if etf in close.columns:
                    r = close.loc[date, etf] / close.loc[prev_date, etf] - 1
                    daily_ret += r / len(current_holding)
            portfolio_equity.iloc[i] = portfolio_equity.iloc[i-1] * (1 + daily_ret)
        running_max = max(running_max, portfolio_equity.iloc[i])
        current_drawdown = (portfolio_equity.iloc[i] - running_max) / running_max

        # 月初清空黑名单
        month_key = (date.year, date.month)
        if current_month != month_key:
            current_month = month_key
            blacklist = set()

        if date not in candidate_score.index:
            continue

        score_today = candidate_score.loc[date]
        uptrend_today = uptrend.loc[date] if date in uptrend.index else pd.Series(False, index=close.columns)
        early_today = early_stage.loc[date] if date in early_stage.index else pd.Series(False, index=close.columns)

        # 防御判断
        in_defense = False
        if use_defense:
            if date in momentum.index:
                mom_today = momentum.loc[date]
                all_negative = (mom_today.dropna() <= 0).all()
            else:
                all_negative = False
            deep_dd = current_drawdown < drawdown_threshold
            if all_negative or deep_dd:
                in_defense = True
                n_defense += 1

        if in_defense:
            new_holding = [e for e in defensive_etfs if e in close.columns]
            if set(new_holding) != set(current_holding):
                n_switches += 1
            current_holding = new_holding
            entry_prices = {e: close.loc[date, e] for e in new_holding}
            highest_prices = {e: close.loc[date, e] for e in new_holding}
            take_profit_pending = {}
        else:
            # === 止损检查（即时执行，v4.1去掉"跌出上涨区间"条件）===
            if current_holding:
                new_holding = []
                for etf in current_holding:
                    if etf not in entry_prices:
                        new_holding.append(etf)
                        continue
                    price = close.loc[date, etf]
                    if np.isnan(price):
                        new_holding.append(etf)
                        continue
                    if etf in highest_prices:
                        highest_prices[etf] = max(highest_prices[etf], price)
                    else:
                        highest_prices[etf] = price

                    # v4.1: 只保留成本止损 + 跟踪止损（去掉跌出上涨区间）
                    cost_ret = price / entry_prices[etf] - 1
                    trail_ret = price / highest_prices[etf] - 1

                    if cost_ret < cost_stop_loss:
                        blacklist.add(etf)
                        n_stop_loss += 1
                    elif trail_ret < trailing_stop_loss:
                        blacklist.add(etf)
                        n_stop_loss += 1
                    else:
                        new_holding.append(etf)

                if len(new_holding) != len(current_holding):
                    current_holding = new_holding
                    for etf in list(entry_prices.keys()):
                        if etf not in current_holding:
                            del entry_prices[etf]
                            take_profit_pending.pop(etf, None)
                    for etf in list(highest_prices.keys()):
                        if etf not in current_holding:
                            del highest_prices[etf]

            # === 止盈检查（2日确认）===
            if current_holding:
                new_holding = []
                for etf in current_holding:
                    if etf not in entry_prices or etf in defensive_etfs:
                        new_holding.append(etf)
                        continue
                    price = close.loc[date, etf]
                    if np.isnan(price):
                        new_holding.append(etf)
                        continue
                    profit = price / entry_prices[etf] - 1

                    # 硬上限25%立即止盈
                    if profit >= hard_take_profit:
                        blacklist.add(etf)
                        n_take_profit += 1
                        continue

                    # 条件止盈10% + 2日确认
                    if profit >= take_profit_threshold:
                        # 检查是否有上涨初期候选
                        early_candidates = early_today[early_today].index.tolist()
                        early_candidates = [e for e in early_candidates if e not in blacklist and e not in current_holding]
                        if early_candidates:
                            take_profit_pending[etf] = take_profit_pending.get(etf, 0) + 1
                            if take_profit_pending[etf] >= take_profit_confirm_days:
                                n_take_profit += 1
                                continue  # 不加入new_holding，触发换仓
                        else:
                            take_profit_pending[etf] = 0
                    else:
                        take_profit_pending[etf] = 0

                    new_holding.append(etf)

                if len(new_holding) != len(current_holding):
                    current_holding = new_holding
                    for etf in list(entry_prices.keys()):
                        if etf not in current_holding:
                            del entry_prices[etf]
                            take_profit_pending.pop(etf, None)
                    for etf in list(highest_prices.keys()):
                        if etf not in current_holding:
                            del highest_prices[etf]

            # === 动量衰减检查（v4.1: 连续3日确认）===
            if current_holding and date in momentum.index:
                mom_today = momentum.loc[date]
                valid_mom = mom_today[mom_today > 0].sort_values(ascending=False)
                top_n_codes = set(valid_mom.head(momentum_rank_threshold).index)
                new_holding = []
                for etf in current_holding:
                    if etf in defensive_etfs:
                        new_holding.append(etf)
                    elif etf in top_n_codes:
                        new_holding.append(etf)
                        momentum_decay_pending.pop(etf, None)
                    else:
                        # 累计确认天数
                        momentum_decay_pending[etf] = momentum_decay_pending.get(etf, 0) + 1
                        if momentum_decay_pending[etf] >= momentum_decay_confirm_days:
                            n_momentum_decay += 1
                            momentum_decay_pending.pop(etf, None)
                        else:
                            new_holding.append(etf)
                if len(new_holding) != len(current_holding):
                    current_holding = new_holding
                    for etf in list(entry_prices.keys()):
                        if etf not in current_holding:
                            del entry_prices[etf]
                            take_profit_pending.pop(etf, None)
                            momentum_decay_pending.pop(etf, None)
                    for etf in list(highest_prices.keys()):
                        if etf not in current_holding:
                            del highest_prices[etf]

            # === 补仓：v4.1改为周频（每周最多补1次）===
            current_week = date.isocalendar()[1]
            can_rebalance = (not rebalance_weekly) or (last_rebalance_week != current_week)
            if len(current_holding) < hold_count and not in_defense and can_rebalance:
                last_rebalance_week = current_week
                valid_scores = score_today[score_today > 0].sort_values(ascending=False)
                valid_scores = valid_scores[~valid_scores.index.isin(blacklist)]
                valid_scores = valid_scores[~valid_scores.index.isin(current_holding)]
                needed = hold_count - len(current_holding)
                if len(valid_scores) > 0:
                    new_picks = valid_scores.head(needed).index.tolist()
                    if new_picks:
                        n_switches += 1
                        current_holding.extend(new_picks)
                        for etf in new_picks:
                            entry_prices[etf] = close.loc[date, etf]
                            highest_prices[etf] = close.loc[date, etf]

        # 写入当日权重
        if current_holding:
            weight = 1.0 / len(current_holding)
            for etf in current_holding:
                signal.loc[date, etf] = weight

    signal = signal.shift(1).fillna(0)

    info = {
        "n_scan": n_scan,
        "n_switches": n_switches,
        "n_stop_loss": n_stop_loss,
        "n_take_profit": n_take_profit,
        "n_momentum_decay": n_momentum_decay,
        "n_defense": n_defense,
        "defensive_etfs": defensive_etfs,
        "hold_count": hold_count,
    }
    return signal, info


def strategy_unified_v4(
    close, ohlcv_dict,
    ma_short=20, ma_long=60,
    momentum_window=25,
    hold_count=4,
    take_profit_threshold=0.10,
    hard_take_profit=0.25,
    momentum_rank_threshold=6,
    momentum_decay_confirm_days=3,
    take_profit_confirm_days=2,
    cost_stop_loss=-0.12,
    trailing_stop_loss=-0.10,
    use_defense=True,
    drawdown_threshold=-0.15,
    rebalance_weekly=True,
    use_rsrs=True,
    use_ma_filter=True,
    use_reversal_filter=True,
    rsrs_window=18,
    cost=0.0015,
):
    """v4.1统一策略接口

    签名兼容验证框架: (close, ohlcv_dict, **kwargs) -> (equity, metrics, signal)
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "v1_3etf"))
    from backtest import run_backtest, calc_metrics as bt_calc_metrics

    # 识别上涨区间
    uptrend, fresh_bt, early_stage = identify_uptrend(
        close, ma_short=ma_short, ma_long=ma_long, momentum_window=momentum_window
    )

    # 计算候选得分
    candidate_score = calculate_candidate_score(
        close, uptrend, early_stage, momentum_window=momentum_window
    )

    # 回归动量（用于防御判断和动量衰减）
    momentum = calculate_regression_momentum(close, window=momentum_window)

    # 生成信号
    signal, info = generate_signal_v4(
        close, uptrend, early_stage, candidate_score, momentum,
        hold_count=hold_count,
        take_profit_threshold=take_profit_threshold,
        hard_take_profit=hard_take_profit,
        momentum_rank_threshold=momentum_rank_threshold,
        momentum_decay_confirm_days=momentum_decay_confirm_days,
        take_profit_confirm_days=take_profit_confirm_days,
        cost_stop_loss=cost_stop_loss,
        trailing_stop_loss=trailing_stop_loss,
        use_defense=use_defense,
        drawdown_threshold=drawdown_threshold,
        rebalance_weekly=rebalance_weekly,
        cost=cost,
    )

    bt = run_backtest(close, signal, cost=cost)
    metrics = bt_calc_metrics(bt["returns"])
    metrics["n_switches"] = info["n_switches"]
    metrics["n_stop_loss"] = info["n_stop_loss"]
    metrics["n_take_profit"] = info["n_take_profit"]
    metrics["n_momentum_decay"] = info["n_momentum_decay"]
    metrics["n_defense"] = info["n_defense"]
    return bt["equity"], metrics, signal
