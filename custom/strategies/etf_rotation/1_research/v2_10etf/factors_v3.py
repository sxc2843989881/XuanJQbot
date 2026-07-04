"""ETF轮动策略 优化版 v3

基于 A/B 双角色共识的4项优化：
1. 持仓数 2→4 + 等权 25%/只（降低 idiosyncratic 风险）
2. 调仓阈值 10% + 月末信号次月初确认（降换手）
3. 防御机制：全池动量为负 OR 回撤>15% → 防御ETF（债券+消费 50/50）
4. L2止损：12%成本止损 + 10%跟踪止损

知识库依据：
- 04_风险控制机制.md L377（持仓数）、L334（调仓阈值）、L258（防御）、L170（止损）
- 02_动量因子计算方法.md L356（等权稳健无偏）
"""
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
from factors_v2 import calculate_composite_score, calculate_regression_momentum


def generate_signal_v3(
    close: pd.DataFrame,
    score: pd.DataFrame,
    momentum: pd.DataFrame,
    rebalance_freq: str = "M",
    hold_count: int = 4,
    switch_threshold: float = 0.10,
    empty_threshold: float = 0.0,
    use_defense: bool = True,
    defensive_etfs: Optional[List[str]] = None,
    drawdown_threshold: float = -0.15,
    use_stop_loss: bool = True,
    cost_stop_loss: float = -0.12,
    trailing_stop_loss: float = -0.10,
) -> Tuple[pd.DataFrame, dict]:
    """优化版信号生成 v3

    Args:
        close: 收盘价 DataFrame
        score: 综合评分 DataFrame（来自 calculate_composite_score）
        momentum: 回归动量 DataFrame（用于判断全池动量）
        rebalance_freq: 调仓频率
        hold_count: 持仓数量（默认4）
        switch_threshold: 调仓阈值（新标的得分需超当前持仓得分此比例才换）
        empty_threshold: 空仓阈值
        use_defense: 是否启用防御机制
        defensive_etfs: 防御ETF列表（默认取波动率最低的2只）
        drawdown_threshold: 防御触发的回撤阈值（-0.15 = -15%）
        use_stop_loss: 是否启用L2止损
        cost_stop_loss: 成本止损线（-0.12 = -12%）
        trailing_stop_loss: 跟踪止损线（-0.10 = -10%）

    Returns:
        signal: 持仓权重 DataFrame
        info: 状态信息 dict（换手次数、防御触发次数、止损次数等）
    """
    # 调仓日
    if rebalance_freq == "M":
        rebalance_idx = close.groupby(close.index.to_period("M")).apply(lambda x: x.index[-1])
    elif rebalance_freq == "W":
        rebalance_idx = close.groupby(close.index.to_period("W")).apply(lambda x: x.index[-1])
    else:
        raise ValueError(f"不支持的频率: {rebalance_freq}")
    rebalance_set = set(rebalance_idx)

    # 默认防御ETF：波动率最低的2只
    if defensive_etfs is None:
        daily_ret = close.pct_change().fillna(0)
        vols = daily_ret.std() * np.sqrt(252)
        defensive_etfs = vols.nsmallest(2).index.tolist()
    defensive_set = set(defensive_etfs)

    # 逐日前向填充 + 止损 + 防御
    signal = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    current_holding = []  # list of etf codes
    entry_prices = {}  # etf -> 买入价
    highest_prices = {}  # etf -> 持仓期间最高价
    blacklist = set()  # 当月止损过的ETF
    current_month = None

    # 状态统计
    n_rebalances = 0
    n_switches = 0  # 实际换仓次数
    n_defense_triggered = 0
    n_stop_loss = 0
    n_threshold_skip = 0  # 阈值跳过次数

    # 组合equity用于回撤判断
    portfolio_equity = pd.Series(1.0, index=close.index)
    running_max = 1.0

    for i, date in enumerate(close.index):
        # 更新组合equity
        if i > 0 and current_holding:
            prev_date = close.index[i-1]
            daily_ret = 0.0
            for etf in current_holding:
                if etf in close.columns:
                    ret = close.loc[date, etf] / close.loc[prev_date, etf] - 1
                    daily_ret += ret / len(current_holding)
            portfolio_equity.iloc[i] = portfolio_equity.iloc[i-1] * (1 + daily_ret)
        running_max = max(running_max, portfolio_equity.iloc[i])
        current_drawdown = (portfolio_equity.iloc[i] - running_max) / running_max

        # 月初清空止损黑名单
        month_key = (date.year, date.month)
        if current_month != month_key:
            current_month = month_key
            blacklist = set()

        # 止损检查（每日）
        if use_stop_loss and current_holding:
            new_holding = []
            for etf in current_holding:
                if etf not in entry_prices:
                    new_holding.append(etf)
                    continue
                price = close.loc[date, etf]
                if np.isnan(price):
                    new_holding.append(etf)
                    continue
                # 更新最高价
                if etf in highest_prices:
                    highest_prices[etf] = max(highest_prices[etf], price)
                else:
                    highest_prices[etf] = price
                # 成本止损
                cost_ret = price / entry_prices[etf] - 1
                # 跟踪止损
                trail_ret = price / highest_prices[etf] - 1
                if cost_ret < cost_stop_loss or trail_ret < trailing_stop_loss:
                    blacklist.add(etf)
                    n_stop_loss += 1
                else:
                    new_holding.append(etf)
            if len(new_holding) != len(current_holding):
                current_holding = new_holding
                # 清理被止损ETF的入场价记录
                for etf in list(entry_prices.keys()):
                    if etf not in current_holding:
                        del entry_prices[etf]
                for etf in list(highest_prices.keys()):
                    if etf not in current_holding:
                        del highest_prices[etf]

        # 调仓日决策
        if date in rebalance_set:
            n_rebalances += 1
            if date not in score.index:
                continue
            score_today = score.loc[date]

            # 防御判断
            in_defense = False
            if use_defense:
                # 条件1：全池动量为负
                if date in momentum.index:
                    mom_today = momentum.loc[date]
                    all_negative = (mom_today.dropna() <= 0).all()
                else:
                    all_negative = False
                # 条件2：组合回撤 > 阈值
                deep_drawdown = current_drawdown < drawdown_threshold
                if all_negative or deep_drawdown:
                    in_defense = True
                    n_defense_triggered += 1

            if in_defense:
                # 防御持仓
                new_holding = [e for e in defensive_etfs if e in close.columns]
                current_holding = new_holding
                entry_prices = {e: close.loc[date, e] for e in new_holding}
                highest_prices = {e: close.loc[date, e] for e in new_holding}
            else:
                # 正常选股
                valid_scores = score_today[score_today > 0].sort_values(ascending=False)
                # 排除当月止损黑名单
                valid_scores = valid_scores[~valid_scores.index.isin(blacklist)]

                if len(valid_scores) == 0 or valid_scores.iloc[0] < empty_threshold:
                    current_holding = []
                    entry_prices = {}
                    highest_prices = {}
                else:
                    top_n = valid_scores.head(hold_count).index.tolist()

                    # 调仓阈值检查：新标的得分需超当前持仓得分10%才换
                    if current_holding and switch_threshold > 0:
                        cur_scores = [score_today.get(e, 0) for e in current_holding]
                        cur_mean = np.mean(cur_scores) if cur_scores else 0
                        new_scores = [score_today.get(e, 0) for e in top_n]
                        new_mean = np.mean(new_scores) if new_scores else 0
                        if cur_mean > 0 and new_mean <= cur_mean * (1 + switch_threshold):
                            # 不换仓
                            n_threshold_skip += 1
                        else:
                            if set(top_n) != set(current_holding):
                                n_switches += 1
                            current_holding = top_n
                            entry_prices = {e: close.loc[date, e] for e in top_n}
                            highest_prices = {e: close.loc[date, e] for e in top_n}
                    else:
                        if set(top_n) != set(current_holding):
                            n_switches += 1
                        current_holding = top_n
                        entry_prices = {e: close.loc[date, e] for e in top_n}
                        highest_prices = {e: close.loc[date, e] for e in top_n}

        # 写入当日权重
        if current_holding:
            weight = 1.0 / len(current_holding)
            for etf in current_holding:
                signal.loc[date, etf] = weight

    # shift(1)避免未来函数
    signal = signal.shift(1).fillna(0)

    info = {
        "n_rebalances": n_rebalances,
        "n_switches": n_switches,
        "n_defense_triggered": n_defense_triggered,
        "n_stop_loss": n_stop_loss,
        "n_threshold_skip": n_threshold_skip,
        "defensive_etfs": defensive_etfs,
        "hold_count": hold_count,
    }
    return signal, info


def strategy_unified_v3(
    close, ohlcv_dict,
    momentum_window=25, rsrs_window=18,
    ma_short=20, ma_long=60,
    use_rsrs=True, use_ma_filter=True, use_reversal_filter=True,
    rebalance_freq="M", hold_count=4,
    switch_threshold=0.10,
    use_defense=True, drawdown_threshold=-0.15,
    use_stop_loss=True, cost_stop_loss=-0.12, trailing_stop_loss=-0.10,
    cost=0.0015,
):
    """统一策略接口 v3

    签名兼容验证框架: (close, ohlcv_dict, **kwargs) -> (equity, metrics, signal)
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "v1_3etf"))
    from backtest import run_backtest, calc_metrics as bt_calc_metrics

    score, momentum, _, _, _ = calculate_composite_score(
        close, ohlcv_dict,
        momentum_window=momentum_window,
        rsrs_window=rsrs_window,
        ma_short=ma_short,
        ma_long=ma_long,
        use_rsrs=use_rsrs,
        use_ma_filter=use_ma_filter,
        use_reversal_filter=use_reversal_filter,
    )
    signal, info = generate_signal_v3(
        close, score, momentum,
        rebalance_freq=rebalance_freq,
        hold_count=hold_count,
        switch_threshold=switch_threshold,
        use_defense=use_defense,
        drawdown_threshold=drawdown_threshold,
        use_stop_loss=use_stop_loss,
        cost_stop_loss=cost_stop_loss,
        trailing_stop_loss=trailing_stop_loss,
    )
    bt = run_backtest(close, signal, cost=cost)
    metrics = bt_calc_metrics(bt["returns"])
    metrics["n_switches"] = info["n_switches"]
    metrics["n_defense"] = info["n_defense_triggered"]
    metrics["n_stop_loss"] = info["n_stop_loss"]
    return bt["equity"], metrics, signal
