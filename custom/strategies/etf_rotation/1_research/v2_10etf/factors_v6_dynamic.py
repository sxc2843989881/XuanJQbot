"""v6策略 - 动态候选池版本

适配20只ETF动态上市场景：
1. 因子计算时只考虑已上市的ETF
2. 选Top4时只从已上市的ETF中选
3. 防御机制中的"全池动量"也只考虑已上市的
4. 已上市ETF不足4只时，只持有已上市的

v6核心配置（A/B共识最优）：
- hold_count=4
- switch_threshold=0.0 (无调仓阈值)
- use_stop_loss=False (无止损)
- drawdown_threshold=-0.08 (防御-8%)
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "v1_3etf"))

from factors_v2 import (
    calculate_regression_momentum,
    calculate_rsrs_panel,
    calculate_dual_ma_filter,
    calculate_reversal_filter,
)
from backtest import run_backtest, calc_metrics
from data_generator_v3 import get_dynamic_universe


def calculate_composite_score_dynamic(
    close: pd.DataFrame,
    ohlcv_dict: dict,
    universe: pd.DataFrame,
    momentum_window: int = 25,
    rsrs_window: int = 18,
    ma_short: int = 20,
    ma_long: int = 60,
) -> pd.DataFrame:
    """计算综合评分（动态候选池版）

    未上市的ETF得分为0，不参与选股
    关键：用ffill填充NaN，让因子函数能正常计算，再用universe过滤
    """
    # 用ffill填充未上市的NaN（用上市第一天的价格向前填充不影响，因为universe会过滤）
    close_filled = close.ffill().bfill()

    # 1. 回归动量
    momentum = calculate_regression_momentum(close_filled, window=momentum_window)

    # 2. RSRS质量因子
    rsrs = calculate_rsrs_panel(ohlcv_dict, window=rsrs_window)
    rsrs_score = rsrs.clip(-2, 2) / 4 + 0.5

    # 3. 综合得分 = 动量 × 质量
    score = momentum * rsrs_score

    # 4. 双均线过滤
    ma_filter = calculate_dual_ma_filter(close_filled, ma_short, ma_long)
    score = score.where(ma_filter, 0)

    # 5. 趋势反转过滤
    reversal_filter = calculate_reversal_filter(close_filled)
    score = score.where(reversal_filter, 0)

    # 6. 负得分设为0
    score = score.where(score > 0, 0)

    # 7. 动态候选池过滤：未上市的ETF得分为0
    score = score.where(universe, 0)

    # 8. NaN处理
    score = score.fillna(0)

    return score


def generate_signal_v6_dynamic(
    close: pd.DataFrame,
    score: pd.DataFrame,
    universe: pd.DataFrame,
    momentum: pd.DataFrame,
    hold_count: int = 4,
    switch_threshold: float = 0.0,
    use_defense: bool = True,
    defensive_etfs: list = None,
    drawdown_threshold: float = -0.08,
    rebalance_freq: str = "M",
    cost: float = 0.0015,
) -> tuple:
    """生成v6信号（动态候选池版）

    核心逻辑：
    1. 月频调仓，选Top4
    2. 无调仓阈值（switch_threshold=0.0）
    3. 无止损
    4. 防御机制：组合回撤>8%切防御ETF
    5. 只从已上市ETF中选

    Returns:
        (signal, stats_dict)
    """
    if defensive_etfs is None:
        # 默认用债券ETF做防御
        defensive_etfs = [c for c in close.columns if "BOND" in c]
        if not defensive_etfs:
            defensive_etfs = [close.columns[0]]

    # 调仓日
    if rebalance_freq == "M":
        rebalance_idx = close.index.to_series().groupby(
            [close.index.year, close.index.month]
        ).last().tolist()
    elif rebalance_freq == "W":
        rebalance_idx = close.index.to_series().groupby(
            [close.index.year, close.index.isocalendar().week]
        ).last().tolist()
    else:
        rebalance_idx = close.index.tolist()
    rebalance_set = set(rebalance_idx)

    # 初始化
    signal = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    current_holding = None
    current_entry_prices = None
    equity_curve = [1.0]
    peak_equity = 1.0

    stats = {
        "n_switches": 0,
        "n_defense": 0,
        "n_stop_loss": 0,
        "holding_history": [],
    }

    daily_ret = close.pct_change().fillna(0)

    for i, date in enumerate(close.index):
        # 计算当日组合收益
        if current_holding is not None and i > 0:
            port_ret = 0.0
            for code in current_holding:
                if code in daily_ret.columns:
                    port_ret += daily_ret.loc[date, code] * (1.0 / len(current_holding))
            new_equity = equity_curve[-1] * (1 + port_ret)
            equity_curve.append(new_equity)
            peak_equity = max(peak_equity, new_equity)
        else:
            if i > 0:
                equity_curve.append(equity_curve[-1])
            peak_equity = max(peak_equity, equity_curve[-1])

        # 计算当前回撤
        current_equity = equity_curve[-1]
        drawdown = (current_equity - peak_equity) / peak_equity if peak_equity > 0 else 0

        # 判断是否防御
        in_defense = False
        if use_defense and drawdown < drawdown_threshold:
            # 检查防御ETF是否已上市
            available_defensive = [c for c in defensive_etfs if universe.loc[date, c]]
            if available_defensive:
                in_defense = True

        # 检查全池动量是否全负
        if use_defense and date in momentum.index:
            mom_today = momentum.loc[date]
            # 只看已上市的ETF
            available_mask = universe.loc[date]
            available_mom = mom_today[available_mask].dropna()
            if len(available_mom) > 0 and (available_mom <= 0).all():
                available_defensive = [c for c in defensive_etfs if universe.loc[date, c]]
                if available_defensive:
                    in_defense = True

        # 调仓日决策
        if date in rebalance_set:
            if in_defense:
                # 防御模式：持有防御ETF
                available_defensive = [c for c in defensive_etfs if universe.loc[date, c]]
                if available_defensive:
                    new_holding = available_defensive[:hold_count]
                    if current_holding != new_holding:
                        stats["n_defense"] += 1
                        stats["n_switches"] += 1
                        stats["holding_history"].append((date, "DEFENSE", new_holding))
                        current_holding = new_holding
            else:
                # 正常模式：选Top4
                if date in score.index:
                    score_today = score.loc[date]
                    available_mask = universe.loc[date]
                    available_scores = score_today[available_mask]
                    valid_scores = available_scores[available_scores > 0].sort_values(ascending=False)

                    if len(valid_scores) >= 1:
                        n_hold = min(hold_count, len(valid_scores))
                        new_holding = valid_scores.head(n_hold).index.tolist()

                        # 检查是否需要换仓
                        if current_holding != new_holding:
                            # 无调仓阈值（switch_threshold=0.0），直接换
                            if switch_threshold == 0.0:
                                stats["n_switches"] += 1
                                stats["holding_history"].append((date, "REBALANCE", new_holding))
                                current_holding = new_holding
                            else:
                                # 有调仓阈值时检查
                                if current_holding is None:
                                    stats["n_switches"] += 1
                                    current_holding = new_holding
                                else:
                                    new_score = valid_scores.head(n_hold).mean()
                                    current_scores = [score_today.get(c, 0) for c in current_holding if c in score_today.index]
                                    current_avg = np.mean(current_scores) if current_scores else 0
                                    if new_score > current_avg * (1 + switch_threshold):
                                        stats["n_switches"] += 1
                                        current_holding = new_holding

        # 设置当日信号
        if current_holding is not None:
            weight = 1.0 / len(current_holding)
            for code in current_holding:
                signal.loc[date, code] = weight

    # shift(1)避免未来函数
    signal = signal.shift(1).fillna(0)

    return signal, stats


def strategy_unified_v6_dynamic(
    close: pd.DataFrame,
    ohlcv_dict: dict,
    list_dates: dict = None,
    hold_count: int = 4,
    switch_threshold: float = 0.0,
    use_stop_loss: bool = False,
    drawdown_threshold: float = -0.08,
    use_defense: bool = True,
    cost_stop_loss: float = -0.12,
    trailing_stop_loss: float = -0.10,
    momentum_window: int = 25,
    rsrs_window: int = 18,
    ma_short: int = 20,
    ma_long: int = 60,
    rebalance_freq: str = "M",
    cost: float = 0.0015,
) -> tuple:
    """v6策略统一接口（动态候选池版）

    Args:
        close: 收盘价（未上市的为NaN）
        ohlcv_dict: OHLCV数据
        list_dates: 上市日期字典
        hold_count: 持仓数量
        switch_threshold: 调仓阈值（0.0=无阈值）
        use_stop_loss: 是否使用止损
        drawdown_threshold: 防御回撤阈值
        ...

    Returns:
        (equity, metrics, signal)
    """
    # 获取动态候选池
    if list_dates is None:
        universe = pd.DataFrame(True, index=close.index, columns=close.columns)
    else:
        universe = get_dynamic_universe(close, list_dates)

    # 用ffill填充NaN用于回测计算（未上市的不参与信号，不影响）
    close_filled = close.ffill().bfill()

    # 计算综合评分
    score = calculate_composite_score_dynamic(
        close, ohlcv_dict, universe,
        momentum_window=momentum_window,
        rsrs_window=rsrs_window,
        ma_short=ma_short,
        ma_long=ma_long,
    )

    # 计算动量（用于防御判断）
    momentum = calculate_regression_momentum(close_filled, window=momentum_window)

    # 生成信号
    signal, stats = generate_signal_v6_dynamic(
        close_filled, score, universe, momentum,
        hold_count=hold_count,
        switch_threshold=switch_threshold,
        use_defense=use_defense,
        drawdown_threshold=drawdown_threshold,
        rebalance_freq=rebalance_freq,
        cost=cost,
    )

    # 回测（用填充后的close，避免NaN传播）
    bt_result = run_backtest(close_filled, signal, cost=cost)
    metrics = calc_metrics(bt_result["returns"])
    metrics["n_switches"] = stats["n_switches"]
    metrics["n_defense"] = stats["n_defense"]
    metrics["n_stop_loss"] = stats["n_stop_loss"]

    return bt_result["equity"], metrics, signal


if __name__ == "__main__":
    from data_generator_v3 import generate_simulation_data_v3, print_universe_evolution

    print("=" * 70)
    print("v6策略 - 动态候选池版本测试")
    print("=" * 70)

    # 生成数据
    close_df, ohlcv_dict, states, list_dates = generate_simulation_data_v3(n_years=13, seed=42)
    print(f"数据: {close_df.shape[0]}交易日, {close_df.shape[1]}ETF")
    print_universe_evolution(close_df, list_dates)

    # 跑v6策略
    print("\n跑v6策略（动态候选池版）...")
    equity, metrics, signal = strategy_unified_v6_dynamic(
        close_df, ohlcv_dict, list_dates,
        hold_count=4,
        switch_threshold=0.0,
        use_stop_loss=False,
        drawdown_threshold=-0.08,
    )

    print(f"\nv6策略结果（13年回测）:")
    print(f"  年化收益: {metrics['annual_return']*100:.2f}%")
    print(f"  Sharpe:   {metrics['sharpe']:.3f}")
    print(f"  最大回撤: {metrics['max_drawdown']*100:.2f}%")
    print(f"  Calmar:   {metrics['calmar']:.3f}")
    print(f"  换手次数: {metrics['n_switches']}")
    print(f"  防御触发: {metrics['n_defense']}")

    # 等权基准（动态：等权持有当日所有已上市ETF）
    print("\n计算动态等权基准...")
    universe = get_dynamic_universe(close_df, list_dates)
    daily_ret = close_df.pct_change().fillna(0)

    # 动态等权：每日等权持有所有已上市ETF
    n_available = universe.sum(axis=1)
    bench_weights = universe.div(n_available, axis=0).fillna(0)
    bench_ret = (daily_ret * bench_weights).sum(axis=1)
    bench_equity = (1 + bench_ret).cumprod()

    bench_metrics = calc_metrics(bench_ret)
    print(f"\n动态等权基准:")
    print(f"  年化收益: {bench_metrics['annual_return']*100:.2f}%")
    print(f"  Sharpe:   {bench_metrics['sharpe']:.3f}")
    print(f"  最大回撤: {bench_metrics['max_drawdown']*100:.2f}%")
    print(f"\nvs基准: Sharpe差 = {metrics['sharpe'] - bench_metrics['sharpe']:+.3f}")
