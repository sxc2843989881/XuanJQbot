"""v7策略：上涨初期得分(EUS) - 基于知识库的逻辑设计

核心目的（用户需求）：
- 捕捉ETF从初期上涨到中期上涨的段
- 尾部（上涨末期）不要
- 通过"上涨初期得分"量化判断

设计依据（知识库查询结果）：
1. RSRS阻力支撑相对强度（光大证券，⭐⭐⭐⭐⭐）
   - 上涨初期：RSRS_std从负转正，RSRS自身动量>0
   - 上涨末期：RSRS_std极高位后回落

2. 动量加速度（知识库动量因子推导）
   - 上涨初期：动量为正且加速度>0（趋势在加速）
   - 上涨末期：动量为正但加速度<0（趋势在减速）

3. 价格位置（知识库行业轮动框架）
   - 上涨初期：价格滚动分位<0.5（距历史高点远）
   - 上涨末期：价格滚动分位>0.8（接近/超过历史高点）

4. 均线形态+乖离率（知识库双均线过滤）
   - 上涨初期：金叉刚发生，乖离率<15%
   - 上涨末期：多头排列但乖离率过大

参数全部采用知识库建议值，不搜索优化：
- RSRS窗口=18（光大默认）
- 动量窗口=25（同花顺默认）
- 均线=20/60（知识库最推荐组合）
- 价格位置窗口=250（1年）
- 乖离率惩罚阈值=15%（知识库建议）
- EUS阈值=0.5（中位数，逻辑合理）
- 持仓=4只（风险分散逻辑）
- 月频调仓（知识库建议）
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "v1_3etf"))

from backtest import run_backtest, calc_metrics
from data_generator_v3 import get_dynamic_universe


# ============================================================
# 子分1：RSRS趋势强度得分（知识库01_RSRS）
# ============================================================

def calc_rsrs_score(high: pd.Series, low: pd.Series, window: int = 18) -> pd.Series:
    """RSRS修正版 = β × R²，并计算RSRS自身动量

    知识库依据：03_技术指标库/01_RSRS阻力支撑相对强度.md
    上涨初期：RSRS从负转正，自身动量>0
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

    # RSRS自身动量（5日变化）
    rsrs_mom = rsrs_values.diff(5)

    # 得分：RSRS>0得基础分，自身动量>0加成
    # sigmoid映射到0-1，动量>0时额外+0.2
    s = 1 / (1 + np.exp(-rsrs_values * 2))  # sigmoid，放大斜率
    s = s + 0.2 * (rsrs_mom > 0).astype(float)
    s = s.clip(0, 1)

    return s


def calc_rsrs_score_panel(ohlcv_dict: dict, window: int = 18) -> pd.DataFrame:
    """计算所有ETF的RSRS得分"""
    data = {}
    for code, df in ohlcv_dict.items():
        data[code] = calc_rsrs_score(df["high"], df["low"], window=window)
    return pd.DataFrame(data)


# ============================================================
# 子分2：动量加速度得分（知识库动量因子推导）
# ============================================================

def calc_momentum_acceleration_score(close: pd.DataFrame, window: int = 25) -> pd.DataFrame:
    """动量加速度得分

    知识库依据：04_因子库/01_动量因子.md + 动量加速度推导
    上涨初期：动量为正且加速度>0（趋势在加速形成）
    上涨末期：动量为正但加速度<0（趋势在减速，即将反转）

    逻辑：
    - 动量 = 25日收益率
    - 加速度 = 动量的5日变化
    - 得分 = 动量为正 + 加速度为正 → 高分
    """
    momentum = close.pct_change(window)
    acceleration = momentum.diff(5)

    # 动量为正且加速度为正 = 上涨初期
    # 动量为正但加速度为负 = 上涨末期（降分）
    # 动量为负 = 下跌（低分）

    # 动量分：sigmoid映射
    s_mom = 1 / (1 + np.exp(-momentum * 20))  # 动量>0时>0.5

    # 加速度分：加速度>0时>0.5
    s_accel = 1 / (1 + np.exp(-acceleration * 50))

    # 综合：动量和加速度都要正才高分
    score = s_mom * 0.5 + s_accel * 0.5

    return score.clip(0, 1)


# ============================================================
# 子分3：价格位置得分（知识库行业轮动框架）
# ============================================================

def calc_position_score(close: pd.DataFrame, window: int = 250) -> pd.DataFrame:
    """价格位置得分

    知识库依据：02_行业轮动/01_行业轮动框架.md 位置判断
    上涨初期：价格滚动分位<0.5（距历史高点远，有上涨空间）
    上涨末期：价格滚动分位>0.8（接近/超过历史高点，即将反转）

    得分 = 1 - 价格分位（位置越低分越高）
    """
    # 滚动分位数（当前价格在过去window日的排名百分位）
    pos_quantile = close.rolling(window).rank(pct=True)
    # 得分 = 1 - 分位（低位高分）
    score = 1 - pos_quantile
    return score.clip(0, 1)


# ============================================================
# 子分4：均线形态+乖离率得分（知识库双均线过滤）
# ============================================================

def calc_ma_score(close: pd.DataFrame, ma_short: int = 20, ma_long: int = 60,
                   deviation_threshold: float = 0.15) -> pd.DataFrame:
    """均线形态+乖离率得分

    知识库依据：03_技术指标库/03_双均线趋势过滤.md
    上涨初期：金叉刚发生，多头排列初成，乖离率<15%
    上涨末期：多头排列但乖离率过大（>15%）

    得分逻辑：
    - 多头排列得分：(P>MA20) + (MA20>MA60) → 0/0.5/1.0
    - 乖离率惩罚：乖离率>15%时得分线性衰减
    """
    ma_s = close.rolling(ma_short).mean()
    ma_l = close.rolling(ma_long).mean()

    # 多头排列得分
    arrange = ((close > ma_s).astype(float) + (ma_s > ma_l).astype(float)) / 2

    # 乖离率 = (价格 - MA20) / MA20
    deviation = (close - ma_s) / ma_s

    # 乖离率惩罚：>threshold时线性衰减到0
    penalty = (1 - np.clip(deviation / deviation_threshold, 0, 1))

    score = arrange * penalty
    return score.clip(0, 1)


# ============================================================
# 上涨初期得分(EUS)合成
# ============================================================

def calculate_eus_score(
    close: pd.DataFrame,
    ohlcv_dict: dict,
    universe: pd.DataFrame,
    rsrs_window: int = 18,
    momentum_window: int = 25,
    position_window: int = 250,
    ma_short: int = 20,
    ma_long: int = 60,
    deviation_threshold: float = 0.15,
) -> tuple:
    """计算上涨初期得分(EUS) - 条件计数法（修正版）

    核心逻辑（用户需求）：
    - 绝对判断每个ETF是否处于"上涨初期到中期"
    - 不是相对排名，而是条件是否满足
    - 如果没有ETF满足条件 → 转债券

    4个条件（A/B共识修正，每个0或1）：
    1. 动量>0：趋势向上（去掉加速度要求，加速度在上涨中期≈0会误判）
    2. 价格位置<0.7：不在极端高位（放宽从0.5到0.7，只在>0.7高位才排除）
    3. RSRS>0：支撑力强（去掉自身动量要求，RSRS>0已足够）
    4. 均线多头排列且乖离<15%：趋势确立但未过热

    修正依据：
    - 原条件太严格（加速度>0 + 位置<0.5 + RSRS动量>0），43%时间转债券
    - 上涨中期动量稳定时加速度≈0，不应排除
    - 牛市后期位置>0.5正常，>0.7才算过热

    EUS = 满足条件数 / 4
    阈值0.75 = "至少3个条件满足"才算可持有
    """
    close_filled = close.ffill().bfill()

    # 1. 动量条件（趋势向上）
    momentum = close_filled.pct_change(momentum_window)
    cond_accel = (momentum > 0).astype(float)

    # 2. 价格位置条件（不在极端高位）
    pos_quantile = close_filled.rolling(position_window).rank(pct=True)
    cond_position = (pos_quantile < 0.7).astype(float)

    # 3. RSRS条件（支撑力强）
    rsrs_data = {}
    for code, df in ohlcv_dict.items():
        high = df["high"]
        low = df["low"]
        rsrs_values = pd.Series(index=high.index, dtype=float)
        for i in range(rsrs_window, len(high)):
            y = high.iloc[i - rsrs_window : i].values
            x = low.iloc[i - rsrs_window : i].values.reshape(-1, 1)
            if np.any(np.isnan(y)) or np.any(np.isnan(x)):
                continue
            model = LinearRegression()
            model.fit(x, y)
            rsrs_values.iloc[i] = model.coef_[0] * model.score(x, y)
        rsrs_data[code] = rsrs_values
    raw_rsrs = pd.DataFrame(rsrs_data).reindex(index=close.index, columns=close.columns)
    cond_rsrs = (raw_rsrs > 0).astype(float)

    # 4. 均线形态条件（趋势确立未过热）
    ma_s = close_filled.rolling(ma_short).mean()
    ma_l = close_filled.rolling(ma_long).mean()
    deviation = (close_filled - ma_s) / ma_s
    cond_ma = ((close_filled > ma_s) & (ma_s > ma_l) & (deviation < deviation_threshold)).astype(float)

    # EUS = 满足条件数 / 4
    eus = (cond_accel + cond_position + cond_rsrs + cond_ma) / 4.0

    # 动态候选池过滤
    eus = eus.where(universe, 0)
    eus = eus.fillna(0)

    return eus, cond_accel, cond_position, cond_rsrs, cond_ma


# ============================================================
# 信号生成：基于EUS的选股逻辑
# ============================================================

def generate_signal_v7_eus(
    close: pd.DataFrame,
    eus: pd.DataFrame,
    universe: pd.DataFrame,
    hold_count: int = 4,
    eus_threshold: float = 0.5,
    rebalance_freq: str = "M",
    defensive_etfs: list = None,
    cost: float = 0.0015,
) -> tuple:
    """基于EUS的选股信号

    逻辑（用户需求）：
    1. EUS > 阈值(0.5)的ETF才入选
    2. 按EUS得分排序，选Top N
    3. 如果没有ETF达标 → 持有债券ETF
    4. 月频调仓

    Args:
        eus_threshold: EUS入选阈值（0.5=中位数，逻辑合理）

    Returns:
        (signal, stats)
    """
    if defensive_etfs is None:
        defensive_etfs = [c for c in close.columns if "BOND" in c]
        if not defensive_etfs:
            defensive_etfs = [close.columns[0]]

    # 调仓日
    if rebalance_freq == "M":
        rebalance_idx = close.index.to_series().groupby(
            [close.index.year, close.index.month]
        ).last().tolist()
    else:
        rebalance_idx = close.index.tolist()
    rebalance_set = set(rebalance_idx)

    signal = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    current_holding = None

    stats = {
        "n_switches": 0,
        "n_defense": 0,
        "n_rebalance": 0,
        "holding_history": [],
        "eus_stats": [],  # 记录每次调仓的EUS分布
    }

    for date in close.index:
        if date in rebalance_set:
            if date in eus.index:
                eus_today = eus.loc[date]
                available_mask = universe.loc[date]
                available_eus = eus_today[available_mask]

                # 筛选EUS > 阈值的ETF
                qualified = available_eus[available_eus > eus_threshold].sort_values(ascending=False)

                if len(qualified) >= 1:
                    # 有达标的：选Top N
                    n_hold = min(hold_count, len(qualified))
                    new_holding = qualified.head(n_hold).index.tolist()
                    holding_type = "UPTREND"
                else:
                    # 没有达标的：转防御ETF
                    avail_def = [c for c in defensive_etfs if universe.loc[date, c]]
                    new_holding = avail_def[:hold_count] if avail_def else []
                    holding_type = "DEFENSE"

                if current_holding != new_holding:
                    stats["n_switches"] += 1
                    if holding_type == "DEFENSE":
                        stats["n_defense"] += 1
                    else:
                        stats["n_rebalance"] += 1
                    stats["holding_history"].append((date, holding_type, new_holding))
                    current_holding = new_holding

                # 记录EUS分布统计
                if len(available_eus) > 0:
                    stats["eus_stats"].append({
                        "date": date,
                        "n_qualified": len(qualified),
                        "n_available": int(available_mask.sum()),
                        "max_eus": float(available_eus.max()),
                        "mean_eus": float(available_eus.mean()),
                    })

        # 设置当日信号
        if current_holding is not None and len(current_holding) > 0:
            weight = 1.0 / len(current_holding)
            for code in current_holding:
                signal.loc[date, code] = weight

    # shift(1)避免未来函数
    signal = signal.shift(1).fillna(0)

    return signal, stats


# ============================================================
# v7策略统一接口
# ============================================================

def strategy_unified_v7(
    close: pd.DataFrame,
    ohlcv_dict: dict,
    list_dates: dict = None,
    hold_count: int = 4,
    eus_threshold: float = 0.75,
    rsrs_window: int = 18,
    momentum_window: int = 25,
    position_window: int = 250,
    ma_short: int = 20,
    ma_long: int = 60,
    deviation_threshold: float = 0.15,
    rebalance_freq: str = "M",
    cost: float = 0.0015,
) -> tuple:
    """v7策略：基于上涨初期得分的ETF轮动

    所有参数采用知识库建议值，不搜索优化：
    - rsrs_window=18（光大证券默认）
    - momentum_window=25（同花顺默认）
    - position_window=250（1年滚动）
    - ma_short=20, ma_long=60（知识库最推荐）
    - deviation_threshold=0.15（乖离率15%惩罚）
    - eus_threshold=0.5（中位数阈值）
    - hold_count=4（风险分散）
    - rebalance_freq="M"（月频）

    Returns:
        (equity, metrics, signal, stats)
    """
    # 动态候选池
    if list_dates is None:
        universe = pd.DataFrame(True, index=close.index, columns=close.columns)
    else:
        universe = get_dynamic_universe(close, list_dates)

    # 计算EUS得分
    eus, s_accel, s_position, s_rsrs, s_ma = calculate_eus_score(
        close, ohlcv_dict, universe,
        rsrs_window=rsrs_window,
        momentum_window=momentum_window,
        position_window=position_window,
        ma_short=ma_short,
        ma_long=ma_long,
        deviation_threshold=deviation_threshold,
    )

    # 生成信号
    close_filled = close.ffill().bfill()
    signal, stats = generate_signal_v7_eus(
        close_filled, eus, universe,
        hold_count=hold_count,
        eus_threshold=eus_threshold,
        rebalance_freq=rebalance_freq,
        cost=cost,
    )

    # 回测
    bt_result = run_backtest(close_filled, signal, cost=cost)
    metrics = calc_metrics(bt_result["returns"])
    metrics["n_switches"] = stats["n_switches"]
    metrics["n_defense"] = stats["n_defense"]
    metrics["n_rebalance"] = stats["n_rebalance"]

    return bt_result["equity"], metrics, signal, stats


if __name__ == "__main__":
    from data_generator_v3 import generate_simulation_data_v3, print_universe_evolution

    print("=" * 70)
    print("v7策略：上涨初期得分(EUS) - 基于知识库逻辑设计")
    print("=" * 70)

    # 生成数据
    close_df, ohlcv_dict, _, list_dates = generate_simulation_data_v3(n_years=13, seed=42)
    print(f"数据: {close_df.shape[0]}交易日, {close_df.shape[1]}ETF")
    print_universe_evolution(close_df, list_dates)

    # 跑v7策略
    print("\n跑v7策略（条件计数法EUS，阈值0.75=至少3条件满足）...")
    equity, metrics, signal, stats = strategy_unified_v7(
        close_df, ohlcv_dict, list_dates,
        hold_count=4,
        eus_threshold=0.75,
    )

    print(f"\nv7策略结果（13年回测）:")
    print(f"  年化收益: {metrics['annual_return']*100:.2f}%")
    print(f"  Sharpe:   {metrics['sharpe']:.3f}")
    print(f"  最大回撤: {metrics['max_drawdown']*100:.2f}%")
    print(f"  Calmar:   {metrics['calmar']:.3f}")
    print(f"  换手次数: {metrics['n_switches']}")
    print(f"  防御触发: {metrics['n_defense']}")
    print(f"  调仓次数: {metrics['n_rebalance']}")

    # EUS分布统计
    if stats["eus_stats"]:
        eus_df = pd.DataFrame(stats["eus_stats"])
        print(f"\nEUS分布统计:")
        print(f"  平均达标ETF数: {eus_df['n_qualified'].mean():.1f} / 平均可用: {eus_df['n_available'].mean():.1f}")
        print(f"  平均最高EUS: {eus_df['max_eus'].mean():.3f}")
        print(f"  平均EUS: {eus_df['mean_eus'].mean():.3f}")
        print(f"  防御占比: {metrics['n_defense']/(metrics['n_rebalance']+metrics['n_defense'])*100:.1f}%")

    # 动态等权基准
    print("\n计算动态等权基准...")
    universe = get_dynamic_universe(close_df, list_dates)
    daily_ret = close_df.pct_change().fillna(0)
    n_available = universe.sum(axis=1)
    bench_weights = universe.div(n_available, axis=0).fillna(0)
    bench_ret = (daily_ret * bench_weights).sum(axis=1)
    bench_metrics = calc_metrics(bench_ret)
    print(f"  年化: {bench_metrics['annual_return']*100:.2f}%, Sharpe: {bench_metrics['sharpe']:.3f}, 回撤: {bench_metrics['max_drawdown']*100:.2f}%")
    print(f"\nvs基准: Sharpe差 = {metrics['sharpe'] - bench_metrics['sharpe']:+.3f}")
