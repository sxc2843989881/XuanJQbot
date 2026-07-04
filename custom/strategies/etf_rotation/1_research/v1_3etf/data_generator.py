"""阶段1：模拟ETF数据生成器

使用几何布朗运动(GBM) + 动量持续性 + 市场状态切换 生成3个模拟ETF的日线数据。

3个ETF代表3类资产：
- ETF-A：大盘股（稳健成长，μ=8%, σ=20%）
- ETF-B：小盘股（高弹性高波动，μ=12%, σ=30%）
- ETF-C：商品（低相关避险，μ=5%, σ=15%）

模型：
    S_t = S_{t-1} × exp( (μ + θ·M_t - 0.5·σ²)·dt + σ·dW_t )

    M_t = 动量持续性项 = 过去N日收益率均值
    θ   = 动量持续性强度（0~1，越大动量效应越强）
    市场状态用3状态马尔可夫链切换 μ 的偏移
"""
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta


# ============================================================
# 1. 模拟ETF参数配置
# ============================================================

SIMULATED_ETFS = {
    "ETF-A": {
        "name": "模拟大盘股ETF",
        "mu": 0.08,        # 年化漂移率 8%
        "sigma": 0.20,     # 年化波动率 20%
        "theta": 0.3,      # 动量持续性中等
        "init_price": 1.0,
    },
    "ETF-B": {
        "name": "模拟小盘股ETF",
        "mu": 0.12,        # 年化漂移率 12%（高成长）
        "sigma": 0.30,     # 年化波动率 30%（高波动）
        "theta": 0.5,      # 动量持续性强（小盘动量效应显著）
        "init_price": 1.0,
    },
    "ETF-C": {
        "name": "模拟商品ETF",
        "mu": 0.05,        # 年化漂移率 5%（低成长）
        "sigma": 0.15,     # 年化波动率 15%（低波动）
        "theta": 0.1,      # 动量持续性弱（商品反转多）
        "init_price": 1.0,
    },
}

# 市场状态参数（3状态马尔可夫链）
# 状态：0=牛市, 1=震荡, 2=熊市
MARKET_STATES = {
    0: {"name": "牛市", "mu_shift": 0.08, "sigma_mult": 0.9},   # μ上移, σ略降
    1: {"name": "震荡", "mu_shift": 0.0, "sigma_mult": 1.2},    # μ不变, σ放大
    2: {"name": "熊市", "mu_shift": -0.12, "sigma_mult": 1.3},  # μ下移, σ放大
}

# 状态转移矩阵（每行当前状态，每列下一状态）
TRANSITION_MATRIX = np.array([
    # ->牛   震荡   熊
    [0.95,  0.04,  0.01],  # 当前牛市
    [0.20,  0.70,  0.10],  # 当前震荡
    [0.10,  0.30,  0.60],  # 当前熊市
])


# ============================================================
# 2. 数据生成核心
# ============================================================

def generate_market_states(n_days: int, seed: int = 42) -> np.ndarray:
    """生成市场状态序列（3状态马尔可夫链）

    Args:
        n_days: 天数
        seed: 随机种子

    Returns:
        np.ndarray: 每日市场状态（0=牛, 1=震荡, 2=熊）
    """
    rng = np.random.default_rng(seed)
    states = np.zeros(n_days, dtype=int)
    states[0] = 0  # 起始为牛市

    for t in range(1, n_days):
        current = states[t - 1]
        states[t] = rng.choice(3, p=TRANSITION_MATRIX[current])

    return states


def generate_etf_prices(
    params: dict,
    n_days: int,
    market_states: np.ndarray,
    seed: int,
    momentum_window: int = 20,
    dates: pd.DatetimeIndex = None,
) -> pd.Series:
    """生成单个ETF的价格序列

    模型：S_t = S_{t-1} × exp( (μ + θ·M_t - 0.5·σ²)·dt + σ·dW_t )
    其中 M_t 是过去 momentum_window 日收益率均值（动量持续性项）

    Args:
        params: ETF参数字典 {mu, sigma, theta, init_price, name}
        n_days: 天数
        market_states: 市场状态序列
        seed: 随机种子
        momentum_window: 动量计算窗口
        dates: 日期索引（若提供则用dates作为Series索引）

    Returns:
        pd.Series: 收盘价序列
    """
    rng = np.random.default_rng(seed)
    dt = 1 / 252  # 日频

    mu = params["mu"]
    sigma = params["sigma"]
    theta = params["theta"]
    S = np.zeros(n_days)
    S[0] = params["init_price"]

    # 日收益率序列（用于计算动量持续性项）
    daily_returns = np.zeros(n_days)

    for t in range(1, n_days):
        state = market_states[t]
        state_params = MARKET_STATES[state]

        # 市场状态调整后的μ和σ
        mu_adj = mu + state_params["mu_shift"]
        sigma_adj = sigma * state_params["sigma_mult"]

        # 动量持续性项：过去N日平均收益率
        if t >= momentum_window:
            M_t = np.mean(daily_returns[t - momentum_window : t])
        else:
            M_t = np.mean(daily_returns[:t]) if t > 0 else 0

        # GBM + 动量持续性
        dW = rng.standard_normal() * np.sqrt(dt)
        drift = (mu_adj + theta * M_t - 0.5 * sigma_adj ** 2) * dt
        diffusion = sigma_adj * dW

        daily_ret = drift + diffusion
        S[t] = S[t - 1] * np.exp(daily_ret)
        daily_returns[t] = daily_ret

    if dates is not None:
        return pd.Series(S, index=dates, name=params["name"])
    return pd.Series(S, name=params["name"])


def generate_simulation_data(
    n_years: int = 5,
    seed: int = 42,
    start_date: str = "2021-01-04",
) -> pd.DataFrame:
    """生成3个模拟ETF的日线数据

    Args:
        n_years: 模拟年数
        seed: 随机种子
        start_date: 起始日期

    Returns:
        DataFrame: index=date, columns=[ETF-A, ETF-B, ETF-C] close prices
    """
    n_days = n_years * 252  # 252交易日/年
    dates = pd.bdate_range(start=start_date, periods=n_days)

    # 生成共享的市场状态序列
    market_states = generate_market_states(n_days, seed=seed)

    # 为每个ETF生成价格
    price_data = {}
    for i, (code, params) in enumerate(SIMULATED_ETFS.items()):
        # 每个ETF用不同的随机种子，但共享市场状态
        prices = generate_etf_prices(params, n_days, market_states, seed=seed + i * 100, dates=dates)
        price_data[code] = prices

    close_df = pd.DataFrame(price_data, index=dates)
    close_df.index.name = "date"

    # 同时生成OHLC和成交量（用于后续阶段的双均线过滤等）
    # 简化：open=close*(1+小噪声), high/low=close±范围, volume=随机
    rng = np.random.default_rng(seed + 999)
    ohlcv = {}
    for code in SIMULATED_ETFS.keys():
        close = close_df[code].values
        noise = rng.standard_normal(len(close)) * 0.002
        op = close * (1 + noise)
        hi = np.maximum(close, op) * (1 + np.abs(rng.standard_normal(len(close))) * 0.005)
        lo = np.minimum(close, op) * (1 - np.abs(rng.standard_normal(len(close))) * 0.005)
        vol = rng.integers(1e6, 1e8, size=len(close)).astype(float)
        ohlcv[code] = pd.DataFrame({
            "open": op, "high": hi, "low": lo, "close": close, "volume": vol,
        }, index=close_df.index)

    return close_df, ohlcv, market_states


# ============================================================
# 3. 主入口（测试用）
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("模拟ETF数据生成器（阶段1：3个ETF）")
    print("=" * 60)

    close_df, ohlcv, states = generate_simulation_data(n_years=5, seed=42)

    print(f"\n生成数据: {close_df.shape[0]} 个交易日, {close_df.shape[1]} 个ETF")
    print(f"日期范围: {close_df.index[0].date()} ~ {close_df.index[-1].date()}")

    # 市场状态统计
    state_names = {0: "牛市", 1: "震荡", 2: "熊市"}
    print("\n市场状态分布:")
    for s, name in state_names.items():
        cnt = (states == s).sum()
        print(f"  {name}: {cnt} 天 ({cnt/len(states)*100:.1f}%)")

    # 各ETF统计
    print("\n各ETF统计:")
    for code in close_df.columns:
        close = close_df[code]
        total_ret = close.iloc[-1] / close.iloc[0] - 1
        years = len(close) / 252
        annual_ret = (1 + total_ret) ** (1 / years) - 1
        daily_ret = close.pct_change().dropna()
        annual_vol = daily_ret.std() * np.sqrt(252)
        running_max = close.cummax()
        drawdown = (close - running_max) / running_max
        max_dd = drawdown.min()
        print(f"  {code} ({SIMULATED_ETFS[code]['name']}):")
        print(f"    长期年化={annual_ret*100:.2f}%, 波动率={annual_vol*100:.2f}%, 最大回撤={max_dd*100:.2f}%")
        print(f"    起始价={close.iloc[0]:.4f}, 终止价={close.iloc[-1]:.4f}")

    print("\n=== 完成 ===")
