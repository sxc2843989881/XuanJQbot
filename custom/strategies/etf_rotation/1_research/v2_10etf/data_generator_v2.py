"""阶段2：10个模拟ETF数据生成器

覆盖5类资产×2风格，每个ETF有不同参数。
相比阶段1的3个ETF，10个标的能提供更好的截面动量区分度。

10个ETF：
- ETF-LV：大盘价值（低波动，稳健）
- ETF-LG：大盘成长（中等波动，趋势性强）
- ETF-SV：小盘价值（中等波动，反转多）
- ETF-SG：小盘成长（高波动，动量强）
- ETF-CY：周期（高波动，强动量）
- ETF-TE：科技（高波动，强动量）
- ETF-CO：消费（低波动，防御）
- ETF-ME：医药（中波动，防御）
- ETF-BO：债券（极低波动，避险）
- ETF-OV：海外（中等波动，低相关）
"""
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# 复用阶段1的市场状态机制
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "v1_3etf"))
from data_generator import generate_market_states, MARKET_STATES, TRANSITION_MATRIX


# ============================================================
# 10个模拟ETF参数（覆盖5类资产×2风格）
# ============================================================

SIMULATED_ETFS_V2 = {
    "ETF-LV": {"name": "模拟大盘价值ETF", "mu": 0.06, "sigma": 0.18, "theta": 0.2, "init_price": 1.0},
    "ETF-LG": {"name": "模拟大盘成长ETF", "mu": 0.10, "sigma": 0.22, "theta": 0.4, "init_price": 1.0},
    "ETF-SV": {"name": "模拟小盘价值ETF", "mu": 0.05, "sigma": 0.25, "theta": 0.15, "init_price": 1.0},
    "ETF-SG": {"name": "模拟小盘成长ETF", "mu": 0.13, "sigma": 0.32, "theta": 0.5, "init_price": 1.0},
    "ETF-CY": {"name": "模拟周期ETF",     "mu": 0.08, "sigma": 0.30, "theta": 0.45, "init_price": 1.0},
    "ETF-TE": {"name": "模拟科技ETF",     "mu": 0.14, "sigma": 0.28, "theta": 0.55, "init_price": 1.0},
    "ETF-CO": {"name": "模拟消费ETF",     "mu": 0.07, "sigma": 0.16, "theta": 0.2, "init_price": 1.0},
    "ETF-ME": {"name": "模拟医药ETF",     "mu": 0.09, "sigma": 0.24, "theta": 0.3, "init_price": 1.0},
    "ETF-BO": {"name": "模拟债券ETF",     "mu": 0.04, "sigma": 0.05, "theta": 0.05, "init_price": 1.0},
    "ETF-OV": {"name": "模拟海外ETF",     "mu": 0.11, "sigma": 0.20, "theta": 0.35, "init_price": 1.0},
}


def generate_etf_prices_v2(
    params: dict,
    n_days: int,
    market_states: np.ndarray,
    seed: int,
    momentum_window: int = 20,
    dates: pd.DatetimeIndex = None,
) -> pd.DataFrame:
    """生成单个ETF的OHLCV数据（阶段2需要high/low用于RSRS）

    模型：S_t = S_{t-1} × exp( (μ + θ·M_t - 0.5·σ²)·dt + σ·dW_t )
    """
    rng = np.random.default_rng(seed)
    dt = 1 / 252

    mu = params["mu"]
    sigma = params["sigma"]
    theta = params["theta"]

    S = np.zeros(n_days)
    S[0] = params["init_price"]
    daily_returns = np.zeros(n_days)

    # 同时生成high/low/open/close用于RSRS计算
    highs = np.zeros(n_days)
    lows = np.zeros(n_days)
    opens = np.zeros(n_days)
    volumes = np.zeros(n_days)

    highs[0] = S[0] * 1.005
    lows[0] = S[0] * 0.995
    opens[0] = S[0]

    for t in range(1, n_days):
        state = market_states[t]
        state_params = MARKET_STATES[state]
        mu_adj = mu + state_params["mu_shift"]
        sigma_adj = sigma * state_params["sigma_mult"]

        # 动量持续性项
        if t >= momentum_window:
            M_t = np.mean(daily_returns[t - momentum_window : t])
        else:
            M_t = np.mean(daily_returns[:t]) if t > 0 else 0

        # GBM + 动量持续性
        dW = rng.standard_normal() * np.sqrt(dt)
        drift = (mu_adj + theta * M_t - 0.5 * sigma_adj ** 2) * dt
        diffusion = sigma_adj * dW
        daily_ret = drift + diffusion

        opens[t] = S[t - 1]
        S[t] = S[t - 1] * np.exp(daily_ret)
        daily_returns[t] = daily_ret

        # 生成high/low（日内波动范围，与sigma相关）
        intraday_range = abs(sigma_adj) * np.sqrt(dt) * (0.5 + rng.random())
        highs[t] = max(opens[t], S[t]) * (1 + intraday_range * rng.random())
        lows[t] = min(opens[t], S[t]) * (1 - intraday_range * rng.random())

        # 成交量（随机）
        volumes[t] = rng.integers(1e6, 1e8)

    df = pd.DataFrame({
        "open": opens, "high": highs, "low": lows, "close": S, "volume": volumes,
    }, index=dates)
    return df


def generate_simulation_data_v2(
    n_years: int = 5,
    seed: int = 42,
    start_date: str = "2021-01-04",
) -> tuple:
    """生成10个模拟ETF的日线OHLCV数据

    Returns:
        (close_df, ohlcv_dict, market_states)
        close_df: 收盘价DataFrame, index=date, columns=ETF代码
        ohlcv_dict: {ETF代码: DataFrame[open,high,low,close,volume]}
        market_states: 市场状态序列
    """
    n_days = n_years * 252
    dates = pd.bdate_range(start=start_date, periods=n_days)
    market_states = generate_market_states(n_days, seed=seed)

    ohlcv_dict = {}
    close_data = {}

    for i, (code, params) in enumerate(SIMULATED_ETFS_V2.items()):
        df = generate_etf_prices_v2(
            params, n_days, market_states,
            seed=seed + i * 100, dates=dates,
        )
        ohlcv_dict[code] = df
        close_data[code] = df["close"]

    close_df = pd.DataFrame(close_data, index=dates)
    close_df.index.name = "date"

    return close_df, ohlcv_dict, market_states


if __name__ == "__main__":
    print("=" * 60)
    print("阶段2：10个模拟ETF数据生成器")
    print("=" * 60)

    close_df, ohlcv_dict, states = generate_simulation_data_v2(n_years=5, seed=42)

    print(f"\n生成数据: {close_df.shape[0]} 交易日, {close_df.shape[1]} ETF")
    print(f"日期范围: {close_df.index[0].date()} ~ {close_df.index[-1].date()}")

    print("\n各ETF统计:")
    for code in close_df.columns:
        close = close_df[code]
        total_ret = close.iloc[-1] / close.iloc[0] - 1
        years = len(close) / 252
        annual_ret = (1 + total_ret) ** (1 / years) - 1
        daily_ret = close.pct_change().dropna()
        annual_vol = daily_ret.std() * np.sqrt(252)
        print(f"  {code} ({SIMULATED_ETFS_V2[code]['name']}): "
              f"年化={annual_ret*100:.2f}%, 波动={annual_vol*100:.2f}%")

    # 相关性矩阵
    print("\n相关性矩阵（前5×5）:")
    corr = close_df.pct_change().corr()
    print(corr.iloc[:5, :5].round(2))
