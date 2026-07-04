"""阶段3：20只模拟ETF数据生成器（动态上市日期）

模拟真实ETF市场的关键特征：
1. 20只ETF覆盖7大类资产
2. 每只ETF有不同的上市日期（2007-2019年）
3. 回测从2013年开始，初始只有部分ETF存在
4. 动态候选池：ETF上市后自动加入候选

模拟ETF参考真实ETF的上市时间和参数特征：
- 早期ETF（2007-2011）：宽基、红利、消费等
- 中期ETF（2012-2015）：行业主题、医药、海外等
- 晚期ETF（2016-2019）：科技、科创板、新兴产业等
"""
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# 复用阶段1的市场状态机制
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "v1_3etf"))
from data_generator import generate_market_states, MARKET_STATES, TRANSITION_MATRIX


# ============================================================
# 20个模拟ETF参数（覆盖7类资产，不同上市日期）
# ============================================================

SIMULATED_ETFS_V3 = {
    # === 早期ETF（2007-2011上市，2013年回测开始时已存在）===
    "ETF-RED":  {"name": "模拟红利ETF",       "category": "价值",   "mu": 0.07, "sigma": 0.17, "theta": 0.2,  "init_price": 1.0, "list_date": "2007-01-15"},
    "ETF-HS300":{"name": "模拟沪深300ETF",    "category": "宽基",   "mu": 0.09, "sigma": 0.21, "theta": 0.35, "init_price": 1.0, "list_date": "2007-08-01"},
    "ETF-CONS": {"name": "模拟消费ETF",       "category": "消费",   "mu": 0.08, "sigma": 0.18, "theta": 0.25, "init_price": 1.0, "list_date": "2010-05-20"},
    "ETF-UTIL": {"name": "模拟公用事业ETF",   "category": "防御",   "mu": 0.06, "sigma": 0.15, "theta": 0.15, "init_price": 1.0, "list_date": "2010-11-08"},
    "ETF-FIN":  {"name": "模拟金融ETF",       "category": "金融",   "mu": 0.08, "sigma": 0.24, "theta": 0.3,  "init_price": 1.0, "list_date": "2011-03-15"},
    "ETF-BOND": {"name": "模拟国债ETF",       "category": "债券",   "mu": 0.04, "sigma": 0.05, "theta": 0.05, "init_price": 1.0, "list_date": "2011-06-30"},

    # === 中期ETF（2012-2015上市，2013年部分存在）===
    "ETF-MED":  {"name": "模拟医药ETF",       "category": "医药",   "mu": 0.09, "sigma": 0.23, "theta": 0.3,  "init_price": 1.0, "list_date": "2012-08-15"},
    "ETF-CYB":  {"name": "模拟创业板ETF",     "category": "成长",   "mu": 0.11, "sigma": 0.28, "theta": 0.45, "init_price": 1.0, "list_date": "2013-01-01"},
    "ETF-ENRG": {"name": "模拟能源ETF",       "category": "周期",   "mu": 0.07, "sigma": 0.26, "theta": 0.35, "init_price": 1.0, "list_date": "2013-06-10"},
    "ETF-MIL":  {"name": "模拟军工ETF",       "category": "主题",   "mu": 0.10, "sigma": 0.27, "theta": 0.4,  "init_price": 1.0, "list_date": "2014-03-20"},
    "ETF-REIT": {"name": "模拟地产ETF",       "category": "金融",   "mu": 0.07, "sigma": 0.22, "theta": 0.25, "init_price": 1.0, "list_date": "2014-09-01"},
    "ETF-OVS":  {"name": "模拟海外ETF",       "category": "海外",   "mu": 0.11, "sigma": 0.20, "theta": 0.35, "init_price": 1.0, "list_date": "2013-11-15"},

    # === 晚期ETF（2016-2019上市，2013年时不存在）===
    "ETF-TECH": {"name": "模拟科技ETF",       "category": "科技",   "mu": 0.13, "sigma": 0.26, "theta": 0.5,  "init_price": 1.0, "list_date": "2016-04-15"},
    "ETF-COMP":{"name": "模拟计算机ETF",     "category": "科技",   "mu": 0.14, "sigma": 0.28, "theta": 0.55, "init_price": 1.0, "list_date": "2017-02-10"},
    "ETF-EV":   {"name": "模拟新能源ETF",     "category": "主题",   "mu": 0.15, "sigma": 0.32, "theta": 0.55, "init_price": 1.0, "list_date": "2018-01-20"},
    "ETF-CHIP": {"name": "模拟芯片ETF",       "category": "科技",   "mu": 0.16, "sigma": 0.33, "theta": 0.6,  "init_price": 1.0, "list_date": "2018-06-15"},
    "ETF-COMD": {"name": "模拟商品ETF",       "category": "周期",   "mu": 0.06, "sigma": 0.24, "theta": 0.3,  "init_price": 1.0, "list_date": "2016-10-08"},
    "ETF-ENVR": {"name": "模拟环保ETF",       "category": "主题",   "mu": 0.08, "sigma": 0.25, "theta": 0.35, "init_price": 1.0, "list_date": "2017-09-15"},
    "ETF-SCI":  {"name": "模拟科创ETF",       "category": "成长",   "mu": 0.14, "sigma": 0.30, "theta": 0.5,  "init_price": 1.0, "list_date": "2019-07-22"},
    "ETF-500":  {"name": "模拟中证500ETF",    "category": "宽基",   "mu": 0.10, "sigma": 0.25, "theta": 0.4,  "init_price": 1.0, "list_date": "2015-05-15"},
}


def generate_etf_prices_v3(
    params: dict,
    n_days: int,
    market_states: np.ndarray,
    seed: int,
    momentum_window: int = 20,
    dates: pd.DatetimeIndex = None,
    list_date: str = None,
) -> pd.DataFrame:
    """生成单个ETF的OHLCV数据（带上市日期）

    模型：S_t = S_{t-1} × exp( (μ + θ·M_t - 0.5·σ²)·dt + σ·dW_t )

    上市日期前数据为NaN，模拟ETF未上市
    """
    rng = np.random.default_rng(seed)
    dt = 1 / 252

    mu = params["mu"]
    sigma = params["sigma"]
    theta = params["theta"]

    S = np.zeros(n_days)
    daily_returns = np.zeros(n_days)

    highs = np.zeros(n_days)
    lows = np.zeros(n_days)
    opens = np.zeros(n_days)
    volumes = np.zeros(n_days)

    # 上市日期前的数据设为NaN
    list_dt = pd.Timestamp(list_date) if list_date else dates[0]
    mask_listed = dates >= list_dt

    # 找到上市第一天
    first_listed_idx = np.argmax(mask_listed) if mask_listed.any() else 0

    S[0] = params["init_price"]
    highs[0] = S[0] * 1.005
    lows[0] = S[0] * 0.995
    opens[0] = S[0]

    for t in range(1, n_days):
        state = market_states[t]
        state_params = MARKET_STATES[state]
        mu_adj = mu + state_params["mu_shift"]
        sigma_adj = sigma * state_params["sigma_mult"]

        if t >= momentum_window:
            M_t = np.mean(daily_returns[t - momentum_window : t])
        else:
            M_t = np.mean(daily_returns[:t]) if t > 0 else 0

        dW = rng.standard_normal() * np.sqrt(dt)
        drift = (mu_adj + theta * M_t - 0.5 * sigma_adj ** 2) * dt
        diffusion = sigma_adj * dW
        daily_ret = drift + diffusion

        opens[t] = S[t - 1]
        S[t] = S[t - 1] * np.exp(daily_ret)
        daily_returns[t] = daily_ret

        intraday_range = abs(sigma_adj) * np.sqrt(dt) * (0.5 + rng.random())
        highs[t] = max(opens[t], S[t]) * (1 + intraday_range * rng.random())
        lows[t] = min(opens[t], S[t]) * (1 - intraday_range * rng.random())
        volumes[t] = rng.integers(1e6, 1e8)

    df = pd.DataFrame({
        "open": opens, "high": highs, "low": lows, "close": S, "volume": volumes,
    }, index=dates)

    # 上市日期前设为NaN
    df.loc[~mask_listed] = np.nan

    return df


def generate_simulation_data_v3(
    n_years: int = 13,
    seed: int = 42,
    start_date: str = "2013-01-04",
) -> tuple:
    """生成20个模拟ETF的日线OHLCV数据（动态上市）

    Args:
        n_years: 回测年数（2013-2026 = 13年）
        seed: 随机种子
        start_date: 回测开始日期

    Returns:
        (close_df, ohlcv_dict, market_states, list_dates)
        close_df: 收盘价DataFrame（未上市的为NaN）
        ohlcv_dict: {ETF代码: DataFrame}
        market_states: 市场状态序列
        list_dates: {ETF代码: 上市日期}
    """
    n_days = n_years * 252
    dates = pd.bdate_range(start=start_date, periods=n_days)
    market_states = generate_market_states(n_days, seed=seed)

    ohlcv_dict = {}
    close_data = {}
    list_dates = {}

    for i, (code, params) in enumerate(SIMULATED_ETFS_V3.items()):
        df = generate_etf_prices_v3(
            params, n_days, market_states,
            seed=seed + i * 100, dates=dates,
            list_date=params["list_date"],
        )
        ohlcv_dict[code] = df
        close_data[code] = df["close"]
        list_dates[code] = params["list_date"]

    close_df = pd.DataFrame(close_data, index=dates)
    close_df.index.name = "date"

    return close_df, ohlcv_dict, market_states, list_dates


def get_dynamic_universe(close_df: pd.DataFrame, list_dates: dict) -> pd.DataFrame:
    """获取动态候选池：每个日期哪些ETF已上市

    Args:
        close_df: 收盘价DataFrame（未上市的为NaN）
        list_dates: {ETF代码: 上市日期}

    Returns:
        DataFrame: True=已上市可交易, False=未上市
    """
    universe = pd.DataFrame(True, index=close_df.index, columns=close_df.columns)
    for code, list_date in list_dates.items():
        list_dt = pd.Timestamp(list_date)
        universe.loc[universe.index < list_dt, code] = False
    return universe


def print_universe_evolution(close_df: pd.DataFrame, list_dates: dict):
    """打印候选池随时间的演变"""
    universe = get_dynamic_universe(close_df, list_dates)

    print("\n候选池演变:")
    print(f"{'年份':<6} {'已上市ETF数':<12} {'新上市ETF':<30}")
    print("-" * 55)

    prev_codes = set()
    for year in range(close_df.index[0].year, close_df.index[-1].year + 1):
        year_end = close_df[close_df.index.year == year].index[-1]
        current_codes = set(universe.columns[universe.loc[year_end]])
        new_codes = current_codes - prev_codes
        new_names = ", ".join(sorted(new_codes)) if new_codes else "—"
        print(f"{year:<6} {len(current_codes):<12} {new_names:<30}")
        prev_codes = current_codes


if __name__ == "__main__":
    print("=" * 70)
    print("阶段3：20只模拟ETF数据生成器（动态上市日期）")
    print("=" * 70)

    close_df, ohlcv_dict, states, list_dates = generate_simulation_data_v3(n_years=13, seed=42)

    print(f"\n生成数据: {close_df.shape[0]} 交易日, {close_df.shape[1]} ETF")
    print(f"日期范围: {close_df.index[0].date()} ~ {close_df.index[-1].date()}")

    print_universe_evolution(close_df, list_dates)

    print("\n各ETF统计（已上市部分）:")
    for code in close_df.columns:
        close = close_df[code].dropna()
        if len(close) < 100:
            continue
        total_ret = close.iloc[-1] / close.iloc[0] - 1
        years = len(close) / 252
        annual_ret = (1 + total_ret) ** (1 / years) - 1 if years > 0 else 0
        daily_ret = close.pct_change().dropna()
        annual_vol = daily_ret.std() * np.sqrt(252)
        list_date = list_dates[code]
        print(f"  {code:<10} ({SIMULATED_ETFS_V3[code]['name']:<12}) "
              f"上市:{list_date} | 年化={annual_ret*100:.2f}% | 波动={annual_vol*100:.2f}% | "
              f"数据{len(close)}天")
