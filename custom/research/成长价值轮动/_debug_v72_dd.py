"""Debug: check V72 max DD period consistency"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

STYLE_ROTATION_DIR = Path(r"c:\caches\sxc\style_rotation_strategy")
DATA_DIR = STYLE_ROTATION_DIR / "data"
sys.path.insert(0, str(STYLE_ROTATION_DIR))
from backtest_module.backtest_engine import (
    BacktestInput, BacktestConfig,
    run_backtest_engine, run_backtest_engine_weighted,
)

# 数据加载
g_raw = pd.read_csv(DATA_DIR / "index_480080.csv")
v_raw = pd.read_csv(DATA_DIR / "index_480081.csv")
for d in (g_raw, v_raw):
    d["date"] = pd.to_datetime(d["date"])
    d["close"] = pd.to_numeric(d["close"], errors="coerce")
    if "open" in d.columns:
        d["open"] = pd.to_numeric(d["open"], errors="coerce")

g_df = g_raw.set_index("date").sort_index()
v_df = v_raw.set_index("date").sort_index()
common = g_df.index.intersection(v_df.index)
g_df = g_df.loc[common]
v_df = v_df.loc[common]

g_close = g_df["close"].astype(float)
v_close = v_df["close"].astype(float)
g_open = g_df["open"].astype(float) if "open" in g_df.columns else g_close.shift(1).fillna(g_close)
v_open = v_df["open"].astype(float) if "open" in v_df.columns else v_close.shift(1).fillna(v_close)
g_open = g_open.fillna(g_close.shift(1)).fillna(g_close)
v_open = v_open.fillna(v_close.shift(1)).fillna(v_close)
valid = ~(g_close.isna() | v_close.isna())
g_close, v_close, g_open, v_open = g_close[valid], v_close[valid], g_open[valid], v_open[valid]

# 因子
F1, F2, MA_W, MA_PCT, CUT, MIN_HOLD = 0.5, 5.0, 75, 0.97, 0.1, 4

ratio = (g_close / v_close).shift(1)
ratio_ma20 = ratio.rolling(20).mean()
ratio_dev = ratio / ratio_ma20 - 1
f1_signal = np.tanh(ratio_dev * 30) * F1
style_score = f1_signal.copy()

g_roc21 = g_close.pct_change(21).shift(1)
v_roc21 = v_close.pct_change(21).shift(1)
g_accel = g_roc21 - g_roc21.shift(10)
v_accel = v_roc21 - v_roc21.shift(10)
accel_diff = (g_accel - v_accel).clip(-0.02, 0.02)
f2_signal = accel_diff * F2
style_score = style_score + f2_signal

def rolling_slope_r2(close_series, window=63):
    y = np.log(close_series).astype(float)
    n = window
    x = np.arange(n).astype(float)
    x_mean = x.mean()
    x_var = ((x - x_mean) ** 2).sum()
    sum_xy = y.rolling(n).apply(lambda yw: np.sum(yw * x), raw=True)
    sum_y = y.rolling(n).sum()
    sum_y2 = y.rolling(n).apply(lambda yw: np.sum(yw * yw), raw=True)
    cov_num = sum_xy - n * x_mean * (sum_y / n)
    slope = cov_num / x_var
    y_var = sum_y2 - (sum_y ** 2) / n
    r2 = (slope ** 2 * x_var) / y_var.where(y_var != 0, np.nan)
    r2 = r2.clip(0, 1)
    return slope, r2

slope_g, r2_g = rolling_slope_r2(g_close.shift(1), 63)
slope_v, r2_v = rolling_slope_r2(v_close.shift(1), 63)
smom_g = slope_g * r2_g
smom_v = slope_v * r2_v

g_ma75 = g_close.shift(1).rolling(MA_W).mean()
v_ma75 = v_close.shift(1).rolling(MA_W).mean()
both_below = (g_close.shift(1) < g_ma75 * MA_PCT) & (v_close.shift(1) < v_ma75 * MA_PCT)

df = pd.DataFrame({
    "g_close": g_close, "v_close": v_close,
    "g_open": g_open, "v_open": v_open,
    "ratio_dev": ratio_dev,
    "f1_signal": f1_signal, "f2_signal": f2_signal,
    "style_score": style_score,
    "smom_g": smom_g, "smom_v": smom_v,
    "both_below": both_below,
})
df_wk = df.resample("W-FRI").last().dropna(subset=["style_score", "smom_g", "smom_v"]).iloc[1:]


def generate_v72_signals(df_wk_in, use_ma75=True):
    position = pd.Series(np.nan, index=df_wk_in.index)
    weight = pd.Series(1.0, index=df_wk_in.index)
    current_pos = None
    hold_weeks = 0
    cover_count = 0
    ma75_count = 0
    for i in range(len(df_wk_in)):
        row = df_wk_in.iloc[i]
        score_i = row["style_score"]
        smom_g_i = row["smom_g"]
        smom_v_i = row["smom_v"]
        both_below_i = bool(row["both_below"]) if use_ma75 else False
        if score_i > 0:
            decided_pos = "growth"
        else:
            if current_pos == "growth" and smom_g_i > 0 and smom_v_i <= 0:
                decided_pos = "growth"
                cover_count += 1
            elif current_pos is None:
                decided_pos = "value" if smom_v_i > smom_g_i else "growth"
            else:
                decided_pos = "value"
        if both_below_i:
            ma75_count += 1
            target_weight = CUT
            target_pos = current_pos if current_pos is not None else decided_pos
        else:
            target_weight = 1.0
            target_pos = decided_pos
        if current_pos is None:
            current_pos = target_pos
            hold_weeks = 1
        elif target_pos != current_pos and hold_weeks >= MIN_HOLD:
            current_pos = target_pos
            hold_weeks = 1
        else:
            hold_weeks += 1
        position.iloc[i] = 1.0 if current_pos == "growth" else 0.0
        weight.iloc[i] = target_weight
    return position, weight, cover_count, ma75_count


pos_v72, w_v72, cover_v72, ma75_v72 = generate_v72_signals(df_wk, use_ma75=True)

# 信号→日频
df_wk_use = df_wk.copy()
df_wk_use["signal_wk"] = pos_v72.map({1.0: "growth", 0.0: "value"})
df_wk_use["weight_wk"] = w_v72.values

df_daily = df.reset_index().rename(columns={"index": "date"}).sort_values("date")
df_wk_reset = df_wk_use.reset_index().rename(columns={"index": "date"}).sort_values("date")[
    ["date", "signal_wk", "weight_wk"]
]
df_daily = pd.merge_asof(df_daily, df_wk_reset, on="date", direction="backward")
df_daily = df_daily.set_index("date").sort_index()
df_bt = df_daily.dropna(subset=["signal_wk"]).copy()

bt_input = BacktestInput(
    dates=df_bt.index.strftime("%Y-%m-%d").values,
    value_open=df_bt["v_open"].values.astype(np.float64),
    value_close=df_bt["v_close"].values.astype(np.float64),
    growth_open=df_bt["g_open"].values.astype(np.float64),
    growth_close=df_bt["g_close"].values.astype(np.float64),
    signal=df_bt["signal_wk"].values.astype(str),
)
pos_weight = df_bt["weight_wk"].values.astype(np.float64)

config = BacktestConfig(commission=0.0001, impact_slippage=0.0, apply_gap_slippage=True)
result_v72 = run_backtest_engine_weighted(bt_input, config, pos_weight)

print(f"\n=== V72 NAV 分析 ===")
print(f"Backtest 开始日: {df_bt.index[0]:%Y-%m-%d}")
print(f"Backtest 结束日: {df_bt.index[-1]:%Y-%m-%d}")
print(f"总天数: {len(df_bt)}")
print(f"NAV[0] = {result_v72.nav[0]:,.0f}")
print(f"NAV[-1] = {result_v72.nav[-1]:,.0f}")
print(f"Engine max_dd = {result_v72.metrics['max_dd']*100:.4f}%")

nav_v72 = pd.Series(result_v72.nav, index=pd.to_datetime(result_v72.dates))

# 全局 min/max
global_min_idx = nav_v72.idxmin()
global_max_idx = nav_v72.idxmax()
print(f"\n全局最小 NAV: {nav_v72.loc[global_min_idx]:,.0f} @ {global_min_idx:%Y-%m-%d}")
print(f"全局最大 NAV: {nav_v72.loc[global_max_idx]:,.0f} @ {global_max_idx:%Y-%m-%d}")

# Peak before trough
peak_before_min = nav_v72.loc[:global_min_idx].idxmax()
print(f"全局最小之前的 Peak: {nav_v72.loc[peak_before_min]:,.0f} @ {peak_before_min:%Y-%m-%d}")
print(f"Global max DD = {(nav_v72.loc[global_min_idx] - nav_v72.loc[peak_before_min])/nav_v72.loc[peak_before_min]*100:.4f}%")

# 2013 年内分析
nav_2013 = nav_v72[nav_v72.index.year == 2013]
print(f"\n=== 2013 年内分析 ===")
print(f"2013 数据点数: {len(nav_2013)}")
if len(nav_2013) > 0:
    print(f"2013 第一天: {nav_2013.index[0]:%Y-%m-%d}, NAV={nav_2013.iloc[0]:,.0f}")
    print(f"2013 最后一天: {nav_2013.index[-1]:%Y-%m-%d}, NAV={nav_2013.iloc[-1]:,.0f}")
    print(f"2013 NAV min: {nav_2013.min():,.0f} @ {nav_2013.idxmin():%Y-%m-%d}")
    print(f"2013 NAV max: {nav_2013.max():,.0f} @ {nav_2013.idxmax():%Y-%m-%d}")
    peak_2013 = nav_2013.cummax()
    dd_2013 = (nav_2013 - peak_2013) / peak_2013
    print(f"2013 within-year max DD: {dd_2013.min()*100:.4f}%")
    print(f"  at trough: {dd_2013.idxmin():%Y-%m-%d}, NAV={nav_2013.loc[dd_2013.idxmin()]:,.0f}")
    print(f"  at peak: {nav_2013.loc[:dd_2013.idxmin()].idxmax():%Y-%m-%d}, NAV={nav_2013.loc[:dd_2013.idxmin()].max():,.0f}")

# 2015 年内分析
nav_2015 = nav_v72[nav_v72.index.year == 2015]
print(f"\n=== 2015 年内分析 ===")
print(f"2015 数据点数: {len(nav_2015)}")
if len(nav_2015) > 0:
    print(f"2015 第一天: {nav_2015.index[0]:%Y-%m-%d}, NAV={nav_2015.iloc[0]:,.0f}")
    print(f"2015 最后一天: {nav_2015.index[-1]:%Y-%m-%d}, NAV={nav_2015.iloc[-1]:,.0f}")
    peak_2015 = nav_2015.cummax()
    dd_2015 = (nav_2015 - peak_2015) / peak_2015
    print(f"2015 within-year max DD: {dd_2015.min()*100:.4f}%")
    print(f"  at trough: {dd_2015.idxmin():%Y-%m-%d}, NAV={nav_2015.loc[dd_2015.idxmin()]:,.0f}")

# 2015-2016 跨年分析
nav_2015_2016 = nav_v72[(nav_v72.index >= "2015-01-01") & (nav_v72.index <= "2016-12-31")]
peak_1516 = nav_2015_2016.cummax()
dd_1516 = (nav_2015_2016 - peak_1516) / peak_1516
print(f"\n=== 2015-2016 跨年分析 ===")
print(f"2015-2016 数据点数: {len(nav_2015_2016)}")
print(f"2015-2016 max DD: {dd_1516.min()*100:.4f}%")
print(f"  at trough: {dd_1516.idxmin():%Y-%m-%d}, NAV={nav_2015_2016.loc[dd_1516.idxmin()]:,.0f}")
print(f"  at peak: {nav_2015_2016.loc[:dd_1516.idxmin()].idxmax():%Y-%m-%d}, NAV={nav_2015_2016.loc[:dd_1516.idxmin()].max():,.0f}")
