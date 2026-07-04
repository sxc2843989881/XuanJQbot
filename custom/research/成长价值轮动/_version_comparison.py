"""V72/V74/V75 多版本公平对比脚本（QBot引擎）

V72 逻辑 = V74 逻辑 + MA75择时
  - 在 V74 双因子(f1=0.5比价MA20 + f2=5.0动量加速度) + 零波动率覆盖基础上
    加回 MA75 择时：当 g_close<MA75*0.97 AND v_close<MA75*0.97 时降仓到10%(cut=0.1)
  - MA75 触发时降仓到10%(非空仓)，方向保持当前持仓

公平对比四版本：
  - V72:          V74 + MA75择时(降仓10%)
  - V74:          V72双因子 + 零波动率覆盖（无MA75）
  - V75:          V74 + 3日动量门控
  - V72_no_ma75:  V72逻辑但去掉MA75 (验证定位：应≈V74)

全部使用 QBot 回测引擎（T+1开盘执行、跳空滑点、手续费）。
V72 的10%降仓通过 run_backtest_engine_weighted 实现（复用QBot引擎核心逻辑+权重维度）。
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# 路径配置
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent
STYLE_ROTATION_DIR = Path(r"c:\caches\sxc\style_rotation_strategy")
DATA_DIR = STYLE_ROTATION_DIR / "data"
OUTPUT_CSV = SCRIPT_DIR / "version_comparison.csv"

sys.path.insert(0, str(STYLE_ROTATION_DIR))
from backtest_module.backtest_engine import (
    BacktestInput, BacktestConfig, run_backtest_engine, run_backtest_engine_weighted
)


# ============================================================
# 1. 数据加载
# ============================================================
print("=" * 80)
print("V72/V74/V75 多版本公平对比（QBot引擎）")
print("=" * 80)

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

print(f"\n[数据] 共 {len(g_close)} 个交易日")
print(f"  区间: {g_close.index[0]:%Y-%m-%d} ~ {g_close.index[-1]:%Y-%m-%d}")


# ============================================================
# 2. 因子计算 — 全部shift(1)防未来函数
# ============================================================
print("\n[因子层] V72双因子(f1+f2) + 63日斜率×R² + MA75 + 3日动量")

# V72参数（基于逻辑，非搜索拟合）
F1 = 0.5
F2 = 5.0
MA_W = 75       # MA75择时窗口
MA_PCT = 0.97   # 跌破阈值
CUT = 0.1       # 降仓到10%

# ---- 因子1：比价MA20偏离（V72核心）----
ratio = (g_close / v_close).shift(1)
ratio_ma20 = ratio.rolling(20).mean()
ratio_dev = ratio / ratio_ma20 - 1
f1_signal = np.tanh(ratio_dev * 30) * F1
style_score = f1_signal.copy()

# ---- 因子2：动量加速度（V72核心）----
g_roc21 = g_close.pct_change(21).shift(1)
v_roc21 = v_close.pct_change(21).shift(1)
g_accel = g_roc21 - g_roc21.shift(10)
v_accel = v_roc21 - v_roc21.shift(10)
accel_diff = (g_accel - v_accel).clip(-0.02, 0.02)
f2_signal = accel_diff * F2
style_score = style_score + f2_signal

# ---- 辅助：63日对数价格回归斜率×R²（零波动率覆盖用）----
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

# ---- MA75择时（V72用）----
g_ma75 = g_close.shift(1).rolling(MA_W).mean()
v_ma75 = v_close.shift(1).rolling(MA_W).mean()
both_below = (g_close.shift(1) < g_ma75 * MA_PCT) & (v_close.shift(1) < v_ma75 * MA_PCT)

# ---- V75: 3日短期动量（门控用）----
g_mom3 = g_close.pct_change(3).shift(1)
v_mom3 = v_close.pct_change(3).shift(1)

print(f"  style_score: mean={style_score.mean():.4f}, std={style_score.std():.4f}")
print(f"  both_below(MA75触发)周数占比: {both_below.resample('W-FRI').last().mean()*100:.1f}%")


# ============================================================
# 3. 周频采样（W-FRI）
# ============================================================
print("\n[采样层] 周频W-FRI")

df = pd.DataFrame({
    "g_close": g_close, "v_close": v_close,
    "g_open": g_open, "v_open": v_open,
    "ratio_dev": ratio_dev,
    "f1_signal": f1_signal, "f2_signal": f2_signal,
    "style_score": style_score,
    "smom_g": smom_g, "smom_v": smom_v,
    "both_below": both_below,
    "g_mom3": g_mom3, "v_mom3": v_mom3,
})

df_wk = df.resample("W-FRI").last().dropna(
    subset=["style_score", "smom_g", "smom_v"]
).iloc[1:]

MIN_HOLD = 4  # 最小持有期4周


# ============================================================
# 4. 信号生成函数
# ============================================================
def generate_v74_signals(df_wk):
    """V74: V72双因子 + 零波动率覆盖（无MA75）
    返回: (position_series[1.0/0.0], weight_series, cover_count)
    """
    position = pd.Series(np.nan, index=df_wk.index)
    weight = pd.Series(1.0, index=df_wk.index)
    current_pos = None
    hold_weeks = 0
    cover_count = 0

    for i in range(len(df_wk)):
        row = df_wk.iloc[i]
        score_i = row["style_score"]
        smom_g_i = row["smom_g"]
        smom_v_i = row["smom_v"]

        if score_i > 0:
            target = "growth"
        else:
            if current_pos == "growth" and smom_g_i > 0 and smom_v_i <= 0:
                target = "growth"
                cover_count += 1
            elif current_pos is None:
                target = "value" if smom_v_i > smom_g_i else "growth"
            else:
                target = "value"

        if current_pos is None:
            current_pos = target
            hold_weeks = 1
        elif target != current_pos and hold_weeks >= MIN_HOLD:
            current_pos = target
            hold_weeks = 1
        else:
            hold_weeks += 1

        position.iloc[i] = 1.0 if current_pos == "growth" else 0.0
        weight.iloc[i] = 1.0

    return position, weight, cover_count


def generate_v72_signals(df_wk, use_ma75=True):
    """V72: V74逻辑 + MA75择时(降仓10%)
    MA75触发时降仓到10%，方向保持当前持仓。
    返回: (position_series, weight_series, cover_count, ma75_count)
    """
    position = pd.Series(np.nan, index=df_wk.index)
    weight = pd.Series(1.0, index=df_wk.index)
    current_pos = None
    hold_weeks = 0
    cover_count = 0
    ma75_count = 0

    for i in range(len(df_wk)):
        row = df_wk.iloc[i]
        score_i = row["style_score"]
        smom_g_i = row["smom_g"]
        smom_v_i = row["smom_v"]
        both_below_i = bool(row["both_below"]) if use_ma75 else False

        # 先按V74逻辑决定方向
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

        # MA75择时（最高优先级）
        if both_below_i:
            ma75_count += 1
            target_weight = CUT  # 0.1 降仓
            target_pos = current_pos if current_pos is not None else decided_pos
        else:
            target_weight = 1.0
            target_pos = decided_pos

        # 状态机：最小持有期（只对方向变化生效，权重变化不生效）
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


def generate_v75_signals(df_wk):
    """V75: V74 + 3日动量门控（仅拦截growth->value，要求v_mom3>g_mom3才切换）
    返回: (position_series, weight_series, cover_count, gate_block_count)
    """
    position = pd.Series(np.nan, index=df_wk.index)
    weight = pd.Series(1.0, index=df_wk.index)
    current_pos = None
    hold_weeks = 0
    cover_count = 0
    gate_block = 0

    for i in range(len(df_wk)):
        row = df_wk.iloc[i]
        score_i = row["style_score"]
        smom_g_i = row["smom_g"]
        smom_v_i = row["smom_v"]
        g_mom3_i = row["g_mom3"]
        v_mom3_i = row["v_mom3"]
        cover_this_iter = False

        if score_i > 0:
            target = "growth"
        else:
            if current_pos == "growth" and smom_g_i > 0 and smom_v_i <= 0:
                target = "growth"
                cover_count += 1
                cover_this_iter = True
            elif current_pos is None:
                target = "value" if smom_v_i > smom_g_i else "growth"
            else:
                target = "value"

        if current_pos is None:
            current_pos = target
            hold_weeks = 1
            position.iloc[i] = 1.0 if target == "growth" else 0.0
        elif target != current_pos and hold_weeks >= MIN_HOLD:
            gate_pass = True
            if not cover_this_iter and target == "value" and current_pos == "growth":
                if pd.isna(g_mom3_i) or pd.isna(v_mom3_i):
                    gate_pass = True
                else:
                    gate_pass = v_mom3_i > g_mom3_i
            if not gate_pass:
                gate_block += 1
                hold_weeks += 1
                position.iloc[i] = 1.0 if current_pos == "growth" else 0.0
            else:
                current_pos = target
                hold_weeks = 1
                position.iloc[i] = 1.0 if target == "growth" else 0.0
        else:
            hold_weeks += 1
            position.iloc[i] = 1.0 if current_pos == "growth" else 0.0

    return position, weight, cover_count, gate_block


# ============================================================
# 5. 信号→日频 + 回测
# ============================================================
def signals_to_daily(df, df_wk, position, weight, version_label):
    """周频信号扩展回日频，构造 BacktestInput + position_weight"""
    df_wk_use = df_wk.copy()
    df_wk_use["signal_wk"] = position.map({1.0: "growth", 0.0: "value"})
    df_wk_use["weight_wk"] = weight.values

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
    print(f"  [{version_label}] 回测区间: {df_bt.index[0]:%Y-%m-%d} ~ {df_bt.index[-1]:%Y-%m-%d} "
          f"({len(df_bt)} 天), 降仓周数(权重<1): {(pos_weight < 1.0).sum()}")
    return bt_input, pos_weight, df_bt


def run_version(df, df_wk, position, weight, version_label, use_weighted=False):
    """运行单版本回测"""
    bt_input, pos_weight, df_bt = signals_to_daily(df, df_wk, position, weight, version_label)
    config = BacktestConfig(commission=0.0001, impact_slippage=0.0, apply_gap_slippage=True)

    if use_weighted:
        result = run_backtest_engine_weighted(bt_input, config, pos_weight)
    else:
        result = run_backtest_engine(bt_input, config)
    return result, df_bt


# ============================================================
# 6. 生成信号并回测
# ============================================================
print("\n[决策层] 生成4版本信号")

# V74
pos_v74, w_v74, cover_v74 = generate_v74_signals(df_wk)
# V72 (V74 + MA75)
pos_v72, w_v72, cover_v72, ma75_v72 = generate_v72_signals(df_wk, use_ma75=True)
# V72_no_ma75 (验证定位)
pos_v72_nm, w_v72_nm, cover_v72_nm, ma75_v72_nm = generate_v72_signals(df_wk, use_ma75=False)
# V75
pos_v75, w_v75, cover_v75, gate_v75 = generate_v75_signals(df_wk)

print(f"  V74:           持成长{ (pos_v74==1.0).mean()*100:.1f}%, 覆盖{cover_v74}次")
print(f"  V72(含MA75):   持成长{ (pos_v72==1.0).mean()*100:.1f}%, 覆盖{cover_v72}次, MA75触发{ma75_v72}周")
print(f"  V72_no_ma75:   持成长{ (pos_v72_nm==1.0).mean()*100:.1f}%, 覆盖{cover_v72_nm}次, MA75触发{ma75_v72_nm}周")
print(f"  V75:           持成长{ (pos_v75==1.0).mean()*100:.1f}%, 覆盖{cover_v75}次, 门控拦截{gate_v75}次")

# 信号一致性验证：V72_no_ma75 的信号应该等于 V74
sig_match = (pos_v72_nm == pos_v74).all()
weight_match = (w_v72_nm == w_v74).all()
print(f"\n  [验证] V72_no_ma75 信号 == V74 信号?  position: {sig_match}, weight: {weight_match}")

print("\n[回测] 调用QBot引擎")
result_v74, df_bt_v74 = run_version(df, df_wk, pos_v74, w_v74, "V74", use_weighted=False)
result_v72, df_bt_v72 = run_version(df, df_wk, pos_v72, w_v72, "V72", use_weighted=True)
result_v72nm, df_bt_v72nm = run_version(df, df_wk, pos_v72_nm, w_v72_nm, "V72_no_ma75", use_weighted=False)
result_v75, df_bt_v75 = run_version(df, df_wk, pos_v75, w_v75, "V75", use_weighted=False)

# V72_no_ma75 用weighted引擎(全1权重)再跑一次，确保引擎一致
result_v72nm_w, _ = run_version(df, df_wk, pos_v72_nm, w_v72_nm, "V72_no_ma75(weighted)", use_weighted=True)


# ============================================================
# 7. 零波动率场景测试（持成长占比）
# ============================================================
print("\n[零波动率场景] 测试各版本持成长占比")

def calc_zero_vol_growth_pct(df_wk_input, position_col):
    """零波动率场景下持成长占比"""
    growth_pct = (df_wk_input[position_col] == 1.0).mean() * 100
    return growth_pct


# 构造零波动率场景（成长直线匀速上涨）
log_start = np.log(g_close.iloc[0])
log_end = np.log(g_close.iloc[-1])
log_prices = np.linspace(log_start, log_end, len(g_close))
g_zero = pd.Series(np.exp(log_prices), index=g_close.index)

ratio_zero = (g_zero / v_close).shift(1)
ratio_ma20_zero = ratio_zero.rolling(20).mean()
ratio_dev_zero = ratio_zero / ratio_ma20_zero - 1
f1_zero = np.tanh(ratio_dev_zero * 30) * F1
style_zero = f1_zero.copy()

g_roc21_zero = g_zero.pct_change(21).shift(1)
g_accel_zero = g_roc21_zero - g_roc21_zero.shift(10)
v_roc21_zero = v_close.pct_change(21).shift(1)
v_accel_zero = v_roc21_zero - v_roc21_zero.shift(10)
accel_diff_zero = (g_accel_zero - v_accel_zero).clip(-0.02, 0.02)
f2_zero = accel_diff_zero * F2
style_zero = style_zero + f2_zero

slope_g_zero, r2_g_zero = rolling_slope_r2(g_zero.shift(1), 63)
slope_v_zero, r2_v_zero = rolling_slope_r2(v_close.shift(1), 63)
smom_g_zero = slope_g_zero * r2_g_zero
smom_v_zero = slope_v_zero * r2_v_zero

g_mom3_zero = g_zero.pct_change(3).shift(1)
v_mom3_zero = v_close.pct_change(3).shift(1)

g_ma75_zero = g_zero.shift(1).rolling(MA_W).mean()
both_below_zero = (g_zero.shift(1) < g_ma75_zero * MA_PCT) & (v_close.shift(1) < v_ma75 * MA_PCT)

df_zero = pd.DataFrame({
    "g_close": g_zero, "v_close": v_close,
    "style_score": style_zero,
    "smom_g": smom_g_zero, "smom_v": smom_v_zero,
    "both_below": both_below_zero,
    "g_mom3": g_mom3_zero, "v_mom3": v_mom3_zero,
})
df_zero_wk = df_zero.resample("W-FRI").last().dropna(
    subset=["style_score", "smom_g", "smom_v"]
).iloc[1:]

# 零波动率场景下各版本信号
pos_v74_z, _, cover_v74_z = generate_v74_signals(df_zero_wk)
pos_v72_z, _, cover_v72_z, ma75_v72_z = generate_v72_signals(df_zero_wk, use_ma75=True)
pos_v72nm_z, _, _, _ = generate_v72_signals(df_zero_wk, use_ma75=False)
pos_v75_z, _, cover_v75_z, gate_v75_z = generate_v75_signals(df_zero_wk)

zero_pct = {
    "V72": (pos_v72_z == 1.0).mean() * 100,
    "V74": (pos_v74_z == 1.0).mean() * 100,
    "V75": (pos_v75_z == 1.0).mean() * 100,
    "V72_no_ma75": (pos_v72nm_z == 1.0).mean() * 100,
}
print(f"  零波动率场景持成长占比: V72={zero_pct['V72']:.1f}%, V74={zero_pct['V74']:.1f}%, "
      f"V75={zero_pct['V75']:.1f}%, V72_no_ma75={zero_pct['V72_no_ma75']:.1f}%")


# ============================================================
# 8. 指标汇总
# ============================================================
print("\n" + "=" * 80)
print("多版本指标对比表")
print("=" * 80)

def get_metrics_row(result, label, factor_desc, note, cover_count):
    m = result.metrics
    return {
        "版本": label,
        "因子构成": factor_desc,
        "年化收益": m["annual_ret"],
        "Sharpe": m["sharpe"],
        "最大回撤": m["max_dd"],
        "Calmar": m["calmar"],
        "调仓次数": m["num_trades"],
        "零波动率持成长%": zero_pct[label],
        "覆盖触发次数": cover_count,
        "备注": note,
    }

rows = []
rows.append(get_metrics_row(
    result_v72, "V72",
    "f1(0.5)+f2(5.0)+零波动率覆盖+MA75择时(降仓10%)",
    "MA75触发降仓10%", cover_v72
))
rows.append(get_metrics_row(
    result_v74, "V74",
    "f1(0.5)+f2(5.0)+零波动率覆盖",
    "无MA75", cover_v74
))
rows.append(get_metrics_row(
    result_v75, "V75",
    "V74+3日动量门控(拦截g->v)",
    f"门控拦截{gate_v75}次", cover_v75
))
rows.append(get_metrics_row(
    result_v72nm, "V72_no_ma75",
    "f1(0.5)+f2(5.0)+零波动率覆盖(去MA75)",
    "验证定位:应≈V74", cover_v72_nm
))

metrics_df = pd.DataFrame(rows)

# 打印表格
print(f"\n{'版本':<14} {'因子构成':<40} {'年化收益':>9} {'Sharpe':>8} {'最大回撤':>9} {'Calmar':>8} {'调仓':>5} {'零波持成长%':>11}")
print("-" * 110)
for _, r in metrics_df.iterrows():
    print(f"{r['版本']:<14} {r['因子构成']:<40} {r['年化收益']*100:>8.2f}% {r['Sharpe']:>8.3f} "
          f"{r['最大回撤']*100:>8.2f}% {r['Calmar']:>8.3f} {r['调仓次数']:>5} {r['零波动率持成长%']:>10.1f}%")

print(f"\n{'版本':<14} {'备注':<30} {'覆盖触发':>8}")
print("-" * 60)
for _, r in metrics_df.iterrows():
    print(f"{r['版本']:<14} {r['备注']:<30} {r['覆盖触发次数']:>8}")

# 保存CSV
metrics_df_csv = metrics_df.copy()
metrics_df_csv["年化收益"] = (metrics_df_csv["年化收益"] * 100).round(2).astype(str) + "%"
metrics_df_csv["最大回撤"] = (metrics_df_csv["最大回撤"] * 100).round(2).astype(str) + "%"
metrics_df_csv["Sharpe"] = metrics_df_csv["Sharpe"].round(3)
metrics_df_csv["Calmar"] = metrics_df_csv["Calmar"].round(3)
metrics_df_csv["零波动率持成长%"] = metrics_df_csv["零波动率持成长%"].round(1)
metrics_df_csv.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
print(f"\n  指标表已保存: {OUTPUT_CSV}")


# ============================================================
# 9. V72 vs V74 差异分析
# ============================================================
print("\n" + "=" * 80)
print("V72 vs V74 差异分析（确认MA75导致）")
print("=" * 80)

m_v72 = result_v72.metrics
m_v74 = result_v74.metrics
m_v75 = result_v75.metrics
m_v72nm = result_v72nm.metrics
m_v72nm_w = result_v72nm_w.metrics

print(f"\n  {'指标':<12} {'V72':>10} {'V74':>10} {'V72-V74':>10} {'解读':<30}")
print(f"  {'-'*75}")
print(f"  {'年化收益':<12} {m_v72['annual_ret']*100:>9.2f}% {m_v74['annual_ret']*100:>9.2f}% "
      f"{(m_v72['annual_ret']-m_v74['annual_ret'])*100:>+9.2f}pp {'MA75降仓影响收益':<30}")
print(f"  {'Sharpe':<12} {m_v72['sharpe']:>10.3f} {m_v74['sharpe']:>10.3f} "
      f"{m_v72['sharpe']-m_v74['sharpe']:>+10.3f} {'':<30}")
print(f"  {'最大回撤':<12} {m_v72['max_dd']*100:>9.2f}% {m_v74['max_dd']*100:>9.2f}% "
      f"{(m_v72['max_dd']-m_v74['max_dd'])*100:>+9.2f}pp {'MA75降仓应降低回撤':<30}")
print(f"  {'Calmar':<12} {m_v72['calmar']:>10.3f} {m_v74['calmar']:>10.3f} "
      f"{m_v72['calmar']-m_v74['calmar']:>+10.3f} {'':<30}")
print(f"  {'调仓次数':<12} {m_v72['num_trades']:>10} {m_v74['num_trades']:>10} "
      f"{m_v72['num_trades']-m_v74['num_trades']:>+10} {'MA75触发额外调仓':<30}")

print(f"\n  V72 MA75触发周数: {ma75_v72}周, 占比{ma75_v72/len(df_wk)*100:.1f}%")
print(f"  V72 降仓(权重<1)交易日数: {(w_v72 < 1.0).sum()}周")


# ============================================================
# 10. V72_no_ma75 ≈ V74 验证
# ============================================================
print("\n" + "=" * 80)
print("V72_no_ma75 ≈ V74 验证（定位退化原因）")
print("=" * 80)

print(f"\n  信号一致性: position {'一致' if sig_match else '不一致'}, "
      f"weight {'一致' if weight_match else '不一致'}")
print(f"\n  {'指标':<12} {'V72_no_ma75':>14} {'V74':>14} {'差异':>14} {'是否一致':<10}")
print(f"  {'-'*65}")
for key, label in [("annual_ret", "年化收益"), ("sharpe", "Sharpe"),
                   ("max_dd", "最大回撤"), ("calmar", "Calmar"),
                   ("num_trades", "调仓次数"), ("final_multiple", "最终倍率")]:
    v_nm = m_v72nm[key]
    v_74 = m_v74[key]
    diff = v_nm - v_74
    if key in ("annual_ret", "max_dd"):
        print(f"  {label:<12} {v_nm*100:>13.4f}% {v_74*100:>13.4f}% {diff*100:>+13.4f}pp "
              f"{'是' if abs(diff) < 1e-6 else '否':<10}")
    elif key == "num_trades":
        print(f"  {label:<12} {v_nm:>14} {v_74:>14} {diff:>+14} "
              f"{'是' if diff == 0 else '否':<10}")
    else:
        print(f"  {label:<12} {v_nm:>14.6f} {v_74:>14.6f} {diff:>+14.6f} "
              f"{'是' if abs(diff) < 1e-6 else '否':<10}")

# weighted引擎(全1权重) vs 原引擎 一致性验证
print(f"\n  [引擎验证] weighted(全1权重) vs 原引擎 (V72_no_ma75):")
print(f"    年化: weighted={m_v72nm_w['annual_ret']*100:.4f}% vs 原引擎={m_v72nm['annual_ret']*100:.4f}% "
      f"(差{(m_v72nm_w['annual_ret']-m_v72nm['annual_ret'])*100:.6f}pp)")
print(f"    调仓: weighted={m_v72nm_w['num_trades']} vs 原引擎={m_v72nm['num_trades']}")


# ============================================================
# 11. 年度收益对比
# ============================================================
print("\n" + "=" * 80)
print("年度收益对比 (2013-2025)")
print("=" * 80)

def annual_returns(result, df_bt):
    rdf = result.to_dataframe()
    rdf["date"] = pd.to_datetime(rdf["date"])
    rdf = rdf.set_index("date")
    rdf["year"] = rdf.index.year
    return rdf.groupby("year")["daily_ret"].apply(lambda x: (1 + x).prod() - 1)

ann_v72 = annual_returns(result_v72, df_bt_v72)
ann_v74 = annual_returns(result_v74, df_bt_v74)
ann_v75 = annual_returns(result_v75, df_bt_v75)
ann_v72nm = annual_returns(result_v72nm, df_bt_v72nm)

# 基准
g_close_bt = g_close.loc[df_bt_v74.index[0]:]
v_close_bt = v_close.loc[df_bt_v74.index[0]:]
g_annual = g_close_bt.resample("Y").last().pct_change().dropna()
v_annual = v_close_bt.resample("Y").last().pct_change().dropna()
g_annual.index = g_annual.index.year
v_annual.index = v_annual.index.year

years = sorted(set(ann_v72.index) | set(ann_v74.index) | set(ann_v75.index))
print(f"\n  {'年份':<6} {'V72':>9} {'V74':>9} {'V75':>9} {'V72_nm':>9} {'成长100':>9} {'价值100':>9} {'V72-V74':>9}")
print(f"  {'-'*75}")
for y in years:
    v72_r = ann_v72.get(y, np.nan) * 100
    v74_r = ann_v74.get(y, np.nan) * 100
    v75_r = ann_v75.get(y, np.nan) * 100
    v72nm_r = ann_v72nm.get(y, np.nan) * 100
    g_r = g_annual.get(y, np.nan) * 100
    v_r = v_annual.get(y, np.nan) * 100
    diff = (ann_v72.get(y, 0) - ann_v74.get(y, 0)) * 100
    print(f"  {y:<6} {v72_r:>8.2f}% {v74_r:>8.2f}% {v75_r:>8.2f}% {v72nm_r:>8.2f}% "
          f"{g_r:>8.2f}% {v_r:>8.2f}% {diff:>+8.2f}pp")

# 年度收益CSV
ann_df = pd.DataFrame({
    "年份": years,
    "V72(%)": [ann_v72.get(y, np.nan) * 100 for y in years],
    "V74(%)": [ann_v74.get(y, np.nan) * 100 for y in years],
    "V75(%)": [ann_v75.get(y, np.nan) * 100 for y in years],
    "V72_no_ma75(%)": [ann_v72nm.get(y, np.nan) * 100 for y in years],
    "成长100(%)": [g_annual.get(y, np.nan) * 100 for y in years],
    "价值100(%)": [v_annual.get(y, np.nan) * 100 for y in years],
    "V72-V74差(pp)": [(ann_v72.get(y, 0) - ann_v74.get(y, 0)) * 100 for y in years],
})
ann_df = ann_df.round(2)
ann_csv = SCRIPT_DIR / "version_annual_returns.csv"
ann_df.to_csv(ann_csv, index=False, encoding="utf-8-sig")
print(f"\n  年度收益表已保存: {ann_csv}")


# ============================================================
# 12. 总结
# ============================================================
print("\n" + "=" * 80)
print("总结")
print("=" * 80)
print(f"  1. V72 vs V74: MA75触发{ma75_v72}周(占比{ma75_v72/len(df_wk)*100:.1f}%), "
      f"年化差{(m_v72['annual_ret']-m_v74['annual_ret'])*100:+.2f}pp, "
      f"回撤差{(m_v72['max_dd']-m_v74['max_dd'])*100:+.2f}pp")
print(f"  2. V72_no_ma75 ≈ V74: 信号{'一致' if sig_match else '不一致'}, "
      f"年化差{(m_v72nm['annual_ret']-m_v74['annual_ret'])*100:+.6f}pp")
print(f"  3. V75 门控拦截{gate_v75}次, 年化{(m_v75['annual_ret'])*100:.2f}%")
print(f"  4. 零波动率场景持成长: V72={zero_pct['V72']:.1f}%, V74={zero_pct['V74']:.1f}%, "
      f"V75={zero_pct['V75']:.1f}%")
print(f"\n  文件:")
print(f"    指标对比表: {OUTPUT_CSV}")
print(f"    年度收益表: {ann_csv}")
print(f"    QBot引擎扩展: {STYLE_ROTATION_DIR / 'backtest_module' / 'backtest_engine.py'} (新增run_backtest_engine_weighted)")
print("=" * 80)
