"""V72 vs V74 最大回撤差异验证脚本（10.85pp 差异是否真实合理）

核心问题:
  V72 (V74 + MA75 降仓10%) 最大回撤 -42.46%
  V74 (无 MA75)            最大回撤 -53.30%
  差异 10.85pp —— 是真实择时效果，还是算法假象?

验证方法:
  1. 复现 MA75 触发日期，对比 V74 最大回撤期（对齐分析）
  2. 安慰剂测试 A: 随机触发(9.5%率) 重复10次取平均 — 排除算法假象
  3. 安慰剂测试 B: 牛市触发(涨幅最大10%周) — 验证时机重要性
  4. 逐年分析: 每年 MA75 触发周数 + V72 vs V74 回撤差异
  5. 数值合理性: 预期回撤差异 = V74回撤期跌幅 × 90% × MA75触发占比

依赖:
  - QBot 引擎: c:\\caches\\sxc\\style_rotation_strategy\\backtest_module\\backtest_engine.py
  - 数据: c:\\caches\\sxc\\style_rotation_strategy\\data\\index_480080.csv, index_480081.csv
  - run_backtest_engine_weighted 实现 V72 降仓 (已验证全1权重等价原引擎)
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
STYLE_ROTATION_DIR = Path(r"c:\caches\sxc\style_rotation_strategy")
DATA_DIR = STYLE_ROTATION_DIR / "data"
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "_verify_output"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

sys.path.insert(0, str(STYLE_ROTATION_DIR))
from backtest_module.backtest_engine import (
    BacktestInput, BacktestConfig,
    run_backtest_engine, run_backtest_engine_weighted,
)

# 固定随机种子，保证可复现
GLOBAL_SEED = 20260701


# ============================================================
# 1. 数据加载 + 因子计算（与 _version_comparison.py 完全一致）
# ============================================================
print("=" * 80)
print("V72 vs V74 最大回撤差异验证（10.85pp 差异是否真实合理）")
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

# V72/V74 参数
F1 = 0.5
F2 = 5.0
MA_W = 75
MA_PCT = 0.97
CUT = 0.1
MIN_HOLD = 4

# ---- 因子1：比价MA20偏离 ----
ratio = (g_close / v_close).shift(1)
ratio_ma20 = ratio.rolling(20).mean()
ratio_dev = ratio / ratio_ma20 - 1
f1_signal = np.tanh(ratio_dev * 30) * F1
style_score = f1_signal.copy()

# ---- 因子2：动量加速度 ----
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

# ---- MA75择时触发条件 ----
g_ma75 = g_close.shift(1).rolling(MA_W).mean()
v_ma75 = v_close.shift(1).rolling(MA_W).mean()
both_below = (g_close.shift(1) < g_ma75 * MA_PCT) & (v_close.shift(1) < v_ma75 * MA_PCT)


# ============================================================
# 2. 周频采样 + 信号生成（与 _version_comparison.py 完全一致）
# ============================================================
df = pd.DataFrame({
    "g_close": g_close, "v_close": v_close,
    "g_open": g_open, "v_open": v_open,
    "ratio_dev": ratio_dev,
    "f1_signal": f1_signal, "f2_signal": f2_signal,
    "style_score": style_score,
    "smom_g": smom_g, "smom_v": smom_v,
    "both_below": both_below,
})

df_wk = df.resample("W-FRI").last().dropna(
    subset=["style_score", "smom_g", "smom_v"]
).iloc[1:]


def generate_v74_signals(df_wk_in):
    """V74: V72双因子 + 零波动率覆盖（无MA75）
    返回: (position[1.0/0.0], weight[全1.0], cover_count)
    """
    position = pd.Series(np.nan, index=df_wk_in.index)
    weight = pd.Series(1.0, index=df_wk_in.index)
    current_pos = None
    hold_weeks = 0
    cover_count = 0

    for i in range(len(df_wk_in)):
        row = df_wk_in.iloc[i]
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


def generate_v72_signals(df_wk_in, use_ma75=True):
    """V72: V74逻辑 + MA75择时(降仓10%)
    返回: (position, weight, cover_count, ma75_count)
    """
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

        # MA75择时（最高优先级）
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


# ============================================================
# 3. 信号→日频 + 回测（与 _version_comparison.py 完全一致）
# ============================================================
def signals_to_daily(df_in, df_wk_in, position, weight):
    """周频信号扩展回日频，构造 BacktestInput + position_weight"""
    df_wk_use = df_wk_in.copy()
    df_wk_use["signal_wk"] = position.map({1.0: "growth", 0.0: "value"})
    df_wk_use["weight_wk"] = weight.values

    df_daily = df_in.reset_index().rename(columns={"index": "date"}).sort_values("date")
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
    return bt_input, pos_weight, df_bt


def run_version(df_in, df_wk_in, position, weight, use_weighted=False, label=""):
    bt_input, pos_weight, df_bt = signals_to_daily(df_in, df_wk_in, position, weight)
    config = BacktestConfig(commission=0.0001, impact_slippage=0.0, apply_gap_slippage=True)

    if use_weighted:
        result = run_backtest_engine_weighted(bt_input, config, pos_weight)
    else:
        result = run_backtest_engine(bt_input, config)
    return result, df_bt


# ============================================================
# 4. 复现 V74 + V72 + V72_no_ma75，验证数字对齐
# ============================================================
print("\n[复现] 生成 V74 / V72 / V72_no_ma75 信号并回测...")

pos_v74, w_v74, cover_v74 = generate_v74_signals(df_wk)
pos_v72, w_v72, cover_v72, ma75_v72 = generate_v72_signals(df_wk, use_ma75=True)
pos_v72nm, w_v72nm, cover_v72nm, ma75_v72nm = generate_v72_signals(df_wk, use_ma75=False)

print(f"  V74:           持成长{(pos_v74==1.0).mean()*100:.1f}%, 覆盖{cover_v74}次")
print(f"  V72(含MA75):   持成长{(pos_v72==1.0).mean()*100:.1f}%, 覆盖{cover_v72}次, MA75触发{ma75_v72}周 "
      f"({ma75_v72/len(df_wk)*100:.1f}%)")
print(f"  V72_no_ma75:   持成长{(pos_v72nm==1.0).mean()*100:.1f}%, 覆盖{cover_v72nm}次, MA75触发{ma75_v72nm}周")

# 信号一致性验证
sig_match = (pos_v72nm == pos_v74).all()
weight_match = (w_v72nm == w_v74).all()
print(f"\n  [验证] V72_no_ma75 信号 == V74 信号?  position: {sig_match}, weight: {weight_match}")

# 运行回测
result_v74, df_bt_v74 = run_version(df, df_wk, pos_v74, w_v74, use_weighted=False, label="V74")
result_v72, df_bt_v72 = run_version(df, df_wk, pos_v72, w_v72, use_weighted=True, label="V72")
result_v72nm, df_bt_v72nm = run_version(df, df_wk, pos_v72nm, w_v72nm, use_weighted=False, label="V72_no_ma75")
result_v72nm_w, _ = run_version(df, df_wk, pos_v72nm, w_v72nm, use_weighted=True, label="V72_no_ma75_w")

m_v72 = result_v72.metrics
m_v74 = result_v74.metrics
m_v72nm = result_v72nm.metrics
m_v72nm_w = result_v72nm_w.metrics

print(f"\n  {'指标':<12} {'V72':>10} {'V74':>10} {'V72_no_ma75':>14} {'V72_no_ma75_w':>16}")
print(f"  {'-'*68}")
for key, lbl in [("annual_ret", "年化收益"), ("sharpe", "Sharpe"),
                 ("max_dd", "最大回撤"), ("calmar", "Calmar"),
                 ("num_trades", "调仓次数")]:
    if key in ("annual_ret", "max_dd"):
        print(f"  {lbl:<12} {m_v72[key]*100:>9.4f}% {m_v74[key]*100:>9.4f}% "
              f"{m_v72nm[key]*100:>13.4f}% {m_v72nm_w[key]*100:>15.4f}%")
    elif key == "num_trades":
        print(f"  {lbl:<12} {m_v72[key]:>10} {m_v74[key]:>10} "
              f"{m_v72nm[key]:>14} {m_v72nm_w[key]:>16}")
    else:
        print(f"  {lbl:<12} {m_v72[key]:>10.4f} {m_v74[key]:>10.4f} "
              f"{m_v72nm[key]:>14.4f} {m_v72nm_w[key]:>16.4f}")

# 关键差异
diff_dd = (m_v72["max_dd"] - m_v74["max_dd"]) * 100
print(f"\n  V72 - V74 最大回撤差异: {diff_dd:+.4f}pp")
print(f"  V72_no_ma75 - V74 最大回撤差异: {(m_v72nm['max_dd']-m_v74['max_dd'])*100:+.8f}pp (应为0)")
print(f"  V72_no_ma75_w - V74 最大回撤差异: {(m_v72nm_w['max_dd']-m_v74['max_dd'])*100:+.8f}pp (应为0，验证引擎一致)")


# ============================================================
# 5. 任务1: MA75 触发日期 vs V74 回撤期对齐分析
# ============================================================
print("\n" + "=" * 80)
print("任务1: MA75 触发日期 vs V74 回撤期对齐分析")
print("=" * 80)

# 取出 MA75 触发的周（基于 weekly df_wk 的 both_below 列）
trigger_weeks = df_wk[df_wk["both_below"] == True].index
print(f"\n  MA75 触发周数(weekly): {len(trigger_weeks)} 周")
print(f"  MA75 触发率: {len(trigger_weeks)/len(df_wk)*100:.2f}%")

# 列出所有触发周
print(f"\n  MA75 触发周列表（前20个 + 后5个）:")
print(f"  {'序号':>4} {'周日期':>12}")
print(f"  {'-'*20}")
for i, d in enumerate(trigger_weeks):
    if i < 20 or i >= len(trigger_weeks) - 5:
        print(f"  {i+1:>4} {d:%Y-%m-%d}")
    elif i == 20:
        print(f"  ... (省略中间 {len(trigger_weeks)-25} 周) ...")

# ---- V74 最大回撤期（peak → trough）----
nav_v74 = pd.Series(result_v74.nav, index=pd.to_datetime(result_v74.dates))
peak_v74 = nav_v74.cummax()
dd_v74 = (nav_v74 - peak_v74) / peak_v74

# 找到最大回撤的 peak 和 trough
trough_idx = dd_v74.idxmin()
peak_idx = nav_v74.loc[:trough_idx].idxmax()
peak_nav = nav_v74.loc[peak_idx]
trough_nav = nav_v74.loc[trough_idx]
max_dd_v74 = dd_v74.loc[trough_idx]

print(f"\n  V74 最大回撤期:")
print(f"    Peak  日: {peak_idx:%Y-%m-%d}  NAV={peak_nav:,.0f}")
print(f"    Trough日: {trough_idx:%Y-%m-%d}  NAV={trough_nav:,.0f}")
print(f"    最大回撤: {max_dd_v74*100:.2f}%")
print(f"    回撤持续: {(trough_idx - peak_idx).days} 天")

# ---- V72 最大回撤期（用 running-peak 法，与引擎一致）----
nav_v72 = pd.Series(result_v72.nav, index=pd.to_datetime(result_v72.dates))
running_max_v72 = nav_v72.cummax()
dd_v72_series = (nav_v72 - running_max_v72) / running_max_v72
trough_idx_v72 = dd_v72_series.idxmin()
peak_idx_v72 = nav_v72.loc[:trough_idx_v72].idxmax()

print(f"\n  V72 自己的最大回撤期（running-peak 法，与引擎一致）:")
print(f"    Peak  日: {peak_idx_v72:%Y-%m-%d}  NAV={nav_v72.loc[peak_idx_v72]:,.0f}")
print(f"    Trough日: {trough_idx_v72:%Y-%m-%d}  NAV={nav_v72.loc[trough_idx_v72]:,.0f}")
print(f"    最大回撤: {m_v72['max_dd']*100:.4f}%  (引擎报告 {m_v72['max_dd']*100:.4f}%)")

print(f"\n  V72 vs V74 max_dd 期对比:")
print(f"    V74: Peak={peak_idx:%Y-%m-%d}, Trough={trough_idx:%Y-%m-%d}, DD={m_v74['max_dd']*100:.2f}%, "
      f"持续{(trough_idx-peak_idx).days}天")
print(f"    V72: Peak={peak_idx_v72:%Y-%m-%d}, Trough={trough_idx_v72:%Y-%m-%d}, DD={m_v72['max_dd']*100:.2f}%, "
      f"持续{(trough_idx_v72-peak_idx_v72).days}天")
print(f"    Peak 同日? {peak_idx == peak_idx_v72}  "
      f"(V72 trough 比 V74 晚 {(trough_idx_v72 - trough_idx).days} 天)")

# 同时列出 V72 在 V74 回撤期的表现（同 peak/trough 窗口）
v72_at_v74_peak = nav_v72.loc[:peak_idx].max()
v72_at_v74_trough = nav_v72.loc[trough_idx]
v72_dd_in_v74_period = (v72_at_v74_trough - v72_at_v74_peak) / v72_at_v74_peak if v72_at_v74_peak > 0 else 0
print(f"\n  V74 回撤期窗口内 V72 表现（同 peak/trough 日）:")
print(f"    V72 在 V74 peak 时的最大值: {v72_at_v74_peak:,.0f}")
print(f"    V72 在 V74 trough 时的值:   {v72_at_v74_trough:,.0f}")
print(f"    V72 在该窗口回撤: {v72_dd_in_v74_period*100:.2f}%")
print(f"    差异: V74回撤 - V72该窗口回撤 = {(max_dd_v74 - v72_dd_in_v74_period)*100:+.2f}pp")

# ---- 检查 MA75 触发周是否集中在 V74 回撤期 ----
trigger_in_dd = trigger_weeks[(trigger_weeks >= peak_idx) & (trigger_weeks <= trough_idx)]
trigger_out_dd = trigger_weeks[(trigger_weeks < peak_idx) | (trigger_weeks > trough_idx)]
print(f"\n  MA75 触发周 vs V74 回撤期重叠:")
print(f"    V74 回撤期内触发:  {len(trigger_in_dd)} 周  ({len(trigger_in_dd)/len(trigger_weeks)*100:.1f}% of triggers)")
print(f"    V74 回撤期外触发:  {len(trigger_out_dd)} 周  ({len(trigger_out_dd)/len(trigger_weeks)*100:.1f}% of triggers)")

# 用更宽的口径：V74 回撤期 + 前后 3 个月（择时可能领先/滞后）
from pandas.tseries.offsets import DateOffset
window_start = peak_idx - DateOffset(months=3)
window_end = trough_idx + DateOffset(months=3)
trigger_near_dd = trigger_weeks[(trigger_weeks >= window_start) & (trigger_weeks <= window_end)]
print(f"\n  扩展窗口（peak前3月 ~ trough后3月）内触发: {len(trigger_near_dd)} 周 "
      f"({len(trigger_near_dd)/len(trigger_weeks)*100:.1f}%)")

# 扩展到 V72 max_dd 期（trough 比 V74 晚 ~3个月，到 2016-05-11）
print(f"\n  注：V72 的 max_dd 期 trough 在 {trough_idx_v72:%Y-%m-%d}（比 V74 晚 {(trough_idx_v72-trough_idx).days} 天）")
trigger_in_v72_dd = trigger_weeks[(trigger_weeks >= peak_idx_v72) & (trigger_weeks <= trough_idx_v72)]
print(f"    V72 max_dd 期内触发: {len(trigger_in_v72_dd)} 周 "
      f"({len(trigger_in_v72_dd)/len(trigger_weeks)*100:.1f}% of triggers)")

# 把所有触发周按年份分布打印
trigger_year_dist = pd.Series(trigger_weeks.year).value_counts().sort_index()
print(f"\n  MA75 触发周按年份分布:")
print(f"  {'年份':>6} {'触发周数':>8}")
print(f"  {'-'*18}")
for y, c in trigger_year_dist.items():
    print(f"  {y:>6} {c:>8}")

# 把 V74 回撤期也按年标注
print(f"\n  V74 最大回撤期跨越年份: {peak_idx.year} ~ {trough_idx.year}")


# ============================================================
# 6. 任务2: 安慰剂测试
# ============================================================
print("\n" + "=" * 80)
print("任务2: 安慰剂测试（关键！排除算法假象）")
print("=" * 80)

# 构造"假触发"权重数组：
#   保留 V74 的方向信号不变，只把 weight 在某些周降到 CUT=0.1
#   假触发的周通过随机或牛市条件选择，触发率与 MA75 相同
n_wk = len(df_wk)
n_trigger_target = int(ma75_v72)  # MA75 实际触发周数
target_rate = n_trigger_target / n_wk
print(f"\n  安慰剂测试参数:")
print(f"    目标触发周数: {n_trigger_target} (与 MA75 一致)")
print(f"    目标触发率: {target_rate*100:.2f}%")
print(f"    触发时降仓: {CUT*100:.0f}%")
print(f"    重复次数: 10 (取平均)")

# V74 方向信号 → 周频 → 用 weighted 引擎跑（weight=1.0 时等价 V74）
# 我们已在 V72_no_ma75_w 验证 weighted 引擎与 V74 引擎等价
v74_dir_pos = pos_v74.copy()  # 方向信号固定为 V74

def run_with_fake_trigger(trigger_mask, pos_dir=pos_v74, label=""):
    """用假触发掩码构造 weight 数组并跑 weighted 引擎。

    Args:
        trigger_mask: bool Series, index=df_wk.index, True=触发周(降仓到CUT)
        pos_dir: 方向信号 (1.0=成长, 0.0=价值)
    Returns:
        BacktestResult
    """
    # 构造 weight: 触发周 CUT, 非触发周 1.0
    # 注意：V72 真实逻辑里，触发周 direction = 当前持仓(不变)，weight=CUT
    #       这里安慰剂保持方向信号不变，只改 weight
    weight = pd.Series(1.0, index=df_wk.index)
    weight[trigger_mask] = CUT

    # 用 V74 方向信号 + 假 weight 跑 weighted 引擎
    result, _ = run_version(df, df_wk, pos_dir, weight, use_weighted=True)
    return result


# ---- 测试A: 随机触发 ----
print(f"\n[测试A] 随机触发（9.5%触发率，固定seed，10次取平均）")
print(f"  {'-'*70}")
np.random.seed(GLOBAL_SEED)

random_test_results = []
random_test_details = []
for trial in range(10):
    # 在 weekly index 上随机选 n_trigger_target 个周作为触发
    rng = np.random.RandomState(GLOBAL_SEED + trial)
    all_idx = np.arange(n_wk)
    rng.shuffle(all_idx)
    trigger_indices = sorted(all_idx[:n_trigger_target])
    trigger_mask = pd.Series(False, index=df_wk.index)
    trigger_mask.iloc[trigger_indices] = True

    n_actual = trigger_mask.sum()
    result = run_with_fake_trigger(trigger_mask, pos_v74, label=f"random_{trial}")
    m = result.metrics
    random_test_results.append({
        "trial": trial + 1,
        "n_trigger": int(n_actual),
        "annual_ret": m["annual_ret"],
        "max_dd": m["max_dd"],
        "sharpe": m["sharpe"],
        "calmar": m["calmar"],
        "num_trades": m["num_trades"],
    })
    random_test_details.append(result)
    print(f"  trial {trial+1:>2}: 触发{n_actual}周, 年化={m['annual_ret']*100:>6.2f}%, "
          f"回撤={m['max_dd']*100:>6.2f}%, Sharpe={m['sharpe']:.3f}, Calmar={m['calmar']:.3f}, "
          f"调仓={m['num_trades']}")

# 平均
avg_random = {
    "annual_ret": np.mean([r["annual_ret"] for r in random_test_results]),
    "max_dd": np.mean([r["max_dd"] for r in random_test_results]),
    "sharpe": np.mean([r["sharpe"] for r in random_test_results]),
    "calmar": np.mean([r["calmar"] for r in random_test_results]),
    "num_trades": np.mean([r["num_trades"] for r in random_test_results]),
}
print(f"\n  随机触发平均:")
print(f"    年化={avg_random['annual_ret']*100:.2f}%, 回撤={avg_random['max_dd']*100:.2f}%, "
      f"Sharpe={avg_random['sharpe']:.3f}, Calmar={avg_random['calmar']:.3f}")
print(f"    随机触发 vs V74(回撤{m_v74['max_dd']*100:.2f}%): "
      f"回撤改善 {(avg_random['max_dd']-m_v74['max_dd'])*100:+.2f}pp")
print(f"    MA75 真实 vs V74: 回撤改善 {(m_v72['max_dd']-m_v74['max_dd'])*100:+.2f}pp")
print(f"    MA75 真实 vs 随机平均: 回撤改善差额 {(m_v72['max_dd']-avg_random['max_dd'])*100:+.2f}pp "
      f"(MA75 比随机多改善 {(avg_random['max_dd']-m_v72['max_dd'])*100:.2f}pp)")


# ---- 测试B: 牛市触发（最差时机）----
print(f"\n[测试B] 牛市触发（涨幅最大10%周触发降仓，最差时机）")
print(f"  {'-'*70}")

# 计算每周的市场涨幅（成长+价值等权）
g_weekly_ret = df_wk["g_close"].pct_change()
v_weekly_ret = df_wk["v_close"].pct_change()
mkt_weekly_ret = (g_weekly_ret + v_weekly_ret) / 2

# 选涨幅最大的 n_trigger_target 周
mkt_ret_sorted = mkt_weekly_ret.sort_values(ascending=False)
bull_trigger_indices = mkt_ret_sorted.head(n_trigger_target).index
bull_trigger_mask = pd.Series(False, index=df_wk.index)
bull_trigger_mask.loc[bull_trigger_indices] = True

print(f"  牛市触发周数: {int(bull_trigger_mask.sum())}")
print(f"  这些周的平均市场涨幅: {mkt_weekly_ret.loc[bull_trigger_indices].mean()*100:+.2f}%")
print(f"  这些周的市场涨幅范围: {mkt_weekly_ret.loc[bull_trigger_indices].min()*100:+.2f}% ~ "
      f"{mkt_weekly_ret.loc[bull_trigger_indices].max()*100:+.2f}%")

result_bull = run_with_fake_trigger(bull_trigger_mask, pos_v74, label="bull")
m_bull = result_bull.metrics
print(f"\n  牛市触发结果:")
print(f"    年化={m_bull['annual_ret']*100:.2f}%, 回撤={m_bull['max_dd']*100:.2f}%, "
      f"Sharpe={m_bull['sharpe']:.3f}, Calmar={m_bull['calmar']:.3f}, 调仓={m_bull['num_trades']}")
print(f"    牛市触发 vs V74: 回撤改善 {(m_bull['max_dd']-m_v74['max_dd'])*100:+.2f}pp")
print(f"    牛市触发 vs MA75 真实: 回撤改善差额 {(m_bull['max_dd']-m_v72['max_dd'])*100:+.2f}pp "
      f"(MA75 比牛市触发多改善 {(m_bull['max_dd']-m_v72['max_dd'])*100:.2f}pp)")


# ============================================================
# 7. 任务3: 逐年分析
# ============================================================
print("\n" + "=" * 80)
print("任务3: 逐年分析（MA75 触发周数 + V72/V74 回撤差异）")
print("=" * 80)

# 计算每年的 rolling max drawdown（在该年内）
def annual_max_drawdown(nav_series):
    """计算给定 nav 序列的最大回撤"""
    if len(nav_series) < 2:
        return 0.0, None, None
    peak = nav_series.cummax()
    dd = (nav_series - peak) / peak
    trough_idx = dd.idxmin()
    peak_idx = nav_series.loc[:trough_idx].idxmax()
    return dd.min(), peak_idx, trough_idx


# 按年聚合 nav，计算每年内最大回撤
nav_v74_series = pd.Series(result_v74.nav, index=pd.to_datetime(result_v74.dates))
nav_v72_series = pd.Series(result_v72.nav, index=pd.to_datetime(result_v72.dates))

years = sorted(set(nav_v74_series.index.year) | set(nav_v72_series.index.year))

print(f"\n  {'年份':>6} {'MA75触发':>10} {'V74年化收益':>12} {'V72年化收益':>12} "
      f"{'V74年内回撤':>12} {'V72年内回撤':>12} {'回撤差异':>10}")
print(f"  {'-'*88}")

for year in years:
    # MA75 触发周数（按年）
    n_trig_y = int((trigger_weeks.year == year).sum())

    # V74 / V72 该年 nav
    v74_year_nav = nav_v74_series[nav_v74_series.index.year == year]
    v72_year_nav = nav_v72_series[nav_v72_series.index.year == year]
    if len(v74_year_nav) < 2 or len(v72_year_nav) < 2:
        continue

    # 年内收益
    v74_year_ret = v74_year_nav.iloc[-1] / v74_year_nav.iloc[0] - 1
    v72_year_ret = v72_year_nav.iloc[-1] / v72_year_nav.iloc[0] - 1

    # 年内最大回撤（基于跨年 peak，正确做法是用全局 peak 计算年内 dd）
    # 这里用"年内 nav 序列的 cummax"作为 peak（年内最大回撤）
    v74_dd, v74_pk, v74_tr = annual_max_drawdown(v74_year_nav)
    v72_dd, v72_pk, v72_tr = annual_max_drawdown(v72_year_nav)

    diff = (v72_dd - v74_dd) * 100
    flag = " <<<" if abs(diff) > 5 else ""
    print(f"  {year:>6} {n_trig_y:>10} {v74_year_ret*100:>11.2f}% {v72_year_ret*100:>11.2f}% "
          f"{v74_dd*100:>11.2f}% {v72_dd*100:>11.2f}% {diff:>+9.2f}pp{flag}")


# ============================================================
# 8. 任务4: 数值合理性检查
# ============================================================
print("\n" + "=" * 80)
print("任务4: 数值合理性检查")
print("=" * 80)

# V74 回撤期内市场跌幅（成长 + 价值等权）
g_in_dd = g_close.loc[peak_idx:trough_idx]
v_in_dd = v_close.loc[peak_idx:trough_idx]
g_drop_in_dd = g_in_dd.iloc[-1] / g_in_dd.iloc[0] - 1
v_drop_in_dd = v_in_dd.iloc[-1] / v_in_dd.iloc[0] - 1
mkt_drop_in_dd = (g_drop_in_dd + v_drop_in_dd) / 2

# 实际上 V74 是仓位轮动，要看 V74 持仓实际跌幅
# V74 在回撤期内持仓比例（成长 vs 价值）
pos_v74_series = pd.Series(result_v74.position, index=pd.to_datetime(result_v74.dates))
pos_v74_in_dd = pos_v74_series.loc[peak_idx:trough_idx]
n_growth_dd = (pos_v74_in_dd == "growth").sum()
n_value_dd = (pos_v74_in_dd == "value").sum()
n_total_dd = len(pos_v74_in_dd)
print(f"\n  V74 回撤期内持仓分布:")
print(f"    持成长: {n_growth_dd} 天 ({n_growth_dd/n_total_dd*100:.1f}%)")
print(f"    持价值: {n_value_dd} 天 ({n_value_dd/n_total_dd*100:.1f}%)")
print(f"    总天数: {n_total_dd} 天")

# 计算回撤期内 MA75 触发占比
# 把 daily both_below 取出来
both_below_daily = df["both_below"]
both_below_in_dd = both_below_daily.loc[peak_idx:trough_idx]
n_ma75_trigger_in_dd = int(both_below_in_dd.sum())
ma75_trigger_pct_in_dd = n_ma75_trigger_in_dd / n_total_dd

print(f"\n  V74 回撤期内 MA75 触发情况（按日频）:")
print(f"    触发天数: {n_ma75_trigger_in_dd} 天 / {n_total_dd} 天 ({ma75_trigger_pct_in_dd*100:.1f}%)")

# V74 在回撤期实际跌幅（已知 max_dd_v74）
print(f"\n  V74 回撤期实际跌幅: {max_dd_v74*100:.2f}%")

# 预期回撤改善 = V74回撤期跌幅 × 90% × MA75触发占比
# 直觉：MA75 触发的日子里，V72 仓位降到 10%（少承受 90% 的损失）
#        非触发日 V72 = V74
#        所以 V72 在 V74 回撤期内的回撤应该 ≈ V74回撤 × (1 - 0.9 × 触发占比)
expected_v72_dd = max_dd_v74 * (1 - 0.9 * ma75_trigger_pct_in_dd)
expected_improvement = max_dd_v74 - expected_v72_dd

print(f"\n  预期 V72 回撤（简单线性模型）:")
print(f"    模型: V72_dd ≈ V74_dd × (1 - 0.9 × MA75触发占比)")
print(f"    = {max_dd_v74*100:.2f}% × (1 - 0.9 × {ma75_trigger_pct_in_dd:.3f})")
print(f"    = {max_dd_v74*100:.2f}% × {1 - 0.9*ma75_trigger_pct_in_dd:.3f}")
print(f"    = {expected_v72_dd*100:.2f}%")
print(f"    预期改善: {expected_improvement*100:.2f}pp")
print(f"  实际 V72 在 V74 回撤期的回撤: {v72_dd_in_v74_period*100:.2f}%")
print(f"  实际改善: {(max_dd_v74 - v72_dd_in_v74_period)*100:.2f}pp")
print(f"  模型 vs 实际差异: {(expected_improvement - (max_dd_v74 - v72_dd_in_v74_period))*100:+.2f}pp")

# 注意：V72 的全局 max_dd 期已在任务1计算（peak_idx_v72, trough_idx_v72）
# 这里直接复用，进行 MA75 触发占比分析
print(f"\n  V72 max_dd 期: {peak_idx_v72:%Y-%m-%d} ~ {trough_idx_v72:%Y-%m-%d} "
      f"({(trough_idx_v72-peak_idx_v72).days} 天)")
print(f"    V72 max_dd = {m_v72['max_dd']*100:.4f}% (引擎报告 {m_v72['max_dd']*100:.4f}%)")

# V72 自己回撤期内的 MA75 触发情况
both_below_in_v72_dd = both_below_daily.loc[peak_idx_v72:trough_idx_v72]
n_ma75_in_v72_dd = int(both_below_in_v72_dd.sum())
n_total_v72_dd = len(both_below_in_v72_dd)
ma75_pct_v72_dd = n_ma75_in_v72_dd / n_total_v72_dd
print(f"    MA75 触发: {n_ma75_in_v72_dd} 天 / {n_total_v72_dd} 天 ({ma75_pct_v72_dd*100:.1f}%)")

# V74 在 V72 max_dd 期内的回撤（更公平的窗口对比）
v74_at_v72_peak = nav_v74.loc[:peak_idx_v72].max()
v74_at_v72_trough = nav_v74.loc[trough_idx_v72]
v74_dd_in_v72_period = (v74_at_v72_trough - v74_at_v72_peak) / v74_at_v72_peak if v74_at_v72_peak > 0 else 0
print(f"\n  V72 max_dd 期内 V74 表现（同窗口对比）:")
print(f"    V74 在 V72 peak 时的最大值: {v74_at_v72_peak:,.0f}")
print(f"    V74 在 V72 trough 时的值:   {v74_at_v72_trough:,.0f}")
print(f"    V74 在该窗口回撤: {v74_dd_in_v72_period*100:.2f}%")
print(f"    V72 在该窗口回撤: {m_v72['max_dd']*100:.2f}%")
print(f"    差异（V72 - V74）: {(m_v72['max_dd'] - v74_dd_in_v72_period)*100:+.2f}pp")
print(f"    注：V72 trough 比 V74 trough 晚 {(trough_idx_v72-trough_idx).days} 天，")
print(f"         V74 在此期间可能已部分反弹，故 V74 该窗口回撤 < V74 真实 max_dd {-53.30}%")

# 在 V72 max_dd 期内的线性模型验证
expected_v72_dd_in_v72 = v74_dd_in_v72_period * (1 - 0.9 * ma75_pct_v72_dd)
expected_imp_v72 = v74_dd_in_v72_period - expected_v72_dd_in_v72
actual_imp_v72 = v74_dd_in_v72_period - m_v72["max_dd"]
print(f"\n  线性模型验证（V72 max_dd 期）:")
print(f"    模型: V72_dd ≈ V74_dd × (1 - 0.9 × MA75触发占比)")
print(f"    = {v74_dd_in_v72_period*100:.2f}% × (1 - 0.9 × {ma75_pct_v72_dd:.3f})")
print(f"    = {v74_dd_in_v72_period*100:.2f}% × {1 - 0.9*ma75_pct_v72_dd:.3f}")
print(f"    = {expected_v72_dd_in_v72*100:.2f}%")
print(f"    预期改善: {expected_imp_v72*100:.2f}pp")
print(f"    实际 V72 max_dd: {m_v72['max_dd']*100:.2f}%")
print(f"    实际改善: {actual_imp_v72*100:.2f}pp")
print(f"    模型 vs 实际差异: {(expected_imp_v72 - actual_imp_v72)*100:+.2f}pp")


# ============================================================
# 9. 汇总表
# ============================================================
print("\n" + "=" * 80)
print("汇总表：MA75 真实 vs 安慰剂测试")
print("=" * 80)

print(f"\n  {'方案':<22} {'触发率':>8} {'触发周数':>8} {'年化收益':>10} {'最大回撤':>10} "
      f"{'Sharpe':>8} {'Calmar':>8} {'回撤改善':>10}")
print(f"  {'-'*92}")
print(f"  {'V74(无MA75)':<22} {'0%':>8} {0:>8} "
      f"{m_v74['annual_ret']*100:>9.2f}% {m_v74['max_dd']*100:>9.2f}% "
      f"{m_v74['sharpe']:>8.3f} {m_v74['calmar']:>8.3f} {'(基准)':>10}")
print(f"  {'V72(MA75真实)':<22} {target_rate*100:>7.2f}% {ma75_v72:>8} "
      f"{m_v72['annual_ret']*100:>9.2f}% {m_v72['max_dd']*100:>9.2f}% "
      f"{m_v72['sharpe']:>8.3f} {m_v72['calmar']:>8.3f} "
      f"{(m_v72['max_dd']-m_v74['max_dd'])*100:>+9.2f}pp")
print(f"  {'随机触发平均(10次)':<22} {target_rate*100:>7.2f}% {n_trigger_target:>8} "
      f"{avg_random['annual_ret']*100:>9.2f}% {avg_random['max_dd']*100:>9.2f}% "
      f"{avg_random['sharpe']:>8.3f} {avg_random['calmar']:>8.3f} "
      f"{(avg_random['max_dd']-m_v74['max_dd'])*100:>+9.2f}pp")
print(f"  {'牛市触发(最差时机)':<22} {target_rate*100:>7.2f}% {n_trigger_target:>8} "
      f"{m_bull['annual_ret']*100:>9.2f}% {m_bull['max_dd']*100:>9.2f}% "
      f"{m_bull['sharpe']:>8.3f} {m_bull['calmar']:>8.3f} "
      f"{(m_bull['max_dd']-m_v74['max_dd'])*100:>+9.2f}pp")

# 单次随机测试的回撤范围（看 MA75 真实是否在分布外）
random_dds = [r["max_dd"] for r in random_test_results]
mean_random_dd = np.mean(random_dds)
std_random_dd = np.std(random_dds, ddof=1)
z_score = (m_v72['max_dd'] - mean_random_dd) / std_random_dd if std_random_dd > 0 else 0
# z>0 表示 V72 回撤 > 随机平均（即更接近0，回撤更小=更好）
print(f"\n  随机触发回撤分布: min={min(random_dds)*100:.2f}%, max={max(random_dds)*100:.2f}%, "
      f"mean={mean_random_dd*100:.2f}%, std={std_random_dd*100:.2f}pp")
print(f"  MA75 真实回撤:    {m_v72['max_dd']*100:.2f}%")
print(f"  z-score = {z_score:+.2f}  (正值=比随机平均回撤更小=MA75更有效)")
print(f"  MA75 真实回撤优于随机测试中的最好结果? "
      f"{m_v72['max_dd'] > max(random_dds)} "
      f"(MA75_dd={m_v72['max_dd']*100:.2f}% > 随机最好={max(random_dds)*100:.2f}%)")
print(f"  10次随机测试中, 有多少次比 MA75 真实更优(回撤更小): "
      f"{sum(1 for d in random_dds if d > m_v72['max_dd'])} / 10")

# 保存随机测试明细到 CSV
random_df = pd.DataFrame(random_test_results)
random_df.to_csv(OUTPUT_DIR / "placebo_random_test.csv", index=False, encoding="utf-8-sig")

# 保存触发周列表
trigger_df = pd.DataFrame({
    "trigger_week": trigger_weeks,
    "in_v74_dd_period": [(peak_idx <= d <= trough_idx) for d in trigger_weeks],
})
trigger_df.to_csv(OUTPUT_DIR / "ma75_trigger_weeks.csv", index=False, encoding="utf-8-sig")

print(f"\n  输出文件:")
print(f"    {OUTPUT_DIR / 'placebo_random_test.csv'}")
print(f"    {OUTPUT_DIR / 'ma75_trigger_weeks.csv'}")

print("\n" + "=" * 80)
print("验证脚本完成。")
print("=" * 80)
