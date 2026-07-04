"""V74 策略 — V72双因子核心 + 零波动率场景精准覆盖（QBot引擎）

设计依据：
  用户原话：
  1. "我只是让你把成长100长期稳定上涨的情况保持住"
  2. "假如MA20触发了切换到价值的信号，那说明价值也在突然上涨"
  3. "可以判断成长100和价值100的绝对动量，就是相当于是涨的斜率"
  4. "谁的上涨的斜率高，那么就持有谁"
  5. "还是以比价为主，其他的策略为辅"
  6. "使用比价因子效果比动量好——因为比价均线过滤噪声，纯动量噪声大误判多"

核心架构（基于V72成功经验 + 用户洞察）：
  1. 主驱动（V72核心，保留f1+f2，去掉MA75）：
     - 因子1（f1=0.5）：比价MA20偏离 — tanh(ratio_dev*30)*0.5
     - 因子2（f2=5.0）：动量加速度 — (g_accel-v_accel).clip(-0.02,0.02)*5.0
     - style_score = f1 + f2，>0倾向成长，<=0倾向价值

  2. 零波动率场景覆盖（NEW，精准触发）：
     - 63日对数价格回归斜率×R²（平滑动量）
     - 仅当 style_score<=0（本应切价值）且 smom_g>0 AND smom_v<=0 时覆盖持成长
     - 即：只在"成长在涨+价值真没涨"时覆盖（真正的零波动率场景特征）

  3. 周频采样（W-FRI）+ 状态机 + 最小持有期4周（V72保留）

  4. 二元全仓（V72结构，去掉MA75降仓）

参数全部有据（非拟合）：
  - ratio MA: 20日（V72核心）
  - 加速度周期: 21日/10日（V72核心）
  - 斜率窗口: 63日（A股动量1-3月上界）
  - 斜率方法: 对数价格回归斜率×R²（R²过滤伪正动量）
  - 采样: 周频W-FRI（V72已验证）
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ============================================================
# 路径配置
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent
STYLE_ROTATION_DIR = Path(r"c:\caches\sxc\style_rotation_strategy")
DATA_DIR = STYLE_ROTATION_DIR / "data"
OUTPUT_DIR = (SCRIPT_DIR.parent / "回测结果").resolve()
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

sys.path.insert(0, str(STYLE_ROTATION_DIR))
from backtest_module.backtest_engine import BacktestInput, BacktestConfig, run_backtest_engine


# ============================================================
# 1. 数据加载
# ============================================================
print("=" * 80)
print("V74 策略 — V72双因子核心 + 零波动率场景精准覆盖（QBot引擎）")
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
print("\n[因子层] V72双因子(f1+f2) + 63日斜率×R²零波动率覆盖")

# V72参数（基于逻辑，非搜索拟合）
F1 = 0.5
F2 = 5.0

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
    """滚动回归：返回 (slope, r2)
    slope: 对数价格的线性回归斜率（per day）
    r2: R²判定系数（0~1）
    """
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

print(f"  style_score 描述: mean={style_score.mean():.4f}, std={style_score.std():.4f}")
print(f"  smom_g 描述: mean={smom_g.mean()*100:.4f}%, std={smom_g.std()*100:.4f}%")
print(f"  smom_v 描述: mean={smom_v.mean()*100:.4f}%, std={smom_v.std()*100:.4f}%")


# ============================================================
# 3. 周频采样 + 状态机决策（W-FRI）
# ============================================================
print("\n[决策层] 周频采样 + 状态机 + 零波动率覆盖")

df = pd.DataFrame({
    "g_close": g_close, "v_close": v_close,
    "g_open": g_open, "v_open": v_open,
    "ratio_dev": ratio_dev,
    "f1_signal": f1_signal, "f2_signal": f2_signal,
    "style_score": style_score,
    "smom_g": smom_g, "smom_v": smom_v,
})

df_wk = df.resample("W-FRI").last().dropna(
    subset=["style_score", "smom_g", "smom_v"]
).iloc[1:]

MIN_HOLD = 4  # 最小持有期4周（约20个交易日）

position = pd.Series(np.nan, index=df_wk.index)
current_pos = None
hold_weeks = 0
cover_triggered = 0

for i in range(len(df_wk)):
    row = df_wk.iloc[i]
    score_i = row["style_score"]
    smom_g_i = row["smom_g"]
    smom_v_i = row["smom_v"]

    # 主信号：V72的f1+f2
    if score_i > 0:
        target = "growth"
    else:
        # score<=0：本应切价值，加入零波动率覆盖判断
        # 用户洞察：只在"成长在涨+价值真没涨"时覆盖
        if current_pos == "growth" and smom_g_i > 0 and smom_v_i <= 0:
            # 零波动率场景特征：成长在涨 + 价值斜率非正
            target = "growth"
            cover_triggered += 1
        elif current_pos is None:
            target = "value" if smom_v_i > smom_g_i else "growth"
        else:
            target = "value"

    # 状态机：最小持有期检查
    if current_pos is None:
        current_pos = target
        hold_weeks = 1
        position.iloc[i] = 1.0 if target == "growth" else 0.0
    elif target != current_pos and hold_weeks >= MIN_HOLD:
        current_pos = target
        hold_weeks = 1
        position.iloc[i] = 1.0 if target == "growth" else 0.0
    else:
        hold_weeks += 1
        position.iloc[i] = 1.0 if current_pos == "growth" else 0.0

df_wk["pos"] = position
df_wk = df_wk.dropna(subset=["pos"])
print(f"  周频样本数: {len(df_wk)}")
print(f"  持成长周数: {(df_wk['pos']==1.0).sum()} ({(df_wk['pos']==1.0).mean()*100:.1f}%)")
print(f"  持价值周数: {(df_wk['pos']==0.0).sum()} ({(df_wk['pos']==0.0).mean()*100:.1f}%)")
print(f"  零波动率覆盖触发次数: {cover_triggered}")


# ============================================================
# 4. 周频信号扩展回日频 + 调用QBot引擎
# ============================================================
print("\n[回测引擎] QBot引擎（周频决策→日频T+1开盘执行）")

df_wk["signal_wk"] = df_wk["pos"].map({1.0: "growth", 0.0: "value"})
df_daily = df.reset_index().rename(columns={"index": "date"}).sort_values("date")
df_wk_reset = df_wk.reset_index().rename(columns={"index": "date"}).sort_values("date")[["date", "signal_wk"]]
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

config = BacktestConfig(
    commission=0.0001,
    impact_slippage=0.0,
    apply_gap_slippage=True,
)

print(f"  配置: {config.describe()}")
print(f"  回测区间: {df_bt.index[0]:%Y-%m-%d} ~ {df_bt.index[-1]:%Y-%m-%d} ({len(df_bt)} 天)")

result = run_backtest_engine(bt_input, config)
metrics = result.metrics


# ============================================================
# 5. 业绩指标打印
# ============================================================
print("\n" + "=" * 80)
print("业绩指标")
print("=" * 80)
print(result.summary("V74策略 — V72双因子 + 零波动率覆盖"))


# ============================================================
# 6. 持仓分布与覆盖触发统计
# ============================================================
print("\n[持仓分布统计]（实际持仓，含T+1执行延迟）")
pos_series = pd.Series(result.position, index=pd.to_datetime(result.dates))
pos_counts = pos_series.value_counts()
total_days = len(pos_series)
print(f"  {'持仓':<8} {'天数':>8} {'占比':>8}")
for p in ["growth", "value", "cash"]:
    n = pos_counts.get(p, 0)
    print(f"  {p:<8} {n:>8} {n/total_days*100:>7.1f}%")

# 覆盖触发率统计
ratio_dev_neg = (df_wk["style_score"] <= 0).sum()
growth_held_when_neg = ((df_wk["style_score"] <= 0) & (df_wk["pos"] == 1.0)).sum()
cover_rate = growth_held_when_neg / ratio_dev_neg * 100 if ratio_dev_neg > 0 else 0
print(f"\n  覆盖触发统计（style_score<=0时仍持成长的比例）:")
print(f"    style_score<=0 周数:    {ratio_dev_neg}")
print(f"    其中仍持成长周数:       {growth_held_when_neg}")
print(f"    覆盖触发率:             {cover_rate:.1f}%  (目标>5% 非装饰性)")


# ============================================================
# 7. 年度收益对比
# ============================================================
print("\n" + "=" * 80)
print("年度收益对比")
print("=" * 80)

result_df = result.to_dataframe()
result_df["date"] = pd.to_datetime(result_df["date"])
result_df = result_df.set_index("date")
result_df["year"] = result_df.index.year
annual_ret = result_df.groupby("year")["daily_ret"].apply(lambda x: (1 + x).prod() - 1)

g_close_bt = g_close.loc[df_bt.index[0]:]
v_close_bt = v_close.loc[df_bt.index[0]:]
g_annual = g_close_bt.resample("Y").last().pct_change().dropna()
v_annual = v_close_bt.resample("Y").last().pct_change().dropna()
g_annual.index = g_annual.index.year
v_annual.index = v_annual.index.year

print(f"\n  {'年份':<6} {'V74':>10} {'成长100':>10} {'价值100':>10} {'V74-成长':>10}")
print(f"  {'-'*55}")
for year in annual_ret.index:
    v74_ret = annual_ret[year]
    g_ret = g_annual.get(year, 0)
    v_ret = v_annual.get(year, 0)
    diff = v74_ret - g_ret
    flag = "OK" if diff > 0.01 else ("XX" if diff < -0.01 else "")
    print(f"  {year:<6} {v74_ret*100:>9.2f}% {g_ret*100:>9.2f}% {v_ret*100:>9.2f}% {diff*100:>+9.2f}pp {flag}")


# ============================================================
# 8. 零波动率场景专项测试
# ============================================================
print("\n" + "=" * 80)
print("零波动率场景专项测试（成长直线匀速上涨）")
print("=" * 80)

log_start = np.log(g_close.iloc[0])
log_end = np.log(g_close.iloc[-1])
log_prices = np.linspace(log_start, log_end, len(g_close))
g_zero = pd.Series(np.exp(log_prices), index=g_close.index)

# 零波动率场景下重新计算V74信号
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

df_zero = pd.DataFrame({
    "g_close": g_zero, "v_close": v_close,
    "style_score": style_zero,
    "smom_g": smom_g_zero, "smom_v": smom_v_zero,
})
df_zero_wk = df_zero.resample("W-FRI").last().dropna(
    subset=["style_score", "smom_g", "smom_v"]
).iloc[1:]

# 状态机决策（与主策略相同）
position_zero = pd.Series(np.nan, index=df_zero_wk.index)
current_pos = None
hold_weeks = 0
for i in range(len(df_zero_wk)):
    row = df_zero_wk.iloc[i]
    if row["style_score"] > 0:
        target = "growth"
    else:
        if current_pos == "growth" and row["smom_g"] > 0 and row["smom_v"] <= 0:
            target = "growth"
        elif current_pos is None:
            target = "value" if row["smom_v"] > row["smom_g"] else "growth"
        else:
            target = "value"

    if current_pos is None:
        current_pos = target
        hold_weeks = 1
        position_zero.iloc[i] = 1.0 if target == "growth" else 0.0
    elif target != current_pos and hold_weeks >= MIN_HOLD:
        current_pos = target
        hold_weeks = 1
        position_zero.iloc[i] = 1.0 if target == "growth" else 0.0
    else:
        hold_weeks += 1
        position_zero.iloc[i] = 1.0 if current_pos == "growth" else 0.0

df_zero_wk["pos"] = position_zero
df_zero_wk = df_zero_wk.dropna(subset=["pos"])

growth_pct_zero = (df_zero_wk["pos"] == 1.0).mean() * 100
print(f"  零波动率场景持成长占比: {growth_pct_zero:.1f}%  (目标≥80% {'OK' if growth_pct_zero >= 80 else 'XX'})")
print(f"  零波动率场景持成长周数: {(df_zero_wk['pos']==1.0).sum()} / {len(df_zero_wk)}")


# ============================================================
# 9. 可视化图表
# ============================================================
print("\n" + "=" * 80)
print("生成可视化图表")
print("=" * 80)

COLOR_GROWTH = "#E74C3C"
COLOR_VALUE = "#3498DB"
COLOR_CASH = "#95A5A6"

# ---- 图1：调仓换色资金曲线 ----
print("  [1/2] 调仓换色资金曲线...")

fig, ax = plt.subplots(figsize=(16, 8))

init_cash = config.start_cash
g_nav = init_cash * g_close_bt / g_close_bt.iloc[0]
v_nav = init_cash * v_close_bt / v_close_bt.iloc[0]
ax.plot(g_nav.index, g_nav / 10000, color=COLOR_GROWTH, alpha=0.20, linewidth=0.8, label="成长100")
ax.plot(v_nav.index, v_nav / 10000, color=COLOR_VALUE, alpha=0.20, linewidth=0.8, label="价值100")

nav_vals = result.nav / 10000
dates_nav = pd.to_datetime(result.dates)
positions = result.position

segments = []
current_color = None
seg_start = 0
for i in range(len(positions)):
    if positions[i] == "growth":
        color = COLOR_GROWTH
    elif positions[i] == "value":
        color = COLOR_VALUE
    else:
        color = COLOR_CASH

    if current_color is None:
        current_color = color
        seg_start = i
    elif color != current_color:
        seg_dates = dates_nav[seg_start:i+1]
        seg_vals = nav_vals[seg_start:i+1]
        segments.append((seg_dates, seg_vals, current_color))
        seg_start = i
        current_color = color

seg_dates = dates_nav[seg_start:]
seg_vals = nav_vals[seg_start:]
segments.append((seg_dates, seg_vals, current_color))

for seg_dates, seg_vals, color in segments:
    ax.plot(seg_dates, seg_vals, color=color, linewidth=2.0)

nav_series = pd.Series(result.nav, index=dates_nav)
rolling_max = nav_series.cummax()
dd = (nav_series - rolling_max) / rolling_max
ax.fill_between(dates_nav, 0, nav_vals, where=(dd.values < -0.05),
                color="gray", alpha=0.08, label="回撤>5%区间")

trades_df = result.trades_to_dataframe()
if len(trades_df) > 0:
    trades_df["trade_date"] = pd.to_datetime(trades_df["trade_date"])
    for _, t in trades_df.iterrows():
        if t["trade_date"] in dates_nav:
            idx = dates_nav.get_loc(t["trade_date"])
            color = COLOR_GROWTH if t["position"] == "growth" else (
                    COLOR_VALUE if t["position"] == "value" else COLOR_CASH)
            ax.scatter(t["trade_date"], nav_vals[idx], color=color, s=25,
                       zorder=5, edgecolors="black", linewidth=0.5)

ax.set_title("V74策略 — 调仓换色资金曲线\n红=持成长100  蓝=持价值100", fontsize=14)
ax.set_ylabel("净值 (万元)", fontsize=12)
ax.legend(loc="upper left", fontsize=10)
ax.grid(True, alpha=0.3)

textstr = (f"年化收益: {metrics['annual_ret']*100:.2f}%\n"
           f"Sharpe: {metrics['sharpe']:.3f}\n"
           f"最大回撤: {metrics['max_dd']*100:.2f}%\n"
           f"Calmar: {metrics['calmar']:.3f}\n"
           f"调仓次数: {metrics['num_trades']}\n"
           f"覆盖触发: {cover_triggered}次 ({cover_rate:.1f}%)")
props = dict(boxstyle="round", facecolor="wheat", alpha=0.8)
ax.text(0.98, 0.02, textstr, transform=ax.transAxes, fontsize=9,
        verticalalignment="bottom", horizontalalignment="right", bbox=props)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "v74_equity_colored.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"    保存: v74_equity_colored.png")


# ---- 图2：因子走势图（4子图） ----
print("  [2/2] 因子走势图...")

fig, axes = plt.subplots(4, 1, figsize=(16, 14), sharex=True)

# 子图1：主信号 - style_score
ax1 = axes[0]
score_vals = df_wk["style_score"].values
ax1.plot(df_wk.index, score_vals, color="#2C3E50", linewidth=1, alpha=0.7, label="style_score = f1+f2")
ax1.axhline(0, color="black", linewidth=0.5)
ax1.fill_between(df_wk.index, score_vals, 0,
                 where=(score_vals > 0), color=COLOR_GROWTH, alpha=0.15, label=">0 倾向成长")
ax1.fill_between(df_wk.index, score_vals, 0,
                 where=(score_vals <= 0), color=COLOR_VALUE, alpha=0.15, label="<=0 倾向价值(可覆盖)")
ax1.set_ylabel("style_score", fontsize=10)
ax1.set_title("主驱动：V72双因子 style_score = f1(比价MA20) + f2(动量加速度)", fontsize=12)
ax1.legend(loc="upper left", fontsize=8)
ax1.grid(True, alpha=0.3)

# 子图2：辅助信号 - 63日回归斜率×R² 比较
ax2 = axes[1]
smom_g_vals = df_wk["smom_g"].values * 100
smom_v_vals = df_wk["smom_v"].values * 100
ax2.plot(df_wk.index, smom_g_vals, color=COLOR_GROWTH, linewidth=1.2, alpha=0.8, label="成长 smom_g (%)")
ax2.plot(df_wk.index, smom_v_vals, color=COLOR_VALUE, linewidth=1.2, alpha=0.8, label="价值 smom_v (%)")
ax2.axhline(0, color="black", linewidth=0.5)
ax2.set_ylabel("回归斜率×R² (%)", fontsize=10)
ax2.set_title("辅助：63日对数价格回归斜率×R² — 零波动率覆盖用", fontsize=12)
ax2.legend(loc="upper left", fontsize=8)
ax2.grid(True, alpha=0.3)

# 子图3：覆盖触发事件
ax3 = axes[2]
cover_trigger = ((df_wk["style_score"] <= 0) & (df_wk["smom_g"] > 0) & (df_wk["smom_v"] <= 0)).astype(int)
ax3.plot(df_wk.index, cover_trigger, color="#8E44AD", linewidth=1, alpha=0.7, drawstyle="steps-post")
ax3.fill_between(df_wk.index, 0, cover_trigger, color="#8E44AD", alpha=0.3, label="覆盖触发（持成长）")
ax3.set_ylabel("覆盖触发", fontsize=10)
ax3.set_title(f"覆盖触发时间线（style_score<=0 且 smom_g>0 且 smom_v<=0） — 触发{cover_triggered}次", fontsize=12)
ax3.set_yticks([0, 1])
ax3.set_yticklabels(["不覆盖", "覆盖"])
ax3.legend(loc="upper left", fontsize=8)
ax3.grid(True, alpha=0.3)

# 子图4：持仓状态时间线
ax4 = axes[3]
pos_num = pd.Series(positions, index=dates_nav).map({"growth": 2, "value": 1, "cash": 0})
pos_colors = pd.Series(positions, index=dates_nav).map(
    {"growth": COLOR_GROWTH, "value": COLOR_VALUE, "cash": COLOR_CASH})
ax4.scatter(dates_nav, pos_num, c=pos_colors.values, s=3, alpha=0.7)
ax4.set_yticks([0, 1, 2])
ax4.set_yticklabels(["cash", "value", "growth"])
ax4.set_ylabel("持仓状态", fontsize=10)
ax4.set_title("持仓状态时间线（周频决策+T+1执行）", fontsize=12)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "v74_factors.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"    保存: v74_factors.png")


# ============================================================
# 10. 总结
# ============================================================
print("\n" + "=" * 80)
print("V74策略完成")
print("=" * 80)
print(f"  年化收益: {metrics['annual_ret']*100:.2f}%  (目标>=39% V72基准)  {'OK' if metrics['annual_ret'] >= 0.39 else 'XX'}")
print(f"  Sharpe:   {metrics['sharpe']:.3f}  (V72基准1.397)")
print(f"  最大回撤: {metrics['max_dd']*100:.2f}%  (V72基准-29.13%)")
print(f"  Calmar:   {metrics['calmar']:.3f}")
print(f"  调仓次数: {metrics['num_trades']} 次  (V72约144次)")
print(f"  覆盖触发: {cover_triggered}次 ({cover_rate:.1f}%)  (目标>5% 非装饰性)")
print(f"  零波动率场景持成长: {growth_pct_zero:.1f}%  (目标>=80%)  {'OK' if growth_pct_zero >= 80 else 'XX'}")
print(f"\n  图表已保存到: {OUTPUT_DIR}")
print("=" * 80)
