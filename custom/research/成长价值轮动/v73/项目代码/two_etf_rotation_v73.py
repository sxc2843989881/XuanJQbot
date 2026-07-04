"""V73 策略 — 3因子Dual Momentum日频方案（QBot引擎）

基于V72思路重构，使用QBot回测引擎（日频）：
  1. 因子1（主驱动）：成长自身20日绝对动量 — np.tanh(g_abs_mom_20 * 30) * 0.5
  2. 因子2（辅助确认）：比价MA20方向降权 — np.tanh(ratio_dev * 30) * 0.25
  3. 因子3（Dual Momentum择时）：63日绝对动量阈值0，赢家动量不足退避cash

四状态决策（二元全仓）：
  style_score = f1_signal + f2_signal
  - style_score > 0 且 g_abs_mom_63 > 0  → 'growth'
  - style_score <= 0 且 v_abs_mom_63 > 0 → 'value'
  - 否则                                 → 'cash'

去掉MA75降仓到10%逻辑 — V73不再使用MA75择时。

信号由shift(1)的因子数据生成（避免未来函数），QBot引擎自动T+1开盘执行。
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
# 项目代码/ -> v73/ -> 成长价值轮动/ -> research/ -> custom/ -> sxc/ -> style_rotation_strategy
STYLE_ROTATION_DIR = (SCRIPT_DIR / "../../../../../style_rotation_strategy").resolve()
if not STYLE_ROTATION_DIR.exists():
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
print("V73 策略 — 3因子Dual Momentum日频方案（QBot引擎）")
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

# 对齐日期（取交集）
common = g_df.index.intersection(v_df.index)
g_df = g_df.loc[common]
v_df = v_df.loc[common]

g_close = g_df["close"].astype(float)
v_close = v_df["close"].astype(float)

# 处理open缺失：优先用原始open，缺失则用前一日close作为open近似
g_open_raw = g_df["open"].astype(float) if "open" in g_df.columns else pd.Series(np.nan, index=g_df.index)
v_open_raw = v_df["open"].astype(float) if "open" in v_df.columns else pd.Series(np.nan, index=v_df.index)

g_open = g_open_raw.fillna(g_close.shift(1)).fillna(g_close)
v_open = v_open_raw.fillna(v_close.shift(1)).fillna(v_close)

# 去除close为NaN的日期
valid = ~(g_close.isna() | v_close.isna())
g_close = g_close[valid]
v_close = v_close[valid]
g_open = g_open[valid]
v_open = v_open[valid]

print(f"\n[数据] 共 {len(g_close)} 个交易日")
print(f"  区间: {g_close.index[0]:%Y-%m-%d} ~ {g_close.index[-1]:%Y-%m-%d}")
print(f"  原始open非空: {g_open_raw.notna().sum()} / {len(g_open_raw)} (其余用前一日close近似)")


# ============================================================
# 2. 因子计算 — 全部shift(1)防止未来函数
# ============================================================
print("\n[因子层] 3因子Dual Momentum（全部shift(1)）")

# ---- 因子1（主驱动）：成长自身20日绝对动量 ----
g_abs_mom_20 = g_close.pct_change(20).shift(1)  # T日用T-1及之前数据
f1_signal = np.tanh(g_abs_mom_20 * 30) * 0.5
# 成长稳步上涨时 g_abs_mom_20 > 0 → f1 > 0 明确持成长

# ---- 因子2（辅助确认）：比价MA20方向降权 ----
ratio = (g_close / v_close).shift(1)
ratio_ma20 = ratio.rolling(20).mean()
ratio_dev = ratio / ratio_ma20 - 1
f2_signal = np.tanh(ratio_dev * 30) * 0.25  # 原0.5降权一半

# ---- 因子3（Dual Momentum择时）：63日绝对动量阈值0 ----
g_abs_mom_63 = g_close.pct_change(63).shift(1)
v_abs_mom_63 = v_close.pct_change(63).shift(1)

# 综合风格得分
style_score = f1_signal + f2_signal

# 组装DataFrame
df = pd.DataFrame(index=g_close.index)
df["g_close"] = g_close
df["v_close"] = v_close
df["g_open"] = g_open
df["v_open"] = v_open
df["g_abs_mom_20"] = g_abs_mom_20
df["g_abs_mom_63"] = g_abs_mom_63
df["v_abs_mom_63"] = v_abs_mom_63
df["ratio_dev"] = ratio_dev
df["f1_signal"] = f1_signal
df["f2_signal"] = f2_signal
df["style_score"] = style_score

# ---- 四状态二元全仓决策（向量化）----
# NaN在比较中返回False，自然落入'cash'分支
style_arr = df["style_score"].values
g63_arr = df["g_abs_mom_63"].values
v63_arr = df["v_abs_mom_63"].values

signal_arr = np.where(
    (style_arr > 0) & (g63_arr > 0), "growth",
    np.where(
        (style_arr <= 0) & (v63_arr > 0), "value",
        "cash"
    )
)
df["signal"] = signal_arr

print(f"\n[信号统计]（全区间）")
counts = df["signal"].value_counts()
for sig in ["growth", "value", "cash"]:
    n = counts.get(sig, 0)
    print(f"  {sig:>6}: {n:>5} 天 ({n/len(df)*100:5.1f}%)")


# ============================================================
# 3. 调用QBot回测引擎
# ============================================================
print("\n[回测引擎] QBot日频回测")

# 从因子全部首次可用日开始（前63+1=64天数据不足）
first_valid = df.dropna(subset=["g_abs_mom_63", "v_abs_mom_63", "style_score"]).index[0]
print(f"  因子首次全部可用日: {first_valid:%Y-%m-%d}")

df_bt = df.loc[first_valid:].copy()

bt_input = BacktestInput(
    dates=df_bt.index.strftime("%Y-%m-%d").values,
    value_open=df_bt["v_open"].values.astype(np.float64),
    value_close=df_bt["v_close"].values.astype(np.float64),
    growth_open=df_bt["g_open"].values.astype(np.float64),
    growth_close=df_bt["g_close"].values.astype(np.float64),
    signal=df_bt["signal"].values.astype(str),
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
# 4. 业绩指标打印
# ============================================================
print("\n" + "=" * 80)
print("业绩指标")
print("=" * 80)
print(result.summary("V73策略 — 3因子Dual Momentum"))


# ============================================================
# 5. 持仓分布统计
# ============================================================
print("\n[持仓分布统计]（实际持仓，含T+1执行延迟）")
pos_series = pd.Series(result.position, index=pd.to_datetime(result.dates))
pos_counts = pos_series.value_counts()
total_days = len(pos_series)
print(f"  {'持仓':<8} {'天数':>8} {'占比':>8}")
for p in ["growth", "value", "cash"]:
    n = pos_counts.get(p, 0)
    print(f"  {p:<8} {n:>8} {n/total_days*100:>7.1f}%")

# 成长稳步上涨时持成长占比验证
# 定义"成长稳步上涨"区间：style_score>0 且 g_abs_mom_63>0
g_up_mask = (df_bt["style_score"] > 0) & (df_bt["g_abs_mom_63"] > 0)
# 对齐到pos_series的索引
g_up_aligned = g_up_mask.reindex(pos_series.index, fill_value=False)
g_up_days = g_up_aligned.sum()
g_up_growth = ((pos_series == "growth") & g_up_aligned).sum()
growth_hit_rate = g_up_growth / g_up_days * 100 if g_up_days > 0 else 0
print(f"\n  成长占优区间（style_score>0 且 g_abs_mom_63>0）:")
print(f"    总天数:     {g_up_days}")
print(f"    实际持成长: {g_up_growth} ({growth_hit_rate:.1f}%)")
print(f"    目标:       ≥80%  {'OK' if growth_hit_rate >= 80 else 'XX'}")


# ============================================================
# 6. 年度收益对比
# ============================================================
print("\n" + "=" * 80)
print("年度收益对比")
print("=" * 80)

result_df = result.to_dataframe()
result_df["date"] = pd.to_datetime(result_df["date"])
result_df = result_df.set_index("date")
result_df["year"] = result_df.index.year
annual_ret = result_df.groupby("year")["daily_ret"].apply(lambda x: (1 + x).prod() - 1)

# 基准：成长100/价值100年度收益
g_close_bt = g_close.loc[df_bt.index[0]:]
v_close_bt = v_close.loc[df_bt.index[0]:]
g_annual = g_close_bt.resample("Y").last().pct_change().dropna()
v_annual = v_close_bt.resample("Y").last().pct_change().dropna()
# 转为以年份整数为索引，方便查找
g_annual.index = g_annual.index.year
v_annual.index = v_annual.index.year

print(f"\n  {'年份':<6} {'V73':>10} {'成长100':>10} {'价值100':>10} {'V73-成长':>10}")
print(f"  {'-'*55}")
for year in annual_ret.index:
    v73_ret = annual_ret[year]
    g_ret = g_annual.get(year, 0)
    v_ret = v_annual.get(year, 0)
    diff = v73_ret - g_ret
    flag = "OK" if diff > 0.01 else ("XX" if diff < -0.01 else "")
    print(f"  {year:<6} {v73_ret*100:>9.2f}% {g_ret*100:>9.2f}% {v_ret*100:>9.2f}% {diff*100:>+9.2f}pp {flag}")


# ============================================================
# 7. 可视化图表
# ============================================================
print("\n" + "=" * 80)
print("生成可视化图表")
print("=" * 80)

COLOR_GROWTH = "#E74C3C"   # 红色 = 持成长
COLOR_VALUE = "#3498DB"     # 蓝色 = 持价值
COLOR_CASH = "#95A5A6"      # 灰色 = 空仓

# ---- 图1：调仓换色资金曲线 ----
print("  [1/2] 调仓换色资金曲线...")

fig, ax = plt.subplots(figsize=(16, 8))

# 基准曲线（成长100/价值100归一化）
init_cash = config.start_cash
g_nav = init_cash * g_close_bt / g_close_bt.iloc[0]
v_nav = init_cash * v_close_bt / v_close_bt.iloc[0]
ax.plot(g_nav.index, g_nav / 10000, color=COLOR_GROWTH, alpha=0.20, linewidth=0.8, label="成长100")
ax.plot(v_nav.index, v_nav / 10000, color=COLOR_VALUE, alpha=0.20, linewidth=0.8, label="价值100")

# 策略曲线（分段着色）
nav_vals = result.nav / 10000
dates_nav = pd.to_datetime(result.dates)
positions = result.position

# 分段着色
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

# 回撤阴影
nav_series = pd.Series(result.nav, index=dates_nav)
rolling_max = nav_series.cummax()
dd = (nav_series - rolling_max) / rolling_max
ax.fill_between(dates_nav, 0, nav_vals, where=(dd.values < -0.05),
                color="gray", alpha=0.08, label="回撤>5%区间")

# 标记调仓点
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

ax.set_title("V73策略 — 调仓换色资金曲线\n红=持成长100  蓝=持价值100  灰=cash空仓", fontsize=14)
ax.set_ylabel("净值 (万元)", fontsize=12)
ax.legend(loc="upper left", fontsize=10)
ax.grid(True, alpha=0.3)

textstr = (f"年化收益: {metrics['annual_ret']*100:.2f}%\n"
           f"Sharpe: {metrics['sharpe']:.3f}\n"
           f"最大回撤: {metrics['max_dd']*100:.2f}%\n"
           f"Calmar: {metrics['calmar']:.3f}\n"
           f"调仓次数: {metrics['num_trades']}")
props = dict(boxstyle="round", facecolor="wheat", alpha=0.8)
ax.text(0.98, 0.02, textstr, transform=ax.transAxes, fontsize=9,
        verticalalignment="bottom", horizontalalignment="right", bbox=props)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "v73_equity_colored.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"    保存: v73_equity_colored.png")


# ---- 图2：因子走势图（4子图） ----
print("  [2/2] 因子走势图...")

fig, axes = plt.subplots(4, 1, figsize=(16, 14), sharex=True)

# 子图1：因子1 — 成长20日绝对动量
ax1 = axes[0]
g_mom20_vals = df_bt["g_abs_mom_20"].values * 100
ax1.plot(df_bt.index, g_mom20_vals, color="#8E44AD", linewidth=1, alpha=0.7, label="成长20日绝对动量(%)")
ax1.axhline(0, color="black", linewidth=0.5)
ax1.fill_between(df_bt.index, g_mom20_vals, 0,
                 where=(g_mom20_vals > 0), color="red", alpha=0.15)
ax1.fill_between(df_bt.index, g_mom20_vals, 0,
                 where=(g_mom20_vals < 0), color="blue", alpha=0.15)
ax1.set_ylabel("20日动量 (%)", fontsize=10)
ax1.set_title("因子1：成长自身20日绝对动量（主驱动，权重0.5，>0倾向成长）", fontsize=12)
ax1.legend(loc="upper left", fontsize=8)
ax1.grid(True, alpha=0.3)

# 子图2：因子2 — 比价MA20方向
ax2 = axes[1]
ratio_dev_vals = df_bt["ratio_dev"].values * 100
ax2.plot(df_bt.index, ratio_dev_vals, color="#16A085", linewidth=1, alpha=0.7, label="比价偏离MA20(%)")
ax2.axhline(0, color="black", linewidth=0.5)
ax2.fill_between(df_bt.index, ratio_dev_vals, 0,
                 where=(ratio_dev_vals > 0), color="red", alpha=0.15)
ax2.fill_between(df_bt.index, ratio_dev_vals, 0,
                 where=(ratio_dev_vals < 0), color="blue", alpha=0.15)
ax2.set_ylabel("比价偏离 (%)", fontsize=10)
ax2.set_title("因子2：比价MA20方向（辅助确认，降权0.25）", fontsize=12)
ax2.legend(loc="upper left", fontsize=8)
ax2.grid(True, alpha=0.3)

# 子图3：综合风格得分
ax3 = axes[2]
score_vals = df_bt["style_score"].values
ax3.plot(df_bt.index, score_vals, color="#2C3E50", linewidth=1.2, label="综合风格得分 = f1 + f2")
ax3.axhline(0, color="black", linewidth=1, linestyle="-")
ax3.fill_between(df_bt.index, score_vals, 0,
                 where=(score_vals > 0), color=COLOR_GROWTH, alpha=0.2, label="选成长")
ax3.fill_between(df_bt.index, score_vals, 0,
                 where=(score_vals < 0), color=COLOR_VALUE, alpha=0.2, label="选价值")
ax3.set_ylabel("综合得分", fontsize=10)
ax3.set_title("综合风格得分（>0选成长，<=0选价值，需63日动量确认）", fontsize=12)
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
ax4.set_title("持仓状态时间线（含63日Dual Momentum择时退避cash）", fontsize=12)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "v73_factors.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"    保存: v73_factors.png")


# ============================================================
# 8. 总结
# ============================================================
print("\n" + "=" * 80)
print("V73策略完成")
print("=" * 80)
print(f"  年化收益: {metrics['annual_ret']*100:.2f}%  (基准V72目标>=39%)")
print(f"  Sharpe:   {metrics['sharpe']:.3f}  (目标>=1.3)")
print(f"  最大回撤: {metrics['max_dd']*100:.2f}%")
print(f"  Calmar:   {metrics['calmar']:.3f}")
print(f"  调仓次数: {metrics['num_trades']} 次")
print(f"  成长占优区间持成长占比: {growth_hit_rate:.1f}% (目标>=80%)")
print(f"\n  图表已保存到: {OUTPUT_DIR}")
print("=" * 80)
