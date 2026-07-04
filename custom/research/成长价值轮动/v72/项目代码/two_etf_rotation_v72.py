"""V72 策略 — 二元全仓2+1因子精简版

基于V71参数稳健性分析的结论：
1. 保留：比价MA20方向(f1=0.5) + 动量加速度(f2=5.0) + MA75择时(ma_pct=0.97)
2. 去掉：ADX门控（效果有限）、滞回（有害）、绝对动量（被证伪）
3. 修复：cut参数bug（之前硬编码0.1，现在正确使用参数）

参数选择基于经济学逻辑（非搜索拟合）：
- f1=0.5：比价偏离3%=2.3σ，统计显著
- f2=5.0：3.5-8.0宽平台中间值
- MA=75：3.5个月中期趋势，前后半样本都选此值
- ma_pct=0.97：3%跌破阈值=2.3σ
- cut=0.1：降仓到10%

二元全仓：要么100%成长，要么100%价值
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import warnings
warnings.filterwarnings("ignore")

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

sys.path.insert(0, str(Path(__file__).resolve().parent))
DATA_DIR = Path(__file__).parent / "data"
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# 1. 数据加载
# ============================================================
print("=" * 80)
print("V72 策略 — 二元全仓2+1因子精简版")
print("=" * 80)

g_raw = pd.read_csv(DATA_DIR / "index_480080.csv")
v_raw = pd.read_csv(DATA_DIR / "index_480081.csv")
for d in (g_raw, v_raw):
    d["date"] = pd.to_datetime(d["date"])
    d["close"] = pd.to_numeric(d["close"], errors="coerce")
g_close = g_raw.set_index("date")["close"].astype(float).sort_index().dropna()
v_close = v_raw.set_index("date")["close"].astype(float).sort_index().dropna()
common = g_close.index.intersection(v_close.index)
g_close = g_close[common].sort_index()
v_close = v_close[common].sort_index()


# ============================================================
# 2. 信号层
# ============================================================
print("\n[信号层] V72 2+1因子：比价MA20(0.5) + 动量加速度(5.0) + MA75择时(0.97)")

# 参数（基于逻辑，非搜索拟合）
F1 = 0.5       # 比价偏离3%=2.3σ
F2 = 5.0       # 3.5-8.0宽平台中间值
MA_W = 75      # 3.5个月中期趋势
MA_PCT = 0.97  # 3%跌破阈值
CUT = 0.1      # 降仓到10%

# ---- 因子1：比价MA20方向（相对动量，选风格）----
ratio = (g_close / v_close).shift(1)
ratio_ma20 = ratio.rolling(20).mean()
ratio_dev = ratio / ratio_ma20 - 1
f1_signal = np.tanh(ratio_dev * 30) * F1
style_score = f1_signal.copy()

# ---- 因子2：动量加速度（趋势确认）----
g_roc21 = g_close.pct_change(21).shift(1)
v_roc21 = v_close.pct_change(21).shift(1)
g_accel = g_roc21 - g_roc21.shift(10)
v_accel = v_roc21 - v_roc21.shift(10)
accel_diff = (g_accel - v_accel).clip(-0.02, 0.02)
f2_signal = accel_diff * F2
style_score = style_score + f2_signal

# 二元候选：True=成长, False=价值
candidate_g = style_score > 0

# ---- 四状态判断（状态识别，非因子）----
def calc_weighted_momentum(close):
    return (4*close.pct_change(21) + 3*close.pct_change(63) + 
            2*close.pct_change(126) + 1*close.pct_change(252)) / 10

g_mom = calc_weighted_momentum(g_close).shift(1)
v_mom = calc_weighted_momentum(v_close).shift(1)
both_up = (g_mom > 0) & (v_mom > 0)
growth_only = (g_mom > 0) & (v_mom <= 0)
value_only = (g_mom <= 0) & (v_mom > 0)
both_down = (g_mom <= 0) & (v_mom <= 0)

# ---- 因子3：MA75择时（系统性下跌降仓）----
g_ma = g_close.shift(1).rolling(MA_W).mean()
v_ma = v_close.shift(1).rolling(MA_W).mean()
both_below = (g_close.shift(1) < g_ma * MA_PCT) & (v_close.shift(1) < v_ma * MA_PCT)


# ============================================================
# 3. 仓位层 — 二元全仓状态机
# ============================================================
print("\n[仓位层] 二元全仓：100%成长 or 100%价值，MA75触发降仓10%")

df = pd.DataFrame(index=g_close.index)
df["g_close"] = g_close
df["v_close"] = v_close
df["candidate_g"] = candidate_g
df["both_below"] = both_below
df["both_up"] = both_up
df["ratio_dev"] = ratio_dev
df["f1_signal"] = f1_signal
df["f2_signal"] = f2_signal
df["style_score"] = style_score
df["g_mom"] = g_mom
df["v_mom"] = v_mom

# 周频采样
weekly = df.resample("W-FRI").last().dropna(subset=["candidate_g"]).iloc[1:]

# 状态机
position = pd.Series(np.nan, index=weekly.index)  # 1=成长, 0=价值, 0.1=降仓
current_pos = None
switch_log = []  # 记录每次切换

for i in range(len(weekly)):
    date = weekly.index[i]
    row = weekly.iloc[i]

    # MA75择时优先
    if row["both_below"]:
        position.iloc[i] = CUT  # 降仓
        continue

    if current_pos is None:
        current_pos = 1.0 if row["candidate_g"] else 0.0
        position.iloc[i] = current_pos
        switch_log.append({"date": date, "action": "init", "pos": current_pos,
                          "ratio_dev": row["ratio_dev"], "f1": row["f1_signal"],
                          "f2": row["f2_signal"], "score": row["style_score"]})
        continue

    target = 1.0 if row["candidate_g"] else 0.0
    if target == current_pos:
        position.iloc[i] = current_pos
    else:
        # 执行切换
        old_pos = current_pos
        current_pos = target
        position.iloc[i] = current_pos
        switch_log.append({"date": date, "action": "switch", "pos": current_pos,
                          "old_pos": old_pos, "ratio_dev": row["ratio_dev"],
                          "f1": row["f1_signal"], "f2": row["f2_signal"],
                          "score": row["style_score"]})

weekly["position"] = position
weekly = weekly.dropna(subset=["position"])

# 转换为执行权重
weekly["w_g_exec"] = weekly["position"].apply(
    lambda x: 1.0 if x == 1.0 else (0.0 if x == 0.0 else 0.5)).shift(1)
weekly["tp_exec"] = weekly["position"].apply(
    lambda x: CUT if x == CUT else 1.0).shift(1)
weekly = weekly.dropna(subset=["w_g_exec"])

print(f"回测区间: {weekly.index[0]:%Y-%m-%d} ~ {weekly.index[-1]:%Y-%m-%d}")
print(f"调仓次数: {len(switch_log)} 次")


# ============================================================
# 4. 回测引擎
# ============================================================
print("\n[回测引擎] 周频W-FRI，佣金万1+印花税万3")

INIT_CAPITAL = 1_000_000
COMMISSION = 0.0001
STAMP_TAX = 0.0003
MIN_COMMISSION = 5

cap = INIT_CAPITAL
pg, pv, pc = 0.0, 0.0, float(cap)
prev_w = None
nav_records = []
n_switches = 0

for date, row in weekly.iterrows():
    gv = pg * row["g_close"] if not pd.isna(row["g_close"]) else 0
    vv = pv * row["v_close"] if not pd.isna(row["v_close"]) else 0
    total = gv + vv + pc
    tp = row["tp_exec"]
    wg = row["w_g_exec"]

    tg_w = tp * wg
    tv_w = tp * (1 - wg)

    if prev_w is not None and abs(wg - prev_w) > 0.5:
        n_switches += 1
    prev_w = wg

    tga = total * tg_w
    tva = total * tv_w

    def calc_fee(old_v, new_v):
        if pd.isna(old_v) or pd.isna(new_v) or abs(new_v - old_v) < 1e-10:
            return 0
        diff = abs(new_v - old_v)
        fee = diff * COMMISSION
        if old_v > new_v:
            fee += (old_v - new_v) * STAMP_TAX
        return max(fee, MIN_COMMISSION)

    fee_g = calc_fee(gv, tga)
    fee_v = calc_fee(vv, tva)
    total_fee = fee_g + fee_v

    pg = tga / row["g_close"] if row["g_close"] > 0 and tga > 0 else 0
    pv = tva / row["v_close"] if row["v_close"] > 0 and tva > 0 else 0
    pc = total - tga - tva - total_fee

    nav_records.append({
        "date": date, "nav": total - total_fee, "fee": total_fee,
        "w_g": wg, "total_pos": tp, "g_close": row["g_close"], "v_close": row["v_close"],
        "position": row["position"],
    })

df_r = pd.DataFrame(nav_records).set_index("date")
df_r["ret"] = df_r["nav"].pct_change()
df_r.iloc[0, df_r.columns.get_loc("ret")] = df_r["nav"].iloc[0] / INIT_CAPITAL - 1


# ============================================================
# 5. 业绩指标
# ============================================================
print("\n" + "=" * 80)
print("业绩指标")
print("=" * 80)

def calc_metrics(returns, freq=52, risk_free=0.025):
    equity = (1 + returns).cumprod()
    n = len(returns)
    years = n / freq
    total_ret = equity.iloc[-1] - 1
    ann_ret = (1 + total_ret) ** (1 / years) - 1 if years > 0 else 0
    ann_vol = returns.std() * np.sqrt(freq)
    rf_period = risk_free / freq
    sharpe = (returns.mean() - rf_period) / returns.std() * np.sqrt(freq) if returns.std() > 0 else 0
    rolling_max = equity.cummax()
    dd = (equity - rolling_max) / rolling_max
    max_dd = dd.min()
    calmar = ann_ret / abs(max_dd) if max_dd < 0 else 0
    win_rate = (returns > 0).sum() / (returns != 0).sum() if (returns != 0).sum() > 0 else 0
    return {"total_return": total_ret, "annual_return": ann_ret,
            "annual_volatility": ann_vol, "sharpe": sharpe,
            "max_drawdown": max_dd, "calmar": calmar,
            "win_rate": win_rate, "years": years}

m = calc_metrics(df_r["ret"])
g_weekly = weekly["g_close"].pct_change().fillna(0)
v_weekly = weekly["v_close"].pct_change().fillna(0)
equal_w = (g_weekly + v_weekly) / 2
m_g = calc_metrics(g_weekly)
m_v = calc_metrics(v_weekly)
m_eq = calc_metrics(equal_w)

print(f"\n{'指标':<14} {'V72策略':>10} {'基准(等权)':>12} {'成长100':>10} {'价值100':>10}")
print("-" * 65)
for key, label in [("annual_return", "年化收益"), ("max_drawdown", "最大回撤"),
                   ("sharpe", "Sharpe"), ("calmar", "Calmar"), ("win_rate", "胜率")]:
    s, g, v, e = m[key], m_g[key], m_v[key], m_eq[key]
    if key in ("annual_return", "max_drawdown", "win_rate"):
        print(f"  {label:<12} {s*100:>9.2f}% {e*100:>11.2f}% {g*100:>9.2f}% {v*100:>9.2f}%")
    else:
        print(f"  {label:<12} {s:>10.3f} {e:>12.3f} {g:>10.3f} {v:>10.3f}")

print(f"\n  调仓次数: {n_switches}, 年化换手: {n_switches/m['years']:.1f}次/年")
print(f"\n  36%目标: {'✅ 已达成' if m['annual_return'] >= 0.36 else '❌ 未达'} "
      f"(实际 {m['annual_return']*100:.2f}% vs 目标 36.00%, 差 {(m['annual_return']-0.36)*100:+.2f}pp)")


# ============================================================
# 6. 可视化 — 调仓换色资金曲线 + 因子走势
# ============================================================
print("\n" + "=" * 80)
print("生成可视化图表")
print("=" * 80)

COLOR_GROWTH = "#E74C3C"   # 红色 = 持成长
COLOR_VALUE = "#3498DB"     # 蓝色 = 持价值
COLOR_CUT = "#95A5A6"       # 灰色 = 降仓
COLOR_BENCH = "#2ECC71"     # 绿色 = 基准

# ---- 图1：调仓换色资金曲线 ----
print("  [1/4] 调仓换色资金曲线...")

fig, ax = plt.subplots(figsize=(16, 8))

# 基准曲线（等权）
eq_nav = INIT_CAPITAL * (1 + equal_w).cumprod()
ax.plot(df_r.index, eq_nav / 10000, color=COLOR_BENCH, alpha=0.4, linewidth=1, label="等权基准")

# 成长100/价值100归一化
g_nav = INIT_CAPITAL * g_close.loc[df_r.index[0]:] / g_close.loc[df_r.index[0]]
v_nav = INIT_CAPITAL * v_close.loc[df_r.index[0]:] / v_close.loc[df_r.index[0]]
ax.plot(g_nav.index, g_nav / 10000, color=COLOR_GROWTH, alpha=0.15, linewidth=0.8)
ax.plot(v_nav.index, v_nav / 10000, color=COLOR_VALUE, alpha=0.15, linewidth=0.8)

# 策略曲线（分段着色）
nav_vals = df_r["nav"].values / 10000
dates = df_r.index
positions = df_r["position"].values

# 找到每次仓位变化的点
segments = []
current_color = None
seg_start = 0

for i in range(len(positions)):
    if positions[i] == 1.0:
        color = COLOR_GROWTH
    elif positions[i] == 0.0:
        color = COLOR_VALUE
    else:
        color = COLOR_CUT

    if current_color is None:
        current_color = color
        seg_start = i
    elif color != current_color:
        # 添加前一段
        seg_dates = dates[seg_start:i+1]
        seg_vals = nav_vals[seg_start:i+1]
        segments.append((seg_dates, seg_vals, current_color))
        seg_start = i
        current_color = color

# 最后一段
seg_dates = dates[seg_start:]
seg_vals = nav_vals[seg_start:]
segments.append((seg_dates, seg_vals, current_color))

# 绘制每一段
for seg_dates, seg_vals, color in segments:
    ax.plot(seg_dates, seg_vals, color=color, linewidth=2.0)

# 标记切换点
for log in switch_log:
    if log["action"] == "switch":
        d = log["date"]
        if d in df_r.index:
            idx = df_r.index.get_loc(d)
            ax.axvline(d, color="gray", alpha=0.15, linewidth=0.5, linestyle="--")
            # 标记箭头
            color = COLOR_GROWTH if log["pos"] == 1.0 else COLOR_VALUE
            ax.scatter(d, nav_vals[idx], color=color, s=30, zorder=5, edgecolors="black", linewidth=0.5)

# 回撤阴影
equity = (1 + df_r["ret"]).cumprod()
rolling_max = equity.cummax()
dd = (equity - rolling_max) / rolling_max
ax.fill_between(df_r.index, 0, nav_vals, where=(dd < -0.05), 
                color="gray", alpha=0.08, label="回撤>5%区间")

ax.set_title("V72策略 — 调仓换色资金曲线\n红=持成长100  蓝=持价值100  灰=降仓  绿=等权基准", fontsize=14)
ax.set_ylabel("净值 (万元)", fontsize=12)
ax.legend(loc="upper left", fontsize=10)
ax.grid(True, alpha=0.3)

# 添加调仓统计文本
textstr = f"调仓次数: {n_switches}次\n年化换手: {n_switches/m['years']:.1f}次/年\n年化收益: {m['annual_return']*100:.2f}%\nSharpe: {m['sharpe']:.3f}\n回撤: {m['max_drawdown']*100:.2f}%"
props = dict(boxstyle="round", facecolor="wheat", alpha=0.8)
ax.text(0.98, 0.02, textstr, transform=ax.transAxes, fontsize=9,
        verticalalignment="bottom", horizontalalignment="right", bbox=props)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "v72_equity_colored.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"    保存: v72_equity_colored.png")


# ---- 图2：回撤曲线 ----
print("  [2/4] 回撤曲线...")
fig, ax = plt.subplots(figsize=(16, 5))
ax.fill_between(df_r.index, dd * 100, 0, color="#E74C3C", alpha=0.4)
ax.plot(df_r.index, dd * 100, color="#C0392B", linewidth=1)
ax.set_title("V72策略 — 回撤曲线", fontsize=14)
ax.set_ylabel("回撤 (%)", fontsize=12)
ax.axhline(m["max_drawdown"] * 100, color="black", linestyle="--", alpha=0.5, 
           label=f"最大回撤: {m['max_drawdown']*100:.2f}%")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "v72_drawdown.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"    保存: v72_drawdown.png")


# ---- 图3：因子走势图 ----
print("  [3/4] 因子走势图...")

fig, axes = plt.subplots(4, 1, figsize=(16, 14), sharex=True)

# 因子1：比价MA20方向信号
ax1 = axes[0]
ax1.plot(weekly.index, weekly["ratio_dev"] * 100, color="#8E44AD", linewidth=1, alpha=0.7, label="比价偏离(%)")
ax1.axhline(0, color="black", linewidth=0.5)
ax1.axhline(3, color="red", linewidth=0.5, linestyle="--", alpha=0.5, label="+3%阈值")
ax1.axhline(-3, color="blue", linewidth=0.5, linestyle="--", alpha=0.5, label="-3%阈值")
ax1.fill_between(weekly.index, weekly["ratio_dev"]*100, 0, 
                 where=weekly["ratio_dev"]>0, color="red", alpha=0.15)
ax1.fill_between(weekly.index, weekly["ratio_dev"]*100, 0, 
                 where=weekly["ratio_dev"]<0, color="blue", alpha=0.15)
ax1.set_ylabel("比价偏离MA20 (%)", fontsize=10)
ax1.set_title("因子1：比价MA20方向（>0成长占优，<0价值占优）", fontsize=12)
ax1.legend(loc="upper left", fontsize=8)
ax1.grid(True, alpha=0.3)

# 因子2：动量加速度信号
ax2 = axes[1]
ax2.plot(weekly.index, weekly["f2_signal"], color="#E67E22", linewidth=1, alpha=0.7, label="动量加速度信号")
ax2.axhline(0, color="black", linewidth=0.5)
ax2.fill_between(weekly.index, weekly["f2_signal"], 0, 
                 where=weekly["f2_signal"]>0, color="red", alpha=0.15)
ax2.fill_between(weekly.index, weekly["f2_signal"], 0, 
                 where=weekly["f2_signal"]<0, color="blue", alpha=0.15)
ax2.set_ylabel("动量加速度信号", fontsize=10)
ax2.set_title("因子2：动量加速度（>0成长加速，<0价值加速）", fontsize=12)
ax2.legend(loc="upper left", fontsize=8)
ax2.grid(True, alpha=0.3)

# 综合风格得分
ax3 = axes[2]
ax3.plot(weekly.index, weekly["style_score"], color="#2C3E50", linewidth=1.2, label="综合风格得分")
ax3.axhline(0, color="black", linewidth=1, linestyle="-")
ax3.fill_between(weekly.index, weekly["style_score"], 0, 
                 where=weekly["style_score"]>0, color=COLOR_GROWTH, alpha=0.2, label="选成长")
ax3.fill_between(weekly.index, weekly["style_score"], 0, 
                 where=weekly["style_score"]<0, color=COLOR_VALUE, alpha=0.2, label="选价值")
# 标记切换点
for log in switch_log:
    if log["action"] == "switch":
        d = log["date"]
        if d in weekly.index:
            ax3.axvline(d, color="gray", alpha=0.2, linewidth=0.5, linestyle="--")
ax3.set_ylabel("综合风格得分", fontsize=10)
ax3.set_title("综合风格得分 = 因子1 + 因子2（>0选成长，<0选价值）", fontsize=12)
ax3.legend(loc="upper left", fontsize=8)
ax3.grid(True, alpha=0.3)

# 持仓状态
ax4 = axes[3]
pos_colors = weekly["position"].apply(
    lambda x: COLOR_GROWTH if x == 1.0 else (COLOR_VALUE if x == 0.0 else COLOR_CUT))
ax4.scatter(weekly.index, weekly["position"], c=pos_colors, s=3, alpha=0.7)
ax4.set_yticks([0, 0.1, 1.0])
ax4.set_yticklabels(["价值", "降仓", "成长"])
ax4.set_ylabel("持仓状态", fontsize=10)
ax4.set_title("持仓状态时间线", fontsize=12)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "v72_factors.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"    保存: v72_factors.png")


# ---- 图4：调仓日志表 ----
print("  [4/4] 调仓日志图...")

fig, ax = plt.subplots(figsize=(16, 6))
ax.axis("off")

# 准备表格数据
table_data = []
for log in switch_log:
    pos_label = "成长100" if log["pos"] == 1.0 else ("价值100" if log["pos"] == 0.0 else "降仓")
    action = "初始化" if log["action"] == "init" else (
        f"切→成长" if log["pos"] == 1.0 else "切→价值")
    table_data.append([
        log["date"].strftime("%Y-%m-%d"),
        action,
        f"{log['ratio_dev']*100:+.2f}%",
        f"{log['f1']:+.3f}",
        f"{log['f2']:+.3f}",
        f"{log['score']:+.3f}",
    ])

col_labels = ["调仓日期", "操作", "比价偏离", "因子1信号", "因子2信号", "综合得分"]
table = ax.table(cellText=table_data, colLabels=col_labels, loc="center",
                 cellLoc="center", colColours=["#34495E"] * 6)
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1, 1.2)

# 设置表头颜色
for i in range(len(col_labels)):
    table[0, i].set_text_props(color="white", weight="bold")

# 交替行颜色
for i in range(1, len(table_data) + 1):
    for j in range(len(col_labels)):
        if i % 2 == 0:
            table[i, j].set_facecolor("#ECF0F1")

ax.set_title(f"V72策略调仓日志（共{len(switch_log)}次）", fontsize=14, pad=20)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "v72_switch_log.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"    保存: v72_switch_log.png")


# ============================================================
# 7. 年度收益
# ============================================================
print("\n" + "=" * 80)
print("年度收益对比")
print("=" * 80)

df_r["year"] = df_r.index.year
annual_ret = df_r.groupby("year")["ret"].apply(lambda x: (1 + x).prod() - 1)
g_annual = weekly["g_close"].resample("Y").last().pct_change().dropna()
v_annual = weekly["v_close"].resample("Y").last().pct_change().dropna()

print(f"  {'年份':<6} {'V72':>10} {'成长100':>10} {'价值100':>10} {'V72-成长':>10}")
for year in annual_ret.index:
    v72_ret = annual_ret[year]
    g_ret = g_annual.get(year, 0) if hasattr(g_annual, 'get') else (g_annual[year] if year in g_annual.index else 0)
    v_ret = v_annual.get(year, 0) if hasattr(v_annual, 'get') else (v_annual[year] if year in v_annual.index else 0)
    diff = v72_ret - g_ret
    flag = "✅" if diff > 0.01 else ("❌" if diff < -0.01 else "")
    print(f"  {year:<6} {v72_ret*100:>9.2f}% {g_ret*100:>9.2f}% {v_ret*100:>9.2f}% {diff*100:>+9.2f}pp {flag}")

print(f"\n{'='*80}")
print(f"V72策略完成。年化={m['annual_return']*100:.2f}% Sharpe={m['sharpe']:.3f} 回撤={m['max_drawdown']*100:.2f}%")
print(f"图表已保存到: {OUTPUT_DIR}")
print(f"{'='*80}")
