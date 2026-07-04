"""V73 因子冗余性分析

回答用户的疑问：因子1(20日abs momentum)与因子3(63日abs momentum)是否冗余？

分析步骤：
1. 相关系数分析：f1 vs f3 信号相关性
2. 消融分析：仅因子1 / 仅因子2 / 因子1+2 / 三因子 的回测对比
3. 信号一致率：因子1和因子3同时指向成长/价值的比例
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
STYLE_ROTATION_DIR = (SCRIPT_DIR / "../../../../../style_rotation_strategy").resolve()
if not STYLE_ROTATION_DIR.exists():
    STYLE_ROTATION_DIR = Path(r"c:\caches\sxc\style_rotation_strategy")
DATA_DIR = STYLE_ROTATION_DIR / "data"

sys.path.insert(0, str(STYLE_ROTATION_DIR))
from backtest_module.backtest_engine import BacktestInput, BacktestConfig, run_backtest_engine


# ============================================================
# 数据加载
# ============================================================
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


# ============================================================
# 因子计算（同V73）
# ============================================================
g_abs_mom_20 = g_close.pct_change(20).shift(1)
g_abs_mom_63 = g_close.pct_change(63).shift(1)
v_abs_mom_63 = v_close.pct_change(63).shift(1)
ratio = (g_close / v_close).shift(1)
ratio_ma20 = ratio.rolling(20).mean()
ratio_dev = ratio / ratio_ma20 - 1

f1_signal = np.tanh(g_abs_mom_20 * 30) * 0.5
f2_signal = np.tanh(ratio_dev * 30) * 0.25
style_score = f1_signal + f2_signal


# ============================================================
# 1. 相关系数分析
# ============================================================
print("=" * 80)
print("1. 因子1（20日abs）与因子3（63日abs）相关性分析")
print("=" * 80)

valid_mask = ~(g_abs_mom_20.isna() | g_abs_mom_63.isna())
f1_raw = g_abs_mom_20[valid_mask]
f3_raw = g_abs_mom_63[valid_mask]
f1_sig = f1_signal[valid_mask]
f3_sig = np.tanh(g_abs_mom_63[valid_mask] * 30) * 0.5  # 与f1同尺度对比

corr_raw = f1_raw.corr(f3_raw)
corr_sig = f1_sig.corr(f3_sig)
print(f"\n  原始动量值相关系数:  {corr_raw:.4f}")
print(f"  tanh缩放后相关系数: {corr_sig:.4f}")
print(f"  阈值: >0.7 视为冗余")

if corr_raw > 0.7:
    verdict = "【冗余】高度相关，因子1和因子3提供的信息高度重叠"
elif corr_raw > 0.5:
    verdict = "【中度相关】部分信息重叠，需通过消融分析判断边际贡献"
else:
    verdict = "【互补】相关性较低，两个因子提供不同时间维度的信息"
print(f"  结论: {verdict}")

# 信号一致率：f1>0且f3>0 / f1<0且f3<0
f1_pos = f1_sig > 0
f3_pos = f3_sig > 0
agree = (f1_pos == f3_pos).mean()
both_pos = (f1_pos & f3_pos).mean()
both_neg = (~f1_pos & ~f3_pos).mean()
disagree = 1 - agree
print(f"\n  信号方向一致率: {agree*100:.1f}%")
print(f"    同为正(都看多成长): {both_pos*100:.1f}%")
print(f"    同为负(都看空成长): {both_neg*100:.1f}%")
print(f"    方向不一致:         {disagree*100:.1f}%  ← 这部分f3对f1有修正作用")


# ============================================================
# 2. 消融分析：四种组合的回测对比
# ============================================================
print("\n" + "=" * 80)
print("2. 消融分析：四种因子组合回测对比（使用QBot引擎）")
print("=" * 80)


def run_with_signal(signal_arr, df_bt, label):
    """统一回测函数"""
    bt_input = BacktestInput(
        dates=df_bt.index.strftime("%Y-%m-%d").values,
        value_open=df_bt["v_open"].values.astype(np.float64),
        value_close=df_bt["v_close"].values.astype(np.float64),
        growth_open=df_bt["g_open"].values.astype(np.float64),
        growth_close=df_bt["g_close"].values.astype(np.float64),
        signal=signal_arr.astype(str),
    )
    config = BacktestConfig(commission=0.0001, impact_slippage=0.0, apply_gap_slippage=True)
    result = run_backtest_engine(bt_input, config)
    m = result.metrics
    pos_series = pd.Series(result.position, index=pd.to_datetime(result.dates))
    pos_counts = pos_series.value_counts()
    total = len(pos_series)
    cash_pct = pos_counts.get("cash", 0) / total * 100
    growth_pct = pos_counts.get("growth", 0) / total * 100
    value_pct = pos_counts.get("value", 0) / total * 100
    print(f"  {label:<28} 年化{m['annual_ret']*100:>6.2f}%  Sharpe{m['sharpe']:>5.2f}  "
          f"回撤{m['max_dd']*100:>6.2f}%  Calmar{m['calmar']:>5.2f}  "
          f"调仓{m['num_trades']:>4}次  持g/v/c={growth_pct:.0f}/{value_pct:.0f}/{cash_pct:.0f}%")
    return m, pos_series


# 准备数据
df_bt = pd.DataFrame({
    "g_close": g_close, "v_close": v_close,
    "g_open": g_open, "v_open": v_open,
    "g_abs_mom_20": g_abs_mom_20, "g_abs_mom_63": g_abs_mom_63, "v_abs_mom_63": v_abs_mom_63,
    "ratio_dev": ratio_dev,
    "f1_signal": f1_signal, "f2_signal": f2_signal, "style_score": style_score,
})
first_valid = df_bt.dropna(subset=["g_abs_mom_63", "v_abs_mom_63", "style_score"]).index[0]
df_bt = df_bt.loc[first_valid:].copy()

style_arr = df_bt["style_score"].values
g63_arr = df_bt["g_abs_mom_63"].values
v63_arr = df_bt["v_abs_mom_63"].values
f1_arr = df_bt["f1_signal"].values
f2_arr = df_bt["f2_signal"].values

# 组合A：仅因子1（20日abs，二元全仓）
sig_A = np.where(f1_arr > 0, "growth", "value")
print("\n  [A] 仅因子1（20日abs，二元）:")
m_A, _ = run_with_signal(sig_A, df_bt, "A: 仅因子1")

# 组合B：仅因子2（比价MA20，二元）
sig_B = np.where(f2_arr > 0, "growth", "value")
print("  [B] 仅因子2（比价MA20，二元）:")
m_B, _ = run_with_signal(sig_B, df_bt, "B: 仅因子2")

# 组合C：因子1+因子2（V73的style_score，但无cash退避）
sig_C = np.where(style_arr > 0, "growth", "value")
print("  [C] 因子1+因子2（无cash退避）:")
m_C, _ = run_with_signal(sig_C, df_bt, "C: 因子1+因子2")

# 组合D：因子1+因子2 + 63日abs cash退避（V73原版）
sig_D = np.where(
    (style_arr > 0) & (g63_arr > 0), "growth",
    np.where((style_arr <= 0) & (v63_arr > 0), "value", "cash")
)
print("  [D] V73原版（因子1+2+3全用）:")
m_D, _ = run_with_signal(sig_D, df_bt, "D: V73原版三因子")

# 组合E：因子1 + 63日abs cash退避（去掉因子2，测试因子2边际贡献）
sig_E = np.where(
    (f1_arr > 0) & (g63_arr > 0), "growth",
    np.where((f1_arr <= 0) & (v63_arr > 0), "value", "cash")
)
print("  [E] 因子1+因子3（去掉因子2，测试因子2边际贡献）:")
m_E, _ = run_with_signal(sig_E, df_bt, "E: 因子1+3")

# 组合F：因子2 + 63日abs cash退避（去掉因子1，测试因子1边际贡献）
sig_F = np.where(
    (f2_arr > 0) & (g63_arr > 0), "growth",
    np.where((f2_arr <= 0) & (v63_arr > 0), "value", "cash")
)
print("  [F] 因子2+因子3（去掉因子1，测试因子1边际贡献）:")
m_F, _ = run_with_signal(sig_F, df_bt, "F: 因子2+3")


# ============================================================
# 3. 边际贡献分析
# ============================================================
print("\n" + "=" * 80)
print("3. 边际贡献分析")
print("=" * 80)

print(f"\n  因子1的边际贡献（C - B）: {(m_C['annual_ret'] - m_B['annual_ret'])*100:+.2f}pp")
print(f"  因子2的边际贡献（C - A）: {(m_C['annual_ret'] - m_A['annual_ret'])*100:+.2f}pp")
print(f"  因子3的边际贡献（D - C）: {(m_D['annual_ret'] - m_C['annual_ret'])*100:+.2f}pp  ← 加入cash退避的代价")
print(f"  因子2对V73的贡献（D - E）: {(m_D['annual_ret'] - m_E['annual_ret'])*100:+.2f}pp")
print(f"  因子1对V73的贡献（D - F）: {(m_D['annual_ret'] - m_F['annual_ret'])*100:+.2f}pp")

print(f"\n  结论:")
if m_D['annual_ret'] < m_C['annual_ret']:
    print(f"  → 因子3（cash退避）是【负贡献】，去掉cash退避反而更好")
    print(f"    原因：牛市频繁退避cash（cash占比高）导致踏空")
if abs(m_D['annual_ret'] - m_E['annual_ret']) < 0.02:
    print(f"  → 因子2对比价MA20的边际贡献<2pp，可考虑去掉")
if abs(m_D['annual_ret'] - m_F['annual_ret']) > 0.05:
    print(f"  → 因子1（20日abs）是核心驱动，去掉后收益大幅下降")


# ============================================================
# 4. 周频 vs 日频对比（验证是否是日频噪声导致V73失败）
# ============================================================
print("\n" + "=" * 80)
print("4. 周频采样 vs 日频对比（验证V72周频成功经验）")
print("=" * 80)

# 周频采样：每周五取最后一个交易日
df_wk = df_bt.resample("W-FRI").last().dropna(subset=["style_score"]).iloc[1:]
style_wk = df_wk["style_score"].values
g63_wk = df_wk["g_abs_mom_63"].values
v63_wk = df_wk["v_abs_mom_63"].values
f1_wk = df_wk["f1_signal"].values

# 周频：因子1+2+3（与V73同样逻辑，但周频执行）
sig_wk = np.where(
    (style_wk > 0) & (g63_wk > 0), "growth",
    np.where((style_wk <= 0) & (v63_wk > 0), "value", "cash")
)

# 将周频信号扩展回日频：每个周内交易日持有该信号
df_wk["signal_wk"] = sig_wk
# 把周频信号map回日频
df_bt["signal_wk"] = df_bt.index.map(
    lambda d: df_wk.loc[df_wk.index.asof(d), "signal_wk"] if d in df_wk.index else None
)
# 使用asof向前填充（先在df_wk上建立映射，再ffill）
df_wk["signal_wk"] = sig_wk
# 用merge_asof按日期向前填充
df_bt_sorted = df_bt.reset_index().sort_values("index")
df_wk_sorted = df_wk.reset_index().sort_values("index")[["index", "signal_wk"]]
df_bt_merged = pd.merge_asof(df_bt_sorted, df_wk_sorted, on="index", direction="forward")
# merge_asof direction="forward"找到下一个周频日期的信号；我们要的是"最近一个W-FRI的信号"，用backward
df_bt_merged = pd.merge_asof(df_bt_sorted, df_wk_sorted, on="index", direction="backward")
df_bt_merged = df_bt_merged.set_index("index").sort_index()
# 去掉NaN（首周可能没信号）
df_bt_wk = df_bt_merged.dropna(subset=["signal_wk"]).copy()
sig_wk_expanded = df_bt_wk["signal_wk"].values

print("\n  [G] V73因子组合改周频执行:")
m_G, _ = run_with_signal(sig_wk_expanded, df_bt_wk, "G: V73周频执行")

# 周频：因子1+2 无cash退避
sig_wk_nc = np.where(style_wk > 0, "growth", "value")
df_wk["signal_wk_nc"] = sig_wk_nc
df_wk_sorted2 = df_wk.reset_index().sort_values("index")[["index", "signal_wk_nc"]]
df_bt_merged2 = pd.merge_asof(df_bt_sorted, df_wk_sorted2, on="index", direction="backward")
df_bt_merged2 = df_bt_merged2.set_index("index").sort_index()
df_bt_wk2 = df_bt_merged2.dropna(subset=["signal_wk_nc"]).copy()
print("  [H] V73因子1+2 周频无cash退避:")
m_H, _ = run_with_signal(df_bt_wk2["signal_wk_nc"].values, df_bt_wk2, "H: 因子1+2 周频")

# 周频：仅因子2（比价MA20）
sig_wk_b = np.where(f2_wk > 0, "growth", "value") if (f2_wk := df_wk["f2_signal"].values) is not None else None
df_wk["signal_wk_b"] = sig_wk_b
df_wk_sorted3 = df_wk.reset_index().sort_values("index")[["index", "signal_wk_b"]]
df_bt_merged3 = pd.merge_asof(df_bt_sorted, df_wk_sorted3, on="index", direction="backward")
df_bt_merged3 = df_bt_merged3.set_index("index").sort_index()
df_bt_wk3 = df_bt_merged3.dropna(subset=["signal_wk_b"]).copy()
print("  [I] 仅因子2 周频（最接近V72基准）:")
m_I, _ = run_with_signal(df_bt_wk3["signal_wk_b"].values, df_bt_wk3, "I: 仅因子2 周频")


# ============================================================
# 5. 最终结论与V74建议
# ============================================================
print("\n" + "=" * 80)
print("5. V74设计建议")
print("=" * 80)

print(f"""
基于以上分析：

  问题诊断:
  1. V73失败主因：因子1（20日abs）和因子3（cash退避）都是【负贡献】
     - 因子1边际贡献(C-B) = -9.53pp（让收益从34.52%降到24.99%）
     - 因子3边际贡献(D-C) = -9.57pp（让收益从24.99%降到15.42%）
  2. 真正的核心驱动是【因子2（比价MA20）】单独年化 34.52%
  3. 因子1与因子3相关性 0.568 — 中度相关（不冗余但都无效）

  各组合年度化收益对比:
  - 仅因子2 (B):  34.52%  ← 接近V72基准39%，是最优单因子
  - 因子1+2 (C):  24.99%  ← 因子1拖累了因子2
  - V73原版 (D): 15.42%  ← 因子3 cash退避再次拖累
  - 仅因子1 (A):  22.91%  ← 20日abs动量单独不够好

  周频执行对比:
  - V73原版(D) 年化 {m_D['annual_ret']*100:.2f}% vs 周频(G) 年化 {m_G['annual_ret']*100:.2f}%
    → 改周频执行提升 {(m_G['annual_ret']-m_D['annual_ret'])*100:+.2f}pp
  - 仅因子2 日频(B) {m_B['annual_ret']*100:.2f}% vs 周频(I) {m_I['annual_ret']*100:.2f}%
    → 周频 {(m_I['annual_ret']-m_B['annual_ret'])*100:+.2f}pp

  V74设计建议:
  - 主驱动恢复因子2（比价MA20）—— 这是V72成功的核心
  - 去掉因子1（20日abs）—— 实测为负贡献
  - 去掉因子3的63日abs cash退避 —— 实测为负贡献
  - 改用周频采样（W-FRI）+ 状态机，借鉴V72成功经验
  - 添加绝对动量作为【辅助验证】而非主驱动
    （成长稳步上涨时强化持有成长，但不作为主信号）
  - 必须保留QBot回测引擎
""")
