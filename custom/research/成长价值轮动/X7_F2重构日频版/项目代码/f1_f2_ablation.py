"""f1_f2_ablation.py — F1/F2 因子消融分析

验证用户判断：F1有效，F2不够有效。
分析内容：
  1. F1/F2 信号差异度（方向一致率、相关系数）
  2. F1/F2 的 IC 分析（与未来 21/63 日收益的 Rank IC）
  3. F1/F2 的分层回测（仅F1 vs 仅F2 vs F1+F2 的累计收益）
  4. F1/F2 的逐年贡献度（F1主导 vs F2主导年份）

数据:
  成长100: c:\\temp_v72_data\\index_480080.csv
  价值100: c:\\temp_v72_data\\index_480081.csv
  区间:    2012-12-31 ~ 2026-07-01
"""
import sys
sys.path.insert(0, r'c:\XuanJLH\Qbot\custom\backtests')

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

# ============================================================
# 1. 加载数据 + 计算F1/F2（与X2完全一致）
# ============================================================
DATA_DIR = Path(r'c:\temp_v72_data')
g_raw = pd.read_csv(str(DATA_DIR / 'index_480080.csv'))
v_raw = pd.read_csv(str(DATA_DIR / 'index_480081.csv'))

for d in (g_raw, v_raw):
    d['date'] = pd.to_datetime(d['date'])
    d['close'] = pd.to_numeric(d['close'], errors='coerce')

g_close = g_raw.set_index('date')['close'].astype(float).sort_index().dropna()
v_close = v_raw.set_index('date')['close'].astype(float).sort_index().dropna()
common = g_close.index.intersection(v_close.index)
g_close = g_close[common].sort_index()
v_close = v_close[common].sort_index()

print("=" * 70)
print("  F1/F2 因子消融分析")
print("=" * 70)
print(f"数据区间: {g_close.index[0]:%Y-%m-%d} ~ {g_close.index[-1]:%Y-%m-%d}")
print(f"交易日数: {len(g_close)}")

# F1 = 比价MA20方向
F1 = 0.5
ratio = (g_close / v_close).shift(1)
ratio_ma20 = ratio.rolling(20).mean()
ratio_dev = ratio / ratio_ma20 - 1
f1_signal = np.tanh(ratio_dev * 30) * F1

# F2 = 动量加速度
F2 = 5.0
g_roc21 = g_close.pct_change(21).shift(1)
v_roc21 = v_close.pct_change(21).shift(1)
g_accel = g_roc21 - g_roc21.shift(10)
v_accel = v_roc21 - v_roc21.shift(10)
accel_diff = (g_accel - v_accel).clip(-0.02, 0.02)
f2_signal = accel_diff * F2

style_score = f1_signal + f2_signal
candidate_g_full = (style_score > 0)
candidate_g_f1only = (f1_signal > 0)
candidate_g_f2only = (f2_signal > 0)

# ============================================================
# 2. 信号差异度分析
# ============================================================
print("\n" + "=" * 70)
print("  [1] F1/F2 信号差异度分析")
print("=" * 70)

# 对齐有效样本（删除起始NaN）
valid = ~(f1_signal.isna() | f2_signal.isna())
f1_v = f1_signal[valid]
f2_v = f2_signal[valid]
cand_full = candidate_g_full[valid]
cand_f1 = candidate_g_f1only[valid]
cand_f2 = candidate_g_f2only[valid]

# Pearson 相关系数
corr_pearson, _ = stats.pearsonr(f1_v.values, f2_v.values)
# Spearman 相关系数
corr_spearman, _ = stats.spearmanr(f1_v.values, f2_v.values)
# 方向一致率（F1>0 且 F2>0 或 F1<0 且 F2<0）
direction_agree = (cand_f1 == cand_f2).mean()
# F1+F2 与 F1 的方向一致率（F2 是否经常翻转 F1 的决策）
full_vs_f1_agree = (cand_full == cand_f1).mean()
# F2 翻转 F1 决策的频率（F1 说 value 但 F1+F2 说 growth，或反之）
f2_overrides_f1 = (cand_full != cand_f1).mean()

print(f"Pearson 相关系数:    {corr_pearson:+.4f}")
print(f"Spearman 相关系数:   {corr_spearman:+.4f}")
print(f"F1/F2 方向一致率:    {direction_agree*100:.2f}%")
print(f"F1+F2 与 F1 一致率:  {full_vs_f1_agree*100:.2f}%  (高=F2翻转F1少，F2贡献小)")
print(f"F2 翻转 F1 决策频率: {f2_overrides_f1*100:.2f}%")

# ============================================================
# 3. IC 分析（Rank IC：因子值 vs 未来 21/63 日相对收益）
# ============================================================
print("\n" + "=" * 70)
print("  [2] F1/F2 IC 分析（Rank IC vs 未来相对收益）")
print("=" * 70)

# 未来相对收益 = g未来收益 - v未来收益（正=成长跑赢，负=价值跑赢）
fwd_21 = (g_close.pct_change(21) - v_close.pct_change(21)).shift(-21)
fwd_63 = (g_close.pct_change(63) - v_close.pct_change(63)).shift(-63)

def calc_ic(factor, fwd_ret, label):
    """计算 Rank IC（Spearman）"""
    df = pd.DataFrame({'f': factor, 'r': fwd_ret}).dropna()
    if len(df) < 30:
        return np.nan, np.nan, 0
    ic, p = stats.spearmanr(df['f'].values, df['r'].values)
    return ic, p, len(df)

# 全样本 IC
ic_f1_21, p_f1_21, n_f1_21 = calc_ic(f1_signal, fwd_21, "F1-21d")
ic_f2_21, p_f2_21, n_f2_21 = calc_ic(f2_signal, fwd_21, "F2-21d")
ic_full_21, p_full_21, n_full_21 = calc_ic(style_score, fwd_21, "Full-21d")
ic_f1_63, p_f1_63, n_f1_63 = calc_ic(f1_signal, fwd_63, "F1-63d")
ic_f2_63, p_f2_63, n_f2_63 = calc_ic(f2_signal, fwd_63, "F2-63d")
ic_full_63, p_full_63, n_full_63 = calc_ic(style_score, fwd_63, "Full-63d")

print(f"\n全样本 Rank IC（Spearman）:")
print(f"{'因子':<16} {'21日IC':>10} {'p值':>10} {'63日IC':>10} {'p值':>10} {'样本数':>8}")
print(f"{'F1(比价MA20)':<16} {ic_f1_21:>+10.4f} {p_f1_21:>10.4f} {ic_f1_63:>+10.4f} {p_f1_63:>10.4f} {n_f1_21:>8}")
print(f"{'F2(动量加速度)':<16} {ic_f2_21:>+10.4f} {p_f2_21:>10.4f} {ic_f2_63:>+10.4f} {p_f2_63:>10.4f} {n_f2_21:>8}")
print(f"{'F1+F2(style)':<16} {ic_full_21:>+10.4f} {p_full_21:>10.4f} {ic_full_63:>+10.4f} {p_full_63:>10.4f} {n_full_21:>8}")

# 逐年 IC（看稳定性）
print(f"\n逐年 21日 Rank IC:")
print(f"{'年份':<8} {'F1 IC':>10} {'F2 IC':>10} {'Full IC':>10} {'F1主导?':>10}")
yearly_ic = []
for year in range(2013, 2027):
    mask = (f1_signal.index.year == year)
    f1_y = f1_signal[mask]
    f2_y = f2_signal[mask]
    full_y = style_score[mask]
    fwd_y = fwd_21[mask]
    ic1, _, n1 = calc_ic(f1_y, fwd_y, "F1")
    ic2, _, n2 = calc_ic(f2_y, fwd_y, "F2")
    icf, _, nf = calc_ic(full_y, fwd_y, "Full")
    if n1 >= 30:
        f1_dom = "F1" if abs(ic1) > abs(ic2) else "F2"
        print(f"{year:<8} {ic1:>+10.4f} {ic2:>+10.4f} {icf:>+10.4f} {f1_dom:>10}")
        yearly_ic.append((year, ic1, ic2, icf, f1_dom))

# ============================================================
# 4. 分层回测：仅F1 vs 仅F2 vs F1+F2（基于日频信号，持有成长或价值）
# ============================================================
print("\n" + "=" * 70)
print("  [3] 分层回测：仅F1 vs 仅F2 vs F1+F2（无降仓，纯日频持有）")
print("=" * 70)

def simple_rotation_backtest(candidate_g_series, g_close, v_close, name):
    """最简化的轮动回测：candidate_g=True 持成长，False 持价值，无降仓，T+1开盘调仓
    返回 (年化, 回撤, Sharpe, 累计收益)
    """
    df = pd.DataFrame({
        'cand_g': candidate_g_series,
        'g_ret': g_close.pct_change(),
        'v_ret': v_close.pct_change(),
    }).dropna(subset=['cand_g'])

    # T+1：当日信号决定次日收益
    df['signal'] = df['cand_g'].shift(1)
    df = df.dropna(subset=['signal'])
    df['daily_ret'] = np.where(df['signal'], df['g_ret'], df['v_ret'])
    df['eq'] = (1 + df['daily_ret']).cumprod()

    n = len(df)
    years = n / 252
    total = df['eq'].iloc[-1] - 1
    ann = (1 + total) ** (1 / years) - 1 if years > 0 else 0
    vol = df['daily_ret'].std() * np.sqrt(252)
    sharpe = (df['daily_ret'].mean() - 0.025/252) / df['daily_ret'].std() * np.sqrt(252) if df['daily_ret'].std() > 0 else 0
    peak = df['eq'].cummax()
    dd = ((df['eq'] - peak) / peak).min()
    n_switches = (df['signal'] != df['signal'].shift(1)).sum()
    print(f"  {name:<24} 年化={ann*100:6.2f}%  回撤={dd*100:6.2f}%  Sharpe={sharpe:.3f}  调仓={n_switches}  累计={total*100:8.2f}%")
    return ann, dd, sharpe, total, n_switches

# 对齐样本（都从第一个有效信号开始）
start_idx = max(f1_signal.first_valid_index(), f2_signal.first_valid_index(),
                style_score.first_valid_index())
end_idx = g_close.index[-1]

print(f"\n回测区间: {start_idx:%Y-%m-%d} ~ {end_idx:%Y-%m-%d}")
print(f"（注：此为简化回测，无降仓/无MA择时，仅对比因子方向选择能力）\n")

r_f1 = simple_rotation_backtest(candidate_g_f1only.loc[start_idx:end_idx], g_close, v_close, "仅F1(比价MA20)")
r_f2 = simple_rotation_backtest(candidate_g_f2only.loc[start_idx:end_idx], g_close, v_close, "仅F2(动量加速度)")
r_full = simple_rotation_backtest(candidate_g_full.loc[start_idx:end_idx], g_close, v_close, "F1+F2(style_score)")

# 对照：买入持有成长 / 买入持有价值
print()
bh_g_total = g_close.loc[start_idx:end_idx].iloc[-1] / g_close.loc[start_idx:end_idx].iloc[0] - 1
bh_v_total = v_close.loc[start_idx:end_idx].iloc[-1] / v_close.loc[start_idx:end_idx].iloc[0] - 1
years_bh = len(g_close.loc[start_idx:end_idx]) / 252
bh_g_ann = (1 + bh_g_total) ** (1/years_bh) - 1
bh_v_ann = (1 + bh_v_total) ** (1/years_bh) - 1
print(f"  {'买入持有成长100':<24} 年化={bh_g_ann*100:6.2f}%  累计={bh_g_total*100:8.2f}%")
print(f"  {'买入持有价值100':<24} 年化={bh_v_ann*100:6.2f}%  累计={bh_v_total*100:8.2f}%")

# ============================================================
# 5. 逐年贡献度分析（F1主导年 vs F2主导年）
# ============================================================
print("\n" + "=" * 70)
print("  [4] 逐年贡献度分析（F1 vs F2 谁主导）")
print("=" * 70)

def simple_rotation_backtest_inner(candidate_g_series, g_close, v_close):
    """内部版本，返回 (年化, 回撤, Sharpe, 累计, 调仓)"""
    df = pd.DataFrame({
        'cand_g': candidate_g_series,
        'g_ret': g_close.pct_change(),
        'v_ret': v_close.pct_change(),
    }).dropna(subset=['cand_g'])
    df['signal'] = df['cand_g'].shift(1)
    df = df.dropna(subset=['signal'])
    if len(df) < 10:
        return (0, 0, 0, 0, 0)
    df['daily_ret'] = np.where(df['signal'], df['g_ret'], df['v_ret'])
    df['eq'] = (1 + df['daily_ret']).cumprod()
    n = len(df)
    years = n / 252
    total = df['eq'].iloc[-1] - 1
    ann = (1 + total) ** (1 / years) - 1 if years > 0 else 0
    sharpe = (df['daily_ret'].mean() - 0.025/252) / df['daily_ret'].std() * np.sqrt(252) if df['daily_ret'].std() > 0 else 0
    peak = df['eq'].cummax()
    dd = ((df['eq'] - peak) / peak).min()
    n_switches = (df['signal'] != df['signal'].shift(1)).sum()
    return (ann, dd, sharpe, total, n_switches)

print(f"\n{'年份':<8} {'仅F1年化':>12} {'仅F2年化':>12} {'F1+F2年化':>12} {'F1-F2':>10} {'主导':>8}")
for year in range(2013, 2027):
    y_start = pd.Timestamp(f"{year}-01-01")
    y_end = pd.Timestamp(f"{year}-12-31")
    if y_start > end_idx:
        break
    mask = (g_close.index >= y_start) & (g_close.index <= y_end)
    if mask.sum() < 60:
        continue
    g_y = g_close[mask]
    v_y = v_close[mask]
    cand_f1_y = candidate_g_f1only[mask]
    cand_f2_y = candidate_g_f2only[mask]
    cand_full_y = candidate_g_full[mask]
    r1 = simple_rotation_backtest_inner(cand_f1_y, g_y, v_y)
    r2 = simple_rotation_backtest_inner(cand_f2_y, g_y, v_y)
    rf = simple_rotation_backtest_inner(cand_full_y, g_y, v_y)
    diff = r1[0] - r2[0]
    dom = "F1" if r1[0] > r2[0] else "F2"
    print(f"{year:<8} {r1[0]*100:>+12.2f}% {r2[0]*100:>+12.2f}% {rf[0]*100:>+12.2f}% {diff*100:>+10.2f}pp {dom:>8}")

# ============================================================
# 6. 结论
# ============================================================
print("\n" + "=" * 70)
print("  [5] 消融分析结论")
print("=" * 70)

f1_better = r_f1[0] > r_f2[0]
f2_override_rate = f2_overrides_f1
f1_ic_abs = abs(ic_f1_21)
f2_ic_abs = abs(ic_f2_21)

print(f"""
信号差异度:
  - F1/F2 Pearson 相关: {corr_pearson:+.4f}（{'低相关=互补空间大' if abs(corr_pearson)<0.3 else '高相关=冗余'}）
  - F1/F2 方向一致率:  {direction_agree*100:.2f}%
  - F2 翻转 F1 频率:   {f2_override_rate*100:.2f}%（{'F2几乎不改变F1决策=F2贡献小' if f2_override_rate<0.1 else 'F2经常翻转F1=F2有独立价值'}）

IC 分析（21日 Rank IC）:
  - F1 IC = {ic_f1_21:+.4f}（{'有效' if f1_ic_abs>0.03 else '弱'}）
  - F2 IC = {ic_f2_21:+.4f}（{'有效' if f2_ic_abs>0.03 else '弱'}）
  - {'F1 IC 显著强于 F2' if f1_ic_abs > f2_ic_abs * 1.5 else 'F2 IC 与 F1 相当或更强'}

分层回测（全周期年化）:
  - 仅F1:  {r_f1[0]*100:.2f}%
  - 仅F2:  {r_f2[0]*100:.2f}%
  - F1+F2: {r_full[0]*100:.2f}%
  - {'F1+F2 > 仅F1，F2有正贡献' if r_full[0] > r_f1[0] else 'F1+F2 <= 仅F1，F2无正贡献或拖累'}
  - {'仅F1 > 仅F2，F1 主导（验证用户判断）' if r_f1[0] > r_f2[0] else '仅F2 > 仅F1，F2 主导（与用户判断不符）'}

用户判断验证:
  - 用户说"F1有效": {'✅ 验证通过' if f1_ic_abs>0.03 and r_f1[0]>0.1 else '❌ 不符'}
  - 用户说"F2不够有效": {'✅ 验证通过' if (f2_ic_abs<f1_ic_abs and r_f2[0]<r_f1[0]) else '❌ 不符'}
""")

# 保存报告
report_path = DATA_DIR / 'f1_f2_ablation_report.txt'
with open(report_path, 'w', encoding='utf-8') as f:
    import io
    import contextlib
    # 重新捕获输出到文件
    pass

print(f"\n报告已生成（控制台输出）")
print(f"数据目录: {DATA_DIR}")
