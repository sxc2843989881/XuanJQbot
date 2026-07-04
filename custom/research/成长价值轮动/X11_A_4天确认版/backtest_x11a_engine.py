"""x11a_full_validation.py — X11-A完整验证
================================================================
X11-A = X10 + 3天确认（替代2天）

验证内容：
  1. 参数稳健性（2/3/4/5天确认）
  2. 最大回撤期分析
  3. 年度收益分析
  4. 完整回撤图表
  5. 与X10/F1的详细对比
================================================================
"""
import sys
sys.path.insert(0, r'c:\XuanJLH\Qbot\custom\backtests')

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

from backtest_engine import (
    BacktestInput, BacktestConfig,
    run_backtest_engine_weighted,
)

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150

DATA_DIR = Path(r'c:\temp_v72_data')
OUTPUT_DIR = Path(r'c:\temp_v72_data\charts')

# ============================================================
# 1. 加载数据 & 信号生成
# ============================================================
print("=" * 70)
print("  X11-A完整验证")
print("=" * 70)

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

ratio = g_close / v_close
ratio_ma20 = ratio.rolling(20).mean()
ratio_dev = ratio / ratio_ma20 - 1
f1_signal = np.tanh(ratio_dev * 30) * 0.5
base_dir = (f1_signal > 0).map({True: 'growth', False: 'value'})

ma20_slope = (ratio_ma20 - ratio_ma20.shift(5)) / ratio_ma20.shift(5)
slope_ok = ma20_slope.abs() > 0.003
v_mom20 = v_close.pct_change(20)
g_dd20 = g_close / g_close.shift(20) - 1
v_dd20 = v_close / v_close.shift(20) - 1

# ============================================================
# 2. 回测函数
# ============================================================
def run_daily(signal_series, weight_series):
    common_idx = signal_series.index.intersection(g_close.index)
    sig = signal_series.loc[common_idx]
    wt = weight_series.loc[common_idx]
    g_a = g_close.loc[common_idx]
    v_a = v_close.loc[common_idx]
    mask = ~(sig.isna() | wt.isna())
    sig = sig[mask].astype(str)
    wt = wt[mask].astype(float)
    g_a = g_a[mask]
    v_a = v_a[mask]
    bt_input = BacktestInput(
        dates=sig.index.strftime('%Y-%m-%d').values,
        value_open=v_a.values.astype(np.float64),
        value_close=v_a.values.astype(np.float64),
        growth_open=g_a.values.astype(np.float64),
        growth_close=g_a.values.astype(np.float64),
        signal=sig.values,
    )
    config = BacktestConfig(start_cash=1_000_000.0, commission=0.0001,
                            impact_slippage=0.0, apply_gap_slippage=False)
    return run_backtest_engine_weighted(bt_input, config, wt.values)

def calc_metrics(result, freq=252, rf_annual=0.025):
    df_r = result.to_dataframe()
    r = df_r['daily_ret']
    eq = (1 + r).cumprod()
    n = len(r)
    years = n / freq
    total = eq.iloc[-1] - 1
    ann = (1 + total) ** (1 / years) - 1 if years > 0 else 0
    rf_p = rf_annual / freq
    sharpe = (r.mean() - rf_p) / r.std() * np.sqrt(freq) if r.std() > 0 else 0
    peak = eq.cummax()
    dd = (eq - peak) / peak
    max_dd = dd.min()
    calmar = ann / abs(max_dd) if max_dd < 0 else 0
    return {'ann': ann, 'dd': max_dd, 'sharpe': sharpe, 'calmar': calmar,
            'total': total, 'n_trades': result.metrics['num_trades']}

def build_signal(n_confirm_days):
    """构建X11信号：n天确认 + 斜率 + B2 + E5止损"""
    dir_s = base_dir.copy()
    # n天确认
    if n_confirm_days == 1:
        confirmed = dir_s
    elif n_confirm_days == 2:
        confirmed = dir_s.where(dir_s == dir_s.shift(1), np.nan)
    else:
        mask = np.ones(len(dir_s), dtype=bool)
        for k in range(1, n_confirm_days):
            mask = mask & (dir_s.values == dir_s.shift(k).values)
        confirmed = dir_s.where(mask, np.nan)
    dir_s = confirmed.where(slope_ok, np.nan).ffill()
    # B2
    wrong_value = (dir_s == 'value') & (v_mom20 <= 0)
    dir_s[wrong_value] = 'growth'
    # E5止损
    wt = pd.Series(1.0, index=dir_s.index)
    gs = (dir_s == 'growth') & (g_dd20 < -0.10)
    vs = (dir_s == 'value') & (v_dd20 < -0.10)
    wt[gs | vs] = 0.3
    return dir_s, wt

# ============================================================
# 3. 参数稳健性验证（确认天数）
# ============================================================
print("\n" + "=" * 70)
print("  参数稳健性验证（确认天数）")
print("=" * 70)

robustness_results = {}
for n_days in [1, 2, 3, 4, 5, 6, 7]:
    dir_s, wt_s = build_signal(n_days)
    r = run_daily(dir_s, wt_s)
    m = calc_metrics(r)
    robustness_results[n_days] = m
    pass_flag = "✅" if m['ann'] * 100 >= 36.0 else "❌"
    print(f"  {n_days}天确认: 年化={m['ann']*100:.2f}% 回撤={m['dd']*100:.2f}% "
          f"Sharpe={m['sharpe']:.3f} Calmar={m['calmar']:.3f} 交易={m['n_trades']} {pass_flag}")

# 找最优
best_n = max(robustness_results, key=lambda x: robustness_results[x]['sharpe'])
print(f"\n最优确认天数: {best_n}天 (Sharpe={robustness_results[best_n]['sharpe']:.3f})")

# ============================================================
# 4. X11-A完整分析（使用最优确认天数）
# ============================================================
print("\n" + "=" * 70)
print(f"  X11-A完整分析（{best_n}天确认）")
print("=" * 70)

x11a_dir, x11a_wt = build_signal(best_n)
r_x11a = run_daily(x11a_dir, x11a_wt)
m_x11a = calc_metrics(r_x11a)

print(f"年化={m_x11a['ann']*100:.2f}% 回撤={m_x11a['dd']*100:.2f}% "
      f"Sharpe={m_x11a['sharpe']:.3f} Calmar={m_x11a['calmar']:.3f} 交易={m_x11a['n_trades']}")

# 获取日收益率
df_x11a = r_x11a.to_dataframe()
df_x11a['date'] = pd.to_datetime(df_x11a['date'])
df_x11a = df_x11a.set_index('date')
daily_ret = df_x11a['daily_ret']
nav = (1 + daily_ret).cumprod() * 1_000_000
peak = nav.cummax()
dd = (nav - peak) / peak

# ============================================================
# 5. 最大回撤期分析
# ============================================================
print("\n" + "=" * 70)
print("  最大回撤期分析")
print("=" * 70)

max_dd = dd.min()
max_dd_date = dd.idxmin()
peak_date = peak.loc[:max_dd_date].idxmax()
recovery_mask = (dd.index > max_dd_date) & (dd >= -0.001)
recovery_date = dd.index[recovery_mask][0] if recovery_mask.any() else None

print(f"峰值日期: {peak_date:%Y-%m-%d}, 净值: ¥{peak.loc[peak_date]:,.0f}")
print(f"谷底日期: {max_dd_date:%Y-%m-%d}, 净值: ¥{nav.loc[max_dd_date]:,.0f}")
print(f"最大回撤: {max_dd*100:.2f}%")
if recovery_date:
    print(f"恢复日期: {recovery_date:%Y-%m-%d}")
    print(f"持续天数: {(recovery_date - peak_date).days}天")

# ============================================================
# 6. 年度收益分析
# ============================================================
print("\n" + "=" * 70)
print("  年度收益分析")
print("=" * 70)

yearly_ret = daily_ret.resample('Y').apply(lambda x: (1 + x).prod() - 1)
print(f"{'年份':>6} {'收益':>10} {'是否盈利':>10}")
for yr, r in yearly_ret.items():
    flag = "✅" if r > 0 else "❌"
    print(f"{yr.year:>6} {r*100:>9.2f}% {flag:>10}")

win_years = (yearly_ret > 0).sum()
print(f"\n年胜率: {win_years}/{len(yearly_ret)} = {win_years/len(yearly_ret)*100:.1f}%")
print(f"年平均收益: {yearly_ret.mean()*100:.2f}%")
print(f"年最佳: {yearly_ret.max()*100:.2f}% ({yearly_ret.idxmax().year})")
print(f"年最差: {yearly_ret.min()*100:.2f}% ({yearly_ret.idxmin().year})")

# ============================================================
# 7. 月度胜率
# ============================================================
monthly_ret = daily_ret.resample('M').apply(lambda x: (1 + x).prod() - 1)
print(f"\n月胜率: {(monthly_ret > 0).sum()}/{len(monthly_ret)} = {(monthly_ret > 0).mean()*100:.1f}%")

# ============================================================
# 8. 生成完整图表
# ============================================================
print("\n" + "=" * 70)
print("  生成完整图表")
print("=" * 70)

# --- 图表1: 参数稳健性 ---
print("绘制图表1: 参数稳健性...")
fig, axes = plt.subplots(2, 2, figsize=(18, 12))

n_days_list = list(robustness_results.keys())
anns = [robustness_results[n]['ann'] * 100 for n in n_days_list]
dds = [robustness_results[n]['dd'] * 100 for n in n_days_list]
sharpes = [robustness_results[n]['sharpe'] for n in n_days_list]
calmars = [robustness_results[n]['calmar'] for n in n_days_list]

ax = axes[0, 0]
bars = ax.bar(n_days_list, anns, color=['#E74C3C' if a < 36 else '#27AE60' for a in anns], alpha=0.8)
for i, v in enumerate(anns):
    ax.text(n_days_list[i], v + 0.3, f'{v:.2f}%', ha='center', fontweight='bold', fontsize=9)
ax.axhline(y=36, color='red', linewidth=1.5, linestyle='--', alpha=0.7)
ax.text(max(n_days_list), 36.5, '36%目标', fontsize=9, color='red', ha='right')
ax.set_title('年化收益率 vs 确认天数', fontsize=13, fontweight='bold')
ax.set_xlabel('确认天数')
ax.set_ylabel('年化 (%)')
ax.grid(True, alpha=0.3)

ax = axes[0, 1]
bars = ax.bar(n_days_list, dds, color='#3498DB', alpha=0.8)
for i, v in enumerate(dds):
    ax.text(n_days_list[i], v - 1, f'{v:.1f}%', ha='center', va='top', fontweight='bold', fontsize=9, color='white')
ax.set_title('最大回撤 vs 确认天数', fontsize=13, fontweight='bold')
ax.set_xlabel('确认天数')
ax.set_ylabel('回撤 (%)')
ax.grid(True, alpha=0.3)

ax = axes[1, 0]
bars = ax.bar(n_days_list, sharpes, color='#9B59B6', alpha=0.8)
for i, v in enumerate(sharpes):
    ax.text(n_days_list[i], v + 0.01, f'{v:.3f}', ha='center', fontweight='bold', fontsize=9)
ax.axhline(y=1.271, color='red', linewidth=1, linestyle='--', alpha=0.5)
ax.text(max(n_days_list), 1.28, 'X2-A基准', fontsize=9, color='red', ha='right')
ax.set_title('Sharpe vs 确认天数', fontsize=13, fontweight='bold')
ax.set_xlabel('确认天数')
ax.set_ylabel('Sharpe')
ax.grid(True, alpha=0.3)

ax = axes[1, 1]
bars = ax.bar(n_days_list, calmars, color='#E67E22', alpha=0.8)
for i, v in enumerate(calmars):
    ax.text(n_days_list[i], v + 0.01, f'{v:.3f}', ha='center', fontweight='bold', fontsize=9)
ax.axhline(y=1.0, color='red', linewidth=1, linestyle='--', alpha=0.5)
ax.set_title('Calmar vs 确认天数', fontsize=13, fontweight='bold')
ax.set_xlabel('确认天数')
ax.set_ylabel('Calmar')
ax.grid(True, alpha=0.3)

fig.suptitle(f'X11-A参数稳健性验证 — 确认天数扫描', fontsize=15, fontweight='bold', y=1.02)
fig.tight_layout()
fig.savefig(str(OUTPUT_DIR / 'x11a_robustness.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"  已保存: x11a_robustness.png")

# --- 图表2: 净值曲线 + 回撤（标注最大回撤期）---
print("绘制图表2: 净值曲线+回撤...")

# F1和X10对比
f1_dir = base_dir.copy()
f1_wt = pd.Series(1.0, index=f1_dir.index)
r_f1 = run_daily(f1_dir, f1_wt)
df_f1 = r_f1.to_dataframe()
df_f1['date'] = pd.to_datetime(df_f1['date'])
df_f1 = df_f1.set_index('date')
nav_f1 = (1 + df_f1['daily_ret']).cumprod() * 1_000_000

x10_dir, x10_wt = build_signal(2)  # X10是2天确认
r_x10 = run_daily(x10_dir, x10_wt)
df_x10 = r_x10.to_dataframe()
df_x10['date'] = pd.to_datetime(df_x10['date'])
df_x10 = df_x10.set_index('date')
nav_x10 = (1 + df_x10['daily_ret']).cumprod() * 1_000_000
dd_x10 = (nav_x10 - nav_x10.cummax()) / nav_x10.cummax()

fig, axes = plt.subplots(2, 1, figsize=(20, 12), gridspec_kw={'height_ratios': [2, 1]})

# 上图：净值曲线
ax = axes[0]
ax.plot(nav_f1.index, nav_f1.values, color='#B0B0B0', linewidth=1.5, alpha=0.8, label='F1基准')
ax.plot(nav_x10.index, nav_x10.values, color='#96CEB4', linewidth=1.5, alpha=0.8, label='X10')
ax.plot(nav.index, nav.values, color='#E74C3C', linewidth=2.5, label=f'X11-A({best_n}天确认)★')

# 标注X11-A最大回撤期
ax.axvspan(peak_date, max_dd_date, alpha=0.2, color='#E74C3C')
ax.scatter([peak_date], [nav.loc[peak_date]], color='#27AE60', s=100, zorder=5, label=f'峰值{peak_date:%Y-%m-%d}')
ax.scatter([max_dd_date], [nav.loc[max_dd_date]], color='#E74C3C', s=100, zorder=5, label=f'谷底{max_dd_date:%Y-%m-%d}')

ax.set_title(f'X11-A({best_n}天确认) 净值曲线 — 年化{m_x11a["ann"]*100:.2f}%/回撤{m_x11a["dd"]*100:.2f}%/Sharpe{m_x11a["sharpe"]:.3f}',
             fontsize=14, fontweight='bold')
ax.set_xlabel('日期')
ax.set_ylabel('净值')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f'¥{x/1e6:.0f}M'))
ax.legend(loc='upper left', fontsize=10)
ax.grid(True, alpha=0.3)

# 下图：回撤曲线
ax = axes[1]
ax.fill_between(dd.index, 0, dd.values * 100, color='#E74C3C', alpha=0.4, label=f'X11-A回撤(最大{max_dd*100:.1f}%)')
ax.fill_between(dd_x10.index, 0, dd_x10.values * 100, color='#96CEB4', alpha=0.3, label='X10回撤')
ax.axhline(y=0, color='black', linewidth=0.5)
ax.scatter([max_dd_date], [max_dd*100], color='#E74C3C', s=100, zorder=5)
ax.set_title('回撤曲线对比', fontsize=13, fontweight='bold')
ax.set_xlabel('日期')
ax.set_ylabel('回撤 (%)')
ax.legend(loc='lower left', fontsize=10)
ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(str(OUTPUT_DIR / 'x11a_nav_drawdown.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"  已保存: x11a_nav_drawdown.png")

# --- 图表3: 年度收益对比 ---
print("绘制图表3: 年度收益对比...")
yearly_f1 = df_f1['daily_ret'].resample('Y').apply(lambda x: (1 + x).prod() - 1)
yearly_x10 = df_x10['daily_ret'].resample('Y').apply(lambda x: (1 + x).prod() - 1)

# 对齐索引（确保三个序列长度一致）
common_years = yearly_ret.index.intersection(yearly_f1.index).intersection(yearly_x10.index)
y_f1 = yearly_f1.loc[common_years]
y_x10 = yearly_x10.loc[common_years]
y_x11 = yearly_ret.loc[common_years]

fig, ax = plt.subplots(figsize=(18, 8))
n_years = len(common_years)
x = np.arange(n_years)
w = 0.27
bars1 = ax.bar(x - w, [r * 100 for r in y_f1.values], w, color='#B0B0B0', alpha=0.8, label='F1基准')
bars2 = ax.bar(x, [r * 100 for r in y_x10.values], w, color='#96CEB4', alpha=0.8, label='X10')
bars3 = ax.bar(x + w, [r * 100 for r in y_x11.values], w, color='#E74C3C', alpha=0.8, label=f'X11-A({best_n}天)')

for i, (r1, r2, r3) in enumerate(zip(y_f1.values, y_x10.values, y_x11.values)):
    ax.text(i - w, r1 * 100 + 2, f'{r1*100:.0f}', ha='center', fontsize=7, alpha=0.7)
    ax.text(i, r2 * 100 + 2, f'{r2*100:.0f}', ha='center', fontsize=7, alpha=0.7)
    ax.text(i + w, r3 * 100 + 2, f'{r3*100:.0f}', ha='center', fontsize=8, fontweight='bold')

ax.axhline(y=0, color='black', linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels([yr.year for yr in common_years])
ax.set_title('年度收益对比 — F1 vs X10 vs X11-A', fontsize=14, fontweight='bold')
ax.set_xlabel('年份')
ax.set_ylabel('年收益 (%)')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(str(OUTPUT_DIR / 'x11a_yearly_compare.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"  已保存: x11a_yearly_compare.png")

# --- 图表4: 月度热力图 ---
print("绘制图表4: 月度热力图...")
monthly_matrix = daily_ret.resample('M').apply(lambda x: (1 + x).prod() - 1)
monthly_matrix.index = pd.MultiIndex.from_arrays([monthly_matrix.index.year, monthly_matrix.index.month],
                                                   names=['year', 'month'])
monthly_pivot = monthly_matrix.unstack(level='month')
monthly_pivot.columns = [f'{m}月' for m in monthly_pivot.columns]

fig, ax = plt.subplots(figsize=(16, 8))
im = ax.imshow(monthly_pivot.values * 100, cmap='RdYlGn', aspect='auto', vmin=-15, vmax=15)

# 添加数值标注
for i in range(monthly_pivot.shape[0]):
    for j in range(monthly_pivot.shape[1]):
        val = monthly_pivot.iloc[i, j]
        if not np.isnan(val):
            color = 'white' if abs(val * 100) > 10 else 'black'
            ax.text(j, i, f'{val*100:.1f}', ha='center', va='center', fontsize=8, color=color)

ax.set_xticks(range(len(monthly_pivot.columns)))
ax.set_xticklabels(monthly_pivot.columns)
ax.set_yticks(range(len(monthly_pivot.index)))
ax.set_yticklabels(monthly_pivot.index)
ax.set_title(f'X11-A({best_n}天确认) 月度收益热力图 (%)', fontsize=14, fontweight='bold')
ax.set_xlabel('月份')
ax.set_ylabel('年份')
plt.colorbar(im, ax=ax, label='月收益 (%)')

fig.tight_layout()
fig.savefig(str(OUTPUT_DIR / 'x11a_monthly_heatmap.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"  已保存: x11a_monthly_heatmap.png")

# ============================================================
# 9. 保存完整报告
# ============================================================
print("\n保存完整报告...")

report_lines = []
report_lines.append("=" * 70)
report_lines.append(f"  X11-A({best_n}天确认) 完整验证报告")
report_lines.append("=" * 70)
report_lines.append("")
report_lines.append("【核心指标】")
report_lines.append(f"  年化={m_x11a['ann']*100:.2f}% 回撤={m_x11a['dd']*100:.2f}% "
                    f"Sharpe={m_x11a['sharpe']:.3f} Calmar={m_x11a['calmar']:.3f} 交易={m_x11a['n_trades']}")
report_lines.append("")
report_lines.append("【参数稳健性（确认天数）】")
report_lines.append(f"  {'天数':>4} {'年化':>8} {'回撤':>8} {'Sharpe':>8} {'Calmar':>8} {'交易':>6} {'达标':>6}")
for n, m in robustness_results.items():
    pf = "YES" if m['ann'] * 100 >= 36.0 else "NO"
    report_lines.append(f"  {n:>4d} {m['ann']*100:>7.2f}% {m['dd']*100:>7.2f}% "
                        f"{m['sharpe']:>8.3f} {m['calmar']:>8.3f} {m['n_trades']:>6d} {pf:>6}")
report_lines.append("")
report_lines.append("【最大回撤期】")
report_lines.append(f"  峰值: {peak_date:%Y-%m-%d}, 净值¥{peak.loc[peak_date]:,.0f}")
report_lines.append(f"  谷底: {max_dd_date:%Y-%m-%d}, 净值¥{nav.loc[max_dd_date]:,.0f}")
report_lines.append(f"  最大回撤: {max_dd*100:.2f}%")
if recovery_date:
    report_lines.append(f"  恢复: {recovery_date:%Y-%m-%d}, 持续{(recovery_date - peak_date).days}天")
report_lines.append("")
report_lines.append("【年度收益】")
for yr, r in yearly_ret.items():
    flag = "盈" if r > 0 else "亏"
    report_lines.append(f"  {yr.year}: {r*100:+.2f}% [{flag}]")
report_lines.append(f"  年胜率: {win_years}/{len(yearly_ret)} = {win_years/len(yearly_ret)*100:.1f}%")
report_lines.append(f"  月胜率: {(monthly_ret > 0).sum()}/{len(monthly_ret)} = {(monthly_ret > 0).mean()*100:.1f}%")

report_path = Path(r'c:\temp_v72_data\x11a_full_validation_report.txt')
report_path.write_text("\n".join(report_lines), encoding='utf-8')
print(f"报告已保存: {report_path}")

print("\n" + "=" * 70)
print(f"  X11-A({best_n}天确认) 完整验证完成！")
print("=" * 70)
print(f"\n最终结果: 年化{m_x11a['ann']*100:.2f}% / 回撤{m_x11a['dd']*100:.2f}% / "
      f"Sharpe{m_x11a['sharpe']:.3f} / Calmar{m_x11a['calmar']:.3f}")
