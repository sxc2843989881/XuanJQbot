"""backtest_x12.py — X12策略完整回测
================================================================
X12 = X11-A + F0偏离度空仓过滤

X12策略规则（6层）:
  第1层 F1基础信号: 比价>MA20 → growth, 否则value
  第2层 F0空仓过滤(X12新增): 偏离度<阈值时空仓(weight=0)
  第3层 A1确认: 4天方向一致
  第4层 斜率确认: MA20斜率绝对值>0.3%
  第5层 B2过滤: value方向且价值20日动量<=0 → 强制改growth
  第6层 E5止损: 持仓方向20日跌幅>10%降仓30%(在F0权重基础上叠加)

对比基准: X11-A(39.96%/回撤-34.71%/Sharpe1.307/Calmar1.151)
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
from backtest_engine import (
    BacktestInput, BacktestConfig,
    run_backtest_engine_weighted,
)

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150

DATA_DIR = Path(r'c:\temp_v72_data')
OUTPUT_DIR = Path(r'c:\XuanJLH\Qbot\custom\research\成长价值轮动\X12_偏离度空仓版\回测结果')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 1. 加载数据
# ============================================================
print("=" * 78)
print("  X12策略回测 = X11-A + F0偏离度空仓过滤")
print("=" * 78)

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
print(f"  数据: {g_close.index[0].date()} ~ {g_close.index[-1].date()}, {len(g_close)}天")

# 公共因子
ratio = g_close / v_close
ratio_ma20 = ratio.rolling(20).mean()
ratio_dev = ratio / ratio_ma20 - 1
ma20_slope = (ratio_ma20 - ratio_ma20.shift(5)) / ratio_ma20.shift(5)
slope_ok = ma20_slope.abs() > 0.003
v_mom20 = v_close.pct_change(20)
g_dd20 = g_close / g_close.shift(20) - 1
v_dd20 = v_close / v_close.shift(20) - 1
base_dir = (ratio > ratio_ma20).map({True: 'growth', False: 'value'})

# ============================================================
# 2. X12策略信号生成
# ============================================================
def build_x12_signal(dev_threshold=0.005, n_confirm=4, stop_threshold=0.10, stop_weight=0.30):
    """构建X12信号

    Args:
        dev_threshold: F0偏离度空仓阈值(0.005=0.5%)
        n_confirm: A1确认天数
        stop_threshold: E5止损阈值
        stop_weight: E5止损后仓位
    Returns:
        dir_s: 方向序列(growth/value)
        wt: 仓位序列(0.0-1.0)
    """
    # 第1层 F1基础信号: 比价>MA20方向
    dir_s = base_dir.copy()

    # 第2层 F0空仓过滤(X12新增): 偏离度<阈值空仓
    wt = pd.Series(1.0, index=dir_s.index)
    low_dev = ratio_dev.abs() < dev_threshold
    wt[low_dev] = 0.0

    # 第3层 A1确认: n天方向一致
    mask = np.ones(len(dir_s), dtype=bool)
    for k in range(1, n_confirm):
        mask = mask & (dir_s.values == dir_s.shift(k).values)
    confirmed = dir_s.where(mask, np.nan)

    # 第4层 斜率确认: MA20斜率绝对值>0.3%
    dir_s = confirmed.where(slope_ok, np.nan).ffill()

    # 第5层 B2过滤: value方向且价值20日动量<=0 → 强制改growth
    wrong_value = (dir_s == 'value') & (v_mom20 <= 0)
    dir_s[wrong_value] = 'growth'

    # 第6层 E5止损: 持仓方向20日跌幅>阈值降仓(在F0权重基础上叠加)
    gs = (dir_s == 'growth') & (g_dd20 < -stop_threshold)
    vs = (dir_s == 'value') & (v_dd20 < -stop_threshold)
    wt[gs | vs] = wt[gs | vs] * stop_weight

    return dir_s, wt

def build_x11a_signal(n_confirm=4, stop_threshold=0.10, stop_weight=0.30):
    """构建X11-A信号(用于对比)"""
    dir_s = base_dir.copy()
    mask = np.ones(len(dir_s), dtype=bool)
    for k in range(1, n_confirm):
        mask = mask & (dir_s.values == dir_s.shift(k).values)
    confirmed = dir_s.where(mask, np.nan)
    dir_s = confirmed.where(slope_ok, np.nan).ffill()
    wrong_value = (dir_s == 'value') & (v_mom20 <= 0)
    dir_s[wrong_value] = 'growth'
    wt = pd.Series(1.0, index=dir_s.index)
    gs = (dir_s == 'growth') & (g_dd20 < -stop_threshold)
    vs = (dir_s == 'value') & (v_dd20 < -stop_threshold)
    wt[gs | vs] = stop_weight
    return dir_s, wt

# ============================================================
# 3. 回测函数
# ============================================================
def run_backtest(signal, weight, impact_slippage=0.0):
    common_idx = signal.index.intersection(g_close.index)
    sig = signal.loc[common_idx]
    wt = weight.loc[common_idx]
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
                            impact_slippage=impact_slippage, apply_gap_slippage=False)
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
            'total': total, 'n_trades': result.metrics['num_trades'],
            'daily_ret': r, 'eq': eq}

# ============================================================
# 4. X11-A基准(对比)
# ============================================================
print("\n  [1/5] X11-A基准回测...")
sig_x11a, wt_x11a = build_x11a_signal()
r_x11a = run_backtest(sig_x11a, wt_x11a)
m_x11a = calc_metrics(r_x11a)
print(f"  X11-A: 年化{m_x11a['ann']*100:.2f}% 回撤{m_x11a['dd']*100:.2f}% "
      f"Sharpe{m_x11a['sharpe']:.3f} Calmar{m_x11a['calmar']:.3f} 交易{m_x11a['n_trades']}次")

# ============================================================
# 5. X12参数敏感性(偏离度阈值)
# ============================================================
print("\n" + "=" * 78)
print("  [2/5] X12参数敏感性: 偏离度空仓阈值")
print("=" * 78)
print(f"  {'阈值':>8} {'年化':>8} {'回撤':>8} {'Sharpe':>8} {'Calmar':>8} {'交易':>6} {'空仓天数':>8}")
print(f"  {'-'*60}")

param_results = {}
for thresh in [0.000, 0.003, 0.005, 0.008, 0.010, 0.015, 0.020]:
    sig, wt = build_x12_signal(dev_threshold=thresh)
    r = run_backtest(sig, wt)
    m = calc_metrics(r)
    n_cash = (wt == 0.0).sum()
    param_results[thresh] = m
    label = f"{thresh*100:.1f}%"
    if thresh == 0.000:
        label += "(=X11-A)"
    elif thresh == 0.005:
        label += "(推荐)"
    print(f"  {label:>8} {m['ann']*100:>7.2f}% {m['dd']*100:>7.2f}% "
          f"{m['sharpe']:>8.3f} {m['calmar']:>8.3f} {m['n_trades']:>6} {n_cash:>8}")

# ============================================================
# 6. X12最优参数完整回测
# ============================================================
print("\n" + "=" * 78)
print("  [3/5] X12完整回测(偏离度阈值0.5%)")
print("=" * 78)

sig_x12, wt_x12 = build_x12_signal(dev_threshold=0.005)
r_x12 = run_backtest(sig_x12, wt_x12)
m_x12 = calc_metrics(r_x12)

print(f"\n  X11-A vs X12 对比:")
print(f"  {'指标':<15} {'X11-A':>12} {'X12':>12} {'差异':>12}")
print(f"  {'-'*54}")
print(f"  {'年化收益':<15} {m_x11a['ann']*100:>11.2f}% {m_x12['ann']*100:>11.2f}% {(m_x12['ann']-m_x11a['ann'])*100:>+11.2f}pp")
print(f"  {'最大回撤':<15} {m_x11a['dd']*100:>11.2f}% {m_x12['dd']*100:>11.2f}% {(m_x12['dd']-m_x11a['dd'])*100:>+11.2f}pp")
print(f"  {'Sharpe':<15} {m_x11a['sharpe']:>12.3f} {m_x12['sharpe']:>12.3f} {m_x12['sharpe']-m_x11a['sharpe']:>+12.3f}")
print(f"  {'Calmar':<15} {m_x11a['calmar']:>12.3f} {m_x12['calmar']:>12.3f} {m_x12['calmar']-m_x11a['calmar']:>+12.3f}")
print(f"  {'交易次数':<15} {m_x11a['n_trades']:>12} {m_x12['n_trades']:>12} {m_x12['n_trades']-m_x11a['n_trades']:>+12}")

# 空仓统计
n_cash = (wt_x12 == 0.0).sum()
n_e5 = (wt_x12 == 0.3).sum()
n_full = (wt_x12 == 1.0).sum()
print(f"\n  X12仓位分布:")
print(f"    满仓(100%): {n_full}天 ({n_full/len(wt_x12)*100:.1f}%)")
print(f"    降仓(30%,E5): {n_e5}天 ({n_e5/len(wt_x12)*100:.1f}%)")
print(f"    空仓(0%,F0): {n_cash}天 ({n_cash/len(wt_x12)*100:.1f}%)")

# ============================================================
# 7. X12 + 5/万滑点测试
# ============================================================
print("\n" + "=" * 78)
print("  [4/5] X12滑点敏感性(5/万滑点)")
print("=" * 78)

r_x12_sl5 = run_backtest(sig_x12, wt_x12, impact_slippage=0.0005)
m_x12_sl5 = calc_metrics(r_x12_sl5)
print(f"  X12无滑点:  年化{m_x12['ann']*100:.2f}% 回撤{m_x12['dd']*100:.2f}% "
      f"Sharpe{m_x12['sharpe']:.3f} Calmar{m_x12['calmar']:.3f}")
print(f"  X12 5/万滑点: 年化{m_x12_sl5['ann']*100:.2f}% 回撤{m_x12_sl5['dd']*100:.2f}% "
      f"Sharpe{m_x12_sl5['sharpe']:.3f} Calmar{m_x12_sl5['calmar']:.3f}")

# X11-A 5/万滑点对比
r_x11a_sl5 = run_backtest(sig_x11a, wt_x11a, impact_slippage=0.0005)
m_x11a_sl5 = calc_metrics(r_x11a_sl5)
print(f"  X11-A 5/万滑点(对比): 年化{m_x11a_sl5['ann']*100:.2f}% 回撤{m_x11a_sl5['dd']*100:.2f}% "
      f"Sharpe{m_x11a_sl5['sharpe']:.3f} Calmar{m_x11a_sl5['calmar']:.3f}")

# ============================================================
# 8. 年度收益对比
# ============================================================
print("\n" + "=" * 78)
print("  [5/5] 年度收益对比")
print("=" * 78)
print(f"  {'年份':>6} {'X11-A':>10} {'X12':>10} {'差异':>10}")
print(f"  {'-'*38}")

df_x11a_full = r_x11a.to_dataframe()
df_x11a_full['date'] = pd.to_datetime(df_x11a_full['date'])
df_x11a_full = df_x11a_full.set_index('date')
ret_x11a = df_x11a_full['daily_ret']

df_x12_full = r_x12.to_dataframe()
df_x12_full['date'] = pd.to_datetime(df_x12_full['date'])
df_x12_full = df_x12_full.set_index('date')
ret_x12 = df_x12_full['daily_ret']

for year in sorted(ret_x11a.index.year.unique()):
    r11 = ret_x11a[ret_x11a.index.year == year]
    r12 = ret_x12[ret_x12.index.year == year]
    ann11 = (1 + r11).prod() - 1
    ann12 = (1 + r12).prod() - 1
    diff = ann12 - ann11
    flag = "✅" if diff > 0 else "❌"
    print(f"  {year:>6} {ann11*100:>9.2f}% {ann12*100:>9.2f}% {diff*100:>+9.2f}pp {flag}")

# ============================================================
# 9. 绘图
# ============================================================
print("\n  绘制图表...")

fig, axes = plt.subplots(3, 1, figsize=(14, 12))

# 资金曲线对比
ax1 = axes[0]
ax1.plot(m_x11a['eq'].values, label=f'X11-A (年化{m_x11a["ann"]*100:.2f}%)', color='#3498DB', linewidth=1.5)
ax1.plot(m_x12['eq'].values, label=f'X12 (年化{m_x12["ann"]*100:.2f}%)', color='#E74C3C', linewidth=1.5)
ax1.set_title('X11-A vs X12 资金曲线对比', fontsize=13, fontweight='bold')
ax1.set_ylabel('累计净值')
ax1.legend(loc='best')
ax1.grid(True, alpha=0.3)
ax1.axhline(y=1, color='gray', linestyle='--', alpha=0.5)

# 回撤对比
ax2 = axes[1]
dd_x11a = (m_x11a['eq'] / m_x11a['eq'].cummax() - 1)
dd_x12 = (m_x12['eq'] / m_x12['eq'].cummax() - 1)
ax2.fill_between(range(len(dd_x11a)), dd_x11a.values, 0, alpha=0.4, color='#3498DB', label=f'X11-A (最深{m_x11a["dd"]*100:.1f}%)')
ax2.fill_between(range(len(dd_x12)), dd_x12.values, 0, alpha=0.4, color='#E74C3C', label=f'X12 (最深{m_x12["dd"]*100:.1f}%)')
ax2.set_title('回撤对比', fontsize=13, fontweight='bold')
ax2.set_ylabel('回撤')
ax2.legend(loc='best')
ax2.grid(True, alpha=0.3)

# 偏离度与空仓标记
ax3 = axes[2]
ax3.plot(ratio_dev.values * 100, color='#2ECC71', linewidth=0.8, alpha=0.7, label='偏离率(%)')
ax3.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='±0.5%阈值')
ax3.axhline(y=-0.5, color='red', linestyle='--', alpha=0.5)
ax3.fill_between(range(len(ratio_dev)), -0.5, ratio_dev.values*100, where=(ratio_dev.abs().values < 0.005),
                 alpha=0.3, color='gray', label='空仓区间')
ax3.set_title('比价偏离率与F0空仓区间', fontsize=13, fontweight='bold')
ax3.set_ylabel('偏离率(%)')
ax3.legend(loc='best')
ax3.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(str(OUTPUT_DIR / 'x12_vs_x11a.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"  图表已保存: {OUTPUT_DIR / 'x12_vs_x11a.png'}")

# ============================================================
# 10. 保存报告
# ============================================================
report_path = OUTPUT_DIR / 'x12_backtest_report.txt'
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("=" * 78 + "\n")
    f.write("  X12策略回测报告\n")
    f.write("=" * 78 + "\n\n")
    f.write("  策略: X12 = X11-A + F0偏离度空仓过滤\n")
    f.write("  数据: 2012-12-31 ~ 2026-07-01\n\n")

    f.write("  X11-A vs X12 核心指标对比:\n")
    f.write(f"    年化收益: X11-A {m_x11a['ann']*100:.2f}% → X12 {m_x12['ann']*100:.2f}% ({(m_x12['ann']-m_x11a['ann'])*100:+.2f}pp)\n")
    f.write(f"    最大回撤: X11-A {m_x11a['dd']*100:.2f}% → X12 {m_x12['dd']*100:.2f}% ({(m_x12['dd']-m_x11a['dd'])*100:+.2f}pp)\n")
    f.write(f"    Sharpe:   X11-A {m_x11a['sharpe']:.3f} → X12 {m_x12['sharpe']:.3f} ({m_x12['sharpe']-m_x11a['sharpe']:+.3f})\n")
    f.write(f"    Calmar:   X11-A {m_x11a['calmar']:.3f} → X12 {m_x12['calmar']:.3f} ({m_x12['calmar']-m_x11a['calmar']:+.3f})\n\n")

    f.write("  参数敏感性(偏离度阈值):\n")
    for thresh, m in param_results.items():
        f.write(f"    {thresh*100:.1f}%: 年化{m['ann']*100:.2f}% 回撤{m['dd']*100:.2f}% Sharpe{m['sharpe']:.3f} Calmar{m['calmar']:.3f}\n")

    f.write(f"\n  5/万滑点测试:\n")
    f.write(f"    X12无滑点:   年化{m_x12['ann']*100:.2f}% Calmar{m_x12['calmar']:.3f}\n")
    f.write(f"    X12 5/万滑点: 年化{m_x12_sl5['ann']*100:.2f}% Calmar{m_x12_sl5['calmar']:.3f}\n")
    f.write(f"    X11-A 5/万滑点: 年化{m_x11a_sl5['ann']*100:.2f}% Calmar{m_x11a_sl5['calmar']:.3f}\n")

print(f"  报告已保存: {report_path}")

print("\n" + "=" * 78)
print("  X12回测完成")
print("=" * 78)
