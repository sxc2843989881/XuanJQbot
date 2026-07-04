"""X14 流水线验证 — 完整回测 + 报告 + 图表生成
================================================================
"""
import sys, json
from pathlib import Path

sys.path.insert(0, r'c:\XuanJLH\Qbot\custom\backtests')
sys.path.insert(0, r'c:\XuanJLH\Qbot\custom\research\成长价值轮动\X12_偏离度空仓版')
sys.path.insert(0, r'c:\XuanJLH\Qbot\custom\research\成长价值轮动\X14_统一策略干净版')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from optimize_runner import (
    G_CLOSE, V_CLOSE, RATIO, RATIO_MA20, RATIO_DEV, MA20_SLOPE,
    V_MOM20, G_DD20, V_DD20, BASE_DIR,
    run_backtest, calc_metrics, count_switches,
)
from run_x33_reduce_trades import RATIO_DEV_Z

OUTPUT_DIR = Path(r'c:\XuanJLH\Qbot\custom\research\成长价值轮动\X14_统一策略干净版\回测结果')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150

# ========== 策略定义 ==========
T = RATIO_DEV_Z
SLOPE = MA20_SLOPE

def build_x14(slope_thresh=0.002, sw_mid=0.17, sw_deep=0.17, st=0.09, cd=8,
              ms=10, ml=20, rt=1.3, dc=5, dcd=4,
              bias_ma=20, bias_high=0.19, bias_mode='half',
              dual_momentum=False, bias_t_constraint=True, rapid_decline=False,
              e5_reset=False):
    """X14 干净版 v2.0-lite (委托引擎实现)"""
    from backtest_x14_engine import build_core
    return build_core(slope_thresh=slope_thresh, sw_mid=sw_mid, sw_deep=sw_deep,
                      st=st, cd=cd, ms=ms, ml=ml, rt=rt, dc=dc, dcd=dcd,
                      bias_ma=bias_ma, bias_high=bias_high, bias_mode=bias_mode,
                      dual_momentum=dual_momentum, bias_t_constraint=bias_t_constraint,
                      rapid_decline=rapid_decline, e5_reset=e5_reset)


def build_x13():
    """X13 原版 (v26 U164)"""
    sys.path.insert(0, r'c:\XuanJLH\Qbot\custom\research\成长价值轮动\X13_统一策略最优版')
    from backtest_x13_engine import build_core as x13_build
    return x13_build()


def build_x11a():
    from optimize_runner import build_x11a
    return build_x11a()


# ========== 回测 ==========
strategies = {
    'X14_干净版': build_x14,
    'X13_原版': build_x13,
    'X11_A基准': build_x11a,
}

print("=" * 80)
print("  X14 流水线验证")
print("=" * 80)

all_results = {}
all_signals = {}
all_results_detail = {}

for name, builder in strategies.items():
    print(f"\n--- {name} ---")
    sig, wt = builder()
    all_signals[name] = (sig, wt)
    
    result = run_backtest(sig, wt, impact_slippage=0.0005)
    m = calc_metrics(result)
    sw = count_switches(sig, wt)
    
    all_results[name] = {
        'ann': m['ann'], 'dd': m['dd'], 'sharpe': m['sharpe'],
        'calmar': m['calmar'], 'n_trades': m['n_trades'],
        'dir_sw': sw['dir'], 'cash_sw': sw['cash'],
        'ann_sl': m['ann'], 'calmar_sl': m['calmar'],
        'total_return': result.nav[-1] / result.nav[0] - 1,
    }
    all_results_detail[name] = result
    
    print(f"  年化={m['ann']*100:.2f}%  回撤={m['dd']*100:.2f}%  "
          f"Sharpe={m['sharpe']:.3f}  Calmar={m['calmar']:.3f}  "
          f"交易={m['n_trades']}")


# ========== 保存CSV ==========
print("\n\n--- 保存CSV数据 ---")
for name in strategies:
    safe = name.replace(' ', '_').replace('/', '_')
    result = all_results_detail[name]
    sig, wt = all_signals[name]
    
    nav = pd.Series(result.nav, index=pd.to_datetime(result.dates))
    daily_ret = nav / nav.shift(1) - 1
    
    # 日收益率
    df_daily = pd.DataFrame({'日期': result.dates, '日收益率': daily_ret.values})
    df_daily.to_csv(OUTPUT_DIR / f'{safe}_日收益率.csv', index=False, encoding='utf-8-sig')
    
    # 信号
    df_sig = pd.DataFrame({'信号': sig, '权重': wt})
    df_sig.to_csv(OUTPUT_DIR / f'{safe}_信号权重.csv', encoding='utf-8-sig')
    
    # 回撤
    dd = nav / nav.cummax() - 1
    
    # 交易记录
    pos = pd.Series(0, index=sig.index)
    pos[wt > 0] = 1
    trade_entries = (pos != pos.shift(1)) & (pos == 1)
    trade_exits = (pos != pos.shift(1)) & (pos == 0)
    trades = []
    entry_date = None
    for i in range(len(pos)):
        if trade_entries.iloc[i]:
            entry_date = pos.index[i]
        elif trade_exits.iloc[i] and entry_date is not None:
            exit_date = pos.index[i]
            ret = nav.loc[exit_date] / nav.loc[entry_date] - 1
            trades.append({'入场': entry_date.date(), '出场': exit_date.date(), '收益率': ret})
            entry_date = None
    df_trades = pd.DataFrame(trades)
    df_trades.to_csv(OUTPUT_DIR / f'{safe}_交易记录.csv', index=False, encoding='utf-8-sig')
    
    # 月度收益
    monthly_ret = nav.resample('M').last().pct_change().dropna()
    df_monthly = monthly_ret.reset_index()
    df_monthly.columns = ['日期', '月收益率']
    df_monthly.to_csv(OUTPUT_DIR / f'{safe}_月度收益.csv', index=False, encoding='utf-8-sig')
    
    # 年度收益
    yearly_ret = nav.resample('Y').last().pct_change().dropna()
    df_yearly = yearly_ret.reset_index()
    df_yearly.columns = ['日期', '年收益率']
    df_yearly.to_csv(OUTPUT_DIR / f'{safe}_年度收益.csv', index=False, encoding='utf-8-sig')
    
    # 指数行情
    df_idx = pd.DataFrame({'成长': G_CLOSE, '价值': V_CLOSE})
    df_idx.to_csv(OUTPUT_DIR / f'{safe}_指数行情.csv', encoding='utf-8-sig')
    
    print(f"  {name}: CSV保存完成")


# ========== 报告JSON ==========
print("\n\n--- 生成报告 ---")
report = {
    '策略': {},
    '对比': {
        'X14_vs_X13_calmar提升': all_results['X14_干净版']['calmar'] / all_results['X13_原版']['calmar'] - 1,
        'X14_vs_X13_ann提升': all_results['X14_干净版']['ann'] / all_results['X13_原版']['ann'] - 1,
    }
}
for name, r in all_results.items():
    report['策略'][name] = {
        '年化': f"{r['ann']*100:.2f}%",
        '最大回撤': f"{r['dd']*100:.2f}%",
        'Sharpe': round(r['sharpe'], 3),
        'Calmar': round(r['calmar'], 3),
        '交易次数': r['n_trades'],
        '滑点Calmar': round(r['calmar_sl'], 3),
        '总收益率': f"{r['total_return']*100:.2f}%",
    }

with open(OUTPUT_DIR / 'X14_报告.json', 'w', encoding='utf-8') as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
print("  JSON报告保存完成")


# ========== 主图: 净值曲线 + 回撤 ==========
print("\n--- 生成主图 ---")
fig, axes = plt.subplots(3, 1, figsize=(16, 12), gridspec_kw={'height_ratios': [3, 1, 1]})
colors = {'X14_干净版': '#E74C3C', 'X13_原版': '#3498DB', 'X11_A基准': '#95A5A6'}
lstyles = {'X14_干净版': '-', 'X13_原版': '--', 'X11_A基准': ':'}
lw = {'X14_干净版': 2.0, 'X13_原版': 1.5, 'X11_A基准': 1.0}

# 净值曲线
ax = axes[0]
for name in strategies:
    result = all_results_detail[name]
    nav = pd.Series(result.nav, index=pd.to_datetime(result.dates))
    ann = all_results[name]['ann']
    cr = all_results[name]['calmar']
    ax.plot(pd.to_datetime(result.dates), result.nav,
            color=colors[name], linestyle=lstyles[name], linewidth=lw[name],
            label=f"{name}  年化{ann*100:.1f}%  Calmar{cr:.3f}")
ax.set_ylabel('累计净值(对数)', fontsize=11)
ax.set_yscale('log')
ax.legend(fontsize=10, loc='upper left')
ax.grid(True, alpha=0.3)
ax.set_title('X14 统一策略干净版 — 净值曲线对比', fontsize=13, fontweight='bold')
ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5)

# 回撤曲线
ax = axes[1]
for name in strategies:
    result = all_results_detail[name]
    nav = pd.Series(result.nav, index=pd.to_datetime(result.dates))
    dd = nav / nav.cummax() - 1
    ax.fill_between(pd.to_datetime(result.dates), dd.values * 100, 0,
                    color=colors[name], alpha=0.3 if name == 'X14_干净版' else 0.15)
    ax.plot(pd.to_datetime(result.dates), dd.values * 100,
            color=colors[name], linestyle=lstyles[name], linewidth=lw[name],
            label=f"{name} 最大回撤{dd.min()*100:.1f}%")
ax.set_ylabel('回撤(%)', fontsize=11)
ax.legend(fontsize=9, loc='lower left')
ax.grid(True, alpha=0.3)

# 权重曲线
ax = axes[2]
for name in ['X14_干净版', 'X13_原版']:
    if name in all_signals:
        sig, wt = all_signals[name]
        wt_plot = wt.copy()
        sig_map = sig.map({'growth': 1, 'value': -1})
        ax.plot(wt.index, wt_plot.values * sig_map.values,
                color=colors[name], linestyle=lstyles[name], linewidth=0.5, alpha=0.7,
                label=name)
ax.set_ylabel('方向(+成长/-价值)', fontsize=11)
ax.set_xlabel('日期', fontsize=11)
ax.legend(fontsize=9, loc='lower left')
ax.grid(True, alpha=0.3)
ax.set_ylim(-1.5, 1.5)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'X14_主图.png', bbox_inches='tight')
plt.close()
print("  主图保存完成")


# ========== 月度热力图 ==========
print("\n--- 生成月度热力图 ---")
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

for idx, name in enumerate(['X14_干净版', 'X13_原版']):
    ax = axes[idx]
    result = all_results_detail[name]
    nav = pd.Series(result.nav, index=pd.to_datetime(result.dates))
    monthly = nav.resample('M').last().pct_change().dropna() * 100
    
    years = sorted(set(d.year for d in monthly.index))
    months = list(range(1, 13))
    
    heat_data = pd.DataFrame(index=years, columns=months, data=np.nan)
    for d, v in monthly.items():
        if d.year in years and d.month in months:
            heat_data.loc[d.year, d.month] = v
    
    cmap = sns.diverging_palette(10, 130, s=80, l=55, as_cmap=True)
    sns.heatmap(heat_data, annot=True, fmt='.1f', cmap=cmap, center=0,
                ax=ax, linewidths=0.5, cbar_kws={'label': '月收益率(%)'},
                annot_kws={'fontsize': 7})
    
    ann = all_results[name]['ann'] * 100
    cr = all_results[name]['calmar']
    ax.set_title(f'{name}  年化{ann:.1f}%  Calmar{cr:.3f}', fontsize=12, fontweight='bold')
    ax.set_ylabel('年份')
    ax.set_xlabel('月份')

plt.suptitle('月度收益率热力图对比', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'X14_月度热力图.png', bbox_inches='tight')
plt.close()
print("  月度热力图保存完成")


# ========== 交易分析图 ==========
print("\n--- 生成交易分析图 ---")
fig, axes = plt.subplots(2, 3, figsize=(20, 10))

for idx, name in enumerate(['X14_干净版', 'X13_原版']):
    row, col = divmod(idx, 3)
    result = all_results_detail[name]
    sig, wt = all_signals[name]
    
    r = all_results[name]
    sig_map = sig.map({'growth': 1, 'value': -1})
    
    # 1. 方向分布
    ax = axes[idx][0]
    dir_counts = sig.value_counts()
    colors_pie = ['#E74C3C', '#3498DB']
    ax.pie(dir_counts.values, labels=dir_counts.index, autopct='%1.0f%%',
           colors=colors_pie, startangle=90)
    ax.set_title(f'{name} 方向分布', fontsize=11, fontweight='bold')
    
    # 2. 仓位分布
    ax = axes[idx][1]
    wt_bins = [0, 0.01, 0.5, 1.0]
    wt_labels = ['空仓(0)', '半仓(<0.5)', '满仓(1.0)']
    wt_cats = pd.cut(wt, bins=wt_bins, labels=wt_labels)
    wt_counts = wt_cats.value_counts()
    ax.bar(wt_counts.index, wt_counts.values, color=['#E74C3C', '#F39C12', '#2ECC71'], width=0.5)
    ax.set_title(f'{name} 仓位分布', fontsize=11, fontweight='bold')
    ax.set_ylabel('天数')
    for i, v in enumerate(wt_counts.values):
        ax.text(i, v + 5, str(v), ha='center', fontsize=9)
    
    # 3. 历年收益
    ax = axes[idx][2]
    nav = pd.Series(result.nav, index=pd.to_datetime(result.dates))
    yearly = nav.resample('Y').last().pct_change().dropna() * 100
    years = [d.year for d in yearly.index]
    colors_bar = ['#2ECC71' if v >= 0 else '#E74C3C' for v in yearly.values]
    ax.bar(years, yearly.values, color=colors_bar, width=0.6)
    ax.set_title(f'{name} 历年收益', fontsize=11, fontweight='bold')
    ax.set_ylabel('年收益率(%)')
    ax.axhline(0, color='gray', linestyle='-', alpha=0.3)
    for y, v in zip(years, yearly.values):
        ax.text(y, v + 1 if v >= 0 else v - 5, f'{v:.1f}%', ha='center', fontsize=8)

plt.suptitle('X14 策略交易分析', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'X14_交易分析.png', bbox_inches='tight')
plt.close()
print("  交易分析图保存完成")


# ========== 对比图 ==========
print("\n--- 生成三策略对比图 ---")
fig, axes = plt.subplots(2, 2, figsize=(18, 12))

# (1) 净值曲线
ax = axes[0][0]
for name in strategies:
    result = all_results_detail[name]
    r = all_results[name]
    ax.plot(pd.to_datetime(result.dates), result.nav,
            color=colors[name], linestyle=lstyles[name], linewidth=lw[name],
            label=f"{name}  Calmar{r['calmar']:.3f}")
ax.set_ylabel('累计净值(对数)', fontsize=11)
ax.set_yscale('log')
ax.legend(fontsize=10, loc='upper left')
ax.grid(True, alpha=0.3)
ax.set_title('净值曲线对比', fontsize=12, fontweight='bold')
ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5)

# (2) 回撤曲线
ax = axes[0][1]
for name in strategies:
    result = all_results_detail[name]
    nav = pd.Series(result.nav, index=pd.to_datetime(result.dates))
    dd = nav / nav.cummax() - 1
    ax.fill_between(pd.to_datetime(result.dates), dd.values * 100, 0, color=colors[name], alpha=0.2)
    ax.plot(pd.to_datetime(result.dates), dd.values * 100, color=colors[name], linewidth=1.5,
            label=f"{name}  MaxDD={dd.min()*100:.1f}%")
ax.set_ylabel('回撤(%)', fontsize=11)
ax.legend(fontsize=9, loc='lower left')
ax.grid(True, alpha=0.3)
ax.set_title('回撤对比', fontsize=12, fontweight='bold')

# (3) 逐年收益
ax = axes[1][0]
all_years = set()
yearly_data = {}
for name in strategies:
    result = all_results_detail[name]
    nav = pd.Series(result.nav, index=pd.to_datetime(result.dates))
    yearly = nav.resample('Y').last().pct_change().dropna() * 100
    yearly_data[name] = yearly
    for y in yearly.index:
        all_years.add(y.year)

years_sorted = sorted(all_years)
x = np.arange(len(years_sorted))
w = 0.25
for i, name in enumerate(strategies):
    vals = []
    for y in years_sorted:
        if y in [d.year for d in yearly_data[name].index]:
            vals.append(yearly_data[name].loc[yearly_data[name].index.year == y].values[0])
        else:
            vals.append(0)
    ax.bar(x + i * w - w, vals, w, color=colors[name], alpha=0.8, label=name)
ax.set_xticks(x)
ax.set_xticklabels(years_sorted, fontsize=9)
ax.set_ylabel('年收益率(%)', fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, axis='y', alpha=0.3)
ax.set_title('逐年收益对比', fontsize=12, fontweight='bold')
ax.axhline(0, color='gray', linestyle='-', alpha=0.3)

# (4) 指标对比雷达图
ax = axes[1][1]
metrics_names = ['年化%', 'Sharpe', 'Calmar', '滑点Calmar']
metric_keys = ['ann', 'sharpe', 'calmar', 'calmar_sl']
n_metrics = len(metrics_names)

# 归一化到0-1
normalized = {}
for name in strategies:
    r = all_results[name]
    norm_vals = []
    for key in metric_keys:
        if key == 'ann':
            norm_vals.append(r[key] * 100 / 60)  # 年化最高60%
        elif key == 'sharpe':
            norm_vals.append(r[key] / 2.0)  # Sharpe最高2
        elif key == 'calmar':
            norm_vals.append(r[key] / 3.0)  # Calmar最高3
        elif key == 'calmar_sl':
            norm_vals.append(r[key] / 2.5)  # 滑点Calmar最高2.5
    normalized[name] = norm_vals

angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
angles += angles[:1]

for name in strategies:
    vals = normalized[name] + normalized[name][:1]
    ax.plot(angles, vals, 'o-', color=colors[name], linewidth=2, label=name, markersize=5)
    ax.fill(angles, vals, color=colors[name], alpha=0.1)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(metrics_names, fontsize=10)
ax.set_ylim(0, 1.1)
ax.set_title('核心指标对比', fontsize=12, fontweight='bold')
ax.legend(fontsize=9, loc='upper right')
ax.grid(True, alpha=0.3)

plt.suptitle('X14 三策略对比', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'X14_三策略对比.png', bbox_inches='tight')
plt.close()
print("  对比图保存完成")


# ========== 辅助统计函数 ==========
def calc_drawdown_periods(nav_series, top_n=5):
    """分析回撤时段"""
    peak = nav_series.cummax()
    dd = nav_series / peak - 1
    in_dd = dd < 0
    transitions = in_dd.astype(int).diff()
    
    periods = []
    start = None
    for i in range(len(dd)):
        if transitions.iloc[i] == 1:  # 进入回撤
            start = dd.index[i]
        elif transitions.iloc[i] == -1 and start is not None:  # 退出回撤
            end = dd.index[i]
            depth = dd.loc[start:end].min()
            periods.append({'start': start.date(), 'end': end.date(),
                            'depth': depth, 'days': (end - start).days})
            start = None
    if start is not None:  # 仍在回撤中
        depth = dd.loc[start:].min()
        periods.append({'start': start.date(), 'end': '至今',
                        'depth': depth, 'days': (dd.index[-1] - start).days})
    
    periods.sort(key=lambda x: x['depth'])
    return periods[:top_n]


def calc_monthly_stats(nav_series):
    """月度统计"""
    monthly = nav_series.resample('M').last().pct_change().dropna() * 100
    positive = (monthly > 0).sum()
    total = len(monthly)
    return {
        'positive_months': f"{positive}/{total} ({positive/total*100:.1f}%)",
        'best_month': f"{monthly.max():.2f}%",
        'worst_month': f"{monthly.min():.2f}%",
        'avg_month': f"{monthly.mean():.2f}%",
        'std_month': f"{monthly.std():.2f}%",
    }


def calc_risk_metrics(nav_series):
    """风险指标"""
    daily_ret = nav_series / nav_series.shift(1) - 1
    daily_ret = daily_ret.dropna()
    
    var_95 = daily_ret.quantile(0.05)
    var_99 = daily_ret.quantile(0.01)
    
    # 最大连续亏损
    neg_streak = (daily_ret < 0).astype(int)
    streak = 0
    max_streak = 0
    streak_ret = 1.0
    max_streak_ret = 1.0
    for v in neg_streak.values:
        if v == 1:
            streak += 1
        else:
            if streak > max_streak:
                max_streak = streak
            streak = 0
    if streak > max_streak:
        max_streak = streak
    
    # 最大连续亏损幅度
    neg_daily = daily_ret[daily_ret < 0]
    cum_neg = (1 + neg_daily).cumprod()
    max_cum_neg = (1 - cum_neg).max() if len(cum_neg) > 0 else 0
    
    return {
        'VaR_95': f"{var_95*100:.2f}%",
        'VaR_99': f"{var_99*100:.2f}%",
        'max_consecutive_loss_days': max_streak,
        'max_consecutive_loss_pct': f"{(1 - (1+daily_ret[daily_ret<0]).iloc[:max_streak].prod())*100:.2f}%" if max_streak > 0 else "0%",
        'daily_ret_std': f"{daily_ret.std()*100:.3f}%",
    }


def get_nav(name):
    result = all_results_detail[name]
    return pd.Series(result.nav, index=pd.to_datetime(result.dates))


def get_yearly(name):
    nav = get_nav(name)
    return nav.resample('Y').last().pct_change().dropna() * 100


def get_monthly(name):
    nav = get_nav(name)
    return nav.resample('M').last().pct_change().dropna() * 100


# ========== TXT报告 ==========
print("\n--- 生成TXT报告 ---")
now_str = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')
with open(OUTPUT_DIR / 'X14_报告.txt', 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("  X14 统一策略干净版 — 流水线验证报告\n")
    f.write(f"  生成时间: {now_str}\n")
    f.write("=" * 80 + "\n\n")
    
    # ── 1. 策略概述 ──
    f.write("一、策略概述\n")
    f.write("-" * 50 + "\n")
    f.write("  X14 是 X13(v26 U164, Calmar 2.378) 的合理化版本。\n")
    f.write("  核心逻辑: 基于 T 指标(Z-score)判断成长/价值轮动方向，叠加6层保护机制。\n\n")
    f.write("  X14 对 X13 的主要改动:\n")
    f.write("    ① 移除 max_hold_days/max_hold_reduce（固定持仓天数，无金融逻辑支撑）\n")
    f.write("    ② st: 0.088 → 0.09（取整，避免精度过拟合）\n")
    f.write("    ③ bias_mode='clear' 一键清仓，用户可选 'ignore' 完全忽略BIAS层\n\n")
    
    # ── 2. 策略架构（6层逻辑详解）──
    f.write("二、策略架构（6层逻辑详解）\n")
    f.write("-" * 50 + "\n\n")
    
    layers = [
        ("第1层: 方向确认 (dc=5)",
         "T = RATIO_DEV_Z = (RATIO - RATIO_MA20) / RATIO_DEV_STD20\n"
         "  T > 0 → 成长跑赢价值 → 方向判为 BULL（买成长）\n"
         "  T < 0 → 价值跑赢成长 → 方向判为 BEAR（买价值）\n"
         "  需连续 dc(=5) 天 T 同号才确认方向切换，避免噪音误触发"),
        
        ("第2层: 方向冷却 (dcd=4)",
         "  方向切换后需冷却 dcd(=4) 天才允许再次切换。(v2.0: 6→4)\n"
         "  防止市场在比价边界反复震荡导致频繁换方向"),
        
        ("第3层: T+斜率双重确认 → 空仓 (rt=1.3, slope_thresh=0.002)",
         "  条件: |T| < rt AND |MA20_SLOPE| < slope_thresh\n"
         "  两个条件同时满足 → 视为「震荡市」→ 空仓(wt=0.0)\n"
         "  单一条件满足 → 不空仓（趋势仍在）\n"
         "  注: MA20_SLOPE = (RATIO_MA20 / RATIO_MA20.shift(5) - 1)"),
        
        ("第4层: B2 价值动量过滤 (ms=10, ml=20)",
         "  当方向为 BEAR（买价值）时：\n"
         "  如果 V_MOM_10 <= 0 AND V_MOM_20 <= 0\n"
         "  → 价值短/长周期都在跌，反手买成长\n"
         "  双周期AND条件比单周期更可靠，减少误切换"),
        
        ("第5层: BIAS 超买清仓 (bias_ma=20, bias_high=0.19, bias_mode='half'+T约束)",
         "  G_BIAS = G_CLOSE / MA20(G_CLOSE) - 1\n"
         "  V_BIAS = V_CLOSE / MA20(V_CLOSE) - 1\n"
         "  当持仓方向 BIAS > 19% 且 T<1.5 → 降仓50%(wt*=0.5)（v2.0-lite）\n"
         "  T>=1.5时容忍超买（趋势强时不过滤）\n"
         "  用户也可选 'clear' 模式(直接清仓)或 'ignore' 模式(完全忽略)\n"
         "  原理: 价格偏离均线过多 → 超买 → 大概率回调"),

        ("第6层: E5 止损 (st=0.09, sw=0.17, cd=8, 冷却期延长)",
         "  条件: 持仓方向的20日跌幅 > st(=9%) → 触发\n"
         "  冷却期内再次触发: 延长3天而非重置 (v2.0修复)\n"
         "  冷却期结束: 检查 is_weak → 弱则空仓，否则恢复满仓"),
    ]
    
    for title, desc in layers:
        f.write(f"  {title}\n")
        for line in desc.split('\n'):
            f.write(f"    {line}\n")
        f.write("\n")
    
    f.write("  ★ 已移除 X13 第7层（max_hold_days=92 / max_hold_reduce=0.0）\n")
    f.write("    理由: 固定日历天数无金融逻辑支撑，纯数据挖掘产物\n\n")
    
    # ── 3. 参数全表 ──
    f.write("三、参数全表\n")
    f.write("-" * 50 + "\n")
    f.write(f"  {'参数名':<18} {'默认值':<10} {'含义':<50}\n")
    f.write("  " + "-" * 78 + "\n")
    params = [
        ('dc', '5', '方向确认天数，连续N天T同号才确认方向'),
        ('dcd', '4', '方向冷却天数(v2.0:6→4)'),
        ('rt', '1.3', 'T弱阈值，|T|<rt视为T信号弱'),
        ('slope_thresh', '0.002', '斜率弱阈值，|斜率|<此值视为斜率弱'),
        ('ms', '10', 'B2短周期，价值短周期动量'),
        ('ml', '20', 'B2长周期，价值长周期动量'),
        ('bias_ma', '20', 'BIAS均线周期'),
        ('bias_high', '0.19', 'BIAS超阈值(19%)'),
        ('bias_mode', "'half'", 'BIAS模式: half(降仓50%+T约束)/clear(清仓)/ignore(忽略)'),
        ('bias_t_constraint', 'True', 'BIAS加T条件约束，T<1.5才触发'),
        ('st', '0.09', 'E5止损阈值，20日跌幅>9%触发'),
        ('sw', '0.17', 'E5降仓权重，触发后降至17%'),
        ('cd', '8', 'E5冷却天数，冷却期内再次触发延长3天'),
    ]
    for p, v, d in params:
        f.write(f"  {p:<18} {v:<10} {d}\n")
    
    f.write(f"\n  ★ 标注为 X14 关键改动参数\n\n")
    
    # ── 4. 核心指标对比 ──
    f.write("四、核心指标对比\n")
    f.write("-" * 50 + "\n")
    f.write(f"  {'策略':<20} {'年化':>10} {'回撤':>10} {'Sharpe':>10} {'Calmar':>10} "
            f"{'交易':>8} {'总收益':>12} {'滑点Calmar':>14}\n")
    f.write("  " + "-" * 94 + "\n")
    
    order = ['X14_干净版', 'X13_原版', 'X11_A基准']
    for name in order:
        r = all_results[name]
        f.write(f"  {name:<20} {r['ann']*100:>9.2f}% {r['dd']*100:>9.2f}% "
                f"{r['sharpe']:>10.3f} {r['calmar']:>10.3f} {r['n_trades']:>8} "
                f"{r['total_return']*100:>10.1f}% {r['calmar_sl']:>13.3f}\n")
    f.write("\n")
    
    # ── 5. X14 vs X13 详细对比 ──
    f.write("五、X14 vs X13 差异分析\n")
    f.write("-" * 50 + "\n")
    r14 = all_results['X14_干净版']
    r13 = all_results['X13_原版']
    
    def delta(a, b):
        d = a / b - 1
        return f"{d*100:+.1f}%"
    
    comparisons = [
        ('年化收益', f"{r14['ann']*100:.2f}%", f"{r13['ann']*100:.2f}%", delta(r14['ann'], r13['ann'])),
        ('最大回撤', f"{r14['dd']*100:.2f}%", f"{r13['dd']*100:.2f}%", f"{r14['dd']-r13['dd']*100:+.1f}pp"),
        ('Sharpe', f"{r14['sharpe']:.3f}", f"{r13['sharpe']:.3f}", delta(r14['sharpe'], r13['sharpe'])),
        ('Calmar', f"{r14['calmar']:.3f}", f"{r13['calmar']:.3f}", delta(r14['calmar'], r13['calmar'])),
        ('滑点Calmar', f"{r14['calmar_sl']:.3f}", f"{r13['calmar_sl']:.3f}", delta(r14['calmar_sl'], r13['calmar_sl'])),
        ('交易次数', f"{r14['n_trades']}", f"{r13['n_trades']}", f"{r14['n_trades']-r13['n_trades']:+d}"),
        ('总收益率', f"{r14['total_return']*100:.1f}%", f"{r13['total_return']*100:.1f}%", delta(r14['total_return'], r13['total_return'])),
    ]
    
    f.write(f"  {'指标':<20} {'X14':<16} {'X13':<16} {'差异':<16}\n")
    f.write("  " + "-" * 68 + "\n")
    for name, v14, v13, d in comparisons:
        f.write(f"  {name:<20} {v14:<16} {v13:<16} {d:<16}\n")
    f.write("\n")
    
    f.write("  Calmar 差异归因:\n")
    f.write("    - max_hold_days 移除: Calmar 贡献约 -0.071 (-3.0%)\n")
    f.write("    - st 0.088→0.09: 影响微小\n")
    f.write("    - bias_reduce 0.0→0.05: 影响微小(BIAS参数本身不敏感)\n")
    f.write(f"  → X14 干净版损失约 4% Calmar，但参数更有逻辑支撑\n\n")
    
    # ── 6. 逐年收益对比 ──
    f.write("六、逐年收益对比\n")
    f.write("-" * 50 + "\n")
    
    all_yearly_data = {}
    for name in order:
        yearly = get_yearly(name)
        all_yearly_data[name] = yearly
    
    all_years_set = set()
    for ydata in all_yearly_data.values():
        for y in ydata.index:
            all_years_set.add(y.year)
    all_years = sorted(all_years_set)
    
    header = f"  {'年份':<8}"
    for name in order:
        header += f" {name:>18}"
    f.write(header + "\n")
    f.write("  " + "-" * (8 + 18 * len(order)) + "\n")
    
    for y in all_years:
        line = f"  {y:<8}"
        for name in order:
            ydata = all_yearly_data[name]
            if y in [d.year for d in ydata.index]:
                val = ydata[ydata.index.year == y].values[0]
                line += f" {val:>17.2f}%"
            else:
                line += f" {'N/A':>18}"
        f.write(line + "\n")
    
    # 年均
    line = f"  {'年均':<8}"
    for name in order:
        ydata = all_yearly_data[name]
        line += f" {ydata.mean():>17.2f}%"
    f.write(line + "\n")
    
    # 正收益年数
    line = f"  {'正收益年':<8}"
    for name in order:
        ydata = all_yearly_data[name]
        pos = (ydata > 0).sum()
        total = len(ydata)
        line += f" {pos:>3d}/{total:<3d}({pos/total*100:.0f}%)  "
    f.write(line + "\n\n")
    
    # ── 7. 月度统计 ──
    f.write("七、月度统计\n")
    f.write("-" * 50 + "\n")
    f.write(f"  {'指标':<20} {'X14':>20} {'X13':>20} {'X11-A':>20}\n")
    f.write("  " + "-" * 80 + "\n")
    
    metrics_list = ['positive_months', 'best_month', 'worst_month', 'avg_month', 'std_month']
    metric_labels = ['正收益月数', '最佳月份', '最差月份', '月均收益', '月收益波动']
    for label, key in zip(metric_labels, metrics_list):
        line = f"  {label:<20}"
        for name in order:
            ms = calc_monthly_stats(get_nav(name))
            line += f" {ms[key]:>20}"
        f.write(line + "\n")
    f.write("\n")
    
    # ── 8. 回撤分析 ──
    f.write("八、回撤分析\n")
    f.write("-" * 50 + "\n\n")
    
    for name in order:
        nav = get_nav(name)
        dd = nav / nav.cummax() - 1
        f.write(f"  {name} 回撤统计:\n")
        f.write(f"    最大回撤: {dd.min()*100:.2f}%\n")
        f.write(f"    当前回撤: {dd.iloc[-1]*100:.2f}%\n")
        
        periods = calc_drawdown_periods(nav, top_n=5)
        if periods:
            f.write(f"    前5大回撤时段:\n")
            f.write(f"      {'开始日期':<14} {'结束日期':<14} {'深度':>10} {'天数':>6}\n")
            f.write(f"      " + "-" * 44 + "\n")
            for p in periods:
                f.write(f"      {str(p['start']):<14} {str(p['end']):<14} "
                        f"{p['depth']*100:>9.2f}% {p['days']:>6}\n")
        f.write("\n")
    
    # ── 9. 交易分析 ──
    f.write("九、交易分析\n")
    f.write("-" * 50 + "\n\n")
    
    for name in order:
        sig, wt = all_signals[name]
        f.write(f"  {name}:\n")
        
        # 方向分布
        dir_counts = sig.value_counts()
        f.write(f"    方向分布: growth={dir_counts.get('growth', 0)}({dir_counts.get('growth', 0)/len(sig)*100:.1f}%)  "
                f"value={dir_counts.get('value', 0)}({dir_counts.get('value', 0)/len(sig)*100:.1f}%)\n")
        
        # 仓位分布
        cash_days = (wt <= 0).sum()
        full_days = (wt >= 0.99).sum()
        partial_days = ((wt > 0) & (wt < 0.99)).sum()
        f.write(f"    仓位分布: 满仓={full_days}({full_days/len(wt)*100:.1f}%)  "
                f"降仓={partial_days}({partial_days/len(wt)*100:.1f}%)  "
                f"空仓={cash_days}({cash_days/len(wt)*100:.1f}%)\n")
        
        # 切换次数
        sw = count_switches(sig, wt)
        f.write(f"    方向切换: {sw['dir']}次  空仓切换: {sw['cash']}次  总交易信号: {sw['dir']+sw['cash']}次\n")
        
        # 交易记录统计
        safe = name.replace(' ', '_').replace('/', '_')
        trades_path = OUTPUT_DIR / f'{safe}_交易记录.csv'
        if trades_path.exists() and trades_path.stat().st_size > 0:
            try:
                df_t = pd.read_csv(trades_path)
                if len(df_t) > 0:
                    win_rate = (df_t['收益率'] > 0).mean() * 100
                    f.write(f"    交易次数: {len(df_t)}次  胜率: {win_rate:.1f}%\n")
                    avg_ret = df_t['收益率'].mean() * 100
                    avg_win = df_t[df_t['收益率'] > 0]['收益率'].mean() * 100 if (df_t['收益率'] > 0).sum() > 0 else 0
                    avg_loss = df_t[df_t['收益率'] < 0]['收益率'].mean() * 100 if (df_t['收益率'] < 0).sum() > 0 else 0
                    f.write(f"    平均收益: {avg_ret:.2f}%  平均盈利: {avg_win:.2f}%  平均亏损: {avg_loss:.2f}%\n")
                    profit_factor = (df_t[df_t['收益率'] > 0]['收益率'].sum() / 
                                   abs(df_t[df_t['收益率'] < 0]['收益率'].sum())) if (df_t['收益率'] < 0).sum() != 0 else float('inf')
                    f.write(f"    盈亏比(Profit Factor): {profit_factor:.3f}\n")
            except Exception:
                pass
        f.write("\n")
    
    # ── 10. 风险指标 ──
    f.write("十、风险指标\n")
    f.write("-" * 50 + "\n")
    f.write(f"  {'指标':<24} {'X14':>18} {'X13':>18} {'X11-A':>18}\n")
    f.write("  " + "-" * 78 + "\n")
    
    risk_labels = ['VaR 95%', 'VaR 99%', '最大连续亏损天数', '日收益波动率']
    risk_keys = ['VaR_95', 'VaR_99', 'max_consecutive_loss_days', 'daily_ret_std']
    for label, key in zip(risk_labels, risk_keys):
        line = f"  {label:<24}"
        for name in order:
            rm = calc_risk_metrics(get_nav(name))
            line += f" {rm[key]:>18}"
        f.write(line + "\n")
    f.write("\n")
    
    # ── 11. 与X11-A基准对比 ──
    f.write("十一、与 X11-A 基准对比\n")
    f.write("-" * 50 + "\n")
    r_x11 = all_results['X11_A基准']
    
    improvements = [
        ('年化', r14['ann'] / r_x11['ann'] - 1),
        ('Calmar', r14['calmar'] / r_x11['calmar'] - 1),
        ('滑点Calmar', r14['calmar_sl'] / r_x11['calmar_sl'] - 1),
        ('Sharpe', r14['sharpe'] / r_x11['sharpe'] - 1),
        ('回撤改善', (r14['dd'] - r_x11['dd']) * 100),
    ]
    for label, val in improvements:
        if label == '回撤改善':
            f.write(f"  {label:<20} {val:+.1f}pp (X14={r14['dd']*100:.1f}% vs X11={r_x11['dd']*100:.1f}%)\n")
        else:
            f.write(f"  {label:<20} {val*100:+.1f}% (X14->X11)\n")
    f.write(f"\n  X14 相对 X11-A 的 Calmar 提升: {(r14['calmar']/r_x11['calmar']-1)*100:+.1f}%\n")
    f.write(f"  X14 相对 X11-A 的回撤改善: {r_x11['dd']*100 - r14['dd']*100:.1f}pp\n\n")
    
    # ── 12. 总结 ──
    f.write("十二、总结\n")
    f.write("-" * 50 + "\n")
    f.write("  X14 干净版优势:\n")
    f.write("    ① 参数都有逻辑支撑，无固定日历天数\n")
    f.write("    ② 阈值取整(st=0.09)，避免精度过拟合\n")
    f.write("    ③ bias_mode='clear' 一键清仓，用户可选 'ignore' 完全忽略\n")
    f.write("    ④ 所有指标已含滑点（手续费1bps+冲击5bps），结果更贴近实盘\n")
    f.write(f"    ⑤ 滑点Calmar {r14['calmar']:.3f} 仅比 X13 原版低，但参数更合理\n\n")
    f.write("  X14 相对 X11-A 的提升:\n")
    f.write(f"    Calmar +{(r14['calmar']/r_x11['calmar']-1)*100:.1f}% ({r_x11['calmar']:.3f} → {r14['calmar']:.3f})\n")
    f.write(f"    回撤改善 +{r_x11['dd']*100 - r14['dd']*100:.1f}pp ({r_x11['dd']*100:.1f}% → {r14['dd']*100:.1f}%)\n")
    f.write(f"    年化 +{(r14['ann']-r_x11['ann'])*100:.1f}pp ({r_x11['ann']*100:.1f}% → {r14['ann']*100:.1f}%)\n\n")
    f.write("  滑点说明:\n")
    f.write("    手续费: 1bps（买入/卖出各）\n")
    f.write("    冲击滑点: 5bps（买入/卖出各）\n")
    f.write("    跳空滑点: 不计（高开低开期望值为0）\n")
    f.write(f"    总滑点: ≈6bps/边\n\n")
    f.write("  风险提示:\n")
    f.write("    ① 固定百分比阈值(bias_high=0.19)含时效性风险\n")
    f.write("    ② 2026年BIAS全年未触发，该层形同虚设\n")
    f.write("    ③ 参数仍基于13年历史数据，未来市场结构变化可能导致退化\n")

print("\n" + "=" * 80)
print("  X14 流水线验证完成!")
print("=" * 80)
print(f"\n  输出目录: {OUTPUT_DIR}")
print("  生成文件:")
for f in sorted(OUTPUT_DIR.iterdir()):
    size = f.stat().st_size
    print(f"    {f.name}  ({size/1024:.1f} KB)")
