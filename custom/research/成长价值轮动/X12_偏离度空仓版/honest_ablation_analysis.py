"""honest_ablation_analysis.py — 诚实消融分析
================================================================
用户质疑: 去掉每层后年化收益是否仍然保持较高?
如果是,说明各层可能是装饰性的,策略可能是过拟合拼凑

针对最终最优版X23(z=1.5, slope=0.002)做详细消融
重点分析: 每层对年化收益的贡献 vs 对回撤的贡献
================================================================
"""
import sys
sys.path.insert(0, r'c:\XuanJLH\Qbot\custom\backtests')
sys.path.insert(0, r'c:\XuanJLH\Qbot\custom\research\成长价值轮动\X12_偏离度空仓版')

from pathlib import Path
import numpy as np
import pandas as pd
from optimize_runner import (
    G_CLOSE, V_CLOSE, RATIO, RATIO_MA20, RATIO_DEV, MA20_SLOPE, SLOPE_OK,
    V_MOM20, G_DD20, V_DD20, BASE_DIR,
    run_backtest, calc_metrics, count_switches, test_strategy, print_result,
    build_x11a, build_x12,
)

# 公共因子(最终最优参数)
Z_THRESH = 1.5
SLOPE_THRESH = 0.002  # 最终最优参数
N_CONFIRM = 4
STOP_THRESHOLD = 0.10
STOP_WEIGHT = 0.30

# z-score计算
RATIO_DEV_STD20 = RATIO_DEV.rolling(20).std()
RATIO_DEV_Z = RATIO_DEV / RATIO_DEV_STD20


def calc_full_metrics(signal, weight):
    """计算完整指标"""
    result = run_backtest(signal, weight)
    m = calc_metrics(result)
    # 年度收益
    df = result.to_dataframe()
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    yearly = {}
    for year in sorted(df.index.year.unique()):
        r = df[df.index.year == year]['daily_ret']
        yearly[year] = (1 + r).prod() - 1
    return m, yearly


# ============================================================
# 完整版X23(最终最优)
# ============================================================
def build_x23_full():
    """X23完整版: F1+A1+F0双确认(z-score)+B2+E5"""
    dir_s = BASE_DIR.copy()
    mask = np.ones(len(dir_s), dtype=bool)
    for k in range(1, N_CONFIRM):
        mask = mask & (dir_s.values == dir_s.shift(k).values)
    confirmed = dir_s.where(mask, np.nan)
    wt = pd.Series(1.0, index=dir_s.index)
    low_dev_z = RATIO_DEV_Z.abs() < Z_THRESH
    low_slope = MA20_SLOPE.abs() < SLOPE_THRESH
    both_weak = low_dev_z & low_slope
    wt[both_weak] = 0.0
    dir_s = confirmed.ffill()
    wrong_value = (dir_s == 'value') & (V_MOM20 <= 0)
    dir_s[wrong_value] = 'growth'
    gs = (dir_s == 'growth') & (G_DD20 < -STOP_THRESHOLD)
    vs = (dir_s == 'value') & (V_DD20 < -STOP_THRESHOLD)
    wt[gs | vs] = wt[gs | vs] * STOP_WEIGHT
    return dir_s, wt


# ============================================================
# 去掉各层的版本
# ============================================================
def build_no_f0():
    """去掉F0双确认空仓(保留A1+B2+E5)"""
    dir_s = BASE_DIR.copy()
    mask = np.ones(len(dir_s), dtype=bool)
    for k in range(1, N_CONFIRM):
        mask = mask & (dir_s.values == dir_s.shift(k).values)
    confirmed = dir_s.where(mask, np.nan)
    wt = pd.Series(1.0, index=dir_s.index)
    # 无F0空仓
    dir_s = confirmed.ffill()
    wrong_value = (dir_s == 'value') & (V_MOM20 <= 0)
    dir_s[wrong_value] = 'growth'
    gs = (dir_s == 'growth') & (G_DD20 < -STOP_THRESHOLD)
    vs = (dir_s == 'value') & (V_DD20 < -STOP_THRESHOLD)
    wt[gs | vs] = STOP_WEIGHT
    return dir_s, wt


def build_no_b2():
    """去掉B2过滤(保留F1+A1+F0+E5)"""
    dir_s = BASE_DIR.copy()
    mask = np.ones(len(dir_s), dtype=bool)
    for k in range(1, N_CONFIRM):
        mask = mask & (dir_s.values == dir_s.shift(k).values)
    confirmed = dir_s.where(mask, np.nan)
    wt = pd.Series(1.0, index=dir_s.index)
    low_dev_z = RATIO_DEV_Z.abs() < Z_THRESH
    low_slope = MA20_SLOPE.abs() < SLOPE_THRESH
    both_weak = low_dev_z & low_slope
    wt[both_weak] = 0.0
    dir_s = confirmed.ffill()
    # 无B2
    gs = (dir_s == 'growth') & (G_DD20 < -STOP_THRESHOLD)
    vs = (dir_s == 'value') & (V_DD20 < -STOP_THRESHOLD)
    wt[gs | vs] = wt[gs | vs] * STOP_WEIGHT
    return dir_s, wt


def build_no_e5():
    """去掉E5止损(保留F1+A1+F0+B2)"""
    dir_s = BASE_DIR.copy()
    mask = np.ones(len(dir_s), dtype=bool)
    for k in range(1, N_CONFIRM):
        mask = mask & (dir_s.values == dir_s.shift(k).values)
    confirmed = dir_s.where(mask, np.nan)
    wt = pd.Series(1.0, index=dir_s.index)
    low_dev_z = RATIO_DEV_Z.abs() < Z_THRESH
    low_slope = MA20_SLOPE.abs() < SLOPE_THRESH
    both_weak = low_dev_z & low_slope
    wt[both_weak] = 0.0
    dir_s = confirmed.ffill()
    wrong_value = (dir_s == 'value') & (V_MOM20 <= 0)
    dir_s[wrong_value] = 'growth'
    # 无E5
    return dir_s, wt


def build_no_a1():
    """去掉A1确认(保留F1+F0+B2+E5)"""
    dir_s = BASE_DIR.copy()
    # 无A1确认
    confirmed = dir_s
    wt = pd.Series(1.0, index=dir_s.index)
    low_dev_z = RATIO_DEV_Z.abs() < Z_THRESH
    low_slope = MA20_SLOPE.abs() < SLOPE_THRESH
    both_weak = low_dev_z & low_slope
    wt[both_weak] = 0.0
    dir_s = confirmed.ffill()
    wrong_value = (dir_s == 'value') & (V_MOM20 <= 0)
    dir_s[wrong_value] = 'growth'
    gs = (dir_s == 'growth') & (G_DD20 < -STOP_THRESHOLD)
    vs = (dir_s == 'value') & (V_DD20 < -STOP_THRESHOLD)
    wt[gs | vs] = wt[gs | vs] * STOP_WEIGHT
    return dir_s, wt


def build_only_f1():
    """只剩F1基础信号(裸策略)"""
    dir_s = BASE_DIR.copy()
    wt = pd.Series(1.0, index=dir_s.index)
    return dir_s, wt


def build_f1_a1():
    """F1+A1(裸趋势跟踪)"""
    dir_s = BASE_DIR.copy()
    mask = np.ones(len(dir_s), dtype=bool)
    for k in range(1, N_CONFIRM):
        mask = mask & (dir_s.values == dir_s.shift(k).values)
    confirmed = dir_s.where(mask, np.nan)
    dir_s = confirmed.ffill()
    wt = pd.Series(1.0, index=dir_s.index)
    return dir_s, wt


# ============================================================
# 运行分析
# ============================================================
print("=" * 90)
print("  诚实消融分析: X23(z=1.5, slope=0.002) 每层对年化收益的真实贡献")
print("=" * 90)

configs = [
    ('X23完整版', build_x23_full, 'F1+A1+F0(z)+B2+E5 (完整)'),
    ('去掉F0双确认', build_no_f0, 'F1+A1+B2+E5 (无F0空仓)'),
    ('去掉B2过滤', build_no_b2, 'F1+A1+F0(z)+E5 (无B2)'),
    ('去掉E5止损', build_no_e5, 'F1+A1+F0(z)+B2 (无E5)'),
    ('去掉A1确认', build_no_a1, 'F1+F0(z)+B2+E5 (无A1)'),
    ('裸F1', build_only_f1, '只有F1基础信号'),
    ('F1+A1', build_f1_a1, '裸趋势跟踪'),
]

print(f"\n  {'配置':<20} {'说明':<35} {'年化':>8} {'回撤':>8} {'Sharpe':>8} {'Calmar':>8} {'交易':>6}")
print(f"  {'-'*100}")

all_results = {}
all_yearly = {}
for name, func, desc in configs:
    sig, wt = func()
    m, yearly = calc_full_metrics(sig, wt)
    all_results[name] = m
    all_yearly[name] = yearly
    print(f"  {name:<20} {desc:<35} {m['ann']*100:>7.2f}% {m['dd']*100:>7.2f}% "
          f"{m['sharpe']:>8.3f} {m['calmar']:>8.3f} {m['n_trades']:>6}")

# ============================================================
# 年化收益变化分析(核心)
# ============================================================
print("\n" + "=" * 90)
print("  【核心分析】每层对年化收益的真实贡献")
print("=" * 90)

full_ann = all_results['X23完整版']['ann']
full_dd = all_results['X23完整版']['dd']
full_calmar = all_results['X23完整版']['calmar']

print(f"\n  基准: X23完整版 年化{full_ann*100:.2f}% 回撤{full_dd*100:.2f}% Calmar{full_calmar:.3f}")
print(f"\n  {'去掉的层':<20} {'年化':>8} {'年化变化':>10} {'回撤':>8} {'回撤变化':>10} {'Calmar':>8} {'收益贡献':>10} {'风控贡献':>10}")
print(f"  {'-'*100}")

for name in ['去掉F0双确认', '去掉B2过滤', '去掉E5止损', '去掉A1确认']:
    m = all_results[name]
    ann_change = (m['ann'] - full_ann) * 100
    dd_change = (m['dd'] - full_dd) * 100
    calmar_change = m['calmar'] - full_calmar

    # 判断贡献类型
    if ann_change < -1:
        rev_contrib = "正贡献"
    elif ann_change > 1:
        rev_contrib = "负贡献"  # 去掉后收益升高=该层牺牲收益
    else:
        rev_contrib = "中性"

    if dd_change > 2:
        risk_contrib = "正贡献"
    elif dd_change < -2:
        risk_contrib = "负贡献"
    else:
        risk_contrib = "中性"

    print(f"  {name:<20} {m['ann']*100:>7.2f}% {ann_change:>+9.2f}pp {m['dd']*100:>7.2f}% "
          f"{dd_change:>+9.2f}pp {m['calmar']:>8.3f} {rev_contrib:>10} {risk_contrib:>10}")

# ============================================================
# 与裸策略对比(关键过拟合检测)
# ============================================================
print("\n" + "=" * 90)
print("  【过拟合检测】与裸策略对比")
print("=" * 90)

bare_ann = all_results['裸F1']['ann']
bare_dd = all_results['裸F1']['dd']
f1a1_ann = all_results['F1+A1']['ann']
f1a1_dd = all_results['F1+A1']['dd']

print(f"\n  裸F1(只有方向判断):     年化{bare_ann*100:.2f}% 回撤{bare_dd*100:.2f}%")
print(f"  F1+A1(裸趋势跟踪):     年化{f1a1_ann*100:.2f}% 回撤{f1a1_dd*100:.2f}%")
print(f"  X23完整版:              年化{full_ann*100:.2f}% 回撤{full_dd*100:.2f}%")
print(f"\n  完整版 vs 裸F1:")
print(f"    年化提升: {(full_ann-bare_ann)*100:+.2f}pp")
print(f"    回撤改善: {(full_dd-bare_dd)*100:+.2f}pp")
print(f"\n  完整版 vs F1+A1:")
print(f"    年化提升: {(full_ann-f1a1_ann)*100:+.2f}pp")
print(f"    回撤改善: {(full_dd-f1a1_dd)*100:+.2f}pp")

# ============================================================
# 年度收益对比(看是否每年都有效)
# ============================================================
print("\n" + "=" * 90)
print("  【年度收益对比】各层在不同年份的表现")
print("=" * 90)

years = sorted(all_yearly['X23完整版'].keys())
print(f"\n  {'年份':>6} ", end='')
for name in ['X23完整版', '去掉F0双确认', '去掉B2过滤', '去掉E5止损', '裸F1']:
    print(f" {name:>12}", end='')
print()
print(f"  {'-'*70}")

for year in years:
    print(f"  {year:>6} ", end='')
    for name in ['X23完整版', '去掉F0双确认', '去掉B2过滤', '去掉E5止损', '裸F1']:
        ret = all_yearly[name].get(year, 0)
        print(f" {ret*100:>11.2f}%", end='')
    print()

# ============================================================
# 诚实结论
# ============================================================
print("\n" + "=" * 90)
print("  【诚实结论】")
print("=" * 90)

print(f"""
  1. 年化收益来源分析:
     - 裸F1: {bare_ann*100:.2f}%
     - F1+A1: {f1a1_ann*100:.2f}%
     - X23完整: {full_ann*100:.2f}%
     - 完整版比裸F1高{(full_ann-bare_ann)*100:.2f}pp,比F1+A1高{(full_ann-f1a1_ann)*100:.2f}pp

  2. 各层真实贡献类型:""")

for name in ['去掉F0双确认', '去掉B2过滤', '去掉E5止损', '去掉A1确认']:
    m = all_results[name]
    ann_change = (m['ann'] - full_ann) * 100
    dd_change = (m['dd'] - full_dd) * 100
    if ann_change > 1:
        rev_type = "牺牲收益换风控"
    elif ann_change < -1:
        rev_type = "收益增强"
    else:
        rev_type = "对收益影响小"
    print(f"     - {name}: 年化{ann_change:+.2f}pp 回撤{dd_change:+.2f}pp → {rev_type}")

print(f"""
  3. 过拟合风险评估:
     - 如果去掉某层后年化收益几乎不变(±1pp内),该层可能是装饰性的
     - 如果去掉某层后年化收益大幅下降(>3pp),该层是收益核心
     - 如果去掉某层后年化收益反而升高,该层是纯粹风控层(牺牲收益)""")

# 统计装饰性因子
decorative_count = 0
for name in ['去掉F0双确认', '去掉B2过滤', '去掉E5止损', '去掉A1确认']:
    m = all_results[name]
    ann_change = abs((m['ann'] - full_ann) * 100)
    if ann_change < 1:
        decorative_count += 1

print(f"\n     装饰性因子数量(<1pp变化): {decorative_count}/4")

if decorative_count >= 2:
    print(f"     ⚠️ 警告: 存在{decorative_count}个装饰性因子,过拟合风险较高!")
elif decorative_count >= 1:
    print(f"     ⚠️ 注意: 存在{decorative_count}个装饰性因子,需关注")
else:
    print(f"     ✅ 各层对收益都有明显影响,装饰性因子少")
