"""BIAS信号触发、交易次数、调仓间隔分析"""
import sys
sys.path.insert(0, r'c:\XuanJLH\Qbot\custom\backtests')
sys.path.insert(0, r'c:\XuanJLH\Qbot\custom\research\成长价值轮动\X12_偏离度空仓版')
sys.path.insert(0, r'c:\XuanJLH\Qbot\custom\research\成长价值轮动\X14_统一策略干净版')

import numpy as np
import pandas as pd
from optimize_runner import (
    G_CLOSE, V_CLOSE, RATIO, RATIO_MA20, RATIO_DEV, MA20_SLOPE,
    V_MOM20, G_DD20, V_DD20, BASE_DIR,
    run_backtest, calc_metrics, count_switches,
)
from run_x33_reduce_trades import RATIO_DEV_Z
from backtest_x14_engine import build_core

T = RATIO_DEV_Z
SLOPE = MA20_SLOPE

# 获取信号和权重
sig, wt = build_core(bias_mode='clear')
result = run_backtest(sig, wt)
m = calc_metrics(result)
sw = count_switches(sig, wt)

df = result.to_dataframe()

# 找所有权重变化的日子（调仓日）
wt_changes = []
prev_wt = None
prev_dir = None
for i in range(len(wt)):
    d = wt.index[i]
    w = wt.iloc[i]
    s = sig.iloc[i] if i < len(sig) else None
    if prev_wt is not None and (w != prev_wt or s != prev_dir):
        wt_changes.append({
            'date': d,
            'prev_wt': prev_wt,
            'new_wt': w,
            'prev_dir': prev_dir,
            'new_dir': s,
            'change_type': '仓位' if w != prev_wt else '方向'
        })
    prev_wt = w
    prev_dir = s

# 找BIAS触发日
bias_ma = 20
bias_high = 0.19
G_MA = G_CLOSE.rolling(bias_ma).mean()
V_MA = V_CLOSE.rolling(bias_ma).mean()
G_BIAS = G_CLOSE / G_MA - 1
V_BIAS = V_CLOSE / V_MA - 1

g_bias_trig = (sig == 'growth') & (G_BIAS > bias_high)
v_bias_trig = (sig == 'value') & (V_BIAS > bias_high)
bias_trig_days = g_bias_trig | v_bias_trig
bias_dates = bias_trig_days[bias_trig_days].index

print("=" * 80)
print("  BIAS信号触发与调仓间隔分析")
print("=" * 80)

# 1. 基本统计
print(f"\n--- 基本统计 ---")
print(f"  BIAS信号触发天数: {len(bias_dates)}天")
print(f"  BIAS离散事件: 6次")
print(f"  总交易次数: {m['n_trades']}")
print(f"  方向切换: {sw['dir']}次")
print(f"  空仓切换: {sw['cash']}次")
print(f"  总调仓次数: {len(wt_changes)}次 (仓位或方向变动)")
print(f"  回测天数: {len(wt)}天")
print(f"  平均每几天调仓: {len(wt)/max(len(wt_changes),1):.1f}天")

# 2. BIAS触发与调仓的关系
print(f"\n--- BIAS触发当日是否同时有调仓 ---")
for d in bias_dates:
    nearby = [c for c in wt_changes if abs((c['date'] - d).days) <= 1]
    if nearby:
        for c in nearby:
            print(f"  BIAS触发 {d.date()}, 附近有调仓: {c['date'].date()} "
                  f"{c['prev_dir']}→{c['new_dir']} wt:{c['prev_wt']}→{c['new_wt']}")
    else:
        print(f"  BIAS触发 {d.date()}, 当日/次日无调仓")

# 3. 最近20次调仓（按时间倒序）
print(f"\n--- 最近20次调仓记录（倒序）---")
print(f"  {'日期':<14s} {'方向变化':<18s} {'权重变化':<16s} {'间隔(天)':<10s} {'说明':<20s}")
print("  " + "-" * 78)

recent = sorted(wt_changes, key=lambda x: x['date'], reverse=True)[:20]
prev_date = None
for c in recent:
    gap = (prev_date - c['date']).days if prev_date else 0
    gap_str = f"{gap}天" if gap else "—"
    note = ""
    if c['date'] in bias_dates:
        note = "← BIAS清仓"
    elif c['new_wt'] == 0.0 and c['prev_wt'] > 0:
        note = "← 空仓"
    elif c['new_wt'] > 0 and c['prev_wt'] == 0.0:
        note = "← 恢复仓位"
    print(f"  {c['date'].date():<12s}  {str(c['prev_dir']):>6s}→{str(c['new_dir']):<6s}  "
          f"{c['prev_wt']:>4.2f}→{c['new_wt']:<4.2f}  {gap_str:<10s} {note:<20s}")
    prev_date = c['date']

# 4. 调仓间隔分布
print(f"\n--- 调仓间隔分布 ---")
gaps = []
prev_date = None
for c in sorted(wt_changes, key=lambda x: x['date']):
    if prev_date:
        gap = (c['date'] - prev_date).days
        gaps.append(gap)
    prev_date = c['date']

gap_series = pd.Series(gaps)
print(f"  最小间隔: {gap_series.min()}天")
print(f"  最大间隔: {gap_series.max()}天")
print(f"  平均间隔: {gap_series.mean():.1f}天")
print(f"  中位间隔: {gap_series.median():.0f}天")
print(f"  ≤1天(连续调仓): {(gap_series<=1).sum()}次 ({((gap_series<=1).sum()/len(gap_series))*100:.1f}%)")
print(f"  1-5天: {((gap_series>1)&(gap_series<=5)).sum()}次")
print(f"  5-10天: {((gap_series>5)&(gap_series<=10)).sum()}次")
print(f"  10-30天: {((gap_series>10)&(gap_series<=30)).sum()}次")
print(f"  >30天: {((gap_series>30)&(gap_series<=100)).sum()}次")
print(f"  >100天: {(gap_series>100).sum()}次")

# 5. BIAS触发后多久恢复仓位
print(f"\n--- BIAS清仓后多久恢复仓位 ---")
for d in bias_dates:
    # 找之后第一个wt>0的日子
    idx = wt.index.get_loc(d)
    if idx + 1 < len(wt):
        future = wt.iloc[idx+1:]
        recovery_idx = future[future > 0].index
        if len(recovery_idx) > 0:
            recovery_date = recovery_idx[0]
            days_to_recover = (recovery_date - d).days
            print(f"  {d.date()}: 清仓→恢复({recovery_date.date()}) 间隔{days_to_recover}天")

# 6. 历年调仓次数
print(f"\n--- 历年调仓次数 ---")
wt_changes_series = pd.DataFrame(wt_changes)
wt_changes_series['date'] = pd.to_datetime(wt_changes_series['date'])
wt_changes_series['year'] = wt_changes_series['date'].dt.year
yearly = wt_changes_series.groupby('year').size()
for yr in range(2013, 2027):
    cnt = yearly.get(yr, 0)
    print(f"  {yr}: {cnt}次调仓")
