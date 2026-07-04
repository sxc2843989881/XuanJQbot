"""拆解调仓类型: 方向切换 vs 权重变动"""
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
sig, wt = build_core(bias_mode='clear')

# 爬一遍数据，标记每次变动的原因
slope_thresh=0.002; rt=1.3; st=0.09; sw=0.17; cd=8
ms=10; ml=20; dc=5; dcd=6; bias_ma=20; bias_high=0.19

V_MOM_S = V_CLOSE.pct_change(ms)
V_MOM_L = V_CLOSE.pct_change(ml)

# 判断每个日期的状态
weak_slope = SLOPE.abs() < slope_thresh
weak_t = T.abs() < rt
is_weak = weak_t & weak_slope

G_MA = G_CLOSE.rolling(bias_ma).mean()
V_MA = V_CLOSE.rolling(bias_ma).mean()
G_BIAS = G_CLOSE / G_MA - 1
V_BIAS = V_CLOSE / V_MA - 1

bias_g = (sig == 'growth') & (G_BIAS > bias_high)
bias_v = (sig == 'value') & (V_BIAS > bias_high)
bias_trig = bias_g | bias_v

# 拆解调仓
changes = []
prev_wt = None
prev_dir = None
for i in range(len(wt)):
    d = wt.index[i]
    w = wt.iloc[i]
    s = sig.iloc[i] if i < len(sig) else None
    if prev_wt is not None and (w != prev_wt or s != prev_dir):
        dir_changed = (s != prev_dir)
        wt_changed = (w != prev_wt)
        
        # 判断原因
        reason = ""
        if dir_changed:
            reason = "方向切换"
        elif wt_changed:
            if w == 0.0 and prev_wt > 0:
                if bias_trig.iloc[i]:
                    reason = "BIAS清仓"
                elif is_weak.iloc[i]:
                    reason = "T+斜率弱→空仓"
                else:
                    reason = "E5或其他清仓"
            elif w > 0 and prev_wt == 0.0:
                reason = "恢复仓位"
            elif w < prev_wt and w > 0:
                reason = "E5降仓"
            elif w > prev_wt:
                reason = "E5恢复"
        
        changes.append({
            'date': d, 'prev_wt': prev_wt, 'new_wt': w,
            'prev_dir': prev_dir, 'new_dir': s,
            'dir_changed': dir_changed, 'wt_changed': wt_changed,
            'reason': reason
        })
    prev_wt = w
    prev_dir = s

print("=" * 70)
print("  调仓类型拆解 (416次的构成)")
print("=" * 70)

# 按原因分类
from collections import Counter
reason_counts = Counter(c['reason'] for c in changes)
print(f"\n  {'原因':<20s} {'次数':>6s} {'占比':>8s}")
print("  " + "-" * 36)
total = len(changes)
for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
    print(f"  {reason:<20s} {count:>6d} {count/total*100:>7.1f}%")

# 方向切换 vs 权重变动
dir_only = sum(1 for c in changes if c['dir_changed'] and not c['wt_changed'])
wt_only = sum(1 for c in changes if c['wt_changed'] and not c['dir_changed'])
both = sum(1 for c in changes if c['dir_changed'] and c['wt_changed'])

print(f"\n  {'类型':<20s} {'次数':>6s} {'占比':>8s}")
print("  " + "-" * 36)
print(f"  {'方向切换(权重不变)':<20s} {dir_only:>6d} {dir_only/total*100:>7.1f}%")
print(f"  {'权重变动(方向不变)':<20s} {wt_only:>6d} {wt_only/total*100:>7.1f}%")
print(f"  {'方向+权重同时':<20s} {both:>6d} {both/total*100:>7.1f}%")

# 权重变动细分
wt_change_reasons = Counter(c['reason'] for c in changes if c['wt_changed'] and not c['dir_changed'])
print(f"\n  权重变动内部细分:")
for r, cnt in sorted(wt_change_reasons.items(), key=lambda x: -x[1]):
    if r != '方向切换':
        print(f"    {r:<20s}: {cnt:>4d}次 ({cnt/wt_only*100:.0f}%)")

# 重点: BIAS清仓对调仓的贡献
bias_related = sum(1 for c in changes if 'BIAS' in c['reason'])
print(f"\n  BIAS相关调仓: {bias_related}次 ({bias_related/total*100:.1f}%)")
print(f"    (触发{bias_trig.sum()}天 → 清仓+恢复 共{bias_related}次操作)")

# E5相关
e5_total = sum(1 for c in changes if 'E5' in c['reason'] or '降仓' in c['reason'] or '恢复' in c['reason'] and 'BIAS' not in c['reason'])
print(f"  E5止损相关调仓: 次数在'降仓'+'恢复'中已包含")

# 空仓相关(非BIAS)
cash_related = sum(1 for c in changes if c['reason'] in ['T+斜率弱→空仓', '恢复仓位'])
print(f"  T+斜率空仓相关: {cash_related}次 ({cash_related/total*100:.1f}%)")
