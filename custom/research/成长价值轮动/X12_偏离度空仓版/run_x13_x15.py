"""run_x13_x15.py — 第1-3轮优化: 统一F0与斜率确认的三种方案
================================================================
X13: 删除斜率层，只保留F0空仓（偏离度统一判断趋势）
X14: 删除F0，斜率不满足时空仓（斜率统一判断趋势）
X15: F0加滞回机制（保留斜率，减少交易切换）
================================================================
"""
import sys
sys.path.insert(0, r'c:\XuanJLH\Qbot\custom\research\成长价值轮动\X12_偏离度空仓版')
from optimize_runner import *
import numpy as np
import pandas as pd

# ============================================================
# X13: 删除斜率确认层，只保留F0空仓
# ============================================================
def build_x13(dev_threshold=0.005):
    """X13: F1+F0(偏离度空仓)+A1(4天)+B2+E5 — 删除斜率层

    逻辑: F0的偏离度已经判断了"趋势不明确"，不需要斜率再判断。
    A1确认后直接ffill方向，不经过斜率过滤。
    消除F0与斜率的矛盾——只有一个"趋势不明确"判断（偏离度）。
    """
    dir_s = BASE_DIR.copy()

    # F0空仓过滤
    wt = pd.Series(1.0, index=dir_s.index)
    low_dev = RATIO_DEV.abs() < dev_threshold
    wt[low_dev] = 0.0

    # A1四天确认（无斜率过滤，直接ffill）
    mask = np.ones(len(dir_s), dtype=bool)
    for k in range(1, 4):
        mask = mask & (dir_s.values == dir_s.shift(k).values)
    confirmed = dir_s.where(mask, np.nan)
    dir_s = confirmed.ffill()  # 直接ffill，不经过斜率过滤

    # B2过滤
    wrong_value = (dir_s == 'value') & (V_MOM20 <= 0)
    dir_s[wrong_value] = 'growth'

    # E5止损
    gs = (dir_s == 'growth') & (G_DD20 < -0.10)
    vs = (dir_s == 'value') & (V_DD20 < -0.10)
    wt[gs | vs] = wt[gs | vs] * 0.30

    return dir_s, wt


# ============================================================
# X14: 删除F0，斜率不满足时空仓（替代ffill）
# ============================================================
def build_x14():
    """X14: F1+A1(4天)+斜率(不满足→空仓)+B2+E5 — 删除F0

    逻辑: 斜率不满足=趋势不明确=空仓，统一为空仓逻辑。
    原X11-A的斜率不满足时ffill（维持方向），现改为空仓（weight=0）。
    消除F0与斜率的矛盾——只用斜率判断"趋势不明确"。
    """
    dir_s = BASE_DIR.copy()

    # A1四天确认
    mask = np.ones(len(dir_s), dtype=bool)
    for k in range(1, 4):
        mask = mask & (dir_s.values == dir_s.shift(k).values)
    confirmed = dir_s.where(mask, np.nan)

    # 斜率确认: 不满足→空仓（而非ffill维持方向）
    wt = pd.Series(1.0, index=dir_s.index)
    slope_fail = ~SLOPE_OK
    wt[slope_fail] = 0.0  # 斜率不满足时空仓
    dir_s = confirmed.ffill()  # 方向仍ffill，但仓位为0

    # B2过滤
    wrong_value = (dir_s == 'value') & (V_MOM20 <= 0)
    dir_s[wrong_value] = 'growth'

    # E5止损
    gs = (dir_s == 'growth') & (G_DD20 < -0.10)
    vs = (dir_s == 'value') & (V_DD20 < -0.10)
    wt[gs | vs] = wt[gs | vs] * 0.30

    return dir_s, wt


# ============================================================
# X15: F0加滞回机制（保留斜率确认）
# ============================================================
def build_x15(low_thresh=0.003, high_thresh=0.005):
    """X15: F1+F0(滞回)+A1(4天)+斜率+B2+E5

    逻辑: 偏离度<低位→空仓，偏离度>高位→恢复满仓。
    中间区域维持当前状态，避免阈值附近抖动。
    保留斜率确认（但F0滞回减少空仓切换次数）。
    """
    dir_s = BASE_DIR.copy()

    # F0滞回空仓
    wt = pd.Series(1.0, index=dir_s.index)
    abs_dev = RATIO_DEV.abs()
    in_cash = False
    for i in range(len(abs_dev)):
        if pd.isna(abs_dev.iloc[i]):
            continue
        if not in_cash and abs_dev.iloc[i] < low_thresh:
            in_cash = True
        elif in_cash and abs_dev.iloc[i] > high_thresh:
            in_cash = False
        if in_cash:
            wt.iloc[i] = 0.0

    # A1四天确认 + 斜率确认
    mask = np.ones(len(dir_s), dtype=bool)
    for k in range(1, 4):
        mask = mask & (dir_s.values == dir_s.shift(k).values)
    confirmed = dir_s.where(mask, np.nan)
    dir_s = confirmed.where(SLOPE_OK, np.nan).ffill()

    # B2过滤
    wrong_value = (dir_s == 'value') & (V_MOM20 <= 0)
    dir_s[wrong_value] = 'growth'

    # E5止损
    gs = (dir_s == 'growth') & (G_DD20 < -0.10)
    vs = (dir_s == 'value') & (V_DD20 < -0.10)
    wt[gs | vs] = wt[gs | vs] * 0.30

    return dir_s, wt


# ============================================================
# 运行对比
# ============================================================
if __name__ == '__main__':
    print("=" * 78)
    print("  第1-3轮优化: 统一F0与斜率确认的三种方案")
    print("=" * 78)

    results = []

    # 基准
    print("\n  --- 基准 ---")
    r = test_strategy("X11-A", build_x11a, desc="F1+A1+斜率+B2+E5 (无F0)")
    print_result(r); results.append(r)

    r = test_strategy("X12", build_x12, {'dev_threshold': 0.005}, "F1+F0(0.5%)+A1+斜率+B2+E5")
    print_result(r); results.append(r)

    # X13: 删除斜率，只保留F0
    print("\n  --- 第1轮 X13 ---")
    r = test_strategy("X13", build_x13, {'dev_threshold': 0.005},
                      "删除斜率层，F0统一判断趋势(偏离度<0.5%→空仓)")
    print_result(r); results.append(r)

    # X13参数敏感性
    print("\n  X13参数敏感性:")
    for thresh in [0.003, 0.005, 0.008, 0.010]:
        r = test_strategy(f"X13({thresh*100:.1f}%)", build_x13, {'dev_threshold': thresh},
                          f"偏离度<{thresh*100:.1f}%空仓")
        print_result(r)

    # X14: 删除F0，斜率不满足→空仓
    print("\n  --- 第2轮 X14 ---")
    r = test_strategy("X14", build_x14, desc="删除F0，斜率不满足→空仓(替代ffill)")
    print_result(r); results.append(r)

    # X15: F0滞回
    print("\n  --- 第3轮 X15 ---")
    r = test_strategy("X15", build_x15, {'low_thresh': 0.003, 'high_thresh': 0.005},
                      "F0滞回(0.3%→0.5%)+保留斜率")
    print_result(r); results.append(r)

    # X15参数敏感性
    print("\n  X15参数敏感性:")
    for low, high in [(0.002, 0.005), (0.003, 0.008), (0.003, 0.010)]:
        r = test_strategy(f"X15({low*100:.1f}%→{high*100:.1f}%)", build_x15,
                          {'low_thresh': low, 'high_thresh': high},
                          f"滞回{low*100:.1f}%→{high*100:.1f}%")
        print_result(r)

    # 汇总
    print("\n" + "=" * 78)
    print("  汇总对比")
    print("=" * 78)
    print(f"  {'版本':<25} {'年化':>7} {'回撤':>7} {'Sharpe':>7} {'Calmar':>7} {'交易':>5} {'空仓切换':>6}")
    print(f"  {'-'*70}")
    for r in results:
        print(f"  {r['name']:<25} {r['ann']*100:>6.2f}% {r['dd']*100:>6.2f}% "
              f"{r['sharpe']:>7.3f} {r['calmar']:>7.3f} {r['n_trades']:>5} {r['cash_switches']:>6}")
