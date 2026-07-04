"""run_x16_x18.py — 第4-6轮优化: 基于X14进一步降低交易次数
================================================================
X14基准: 删除F0，斜率不满足→空仓 (年化40.80%/回撤-28.69%/交易390次)

X16: X14 + 斜率空仓加确认机制(连续N天)
X17: X14 + 斜率空仓加滞回(斜率<低位→空仓, >高位→恢复)
X18: X14 + 偏离度+斜率双重确认(两者都弱才空仓)
================================================================
"""
import sys
sys.path.insert(0, r'c:\XuanJLH\Qbot\custom\research\成长价值轮动\X12_偏离度空仓版')
from optimize_runner import *
import numpy as np
import pandas as pd

# ============================================================
# X14基准(复用)
# ============================================================
def build_x14():
    dir_s = BASE_DIR.copy()
    mask = np.ones(len(dir_s), dtype=bool)
    for k in range(1, 4):
        mask = mask & (dir_s.values == dir_s.shift(k).values)
    confirmed = dir_s.where(mask, np.nan)
    wt = pd.Series(1.0, index=dir_s.index)
    slope_fail = ~SLOPE_OK
    wt[slope_fail] = 0.0
    dir_s = confirmed.ffill()
    wrong_value = (dir_s == 'value') & (V_MOM20 <= 0)
    dir_s[wrong_value] = 'growth'
    gs = (dir_s == 'growth') & (G_DD20 < -0.10)
    vs = (dir_s == 'value') & (V_DD20 < -0.10)
    wt[gs | vs] = wt[gs | vs] * 0.30
    return dir_s, wt


# ============================================================
# X16: X14 + 斜率空仓加确认机制(连续N天)
# ============================================================
def build_x16(n_confirm=3):
    """X16: X14 + 斜率空仓加确认

    逻辑: 连续N天斜率不满足才空仓，连续N天斜率满足才恢复满仓。
    避免斜率在阈值附近抖动导致的频繁切换。
    类似A1方向确认，但应用于仓位切换。
    """
    dir_s = BASE_DIR.copy()

    # A1四天确认
    mask = np.ones(len(dir_s), dtype=bool)
    for k in range(1, 4):
        mask = mask & (dir_s.values == dir_s.shift(k).values)
    confirmed = dir_s.where(mask, np.nan)

    # 斜率空仓加确认: 连续N天不满足才空仓
    wt = pd.Series(1.0, index=dir_s.index)
    slope_fail = (~SLOPE_OK).astype(int)
    # 连续N天斜率不满足
    consec_fail = slope_fail.rolling(n_confirm).sum()
    should_cash = consec_fail >= n_confirm

    # 连续N天斜率满足才恢复
    slope_pass = SLOPE_OK.astype(int)
    consec_pass = slope_pass.rolling(n_confirm).sum()
    should_recover = consec_pass >= n_confirm

    # 状态机: 空仓→满仓需要确认，满仓→空仓也需要确认
    in_cash = False
    for i in range(len(wt)):
        if pd.isna(should_cash.iloc[i]):
            continue
        if not in_cash and should_cash.iloc[i]:
            in_cash = True
        elif in_cash and should_recover.iloc[i]:
            in_cash = False
        if in_cash:
            wt.iloc[i] = 0.0

    dir_s = confirmed.ffill()
    wrong_value = (dir_s == 'value') & (V_MOM20 <= 0)
    dir_s[wrong_value] = 'growth'
    gs = (dir_s == 'growth') & (G_DD20 < -0.10)
    vs = (dir_s == 'value') & (V_DD20 < -0.10)
    wt[gs | vs] = wt[gs | vs] * 0.30
    return dir_s, wt


# ============================================================
# X17: X14 + 斜率空仓加滞回
# ============================================================
def build_x17(low_slope=0.0015, high_slope=0.004):
    """X17: X14 + 斜率滞回

    逻辑: 斜率绝对值<低位→空仓，>高位→恢复满仓。
    中间区域维持当前状态，避免抖动。
    """
    dir_s = BASE_DIR.copy()
    mask = np.ones(len(dir_s), dtype=bool)
    for k in range(1, 4):
        mask = mask & (dir_s.values == dir_s.shift(k).values)
    confirmed = dir_s.where(mask, np.nan)

    wt = pd.Series(1.0, index=dir_s.index)
    abs_slope = MA20_SLOPE.abs()
    in_cash = False
    for i in range(len(abs_slope)):
        if pd.isna(abs_slope.iloc[i]):
            continue
        if not in_cash and abs_slope.iloc[i] < low_slope:
            in_cash = True
        elif in_cash and abs_slope.iloc[i] > high_slope:
            in_cash = False
        if in_cash:
            wt.iloc[i] = 0.0

    dir_s = confirmed.ffill()
    wrong_value = (dir_s == 'value') & (V_MOM20 <= 0)
    dir_s[wrong_value] = 'growth'
    gs = (dir_s == 'growth') & (G_DD20 < -0.10)
    vs = (dir_s == 'value') & (V_DD20 < -0.10)
    wt[gs | vs] = wt[gs | vs] * 0.30
    return dir_s, wt


# ============================================================
# X18: X14 + 偏离度+斜率双重确认(两者都弱才空仓)
# ============================================================
def build_x18(dev_thresh=0.005, slope_thresh=0.003):
    """X18: 偏离度+斜率双重确认

    逻辑: 偏离度<阈值 且 斜率<阈值→空仓（两者都弱才空仓）。
    比X14(只看斜率)更保守，比X12(只看偏离度)逻辑更统一。
    两个指标交叉确认，减少假信号。
    """
    dir_s = BASE_DIR.copy()
    mask = np.ones(len(dir_s), dtype=bool)
    for k in range(1, 4):
        mask = mask & (dir_s.values == dir_s.shift(k).values)
    confirmed = dir_s.where(mask, np.nan)

    # 双重确认: 偏离度小 且 斜率小 → 空仓
    wt = pd.Series(1.0, index=dir_s.index)
    low_dev = RATIO_DEV.abs() < dev_thresh
    low_slope = MA20_SLOPE.abs() < slope_thresh
    both_weak = low_dev & low_slope
    wt[both_weak] = 0.0

    dir_s = confirmed.ffill()
    wrong_value = (dir_s == 'value') & (V_MOM20 <= 0)
    dir_s[wrong_value] = 'growth'
    gs = (dir_s == 'growth') & (G_DD20 < -0.10)
    vs = (dir_s == 'value') & (V_DD20 < -0.10)
    wt[gs | vs] = wt[gs | vs] * 0.30
    return dir_s, wt


# ============================================================
# 运行
# ============================================================
if __name__ == '__main__':
    print("=" * 78)
    print("  第4-6轮优化: 基于X14进一步降低交易次数")
    print("=" * 78)

    results = []

    # X14基准
    print("\n  --- X14基准 ---")
    r = test_strategy("X14", build_x14, desc="斜率不满足→空仓(基准)")
    print_result(r); results.append(r)

    # X16: 斜率确认机制
    print("\n  --- 第4轮 X16: 斜率空仓加确认 ---")
    for n in [2, 3, 5]:
        r = test_strategy(f"X16({n}天确认)", build_x16, {'n_confirm': n},
                          f"连续{n}天斜率弱→空仓, 连续{n}天强→恢复")
        print_result(r); results.append(r)

    # X17: 斜率滞回
    print("\n  --- 第5轮 X17: 斜率滞回 ---")
    for low, high in [(0.0015, 0.004), (0.001, 0.003), (0.002, 0.005)]:
        r = test_strategy(f"X17({low*100:.2f}%→{high*100:.2f}%)", build_x17,
                          {'low_slope': low, 'high_slope': high},
                          f"斜率<{low*100:.2f}%→空仓, >{high*100:.2f}%→恢复")
        print_result(r); results.append(r)

    # X18: 双重确认
    print("\n  --- 第6轮 X18: 偏离度+斜率双重确认 ---")
    for dev, slp in [(0.005, 0.003), (0.003, 0.003), (0.005, 0.005)]:
        r = test_strategy(f"X18(dev{dev*100:.1f}%+slp{slp*100:.1f}%)", build_x18,
                          {'dev_thresh': dev, 'slope_thresh': slp},
                          f"偏离度<{dev*100:.1f}%且斜率<{slp*100:.1f}%→空仓")
        print_result(r); results.append(r)

    # 汇总
    print("\n" + "=" * 78)
    print("  汇总对比")
    print("=" * 78)
    print(f"  {'版本':<30} {'年化':>7} {'回撤':>7} {'Sharpe':>7} {'Calmar':>7} {'交易':>5} {'空仓切':>5}")
    print(f"  {'-'*72}")
    for r in results:
        print(f"  {r['name']:<30} {r['ann']*100:>6.2f}% {r['dd']*100:>6.2f}% "
              f"{r['sharpe']:>7.3f} {r['calmar']:>7.3f} {r['n_trades']:>5} {r['cash_switches']:>5}")
