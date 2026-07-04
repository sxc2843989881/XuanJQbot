"""run_x27_x30.py — 第15-18轮优化
================================================================
X27: KST多周期动量替代F1比价MA20
X28: 空仓恢复三重确认(基于X23 z=1.5)
X29: 新增12月绝对动量"是否持有"层
X30: 综合最优组合

新基准(来自X23): X23(z=1.5) 年化41.10% 回撤-26.66% Calmar1.542

知识库建议:
- KST=ROC(10)×1+ROC(15)×2+ROC(20)×3+ROC(30)×4,EMA9信号线(Martin Pring 1992)
- 空仓恢复: 偏离度z>+0.5 + 连续3日 + 斜率>0.3% 三者AND
- 绝对动量: 252日(12月) sweet spot, Moskowitz 2012
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

# ============================================================
# 新因子计算
# ============================================================
# 偏离度z-score(波动率归一化) - X23已验证有效
RATIO_DEV_STD20 = RATIO_DEV.rolling(20).std()
RATIO_DEV_Z = RATIO_DEV / RATIO_DEV_STD20

# KST多周期动量(Martin Pring 1992)
# KST = ROC(10)×1 + ROC(15)×2 + ROC(20)×3 + ROC(30)×4, 信号线EMA9
ROC_10 = RATIO.pct_change(10)
ROC_15 = RATIO.pct_change(15)
ROC_20 = RATIO.pct_change(20)
ROC_30 = RATIO.pct_change(30)
KST_RAW = ROC_10 * 1 + ROC_15 * 2 + ROC_20 * 3 + ROC_30 * 4
KST = KST_RAW.rolling(9).mean()  # EMA9信号线简化为SMA9
KST_DIR = (KST > 0).map({True: 'growth', False: 'value'})

# 绝对动量(12月=252日)
G_ABS_MOM_252 = G_CLOSE.pct_change(252)
V_ABS_MOM_252 = V_CLOSE.pct_change(252)
# 多窗口绝对动量(3/6/12月)
G_ABS_MOM_63 = G_CLOSE.pct_change(63)   # 3月
G_ABS_MOM_126 = G_CLOSE.pct_change(126) # 6月
V_ABS_MOM_63 = V_CLOSE.pct_change(63)
V_ABS_MOM_126 = V_CLOSE.pct_change(126)


# ============================================================
# X23基准(新最优,用于对比)
# ============================================================
def build_x23(z_thresh=1.5, slope_thresh=0.003, n_confirm=4,
              stop_threshold=0.10, stop_weight=0.30):
    """X23: 偏离度z-score化(新基准)"""
    dir_s = BASE_DIR.copy()
    mask = np.ones(len(dir_s), dtype=bool)
    for k in range(1, n_confirm):
        mask = mask & (dir_s.values == dir_s.shift(k).values)
    confirmed = dir_s.where(mask, np.nan)
    wt = pd.Series(1.0, index=dir_s.index)
    low_dev_z = RATIO_DEV_Z.abs() < z_thresh
    low_slope = MA20_SLOPE.abs() < slope_thresh
    both_weak = low_dev_z & low_slope
    wt[both_weak] = 0.0
    dir_s = confirmed.ffill()
    wrong_value = (dir_s == 'value') & (V_MOM20 <= 0)
    dir_s[wrong_value] = 'growth'
    gs = (dir_s == 'growth') & (G_DD20 < -stop_threshold)
    vs = (dir_s == 'value') & (V_DD20 < -stop_threshold)
    wt[gs | vs] = wt[gs | vs] * stop_weight
    return dir_s, wt


# ============================================================
# X27: KST多周期动量替代F1比价MA20
# ============================================================
def build_x27(z_thresh=1.5, slope_thresh=0.003, n_confirm=4,
              stop_threshold=0.10, stop_weight=0.30,
              use_kst_dir=True):
    """X27: F1升级为KST多周期动量

    逻辑: 用KST(RC10×1+RC15×2+RC20×3+RC30×4)替代比价MA20做方向判断
    KST噪音降62%,趋势确认提前2.1周期(Martin Pring 1992)
    """
    if use_kst_dir:
        base_dir = KST_DIR.copy()
    else:
        base_dir = BASE_DIR.copy()

    dir_s = base_dir.copy()
    mask = np.ones(len(dir_s), dtype=bool)
    for k in range(1, n_confirm):
        mask = mask & (dir_s.values == dir_s.shift(k).values)
    confirmed = dir_s.where(mask, np.nan)

    wt = pd.Series(1.0, index=dir_s.index)
    low_dev_z = RATIO_DEV_Z.abs() < z_thresh
    low_slope = MA20_SLOPE.abs() < slope_thresh
    both_weak = low_dev_z & low_slope
    wt[both_weak] = 0.0

    dir_s = confirmed.ffill()
    wrong_value = (dir_s == 'value') & (V_MOM20 <= 0)
    dir_s[wrong_value] = 'growth'
    gs = (dir_s == 'growth') & (G_DD20 < -stop_threshold)
    vs = (dir_s == 'value') & (V_DD20 < -stop_threshold)
    wt[gs | vs] = wt[gs | vs] * stop_weight
    return dir_s, wt


# ============================================================
# X28: 空仓恢复三重确认
# ============================================================
def build_x28(z_entry=1.5, z_exit=0.5, slope_thresh=0.003,
              n_confirm=3, n_recover=3,
              stop_threshold=0.10, stop_weight=0.30):
    """X28: X23+空仓恢复三重确认

    逻辑:
    - 进入空仓: |z|<z_entry 且 |slope|<slope_thresh
    - 恢复满仓: z>z_exit 且 连续n_recover天 且 |slope|>2*slope_thresh (三者AND)
    - 死区内维持空仓状态,避免假恢复
    """
    dir_s = BASE_DIR.copy()
    mask = np.ones(len(dir_s), dtype=bool)
    for k in range(1, n_confirm):
        mask = mask & (dir_s.values == dir_s.shift(k).values)
    confirmed = dir_s.where(mask, np.nan)

    # 滞回状态机
    wt = pd.Series(1.0, index=dir_s.index)
    is_flat = False
    recover_count = 0

    for i in range(len(dir_s)):
        if pd.isna(RATIO_DEV_Z.iloc[i]) or pd.isna(MA20_SLOPE.iloc[i]):
            continue
        z_val = RATIO_DEV_Z.iloc[i]
        z_abs = abs(z_val)
        slope_abs = abs(MA20_SLOPE.iloc[i])

        if not is_flat:
            # 满仓状态
            if z_abs < z_entry and slope_abs < slope_thresh:
                is_flat = True
                wt.iloc[i] = 0.0
                recover_count = 0
            else:
                wt.iloc[i] = 1.0
        else:
            # 空仓状态: 需三重确认才恢复
            # 条件1: z > z_exit (趋势明确偏离)
            # 条件2: 连续n_recover天满足
            # 条件3: |slope| > 2*slope_thresh (斜率确认)
            recover_cond = (z_abs > z_exit) and (slope_abs > 2 * slope_thresh)
            if recover_cond:
                recover_count += 1
                if recover_count >= n_recover:
                    is_flat = False
                    wt.iloc[i] = 1.0
                    recover_count = 0
                else:
                    wt.iloc[i] = 0.0
            else:
                recover_count = 0
                wt.iloc[i] = 0.0

    dir_s = confirmed.ffill()
    wrong_value = (dir_s == 'value') & (V_MOM20 <= 0)
    dir_s[wrong_value] = 'growth'
    gs = (dir_s == 'growth') & (G_DD20 < -stop_threshold)
    vs = (dir_s == 'value') & (V_DD20 < -stop_threshold)
    wt[gs | vs] = wt[gs | vs] * stop_weight
    return dir_s, wt


# ============================================================
# X29: 新增12月绝对动量"是否持有"层
# ============================================================
def build_x29(z_thresh=1.5, slope_thresh=0.003, n_confirm=4,
              abs_mom_thresh=0.0, abs_reduce=0.50,
              stop_threshold=0.10, stop_weight=0.30,
              use_multi_window=False):
    """X29: X23+12月绝对动量层

    逻辑:
    - 单窗口: AbsMom_252<abs_mom_thresh时降仓到abs_reduce
    - 多窗口: 3/6/12月投票,<0票数>=2时降仓
    补充缺失的"是否持有"层(知识库建议)
    """
    dir_s = BASE_DIR.copy()
    mask = np.ones(len(dir_s), dtype=bool)
    for k in range(1, n_confirm):
        mask = mask & (dir_s.values == dir_s.shift(k).values)
    confirmed = dir_s.where(mask, np.nan)

    wt = pd.Series(1.0, index=dir_s.index)
    # F0+斜率双重确认空仓
    low_dev_z = RATIO_DEV_Z.abs() < z_thresh
    low_slope = MA20_SLOPE.abs() < slope_thresh
    both_weak = low_dev_z & low_slope
    wt[both_weak] = 0.0

    # 绝对动量"是否持有"层
    if use_multi_window:
        # 多窗口投票: 3/6/12月,<0票数>=2时降仓
        g_votes = (G_ABS_MOM_63 < 0).astype(int) + (G_ABS_MOM_126 < 0).astype(int) + (G_ABS_MOM_252 < 0).astype(int)
        v_votes = (V_ABS_MOM_63 < 0).astype(int) + (V_ABS_MOM_126 < 0).astype(int) + (V_ABS_MOM_252 < 0).astype(int)
        # 当前方向的绝对动量
        g_weak = (dir_s == 'growth') & (g_votes >= 2)
        v_weak = (dir_s == 'value') & (v_votes >= 2)
        wt[g_weak | v_weak] = wt[g_weak | v_weak] * abs_reduce
    else:
        # 单窗口: 12月绝对动量
        g_weak = (dir_s == 'growth') & (G_ABS_MOM_252 < abs_mom_thresh)
        v_weak = (dir_s == 'value') & (V_ABS_MOM_252 < abs_mom_thresh)
        wt[g_weak | v_weak] = wt[g_weak | v_weak] * abs_reduce

    dir_s = confirmed.ffill()
    wrong_value = (dir_s == 'value') & (V_MOM20 <= 0)
    dir_s[wrong_value] = 'growth'
    gs = (dir_s == 'growth') & (G_DD20 < -stop_threshold)
    vs = (dir_s == 'value') & (V_DD20 < -stop_threshold)
    wt[gs | vs] = wt[gs | vs] * stop_weight
    return dir_s, wt


# ============================================================
# X30: 综合最优组合
# ============================================================
def build_x30(z_thresh=1.5, slope_thresh=0.003, n_confirm=4,
              use_kst=False, use_recover_confirm=False,
              use_abs_mom=False, abs_reduce=0.50,
              z_exit=0.5, n_recover=3,
              stop_threshold=0.10, stop_weight=0.30):
    """X30: 综合最优组合

    可选开关:
    - use_kst: 是否用KST替代F1
    - use_recover_confirm: 是否启用空仓恢复三重确认
    - use_abs_mom: 是否启用12月绝对动量层
    """
    base_dir = KST_DIR.copy() if use_kst else BASE_DIR.copy()
    dir_s = base_dir.copy()
    mask = np.ones(len(dir_s), dtype=bool)
    for k in range(1, n_confirm):
        mask = mask & (dir_s.values == dir_s.shift(k).values)
    confirmed = dir_s.where(mask, np.nan)

    if use_recover_confirm:
        # 带恢复确认的滞回状态机
        wt = pd.Series(1.0, index=dir_s.index)
        is_flat = False
        recover_count = 0
        for i in range(len(dir_s)):
            if pd.isna(RATIO_DEV_Z.iloc[i]) or pd.isna(MA20_SLOPE.iloc[i]):
                continue
            z_abs = abs(RATIO_DEV_Z.iloc[i])
            slope_abs = abs(MA20_SLOPE.iloc[i])
            if not is_flat:
                if z_abs < z_thresh and slope_abs < slope_thresh:
                    is_flat = True
                    wt.iloc[i] = 0.0
                    recover_count = 0
                else:
                    wt.iloc[i] = 1.0
            else:
                recover_cond = (z_abs > z_exit) and (slope_abs > 2 * slope_thresh)
                if recover_cond:
                    recover_count += 1
                    if recover_count >= n_recover:
                        is_flat = False
                        wt.iloc[i] = 1.0
                        recover_count = 0
                    else:
                        wt.iloc[i] = 0.0
                else:
                    recover_count = 0
                    wt.iloc[i] = 0.0
    else:
        # 简单双重确认
        wt = pd.Series(1.0, index=dir_s.index)
        low_dev_z = RATIO_DEV_Z.abs() < z_thresh
        low_slope = MA20_SLOPE.abs() < slope_thresh
        both_weak = low_dev_z & low_slope
        wt[both_weak] = 0.0

    dir_s = confirmed.ffill()
    wrong_value = (dir_s == 'value') & (V_MOM20 <= 0)
    dir_s[wrong_value] = 'growth'

    # 绝对动量层(可选)
    if use_abs_mom:
        g_weak = (dir_s == 'growth') & (G_ABS_MOM_252 < 0)
        v_weak = (dir_s == 'value') & (V_ABS_MOM_252 < 0)
        wt[g_weak | v_weak] = wt[g_weak | v_weak] * abs_reduce

    # E5止损
    gs = (dir_s == 'growth') & (G_DD20 < -stop_threshold)
    vs = (dir_s == 'value') & (V_DD20 < -stop_threshold)
    wt[gs | vs] = wt[gs | vs] * stop_weight
    return dir_s, wt


# ============================================================
# 运行测试
# ============================================================
if __name__ == '__main__':
    print("=" * 78)
    print("  X27-X30: 第15-18轮优化(KST/恢复确认/绝对动量/综合)")
    print("=" * 78)

    results = []

    # 基准
    print("\n  [基准1] X18(双重确认):")
    from run_x23_x26 import build_x18
    info = test_strategy("X18", build_x18, {'dev_thresh': 0.003, 'slope_thresh': 0.003},
                         "偏离度+斜率双重确认")
    print_result(info)
    results.append(info)

    print("\n  [基准2] X23(z=1.5):")
    info = test_strategy("X23", build_x23, {'z_thresh': 1.5},
                         "z-score阈值1.5(新最优)")
    print_result(info)
    results.append(info)

    # X27: KST替代F1
    print("\n" + "=" * 78)
    print("  第15轮 X27: KST多周期动量替代F1")
    print("=" * 78)
    print("  逻辑: KST=ROC10×1+ROC15×2+ROC20×3+ROC30×4,EMA9信号线")
    print("  扫描不同z阈值(配KST):")
    for z in [0.5, 1.0, 1.5, 2.0]:
        info = test_strategy(f"X27(KST,z={z})", build_x27,
                             {'z_thresh': z, 'use_kst_dir': True},
                             f"KST方向+z-score{z}")
        print_result(info)
        results.append(info)
    # 对比: KST vs 原F1(同z=1.5)
    info = test_strategy("X27(F1,z=1.5)", build_x27,
                         {'z_thresh': 1.5, 'use_kst_dir': False},
                         "原F1方向+z-score1.5(对比)")
    print_result(info)
    results.append(info)

    # X28: 空仓恢复三重确认
    print("\n" + "=" * 78)
    print("  第16轮 X28: 空仓恢复三重确认")
    print("=" * 78)
    print("  逻辑: 恢复需z>z_exit+连续n天+斜率确认,三者AND")
    for z_entry, z_exit, n_rec in [(1.5, 0.5, 3), (1.5, 1.0, 3),
                                    (2.0, 0.5, 3), (1.5, 0.5, 5)]:
        info = test_strategy(f"X28(e={z_entry},x={z_exit},n={n_rec})", build_x28,
                             {'z_entry': z_entry, 'z_exit': z_exit, 'n_recover': n_rec},
                             f"进入{z_entry}/退出{z_exit}/{n_rec}天确认")
        print_result(info)
        results.append(info)

    # X29: 12月绝对动量
    print("\n" + "=" * 78)
    print("  第17轮 X29: 新增12月绝对动量层")
    print("=" * 78)
    print("  逻辑: AbsMom_252<0降仓50%,补充'是否持有'层")
    # 单窗口
    for reduce in [0.30, 0.50, 0.70]:
        info = test_strategy(f"X29(单窗口,reduce={reduce})", build_x29,
                             {'abs_reduce': reduce, 'use_multi_window': False},
                             f"12月绝对动量<0降仓{reduce*100:.0f}%")
        print_result(info)
        results.append(info)
    # 多窗口投票
    info = test_strategy("X29(多窗口投票)", build_x29,
                         {'use_multi_window': True, 'abs_reduce': 0.50},
                         "3/6/12月投票,>=2票<0降仓50%")
    print_result(info)
    results.append(info)

    # X30: 综合最优组合
    print("\n" + "=" * 78)
    print("  第18轮 X30: 综合最优组合")
    print("=" * 78)
    print("  逻辑: 测试各机制组合,找最优配置")
    combos = [
        {'use_kst': False, 'use_recover_confirm': False, 'use_abs_mom': False, 'desc': 'X23基准(z=1.5)'},
        {'use_kst': True, 'use_recover_confirm': False, 'use_abs_mom': False, 'desc': 'X23+KST'},
        {'use_kst': False, 'use_recover_confirm': True, 'use_abs_mom': False, 'desc': 'X23+恢复确认'},
        {'use_kst': False, 'use_recover_confirm': False, 'use_abs_mom': True, 'desc': 'X23+绝对动量'},
        {'use_kst': True, 'use_recover_confirm': True, 'use_abs_mom': False, 'desc': 'X23+KST+恢复确认'},
        {'use_kst': True, 'use_recover_confirm': False, 'use_abs_mom': True, 'desc': 'X23+KST+绝对动量'},
        {'use_kst': False, 'use_recover_confirm': True, 'use_abs_mom': True, 'desc': 'X23+恢复确认+绝对动量'},
        {'use_kst': True, 'use_recover_confirm': True, 'use_abs_mom': True, 'desc': 'X23+全部'},
    ]
    for c in combos:
        desc = c.pop('desc')
        info = test_strategy(f"X30({desc})", build_x30,
                             {**c, 'z_thresh': 1.5},
                             desc)
        print_result(info)
        results.append(info)

    # 汇总
    print("\n" + "=" * 78)
    print("  X27-X30 汇总")
    print("=" * 78)
    print(f"  {'版本':<35} {'年化':>8} {'回撤':>8} {'Sharpe':>8} {'Calmar':>8} {'交易':>6}")
    print(f"  {'-'*80}")
    for r in results:
        print(f"  {r['name']:<35} {r['ann']*100:>7.2f}% {r['dd']*100:>7.2f}% "
              f"{r['sharpe']:>8.3f} {r['calmar']:>8.3f} {r['n_trades']:>6}")

    print("\n  完成!")
