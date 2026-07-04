"""run_x23_x26.py — 第11-14轮优化
================================================================
X23: 波动率归一化(z-score替代固定阈值)
X24: 滞回带机制(进入0.3/退出0.5)
X25: 四状态机+分档降仓(50%替代0%)
X26: RSRS领先撤退

知识库建议(07_趋势不明确状态处理):
- 偏离度z-score化 = 偏离度/比价20日波动率,阈值自适应
- 滞回带: 进入0.5%/退出0.8%,死区避免抖动
- 四状态机: HOLD/REDUCE/FLAT,F0和斜率共同作为转换条件
- RSRS z<-0.7领先撤退(光大2017)
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
# 偏离度z-score(波动率归一化)
RATIO_DEV_STD20 = RATIO_DEV.rolling(20).std()
RATIO_DEV_Z = RATIO_DEV / RATIO_DEV_STD20

# 简化RSRS: 用20日价格振幅的z-score(代理阻力支撑相对强度)
def calc_simple_rsrs(close, window=18, n_std=120):
    """简化RSRS: 价格振幅的z-score

    原始RSRS: 18日high-low OLS斜率β的z-score
    简化版: (high-low)/close的z-score,捕捉同样的"阻力支撑相对强度"信号
    """
    high = close.rolling(window).max()
    low = close.rolling(window).min()
    range_ratio = (high - low) / close
    rsrs_z = (range_ratio - range_ratio.rolling(n_std).mean()) / range_ratio.rolling(n_std).std()
    return rsrs_z

# G和V的RSRS
G_RSRS_Z = calc_simple_rsrs(G_CLOSE, window=18, n_std=120)
V_RSRS_Z = calc_simple_rsrs(V_CLOSE, window=18, n_std=120)

# 综合RSRS(取G和V的均值)
RSRS_Z = (G_RSRS_Z + V_RSRS_Z) / 2


# ============================================================
# X18基准(用于对比,偏离度+斜率双重确认)
# ============================================================
def build_x18(dev_thresh=0.003, slope_thresh=0.003, n_confirm=4,
              stop_threshold=0.10, stop_weight=0.30):
    """X18: 偏离度+斜率双重确认"""
    dir_s = BASE_DIR.copy()
    mask = np.ones(len(dir_s), dtype=bool)
    for k in range(1, n_confirm):
        mask = mask & (dir_s.values == dir_s.shift(k).values)
    confirmed = dir_s.where(mask, np.nan)
    wt = pd.Series(1.0, index=dir_s.index)
    low_dev = RATIO_DEV.abs() < dev_thresh
    low_slope = MA20_SLOPE.abs() < slope_thresh
    both_weak = low_dev & low_slope
    wt[both_weak] = 0.0
    dir_s = confirmed.ffill()
    wrong_value = (dir_s == 'value') & (V_MOM20 <= 0)
    dir_s[wrong_value] = 'growth'
    gs = (dir_s == 'growth') & (G_DD20 < -stop_threshold)
    vs = (dir_s == 'value') & (V_DD20 < -stop_threshold)
    wt[gs | vs] = wt[gs | vs] * stop_weight
    return dir_s, wt


# ============================================================
# X23: 波动率归一化(z-score替代固定阈值)
# ============================================================
def build_x23(z_thresh=0.5, slope_thresh=0.003, n_confirm=4,
              stop_threshold=0.10, stop_weight=0.30):
    """X23: 偏离度z-score化

    逻辑: 偏离度除以其20日std,变z-score,跨波动率环境稳定
    空仓条件: |z| < z_thresh 且 |slope| < slope_thresh
    """
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
# X24: 滞回带机制
# ============================================================
def build_x24(dev_entry=0.003, dev_exit=0.005, slope_thresh=0.003,
              n_confirm=4, stop_threshold=0.10, stop_weight=0.30):
    """X24: 滞回带

    逻辑:
    - 进入空仓: |dev| < dev_entry(0.3%) 且 |slope| < slope_thresh
    - 退出空仓: |dev| > dev_exit(0.5%) 或 |slope| > 2*slope_thresh
    - 死区内维持原状,避免阈值抖动
    """
    dir_s = BASE_DIR.copy()
    mask = np.ones(len(dir_s), dtype=bool)
    for k in range(1, n_confirm):
        mask = mask & (dir_s.values == dir_s.shift(k).values)
    confirmed = dir_s.where(mask, np.nan)

    # 滞回状态机
    wt = pd.Series(1.0, index=dir_s.index)
    is_flat = False
    for i in range(len(dir_s)):
        if pd.isna(RATIO_DEV.iloc[i]) or pd.isna(MA20_SLOPE.iloc[i]):
            continue
        dev_abs = abs(RATIO_DEV.iloc[i])
        slope_abs = abs(MA20_SLOPE.iloc[i])
        if not is_flat:
            # 满仓状态: 进入空仓条件(更严)
            if dev_abs < dev_entry and slope_abs < slope_thresh:
                is_flat = True
                wt.iloc[i] = 0.0
        else:
            # 空仓状态: 退出条件(更松)
            if dev_abs > dev_exit or slope_abs > 2 * slope_thresh:
                is_flat = False
                wt.iloc[i] = 1.0
            else:
                wt.iloc[i] = 0.0

    dir_s = confirmed.ffill()
    wrong_value = (dir_s == 'value') & (V_MOM20 <= 0)
    dir_s[wrong_value] = 'growth'
    gs = (dir_s == 'growth') & (G_DD20 < -stop_threshold)
    vs = (dir_s == 'value') & (V_DD20 < -stop_threshold)
    wt[gs | vs] = wt[gs | vs] * stop_weight
    return dir_s, wt


# ============================================================
# X25: 四状态机+分档降仓(50%替代0%)
# ============================================================
def build_x25(dev_thresh=0.003, slope_thresh=0.003, n_confirm=4,
              reduce_weight=0.50, stop_threshold=0.10, stop_weight=0.30):
    """X25: 四状态机+分档降仓

    状态:
    - HOLD: 满仓(100%)
    - REDUCE: 降仓50%(趋势不明确)
    - FLAT: 空仓0%(系统性风险,双跌)
    转换:
    - HOLD→REDUCE: |dev|<thresh 且 |slope|<thresh
    - REDUCE→HOLD: |dev|>2*thresh 或 |slope|>2*thresh
    - HOLD/REDUCE→FLAT: G和V都20日跌幅<-5%
    - FLAT→HOLD: |dev|>2*thresh 且 |slope|>2*thresh
    """
    dir_s = BASE_DIR.copy()
    mask = np.ones(len(dir_s), dtype=bool)
    for k in range(1, n_confirm):
        mask = mask & (dir_s.values == dir_s.shift(k).values)
    confirmed = dir_s.where(mask, np.nan)

    # 状态机
    wt = pd.Series(1.0, index=dir_s.index)
    state = 'HOLD'  # 初始状态
    for i in range(len(dir_s)):
        if pd.isna(RATIO_DEV.iloc[i]) or pd.isna(MA20_SLOPE.iloc[i]):
            continue
        dev_abs = abs(RATIO_DEV.iloc[i])
        slope_abs = abs(MA20_SLOPE.iloc[i])
        g_dd = G_DD20.iloc[i] if not pd.isna(G_DD20.iloc[i]) else 0
        v_dd = V_DD20.iloc[i] if not pd.isna(V_DD20.iloc[i]) else 0
        both_down = (g_dd < -0.05) and (v_dd < -0.05)  # 双跌→系统性风险

        if state == 'HOLD':
            if both_down:
                state = 'FLAT'
                wt.iloc[i] = 0.0
            elif dev_abs < dev_thresh and slope_abs < slope_thresh:
                state = 'REDUCE'
                wt.iloc[i] = reduce_weight
            else:
                wt.iloc[i] = 1.0
        elif state == 'REDUCE':
            if both_down:
                state = 'FLAT'
                wt.iloc[i] = 0.0
            elif dev_abs > 2 * dev_thresh or slope_abs > 2 * slope_thresh:
                state = 'HOLD'
                wt.iloc[i] = 1.0
            else:
                wt.iloc[i] = reduce_weight
        elif state == 'FLAT':
            if dev_abs > 2 * dev_thresh and slope_abs > 2 * slope_thresh:
                state = 'HOLD'
                wt.iloc[i] = 1.0
            else:
                wt.iloc[i] = 0.0

    dir_s = confirmed.ffill()
    wrong_value = (dir_s == 'value') & (V_MOM20 <= 0)
    dir_s[wrong_value] = 'growth'
    gs = (dir_s == 'growth') & (G_DD20 < -stop_threshold)
    vs = (dir_s == 'value') & (V_DD20 < -stop_threshold)
    wt[gs | vs] = wt[gs | vs] * stop_weight
    return dir_s, wt


# ============================================================
# X26: RSRS领先撤退
# ============================================================
def build_x26(rsrs_thresh=-0.7, dev_thresh=0.003, slope_thresh=0.003,
              n_confirm=4, stop_threshold=0.10, stop_weight=0.30,
              rsrs_reduce=0.30):
    """X26: X18+RSRS领先撤退

    新增: RSRS z<rsrs_thresh时领先降仓到rsrs_reduce
    E5(20日跌幅>10%)保留作滞后兜底
    """
    dir_s = BASE_DIR.copy()
    mask = np.ones(len(dir_s), dtype=bool)
    for k in range(1, n_confirm):
        mask = mask & (dir_s.values == dir_s.shift(k).values)
    confirmed = dir_s.where(mask, np.nan)

    wt = pd.Series(1.0, index=dir_s.index)
    # F0+斜率双重确认空仓
    low_dev = RATIO_DEV.abs() < dev_thresh
    low_slope = MA20_SLOPE.abs() < slope_thresh
    both_weak = low_dev & low_slope
    wt[both_weak] = 0.0

    # RSRS领先降仓(在F0基础上)
    rsrs_warn = RSRS_Z < rsrs_thresh
    wt[rsrs_warn & ~both_weak] = rsrs_reduce  # 只在非空仓时降仓

    dir_s = confirmed.ffill()
    wrong_value = (dir_s == 'value') & (V_MOM20 <= 0)
    dir_s[wrong_value] = 'growth'
    # E5滞后止损(兜底)
    gs = (dir_s == 'growth') & (G_DD20 < -stop_threshold)
    vs = (dir_s == 'value') & (V_DD20 < -stop_threshold)
    wt[gs | vs] = wt[gs | vs] * stop_weight
    return dir_s, wt


# ============================================================
# 运行测试
# ============================================================
if __name__ == '__main__':
    print("=" * 78)
    print("  X23-X26: 第11-14轮优化(知识库建议落地)")
    print("=" * 78)

    results = []

    # 基准
    print("\n  [基准1] X11-A:")
    info = test_strategy("X11-A", build_x11a, desc="F1+A1+斜率+B2+E5")
    print_result(info)
    results.append(info)

    print("\n  [基准2] X12(0.5%):")
    info = test_strategy("X12", build_x12, {'dev_threshold': 0.005}, "F1+F0(0.5%)+A1+斜率+B2+E5")
    print_result(info)
    results.append(info)

    print("\n  [基准3] X18(双重确认):")
    info = test_strategy("X18", build_x18, {'dev_thresh': 0.003, 'slope_thresh': 0.003},
                         "偏离度+斜率双重确认")
    print_result(info)
    results.append(info)

    # X23: 波动率归一化
    print("\n" + "=" * 78)
    print("  第11轮 X23: 波动率归一化(z-score)")
    print("=" * 78)
    print("  逻辑: 偏离度/20日std=z-score,跨波动率环境稳定")
    print("  扫描不同z阈值:")
    for z in [0.3, 0.5, 0.7, 1.0, 1.5]:
        info = test_strategy(f"X23(z={z})", build_x23,
                             {'z_thresh': z}, f"z-score阈值{z}")
        print_result(info)
        results.append(info)

    # X24: 滞回带
    print("\n" + "=" * 78)
    print("  第12轮 X24: 滞回带机制")
    print("=" * 78)
    print("  逻辑: 进入0.3/退出0.5,死区避免抖动")
    for entry, exit in [(0.003, 0.005), (0.002, 0.005), (0.003, 0.008), (0.002, 0.008)]:
        info = test_strategy(f"X24(e={entry},x={exit})", build_x24,
                             {'dev_entry': entry, 'dev_exit': exit},
                             f"进入{entry*100:.1f}%/退出{exit*100:.1f}%")
        print_result(info)
        results.append(info)

    # X25: 状态机+分档降仓
    print("\n" + "=" * 78)
    print("  第13轮 X25: 四状态机+分档降仓")
    print("=" * 78)
    print("  逻辑: HOLD/REDUCE/FLAT,REDUCE用50%替代0%")
    for rw in [0.30, 0.50, 0.70]:
        info = test_strategy(f"X25(reduce={rw})", build_x25,
                             {'reduce_weight': rw},
                             f"REDUCE仓位{rw*100:.0f}%")
        print_result(info)
        results.append(info)

    # X26: RSRS领先撤退
    print("\n" + "=" * 78)
    print("  第14轮 X26: RSRS领先撤退")
    print("=" * 78)
    print("  逻辑: RSRS z<-0.7领先降仓30%,E5保留兜底")
    for thresh in [-0.5, -0.7, -1.0, -1.5]:
        info = test_strategy(f"X26(rsrs={thresh})", build_x26,
                             {'rsrs_thresh': thresh},
                             f"RSRS阈值{thresh}")
        print_result(info)
        results.append(info)

    # 汇总
    print("\n" + "=" * 78)
    print("  X23-X26 汇总")
    print("=" * 78)
    print(f"  {'版本':<25} {'年化':>8} {'回撤':>8} {'Sharpe':>8} {'Calmar':>8} {'交易':>6}")
    print(f"  {'-'*70}")
    for r in results:
        print(f"  {r['name']:<25} {r['ann']*100:>7.2f}% {r['dd']*100:>7.2f}% "
              f"{r['sharpe']:>8.3f} {r['calmar']:>8.3f} {r['n_trades']:>6}")

    print("\n  完成!")
