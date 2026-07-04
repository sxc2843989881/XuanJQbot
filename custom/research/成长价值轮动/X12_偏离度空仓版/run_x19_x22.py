"""run_x19_x22.py — 第7-10轮优化: 基于X18最优版继续微调
================================================================
X18基准: 偏离度<0.3%且斜率<0.3%→空仓 (年化41.81%/回撤-30.36%/交易303次)

X19: X18 + E5止损参数优化(阈值/降仓比例)
X20: X18 + A1确认天数优化(3/4/5天)
X21: X18 + B2优化(value弱且growth也跌→空仓而非改growth)
X22: X18 + 综合最优参数组合
================================================================
"""
import sys
sys.path.insert(0, r'c:\XuanJLH\Qbot\custom\research\成长价值轮动\X12_偏离度空仓版')
from optimize_runner import *
import numpy as np
import pandas as pd

# ============================================================
# X18基准(复用)
# ============================================================
def build_x18(dev_thresh=0.003, slope_thresh=0.003, n_confirm=4,
              stop_threshold=0.10, stop_weight=0.30):
    """X18: 偏离度+斜率双重确认空仓"""
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
# X19: E5止损参数优化
# ============================================================
def build_x19(stop_threshold=0.10, stop_weight=0.30):
    """X19: X18 + E5参数优化"""
    return build_x18(stop_threshold=stop_threshold, stop_weight=stop_weight)


# ============================================================
# X20: A1确认天数优化
# ============================================================
def build_x20(n_confirm=4):
    """X20: X18 + A1确认天数优化"""
    return build_x18(n_confirm=n_confirm)


# ============================================================
# X21: B2优化 — value弱且growth也跌→空仓
# ============================================================
def build_x21(dev_thresh=0.003, slope_thresh=0.003, n_confirm=4,
              stop_threshold=0.10, stop_weight=0.30):
    """X21: X18 + B2优化

    原B2: value方向且v_mom20<=0 → 强制改growth
    新B2: value方向且v_mom20<=0 → 检查growth是否也在跌
      - growth也在跌(g_mom20<=0) → 空仓等待
      - growth没跌(g_mom20>0) → 改growth

    逻辑: 价值弱时如果成长也弱，不应强制持有成长，应空仓等待。
    """
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

    # 新B2: value弱时检查growth
    g_mom20 = G_CLOSE.pct_change(20)
    wrong_value = (dir_s == 'value') & (V_MOM20 <= 0)
    growth_also_weak = wrong_value & (g_mom20 <= 0)  # growth也在跌
    growth_ok = wrong_value & (g_mom20 > 0)           # growth没跌

    dir_s[growth_ok] = 'growth'           # growth没跌→改growth
    wt[growth_also_weak] = 0.0            # growth也跌→空仓

    gs = (dir_s == 'growth') & (G_DD20 < -stop_threshold)
    vs = (dir_s == 'value') & (V_DD20 < -stop_threshold)
    wt[gs | vs] = wt[gs | vs] * stop_weight
    return dir_s, wt


# ============================================================
# X22: 综合最优参数组合
# ============================================================
def build_x22(dev_thresh=0.003, slope_thresh=0.003, n_confirm=4,
              stop_threshold=0.10, stop_weight=0.30, use_smart_b2=True):
    """X22: X18 + 最优E5参数 + 智能B2"""
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

    if use_smart_b2:
        g_mom20 = G_CLOSE.pct_change(20)
        wrong_value = (dir_s == 'value') & (V_MOM20 <= 0)
        growth_also_weak = wrong_value & (g_mom20 <= 0)
        growth_ok = wrong_value & (g_mom20 > 0)
        dir_s[growth_ok] = 'growth'
        wt[growth_also_weak] = 0.0
    else:
        wrong_value = (dir_s == 'value') & (V_MOM20 <= 0)
        dir_s[wrong_value] = 'growth'

    gs = (dir_s == 'growth') & (G_DD20 < -stop_threshold)
    vs = (dir_s == 'value') & (V_DD20 < -stop_threshold)
    wt[gs | vs] = wt[gs | vs] * stop_weight
    return dir_s, wt


# ============================================================
# 运行
# ============================================================
if __name__ == '__main__':
    print("=" * 78)
    print("  第7-10轮优化: 基于X18继续微调")
    print("=" * 78)

    results = []

    # X18基准
    print("\n  --- X18基准 ---")
    r = test_strategy("X18", build_x18, desc="偏离度<0.3%且斜率<0.3%→空仓")
    print_result(r); results.append(r)

    # X19: E5止损参数优化
    print("\n  --- 第7轮 X19: E5止损参数优化 ---")
    for stop_thresh, stop_wt in [(0.08, 0.30), (0.10, 0.30), (0.12, 0.30),
                                   (0.10, 0.20), (0.10, 0.40), (0.10, 0.50)]:
        r = test_strategy(f"X19({stop_thresh*100:.0f}%/{stop_wt*100:.0f}%)", build_x19,
                          {'stop_threshold': stop_thresh, 'stop_weight': stop_wt},
                          f"止损{stop_thresh*100:.0f}%降仓{stop_wt*100:.0f}%")
        print_result(r); results.append(r)

    # X20: A1确认天数优化
    print("\n  --- 第8轮 X20: A1确认天数优化 ---")
    for n in [3, 4, 5]:
        r = test_strategy(f"X20({n}天确认)", build_x20, {'n_confirm': n},
                          f"A1确认{n}天")
        print_result(r); results.append(r)

    # X21: B2智能优化
    print("\n  --- 第9轮 X21: B2智能优化(value弱+growth跌→空仓) ---")
    r = test_strategy("X21", build_x21, desc="value弱且growth也跌→空仓等待")
    print_result(r); results.append(r)

    # X21参数敏感性
    for dev, slp in [(0.003, 0.003), (0.005, 0.003), (0.003, 0.005)]:
        r = test_strategy(f"X21(dev{dev*100:.1f}%+slp{slp*100:.1f}%)", build_x21,
                          {'dev_thresh': dev, 'slope_thresh': slp},
                          f"智能B2+偏离度{dev*100:.1f}%+斜率{slp*100:.1f}%")
        print_result(r); results.append(r)

    # X22: 综合最优组合
    print("\n  --- 第10轮 X22: 综合最优组合 ---")
    combos = [
        # (dev, slp, n_confirm, stop_thresh, stop_wt, smart_b2, desc)
        (0.003, 0.003, 4, 0.10, 0.30, True, "智能B2+标准E5"),
        (0.003, 0.003, 4, 0.12, 0.30, True, "智能B2+宽松E5(12%)"),
        (0.003, 0.003, 5, 0.10, 0.30, True, "智能B2+5天确认"),
        (0.005, 0.003, 4, 0.10, 0.30, True, "智能B2+dev0.5%"),
        (0.003, 0.003, 4, 0.10, 0.30, False, "原B2(对比)"),
    ]
    for dev, slp, n_conf, st, sw, smart, desc in combos:
        r = test_strategy(f"X22({desc})", build_x22,
                          {'dev_thresh': dev, 'slope_thresh': slp, 'n_confirm': n_conf,
                           'stop_threshold': st, 'stop_weight': sw, 'use_smart_b2': smart},
                          desc)
        print_result(r); results.append(r)

    # 汇总
    print("\n" + "=" * 78)
    print("  汇总对比(按Calmar排序)")
    print("=" * 78)
    sorted_r = sorted(results, key=lambda x: -x['calmar'])
    print(f"  {'版本':<35} {'年化':>7} {'回撤':>7} {'Sharpe':>7} {'Calmar':>7} {'交易':>5} {'滑点年化':>7}")
    print(f"  {'-'*80}")
    for r in sorted_r[:15]:
        print(f"  {r['name']:<35} {r['ann']*100:>6.2f}% {r['dd']*100:>6.2f}% "
              f"{r['sharpe']:>7.3f} {r['calmar']:>7.3f} {r['n_trades']:>5} {r['ann_slippage5']*100:>6.2f}%")
