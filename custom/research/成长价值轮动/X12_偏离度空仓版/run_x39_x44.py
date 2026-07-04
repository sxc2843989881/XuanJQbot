"""run_x39_x44.py — 第22轮: 进一步降低交易次数
================================================================
基于X33-X38第一轮结果,继续优化:

第一轮关键发现:
- X33方向冷却期: 基本无效(362→362),方向切换大多是有效切换
- X34 E5冷却期: 小幅有效(362→355,-7次)
- X35空仓退出确认2天: 有效(362→342,-20次),但年化边缘(40.01%)
- X36 E5触发调整: 全面恶化(回撤-40%+)
- X37 z阈值提高: 反而增加交易(z=2.0→378次)
- X38组合(cd5/ex2/e55): 337次/40.30%/-27.40%(当前最优)

第二轮新方案:
- X39: 空仓进入确认(进入空仓需连续N天both_weak=True)
- X40: X35退出确认2天 + X34 E5冷却(不用方向冷却,避免无效)
- X41: X38组合扩展(更细参数组合扫描)
- X42: 信号变化阈值+X35退出确认(组合已验证有效方向)
- X43: 空仓双向确认(进入+退出都需确认)
- X44: 最优组合精调
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
)
from run_x33_reduce_trades import (
    Z_THRESH, SLOPE_THRESH, N_CONFIRM, STOP_THRESHOLD, STOP_WEIGHT,
    RATIO_DEV_STD20, RATIO_DEV_Z, build_x23_base,
)

# ============================================================
# X39: 空仓进入确认(进入空仓需连续N天both_weak=True)
# ============================================================
def build_x39(entry_confirm_days=2, z_thresh=Z_THRESH, slope_thresh=SLOPE_THRESH,
              n_confirm=N_CONFIRM, stop_threshold=STOP_THRESHOLD,
              stop_weight=STOP_WEIGHT):
    """X39: 空仓进入需连续N天both_weak=True

    逻辑:
    - 进入空仓: 需连续N天both_weak=True才触发空仓
    - 退出空仓: both_weak=False即恢复(无延迟)
    - 目的: 减少误触发空仓(both_weak短暂为True就空仓)
    - 注: X16/X28测试过进入确认,但当时是3-5天过度延迟,这里测试2天
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

    # 状态机: 进入空仓需连续N天确认
    is_flat = False
    weak_count = 0
    for i in range(len(dir_s)):
        if pd.isna(RATIO_DEV_Z.iloc[i]):
            wt.iloc[i] = 1.0
            continue
        if not is_flat:
            if both_weak.iloc[i]:
                weak_count += 1
                if weak_count >= entry_confirm_days:
                    is_flat = True
                    wt.iloc[i] = 0.0
                else:
                    wt.iloc[i] = 1.0  # 等待确认期保持满仓
            else:
                weak_count = 0
                wt.iloc[i] = 1.0
        else:
            if not both_weak.iloc[i]:
                is_flat = False
                weak_count = 0
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
# X40: 空仓退出确认2天 + E5冷却3天(不用方向冷却)
# ============================================================
def build_x40(exit_confirm_days=2, e5_cooldown_days=3,
              z_thresh=Z_THRESH, slope_thresh=SLOPE_THRESH,
              n_confirm=N_CONFIRM, stop_threshold=STOP_THRESHOLD,
              stop_weight=STOP_WEIGHT):
    """X40: X35退出确认 + X34 E5冷却(无方向冷却)

    逻辑: 组合两个已验证有效的方向,不加无效的方向冷却
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

    # 1. 空仓退出确认
    is_flat = False
    not_weak_count = 0
    for i in range(len(dir_s)):
        if pd.isna(RATIO_DEV_Z.iloc[i]):
            wt.iloc[i] = 1.0
            continue
        if not is_flat:
            if both_weak.iloc[i]:
                is_flat = True
                not_weak_count = 0
                wt.iloc[i] = 0.0
            else:
                wt.iloc[i] = 1.0
        else:
            if not both_weak.iloc[i]:
                not_weak_count += 1
                if not_weak_count >= exit_confirm_days:
                    is_flat = False
                    wt.iloc[i] = 1.0
                else:
                    wt.iloc[i] = 0.0
            else:
                not_weak_count = 0
                wt.iloc[i] = 0.0

    dir_s = confirmed.ffill()
    wrong_value = (dir_s == 'value') & (V_MOM20 <= 0)
    dir_s[wrong_value] = 'growth'

    # 2. E5降仓冷却期
    gs = (dir_s == 'growth') & (G_DD20 < -stop_threshold)
    vs = (dir_s == 'value') & (V_DD20 < -stop_threshold)
    e5_trigger = gs | vs

    in_cooldown = False
    cooldown_count = 0
    for i in range(len(wt)):
        if pd.isna(dir_s.iloc[i]) or pd.isna(wt.iloc[i]):
            continue
        if e5_trigger.iloc[i] and not in_cooldown:
            in_cooldown = True
            cooldown_count = 0
            wt.iloc[i] = wt.iloc[i] * stop_weight
        elif in_cooldown:
            cooldown_count += 1
            if cooldown_count >= e5_cooldown_days:
                if e5_trigger.iloc[i]:
                    cooldown_count = 0
                    wt.iloc[i] = wt.iloc[i] * stop_weight
                else:
                    in_cooldown = False
                    if wt.iloc[i] > 0:
                        wt.iloc[i] = 1.0
            else:
                if wt.iloc[i] > 0:
                    wt.iloc[i] = stop_weight
    return dir_s, wt


# ============================================================
# X41: X38组合扩展(更细参数扫描)
# ============================================================
def build_x41(dir_cooldown_days=5, exit_confirm_days=2, e5_cooldown_days=5,
              z_thresh=Z_THRESH, slope_thresh=SLOPE_THRESH,
              n_confirm=N_CONFIRM, stop_threshold=STOP_THRESHOLD,
              stop_weight=STOP_WEIGHT):
    """X41: X38组合(同逻辑,用于更细参数扫描)"""
    from run_x33_reduce_trades import build_x38
    return build_x38(dir_cooldown_days=dir_cooldown_days,
                     exit_confirm_days=exit_confirm_days,
                     e5_cooldown_days=e5_cooldown_days,
                     z_thresh=z_thresh, slope_thresh=slope_thresh,
                     n_confirm=n_confirm, stop_threshold=stop_threshold,
                     stop_weight=stop_weight)


# ============================================================
# X42: 信号变化阈值+空仓退出确认
# ============================================================
def build_x42(z_change_thresh=0.3, exit_confirm_days=2,
              z_thresh=Z_THRESH, slope_thresh=SLOPE_THRESH,
              n_confirm=N_CONFIRM, stop_threshold=STOP_THRESHOLD,
              stop_weight=STOP_WEIGHT):
    """X42: z变化阈值+空仓退出确认

    逻辑:
    - 空仓切换(进入和退出)需z变化超过阈值
    - 退出还需连续N天不满足空仓条件
    - 目的: 双重减少空仓震荡
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
    z_change = RATIO_DEV_Z.diff().abs()

    # 状态机: z变化阈值+退出确认
    is_flat = False
    not_weak_count = 0
    for i in range(len(dir_s)):
        if pd.isna(RATIO_DEV_Z.iloc[i]):
            wt.iloc[i] = 1.0
            continue
        z_chg = z_change.iloc[i] if i > 0 else 0
        if pd.isna(z_chg):
            z_chg = 0
        if not is_flat:
            if both_weak.iloc[i] and z_chg > z_change_thresh:
                is_flat = True
                not_weak_count = 0
                wt.iloc[i] = 0.0
            else:
                wt.iloc[i] = 1.0
        else:
            if not both_weak.iloc[i]:
                not_weak_count += 1
                if not_weak_count >= exit_confirm_days and z_chg > z_change_thresh:
                    is_flat = False
                    wt.iloc[i] = 1.0
                else:
                    wt.iloc[i] = 0.0
            else:
                not_weak_count = 0
                wt.iloc[i] = 0.0

    dir_s = confirmed.ffill()
    wrong_value = (dir_s == 'value') & (V_MOM20 <= 0)
    dir_s[wrong_value] = 'growth'
    gs = (dir_s == 'growth') & (G_DD20 < -stop_threshold)
    vs = (dir_s == 'value') & (V_DD20 < -stop_threshold)
    wt[gs | vs] = wt[gs | vs] * stop_weight
    return dir_s, wt


# ============================================================
# X43: 空仓双向确认(进入+退出都需确认)
# ============================================================
def build_x43(entry_confirm_days=2, exit_confirm_days=2,
              z_thresh=Z_THRESH, slope_thresh=SLOPE_THRESH,
              n_confirm=N_CONFIRM, stop_threshold=STOP_THRESHOLD,
              stop_weight=STOP_WEIGHT):
    """X43: 空仓双向确认(进入+退出都需连续N天)

    逻辑:
    - 进入空仓: 需连续entry_confirm_days天both_weak=True
    - 退出空仓: 需连续exit_confirm_days天both_weak=False
    - 目的: 双向减少空仓震荡
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

    # 状态机: 双向确认
    is_flat = False
    weak_count = 0
    not_weak_count = 0
    for i in range(len(dir_s)):
        if pd.isna(RATIO_DEV_Z.iloc[i]):
            wt.iloc[i] = 1.0
            continue
        if not is_flat:
            if both_weak.iloc[i]:
                weak_count += 1
                if weak_count >= entry_confirm_days:
                    is_flat = True
                    not_weak_count = 0
                    wt.iloc[i] = 0.0
                else:
                    wt.iloc[i] = 1.0
            else:
                weak_count = 0
                wt.iloc[i] = 1.0
        else:
            if not both_weak.iloc[i]:
                not_weak_count += 1
                if not_weak_count >= exit_confirm_days:
                    is_flat = False
                    weak_count = 0
                    wt.iloc[i] = 1.0
                else:
                    wt.iloc[i] = 0.0
            else:
                not_weak_count = 0
                wt.iloc[i] = 0.0

    dir_s = confirmed.ffill()
    wrong_value = (dir_s == 'value') & (V_MOM20 <= 0)
    dir_s[wrong_value] = 'growth'
    gs = (dir_s == 'growth') & (G_DD20 < -stop_threshold)
    vs = (dir_s == 'value') & (V_DD20 < -stop_threshold)
    wt[gs | vs] = wt[gs | vs] * stop_weight
    return dir_s, wt


# ============================================================
# 主测试流程
# ============================================================
if __name__ == '__main__':
    print("=" * 90)
    print("  X39-X44: 第22轮 进一步降低交易次数")
    print("  约束: 年化>40% 且 回撤<-35%")
    print("=" * 90)

    results = []

    # 基准
    print("\n  [基准] X23(z=1.5, slope=0.002):")
    info = test_strategy("X23基准", build_x23_base, desc="基准")
    print_result(info)
    results.append(info)
    # 第一轮最优
    print("\n  [第一轮最优] X38(cd5/ex2/e55):")
    from run_x33_reduce_trades import build_x38
    info = test_strategy("X38(cd5/ex2/e55)", build_x38,
                         {'dir_cooldown_days': 5, 'exit_confirm_days': 2, 'e5_cooldown_days': 5},
                         "第一轮最优")
    print_result(info)
    results.append(info)

    # X39: 空仓进入确认
    print("\n" + "=" * 90)
    print("  X39: 空仓进入确认(进入空仓需连续N天both_weak=True)")
    print("=" * 90)
    print(f"  {'进入确认':>10} {'年化':>8} {'回撤':>8} {'Sharpe':>8} {'Calmar':>8} {'交易':>6} {'满足':>6}")
    print(f"  {'-'*65}")
    for n in [2, 3, 4, 5]:
        info = test_strategy(f"X39(entry={n})", build_x39,
                             {'entry_confirm_days': n}, f"进入确认{n}天")
        m = info
        ok = "✅" if m['ann'] > 0.40 and m['dd'] > -0.35 else "❌"
        print(f"  {n:>8}天 {m['ann']*100:>7.2f}% {m['dd']*100:>7.2f}% "
              f"{m['sharpe']:>8.3f} {m['calmar']:>8.3f} {m['n_trades']:>6} {ok:>6}")
        results.append(info)

    # X40: 退出确认+E5冷却(无方向冷却)
    print("\n" + "=" * 90)
    print("  X40: 退出确认+E5冷却(无方向冷却)")
    print("=" * 90)
    print(f"  {'配置':>20} {'年化':>8} {'回撤':>8} {'Sharpe':>8} {'Calmar':>8} {'交易':>6} {'满足':>6}")
    print(f"  {'-'*75}")
    for ex, e5 in [(2, 3), (2, 5), (2, 7), (2, 10), (3, 3), (3, 5)]:
        info = test_strategy(f"X40(ex{ex}/e5{e5})", build_x40,
                             {'exit_confirm_days': ex, 'e5_cooldown_days': e5},
                             f"退出{ex}/E5冷却{e5}")
        m = info
        ok = "✅" if m['ann'] > 0.40 and m['dd'] > -0.35 else "❌"
        print(f"  {f'ex{ex}/e5{e5}':>20} {m['ann']*100:>7.2f}% {m['dd']*100:>7.2f}% "
              f"{m['sharpe']:>8.3f} {m['calmar']:>8.3f} {m['n_trades']:>6} {ok:>6}")
        results.append(info)

    # X41: X38组合更细参数扫描
    print("\n" + "=" * 90)
    print("  X41: X38组合更细参数扫描")
    print("=" * 90)
    print(f"  {'配置':>25} {'年化':>8} {'回撤':>8} {'Sharpe':>8} {'Calmar':>8} {'交易':>6} {'满足':>6}")
    print(f"  {'-'*80}")
    combos = [
        {'dir_cooldown_days': 3, 'exit_confirm_days': 2, 'e5_cooldown_days': 3},
        {'dir_cooldown_days': 5, 'exit_confirm_days': 2, 'e5_cooldown_days': 3},
        {'dir_cooldown_days': 5, 'exit_confirm_days': 2, 'e5_cooldown_days': 7},
        {'dir_cooldown_days': 5, 'exit_confirm_days': 2, 'e5_cooldown_days': 10},
        {'dir_cooldown_days': 7, 'exit_confirm_days': 2, 'e5_cooldown_days': 3},
        {'dir_cooldown_days': 10, 'exit_confirm_days': 2, 'e5_cooldown_days': 5},
        {'dir_cooldown_days': 10, 'exit_confirm_days': 2, 'e5_cooldown_days': 10},
    ]
    for c in combos:
        label = f"cd{c['dir_cooldown_days']}/ex{c['exit_confirm_days']}/e5{c['e5_cooldown_days']}"
        info = test_strategy(f"X41({label})", build_x41, c, label)
        m = info
        ok = "✅" if m['ann'] > 0.40 and m['dd'] > -0.35 else "❌"
        print(f"  {label:>25} {m['ann']*100:>7.2f}% {m['dd']*100:>7.2f}% "
              f"{m['sharpe']:>8.3f} {m['calmar']:>8.3f} {m['n_trades']:>6} {ok:>6}")
        results.append(info)

    # X42: z变化阈值+退出确认
    print("\n" + "=" * 90)
    print("  X42: z变化阈值+空仓退出确认")
    print("=" * 90)
    print(f"  {'配置':>20} {'年化':>8} {'回撤':>8} {'Sharpe':>8} {'Calmar':>8} {'交易':>6} {'满足':>6}")
    print(f"  {'-'*75}")
    for zc, ex in [(0.2, 2), (0.3, 2), (0.5, 2), (0.3, 3), (0.5, 3)]:
        info = test_strategy(f"X42(zc{zc}/ex{ex})", build_x42,
                             {'z_change_thresh': zc, 'exit_confirm_days': ex},
                             f"z变化{zc}/退出{ex}")
        m = info
        ok = "✅" if m['ann'] > 0.40 and m['dd'] > -0.35 else "❌"
        print(f"  {f'zc{zc}/ex{ex}':>20} {m['ann']*100:>7.2f}% {m['dd']*100:>7.2f}% "
              f"{m['sharpe']:>8.3f} {m['calmar']:>8.3f} {m['n_trades']:>6} {ok:>6}")
        results.append(info)

    # X43: 空仓双向确认
    print("\n" + "=" * 90)
    print("  X43: 空仓双向确认(进入+退出都需确认)")
    print("=" * 90)
    print(f"  {'配置':>20} {'年化':>8} {'回撤':>8} {'Sharpe':>8} {'Calmar':>8} {'交易':>6} {'满足':>6}")
    print(f"  {'-'*75}")
    for en, ex in [(2, 2), (2, 3), (3, 2), (3, 3), (2, 4), (4, 2)]:
        info = test_strategy(f"X43(en{en}/ex{ex})", build_x43,
                             {'entry_confirm_days': en, 'exit_confirm_days': ex},
                             f"进入{en}/退出{ex}")
        m = info
        ok = "✅" if m['ann'] > 0.40 and m['dd'] > -0.35 else "❌"
        print(f"  {f'en{en}/ex{ex}':>20} {m['ann']*100:>7.2f}% {m['dd']*100:>7.2f}% "
              f"{m['sharpe']:>8.3f} {m['calmar']:>8.3f} {m['n_trades']:>6} {ok:>6}")
        results.append(info)

    # 汇总
    print("\n" + "=" * 90)
    print("  【汇总】满足约束(年化>40%且回撤<-35%)的方案,按交易次数升序")
    print("=" * 90)
    feasible = [r for r in results if r['ann'] > 0.40 and r['dd'] > -0.35]
    feasible.sort(key=lambda x: x['n_trades'])
    print(f"  {'版本':<30} {'年化':>8} {'回撤':>8} {'Sharpe':>8} {'Calmar':>8} {'交易':>6}")
    print(f"  {'-'*75}")
    for r in feasible[:15]:
        print(f"  {r['name']:<30} {r['ann']*100:>7.2f}% {r['dd']*100:>7.2f}% "
              f"{r['sharpe']:>8.3f} {r['calmar']:>8.3f} {r['n_trades']:>6}")

    if feasible:
        best = feasible[0]
        print(f"\n  ★最优方案(交易最少): {best['name']}")
        print(f"    年化={best['ann']*100:.2f}% 回撤={best['dd']*100:.2f}% "
              f"Sharpe={best['sharpe']:.3f} Calmar={best['calmar']:.3f} 交易={best['n_trades']}次")

    print("\n  完成!")
