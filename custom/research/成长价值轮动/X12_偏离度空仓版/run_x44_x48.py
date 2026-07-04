"""run_x44_x48.py — 第23轮: X43(en4/ex2)基础上提升年化
================================================================
第二轮关键发现:
- X43(en4/ex2): 248次/40.05%/-30.36%(交易-31%,但年化边缘)
- X43(en3/ex2): 259次/39.86%(差0.14%达标)
- X40(ex2/e55): 337次/40.54%/-27.40%(年化更安全)
- X23基准: 362次/43.50%/-25.85%

第三轮目标: 在X43(en4/ex2)基础上提升年化到41%+,保持交易<260次
新方案:
- X44: X43(en4/ex2) + E5冷却3天(减少E5恢复交易)
- X45: X43(en4/ex2) + E5冷却5天
- X46: X43(en4/ex2) + E5参数调整(降仓比例20%/40%)
- X47: X43(en4/ex2) + z阈值微调(1.3/1.4/1.6)
- X48: X43(en3/ex2) + E5冷却(尝试让259次版本年化上40%)
- X49: 最优组合精调
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
# 通用: 空仓双向确认 + E5冷却
# ============================================================
def build_x43_with_e5(entry_confirm_days=4, exit_confirm_days=2,
                      e5_cooldown_days=0,  # 0=不冷却
                      z_thresh=Z_THRESH, slope_thresh=SLOPE_THRESH,
                      n_confirm=N_CONFIRM, stop_threshold=STOP_THRESHOLD,
                      stop_weight=STOP_WEIGHT):
    """X43双向确认 + 可选E5冷却期

    逻辑:
    - 空仓进入: 连续entry_confirm_days天both_weak=True
    - 空仓退出: 连续exit_confirm_days天both_weak=False
    - E5冷却: 触发后e5_cooldown_days天内不恢复(0=不冷却)
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

    # 1. 空仓双向确认
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

    # 2. E5降仓冷却期(可选)
    gs = (dir_s == 'growth') & (G_DD20 < -stop_threshold)
    vs = (dir_s == 'value') & (V_DD20 < -stop_threshold)
    e5_trigger = gs | vs

    if e5_cooldown_days > 0:
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
    else:
        wt[gs | vs] = wt[gs | vs] * stop_weight

    return dir_s, wt


# ============================================================
# 主测试流程
# ============================================================
if __name__ == '__main__':
    print("=" * 90)
    print("  X44-X48: 第23轮 X43(en4/ex2)基础上提升年化")
    print("  约束: 年化>40% 且 回撤<-35%")
    print("=" * 90)

    results = []

    # 基准对比
    print("\n  [基准] X23:")
    info = test_strategy("X23基准", build_x23_base, desc="基准")
    print_result(info)
    results.append(info)

    print("\n  [第二轮最优] X43(en4/ex2):")
    info = test_strategy("X43(en4/ex2)", build_x43_with_e5,
                         {'entry_confirm_days': 4, 'exit_confirm_days': 2, 'e5_cooldown_days': 0},
                         "第二轮最优")
    print_result(info)
    results.append(info)

    # X44: X43(en4/ex2) + E5冷却3天
    print("\n" + "=" * 90)
    print("  X44: X43(en4/ex2) + E5冷却N天")
    print("=" * 90)
    print(f"  {'E5冷却':>10} {'年化':>8} {'回撤':>8} {'Sharpe':>8} {'Calmar':>8} {'交易':>6} {'满足':>6}")
    print(f"  {'-'*65}")
    for e5 in [0, 3, 5, 7, 10]:
        info = test_strategy(f"X44(en4/ex2/e5{e5})", build_x43_with_e5,
                             {'entry_confirm_days': 4, 'exit_confirm_days': 2,
                              'e5_cooldown_days': e5},
                             f"E5冷却{e5}天")
        m = info
        ok = "✅" if m['ann'] > 0.40 and m['dd'] > -0.35 else "❌"
        print(f"  {e5:>8}天 {m['ann']*100:>7.2f}% {m['dd']*100:>7.2f}% "
              f"{m['sharpe']:>8.3f} {m['calmar']:>8.3f} {m['n_trades']:>6} {ok:>6}")
        results.append(info)

    # X45: X43(en4/ex2) + E5降仓比例调整
    print("\n" + "=" * 90)
    print("  X45: X43(en4/ex2) + E5降仓比例调整")
    print("=" * 90)
    print(f"  {'降仓比例':>10} {'年化':>8} {'回撤':>8} {'Sharpe':>8} {'Calmar':>8} {'交易':>6} {'满足':>6}")
    print(f"  {'-'*65}")
    for sw in [0.20, 0.25, 0.30, 0.35, 0.40]:
        info = test_strategy(f"X45(sw{sw})", build_x43_with_e5,
                             {'entry_confirm_days': 4, 'exit_confirm_days': 2,
                              'e5_cooldown_days': 0, 'stop_weight': sw},
                             f"降仓{sw*100:.0f}%")
        m = info
        ok = "✅" if m['ann'] > 0.40 and m['dd'] > -0.35 else "❌"
        print(f"  {sw*100:>8.0f}% {m['ann']*100:>7.2f}% {m['dd']*100:>7.2f}% "
              f"{m['sharpe']:>8.3f} {m['calmar']:>8.3f} {m['n_trades']:>6} {ok:>6}")
        results.append(info)

    # X46: X43(en4/ex2) + z阈值微调
    print("\n" + "=" * 90)
    print("  X46: X43(en4/ex2) + z阈值微调")
    print("=" * 90)
    print(f"  {'z阈值':>10} {'年化':>8} {'回撤':>8} {'Sharpe':>8} {'Calmar':>8} {'交易':>6} {'满足':>6}")
    print(f"  {'-'*65}")
    for z in [1.2, 1.3, 1.4, 1.5, 1.6, 1.7]:
        info = test_strategy(f"X46(z{z})", build_x43_with_e5,
                             {'entry_confirm_days': 4, 'exit_confirm_days': 2,
                              'e5_cooldown_days': 0, 'z_thresh': z},
                             f"z={z}")
        m = info
        ok = "✅" if m['ann'] > 0.40 and m['dd'] > -0.35 else "❌"
        print(f"  {z:>10} {m['ann']*100:>7.2f}% {m['dd']*100:>7.2f}% "
              f"{m['sharpe']:>8.3f} {m['calmar']:>8.3f} {m['n_trades']:>6} {ok:>6}")
        results.append(info)

    # X47: X43(en3/ex2) + E5冷却(尝试让259次版本年化上40%)
    print("\n" + "=" * 90)
    print("  X47: X43(en3/ex2) + E5冷却(让259次版本年化上40%)")
    print("=" * 90)
    print(f"  {'E5冷却':>10} {'年化':>8} {'回撤':>8} {'Sharpe':>8} {'Calmar':>8} {'交易':>6} {'满足':>6}")
    print(f"  {'-'*65}")
    for e5 in [0, 3, 5, 7]:
        info = test_strategy(f"X47(en3/ex2/e5{e5})", build_x43_with_e5,
                             {'entry_confirm_days': 3, 'exit_confirm_days': 2,
                              'e5_cooldown_days': e5},
                             f"en3/ex2/E5冷却{e5}")
        m = info
        ok = "✅" if m['ann'] > 0.40 and m['dd'] > -0.35 else "❌"
        print(f"  {e5:>8}天 {m['ann']*100:>7.2f}% {m['dd']*100:>7.2f}% "
              f"{m['sharpe']:>8.3f} {m['calmar']:>8.3f} {m['n_trades']:>6} {ok:>6}")
        results.append(info)

    # X48: 最优组合精调(en4/ex2 + z1.4 + E5冷却5)
    print("\n" + "=" * 90)
    print("  X48: 最优组合精调(en4/ex2 + z微调 + E5冷却)")
    print("=" * 90)
    print(f"  {'配置':>30} {'年化':>8} {'回撤':>8} {'Sharpe':>8} {'Calmar':>8} {'交易':>6} {'满足':>6}")
    print(f"  {'-'*85}")
    combos = [
        {'entry_confirm_days': 4, 'exit_confirm_days': 2, 'e5_cooldown_days': 5, 'z_thresh': 1.4},
        {'entry_confirm_days': 4, 'exit_confirm_days': 2, 'e5_cooldown_days': 5, 'z_thresh': 1.3},
        {'entry_confirm_days': 4, 'exit_confirm_days': 2, 'e5_cooldown_days': 3, 'z_thresh': 1.4},
        {'entry_confirm_days': 4, 'exit_confirm_days': 2, 'e5_cooldown_days': 3, 'z_thresh': 1.3},
        {'entry_confirm_days': 4, 'exit_confirm_days': 2, 'e5_cooldown_days': 5, 'z_thresh': 1.5},
        {'entry_confirm_days': 3, 'exit_confirm_days': 2, 'e5_cooldown_days': 5, 'z_thresh': 1.3},
        {'entry_confirm_days': 3, 'exit_confirm_days': 2, 'e5_cooldown_days': 5, 'z_thresh': 1.4},
    ]
    for c in combos:
        label = f"en{c['entry_confirm_days']}/ex{c['exit_confirm_days']}/e5{c['e5_cooldown_days']}/z{c['z_thresh']}"
        info = test_strategy(f"X48({label})", build_x43_with_e5, c, label)
        m = info
        ok = "✅" if m['ann'] > 0.40 and m['dd'] > -0.35 else "❌"
        print(f"  {label:>30} {m['ann']*100:>7.2f}% {m['dd']*100:>7.2f}% "
              f"{m['sharpe']:>8.3f} {m['calmar']:>8.3f} {m['n_trades']:>6} {ok:>6}")
        results.append(info)

    # 汇总
    print("\n" + "=" * 90)
    print("  【汇总】满足约束(年化>40%且回撤<-35%)的方案,按交易次数升序")
    print("=" * 90)
    feasible = [r for r in results if r['ann'] > 0.40 and r['dd'] > -0.35]
    feasible.sort(key=lambda x: x['n_trades'])
    print(f"  {'版本':<35} {'年化':>8} {'回撤':>8} {'Sharpe':>8} {'Calmar':>8} {'交易':>6}")
    print(f"  {'-'*80}")
    for r in feasible[:20]:
        print(f"  {r['name']:<35} {r['ann']*100:>7.2f}% {r['dd']*100:>7.2f}% "
              f"{r['sharpe']:>8.3f} {r['calmar']:>8.3f} {r['n_trades']:>6}")

    if feasible:
        best = feasible[0]
        print(f"\n  ★最优方案(交易最少): {best['name']}")
        print(f"    年化={best['ann']*100:.2f}% 回撤={best['dd']*100:.2f}% "
              f"Sharpe={best['sharpe']:.3f} Calmar={best['calmar']:.3f} 交易={best['n_trades']}次")

    print("\n  完成!")
