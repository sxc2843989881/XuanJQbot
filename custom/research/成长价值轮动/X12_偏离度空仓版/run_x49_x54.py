"""run_x49_x54.py — 第24轮: 继续降低交易次数
================================================================
第三轮最优:
- X44(en4/ex2/e53): 240次/40.26%/-28.72%(交易最少)
- X44(en4/ex2/e55): 242次/40.64%/-27.92%(年化最高且回撤最浅)

当前交易构成: 方向183次 + 空仓59次 = 242次
方向切换183次未减少,是下一步优化重点

第四轮新方案:
- X49: en5/ex2 + E5冷却(更严格进入确认)
- X50: en4/ex3 + E5冷却(更严格退出确认)
- X51: en4/ex2/e55 + 方向切换需连续N天F1确认(减少方向震荡)
- X52: en4/ex2/e55 + 方向切换最小持有期(强制持有N天)
- X53: en4/ex2/e55 + 5/万滑点验证
- X54: 最优组合最终精调
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
from run_x44_x48 import build_x43_with_e5


# ============================================================
# X51: 双向确认 + 方向切换需连续N天F1确认
# ============================================================
def build_x51(entry_confirm_days=4, exit_confirm_days=2, e5_cooldown_days=5,
              dir_confirm_days=5, z_thresh=Z_THRESH, slope_thresh=SLOPE_THRESH,
              n_confirm=N_CONFIRM, stop_threshold=STOP_THRESHOLD,
              stop_weight=STOP_WEIGHT):
    """X51: 双向确认 + 方向切换需连续N天F1确认

    逻辑:
    - 空仓双向确认(en4/ex2)
    - E5冷却5天
    - 新增: 方向切换需连续dir_confirm_days天F1信号一致(比A1的4天更长)
    - 目的: 减少方向切换183次
    """
    dir_s = BASE_DIR.copy()
    # 方向确认: 连续dir_confirm_days天一致
    mask = np.ones(len(dir_s), dtype=bool)
    for k in range(1, dir_confirm_days):
        mask = mask & (dir_s.values == dir_s.shift(k).values)
    confirmed = dir_s.where(mask, np.nan)

    wt = pd.Series(1.0, index=dir_s.index)
    low_dev_z = RATIO_DEV_Z.abs() < z_thresh
    low_slope = MA20_SLOPE.abs() < slope_thresh
    both_weak = low_dev_z & low_slope

    # 空仓双向确认
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

    # E5降仓冷却期
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
# X52: 双向确认 + 方向切换最小持有期
# ============================================================
def build_x52(entry_confirm_days=4, exit_confirm_days=2, e5_cooldown_days=5,
              dir_min_hold=10, z_thresh=Z_THRESH, slope_thresh=SLOPE_THRESH,
              n_confirm=N_CONFIRM, stop_threshold=STOP_THRESHOLD,
              stop_weight=STOP_WEIGHT):
    """X52: 双向确认 + 方向切换最小持有期

    逻辑:
    - 空仓双向确认(en4/ex2)
    - E5冷却5天
    - 新增: 方向切换后至少持有dir_min_hold天才再切换(冷却期)
    - 但F0空仓和E5降仓仍可触发(风控优先)
    """
    dir_s = BASE_DIR.copy()
    mask = np.ones(len(dir_s), dtype=bool)
    for k in range(1, n_confirm):
        mask = mask & (dir_s.values == dir_s.shift(k).values)
    confirmed = dir_s.where(mask, np.nan)

    # 方向切换最小持有期
    raw_dir = confirmed.ffill()
    cooled_dir = raw_dir.copy()
    last_change_idx = -np.inf
    current_dir = raw_dir.iloc[0]
    for i in range(len(raw_dir)):
        if pd.isna(raw_dir.iloc[i]):
            continue
        if raw_dir.iloc[i] != current_dir:
            if i - last_change_idx >= dir_min_hold:
                current_dir = raw_dir.iloc[i]
                last_change_idx = i
            cooled_dir.iloc[i] = current_dir
        else:
            cooled_dir.iloc[i] = current_dir
    dir_s = cooled_dir

    wt = pd.Series(1.0, index=dir_s.index)
    low_dev_z = RATIO_DEV_Z.abs() < z_thresh
    low_slope = MA20_SLOPE.abs() < slope_thresh
    both_weak = low_dev_z & low_slope

    # 空仓双向确认
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

    wrong_value = (dir_s == 'value') & (V_MOM20 <= 0)
    dir_s[wrong_value] = 'growth'

    # E5降仓冷却期
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
# 主测试流程
# ============================================================
if __name__ == '__main__':
    print("=" * 90)
    print("  X49-X54: 第24轮 继续降低交易次数")
    print("  约束: 年化>40% 且 回撤<-35%")
    print("=" * 90)

    results = []

    # 基准对比
    print("\n  [基准] X23:")
    info = test_strategy("X23基准", build_x23_base, desc="基准")
    print_result(info)
    results.append(info)

    print("\n  [第三轮最优] X44(en4/ex2/e55):")
    info = test_strategy("X44(en4/ex2/e55)", build_x43_with_e5,
                         {'entry_confirm_days': 4, 'exit_confirm_days': 2, 'e5_cooldown_days': 5},
                         "第三轮最优")
    print_result(info)
    results.append(info)

    # X49: en5/ex2 + E5冷却(更严格进入)
    print("\n" + "=" * 90)
    print("  X49: en5/ex2 + E5冷却(更严格进入确认)")
    print("=" * 90)
    print(f"  {'E5冷却':>10} {'年化':>8} {'回撤':>8} {'Sharpe':>8} {'Calmar':>8} {'交易':>6} {'满足':>6}")
    print(f"  {'-'*65}")
    for e5 in [0, 3, 5, 7]:
        info = test_strategy(f"X49(en5/ex2/e5{e5})", build_x43_with_e5,
                             {'entry_confirm_days': 5, 'exit_confirm_days': 2,
                              'e5_cooldown_days': e5},
                             f"en5/ex2/E5冷却{e5}")
        m = info
        ok = "✅" if m['ann'] > 0.40 and m['dd'] > -0.35 else "❌"
        print(f"  {e5:>8}天 {m['ann']*100:>7.2f}% {m['dd']*100:>7.2f}% "
              f"{m['sharpe']:>8.3f} {m['calmar']:>8.3f} {m['n_trades']:>6} {ok:>6}")
        results.append(info)

    # X50: en4/ex3 + E5冷却(更严格退出)
    print("\n" + "=" * 90)
    print("  X50: en4/ex3 + E5冷却(更严格退出确认)")
    print("=" * 90)
    print(f"  {'E5冷却':>10} {'年化':>8} {'回撤':>8} {'Sharpe':>8} {'Calmar':>8} {'交易':>6} {'满足':>6}")
    print(f"  {'-'*65}")
    for e5 in [0, 3, 5, 7]:
        info = test_strategy(f"X50(en4/ex3/e5{e5})", build_x43_with_e5,
                             {'entry_confirm_days': 4, 'exit_confirm_days': 3,
                              'e5_cooldown_days': e5},
                             f"en4/ex3/E5冷却{e5}")
        m = info
        ok = "✅" if m['ann'] > 0.40 and m['dd'] > -0.35 else "❌"
        print(f"  {e5:>8}天 {m['ann']*100:>7.2f}% {m['dd']*100:>7.2f}% "
              f"{m['sharpe']:>8.3f} {m['calmar']:>8.3f} {m['n_trades']:>6} {ok:>6}")
        results.append(info)

    # X51: en4/ex2/e55 + 方向切换需连续N天F1确认
    print("\n" + "=" * 90)
    print("  X51: en4/ex2/e55 + 方向切换需连续N天F1确认")
    print("=" * 90)
    print(f"  {'方向确认':>10} {'年化':>8} {'回撤':>8} {'Sharpe':>8} {'Calmar':>8} {'交易':>6} {'满足':>6}")
    print(f"  {'-'*65}")
    for dc in [5, 6, 7, 10]:
        info = test_strategy(f"X51(dc{dc})", build_x51,
                             {'entry_confirm_days': 4, 'exit_confirm_days': 2,
                              'e5_cooldown_days': 5, 'dir_confirm_days': dc},
                             f"方向确认{dc}天")
        m = info
        ok = "✅" if m['ann'] > 0.40 and m['dd'] > -0.35 else "❌"
        print(f"  {dc:>8}天 {m['ann']*100:>7.2f}% {m['dd']*100:>7.2f}% "
              f"{m['sharpe']:>8.3f} {m['calmar']:>8.3f} {m['n_trades']:>6} {ok:>6}")
        results.append(info)

    # X52: en4/ex2/e55 + 方向切换最小持有期
    print("\n" + "=" * 90)
    print("  X52: en4/ex2/e55 + 方向切换最小持有期")
    print("=" * 90)
    print(f"  {'最小持有':>10} {'年化':>8} {'回撤':>8} {'Sharpe':>8} {'Calmar':>8} {'交易':>6} {'满足':>6}")
    print(f"  {'-'*65}")
    for dh in [5, 7, 10, 15, 20, 30]:
        info = test_strategy(f"X52(dh{dh})", build_x52,
                             {'entry_confirm_days': 4, 'exit_confirm_days': 2,
                              'e5_cooldown_days': 5, 'dir_min_hold': dh},
                             f"最小持有{dh}天")
        m = info
        ok = "✅" if m['ann'] > 0.40 and m['dd'] > -0.35 else "❌"
        print(f"  {dh:>8}天 {m['ann']*100:>7.2f}% {m['dd']*100:>7.2f}% "
              f"{m['sharpe']:>8.3f} {m['calmar']:>8.3f} {m['n_trades']:>6} {ok:>6}")
        results.append(info)

    # X53: 最优组合的5/万滑点验证
    print("\n" + "=" * 90)
    print("  X53: 5/万滑点验证(前3名方案)")
    print("=" * 90)
    # 取当前最优3个
    feasible = [r for r in results if r['ann'] > 0.40 and r['dd'] > -0.35]
    feasible.sort(key=lambda x: x['n_trades'])
    top3 = feasible[:3]
    print(f"  {'版本':<35} {'无滑点年化':>10} {'滑点年化':>10} {'滑点Calmar':>10} {'交易':>6}")
    print(f"  {'-'*75}")
    for r in top3:
        # 重新构建信号测试滑点
        sig, wt = None, None
        if 'en4' in r['name'] and 'e55' in r['name'] and 'X44' in r['name']:
            sig, wt = build_x43_with_e5(entry_confirm_days=4, exit_confirm_days=2, e5_cooldown_days=5)
        elif 'en4' in r['name'] and 'e53' in r['name']:
            sig, wt = build_x43_with_e5(entry_confirm_days=4, exit_confirm_days=2, e5_cooldown_days=3)
        elif 'X44(en4/ex2/e55)' in r['name']:
            sig, wt = build_x43_with_e5(entry_confirm_days=4, exit_confirm_days=2, e5_cooldown_days=5)
        if sig is not None:
            res_sl = run_backtest(sig, wt, impact_slippage=0.0005)
            m_sl = calc_metrics(res_sl)
            print(f"  {r['name']:<35} {r['ann']*100:>9.2f}% {m_sl['ann']*100:>9.2f}% "
                  f"{m_sl['calmar']:>10.3f} {r['n_trades']:>6}")

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
