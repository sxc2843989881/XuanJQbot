"""run_x54_x58.py — 第25轮: 极限降低交易次数 + 滑点验证
================================================================
第四轮最优:
- X49(en5/ex2/e53): 234次/40.61%/-31.39%(交易最少)
- X51(dc5): 235次/40.46%/-28.85%(回撤更浅)
- X49(en5/ex2/e55): 236次/41.00%/-31.17%(年化最高)

第五轮目标:
1. 尝试en6/ex2(更严格进入)
2. 组合X49+X51(en5/ex2/e55+方向确认5天)
3. 修复滑点验证
4. 确定最终最优方案
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
from run_x49_x54 import build_x51


# ============================================================
# X55: en5/ex2/e55 + 方向确认5天(组合X49+X51)
# ============================================================
def build_x55(entry_confirm_days=5, exit_confirm_days=2, e5_cooldown_days=5,
              dir_confirm_days=5, z_thresh=Z_THRESH, slope_thresh=SLOPE_THRESH,
              stop_threshold=STOP_THRESHOLD, stop_weight=STOP_WEIGHT):
    """X55: en5/ex2/e55 + 方向确认5天(组合最优)"""
    return build_x51(entry_confirm_days=entry_confirm_days,
                     exit_confirm_days=exit_confirm_days,
                     e5_cooldown_days=e5_cooldown_days,
                     dir_confirm_days=dir_confirm_days,
                     z_thresh=z_thresh, slope_thresh=slope_thresh,
                     n_confirm=N_CONFIRM, stop_threshold=stop_threshold,
                     stop_weight=stop_weight)


# ============================================================
# 主测试流程
# ============================================================
if __name__ == '__main__':
    print("=" * 90)
    print("  X54-X58: 第25轮 极限降低交易次数 + 滑点验证")
    print("  约束: 年化>40% 且 回撤<-35%")
    print("=" * 90)

    results = []

    # 基准对比
    print("\n  [基准] X23:")
    info = test_strategy("X23基准", build_x23_base, desc="基准")
    print_result(info)
    results.append(info)

    print("\n  [第四轮最优] X49(en5/ex2/e53):")
    info = test_strategy("X49(en5/ex2/e53)", build_x43_with_e5,
                         {'entry_confirm_days': 5, 'exit_confirm_days': 2, 'e5_cooldown_days': 3},
                         "第四轮最优")
    print_result(info)
    results.append(info)

    # X54: en6/ex2 + E5冷却
    print("\n" + "=" * 90)
    print("  X54: en6/ex2 + E5冷却(更严格进入)")
    print("=" * 90)
    print(f"  {'E5冷却':>10} {'年化':>8} {'回撤':>8} {'Sharpe':>8} {'Calmar':>8} {'交易':>6} {'满足':>6}")
    print(f"  {'-'*65}")
    for e5 in [0, 3, 5, 7]:
        info = test_strategy(f"X54(en6/ex2/e5{e5})", build_x43_with_e5,
                             {'entry_confirm_days': 6, 'exit_confirm_days': 2,
                              'e5_cooldown_days': e5},
                             f"en6/ex2/E5冷却{e5}")
        m = info
        ok = "✅" if m['ann'] > 0.40 and m['dd'] > -0.35 else "❌"
        print(f"  {e5:>8}天 {m['ann']*100:>7.2f}% {m['dd']*100:>7.2f}% "
              f"{m['sharpe']:>8.3f} {m['calmar']:>8.3f} {m['n_trades']:>6} {ok:>6}")
        results.append(info)

    # X55: en5/ex2/e55 + 方向确认5天(组合)
    print("\n" + "=" * 90)
    print("  X55: en5/ex2/e55 + 方向确认5天(组合X49+X51)")
    print("=" * 90)
    print(f"  {'方向确认':>10} {'年化':>8} {'回撤':>8} {'Sharpe':>8} {'Calmar':>8} {'交易':>6} {'满足':>6}")
    print(f"  {'-'*65}")
    for dc in [4, 5, 6]:
        info = test_strategy(f"X55(en5/ex2/e55/dc{dc})", build_x55,
                             {'entry_confirm_days': 5, 'exit_confirm_days': 2,
                              'e5_cooldown_days': 5, 'dir_confirm_days': dc},
                             f"en5/ex2/e55/dc{dc}")
        m = info
        ok = "✅" if m['ann'] > 0.40 and m['dd'] > -0.35 else "❌"
        print(f"  {dc:>8}天 {m['ann']*100:>7.2f}% {m['dd']*100:>7.2f}% "
              f"{m['sharpe']:>8.3f} {m['calmar']:>8.3f} {m['n_trades']:>6} {ok:>6}")
        results.append(info)

    # X56: en5/ex3 + E5冷却(更严格退出)
    print("\n" + "=" * 90)
    print("  X56: en5/ex3 + E5冷却(更严格退出)")
    print("=" * 90)
    print(f"  {'E5冷却':>10} {'年化':>8} {'回撤':>8} {'Sharpe':>8} {'Calmar':>8} {'交易':>6} {'满足':>6}")
    print(f"  {'-'*65}")
    for e5 in [0, 3, 5, 7]:
        info = test_strategy(f"X56(en5/ex3/e5{e5})", build_x43_with_e5,
                             {'entry_confirm_days': 5, 'exit_confirm_days': 3,
                              'e5_cooldown_days': e5},
                             f"en5/ex3/E5冷却{e5}")
        m = info
        ok = "✅" if m['ann'] > 0.40 and m['dd'] > -0.35 else "❌"
        print(f"  {e5:>8}天 {m['ann']*100:>7.2f}% {m['dd']*100:>7.2f}% "
              f"{m['sharpe']:>8.3f} {m['calmar']:>8.3f} {m['n_trades']:>6} {ok:>6}")
        results.append(info)

    # X57: 滑点验证(前5名方案)
    print("\n" + "=" * 90)
    print("  X57: 5/万滑点验证(满足约束的前5名)")
    print("=" * 90)
    feasible = [r for r in results if r['ann'] > 0.40 and r['dd'] > -0.35]
    feasible.sort(key=lambda x: x['n_trades'])
    top5 = feasible[:5]
    print(f"  {'版本':<35} {'无滑点年化':>10} {'滑点年化':>10} {'滑点Calmar':>10} {'交易':>6}")
    print(f"  {'-'*75}")
    for r in top5:
        # 根据版本名重建信号
        sig, wt = None, None
        name = r['name']
        if 'en6' in name:
            en = 6
        elif 'en5' in name:
            en = 5
        elif 'en4' in name:
            en = 4
        else:
            en = 4

        if 'ex3' in name:
            ex = 3
        else:
            ex = 2

        if 'e53' in name:
            e5 = 3
        elif 'e55' in name:
            e5 = 5
        elif 'e50' in name:
            e5 = 0
        else:
            e5 = 5

        if 'dc' in name and 'X55' in name:
            # 组合方案
            dc = 5
            sig, wt = build_x55(entry_confirm_days=en, exit_confirm_days=ex,
                                e5_cooldown_days=e5, dir_confirm_days=dc)
        else:
            sig, wt = build_x43_with_e5(entry_confirm_days=en, exit_confirm_days=ex,
                                         e5_cooldown_days=e5)

        if sig is not None:
            res_sl = run_backtest(sig, wt, impact_slippage=0.0005)
            m_sl = calc_metrics(res_sl)
            print(f"  {name:<35} {r['ann']*100:>9.2f}% {m_sl['ann']*100:>9.2f}% "
                  f"{m_sl['calmar']:>10.3f} {r['n_trades']:>6}")

    # 汇总
    print("\n" + "=" * 90)
    print("  【汇总】满足约束(年化>40%且回撤<-35%)的方案,按交易次数升序")
    print("=" * 90)
    feasible = [r for r in results if r['ann'] > 0.40 and r['dd'] > -0.35]
    feasible.sort(key=lambda x: x['n_trades'])
    print(f"  {'版本':<35} {'年化':>8} {'回撤':>8} {'Sharpe':>8} {'Calmar':>8} {'交易':>6}")
    print(f"  {'-'*80}")
    for r in feasible[:15]:
        print(f"  {r['name']:<35} {r['ann']*100:>7.2f}% {r['dd']*100:>7.2f}% "
              f"{r['sharpe']:>8.3f} {r['calmar']:>8.3f} {r['n_trades']:>6}")

    if feasible:
        best = feasible[0]
        print(f"\n  ★最优方案(交易最少): {best['name']}")
        print(f"    年化={best['ann']*100:.2f}% 回撤={best['dd']*100:.2f}% "
              f"Sharpe={best['sharpe']:.3f} Calmar={best['calmar']:.3f} 交易={best['n_trades']}次")

    print("\n  完成!")
