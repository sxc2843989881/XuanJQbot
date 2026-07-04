"""run_x58_x62.py — 第26轮: 极限尝试 + 最终确认
================================================================
第五轮最优:
- X54(en6/ex2/e53): 228次/41.00%/-31.24%(交易最少,年化41%安全)
- X54(en6/ex2/e55): 230次/41.39%/-31.02%(年化最高)
- X55(en5/ex2/e55/dc5): 231次/40.77%/-30.36%(回撤最浅)

滑点验证: X54(en6/ex2/e53)滑点后年化38.86%(低于40%)

第六轮目标:
1. 尝试en7/ex2(极限进入确认)
2. 组合X54+X55(en6/ex2/e53+方向确认5天)
3. 确定最终最优方案并记录
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
# 主测试流程
# ============================================================
if __name__ == '__main__':
    print("=" * 90)
    print("  X58-X62: 第26轮 极限尝试 + 最终确认")
    print("  约束: 年化>40% 且 回撤<-35%")
    print("=" * 90)

    results = []

    # 基准对比
    print("\n  [基准] X23:")
    info = test_strategy("X23基准", build_x23_base, desc="基准")
    print_result(info)
    results.append(info)

    print("\n  [第五轮最优] X54(en6/ex2/e53):")
    info = test_strategy("X54(en6/ex2/e53)", build_x43_with_e5,
                         {'entry_confirm_days': 6, 'exit_confirm_days': 2, 'e5_cooldown_days': 3},
                         "第五轮最优")
    print_result(info)
    results.append(info)

    # X58: en7/ex2 + E5冷却(极限进入)
    print("\n" + "=" * 90)
    print("  X58: en7/ex2 + E5冷却(极限进入确认)")
    print("=" * 90)
    print(f"  {'E5冷却':>10} {'年化':>8} {'回撤':>8} {'Sharpe':>8} {'Calmar':>8} {'交易':>6} {'满足':>6}")
    print(f"  {'-'*65}")
    for e5 in [0, 3, 5, 7]:
        info = test_strategy(f"X58(en7/ex2/e5{e5})", build_x43_with_e5,
                             {'entry_confirm_days': 7, 'exit_confirm_days': 2,
                              'e5_cooldown_days': e5},
                             f"en7/ex2/E5冷却{e5}")
        m = info
        ok = "✅" if m['ann'] > 0.40 and m['dd'] > -0.35 else "❌"
        print(f"  {e5:>8}天 {m['ann']*100:>7.2f}% {m['dd']*100:>7.2f}% "
              f"{m['sharpe']:>8.3f} {m['calmar']:>8.3f} {m['n_trades']:>6} {ok:>6}")
        results.append(info)

    # X59: en8/ex2 + E5冷却(更极限)
    print("\n" + "=" * 90)
    print("  X59: en8/ex2 + E5冷却(更极限进入)")
    print("=" * 90)
    print(f"  {'E5冷却':>10} {'年化':>8} {'回撤':>8} {'Sharpe':>8} {'Calmar':>8} {'交易':>6} {'满足':>6}")
    print(f"  {'-'*65}")
    for e5 in [0, 3, 5]:
        info = test_strategy(f"X59(en8/ex2/e5{e5})", build_x43_with_e5,
                             {'entry_confirm_days': 8, 'exit_confirm_days': 2,
                              'e5_cooldown_days': e5},
                             f"en8/ex2/E5冷却{e5}")
        m = info
        ok = "✅" if m['ann'] > 0.40 and m['dd'] > -0.35 else "❌"
        print(f"  {e5:>8}天 {m['ann']*100:>7.2f}% {m['dd']*100:>7.2f}% "
              f"{m['sharpe']:>8.3f} {m['calmar']:>8.3f} {m['n_trades']:>6} {ok:>6}")
        results.append(info)

    # X60: en6/ex2/e53 + 方向确认5天(组合)
    print("\n" + "=" * 90)
    print("  X60: en6/ex2/e53 + 方向确认(组合X54+X51)")
    print("=" * 90)
    print(f"  {'方向确认':>10} {'年化':>8} {'回撤':>8} {'Sharpe':>8} {'Calmar':>8} {'交易':>6} {'满足':>6}")
    print(f"  {'-'*65}")
    for dc in [4, 5, 6]:
        info = test_strategy(f"X60(en6/ex2/e53/dc{dc})", build_x51,
                             {'entry_confirm_days': 6, 'exit_confirm_days': 2,
                              'e5_cooldown_days': 3, 'dir_confirm_days': dc},
                             f"en6/ex2/e53/dc{dc}")
        m = info
        ok = "✅" if m['ann'] > 0.40 and m['dd'] > -0.35 else "❌"
        print(f"  {dc:>8}天 {m['ann']*100:>7.2f}% {m['dd']*100:>7.2f}% "
              f"{m['sharpe']:>8.3f} {m['calmar']:>8.3f} {m['n_trades']:>6} {ok:>6}")
        results.append(info)

    # X61: en6/ex2/e55 + 方向确认5天(组合)
    print("\n" + "=" * 90)
    print("  X61: en6/ex2/e55 + 方向确认(组合)")
    print("=" * 90)
    print(f"  {'方向确认':>10} {'年化':>8} {'回撤':>8} {'Sharpe':>8} {'Calmar':>8} {'交易':>6} {'满足':>6}")
    print(f"  {'-'*65}")
    for dc in [4, 5, 6]:
        info = test_strategy(f"X61(en6/ex2/e55/dc{dc})", build_x51,
                             {'entry_confirm_days': 6, 'exit_confirm_days': 2,
                              'e5_cooldown_days': 5, 'dir_confirm_days': dc},
                             f"en6/ex2/e55/dc{dc}")
        m = info
        ok = "✅" if m['ann'] > 0.40 and m['dd'] > -0.35 else "❌"
        print(f"  {dc:>8}天 {m['ann']*100:>7.2f}% {m['dd']*100:>7.2f}% "
              f"{m['sharpe']:>8.3f} {m['calmar']:>8.3f} {m['n_trades']:>6} {ok:>6}")
        results.append(info)

    # X62: 滑点验证(前5名)
    print("\n" + "=" * 90)
    print("  X62: 5/万滑点验证(满足约束的前5名)")
    print("=" * 90)
    feasible = [r for r in results if r['ann'] > 0.40 and r['dd'] > -0.35]
    feasible.sort(key=lambda x: x['n_trades'])
    top5 = feasible[:5]
    print(f"  {'版本':<40} {'无滑点':>8} {'滑点年化':>10} {'滑点Calmar':>10} {'交易':>6}")
    print(f"  {'-'*80}")
    for r in top5:
        name = r['name']
        # 解析参数
        en = 6
        ex = 2
        e5 = 3
        dc = 0
        if 'en7' in name: en = 7
        elif 'en8' in name: en = 8
        elif 'en6' in name: en = 6
        elif 'en5' in name: en = 5

        if 'ex3' in name: ex = 3

        if 'e55' in name: e5 = 5
        elif 'e53' in name: e5 = 3
        elif 'e50' in name: e5 = 0

        if 'dc5' in name: dc = 5
        elif 'dc4' in name: dc = 4
        elif 'dc6' in name: dc = 6

        if dc > 0:
            sig, wt = build_x51(entry_confirm_days=en, exit_confirm_days=ex,
                                e5_cooldown_days=e5, dir_confirm_days=dc)
        else:
            sig, wt = build_x43_with_e5(entry_confirm_days=en, exit_confirm_days=ex,
                                         e5_cooldown_days=e5)

        if sig is not None:
            res_sl = run_backtest(sig, wt, impact_slippage=0.0005)
            m_sl = calc_metrics(res_sl)
            print(f"  {name:<40} {r['ann']*100:>7.2f}% {m_sl['ann']*100:>9.2f}% "
                  f"{m_sl['calmar']:>10.3f} {r['n_trades']:>6}")

    # 汇总
    print("\n" + "=" * 90)
    print("  【汇总】满足约束(年化>40%且回撤<-35%)的方案,按交易次数升序")
    print("=" * 90)
    feasible = [r for r in results if r['ann'] > 0.40 and r['dd'] > -0.35]
    feasible.sort(key=lambda x: x['n_trades'])
    print(f"  {'版本':<40} {'年化':>8} {'回撤':>8} {'Sharpe':>8} {'Calmar':>8} {'交易':>6}")
    print(f"  {'-'*85}")
    for r in feasible[:20]:
        print(f"  {r['name']:<40} {r['ann']*100:>7.2f}% {r['dd']*100:>7.2f}% "
              f"{r['sharpe']:>8.3f} {r['calmar']:>8.3f} {r['n_trades']:>6}")

    if feasible:
        best = feasible[0]
        print(f"\n  ★最终最优方案(交易最少): {best['name']}")
        print(f"    年化={best['ann']*100:.2f}% 回撤={best['dd']*100:.2f}% "
              f"Sharpe={best['sharpe']:.3f} Calmar={best['calmar']:.3f} 交易={best['n_trades']}次")
        print(f"    相比X23基准: 交易减少{362-best['n_trades']}次({(362-best['n_trades'])/362*100:.1f}%)")
        print(f"                  年化变化{(best['ann']-0.435)*100:+.2f}pp 回撤变化{best['dd']*100-(-25.85):+.2f}pp")

    print("\n  完成!")
