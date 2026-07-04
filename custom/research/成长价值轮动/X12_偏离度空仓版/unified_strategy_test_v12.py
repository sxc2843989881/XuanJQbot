"""unified_strategy_test_v12.py — 第十二轮: sw甜点细调 + 极限低仓位探索
================================================================
第十一轮最优: sw=0.15让Calmar达2.048
第十二轮: 在sw=0.05-0.15细调, 探索更低仓位的极限
================================================================
"""
import sys
sys.path.insert(0, r'c:\XuanJLH\Qbot\custom\backtests')
sys.path.insert(0, r'c:\XuanJLH\Qbot\custom\research\成长价值轮动\X12_偏离度空仓版')

from pathlib import Path
import numpy as np
import pandas as pd
from optimize_runner import (
    G_CLOSE, V_CLOSE, RATIO, RATIO_MA20, RATIO_DEV, MA20_SLOPE,
    V_MOM20, G_DD20, V_DD20, BASE_DIR,
    run_backtest, calc_metrics, count_switches,
)
from run_x33_reduce_trades import RATIO_DEV_STD20, RATIO_DEV_Z
from unified_strategy_test_v9 import build_v9
from unified_strategy_test_v10 import build_v10_multi_period_t

T = RATIO_DEV_Z
SLOPE = MA20_SLOPE


def test_strategy(name, sig, wt, desc=""):
    result = run_backtest(sig, wt)
    m = calc_metrics(result)
    sw = count_switches(sig, wt)
    result_sl = run_backtest(sig, wt, impact_slippage=0.0005)
    m_sl = calc_metrics(result_sl)
    return {
        'name': name, 'desc': desc,
        'ann': m['ann'], 'dd': m['dd'], 'sharpe': m['sharpe'],
        'calmar': m['calmar'], 'n_trades': m['n_trades'],
        'dir_sw': sw['dir'], 'cash_sw': sw['cash'],
        'ann_sl': m_sl['ann'], 'calmar_sl': m_sl['calmar'],
    }


def print_info(info):
    print(f"  {info['name']}: {info['desc']}")
    print(f"    年化={info['ann']*100:.2f}% 回撤={info['dd']*100:.2f}% "
          f"Sharpe={info['sharpe']:.3f} Calmar={info['calmar']:.3f} "
          f"交易={info['n_trades']}(方向{info['dir_sw']}+仓位{info['cash_sw']})")
    print(f"    5/万滑点: 年化={info['ann_sl']*100:.2f}% Calmar={info['calmar_sl']:.3f}")


if __name__ == '__main__':
    print("=" * 80)
    print("  统一策略第十二轮: sw甜点细调 + 极限低仓位探索")
    print("=" * 80)

    results = []

    # ---- 第一组: sw细调 (st=0.088, cd=5, rt=1.3) ----
    print("\n--- 第一组: sw细调 (st=0.088) ---")
    for sw in [0.05, 0.08, 0.10, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.20]:
        sig, wt = build_v9(reduce_thresh=1.3, stop_threshold=0.088,
                           stop_weight=sw, e5_cooldown=5)
        info = test_strategy(f"U62_sw{sw}", sig, wt, f"sw={sw}")
        print_info(info); results.append(info)

    # ---- 第二组: sw+cd细调 (st=0.088) ----
    print("\n--- 第二组: sw+cd细调 (st=0.088) ---")
    for sw, cd in [(0.10, 6), (0.12, 6), (0.13, 6), (0.14, 6),
                    (0.15, 6), (0.10, 4), (0.12, 4), (0.13, 4),
                    (0.15, 4), (0.13, 5), (0.14, 5), (0.12, 5)]:
        sig, wt = build_v9(reduce_thresh=1.3, stop_threshold=0.088,
                           stop_weight=sw, e5_cooldown=cd)
        info = test_strategy(f"U63_sw{sw}_cd{cd}", sig, wt, f"sw={sw} cd={cd}")
        print_info(info); results.append(info)

    # ---- 第三组: sw+st细调 (cd=5) ----
    print("\n--- 第三组: sw+st细调 (cd=5) ---")
    for sw, st in [(0.10, 0.085), (0.10, 0.088), (0.10, 0.090),
                    (0.12, 0.085), (0.12, 0.088), (0.12, 0.090),
                    (0.13, 0.085), (0.13, 0.088), (0.13, 0.090),
                    (0.14, 0.088), (0.15, 0.085), (0.15, 0.090)]:
        sig, wt = build_v9(reduce_thresh=1.3, stop_threshold=st,
                           stop_weight=sw, e5_cooldown=5)
        info = test_strategy(f"U64_sw{sw}_st{st}", sig, wt, f"sw={sw} st={st}")
        print_info(info); results.append(info)

    # ---- 第四组: rt+sw细调 (st=0.088, cd=5) ----
    print("\n--- 第四组: rt+sw细调 (st=0.088) ---")
    for rt, sw in [(1.28, 0.13), (1.30, 0.13), (1.32, 0.13),
                    (1.28, 0.15), (1.30, 0.15), (1.32, 0.15),
                    (1.25, 0.13), (1.35, 0.13), (1.28, 0.12),
                    (1.30, 0.14), (1.32, 0.14), (1.28, 0.14)]:
        sig, wt = build_v9(reduce_thresh=rt, stop_threshold=0.088,
                           stop_weight=sw, e5_cooldown=5)
        info = test_strategy(f"U65_rt{rt}_sw{sw}", sig, wt, f"rt={rt} sw={sw}")
        print_info(info); results.append(info)

    # ---- 第五组: 四参数联合 (rt+sw+cd+st) ----
    print("\n--- 第五组: 四参数联合 ---")
    for rt, sw, cd, st in [(1.30, 0.13, 5, 0.088), (1.30, 0.13, 6, 0.088),
                            (1.30, 0.14, 5, 0.088), (1.30, 0.14, 6, 0.088),
                            (1.28, 0.13, 5, 0.088), (1.28, 0.13, 6, 0.088),
                            (1.32, 0.13, 5, 0.088), (1.32, 0.13, 6, 0.088),
                            (1.30, 0.13, 5, 0.085), (1.30, 0.13, 5, 0.090),
                            (1.30, 0.12, 5, 0.088), (1.30, 0.15, 6, 0.090)]:
        sig, wt = build_v9(reduce_thresh=rt, stop_threshold=st,
                           stop_weight=sw, e5_cooldown=cd)
        info = test_strategy(f"U66_rt{rt}_sw{sw}_cd{cd}_st{st}", sig, wt,
                              f"rt={rt} sw={sw} cd={cd} st={st}")
        print_info(info); results.append(info)

    # ---- 第六组: 最终最优候选 ----
    print("\n--- 第六组: 最终最优候选 ---")
    final_configs = [
        (1.30, 0.002, 0.0, 5, 5, 0.088, 0.13, 5, "U67_v1"),
        (1.30, 0.002, 0.0, 5, 5, 0.088, 0.14, 5, "U67_v2"),
        (1.30, 0.002, 0.0, 5, 5, 0.088, 0.15, 5, "U67_v3"),
        (1.30, 0.002, 0.0, 5, 5, 0.088, 0.12, 5, "U67_v4"),
        (1.28, 0.002, 0.0, 5, 5, 0.088, 0.13, 5, "U67_v5"),
        (1.32, 0.002, 0.0, 5, 5, 0.088, 0.13, 5, "U67_v6"),
        (1.30, 0.002, 0.0, 5, 6, 0.088, 0.13, 6, "U67_v7"),
        (1.30, 0.002, 0.0, 5, 5, 0.085, 0.13, 5, "U67_v8"),
        (1.30, 0.002, 0.0, 5, 5, 0.090, 0.13, 5, "U67_v9"),
        (1.30, 0.002, 0.0, 5, 5, 0.088, 0.13, 6, "U67_v10"),
    ]
    for rt, sl, rw, dc, dcd, st, sw, e5c, name in final_configs:
        sig, wt = build_v9(reduce_thresh=rt, slope_thresh=sl, reduce_weight=rw,
                           dir_confirm=dc, dir_cooldown=dcd,
                           stop_threshold=st, stop_weight=sw, e5_cooldown=e5c)
        info = test_strategy(name, sig, wt, name.replace("U67_", ""))
        print_info(info); results.append(info)

    # ---- 汇总 ----
    print("\n" + "=" * 80)
    print("  最终汇总 (满足约束: 年化>40% 回撤<-35%)")
    print("=" * 80)
    ok = [r for r in results if r['ann'] > 0.40 and r['dd'] > -0.35]
    ok.sort(key=lambda x: -x['calmar'])
    print(f"\n  满足约束: {len(ok)}个版本\n")
    print(f"  {'名称':<32} {'年化':>7} {'回撤':>7} {'Sharpe':>7} {'Calmar':>7} "
          f"{'交易':>5} {'滑点年化':>8}")
    print("  " + "-" * 84)
    print(f"  {'X61(基准)':<32} {'41.12%':>7} {'-30.21%':>7} {'1.357':>7} "
          f"{'1.361':>7} {'224':>5} {'39.04%':>8}")
    print(f"  {'U57(v11最优)':<32} {'44.92%':>7} {'-21.93%':>7} {'1.509':>7} "
          f"{'2.048':>7} {'365':>5} {'42.35%':>8}")
    for r in ok[:25]:
        print(f"  {r['name']:<32} {r['ann']*100:>6.2f}% {r['dd']*100:>6.2f}% "
              f"{r['sharpe']:>7.3f} {r['calmar']:>7.3f} {r['n_trades']:>5} "
              f"{r['ann_sl']*100:>7.2f}%")

    print(f"\n  Top 15 Calmar:")
    results.sort(key=lambda x: -x['calmar'])
    for r in results[:15]:
        print(f"  {r['name']:<32} 年化{r['ann']*100:.2f}% 回撤{r['dd']*100:.2f}% "
              f"Calmar{r['calmar']:.3f} 交易{r['n_trades']}次 滑点{r['ann_sl']*100:.2f}%")

    best = max(results, key=lambda x: x['calmar'])
    print(f"\n  ★最优: {best['name']} Calmar={best['calmar']:.3f} "
          f"年化={best['ann']*100:.2f}% 回撤={best['dd']*100:.2f}%")
