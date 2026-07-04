"""unified_strategy_test_v15.py — 第十五轮: cd甜点+四参数联合
================================================================
第十四轮突破: cd=8让Calmar达2.164(年化45.60%/回撤-21.08%)
第十五轮: 在cd=6-12细调, 联合sw+st+cd四参数
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
from unified_strategy_test_v13 import build_v13_b2_improved

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
    print("  统一策略第十五轮: cd甜点+四参数联合")
    print("=" * 80)

    results = []

    # ---- 第一组: cd细调 (B2改进, sw=0.15, st=0.088) ----
    print("\n--- 第一组: cd细调 (B2改进, sw=0.15, st=0.088) ---")
    for cd in [5, 6, 7, 8, 9, 10, 11, 12, 15]:
        sig, wt = build_v13_b2_improved(mom_short=10, mom_long=20,
                                         stop_weight=0.15, stop_threshold=0.088,
                                         e5_cooldown=cd)
        info = test_strategy(f"U81_cd{cd}", sig, wt, f"cd={cd}")
        print_info(info); results.append(info)

    # ---- 第二组: cd+sw联合 (B2改进, st=0.088) ----
    print("\n--- 第二组: cd+sw联合 (st=0.088) ---")
    for cd, sw in [(8, 0.13), (8, 0.14), (8, 0.15), (8, 0.16),
                    (6, 0.13), (6, 0.14), (6, 0.15),
                    (10, 0.13), (10, 0.15), (9, 0.14)]:
        sig, wt = build_v13_b2_improved(mom_short=10, mom_long=20,
                                         stop_weight=sw, stop_threshold=0.088,
                                         e5_cooldown=cd)
        info = test_strategy(f"U82_cd{cd}_sw{sw}", sig, wt, f"cd={cd} sw={sw}")
        print_info(info); results.append(info)

    # ---- 第三组: cd+st联合 (B2改进, sw=0.15) ----
    print("\n--- 第三组: cd+st联合 (sw=0.15) ---")
    for cd, st in [(8, 0.085), (8, 0.088), (8, 0.090), (8, 0.092),
                    (6, 0.088), (6, 0.092), (10, 0.088), (10, 0.092),
                    (9, 0.088), (9, 0.092)]:
        sig, wt = build_v13_b2_improved(mom_short=10, mom_long=20,
                                         stop_weight=0.15, stop_threshold=st,
                                         e5_cooldown=cd)
        info = test_strategy(f"U83_cd{cd}_st{st}", sig, wt, f"cd={cd} st={st}")
        print_info(info); results.append(info)

    # ---- 第四组: 四参数联合(cd+sw+st) ----
    print("\n--- 第四组: 四参数联合 ---")
    for cd, sw, st in [(8, 0.13, 0.088), (8, 0.14, 0.088), (8, 0.15, 0.092),
                        (8, 0.13, 0.092), (8, 0.14, 0.092), (6, 0.13, 0.088),
                        (6, 0.14, 0.088), (10, 0.13, 0.088), (10, 0.14, 0.088),
                        (8, 0.13, 0.090), (8, 0.15, 0.090), (9, 0.13, 0.088)]:
        sig, wt = build_v13_b2_improved(mom_short=10, mom_long=20,
                                         stop_weight=sw, stop_threshold=st,
                                         e5_cooldown=cd)
        info = test_strategy(f"U84_cd{cd}_sw{sw}_st{st}", sig, wt,
                              f"cd={cd} sw={sw} st={st}")
        print_info(info); results.append(info)

    # ---- 第五组: cd+rt联合 ----
    print("\n--- 第五组: cd+rt联合 (sw=0.15, st=0.088) ---")
    for cd, rt in [(8, 1.28), (8, 1.30), (8, 1.32), (8, 1.35),
                    (6, 1.30), (10, 1.30), (9, 1.30), (8, 1.25)]:
        # 需要修改build_v13_b2_improved支持rt参数, 这里用build_v9替代
        # build_v9不支持B2改进, 这里临时用build_v9
        from unified_strategy_test_v9 import build_v9
        sig, wt = build_v9(reduce_thresh=rt, stop_threshold=0.088,
                           stop_weight=0.15, e5_cooldown=cd)
        info = test_strategy(f"U85_cd{cd}_rt{rt}", sig, wt, f"cd={cd} rt={rt}")
        print_info(info); results.append(info)

    # ---- 第六组: 最终候选 ----
    print("\n--- 第六组: 最终候选 ---")
    final_configs = [
        ("U86_最优_cd8_sw0.13_st0.088", lambda: build_v13_b2_improved(mom_short=10, mom_long=20, stop_weight=0.13, stop_threshold=0.088, e5_cooldown=8)),
        ("U86_最优_cd8_sw0.14_st0.088", lambda: build_v13_b2_improved(mom_short=10, mom_long=20, stop_weight=0.14, stop_threshold=0.088, e5_cooldown=8)),
        ("U86_最优_cd8_sw0.15_st0.088", lambda: build_v13_b2_improved(mom_short=10, mom_long=20, stop_weight=0.15, stop_threshold=0.088, e5_cooldown=8)),
        ("U86_最优_cd8_sw0.13_st0.092", lambda: build_v13_b2_improved(mom_short=10, mom_long=20, stop_weight=0.13, stop_threshold=0.092, e5_cooldown=8)),
        ("U86_最优_cd8_sw0.14_st0.092", lambda: build_v13_b2_improved(mom_short=10, mom_long=20, stop_weight=0.14, stop_threshold=0.092, e5_cooldown=8)),
        ("U86_最优_cd9_sw0.14_st0.088", lambda: build_v13_b2_improved(mom_short=10, mom_long=20, stop_weight=0.14, stop_threshold=0.088, e5_cooldown=9)),
        ("U86_最优_cd10_sw0.13_st0.088", lambda: build_v13_b2_improved(mom_short=10, mom_long=20, stop_weight=0.13, stop_threshold=0.088, e5_cooldown=10)),
        ("U86_最优_cd6_sw0.13_st0.088", lambda: build_v13_b2_improved(mom_short=10, mom_long=20, stop_weight=0.13, stop_threshold=0.088, e5_cooldown=6)),
        ("U86_v9基准_cd8", lambda: build_v9(reduce_thresh=1.3, stop_threshold=0.088, stop_weight=0.15, e5_cooldown=8)),
        ("U86_v9基准_cd6", lambda: build_v9(reduce_thresh=1.3, stop_threshold=0.088, stop_weight=0.15, e5_cooldown=6)),
    ]
    for name, func in final_configs:
        sig, wt = func()
        info = test_strategy(name, sig, wt, name.replace("U86_最优_", "").replace("U86_", ""))
        print_info(info); results.append(info)

    # ---- 汇总 ----
    print("\n" + "=" * 80)
    print("  最终汇总 (满足约束: 年化>40% 回撤<-35%)")
    print("=" * 80)
    ok = [r for r in results if r['ann'] > 0.40 and r['dd'] > -0.35]
    ok.sort(key=lambda x: -x['calmar'])
    print(f"\n  满足约束: {len(ok)}个版本\n")
    print(f"  {'名称':<36} {'年化':>7} {'回撤':>7} {'Sharpe':>7} {'Calmar':>7} "
          f"{'交易':>5} {'滑点年化':>8} {'滑点Calmar':>10}")
    print("  " + "-" * 92)
    print(f"  {'X61(基准)':<36} {'41.12%':>7} {'-30.21%':>7} {'1.357':>7} "
          f"{'1.361':>7} {'224':>5} {'39.04%':>8} {'1.214':>10}")
    print(f"  {'U77(v14最优)':<36} {'45.60%':>7} {'-21.08%':>7} {'1.539':>7} "
          f"{'2.164':>7} {'388':>5} {'43.08%':>8} {'1.847':>10}")
    for r in ok[:25]:
        print(f"  {r['name']:<36} {r['ann']*100:>6.2f}% {r['dd']*100:>6.2f}% "
              f"{r['sharpe']:>7.3f} {r['calmar']:>7.3f} {r['n_trades']:>5} "
              f"{r['ann_sl']*100:>7.2f}% {r['calmar_sl']:>10.3f}")

    print(f"\n  Top 15 Calmar:")
    results.sort(key=lambda x: -x['calmar'])
    for r in results[:15]:
        print(f"  {r['name']:<36} 年化{r['ann']*100:.2f}% 回撤{r['dd']*100:.2f}% "
              f"Calmar{r['calmar']:.3f} 滑点Calmar{r['calmar_sl']:.3f} 交易{r['n_trades']}次")

    best = max(results, key=lambda x: x['calmar'])
    print(f"\n  ★最优: {best['name']} Calmar={best['calmar']:.3f} "
          f"年化={best['ann']*100:.2f}% 回撤={best['dd']*100:.2f}%")

    best_sl = max(results, key=lambda x: x['calmar_sl'])
    print(f"  ★滑点最优: {best_sl['name']} 滑点Calmar={best_sl['calmar_sl']:.3f} "
          f"滑点年化={best_sl['ann_sl']*100:.2f}%")
