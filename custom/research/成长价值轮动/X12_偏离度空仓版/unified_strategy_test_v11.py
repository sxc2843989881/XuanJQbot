"""unified_strategy_test_v11.py — 第十一轮: Calmar 2.0+ 三参数联合优化
================================================================
第十轮突破: sw=0.2让Calmar达2.004, cd=6让Calmar达2.002
第十一轮: 组合sw+cd+st三参数, 寻找Calmar最高点
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
    print("  统一策略第十一轮: Calmar 2.0+ 三参数联合优化")
    print("=" * 80)

    results = []

    # ---- 第一组: sw+cd联合扫描 (st=0.09, rt=1.3) ----
    print("\n--- 第一组: sw+cd联合扫描 (st=0.09) ---")
    for sw, cd in [(0.20, 5), (0.20, 6), (0.20, 7), (0.15, 5),
                    (0.15, 6), (0.25, 5), (0.25, 6), (0.20, 4),
                    (0.20, 8), (0.10, 5), (0.30, 6), (0.18, 5),
                    (0.22, 5), (0.20, 5)]:
        sig, wt = build_v9(reduce_thresh=1.3, stop_threshold=0.09,
                           stop_weight=sw, e5_cooldown=cd)
        info = test_strategy(f"U56_sw{sw}_cd{cd}", sig, wt, f"sw={sw} cd={cd}")
        print_info(info); results.append(info)

    # ---- 第二组: sw+st联合扫描 (cd=5, rt=1.3) ----
    print("\n--- 第二组: sw+st联合扫描 (cd=5) ---")
    for sw, st in [(0.20, 0.085), (0.20, 0.088), (0.20, 0.090), (0.20, 0.092),
                    (0.15, 0.088), (0.15, 0.090), (0.25, 0.088), (0.25, 0.090),
                    (0.18, 0.088), (0.22, 0.088), (0.20, 0.080), (0.20, 0.095)]:
        sig, wt = build_v9(reduce_thresh=1.3, stop_threshold=st,
                           stop_weight=sw, e5_cooldown=5)
        info = test_strategy(f"U57_sw{sw}_st{st}", sig, wt, f"sw={sw} st={st}")
        print_info(info); results.append(info)

    # ---- 第三组: 三参数联合 (sw+cd+st) ----
    print("\n--- 第三组: 三参数联合(sw+cd+st) ---")
    for sw, cd, st in [(0.20, 6, 0.088), (0.20, 6, 0.085), (0.20, 6, 0.090),
                        (0.18, 6, 0.088), (0.22, 6, 0.088), (0.20, 5, 0.088),
                        (0.20, 7, 0.088), (0.15, 6, 0.088), (0.20, 6, 0.082),
                        (0.20, 6, 0.092), (0.18, 5, 0.088), (0.22, 6, 0.090)]:
        sig, wt = build_v9(reduce_thresh=1.3, stop_threshold=st,
                           stop_weight=sw, e5_cooldown=cd)
        info = test_strategy(f"U58_sw{sw}_cd{cd}_st{st}", sig, wt,
                              f"sw={sw} cd={cd} st={st}")
        print_info(info); results.append(info)

    # ---- 第四组: rt+sw联合 (cd=5, st=0.09) ----
    print("\n--- 第四组: rt+sw联合 (cd=5, st=0.09) ---")
    for rt, sw in [(1.30, 0.20), (1.28, 0.20), (1.32, 0.20), (1.25, 0.20),
                    (1.35, 0.20), (1.30, 0.15), (1.30, 0.25), (1.28, 0.15),
                    (1.32, 0.15), (1.30, 0.18), (1.30, 0.22), (1.28, 0.18)]:
        sig, wt = build_v9(reduce_thresh=rt, stop_threshold=0.09,
                           stop_weight=sw, e5_cooldown=5)
        info = test_strategy(f"U59_rt{rt}_sw{sw}", sig, wt, f"rt={rt} sw={sw}")
        print_info(info); results.append(info)

    # ---- 第五组: 多周期T + sw=0.2 ----
    print("\n--- 第五组: 多周期T + sw=0.2 ---")
    for t_s, t_l, w_s, sw in [(5, 20, 0.3, 0.20), (5, 20, 0.4, 0.20),
                                (5, 20, 0.3, 0.15), (5, 20, 0.3, 0.25),
                                (10, 20, 0.5, 0.20), (5, 20, 0.2, 0.20),
                                (5, 20, 0.3, 0.18), (5, 20, 0.3, 0.22)]:
        sig, wt = build_v10_multi_period_t(t_short=t_s, t_long=t_l,
                                            weights=(w_s, 1-w_s),
                                            reduce_thresh=1.3, stop_threshold=0.09,
                                            stop_weight=sw, e5_cooldown=5)
        info = test_strategy(f"U60_T{t_s}_{t_l}_w{w_s}_sw{sw}", sig, wt,
                              f"T{t_s}/{t_l} w{w_s} sw={sw}")
        print_info(info); results.append(info)

    # ---- 第六组: 最终最优组合验证 ----
    print("\n--- 第六组: 最终最优组合验证 ---")
    final_configs = [
        (1.3, 0.002, 0.0, 5, 5, 0.088, 0.20, 5, "U61_最终_v1"),
        (1.3, 0.002, 0.0, 5, 6, 0.088, 0.20, 6, "U61_最终_v2"),
        (1.3, 0.002, 0.0, 5, 5, 0.088, 0.18, 5, "U61_最终_v3"),
        (1.3, 0.002, 0.0, 5, 5, 0.085, 0.20, 5, "U61_最终_v4"),
        (1.3, 0.002, 0.0, 5, 6, 0.090, 0.20, 6, "U61_最终_v5"),
        (1.3, 0.002, 0.0, 5, 5, 0.090, 0.20, 6, "U61_最终_v6"),
        (1.28, 0.002, 0.0, 5, 5, 0.088, 0.20, 5, "U61_最终_v7"),
        (1.3, 0.002, 0.0, 5, 6, 0.085, 0.20, 6, "U61_最终_v8"),
        (1.3, 0.002, 0.0, 5, 5, 0.088, 0.22, 5, "U61_最终_v9"),
        (1.3, 0.002, 0.0, 5, 6, 0.088, 0.18, 6, "U61_最终_v10"),
    ]
    for rt, sl, rw, dc, dcd, st, sw, e5c, name in final_configs:
        sig, wt = build_v9(reduce_thresh=rt, slope_thresh=sl, reduce_weight=rw,
                           dir_confirm=dc, dir_cooldown=dcd,
                           stop_threshold=st, stop_weight=sw, e5_cooldown=e5c)
        info = test_strategy(name, sig, wt, name.replace("U61_最终_", ""))
        print_info(info); results.append(info)

    # ---- 汇总 ----
    print("\n" + "=" * 80)
    print("  最终汇总 (满足约束: 年化>40% 回撤<-35%)")
    print("=" * 80)
    ok = [r for r in results if r['ann'] > 0.40 and r['dd'] > -0.35]
    ok.sort(key=lambda x: -x['calmar'])
    print(f"\n  满足约束: {len(ok)}个版本\n")
    print(f"  {'名称':<28} {'年化':>7} {'回撤':>7} {'Sharpe':>7} {'Calmar':>7} "
          f"{'交易':>5} {'滑点年化':>8}")
    print("  " + "-" * 80)
    print(f"  {'X61(基准)':<28} {'41.12%':>7} {'-30.21%':>7} {'1.357':>7} "
          f"{'1.361':>7} {'224':>5} {'39.04%':>8}")
    print(f"  {'U52(v10最优)':<28} {'44.73%':>7} {'-22.32%':>7} {'1.500':>7} "
          f"{'2.004':>7} {'364':>5} {'42.11%':>8}")
    for r in ok[:25]:
        print(f"  {r['name']:<28} {r['ann']*100:>6.2f}% {r['dd']*100:>6.2f}% "
              f"{r['sharpe']:>7.3f} {r['calmar']:>7.3f} {r['n_trades']:>5} "
              f"{r['ann_sl']*100:>7.2f}%")

    print(f"\n  Top 15 Calmar:")
    results.sort(key=lambda x: -x['calmar'])
    for r in results[:15]:
        print(f"  {r['name']:<28} 年化{r['ann']*100:.2f}% 回撤{r['dd']*100:.2f}% "
              f"Calmar{r['calmar']:.3f} 交易{r['n_trades']}次 滑点{r['ann_sl']*100:.2f}%")

    # 保存最优结果
    best = max(results, key=lambda x: x['calmar'])
    print(f"\n  ★最优: {best['name']} Calmar={best['calmar']:.3f} "
          f"年化={best['ann']*100:.2f}% 回撤={best['dd']*100:.2f}%")
