"""unified_strategy_test_v16.py — 第十六轮: cd=8甜点深度优化
================================================================
第十五轮最优: cd=8+sw=0.16让Calmar达2.171(年化45.61%/回撤-21.01%)
第十六轮: cd=8基础上的深度优化, 寻找极限
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
    print("  统一策略第十六轮: cd=8甜点深度优化")
    print("=" * 80)

    results = []

    # ---- 第一组: cd=8 + sw细调 ----
    print("\n--- 第一组: cd=8 + sw细调 ---")
    for sw in [0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20, 0.21, 0.22]:
        sig, wt = build_v13_b2_improved(mom_short=10, mom_long=20,
                                         stop_weight=sw, stop_threshold=0.088,
                                         e5_cooldown=8)
        info = test_strategy(f"U87_sw{sw}", sig, wt, f"sw={sw}")
        print_info(info); results.append(info)

    # ---- 第二组: cd=8 + st细调 (sw=0.16) ----
    print("\n--- 第二组: cd=8 + st细调 (sw=0.16) ---")
    for st in [0.080, 0.082, 0.085, 0.086, 0.087, 0.088, 0.089, 0.090, 0.092]:
        sig, wt = build_v13_b2_improved(mom_short=10, mom_long=20,
                                         stop_weight=0.16, stop_threshold=st,
                                         e5_cooldown=8)
        info = test_strategy(f"U88_st{st}", sig, wt, f"st={st}")
        print_info(info); results.append(info)

    # ---- 第三组: cd=8 + B2参数细调 ----
    print("\n--- 第三组: cd=8 + B2参数细调 ---")
    for ms, ml in [(10, 20), (10, 15), (10, 25), (8, 20), (12, 20),
                    (10, 18), (10, 22), (7, 20), (15, 20)]:
        sig, wt = build_v13_b2_improved(mom_short=ms, mom_long=ml,
                                         stop_weight=0.16, stop_threshold=0.088,
                                         e5_cooldown=8)
        info = test_strategy(f"U89_ms{ms}_ml{ml}", sig, wt, f"短{ms}长{ml}")
        print_info(info); results.append(info)

    # ---- 第四组: cd=8 + slope细调 ----
    print("\n--- 第四组: cd=8 + slope细调 ---")
    # 需要修改build_v13_b2_improved支持slope_thresh, 这里临时手动构建
    def build_custom(slope_thresh=0.002, sw=0.16, st=0.088, cd=8,
                     ms=10, ml=20, rt=1.3, dc=5, dcd=5):
        V_MOM_S = V_CLOSE.pct_change(ms)
        V_MOM_L = V_CLOSE.pct_change(ml)
        raw_dir = (T > 0).map({True: 'BULL', False: 'BEAR'})
        mask = np.ones(len(raw_dir), dtype=bool)
        for k in range(1, dc):
            mask = mask & (raw_dir.values == raw_dir.shift(k).values)
        confirmed_dir = raw_dir.where(mask, np.nan).ffill().fillna('BULL')
        if dcd > 0:
            new_dir = confirmed_dir.copy()
            last_switch = -dcd - 1
            prev = confirmed_dir.iloc[0]
            for i in range(len(confirmed_dir)):
                if pd.isna(confirmed_dir.iloc[i]):
                    new_dir.iloc[i] = prev
                    continue
                if confirmed_dir.iloc[i] != prev:
                    if i - last_switch >= dcd:
                        last_switch = i
                        prev = confirmed_dir.iloc[i]
                    new_dir.iloc[i] = prev
                else:
                    new_dir.iloc[i] = prev
            confirmed_dir = new_dir
        dir_raw = confirmed_dir
        weak_slope = SLOPE.abs() < slope_thresh
        weak_t = T.abs() < rt
        is_weak = weak_t & weak_slope
        wt = pd.Series(1.0, index=T.index)
        wt[is_weak] = 0.0
        wrong_value = (dir_raw == 'BEAR') & (V_MOM_S <= 0) & (V_MOM_L <= 0)
        dir_raw[wrong_value] = 'BULL'
        dir_s = dir_raw.map({'BULL': 'growth', 'BEAR': 'value'})
        gs = (dir_s == 'growth') & (G_DD20 < -st)
        vs = (dir_s == 'value') & (V_DD20 < -st)
        e5_trigger = gs | vs
        in_cooldown = False
        cooldown_count = 0
        for i in range(len(wt)):
            if pd.isna(wt.iloc[i]) or pd.isna(dir_s.iloc[i]):
                continue
            if e5_trigger.iloc[i] and not in_cooldown:
                in_cooldown = True
                cooldown_count = 0
                wt.iloc[i] = wt.iloc[i] * sw
            elif in_cooldown:
                cooldown_count += 1
                if cooldown_count >= cd:
                    if e5_trigger.iloc[i]:
                        cooldown_count = 0
                        wt.iloc[i] = wt.iloc[i] * sw
                    else:
                        in_cooldown = False
                        if is_weak.iloc[i]:
                            wt.iloc[i] = 0.0
                        else:
                            wt.iloc[i] = 1.0
                else:
                    if wt.iloc[i] > 0:
                        wt.iloc[i] = sw
        signal = dir_raw.map({'BULL': 'growth', 'BEAR': 'value'})
        return signal, wt

    for sl in [0.0015, 0.0018, 0.002, 0.0022, 0.0025, 0.003]:
        sig, wt = build_custom(slope_thresh=sl)
        info = test_strategy(f"U90_sl{sl}", sig, wt, f"slope={sl}")
        print_info(info); results.append(info)

    # ---- 第五组: cd=8 + rt细调 ----
    print("\n--- 第五组: cd=8 + rt细调 ---")
    for rt in [1.25, 1.28, 1.30, 1.32, 1.35, 1.40]:
        sig, wt = build_custom(rt=rt)
        info = test_strategy(f"U91_rt{rt}", sig, wt, f"rt={rt}")
        print_info(info); results.append(info)

    # ---- 第六组: cd=8 + dc/dcd细调 ----
    print("\n--- 第六组: cd=8 + dc/dcd细调 ---")
    for dc, dcd in [(5, 5), (4, 5), (5, 4), (5, 6), (4, 4), (6, 5), (5, 3), (3, 5)]:
        sig, wt = build_custom(dc=dc, dcd=dcd)
        info = test_strategy(f"U92_dc{dc}_dcd{dcd}", sig, wt, f"dc={dc} dcd={dcd}")
        print_info(info); results.append(info)

    # ---- 第七组: 最终最优候选 ----
    print("\n--- 第七组: 最终最优候选 ---")
    final_configs = [
        ("U93_最终_sw0.16_st0.088_cd8", lambda: build_v13_b2_improved(mom_short=10, mom_long=20, stop_weight=0.16, stop_threshold=0.088, e5_cooldown=8)),
        ("U93_最终_sw0.17_st0.088_cd8", lambda: build_v13_b2_improved(mom_short=10, mom_long=20, stop_weight=0.17, stop_threshold=0.088, e5_cooldown=8)),
        ("U93_最终_sw0.18_st0.088_cd8", lambda: build_v13_b2_improved(mom_short=10, mom_long=20, stop_weight=0.18, stop_threshold=0.088, e5_cooldown=8)),
        ("U93_最终_sw0.16_st0.090_cd8", lambda: build_v13_b2_improved(mom_short=10, mom_long=20, stop_weight=0.16, stop_threshold=0.090, e5_cooldown=8)),
        ("U93_最终_sl0.0018", lambda: build_custom(slope_thresh=0.0018)),
        ("U93_最终_sl0.0022", lambda: build_custom(slope_thresh=0.0022)),
        ("U93_最终_rt1.28", lambda: build_custom(rt=1.28)),
        ("U93_最终_rt1.32", lambda: build_custom(rt=1.32)),
        ("U93_最终_dc4_dcd5", lambda: build_custom(dc=4, dcd=5)),
        ("U93_最终_dc5_dcd4", lambda: build_custom(dc=5, dcd=4)),
    ]
    for name, func in final_configs:
        sig, wt = func()
        info = test_strategy(name, sig, wt, name.replace("U93_最终_", ""))
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
    print(f"  {'U82(v15最优)':<36} {'45.61%':>7} {'-21.01%':>7} {'1.539':>7} "
          f"{'2.171':>7} {'388':>5} {'43.08%':>8} {'1.852':>10}")
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

    # 综合最优(无滑点Calmar和滑点Calmar的几何平均)
    best_combined = max(results, key=lambda x: (x['calmar'] * x['calmar_sl']) ** 0.5)
    print(f"  ★综合最优: {best_combined['name']} "
          f"几何平均Calmar={((best_combined['calmar'] * best_combined['calmar_sl']) ** 0.5):.3f} "
          f"(无滑点{best_combined['calmar']:.3f} + 滑点{best_combined['calmar_sl']:.3f})")
