"""unified_strategy_test_v28.py — 第二十八轮: 滑点Calmar>2.06深度优化
================================================================
v27突破:
  - U164_基准: 无滑点Calmar 2.378 (年化49.79% 回撤-20.94%)
  - U175_cd9_sw0.16: 滑点Calmar 2.059 (滑点年化46.83%)
v28: 滑点Calmar>2.06深度优化 + 探索新方向
  1. cd(8-12) + sw(0.14-0.18) 联合扫描
  2. cd=9 + 其他参数联合(st/bias_high)
  3. 探索新方向(趋势强度/动态持仓)
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
from run_x33_reduce_trades import RATIO_DEV_Z

T = RATIO_DEV_Z
SLOPE = MA20_SLOPE


def build_core(slope_thresh=0.002, sw=0.17, st=0.088, cd=8,
               ms=10, ml=20, rt=1.3, dc=5, dcd=6,
               bias_ma=20, bias_high=0.19, bias_reduce=0.0,
               use_max_hold=True, max_hold_days=92, max_hold_reduce=0.0,
               hold_mode='reset_dir'):
    """v26最优核心"""
    G_MA = G_CLOSE.rolling(bias_ma).mean()
    V_MA = V_CLOSE.rolling(bias_ma).mean()
    G_BIAS = (G_CLOSE / G_MA - 1)
    V_BIAS = (V_CLOSE / V_MA - 1)

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

    extreme_g = (dir_s == 'growth') & (G_BIAS > bias_high)
    extreme_v = (dir_s == 'value') & (V_BIAS > bias_high)
    extreme = extreme_g | extreme_v
    wt[extreme] = wt[extreme] * bias_reduce

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

    if use_max_hold:
        hold_count = 0
        prev_key = None
        for i in range(len(wt)):
            if pd.isna(wt.iloc[i]) or pd.isna(dir_s.iloc[i]):
                continue
            if hold_mode == 'reset_dir':
                key = dir_s.iloc[i]
            elif hold_mode == 'reset_pos':
                key = (dir_s.iloc[i], round(wt.iloc[i], 2))
            else:
                key = None

            if key != prev_key:
                hold_count = 0
                prev_key = key
            else:
                hold_count += 1
                if hold_count >= max_hold_days:
                    if wt.iloc[i] > 0:
                        wt.iloc[i] = wt.iloc[i] * max_hold_reduce

    signal = dir_raw.map({'BULL': 'growth', 'BEAR': 'value'})
    return signal, wt


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
          f"交易={info['n_trades']}")
    print(f"    5/万滑点: 年化={info['ann_sl']*100:.2f}% Calmar={info['calmar_sl']:.3f}")


if __name__ == '__main__':
    print("=" * 80)
    print("  统一策略第二十八轮: 滑点Calmar>2.06深度优化")
    print("=" * 80)

    results = []

    # ---- 基准 ----
    print("\n--- 基准: v27最优(U175_cd9_sw0.16滑点) ---")
    sig, wt = build_core(bias_high=0.19, bias_reduce=0.0, dcd=6, cd=9, sw=0.16,
                         max_hold_days=92, max_hold_reduce=0.0)
    info = test_strategy("U175_基准", sig, wt, "v27滑点最优")
    print_info(info); results.append(info)

    # ---- 第一组: cd(8-12) + sw(0.14-0.18) 联合扫描 ----
    print("\n--- 第一组: cd+sw联合扫描 ---")
    for cd in [8, 9, 10, 11, 12, 13, 14, 15]:
        for sw in [0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20]:
            sig, wt = build_core(bias_high=0.19, bias_reduce=0.0, dcd=6, cd=cd, sw=sw,
                                 max_hold_days=92, max_hold_reduce=0.0)
            info = test_strategy(f"U180_cd{cd}_sw{sw}", sig, wt,
                                 f"cd={cd} sw={sw}")
            print_info(info); results.append(info)

    # ---- 第二组: cd=9 + st/bias_high 联合 ----
    print("\n--- 第二组: cd=9 + st/bias_high联合 ---")
    for st in [0.080, 0.085, 0.088, 0.090, 0.092, 0.095, 0.100]:
        for bh in [0.18, 0.19, 0.20]:
            sig, wt = build_core(bias_high=bh, bias_reduce=0.0, dcd=6, cd=9, sw=0.16, st=st,
                                 max_hold_days=92, max_hold_reduce=0.0)
            info = test_strategy(f"U181_st{st}_BH{bh}", sig, wt,
                                 f"st={st} BH={bh}")
            print_info(info); results.append(info)

    # ---- 第三组: cd=9 + sw=0.16 + dcd 联合 ----
    print("\n--- 第三组: cd=9+sw=0.16+dcd联合 ---")
    for dcd in [4, 5, 6, 7, 8, 10, 12]:
        sig, wt = build_core(bias_high=0.19, bias_reduce=0.0, dcd=dcd, cd=9, sw=0.16,
                             max_hold_days=92, max_hold_reduce=0.0)
        info = test_strategy(f"U182_dcd{dcd}", sig, wt, f"dcd={dcd}")
        print_info(info); results.append(info)

    # ---- 第四组: cd=9 + sw=0.16 + MH/MHr 联合 ----
    print("\n--- 第四组: cd=9+sw=0.16+MH/MHr联合 ---")
    for mh in [88, 90, 92, 94, 96]:
        for mhr in [0.0, 0.02, 0.04, 0.06, 0.08]:
            sig, wt = build_core(bias_high=0.19, bias_reduce=0.0, dcd=6, cd=9, sw=0.16,
                                 max_hold_days=mh, max_hold_reduce=mhr)
            info = test_strategy(f"U183_MH{mh}_MHr{mhr}", sig, wt,
                                 f"MH{mh} MHr={mhr}")
            print_info(info); results.append(info)

    # ---- 第五组: cd=9 + sw=0.16 + rt/dc/slope 联合 ----
    print("\n--- 第五组: cd=9+sw=0.16+rt/dc/slope联合 ---")
    for rt in [1.2, 1.25, 1.3, 1.35, 1.4]:
        for dc in [4, 5, 6, 7]:
            sig, wt = build_core(bias_high=0.19, bias_reduce=0.0, dcd=6, cd=9, sw=0.16,
                                 rt=rt, dc=dc,
                                 max_hold_days=92, max_hold_reduce=0.0)
            info = test_strategy(f"U184_rt{rt}_dc{dc}", sig, wt,
                                 f"rt={rt} dc={dc}")
            print_info(info); results.append(info)

    # ---- 第六组: 极限组合最终验证 ----
    print("\n--- 第六组: 极限组合最终验证 ---")
    final_combos = [
        ("U185_最终_基准", dict(bias_high=0.19, bias_reduce=0.0, dcd=6, cd=9, sw=0.16,
                                max_hold_days=92, max_hold_reduce=0.0)),
        ("U185_最终_cd10_sw0.16", dict(bias_high=0.19, bias_reduce=0.0, dcd=6, cd=10, sw=0.16,
                                        max_hold_days=92, max_hold_reduce=0.0)),
        ("U185_最终_cd9_sw0.15", dict(bias_high=0.19, bias_reduce=0.0, dcd=6, cd=9, sw=0.15,
                                       max_hold_days=92, max_hold_reduce=0.0)),
        ("U185_最终_cd9_sw0.17", dict(bias_high=0.19, bias_reduce=0.0, dcd=6, cd=9, sw=0.17,
                                       max_hold_days=92, max_hold_reduce=0.0)),
        ("U185_最终_cd9_sw0.16_st0.092", dict(bias_high=0.19, bias_reduce=0.0, dcd=6, cd=9, sw=0.16, st=0.092,
                                                max_hold_days=92, max_hold_reduce=0.0)),
        ("U185_最终_cd9_sw0.16_BH0.18", dict(bias_high=0.18, bias_reduce=0.0, dcd=6, cd=9, sw=0.16,
                                              max_hold_days=92, max_hold_reduce=0.0)),
        ("U185_最终_cd10_sw0.17", dict(bias_high=0.19, bias_reduce=0.0, dcd=6, cd=10, sw=0.17,
                                        max_hold_days=92, max_hold_reduce=0.0)),
        ("U185_最终_cd11_sw0.16", dict(bias_high=0.19, bias_reduce=0.0, dcd=6, cd=11, sw=0.16,
                                        max_hold_days=92, max_hold_reduce=0.0)),
        ("U185_最终_cd9_sw0.16_dcd7", dict(bias_high=0.19, bias_reduce=0.0, dcd=7, cd=9, sw=0.16,
                                            max_hold_days=92, max_hold_reduce=0.0)),
    ]
    for name, params in final_combos:
        sig, wt = build_core(**params)
        info = test_strategy(name, sig, wt, name.replace("U185_最终_", ""))
        print_info(info); results.append(info)

    # ---- 汇总 ----
    print("\n" + "=" * 80)
    print("  最终汇总")
    print("=" * 80)
    print(f"\n  {'名称':<46} {'年化':>7} {'回撤':>7} {'Sharpe':>7} {'Calmar':>7} "
          f"{'交易':>5} {'滑点年化':>8} {'滑点Calmar':>10}")
    print("  " + "-" * 108)
    print(f"  {'X61(基准)':<46} {'41.12%':>7} {'-30.21%':>7} {'1.357':>7} "
          f"{'1.361':>7} {'224':>5} {'39.04%':>8} {'1.214':>10}")
    print(f"  {'U164_v26(基准)':<46} {'49.79%':>7} {'-20.94%':>7} {'1.685':>7} "
          f"{'2.378':>7} {'395':>5} {'47.14%':>8} {'2.032':>10}")
    print(f"  {'U175_v27(基准)':<46} {'49.79%':>7} {'-20.94%':>7} {'1.685':>7} "
          f"{'2.378':>7} {'395':>5} {'46.83%':>8} {'2.059':>10}")

    # 排序按滑点Calmar
    results.sort(key=lambda x: -x['calmar_sl'])
    print(f"\n  Top 30 滑点Calmar:")
    for r in results[:30]:
        if r['name'] == 'U175_基准':
            continue
        print(f"  {r['name']:<46} {r['ann']*100:>6.2f}% {r['dd']*100:>6.2f}% "
              f"{r['sharpe']:>7.3f} {r['calmar']:>7.3f} {r['n_trades']:>5} "
              f"{r['ann_sl']*100:>7.2f}% {r['calmar_sl']:>10.3f}")

    print(f"\n  Top 15 无滑点Calmar:")
    results.sort(key=lambda x: -x['calmar'])
    for r in results[:15]:
        print(f"  {r['name']:<46} 年化{r['ann']*100:.2f}% 回撤{r['dd']*100:.2f}% "
              f"Calmar{r['calmar']:.3f} 滑点Calmar{r['calmar_sl']:.3f}")

    best = max(results, key=lambda x: x['calmar'])
    print(f"\n  ★无滑点最优: {best['name']} Calmar={best['calmar']:.3f} "
          f"年化={best['ann']*100:.2f}% 回撤={best['dd']*100:.2f}%")

    best_sl = max(results, key=lambda x: x['calmar_sl'])
    print(f"  ★滑点最优: {best_sl['name']} 滑点Calmar={best_sl['calmar_sl']:.3f} "
          f"滑点年化={best_sl['ann_sl']*100:.2f}%")

    # 综合最优(几何平均)
    best_combined = max(results, key=lambda x: (x['calmar'] * x['calmar_sl']) ** 0.5)
    print(f"  ★综合最优: {best_combined['name']} "
          f"几何平均Calmar={((best_combined['calmar'] * best_combined['calmar_sl']) ** 0.5):.3f} "
          f"(无滑点{best_combined['calmar']:.3f} + 滑点{best_combined['calmar_sl']:.3f})")

    v27_calmar_sl = 2.059
    improved = [r for r in results if r['calmar_sl'] > v27_calmar_sl]
    print(f"\n  超越v27滑点(Calmar>{v27_calmar_sl})的版本: {len(improved)}个")
    for r in improved[:15]:
        print(f"    {r['name']}: 滑点Calmar={r['calmar_sl']:.3f} (+{r['calmar_sl']-v27_calmar_sl:.3f})")
