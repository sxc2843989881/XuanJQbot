"""unified_strategy_test_v27.py — 第二十七轮: MHr=0+cd=9双甜点联合深度优化
================================================================
v26突破:
  - U164_MH92_MHr0.0: Calmar 2.378 (年化49.79% 回撤-20.94%)
  - U163_cd9: 滑点Calmar 2.045 (滑点年化46.70%)
v27: MHr=0+cd=9双甜点联合深度优化
  1. cd(7-12) + MHr(0.0-0.10) + dcd(5-7) 三参数联合
  2. sw/st甜点联合(MHr=0方向)
  3. bias_high甜点细调
  4. 极限组合最终验证
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
    """v26最优核心: Br=0, MHr=0, dcd=6"""
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

    # BIAS过滤(Br=0直接空仓)
    extreme_g = (dir_s == 'growth') & (G_BIAS > bias_high)
    extreme_v = (dir_s == 'value') & (V_BIAS > bias_high)
    extreme = extreme_g | extreme_v
    wt[extreme] = wt[extreme] * bias_reduce

    # E5止损
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

    # 持仓时间限制(MHr=0直接空仓)
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
    print("  统一策略第二十七轮: MHr=0+cd=9双甜点联合深度优化")
    print("=" * 80)

    results = []

    # ---- 基准 ----
    print("\n--- 基准: v26最优(U164_MH92_MHr0.0) ---")
    sig, wt = build_core(bias_high=0.19, bias_reduce=0.0, dcd=6,
                         max_hold_days=92, max_hold_reduce=0.0)
    info = test_strategy("U164_基准", sig, wt, "v26最优")
    print_info(info); results.append(info)

    # ---- 第一组: cd(7-12) + MHr(0.0-0.10) + dcd(5-7) 三参数联合 ----
    print("\n--- 第一组: cd+MHr+dcd三参数联合 ---")
    for cd in [7, 8, 9, 10, 11, 12]:
        for mhr in [0.0, 0.02, 0.04, 0.06, 0.08, 0.10]:
            for dcd in [5, 6, 7]:
                sig, wt = build_core(bias_high=0.19, bias_reduce=0.0, dcd=dcd, cd=cd,
                                     max_hold_days=92, max_hold_reduce=mhr)
                info = test_strategy(f"U170_cd{cd}_MHr{mhr}_dcd{dcd}", sig, wt,
                                     f"cd={cd} MHr={mhr} dcd={dcd}")
                print_info(info); results.append(info)

    # ---- 第二组: sw/st甜点联合(MHr=0方向) ----
    print("\n--- 第二组: sw/st甜点联合(MHr=0) ---")
    for sw in [0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20]:
        for st in [0.080, 0.085, 0.088, 0.090, 0.092, 0.095]:
            sig, wt = build_core(bias_high=0.19, bias_reduce=0.0, dcd=6, sw=sw, st=st,
                                 max_hold_days=92, max_hold_reduce=0.0)
            info = test_strategy(f"U171_sw{sw}_st{st}", sig, wt,
                                 f"sw={sw} st={st}")
            print_info(info); results.append(info)

    # ---- 第三组: bias_high甜点细调(0.185-0.195) ----
    print("\n--- 第三组: bias_high甜点细调 ---")
    for bh in [0.180, 0.185, 0.188, 0.190, 0.192, 0.195, 0.198, 0.200]:
        sig, wt = build_core(bias_high=bh, bias_reduce=0.0, dcd=6,
                             max_hold_days=92, max_hold_reduce=0.0)
        info = test_strategy(f"U172_BH{bh}", sig, wt, f"BH={bh}")
        print_info(info); results.append(info)

    # ---- 第四组: cd=9 + bias_high联合 ----
    print("\n--- 第四组: cd=9 + bias_high联合 ---")
    for bh in [0.180, 0.185, 0.188, 0.190, 0.192, 0.195, 0.198, 0.200]:
        for cd in [8, 9, 10]:
            sig, wt = build_core(bias_high=bh, bias_reduce=0.0, dcd=6, cd=cd,
                                 max_hold_days=92, max_hold_reduce=0.0)
            info = test_strategy(f"U173_BH{bh}_cd{cd}", sig, wt,
                                 f"BH={bh} cd={cd}")
            print_info(info); results.append(info)

    # ---- 第五组: dc/rt/slope细调 ----
    print("\n--- 第五组: dc/rt/slope细调 ---")
    for dc in [4, 5, 6, 7]:
        for rt in [1.2, 1.25, 1.3, 1.35, 1.4]:
            sig, wt = build_core(bias_high=0.19, bias_reduce=0.0, dcd=6, dc=dc, rt=rt,
                                 max_hold_days=92, max_hold_reduce=0.0)
            info = test_strategy(f"U174_dc{dc}_rt{rt}", sig, wt,
                                 f"dc={dc} rt={rt}")
            print_info(info); results.append(info)

    # ---- 第六组: 极限组合最终验证 ----
    print("\n--- 第六组: 极限组合最终验证 ---")
    final_combos = [
        ("U175_最终_基准", dict(bias_high=0.19, bias_reduce=0.0, dcd=6,
                                max_hold_days=92, max_hold_reduce=0.0)),
        ("U175_最终_cd9", dict(bias_high=0.19, bias_reduce=0.0, dcd=6, cd=9,
                                max_hold_days=92, max_hold_reduce=0.0)),
        ("U175_最终_cd10", dict(bias_high=0.19, bias_reduce=0.0, dcd=6, cd=10,
                                 max_hold_days=92, max_hold_reduce=0.0)),
        ("U175_最终_sw0.16", dict(bias_high=0.19, bias_reduce=0.0, dcd=6, sw=0.16,
                                   max_hold_days=92, max_hold_reduce=0.0)),
        ("U175_最终_sw0.18", dict(bias_high=0.19, bias_reduce=0.0, dcd=6, sw=0.18,
                                   max_hold_days=92, max_hold_reduce=0.0)),
        ("U175_最终_st0.09", dict(bias_high=0.19, bias_reduce=0.0, dcd=6, st=0.090,
                                   max_hold_days=92, max_hold_reduce=0.0)),
        ("U175_最终_st0.092", dict(bias_high=0.19, bias_reduce=0.0, dcd=6, st=0.092,
                                    max_hold_days=92, max_hold_reduce=0.0)),
        ("U175_最终_BH0.18", dict(bias_high=0.18, bias_reduce=0.0, dcd=6,
                                   max_hold_days=92, max_hold_reduce=0.0)),
        ("U175_最终_BH0.20", dict(bias_high=0.20, bias_reduce=0.0, dcd=6,
                                   max_hold_days=92, max_hold_reduce=0.0)),
        ("U175_最终_cd9_sw0.16", dict(bias_high=0.19, bias_reduce=0.0, dcd=6, cd=9, sw=0.16,
                                       max_hold_days=92, max_hold_reduce=0.0)),
        ("U175_最终_cd9_st0.092", dict(bias_high=0.19, bias_reduce=0.0, dcd=6, cd=9, st=0.092,
                                        max_hold_days=92, max_hold_reduce=0.0)),
    ]
    for name, params in final_combos:
        sig, wt = build_core(**params)
        info = test_strategy(name, sig, wt, name.replace("U175_最终_", ""))
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
    print(f"  {'U151_v25(基准)':<46} {'49.51%':>7} {'-20.94%':>7} {'1.676':>7} "
          f"{'2.364':>7} {'400':>5} {'46.86%':>8} {'2.020':>10}")
    print(f"  {'U164_v26(基准)':<46} {'49.79%':>7} {'-20.94%':>7} {'1.685':>7} "
          f"{'2.378':>7} {'400':>5} {'47.14%':>8} {'2.032':>10}")

    results.sort(key=lambda x: -x['calmar'])
    for r in results[:30]:
        if r['name'] == 'U164_基准':
            continue
        print(f"  {r['name']:<46} {r['ann']*100:>6.2f}% {r['dd']*100:>6.2f}% "
              f"{r['sharpe']:>7.3f} {r['calmar']:>7.3f} {r['n_trades']:>5} "
              f"{r['ann_sl']*100:>7.2f}% {r['calmar_sl']:>10.3f}")

    print(f"\n  Top 15 Calmar:")
    for r in results[:15]:
        print(f"  {r['name']:<46} 年化{r['ann']*100:.2f}% 回撤{r['dd']*100:.2f}% "
              f"Calmar{r['calmar']:.3f} 滑点Calmar{r['calmar_sl']:.3f}")

    best = max(results, key=lambda x: x['calmar'])
    print(f"\n  ★最优: {best['name']} Calmar={best['calmar']:.3f} "
          f"年化={best['ann']*100:.2f}% 回撤={best['dd']*100:.2f}%")

    best_sl = max(results, key=lambda x: x['calmar_sl'])
    print(f"  ★滑点最优: {best_sl['name']} 滑点Calmar={best_sl['calmar_sl']:.3f} "
          f"滑点年化={best_sl['ann_sl']*100:.2f}%")

    v26_calmar = 2.378
    improved = [r for r in results if r['calmar'] > v26_calmar]
    print(f"\n  超越v26(Calmar>{v26_calmar})的版本: {len(improved)}个")
    for r in improved:
        print(f"    {r['name']}: Calmar={r['calmar']:.3f} (+{r['calmar']-v26_calmar:.3f})")
