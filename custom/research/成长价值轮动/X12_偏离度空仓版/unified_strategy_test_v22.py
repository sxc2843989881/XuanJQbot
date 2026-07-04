"""unified_strategy_test_v22.py — 第二十二轮: 三甜点联合极限优化
================================================================
v21最优: U123_BIAS020_MH90_r0.4 Calmar 2.305
  - BIAS r=0.20 + 持仓90天 + 持仓减仓至0.4
v22: 在三个甜点附近深度优化
  1. BIAS阈值(0.18-0.20) + r(0.15-0.25) + MH90 + MHr(0.3-0.5) 联合扫描
  2. 持仓时间(85-95)细调
  3. 最优组合+v16其他参数联合
  4. 尝试BIAS+持仓时间的"或"逻辑(BIAS触发也重置持仓时间)
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
               ms=10, ml=20, rt=1.3, dc=5, dcd=5,
               # BIAS过滤
               bias_ma=20, bias_high=0.19, bias_reduce=0.20,
               # 持仓时间
               use_max_hold=True, max_hold_days=90, max_hold_reduce=0.4,
               hold_mode='reset_dir'):
    """v21最优核心"""
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

    # BIAS过滤
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

    # 持仓时间限制
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
    print("  统一策略第二十二轮: 三甜点联合极限优化")
    print("=" * 80)

    results = []

    # ---- 基准 ----
    print("\n--- 基准: v21最优(U123_BIAS020_MH90_r0.4) ---")
    sig, wt = build_core(bias_high=0.19, bias_reduce=0.20,
                         use_max_hold=True, max_hold_days=90, max_hold_reduce=0.4)
    info = test_strategy("U123_基准", sig, wt, "v21最优")
    print_info(info); results.append(info)

    # ---- 第一组: BIAS阈值+r+MHr联合扫描 ----
    print("\n--- 第一组: BIAS阈值+r+MHr联合扫描 (MH=90) ---")
    for bh in [0.18, 0.19, 0.20]:
        for br in [0.15, 0.20, 0.25]:
            for mr in [0.3, 0.4, 0.5]:
                sig, wt = build_core(bias_high=bh, bias_reduce=br,
                                      max_hold_days=90, max_hold_reduce=mr)
                info = test_strategy(f"U124_B{bh}_r{br}_MHr{mr}", sig, wt,
                                      f"BIAS>{bh} r={br} MHr={mr}")
                print_info(info); results.append(info)

    # ---- 第二组: 持仓时间细调(85-95) ----
    print("\n--- 第二组: 持仓时间细调 ---")
    for mh in [80, 85, 88, 90, 92, 95, 100]:
        sig, wt = build_core(bias_high=0.19, bias_reduce=0.20,
                              max_hold_days=mh, max_hold_reduce=0.4)
        info = test_strategy(f"U125_MH{mh}", sig, wt, f"持仓{mh}天")
        print_info(info); results.append(info)

    # ---- 第三组: 最优组合+v16参数联合 ----
    print("\n--- 第三组: 最优组合+v16参数联合 ---")
    combos = [
        ("U126_sw0.16", dict(bias_high=0.19, bias_reduce=0.20, sw=0.16,
                              max_hold_days=90, max_hold_reduce=0.4)),
        ("U126_sw0.18", dict(bias_high=0.19, bias_reduce=0.20, sw=0.18,
                              max_hold_days=90, max_hold_reduce=0.4)),
        ("U126_st0.090", dict(bias_high=0.19, bias_reduce=0.20, st=0.090,
                               max_hold_days=90, max_hold_reduce=0.4)),
        ("U126_st0.085", dict(bias_high=0.19, bias_reduce=0.20, st=0.085,
                               max_hold_days=90, max_hold_reduce=0.4)),
        ("U126_cd7", dict(bias_high=0.19, bias_reduce=0.20, cd=7,
                           max_hold_days=90, max_hold_reduce=0.4)),
        ("U126_cd10", dict(bias_high=0.19, bias_reduce=0.20, cd=10,
                            max_hold_days=90, max_hold_reduce=0.4)),
        ("U126_rt1.2", dict(bias_high=0.19, bias_reduce=0.20, rt=1.2,
                             max_hold_days=90, max_hold_reduce=0.4)),
        ("U126_rt1.4", dict(bias_high=0.19, bias_reduce=0.20, rt=1.4,
                             max_hold_days=90, max_hold_reduce=0.4)),
        ("U126_sl0.0018", dict(bias_high=0.19, bias_reduce=0.20, slope_thresh=0.0018,
                                max_hold_days=90, max_hold_reduce=0.4)),
        ("U126_sl0.0022", dict(bias_high=0.19, bias_reduce=0.20, slope_thresh=0.0022,
                                max_hold_days=90, max_hold_reduce=0.4)),
    ]
    for name, params in combos:
        sig, wt = build_core(**params)
        info = test_strategy(name, sig, wt, name.replace("U126_", ""))
        print_info(info); results.append(info)

    # ---- 第四组: BIAS ma + 持仓时间组合 ----
    print("\n--- 第四组: BIAS ma + 持仓时间组合 ---")
    for ma in [15, 20, 25]:
        for mh in [85, 90, 95]:
            sig, wt = build_core(bias_ma=ma, bias_high=0.19, bias_reduce=0.20,
                                  max_hold_days=mh, max_hold_reduce=0.4)
            info = test_strategy(f"U127_ma{ma}_MH{mh}", sig, wt, f"MA{ma}+MH{mh}")
            print_info(info); results.append(info)

    # ---- 第五组: 持仓触发机制reset_pos对比 ----
    print("\n--- 第五组: 持仓触发机制对比 ---")
    for mode in ['reset_dir', 'reset_pos']:
        sig, wt = build_core(bias_high=0.19, bias_reduce=0.20,
                              max_hold_days=90, max_hold_reduce=0.4,
                              hold_mode=mode)
        info = test_strategy(f"U128_mode{mode}", sig, wt, f"触发={mode}")
        print_info(info); results.append(info)

    # ---- 第六组: 最优组合的最终验证 ----
    print("\n--- 第六组: 最优候选最终验证 ---")
    final_combos = [
        ("U129_最终_B0.19_r0.20_MH90_r0.4", dict(bias_high=0.19, bias_reduce=0.20,
                                                    max_hold_days=90, max_hold_reduce=0.4)),
        ("U129_最终_B0.19_r0.20_MH90_r0.3", dict(bias_high=0.19, bias_reduce=0.20,
                                                    max_hold_days=90, max_hold_reduce=0.3)),
        ("U129_最终_B0.19_r0.15_MH90_r0.4", dict(bias_high=0.19, bias_reduce=0.15,
                                                    max_hold_days=90, max_hold_reduce=0.4)),
        ("U129_最终_B0.18_r0.20_MH90_r0.4", dict(bias_high=0.18, bias_reduce=0.20,
                                                    max_hold_days=90, max_hold_reduce=0.4)),
        ("U129_最终_B0.20_r0.20_MH90_r0.4", dict(bias_high=0.20, bias_reduce=0.20,
                                                    max_hold_days=90, max_hold_reduce=0.4)),
        ("U129_最终_B0.19_r0.20_MH85_r0.4", dict(bias_high=0.19, bias_reduce=0.20,
                                                    max_hold_days=85, max_hold_reduce=0.4)),
        ("U129_最终_B0.19_r0.20_MH95_r0.4", dict(bias_high=0.19, bias_reduce=0.20,
                                                    max_hold_days=95, max_hold_reduce=0.4)),
    ]
    for name, params in final_combos:
        sig, wt = build_core(**params)
        info = test_strategy(name, sig, wt, name.replace("U129_最终_", ""))
        print_info(info); results.append(info)

    # ---- 汇总 ----
    print("\n" + "=" * 80)
    print("  最终汇总")
    print("=" * 80)
    print(f"\n  {'名称':<40} {'年化':>7} {'回撤':>7} {'Sharpe':>7} {'Calmar':>7} "
          f"{'交易':>5} {'滑点年化':>8} {'滑点Calmar':>10}")
    print("  " + "-" * 100)
    print(f"  {'X61(基准)':<40} {'41.12%':>7} {'-30.21%':>7} {'1.357':>7} "
          f"{'1.361':>7} {'224':>5} {'39.04%':>8} {'1.214':>10}")
    print(f"  {'U87_v16(基准)':<40} {'45.62%':>7} {'-20.94%':>7} {'1.538':>7} "
          f"{'2.179':>7} {'388':>5} {'43.08%':>8} {'1.857':>10}")
    print(f"  {'U116_v20(基准)':<40} {'47.92%':>7} {'-20.94%':>7} {'1.628':>7} "
          f"{'2.289':>7} {'400':>5} {'45.27%':>8} {'1.951':>10}")
    print(f"  {'U123_v21(基准)':<40} {'48.27%':>7} {'-20.94%':>7} {'1.640':>7} "
          f"{'2.305':>7} {'400':>5} {'45.62%':>8} {'1.966':>10}")

    results.sort(key=lambda x: -x['calmar'])
    for r in results[:25]:
        if r['name'] == 'U123_基准':
            continue
        print(f"  {r['name']:<40} {r['ann']*100:>6.2f}% {r['dd']*100:>6.2f}% "
              f"{r['sharpe']:>7.3f} {r['calmar']:>7.3f} {r['n_trades']:>5} "
              f"{r['ann_sl']*100:>7.2f}% {r['calmar_sl']:>10.3f}")

    print(f"\n  Top 15 Calmar:")
    for r in results[:15]:
        print(f"  {r['name']:<40} 年化{r['ann']*100:.2f}% 回撤{r['dd']*100:.2f}% "
              f"Calmar{r['calmar']:.3f} 滑点Calmar{r['calmar_sl']:.3f}")

    best = max(results, key=lambda x: x['calmar'])
    print(f"\n  ★最优: {best['name']} Calmar={best['calmar']:.3f} "
          f"年化={best['ann']*100:.2f}% 回撤={best['dd']*100:.2f}%")

    v21_calmar = 2.305
    improved = [r for r in results if r['calmar'] > v21_calmar]
    print(f"\n  超越v21(Calmar>{v21_calmar})的版本: {len(improved)}个")
    for r in improved:
        print(f"    {r['name']}: Calmar={r['calmar']:.3f} (+{r['calmar']-v21_calmar:.3f})")
