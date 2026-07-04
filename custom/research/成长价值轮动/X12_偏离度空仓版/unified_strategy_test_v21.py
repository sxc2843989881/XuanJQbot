"""unified_strategy_test_v21.py — 第二十一轮: 持仓时间+BIAS联合甜点
================================================================
v20两个突破:
  - U116_MH90_r0.5: Calmar 2.289 (持仓90天减仓至0.5)
  - U112_BIAS0.19_r0.25: Calmar 2.274 (BIAS减仓比例0.25)
v21: 联合这两个甜点 + 深度优化
  1. 持仓时间(60-120) + 减仓比例(0.4-0.6) 联合扫描
  2. BIAS r=0.25 + 持仓时间 联合
  3. 持仓时间+BIAS+其他参数最优组合
  4. 不同持仓时间触发机制(累积持有/方向切换重置)
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
               bias_ma=20, bias_high=0.19, bias_reduce=0.25,
               # 持仓时间
               use_max_hold=False, max_hold_days=90, max_hold_reduce=0.5,
               # 持仓触发机制: 'reset_dir'方向切换重置, 'reset_pos'仓位变化重置, 'cumulative'累积
               hold_mode='reset_dir'):
    """v20最优+持仓时间+BIAS联合"""
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
            # 重置条件
            if hold_mode == 'reset_dir':
                key = dir_s.iloc[i]
            elif hold_mode == 'reset_pos':
                key = (dir_s.iloc[i], round(wt.iloc[i], 2))
            else:  # cumulative
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
    print("  统一策略第二十一轮: 持仓时间+BIAS联合甜点")
    print("=" * 80)

    results = []

    # ---- 基准 ----
    print("\n--- 基准: v20最优(U116_MH90_r0.5) ---")
    sig, wt = build_core(bias_high=0.19, bias_reduce=0.3,
                         use_max_hold=True, max_hold_days=90, max_hold_reduce=0.5)
    info = test_strategy("U116_基准", sig, wt, "v20最优")
    print_info(info); results.append(info)

    # ---- 第一组: 持仓时间+减仓比例联合扫描 ----
    print("\n--- 第一组: 持仓时间+减仓比例联合扫描 ---")
    for mh in [60, 75, 90, 100, 120]:
        for mr in [0.4, 0.5, 0.6]:
            sig, wt = build_core(bias_high=0.19, bias_reduce=0.3,
                                  use_max_hold=True, max_hold_days=mh, max_hold_reduce=mr)
            info = test_strategy(f"U118_MH{mh}_r{mr}", sig, wt, f"持仓{mh}天减仓至{mr}")
            print_info(info); results.append(info)

    # ---- 第二组: BIAS r=0.25 + 持仓时间联合 ----
    print("\n--- 第二组: BIAS r=0.25 + 持仓时间联合 ---")
    for mh in [60, 75, 90, 100, 120]:
        for mr in [0.4, 0.5, 0.6]:
            sig, wt = build_core(bias_high=0.19, bias_reduce=0.25,
                                  use_max_hold=True, max_hold_days=mh, max_hold_reduce=mr)
            info = test_strategy(f"U119_BIAS025_MH{mh}_r{mr}", sig, wt,
                                  f"BIAS r=0.25 持仓{mh}天减仓至{mr}")
            print_info(info); results.append(info)

    # ---- 第三组: 不同持仓触发机制 ----
    print("\n--- 第三组: 不同持仓触发机制 ---")
    for mode in ['reset_dir', 'reset_pos', 'cumulative']:
        sig, wt = build_core(bias_high=0.19, bias_reduce=0.25,
                              use_max_hold=True, max_hold_days=90, max_hold_reduce=0.5,
                              hold_mode=mode)
        info = test_strategy(f"U120_mode{mode}", sig, wt, f"持仓触发={mode}")
        print_info(info); results.append(info)

    # ---- 第四组: 持仓时间+BIAS更细参数 ----
    print("\n--- 第四组: 持仓时间+BIAS更细参数 ---")
    for bh in [0.18, 0.19, 0.20]:
        for br in [0.20, 0.25, 0.30]:
            sig, wt = build_core(bias_high=bh, bias_reduce=br,
                                  use_max_hold=True, max_hold_days=90, max_hold_reduce=0.5)
            info = test_strategy(f"U121_BIAS{bh}_r{br}_MH90", sig, wt,
                                  f"BIAS>{bh} r={br} MH90")
            print_info(info); results.append(info)

    # ---- 第五组: 持仓时间+其他参数联合 ----
    print("\n--- 第五组: 持仓时间+其他参数联合 ---")
    combos = [
        ("U122_MH90_sw0.16", dict(bias_high=0.19, bias_reduce=0.25, sw=0.16,
                                    use_max_hold=True, max_hold_days=90, max_hold_reduce=0.5)),
        ("U122_MH90_sw0.18", dict(bias_high=0.19, bias_reduce=0.25, sw=0.18,
                                    use_max_hold=True, max_hold_days=90, max_hold_reduce=0.5)),
        ("U122_MH90_st0.090", dict(bias_high=0.19, bias_reduce=0.25, st=0.090,
                                     use_max_hold=True, max_hold_days=90, max_hold_reduce=0.5)),
        ("U122_MH90_cd7", dict(bias_high=0.19, bias_reduce=0.25, cd=7,
                                 use_max_hold=True, max_hold_days=90, max_hold_reduce=0.5)),
        ("U122_MH90_cd10", dict(bias_high=0.19, bias_reduce=0.25, cd=10,
                                  use_max_hold=True, max_hold_days=90, max_hold_reduce=0.5)),
        ("U122_MH90_rt1.2", dict(bias_high=0.19, bias_reduce=0.25, rt=1.2,
                                   use_max_hold=True, max_hold_days=90, max_hold_reduce=0.5)),
        ("U122_MH90_rt1.4", dict(bias_high=0.19, bias_reduce=0.25, rt=1.4,
                                   use_max_hold=True, max_hold_days=90, max_hold_reduce=0.5)),
        ("U122_MH100_r0.5", dict(bias_high=0.19, bias_reduce=0.25,
                                   use_max_hold=True, max_hold_days=100, max_hold_reduce=0.5)),
        ("U122_MH120_r0.4", dict(bias_high=0.19, bias_reduce=0.25,
                                   use_max_hold=True, max_hold_days=120, max_hold_reduce=0.4)),
    ]
    for name, params in combos:
        sig, wt = build_core(**params)
        info = test_strategy(name, sig, wt, name.replace("U122_", ""))
        print_info(info); results.append(info)

    # ---- 第六组: BIAS r=0.20 + 持仓时间最优 ----
    print("\n--- 第六组: BIAS r=0.20 + 持仓时间 ---")
    for mh in [75, 90, 100]:
        for mr in [0.4, 0.5, 0.6]:
            sig, wt = build_core(bias_high=0.19, bias_reduce=0.20,
                                  use_max_hold=True, max_hold_days=mh, max_hold_reduce=mr)
            info = test_strategy(f"U123_BIAS020_MH{mh}_r{mr}", sig, wt,
                                  f"BIAS r=0.20 持仓{mh}天减仓至{mr}")
            print_info(info); results.append(info)

    # ---- 汇总 ----
    print("\n" + "=" * 80)
    print("  最终汇总")
    print("=" * 80)
    print(f"\n  {'名称':<36} {'年化':>7} {'回撤':>7} {'Sharpe':>7} {'Calmar':>7} "
          f"{'交易':>5} {'滑点年化':>8} {'滑点Calmar':>10}")
    print("  " + "-" * 96)
    print(f"  {'X61(基准)':<36} {'41.12%':>7} {'-30.21%':>7} {'1.357':>7} "
          f"{'1.361':>7} {'224':>5} {'39.04%':>8} {'1.214':>10}")
    print(f"  {'U87_v16(基准)':<36} {'45.62%':>7} {'-20.94%':>7} {'1.538':>7} "
          f"{'2.179':>7} {'388':>5} {'43.08%':>8} {'1.857':>10}")
    print(f"  {'U106_v19(基准)':<36} {'47.48%':>7} {'-20.94%':>7} {'1.605':>7} "
          f"{'2.268':>7} {'398':>5} {'44.83%':>8} {'1.933':>10}")
    print(f"  {'U116_v20(基准)':<36} {'47.92%':>7} {'-20.94%':>7} {'1.628':>7} "
          f"{'2.289':>7} {'400':>5} {'45.27%':>8} {'1.951':>10}")

    results.sort(key=lambda x: -x['calmar'])
    for r in results[:25]:
        if r['name'] == 'U116_基准':
            continue
        print(f"  {r['name']:<36} {r['ann']*100:>6.2f}% {r['dd']*100:>6.2f}% "
              f"{r['sharpe']:>7.3f} {r['calmar']:>7.3f} {r['n_trades']:>5} "
              f"{r['ann_sl']*100:>7.2f}% {r['calmar_sl']:>10.3f}")

    print(f"\n  Top 15 Calmar:")
    for r in results[:15]:
        print(f"  {r['name']:<36} 年化{r['ann']*100:.2f}% 回撤{r['dd']*100:.2f}% "
              f"Calmar{r['calmar']:.3f} 滑点Calmar{r['calmar_sl']:.3f}")

    best = max(results, key=lambda x: x['calmar'])
    print(f"\n  ★最优: {best['name']} Calmar={best['calmar']:.3f} "
          f"年化={best['ann']*100:.2f}% 回撤={best['dd']*100:.2f}%")

    v20_calmar = 2.289
    improved = [r for r in results if r['calmar'] > v20_calmar]
    print(f"\n  超越v20(Calmar>{v20_calmar})的版本: {len(improved)}个")
    for r in improved:
        print(f"    {r['name']}: Calmar={r['calmar']:.3f} (+{r['calmar']-v20_calmar:.3f})")
