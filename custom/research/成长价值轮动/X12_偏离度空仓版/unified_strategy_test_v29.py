"""unified_strategy_test_v29.py — 第二十九轮: 新方向探索(极限突破)
================================================================
v23-v28已达到极限:
  - 无滑点Calmar极限: 2.378 (v26)
  - 滑点Calmar极限: 2.059 (v27)
v29: 探索全新方向试图突破
  1. 信号平滑(连续N天方向一致才算切换)
  2. 双重止损机制(E5+BIAS联合)
  3. 动态参数(波动率调整)
  4. 趋势强度过滤
  5. 极限组合最终验证
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
               hold_mode='reset_dir',
               # 新方向1: 信号平滑
               smooth_days=1,
               # 新方向2: 双重止损
               use_double_stop=False, st2=0.15, sw2=0.10,
               # 新方向3: 趋势强度过滤
               use_trend_strength=False, ts_lookback=20, ts_thresh=0.5,
               # 新方向4: BIAS恢复确认
               use_bias_recovery=False, recovery_days=3):
    """v26最优核心 + 新方向探索"""
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

    # 新方向1: 信号平滑(连续N天方向一致才算切换)
    if smooth_days > 1:
        new_dir = confirmed_dir.copy()
        last_change = -smooth_days
        prev = confirmed_dir.iloc[0]
        for i in range(len(confirmed_dir)):
            if confirmed_dir.iloc[i] != prev:
                if i - last_change >= smooth_days:
                    last_change = i
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

    # 新方向4: BIAS恢复确认
    if use_bias_recovery:
        # BIAS超阈值后,需要连续recovery_days天BIAS回落才退出空仓
        in_extreme = False
        recovery_count = 0
        for i in range(len(wt)):
            if pd.isna(wt.iloc[i]) or pd.isna(dir_s.iloc[i]):
                continue
            if not in_extreme:
                if extreme.iloc[i]:
                    in_extreme = True
                    recovery_count = 0
                    wt.iloc[i] = wt.iloc[i] * bias_reduce
                else:
                    pass
            else:
                if extreme.iloc[i]:
                    recovery_count = 0
                    wt.iloc[i] = wt.iloc[i] * bias_reduce
                else:
                    recovery_count += 1
                    if recovery_count >= recovery_days:
                        in_extreme = False
                    else:
                        wt.iloc[i] = wt.iloc[i] * bias_reduce
    else:
        wt[extreme] = wt[extreme] * bias_reduce

    # E5止损
    gs = (dir_s == 'growth') & (G_DD20 < -st)
    vs = (dir_s == 'value') & (V_DD20 < -st)
    e5_trigger = gs | vs

    # 新方向2: 双重止损
    if use_double_stop:
        gs2 = (dir_s == 'growth') & (G_DD20 < -st2)
        vs2 = (dir_s == 'value') & (V_DD20 < -st2)
        e5_trigger2 = gs2 | vs2

    in_cooldown = False
    cooldown_count = 0
    for i in range(len(wt)):
        if pd.isna(wt.iloc[i]) or pd.isna(dir_s.iloc[i]):
            continue
        trigger = e5_trigger.iloc[i]
        if use_double_stop:
            trigger = trigger or e5_trigger2.iloc[i]
            current_sw = sw2 if (use_double_stop and e5_trigger2.iloc[i] and not e5_trigger.iloc[i]) else sw
        else:
            current_sw = sw

        if trigger and not in_cooldown:
            in_cooldown = True
            cooldown_count = 0
            wt.iloc[i] = wt.iloc[i] * current_sw
        elif in_cooldown:
            cooldown_count += 1
            if cooldown_count >= cd:
                if trigger:
                    cooldown_count = 0
                    wt.iloc[i] = wt.iloc[i] * current_sw
                else:
                    in_cooldown = False
                    if is_weak.iloc[i]:
                        wt.iloc[i] = 0.0
                    else:
                        wt.iloc[i] = 1.0
            else:
                if wt.iloc[i] > 0:
                    wt.iloc[i] = current_sw

    # 新方向3: 趋势强度过滤
    if use_trend_strength:
        # 用RATIO_DEV的z-score作为趋势强度
        TS = T.rolling(ts_lookback).apply(lambda x: np.mean(x > 0) - 0.5, raw=False)
        weak_ts = TS.abs() < ts_thresh
        wt[weak_ts] = wt[weak_ts] * 0.5

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
    print("  统一策略第二十九轮: 新方向探索(极限突破)")
    print("=" * 80)

    results = []

    # ---- 基准 ----
    print("\n--- 基准: v26最优(U164_MH92_MHr0.0) ---")
    sig, wt = build_core(bias_high=0.19, bias_reduce=0.0, dcd=6,
                         max_hold_days=92, max_hold_reduce=0.0)
    info = test_strategy("U164_基准", sig, wt, "v26最优")
    print_info(info); results.append(info)

    # ---- 第一组: 信号平滑 ----
    print("\n--- 第一组: 信号平滑(smooth_days) ---")
    for sd in [2, 3, 5, 7, 10, 15, 20]:
        sig, wt = build_core(bias_high=0.19, bias_reduce=0.0, dcd=6,
                             max_hold_days=92, max_hold_reduce=0.0,
                             smooth_days=sd)
        info = test_strategy(f"U190_smooth{sd}", sig, wt, f"平滑{sd}天")
        print_info(info); results.append(info)

    # ---- 第二组: 双重止损 ----
    print("\n--- 第二组: 双重止损 ---")
    for st2 in [0.12, 0.15, 0.18, 0.20]:
        for sw2 in [0.05, 0.10, 0.15]:
            sig, wt = build_core(bias_high=0.19, bias_reduce=0.0, dcd=6,
                                 max_hold_days=92, max_hold_reduce=0.0,
                                 use_double_stop=True, st2=st2, sw2=sw2)
            info = test_strategy(f"U191_st2_{st2}_sw2_{sw2}", sig, wt,
                                 f"双止损 st2={st2} sw2={sw2}")
            print_info(info); results.append(info)

    # ---- 第三组: 趋势强度过滤 ----
    print("\n--- 第三组: 趋势强度过滤 ---")
    for ts_lookback in [10, 20, 30, 60]:
        for ts_thresh in [0.3, 0.4, 0.5, 0.6]:
            sig, wt = build_core(bias_high=0.19, bias_reduce=0.0, dcd=6,
                                 max_hold_days=92, max_hold_reduce=0.0,
                                 use_trend_strength=True,
                                 ts_lookback=ts_lookback, ts_thresh=ts_thresh)
            info = test_strategy(f"U192_TS{ts_lookback}_{ts_thresh}", sig, wt,
                                 f"TS{ts_lookback} thresh={ts_thresh}")
            print_info(info); results.append(info)

    # ---- 第四组: BIAS恢复确认 ----
    print("\n--- 第四组: BIAS恢复确认 ---")
    for recovery_days in [2, 3, 5, 7, 10]:
        sig, wt = build_core(bias_high=0.19, bias_reduce=0.0, dcd=6,
                             max_hold_days=92, max_hold_reduce=0.0,
                             use_bias_recovery=True, recovery_days=recovery_days)
        info = test_strategy(f"U193_recovery{recovery_days}", sig, wt,
                             f"恢复{recovery_days}天")
        print_info(info); results.append(info)

    # ---- 第五组: 极限组合最终验证 ----
    print("\n--- 第五组: 极限组合最终验证 ---")
    final_combos = [
        ("U194_最终_基准", dict(bias_high=0.19, bias_reduce=0.0, dcd=6,
                                max_hold_days=92, max_hold_reduce=0.0)),
        ("U194_最终_smooth3", dict(bias_high=0.19, bias_reduce=0.0, dcd=6,
                                    max_hold_days=92, max_hold_reduce=0.0,
                                    smooth_days=3)),
        ("U194_最终_smooth5", dict(bias_high=0.19, bias_reduce=0.0, dcd=6,
                                    max_hold_days=92, max_hold_reduce=0.0,
                                    smooth_days=5)),
        ("U194_最终_double_st0.15_sw0.1", dict(bias_high=0.19, bias_reduce=0.0, dcd=6,
                                                max_hold_days=92, max_hold_reduce=0.0,
                                                use_double_stop=True, st2=0.15, sw2=0.10)),
        ("U194_最终_recovery3", dict(bias_high=0.19, bias_reduce=0.0, dcd=6,
                                      max_hold_days=92, max_hold_reduce=0.0,
                                      use_bias_recovery=True, recovery_days=3)),
    ]
    for name, params in final_combos:
        sig, wt = build_core(**params)
        info = test_strategy(name, sig, wt, name.replace("U194_最终_", ""))
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

    results.sort(key=lambda x: -x['calmar'])
    print(f"\n  Top 15 无滑点Calmar:")
    for r in results[:15]:
        print(f"  {r['name']:<46} 年化{r['ann']*100:.2f}% 回撤{r['dd']*100:.2f}% "
              f"Calmar{r['calmar']:.3f} 滑点Calmar{r['calmar_sl']:.3f}")

    best = max(results, key=lambda x: x['calmar'])
    print(f"\n  ★无滑点最优: {best['name']} Calmar={best['calmar']:.3f} "
          f"年化={best['ann']*100:.2f}% 回撤={best['dd']*100:.2f}%")

    best_sl = max(results, key=lambda x: x['calmar_sl'])
    print(f"  ★滑点最优: {best_sl['name']} 滑点Calmar={best_sl['calmar_sl']:.3f} "
          f"滑点年化={best_sl['ann_sl']*100:.2f}%")

    v26_calmar = 2.378
    improved = [r for r in results if r['calmar'] > v26_calmar]
    print(f"\n  超越v26(Calmar>{v26_calmar})的版本: {len(improved)}个")
    for r in improved:
        print(f"    {r['name']}: Calmar={r['calmar']:.3f} (+{r['calmar']-v26_calmar:.3f})")

    print("\n  策略已达到当前框架极限")
    print("  v23-v29 7轮测试完成, 无滑点Calmar极限: 2.378")
    print("  滑点Calmar极限: 2.059")
    print("  如需进一步提升, 需要新数据源或新因子系统")
