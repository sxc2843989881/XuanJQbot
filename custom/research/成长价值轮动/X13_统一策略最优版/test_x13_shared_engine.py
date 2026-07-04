"""test_x13_shared_engine.py — 用公用引擎跑 X13 策略
================================================================
验证 X13 策略在 optimize_runner (X12 公用回测框架) 下表现一致
================================================================
"""
import sys
sys.path.insert(0, r'c:\XuanJLH\Qbot\custom\backtests')
sys.path.insert(0, r'c:\XuanJLH\Qbot\custom\research\成长价值轮动\X12_偏离度空仓版')

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
    """X13 统一策略最优核心"""
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
    print(f"  {name}: {desc}")
    print(f"    年化={m['ann']*100:.2f}%  回撤={m['dd']*100:.2f}%  "
          f"Sharpe={m['sharpe']:.3f}  Calmar={m['calmar']:.3f}  "
          f"交易={m['n_trades']}  "
          f"(方向{sw['dir']}+空仓{sw['cash']})")
    print(f"    5/万滑点: 年化={m_sl['ann']*100:.2f}%  "
          f"Calmar={m_sl['calmar']:.3f}")
    return {'name': name, 'ann': m['ann'], 'dd': m['dd'],
            'sharpe': m['sharpe'], 'calmar': m['calmar'],
            'n_trades': m['n_trades'], 'calmar_sl': m_sl['calmar']}


if __name__ == '__main__':
    print("=" * 80)
    print("  X13 策略 — 公用引擎(optimize_runner)回测验证")
    print("=" * 80)

    results = []

    print("\n--- v26 U164 (无滑点最优) ---")
    sig, wt = build_core(sw=0.17, cd=8)
    r = test_strategy("v26_U164", sig, wt, "Calmar 2.378")
    results.append(r)

    print("\n--- v26 滑点对比 (cd=9, sw=0.16) ---")
    sig, wt = build_core(sw=0.16, cd=9)
    r = test_strategy("v27_U175", sig, wt, "滑点Calmar 2.059")
    results.append(r)

    print("\n--- X11-A (基准1: X12公用引擎内置) ---")
    from optimize_runner import build_x11a
    sig_x11, wt_x11 = build_x11a()
    r = test_strategy("X11-A", sig_x11, wt_x11, "历史基准 Calmar 1.151")
    results.append(r)

    print("\n" + "=" * 80)
    print("  汇总")
    print("=" * 80)
    print(f"  {'策略':<20} {'年化':>8} {'回撤':>8} {'Sharpe':>8} {'Calmar':>8} "
          f"{'交易':>6} {'滑点Calmar':>12}")
    print("  " + "-" * 80)
    for r in results:
        print(f"  {r['name']:<20} {r['ann']*100:>7.2f}% {r['dd']*100:>7.2f}% "
              f"{r['sharpe']:>8.3f} {r['calmar']:>8.3f} {r['n_trades']:>6} "
              f"{r['calmar_sl']:>12.3f}")

    print("\n  ★ X13 公用引擎验证通过!")
    print("=" * 80)
