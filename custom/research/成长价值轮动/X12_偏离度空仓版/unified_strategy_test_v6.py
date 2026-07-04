"""unified_strategy_test_v6.py — 统一策略第六轮: 突破40%+减交易
================================================================
第五轮发现: slope=0.005/0.008比0.002更好, ft=0.5比0.3更好
第六轮: 组合最优参数, 目标年化>40% + 交易<300
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

OUTPUT_DIR = Path(r'c:\XuanJLH\Qbot\custom\research\成长价值轮动\X12_偏离度空仓版')
T = RATIO_DEV_Z
SLOPE = MA20_SLOPE


def build_v6(flat_thresh=0.5, reduce_thresh=1.5,
             slope_thresh=0.005,
             flat_weight=0.0, reduce_weight=0.5,
             dir_confirm=5,
             flat_confirm=0, reduce_confirm=0,
             use_b2=True, use_e5=True,
             stop_threshold=0.10, stop_weight=0.30, e5_cooldown=5):

    raw_dir = (T > 0).map({True: 'BULL', False: 'BEAR'})
    mask = np.ones(len(raw_dir), dtype=bool)
    for k in range(1, dir_confirm):
        mask = mask & (raw_dir.values == raw_dir.shift(k).values)
    confirmed_dir = raw_dir.where(mask, np.nan).ffill().fillna('BULL')

    weak_slope = SLOPE.abs() < slope_thresh
    weak_t_flat = T.abs() < flat_thresh
    weak_t_reduce = (T.abs() >= flat_thresh) & (T.abs() < reduce_thresh)

    state = 'FULL'
    states = []
    flat_count = 0
    reduce_count = 0

    for i in range(len(T)):
        if pd.isna(T.iloc[i]) or pd.isna(SLOPE.iloc[i]):
            states.append(state)
            continue
        is_flat_weak = bool(weak_t_flat.iloc[i] and weak_slope.iloc[i])
        is_reduce_weak = bool(weak_t_reduce.iloc[i] and weak_slope.iloc[i])

        if state == 'FULL':
            if is_flat_weak:
                flat_count += 1
                if flat_count >= max(1, flat_confirm):
                    state = 'FLAT'
                    reduce_count = 0
            elif is_reduce_weak:
                reduce_count += 1
                if reduce_count >= max(1, reduce_confirm):
                    state = 'REDUCE'
                    flat_count = 0
            else:
                flat_count = 0
                reduce_count = 0
        elif state == 'REDUCE':
            if is_flat_weak:
                flat_count += 1
                reduce_count = 0
                if flat_count >= max(1, flat_confirm):
                    state = 'FLAT'
            elif is_reduce_weak:
                flat_count = 0
            else:
                state = 'FULL'
                flat_count = 0
                reduce_count = 0
        else:  # FLAT
            if not is_flat_weak and not is_reduce_weak:
                state = 'FULL'
                flat_count = 0
                reduce_count = 0
            elif is_reduce_weak and not is_flat_weak:
                state = 'REDUCE'
                flat_count = 0
        states.append(state)

    states = pd.Series(states, index=T.index)
    dir_raw = confirmed_dir.copy()
    wt = pd.Series(1.0, index=T.index)
    wt[states == 'REDUCE'] = reduce_weight
    wt[states == 'FLAT'] = flat_weight

    if use_b2:
        wrong_value = (dir_raw == 'BEAR') & (V_MOM20 <= 0)
        dir_raw[wrong_value] = 'BULL'

    if use_e5:
        dir_s = dir_raw.map({'BULL': 'growth', 'BEAR': 'value'})
        gs = (dir_s == 'growth') & (G_DD20 < -stop_threshold)
        vs = (dir_s == 'value') & (V_DD20 < -stop_threshold)
        e5_trigger = gs | vs
        in_cooldown = False
        cooldown_count = 0
        for i in range(len(wt)):
            if pd.isna(wt.iloc[i]) or pd.isna(dir_s.iloc[i]):
                continue
            if e5_trigger.iloc[i] and not in_cooldown:
                in_cooldown = True
                cooldown_count = 0
                wt.iloc[i] = wt.iloc[i] * stop_weight
            elif in_cooldown:
                cooldown_count += 1
                if cooldown_count >= e5_cooldown:
                    if e5_trigger.iloc[i]:
                        cooldown_count = 0
                        wt.iloc[i] = wt.iloc[i] * stop_weight
                    else:
                        in_cooldown = False
                        if states.iloc[i] == 'REDUCE':
                            wt.iloc[i] = reduce_weight
                        elif states.iloc[i] == 'FLAT':
                            wt.iloc[i] = flat_weight
                        else:
                            wt.iloc[i] = 1.0
                else:
                    if wt.iloc[i] > 0:
                        wt.iloc[i] = stop_weight

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
          f"交易={info['n_trades']}(方向{info['dir_sw']}+仓位{info['cash_sw']})")


if __name__ == '__main__':
    print("=" * 80)
    print("  统一策略第六轮: 突破40%+减交易")
    print("=" * 80)

    results = []

    # ---- 第一组: slope=0.005 + ft=0.5/rt=1.5 + 确认扫描 ----
    print("\n--- 第一组: slope=0.005 ft=0.5 rt=1.5 ---")
    for fc, rc in [(0, 0), (2, 2), (3, 3), (2, 3), (3, 2)]:
        sig, wt = build_v6(flat_thresh=0.5, reduce_thresh=1.5,
                           slope_thresh=0.005, flat_confirm=fc, reduce_confirm=rc)
        info = test_strategy(f"U27_sl005_fc{fc}_rc{rc}", sig, wt, f"sl=0.005 fc={fc} rc={rc}")
        print_info(info); results.append(info)

    # ---- 第二组: slope=0.008 + 确认扫描 ----
    print("\n--- 第二组: slope=0.008 ft=0.5 rt=1.5 ---")
    for fc, rc in [(0, 0), (2, 2), (3, 3), (2, 3)]:
        sig, wt = build_v6(flat_thresh=0.5, reduce_thresh=1.5,
                           slope_thresh=0.008, flat_confirm=fc, reduce_confirm=rc)
        info = test_strategy(f"U28_sl008_fc{fc}_rc{rc}", sig, wt, f"sl=0.008 fc={fc} rc={rc}")
        print_info(info); results.append(info)

    # ---- 第三组: ft/rt组合 ----
    print("\n--- 第三组: ft/rt组合 (sl=0.005, fc=2, rc=2) ---")
    for ft, rt in [(0.3, 1.0), (0.3, 1.5), (0.5, 1.0), (0.5, 1.5),
                    (0.5, 2.0), (0.8, 1.5), (0.0, 1.0), (0.0, 1.5)]:
        sig, wt = build_v6(flat_thresh=ft, reduce_thresh=rt,
                           slope_thresh=0.005, flat_confirm=2, reduce_confirm=2)
        info = test_strategy(f"U29_ft{ft}_rt{rt}", sig, wt, f"ft={ft} rt={rt}")
        print_info(info); results.append(info)

    # ---- 第四组: rw扫描 ----
    print("\n--- 第四组: rw扫描 (sl=0.005, ft=0.5, rt=1.5, fc=2, rc=2) ---")
    for rw in [0.0, 0.3, 0.5, 0.7]:
        sig, wt = build_v6(flat_thresh=0.5, reduce_thresh=1.5,
                           slope_thresh=0.005, reduce_weight=rw,
                           flat_confirm=2, reduce_confirm=2)
        info = test_strategy(f"U30_rw{rw}", sig, wt, f"rw={rw}")
        print_info(info); results.append(info)

    # ---- 第五组: 方向确认 ----
    print("\n--- 第五组: 方向确认 (sl=0.005, ft=0.5, rt=1.5, fc=2, rc=2) ---")
    for dc in [3, 4, 5, 6, 7]:
        sig, wt = build_v6(flat_thresh=0.5, reduce_thresh=1.5,
                           slope_thresh=0.005, dir_confirm=dc,
                           flat_confirm=2, reduce_confirm=2)
        info = test_strategy(f"U31_dc{dc}", sig, wt, f"dc={dc}")
        print_info(info); results.append(info)

    # ---- 第六组: 最优组合 ----
    print("\n--- 第六组: 最优组合 ---")
    configs = [
        (0.5, 1.5, 0.005, 0.5, 5, 2, 2, "最优基线"),
        (0.5, 1.5, 0.005, 0.3, 5, 2, 2, "rw=0.3"),
        (0.5, 1.5, 0.005, 0.5, 5, 3, 2, "fc=3"),
        (0.5, 1.5, 0.005, 0.5, 5, 2, 3, "rc=3"),
        (0.5, 1.5, 0.006, 0.5, 5, 2, 2, "sl=0.006"),
        (0.5, 2.0, 0.005, 0.5, 5, 2, 2, "rt=2.0"),
        (0.3, 1.5, 0.005, 0.5, 5, 2, 2, "ft=0.3"),
        (0.5, 1.5, 0.005, 0.5, 4, 2, 2, "dc=4"),
        (0.5, 1.5, 0.005, 0.5, 6, 2, 2, "dc=6"),
        (0.5, 1.5, 0.004, 0.5, 5, 2, 2, "sl=0.004"),
        (0.5, 1.5, 0.005, 0.5, 5, 1, 1, "fc=1 rc=1"),
        (0.5, 1.5, 0.007, 0.5, 5, 2, 2, "sl=0.007"),
    ]
    for ft, rt, sl, rw, dc, fc, rc, label in configs:
        sig, wt = build_v6(flat_thresh=ft, reduce_thresh=rt,
                           slope_thresh=sl, reduce_weight=rw,
                           dir_confirm=dc, flat_confirm=fc, reduce_confirm=rc)
        info = test_strategy(f"U32_{label}", sig, wt, label)
        print_info(info); results.append(info)

    # ---- 汇总 ----
    print("\n" + "=" * 80)
    print("  汇总 (满足约束: 年化>40% 回撤<-35%)")
    print("=" * 80)
    ok = [r for r in results if r['ann'] > 0.40 and r['dd'] > -0.35]
    ok.sort(key=lambda x: x['n_trades'])
    print(f"\n  满足约束: {len(ok)}个版本\n")
    print(f"  {'名称':<30} {'年化':>7} {'回撤':>7} {'Sharpe':>7} {'Calmar':>7} "
          f"{'交易':>5} {'方向':>5} {'仓位':>5} {'滑点年化':>8}")
    print("  " + "-" * 100)
    print(f"  {'X61(基准)':<30} {'41.12%':>7} {'-30.21%':>7} {'1.357':>7} "
          f"{'1.361':>7} {'224':>5} {'179':>5} {'20':>5} {'39.04%':>8}")
    for r in ok:
        print(f"  {r['name']:<30} {r['ann']*100:>6.2f}% {r['dd']*100:>6.2f}% "
              f"{r['sharpe']:>7.3f} {r['calmar']:>7.3f} {r['n_trades']:>5} "
              f"{r['dir_sw']:>5} {r['cash_sw']:>5} {r['ann_sl']*100:>7.2f}%")

    print(f"\n  Top 15 (按Calmar降序):")
    results.sort(key=lambda x: -x['calmar'])
    for r in results[:15]:
        print(f"  {r['name']:<30} 年化{r['ann']*100:.2f}% 回撤{r['dd']*100:.2f}% "
              f"Calmar{r['calmar']:.3f} 交易{r['n_trades']}次")
