"""unified_strategy_test_v4.py — 统一策略第四轮: T+斜率组合 + 三档仓位
================================================================
第三轮最优: U14_nw0.7 年化38.06% 回撤-30.09% (差距X61约3pp)
关键发现: 统一版丢失了斜率信息
第四轮: T+斜率双重确认 + 三档仓位 + nw高位细调
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


def build_v4(entry_thresh=1.0, exit_thresh=0.3,
             slope_thresh=0.002, use_slope=True,
             neutral_weight=0.7, dir_confirm=5,
             use_b2=True, use_e5=True,
             stop_threshold=0.10, stop_weight=0.30, e5_cooldown=5):
    """v4: T+斜率双重确认 + 三态状态机

    进入NEUTRAL条件: |T|<entry AND |slope|<slope_thresh
    (双重确认, 减少误触发)
    """
    # 方向确认
    raw_dir = (T > 0).map({True: 'BULL', False: 'BEAR'})
    mask = np.ones(len(raw_dir), dtype=bool)
    for k in range(1, dir_confirm):
        mask = mask & (raw_dir.values == raw_dir.shift(k).values)
    confirmed_dir = raw_dir.where(mask, np.nan).ffill().fillna('BULL')

    # 趋势弱判断
    weak_t = T.abs() < entry_thresh
    if use_slope:
        weak_slope = SLOPE.abs() < slope_thresh
        is_weak = weak_t & weak_slope
    else:
        is_weak = weak_t

    # 状态机
    states = []
    state = 'NEUTRAL'
    for i in range(len(T)):
        t = T.iloc[i]
        d = confirmed_dir.iloc[i]
        w = is_weak.iloc[i] if not pd.isna(is_weak.iloc[i]) else False
        if pd.isna(t):
            states.append(state)
            continue

        if state == 'BULL':
            if t < exit_thresh and w:
                state = 'BEAR' if (t < -entry_thresh and d == 'BEAR') else 'NEUTRAL'
        elif state == 'BEAR':
            if t > -exit_thresh and w:
                state = 'BULL' if (t > entry_thresh and d == 'BULL') else 'NEUTRAL'
        else:
            if not w:
                if t > entry_thresh and d == 'BULL':
                    state = 'BULL'
                elif t < -entry_thresh and d == 'BEAR':
                    state = 'BEAR'
        states.append(state)

    states = pd.Series(states, index=T.index)
    dir_raw = states.where(states != 'NEUTRAL', np.nan).ffill().fillna('BULL')
    wt = pd.Series(1.0, index=T.index)
    wt[states == 'NEUTRAL'] = neutral_weight

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
                        if wt.iloc[i] > 0:
                            wt.iloc[i] = 1.0 if states.iloc[i] != 'NEUTRAL' else neutral_weight
                else:
                    if wt.iloc[i] > 0:
                        wt.iloc[i] = stop_weight

    signal = dir_raw.map({'BULL': 'growth', 'BEAR': 'value'})
    return signal, wt


def build_v4_three_tier(flat_thresh=0.3, reduce_thresh=1.0,
                        slope_thresh=0.002, use_slope=True,
                        flat_weight=0.0, reduce_weight=0.5,
                        dir_confirm=5,
                        use_b2=True, use_e5=True,
                        stop_threshold=0.10, stop_weight=0.30, e5_cooldown=5):
    """v4三档: |T|<flat_thresh → 空仓, <reduce_thresh → 降仓, else满仓"""
    raw_dir = (T > 0).map({True: 'BULL', False: 'BEAR'})
    mask = np.ones(len(raw_dir), dtype=bool)
    for k in range(1, dir_confirm):
        mask = mask & (raw_dir.values == raw_dir.shift(k).values)
    confirmed_dir = raw_dir.where(mask, np.nan).ffill().fillna('BULL')

    weak_t_flat = T.abs() < flat_thresh
    weak_t_reduce = (T.abs() >= flat_thresh) & (T.abs() < reduce_thresh)
    if use_slope:
        weak_slope = SLOPE.abs() < slope_thresh
        is_flat = weak_t_flat & weak_slope
        is_reduce = weak_t_reduce & weak_slope
    else:
        is_flat = weak_t_flat
        is_reduce = weak_t_reduce

    dir_raw = confirmed_dir.copy()
    wt = pd.Series(1.0, index=T.index)
    wt[is_flat] = flat_weight
    wt[is_reduce] = reduce_weight

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
                    # 恢复时重算
                    if not in_cooldown and wt.iloc[i] > 0:
                        if is_flat.iloc[i]:
                            wt.iloc[i] = flat_weight
                        elif is_reduce.iloc[i]:
                            wt.iloc[i] = reduce_weight
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
    print(f"    5/万滑点: 年化={info['ann_sl']*100:.2f}% Calmar={info['calmar_sl']:.3f}")


if __name__ == '__main__':
    print("=" * 80)
    print("  统一策略第四轮: T+斜率双重确认 + 三档仓位")
    print("=" * 80)

    results = []

    # ---- 第一组: T+斜率 vs 纯T (nw高位) ----
    print("\n--- 第一组: T+斜率 vs 纯T (entry=1.0, nw=0.7) ---")
    for use_sl, label in [(True, "T+斜率"), (False, "纯T")]:
        for sl_th in [0.002, 0.003, 0.005]:
            sig, wt = build_v4(entry_thresh=1.0, exit_thresh=0.3,
                               slope_thresh=sl_th, use_slope=use_sl,
                               neutral_weight=0.7, dir_confirm=5)
            info = test_strategy(f"U16_{label}_sl{sl_th}", sig, wt,
                                  f"{label} slope={sl_th}")
            print_info(info); results.append(info)

    # ---- 第二组: nw高位细调 ----
    print("\n--- 第二组: nw高位细调 (entry=1.0, slope=0.002, dc=5) ---")
    for nw in [0.6, 0.7, 0.8, 0.9, 1.0]:
        sig, wt = build_v4(entry_thresh=1.0, exit_thresh=0.3,
                           slope_thresh=0.002, use_slope=True,
                           neutral_weight=nw, dir_confirm=5)
        info = test_strategy(f"U17_nw{nw}", sig, wt, f"nw={nw}")
        print_info(info); results.append(info)

    # ---- 第三组: entry+exit扫描 ----
    print("\n--- 第三组: entry+exit扫描 (slope=0.002, nw=0.7, dc=5) ---")
    for et, xt in [(0.5, 0.0), (0.8, 0.2), (1.0, 0.3), (1.2, 0.5), (1.5, 0.5),
                    (0.5, 0.3), (1.0, 0.0), (1.5, 0.0)]:
        sig, wt = build_v4(entry_thresh=et, exit_thresh=xt,
                           slope_thresh=0.002, use_slope=True,
                           neutral_weight=0.7, dir_confirm=5)
        info = test_strategy(f"U18_e{et}_x{xt}", sig, wt, f"entry={et} exit={xt}")
        print_info(info); results.append(info)

    # ---- 第四组: 三档仓位 ----
    print("\n--- 第四组: 三档仓位 (空仓+降仓+满仓) ---")
    for ft, rt, fw, rw in [(0.3, 1.0, 0.0, 0.5), (0.3, 1.0, 0.0, 0.7),
                            (0.2, 0.8, 0.0, 0.5), (0.5, 1.5, 0.0, 0.5),
                            (0.3, 1.0, 0.3, 0.7), (0.2, 1.0, 0.0, 0.3),
                            (0.3, 1.5, 0.0, 0.5), (0.0, 1.0, 0.0, 0.5)]:
        sig, wt = build_v4_three_tier(flat_thresh=ft, reduce_thresh=rt,
                                       slope_thresh=0.002, use_slope=True,
                                       flat_weight=fw, reduce_weight=rw,
                                       dir_confirm=5)
        info = test_strategy(f"U19_ft{ft}_rt{rt}_fw{fw}_rw{rw}", sig, wt,
                              f"flat<{ft}→{fw}, <{rt}→{rw}")
        print_info(info); results.append(info)

    # ---- 第五组: 最优组合 ----
    print("\n--- 第五组: 最优组合细调 ---")
    for et, xt, nw, dc in [(1.0, 0.3, 0.8, 4), (1.0, 0.3, 0.8, 5),
                            (0.8, 0.2, 0.8, 5), (1.0, 0.0, 0.8, 5),
                            (1.2, 0.3, 0.7, 5), (1.0, 0.3, 0.9, 4),
                            (0.8, 0.0, 0.7, 5), (1.0, 0.3, 0.7, 4)]:
        sig, wt = build_v4(entry_thresh=et, exit_thresh=xt,
                           slope_thresh=0.002, use_slope=True,
                           neutral_weight=nw, dir_confirm=dc)
        info = test_strategy(f"U20_e{et}_x{xt}_nw{nw}_dc{dc}", sig, wt,
                              f"e={et} x={xt} nw={nw} dc={dc}")
        print_info(info); results.append(info)

    # ---- 汇总 ----
    print("\n" + "=" * 80)
    print("  汇总 (按年化降序, 前20)")
    print("=" * 80)
    print(f"  {'名称':<30} {'年化':>7} {'回撤':>7} {'Sharpe':>7} {'Calmar':>7} "
          f"{'交易':>5} {'方向':>5} {'仓位':>5} {'滑点年化':>8}")
    print("  " + "-" * 98)
    print(f"  {'X61(基准)':<30} {'41.12%':>7} {'-30.21%':>7} {'1.357':>7} "
          f"{'1.361':>7} {'224':>5} {'179':>5} {'20':>5} {'39.04%':>8}")

    results.sort(key=lambda x: -x['ann'])
    for r in results[:20]:
        print(f"  {r['name']:<30} {r['ann']*100:>6.2f}% {r['dd']*100:>6.2f}% "
              f"{r['sharpe']:>7.3f} {r['calmar']:>7.3f} {r['n_trades']:>5} "
              f"{r['dir_sw']:>5} {r['cash_sw']:>5} {r['ann_sl']*100:>7.2f}%")

    ok = [r for r in results if r['ann'] > 0.40 and r['dd'] > -0.35]
    print(f"\n  满足约束(年化>40% 回撤<-35%): {len(ok)}个版本")
    for r in ok:
        print(f"    {r['name']}: 年化{r['ann']*100:.2f}% 回撤{r['dd']*100:.2f}% "
              f"交易{r['n_trades']}次 Calmar{r['calmar']:.3f}")
