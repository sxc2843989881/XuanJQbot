"""unified_strategy_test_v5.py — 统一策略第五轮: 三档+状态机减交易
================================================================
第四轮突破: U19三档仓位 年化42.57% 回撤-27.28% Calmar1.561 (超X61!)
问题: 交易412次太多
第五轮: 三档+状态机滞回+冷却期, 目标<300次
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


def build_v5(flat_thresh=0.3, reduce_thresh=1.0,
             slope_thresh=0.002,
             flat_weight=0.0, reduce_weight=0.5,
             dir_confirm=5,
             flat_confirm=0, reduce_confirm=0,  # 进入降仓/空仓的确认天数
             use_b2=True, use_e5=True,
             stop_threshold=0.10, stop_weight=0.30, e5_cooldown=5):
    """v5: 三档仓位 + 状态机确认"""

    # 方向确认
    raw_dir = (T > 0).map({True: 'BULL', False: 'BEAR'})
    mask = np.ones(len(raw_dir), dtype=bool)
    for k in range(1, dir_confirm):
        mask = mask & (raw_dir.values == raw_dir.shift(k).values)
    confirmed_dir = raw_dir.where(mask, np.nan).ffill().fillna('BULL')

    # 趋势弱判断
    weak_slope = SLOPE.abs() < slope_thresh
    weak_t_flat = T.abs() < flat_thresh
    weak_t_reduce = (T.abs() >= flat_thresh) & (T.abs() < reduce_thresh)

    # 状态机: FULL/REDUCE/FLAT
    # FULL → REDUCE: 连续reduce_confirm天满足weak_t_reduce & weak_slope
    # FULL → FLAT: 连续flat_confirm天满足weak_t_flat & weak_slope
    # REDUCE/FLAT → FULL: 不满足弱条件
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
                else:
                    pass  # 待确认
            elif is_reduce_weak:
                reduce_count += 1
                if reduce_count >= max(1, reduce_confirm):
                    state = 'REDUCE'
                    flat_count = 0
                else:
                    pass
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
                # 保持REDUCE
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

    # 方向+权重
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
    print(f"    5/万滑点: 年化={info['ann_sl']*100:.2f}% Calmar={info['calmar_sl']:.3f}")


if __name__ == '__main__':
    print("=" * 80)
    print("  统一策略第五轮: 三档+状态机减交易")
    print("=" * 80)

    results = []

    # ---- 第一组: 基准(无确认) vs 加确认 ----
    print("\n--- 第一组: 确认天数扫描 (ft=0.3, rt=1.0, rw=0.5) ---")
    for fc, rc in [(0, 0), (2, 2), (3, 3), (4, 4), (5, 5), (3, 5), (5, 3)]:
        sig, wt = build_v5(flat_thresh=0.3, reduce_thresh=1.0,
                           flat_confirm=fc, reduce_confirm=rc)
        info = test_strategy(f"U21_fc{fc}_rc{rc}", sig, wt, f"flat确认{fc}天 reduce确认{rc}天")
        print_info(info); results.append(info)

    # ---- 第二组: rt扫描 + 确认5天 ----
    print("\n--- 第二组: rt扫描 (fc=3, rc=3) ---")
    for rt in [0.8, 1.0, 1.2, 1.5, 2.0]:
        sig, wt = build_v5(flat_thresh=0.3, reduce_thresh=rt,
                           flat_confirm=3, reduce_confirm=3)
        info = test_strategy(f"U22_rt{rt}", sig, wt, f"rt={rt} fc=3 rc=3")
        print_info(info); results.append(info)

    # ---- 第三组: rw扫描 + 确认 ----
    print("\n--- 第三组: rw扫描 (ft=0.3, rt=1.0, fc=3, rc=3) ---")
    for rw in [0.3, 0.5, 0.7, 0.8]:
        sig, wt = build_v5(flat_thresh=0.3, reduce_thresh=1.0,
                           reduce_weight=rw, flat_confirm=3, reduce_confirm=3)
        info = test_strategy(f"U23_rw{rw}", sig, wt, f"rw={rw}")
        print_info(info); results.append(info)

    # ---- 第四组: slope阈值扫描 ----
    print("\n--- 第四组: slope阈值扫描 (fc=3, rc=3) ---")
    for sl in [0.001, 0.002, 0.003, 0.005, 0.008]:
        sig, wt = build_v5(flat_thresh=0.3, reduce_thresh=1.0,
                           slope_thresh=sl, flat_confirm=3, reduce_confirm=3)
        info = test_strategy(f"U24_sl{sl}", sig, wt, f"slope={sl}")
        print_info(info); results.append(info)

    # ---- 第五组: 方向确认天数 ----
    print("\n--- 第五组: 方向确认天数 (fc=3, rc=3) ---")
    for dc in [3, 4, 5, 7, 10]:
        sig, wt = build_v5(flat_thresh=0.3, reduce_thresh=1.0,
                           dir_confirm=dc, flat_confirm=3, reduce_confirm=3)
        info = test_strategy(f"U25_dc{dc}", sig, wt, f"方向确认{dc}天")
        print_info(info); results.append(info)

    # ---- 第六组: 最优组合 ----
    print("\n--- 第六组: 最优组合 ---")
    for ft, rt, rw, fc, rc, dc in [
        (0.3, 1.0, 0.5, 3, 5, 5), (0.3, 1.0, 0.5, 5, 3, 5),
        (0.3, 1.5, 0.5, 3, 3, 5), (0.2, 1.0, 0.5, 3, 3, 5),
        (0.3, 1.0, 0.3, 3, 5, 5), (0.3, 1.0, 0.7, 3, 3, 5),
        (0.3, 1.0, 0.5, 4, 4, 5), (0.3, 1.0, 0.5, 3, 3, 7),
        (0.3, 1.0, 0.5, 5, 5, 5), (0.5, 1.5, 0.5, 3, 3, 5),
    ]:
        sig, wt = build_v5(flat_thresh=ft, reduce_thresh=rt, reduce_weight=rw,
                           flat_confirm=fc, reduce_confirm=rc, dir_confirm=dc)
        info = test_strategy(f"U26_ft{ft}_rt{rt}_rw{rw}_fc{fc}_rc{rc}_dc{dc}",
                              sig, wt, f"ft={ft} rt={rt} rw={rw} fc={fc} rc={rc} dc={dc}")
        print_info(info); results.append(info)

    # ---- 汇总 ----
    print("\n" + "=" * 80)
    print("  汇总 (满足约束: 年化>40% 回撤<-35%, 按交易次数升序)")
    print("=" * 80)

    ok = [r for r in results if r['ann'] > 0.40 and r['dd'] > -0.35]
    ok.sort(key=lambda x: x['n_trades'])
    print(f"\n  满足约束: {len(ok)}个版本\n")
    print(f"  {'名称':<40} {'年化':>7} {'回撤':>7} {'Sharpe':>7} {'Calmar':>7} "
          f"{'交易':>5} {'方向':>5} {'仓位':>5}")
    print("  " + "-" * 108)
    print(f"  {'X61(基准)':<40} {'41.12%':>7} {'-30.21%':>7} {'1.357':>7} "
          f"{'1.361':>7} {'224':>5} {'179':>5} {'20':>5}")
    for r in ok:
        print(f"  {r['name']:<40} {r['ann']*100:>6.2f}% {r['dd']*100:>6.2f}% "
              f"{r['sharpe']:>7.3f} {r['calmar']:>7.3f} {r['n_trades']:>5} "
              f"{r['dir_sw']:>5} {r['cash_sw']:>5}")

    # Top 10 全部
    print(f"\n  Top 10 (按Calmar降序):")
    results.sort(key=lambda x: -x['calmar'])
    for r in results[:10]:
        print(f"  {r['name']:<40} 年化{r['ann']*100:.2f}% 回撤{r['dd']*100:.2f}% "
              f"Calmar{r['calmar']:.3f} 交易{r['n_trades']}次")
