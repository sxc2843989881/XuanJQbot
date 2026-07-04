"""unified_strategy_test_v7.py — 第七轮: 三档无状态机+方向冷却期
================================================================
关键发现: 状态机确认损害收益(42.57%→39.10%)
回到第四轮三档(无确认)基础上, 用方向冷却期减少交易
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


def build_v7(flat_thresh=0.3, reduce_thresh=1.5,
             slope_thresh=0.002,
             flat_weight=0.0, reduce_weight=0.5,
             dir_confirm=5, dir_cooldown=0,  # 方向切换冷却期
             use_b2=True, use_e5=True,
             stop_threshold=0.10, stop_weight=0.30, e5_cooldown=5):
    """v7: 三档仓位(无状态机确认) + 方向冷却期"""

    # 方向: T符号 + 确认
    raw_dir = (T > 0).map({True: 'BULL', False: 'BEAR'})
    mask = np.ones(len(raw_dir), dtype=bool)
    for k in range(1, dir_confirm):
        mask = mask & (raw_dir.values == raw_dir.shift(k).values)
    confirmed_dir = raw_dir.where(mask, np.nan).ffill().fillna('BULL')

    # 方向冷却期: 切换后dir_cooldown天内不切换
    if dir_cooldown > 0:
        new_dir = confirmed_dir.copy()
        last_switch = -dir_cooldown - 1
        prev = confirmed_dir.iloc[0]
        for i in range(len(confirmed_dir)):
            if pd.isna(confirmed_dir.iloc[i]):
                new_dir.iloc[i] = prev
                continue
            if confirmed_dir.iloc[i] != prev:
                if i - last_switch >= dir_cooldown:
                    last_switch = i
                    prev = confirmed_dir.iloc[i]
                else:
                    new_dir.iloc[i] = prev
            else:
                new_dir.iloc[i] = prev
        confirmed_dir = new_dir

    dir_raw = confirmed_dir

    # 三档仓位(无状态机)
    weak_slope = SLOPE.abs() < slope_thresh
    weak_t_flat = T.abs() < flat_thresh
    weak_t_reduce = (T.abs() >= flat_thresh) & (T.abs() < reduce_thresh)
    is_flat = weak_t_flat & weak_slope
    is_reduce = weak_t_reduce & weak_slope

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


if __name__ == '__main__':
    print("=" * 80)
    print("  统一策略第七轮: 三档+方向冷却期")
    print("=" * 80)

    results = []

    # ---- 第一组: 方向冷却期扫描 (基准参数) ----
    print("\n--- 第一组: 方向冷却期扫描 (ft=0.3, rt=1.5, sl=0.002, dc=5) ---")
    for cd in [0, 3, 5, 7, 10, 15, 20]:
        sig, wt = build_v7(flat_thresh=0.3, reduce_thresh=1.5,
                           slope_thresh=0.002, dir_confirm=5, dir_cooldown=cd)
        info = test_strategy(f"U33_cd{cd}", sig, wt, f"方向冷却{cd}天")
        print_info(info); results.append(info)

    # ---- 第二组: 方向确认+冷却组合 ----
    print("\n--- 第二组: 方向确认+冷却组合 ---")
    for dc, cd in [(5, 5), (5, 7), (7, 5), (7, 7), (4, 5), (5, 10), (3, 7), (10, 3)]:
        sig, wt = build_v7(flat_thresh=0.3, reduce_thresh=1.5,
                           slope_thresh=0.002, dir_confirm=dc, dir_cooldown=cd)
        info = test_strategy(f"U34_dc{dc}_cd{cd}", sig, wt, f"确认{dc}天+冷却{cd}天")
        print_info(info); results.append(info)

    # ---- 第三组: ft/rt/sl组合 + 冷却 ----
    print("\n--- 第三组: ft/rt/sl组合 + 冷却5天 ---")
    for ft, rt, sl in [(0.3, 1.5, 0.002), (0.3, 1.0, 0.002), (0.5, 1.5, 0.002),
                        (0.3, 1.5, 0.003), (0.2, 1.0, 0.002), (0.3, 2.0, 0.002),
                        (0.0, 1.5, 0.002), (0.3, 1.5, 0.001)]:
        sig, wt = build_v7(flat_thresh=ft, reduce_thresh=rt,
                           slope_thresh=sl, dir_confirm=5, dir_cooldown=5)
        info = test_strategy(f"U35_ft{ft}_rt{rt}_sl{sl}", sig, wt,
                              f"ft={ft} rt={rt} sl={sl} cd=5")
        print_info(info); results.append(info)

    # ---- 第四组: rw + 冷却 ----
    print("\n--- 第四组: rw扫描 (cd=5) ---")
    for rw in [0.0, 0.3, 0.5, 0.7]:
        sig, wt = build_v7(flat_thresh=0.3, reduce_thresh=1.5,
                           slope_thresh=0.002, reduce_weight=rw,
                           dir_confirm=5, dir_cooldown=5)
        info = test_strategy(f"U36_rw{rw}_cd5", sig, wt, f"rw={rw} cd=5")
        print_info(info); results.append(info)

    # ---- 第五组: 最优组合 ----
    print("\n--- 第五组: 最优组合 ---")
    configs = [
        (0.3, 1.5, 0.002, 0.5, 5, 5, "基线"),
        (0.3, 1.5, 0.002, 0.5, 5, 7, "cd=7"),
        (0.3, 1.5, 0.002, 0.3, 5, 5, "rw=0.3"),
        (0.3, 1.5, 0.002, 0.5, 7, 5, "dc=7"),
        (0.3, 1.5, 0.003, 0.5, 5, 5, "sl=0.003"),
        (0.2, 1.5, 0.002, 0.5, 5, 5, "ft=0.2"),
        (0.3, 1.0, 0.002, 0.5, 5, 5, "rt=1.0"),
        (0.3, 1.5, 0.002, 0.5, 5, 10, "cd=10"),
        (0.3, 1.5, 0.002, 0.5, 4, 7, "dc=4 cd=7"),
        (0.5, 1.5, 0.002, 0.5, 5, 5, "ft=0.5"),
        (0.3, 2.0, 0.002, 0.5, 5, 5, "rt=2.0"),
        (0.3, 1.5, 0.002, 0.0, 5, 5, "rw=0.0"),
    ]
    for ft, rt, sl, rw, dc, cd, label in configs:
        sig, wt = build_v7(flat_thresh=ft, reduce_thresh=rt,
                           slope_thresh=sl, reduce_weight=rw,
                           dir_confirm=dc, dir_cooldown=cd)
        info = test_strategy(f"U37_{label}", sig, wt, label)
        print_info(info); results.append(info)

    # ---- 汇总 ----
    print("\n" + "=" * 80)
    print("  汇总 (满足约束: 年化>40% 回撤<-35%, 按交易升序)")
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
