"""unified_strategy_test_v9.py — 第九轮: 最终最优组合验证
================================================================
第八轮发现: st=0.08让Calmar达1.873, rt=1.3让年化达44.28%
第九轮: 组合最优参数 + 验证稳健性
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


def build_v9(flat_thresh=0.3, reduce_thresh=1.3,
             slope_thresh=0.002,
             reduce_weight=0.0,
             dir_confirm=5, dir_cooldown=5,
             use_b2=True, use_e5=True,
             stop_threshold=0.08, stop_weight=0.30, e5_cooldown=5):

    raw_dir = (T > 0).map({True: 'BULL', False: 'BEAR'})
    mask = np.ones(len(raw_dir), dtype=bool)
    for k in range(1, dir_confirm):
        mask = mask & (raw_dir.values == raw_dir.shift(k).values)
    confirmed_dir = raw_dir.where(mask, np.nan).ffill().fillna('BULL')

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
                new_dir.iloc[i] = prev
            else:
                new_dir.iloc[i] = prev
        confirmed_dir = new_dir

    dir_raw = confirmed_dir
    weak_slope = SLOPE.abs() < slope_thresh
    weak_t = T.abs() < reduce_thresh
    is_weak = weak_t & weak_slope

    wt = pd.Series(1.0, index=T.index)
    wt[is_weak] = reduce_weight  # 0.0 = 空仓

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
                        if is_weak.iloc[i]:
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
    print("  统一策略第九轮: 最终最优组合验证")
    print("=" * 80)

    results = []

    # ---- 第一组: rt=1.3 + st=0.08 组合 ----
    print("\n--- 第一组: rt=1.3 + st=0.08 组合 ---")
    for rt, st in [(1.3, 0.08), (1.3, 0.10), (1.5, 0.08), (1.5, 0.10),
                    (1.2, 0.08), (1.0, 0.08), (1.3, 0.06), (1.3, 0.12)]:
        sig, wt = build_v9(reduce_thresh=rt, stop_threshold=st)
        info = test_strategy(f"U44_rt{rt}_st{st}", sig, wt, f"rt={rt} st={st}")
        print_info(info); results.append(info)

    # ---- 第二组: E5参数细调 (rt=1.3) ----
    print("\n--- 第二组: E5参数细调 (rt=1.3) ---")
    for st, sw, cd in [(0.08, 0.3, 5), (0.08, 0.2, 5), (0.08, 0.3, 3),
                        (0.08, 0.3, 8), (0.06, 0.3, 5), (0.07, 0.3, 5),
                        (0.09, 0.3, 5), (0.08, 0.4, 5), (0.08, 0.3, 10)]:
        sig, wt = build_v9(reduce_thresh=1.3, stop_threshold=st,
                           stop_weight=sw, e5_cooldown=cd)
        info = test_strategy(f"U45_st{st}_sw{sw}_cd{cd}", sig, wt,
                              f"st={st} sw={sw} cool={cd}")
        print_info(info); results.append(info)

    # ---- 第三组: slope + rt组合 ----
    print("\n--- 第三组: slope+rt组合 (st=0.08) ---")
    for sl, rt in [(0.002, 1.3), (0.0015, 1.3), (0.002, 1.2),
                    (0.002, 1.0), (0.001, 1.3), (0.0025, 1.3),
                    (0.002, 1.5), (0.0015, 1.5)]:
        sig, wt = build_v9(reduce_thresh=rt, slope_thresh=sl, stop_threshold=0.08)
        info = test_strategy(f"U46_sl{sl}_rt{rt}", sig, wt, f"sl={sl} rt={rt}")
        print_info(info); results.append(info)

    # ---- 第四组: 方向确认+冷却 ----
    print("\n--- 第四组: 方向确认+冷却 (rt=1.3, st=0.08) ---")
    for dc, cd in [(5, 5), (4, 5), (5, 7), (7, 5), (5, 3), (5, 10), (3, 5), (6, 5)]:
        sig, wt = build_v9(reduce_thresh=1.3, stop_threshold=0.08,
                           dir_confirm=dc, dir_cooldown=cd)
        info = test_strategy(f"U47_dc{dc}_cd{cd}", sig, wt, f"dc={dc} cd={cd}")
        print_info(info); results.append(info)

    # ---- 第五组: 最终最优组合 ----
    print("\n--- 第五组: 最终最优组合 ---")
    configs = [
        (1.3, 0.002, 0.0, 5, 5, 0.08, 0.30, 5, "U48_最终最优"),
        (1.3, 0.002, 0.0, 5, 5, 0.08, 0.20, 5, "U48_sw0.2"),
        (1.3, 0.002, 0.0, 5, 5, 0.08, 0.30, 3, "U48_cool3"),
        (1.3, 0.002, 0.0, 5, 5, 0.06, 0.30, 5, "U48_st0.06"),
        (1.2, 0.002, 0.0, 5, 5, 0.08, 0.30, 5, "U48_rt1.2"),
        (1.3, 0.0015, 0.0, 5, 5, 0.08, 0.30, 5, "U48_sl0.0015"),
        (1.3, 0.002, 0.0, 4, 5, 0.08, 0.30, 5, "U48_dc4"),
        (1.5, 0.002, 0.0, 5, 5, 0.08, 0.30, 5, "U48_rt1.5"),
        (1.3, 0.002, 0.0, 5, 7, 0.08, 0.30, 5, "U48_cd7"),
        (1.3, 0.002, 0.0, 5, 5, 0.08, 0.30, 8, "U48_cool8"),
    ]
    for rt, sl, rw, dc, cd, st, sw, e5c, name in configs:
        sig, wt = build_v9(reduce_thresh=rt, slope_thresh=sl, reduce_weight=rw,
                           dir_confirm=dc, dir_cooldown=cd,
                           stop_threshold=st, stop_weight=sw, e5_cooldown=e5c)
        info = test_strategy(name, sig, wt, name.replace("U48_", ""))
        print_info(info); results.append(info)

    # ---- 汇总 ----
    print("\n" + "=" * 80)
    print("  最终汇总 (满足约束: 年化>40% 回撤<-35%)")
    print("=" * 80)
    ok = [r for r in results if r['ann'] > 0.40 and r['dd'] > -0.35]
    ok.sort(key=lambda x: -x['calmar'])
    print(f"\n  满足约束: {len(ok)}个版本\n")
    print(f"  {'名称':<28} {'年化':>7} {'回撤':>7} {'Sharpe':>7} {'Calmar':>7} "
          f"{'交易':>5} {'滑点年化':>8}")
    print("  " + "-" * 80)
    print(f"  {'X61(基准)':<28} {'41.12%':>7} {'-30.21%':>7} {'1.357':>7} "
          f"{'1.361':>7} {'224':>5} {'39.04%':>8}")
    for r in ok:
        print(f"  {r['name']:<28} {r['ann']*100:>6.2f}% {r['dd']*100:>6.2f}% "
              f"{r['sharpe']:>7.3f} {r['calmar']:>7.3f} {r['n_trades']:>5} "
              f"{r['ann_sl']*100:>7.2f}%")

    print(f"\n  Top 10 Calmar:")
    results.sort(key=lambda x: -x['calmar'])
    for r in results[:10]:
        print(f"  {r['name']:<28} 年化{r['ann']*100:.2f}% 回撤{r['dd']*100:.2f}% "
              f"Calmar{r['calmar']:.3f} 交易{r['n_trades']}次 滑点{r['ann_sl']*100:.2f}%")
