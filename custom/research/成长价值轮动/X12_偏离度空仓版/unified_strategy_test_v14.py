"""unified_strategy_test_v14.py — 第十四轮: B2改进基础上的深度优化
================================================================
第十三轮发现: B2(10,20)改进让Calmar达2.049(回撤-21.85%最低)
第十四轮: 在B2改进基础上深度优化, 尝试新因子
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
from unified_strategy_test_v9 import build_v9
from unified_strategy_test_v13 import build_v13_b2_improved

T = RATIO_DEV_Z
SLOPE = MA20_SLOPE
V_MOM10 = V_CLOSE.pct_change(10)
G_MOM10 = G_CLOSE.pct_change(10)
G_MOM20 = G_CLOSE.pct_change(20)


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


def build_v14_b2_extended(mom_short=10, mom_long=20, use_g=False,
                          reduce_thresh=1.3, slope_thresh=0.002,
                          reduce_weight=0.0,
                          dir_confirm=5, dir_cooldown=5,
                          use_e5=True,
                          stop_threshold=0.088, stop_weight=0.15, e5_cooldown=5):
    """B2扩展: 加入G_MOM判断, value方向且V弱且G强 → 改growth"""
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
    wt[is_weak] = reduce_weight

    # B2扩展: value方向 + V弱 + G强 → 改growth
    V_MOM_S = V_CLOSE.pct_change(mom_short)
    V_MOM_L = V_CLOSE.pct_change(mom_long)
    wrong_value = (dir_raw == 'BEAR') & (V_MOM_S <= 0) & (V_MOM_L <= 0)
    if use_g:
        G_MOM_S = G_CLOSE.pct_change(mom_short)
        G_MOM_L = G_CLOSE.pct_change(mom_long)
        # 加入G强条件: G_MOM_S > 0
        wrong_value = wrong_value & (G_MOM_S > 0)
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


def build_v14_e5_dual(stop_threshold_g=0.088, stop_threshold_v=0.088,
                       reduce_thresh=1.3, slope_thresh=0.002,
                       reduce_weight=0.0,
                       dir_confirm=5, dir_cooldown=5,
                       use_b2=True, use_b2_improved=True,
                       stop_weight=0.15, e5_cooldown=5):
    """E5双阈值: growth和value用不同止损阈值"""
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
    wt[is_weak] = reduce_weight

    if use_b2:
        if use_b2_improved:
            V_MOM10 = V_CLOSE.pct_change(10)
            wrong_value = (dir_raw == 'BEAR') & (V_MOM10 <= 0) & (V_MOM20 <= 0)
        else:
            wrong_value = (dir_raw == 'BEAR') & (V_MOM20 <= 0)
        dir_raw[wrong_value] = 'BULL'

    dir_s = dir_raw.map({'BULL': 'growth', 'BEAR': 'value'})
    # E5双阈值
    gs = (dir_s == 'growth') & (G_DD20 < -stop_threshold_g)
    vs = (dir_s == 'value') & (V_DD20 < -stop_threshold_v)
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


def build_v14_e5_trend(stop_threshold=0.088, trend_filter=True,
                       reduce_thresh=1.3, slope_thresh=0.002,
                       reduce_weight=0.0,
                       dir_confirm=5, dir_cooldown=5,
                       use_b2_improved=True,
                       stop_weight=0.15, e5_cooldown=5):
    """E5趋势过滤: 只在趋势反转时触发E5止损"""
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
    wt[is_weak] = reduce_weight

    if use_b2_improved:
        V_MOM10 = V_CLOSE.pct_change(10)
        wrong_value = (dir_raw == 'BEAR') & (V_MOM10 <= 0) & (V_MOM20 <= 0)
    else:
        wrong_value = (dir_raw == 'BEAR') & (V_MOM20 <= 0)
    dir_raw[wrong_value] = 'BULL'

    dir_s = dir_raw.map({'BULL': 'growth', 'BEAR': 'value'})
    gs = (dir_s == 'growth') & (G_DD20 < -stop_threshold)
    vs = (dir_s == 'value') & (V_DD20 < -stop_threshold)
    e5_trigger = gs | vs

    # 趋势过滤: E5只在趋势确认时触发
    if trend_filter:
        # growth方向: G_MOM10 < 0 (短期下跌) 才触发
        # value方向: V_MOM10 < 0 才触发
        G_MOM10 = G_CLOSE.pct_change(10)
        V_MOM10 = V_CLOSE.pct_change(10)
        e5_trigger = e5_trigger & (((dir_s == 'growth') & (G_MOM10 < 0)) |
                                    ((dir_s == 'value') & (V_MOM10 < 0)))

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


if __name__ == '__main__':
    print("=" * 80)
    print("  统一策略第十四轮: B2改进基础上的深度优化")
    print("=" * 80)

    results = []

    # ---- 第一组: B2扩展(V+G组合) ----
    print("\n--- 第一组: B2扩展(V+G组合) ---")
    for ms, ml, ug in [(10, 20, True), (10, 20, False), (5, 20, True),
                        (5, 10, True), (10, 30, True), (15, 30, True)]:
        sig, wt = build_v14_b2_extended(mom_short=ms, mom_long=ml, use_g=ug)
        info = test_strategy(f"U74_ms{ms}_ml{ml}_g{ug}", sig, wt,
                              f"短{ms}长{ml} G过滤={ug}")
        print_info(info); results.append(info)

    # ---- 第二组: B2改进+sw细调 ----
    print("\n--- 第二组: B2改进+sw细调 ---")
    for sw in [0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.20]:
        sig, wt = build_v13_b2_improved(mom_short=10, mom_long=20,
                                         stop_weight=sw, stop_threshold=0.088)
        info = test_strategy(f"U75_B2_sw{sw}", sig, wt, f"B2(10,20) sw={sw}")
        print_info(info); results.append(info)

    # ---- 第三组: B2改进+st细调 ----
    print("\n--- 第三组: B2改进+st细调 ---")
    for st in [0.080, 0.085, 0.088, 0.090, 0.092, 0.095]:
        sig, wt = build_v13_b2_improved(mom_short=10, mom_long=20,
                                         stop_weight=0.15, stop_threshold=st)
        info = test_strategy(f"U76_B2_st{st}", sig, wt, f"B2(10,20) st={st}")
        print_info(info); results.append(info)

    # ---- 第四组: B2改进+cd细调 ----
    print("\n--- 第四组: B2改进+cd细调 ---")
    for cd in [3, 4, 5, 6, 7, 8]:
        sig, wt = build_v13_b2_improved(mom_short=10, mom_long=20,
                                         stop_weight=0.15, stop_threshold=0.088,
                                         e5_cooldown=cd)
        info = test_strategy(f"U77_B2_cd{cd}", sig, wt, f"B2(10,20) cd={cd}")
        print_info(info); results.append(info)

    # ---- 第五组: E5双阈值 ----
    print("\n--- 第五组: E5双阈值 ---")
    for sg, sv in [(0.088, 0.088), (0.08, 0.09), (0.09, 0.08),
                    (0.085, 0.090), (0.090, 0.085), (0.08, 0.10),
                    (0.10, 0.08), (0.088, 0.10)]:
        sig, wt = build_v14_e5_dual(stop_threshold_g=sg, stop_threshold_v=sv,
                                     stop_weight=0.15)
        info = test_strategy(f"U78_sg{sg}_sv{sv}", sig, wt, f"G止损{sg} V止损{sv}")
        print_info(info); results.append(info)

    # ---- 第六组: E5趋势过滤 ----
    print("\n--- 第六组: E5趋势过滤 ---")
    for st, tf in [(0.088, True), (0.088, False), (0.080, True),
                    (0.090, True), (0.085, True), (0.095, True),
                    (0.10, True), (0.088, True)]:
        sig, wt = build_v14_e5_trend(stop_threshold=st, trend_filter=tf,
                                      stop_weight=0.15)
        info = test_strategy(f"U79_st{st}_tf{tf}", sig, wt, f"止损{st} 趋势过滤={tf}")
        print_info(info); results.append(info)

    # ---- 第七组: 综合最优 ----
    print("\n--- 第七组: 综合最优 ---")
    final_configs = [
        ("U80_B2(10,20)_sw0.15_st0.088", lambda: build_v13_b2_improved(mom_short=10, mom_long=20, stop_weight=0.15, stop_threshold=0.088)),
        ("U80_B2(10,20)_sw0.14_st0.088", lambda: build_v13_b2_improved(mom_short=10, mom_long=20, stop_weight=0.14, stop_threshold=0.088)),
        ("U80_B2(10,20)_sw0.16_st0.088", lambda: build_v13_b2_improved(mom_short=10, mom_long=20, stop_weight=0.16, stop_threshold=0.088)),
        ("U80_B2扩展_gTrue", lambda: build_v14_b2_extended(mom_short=10, mom_long=20, use_g=True, stop_weight=0.15, stop_threshold=0.088)),
        ("U80_E5双阈值_0.08_0.09", lambda: build_v14_e5_dual(stop_threshold_g=0.08, stop_threshold_v=0.09, stop_weight=0.15)),
        ("U80_E5趋势过滤_st0.088", lambda: build_v14_e5_trend(stop_threshold=0.088, trend_filter=True, stop_weight=0.15)),
        ("U80_E5趋势过滤_st0.080", lambda: build_v14_e5_trend(stop_threshold=0.080, trend_filter=True, stop_weight=0.15)),
        ("U80_v9基准_sw0.15", lambda: build_v9(reduce_thresh=1.3, stop_threshold=0.088, stop_weight=0.15, e5_cooldown=5)),
    ]
    for name, func in final_configs:
        sig, wt = func()
        info = test_strategy(name, sig, wt, name.replace("U80_", ""))
        print_info(info); results.append(info)

    # ---- 汇总 ----
    print("\n" + "=" * 80)
    print("  最终汇总 (满足约束: 年化>40% 回撤<-35%)")
    print("=" * 80)
    ok = [r for r in results if r['ann'] > 0.40 and r['dd'] > -0.35]
    ok.sort(key=lambda x: -x['calmar'])
    print(f"\n  满足约束: {len(ok)}个版本\n")
    print(f"  {'名称':<32} {'年化':>7} {'回撤':>7} {'Sharpe':>7} {'Calmar':>7} "
          f"{'交易':>5} {'滑点年化':>8}")
    print("  " + "-" * 84)
    print(f"  {'X61(基准)':<32} {'41.12%':>7} {'-30.21%':>7} {'1.357':>7} "
          f"{'1.361':>7} {'224':>5} {'39.04%':>8}")
    print(f"  {'U70(v13最优)':<32} {'44.77%':>7} {'-21.85%':>7} {'1.513':>7} "
          f"{'2.049':>7} {'389':>5} {'42.14%':>8}")
    for r in ok[:25]:
        print(f"  {r['name']:<32} {r['ann']*100:>6.2f}% {r['dd']*100:>6.2f}% "
              f"{r['sharpe']:>7.3f} {r['calmar']:>7.3f} {r['n_trades']:>5} "
              f"{r['ann_sl']*100:>7.2f}%")

    print(f"\n  Top 15 Calmar:")
    results.sort(key=lambda x: -x['calmar'])
    for r in results[:15]:
        print(f"  {r['name']:<32} 年化{r['ann']*100:.2f}% 回撤{r['dd']*100:.2f}% "
              f"Calmar{r['calmar']:.3f} 交易{r['n_trades']}次 滑点{r['ann_sl']*100:.2f}%")

    best = max(results, key=lambda x: x['calmar'])
    print(f"\n  ★最优: {best['name']} Calmar={best['calmar']:.3f} "
          f"年化={best['ann']*100:.2f}% 回撤={best['dd']*100:.2f}%")
