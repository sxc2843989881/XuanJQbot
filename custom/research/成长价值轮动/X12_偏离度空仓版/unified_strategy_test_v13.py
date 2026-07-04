"""unified_strategy_test_v13.py — 第十三轮: 新逻辑突破尝试
================================================================
第十二轮确认: sw=0.15是甜点, Calmar 2.048已是当前框架极限
第十三轮: 尝试新逻辑突破
  1. T指标平滑(5日均值) - 减少噪声
  2. B2过滤改进(V_MOM组合)
  3. T+斜率加权得分 - 替代AND
  4. E5动态阈值(ATR)
  5. 方向T的5日均值
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

T = RATIO_DEV_Z
SLOPE = MA20_SLOPE


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


def build_v13_smooth_t(smooth_window=5, reduce_thresh=1.3,
                        slope_thresh=0.002, reduce_weight=0.0,
                        dir_confirm=5, dir_cooldown=5,
                        use_b2=True, use_e5=True,
                        stop_threshold=0.088, stop_weight=0.15, e5_cooldown=5):
    """T指标平滑: 用T的smooth_window日均值, 减少噪声"""
    T_smooth = T.rolling(smooth_window).mean()

    raw_dir = (T_smooth > 0).map({True: 'BULL', False: 'BEAR'})
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
    weak_t = T_smooth.abs() < reduce_thresh
    is_weak = weak_t & weak_slope

    wt = pd.Series(1.0, index=T_smooth.index)
    wt[is_weak] = reduce_weight

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


def build_v13_score_based(t_weight=0.7, slope_weight=0.3,
                           entry_score=0.8, exit_score=0.3,
                           reduce_thresh=1.3, slope_thresh=0.002,
                           reduce_weight=0.0,
                           dir_confirm=5, dir_cooldown=5,
                           use_b2=True, use_e5=True,
                           stop_threshold=0.088, stop_weight=0.15, e5_cooldown=5):
    """T+斜率加权得分: 替代AND逻辑
    score = t_weight * (|T|<rt) + slope_weight * (|slope|<sl)
    score > entry_score → 空仓
    score < exit_score → 满仓
    """
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

    # 加权得分
    score = pd.Series(0.0, index=T.index)
    score = score + t_weight * weak_t.astype(float)
    score = score + slope_weight * weak_slope.astype(float)

    wt = pd.Series(1.0, index=T.index)
    is_flat = False
    is_weak = pd.Series(False, index=T.index)
    for i in range(len(T)):
        if pd.isna(T.iloc[i]) or pd.isna(SLOPE.iloc[i]):
            wt.iloc[i] = 1.0
            continue
        if not is_flat:
            if score.iloc[i] >= entry_score:
                is_flat = True
                wt.iloc[i] = reduce_weight
                is_weak.iloc[i] = True
            else:
                wt.iloc[i] = 1.0
        else:
            if score.iloc[i] <= exit_score:
                is_flat = False
                wt.iloc[i] = 1.0
            else:
                wt.iloc[i] = reduce_weight
                is_weak.iloc[i] = True

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


def build_v13_b2_improved(mom_short=10, mom_long=20,
                           reduce_thresh=1.3, slope_thresh=0.002,
                           reduce_weight=0.0,
                           dir_confirm=5, dir_cooldown=5,
                           use_e5=True,
                           stop_threshold=0.088, stop_weight=0.15, e5_cooldown=5):
    """B2改进: 短期+长期动量组合判断"""
    V_MOM10 = V_CLOSE.pct_change(mom_short)
    V_MOM_L = V_CLOSE.pct_change(mom_long)

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

    # B2改进: 短期和长期动量都<=0才改growth
    wrong_value = (dir_raw == 'BEAR') & (V_MOM10 <= 0) & (V_MOM_L <= 0)
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


if __name__ == '__main__':
    print("=" * 80)
    print("  统一策略第十三轮: 新逻辑突破尝试")
    print("=" * 80)

    results = []

    # ---- 第一组: T指标平滑 ----
    print("\n--- 第一组: T指标平滑 ---")
    for sw in [3, 5, 7, 10]:
        sig, wt = build_v13_smooth_t(smooth_window=sw)
        info = test_strategy(f"U68_smoothT{sw}", sig, wt, f"T平滑{sw}日")
        print_info(info); results.append(info)

    # ---- 第二组: T+斜率加权得分 ----
    print("\n--- 第二组: T+斜率加权得分 ---")
    for tw, sw_, es, xs in [(0.7, 0.3, 0.8, 0.3), (0.6, 0.4, 0.8, 0.3),
                              (0.5, 0.5, 0.8, 0.3), (0.7, 0.3, 1.0, 0.3),
                              (0.7, 0.3, 0.7, 0.3), (0.7, 0.3, 0.9, 0.4),
                              (0.8, 0.2, 0.8, 0.3), (0.7, 0.3, 0.8, 0.5)]:
        sig, wt = build_v13_score_based(t_weight=tw, slope_weight=sw_,
                                         entry_score=es, exit_score=xs)
        info = test_strategy(f"U69_tw{tw}_sw{sw_}_es{es}", sig, wt,
                              f"tw={tw} sw={sw_} 进{es}出{xs}")
        print_info(info); results.append(info)

    # ---- 第三组: B2改进 ----
    print("\n--- 第三组: B2改进 ---")
    for ms, ml in [(10, 20), (5, 20), (10, 30), (15, 30), (10, 10), (5, 10)]:
        sig, wt = build_v13_b2_improved(mom_short=ms, mom_long=ml)
        info = test_strategy(f"U70_ms{ms}_ml{ml}", sig, wt, f"短{ms}长{ml}")
        print_info(info); results.append(info)

    # ---- 第四组: T平滑 + sw=0.15 ----
    print("\n--- 第四组: T平滑+sw=0.15 ---")
    for sw_t, sw_w in [(3, 0.15), (5, 0.15), (7, 0.15), (5, 0.20),
                        (3, 0.20), (7, 0.20), (5, 0.10), (5, 0.18)]:
        sig, wt = build_v13_smooth_t(smooth_window=sw_t, stop_weight=sw_w)
        info = test_strategy(f"U71_sT{sw_t}_sw{sw_w}", sig, wt,
                              f"T平滑{sw_t} sw={sw_w}")
        print_info(info); results.append(info)

    # ---- 第五组: 综合最优组合 ----
    print("\n--- 第五组: 综合最优组合 ---")
    # T平滑5 + sw=0.15
    sig, wt = build_v13_smooth_t(smooth_window=5, stop_weight=0.15,
                                  stop_threshold=0.088)
    info = test_strategy("U72_平滑5_最优", sig, wt, "T5+sw0.15+st0.088")
    print_info(info); results.append(info)

    # T平滑3 + sw=0.15
    sig, wt = build_v13_smooth_t(smooth_window=3, stop_weight=0.15,
                                  stop_threshold=0.088)
    info = test_strategy("U72_平滑3_最优", sig, wt, "T3+sw0.15+st0.088")
    print_info(info); results.append(info)

    # B2改进(10,20) + sw=0.15
    sig, wt = build_v13_b2_improved(mom_short=10, mom_long=20,
                                     stop_weight=0.15, stop_threshold=0.088)
    info = test_strategy("U72_B2改进_最优", sig, wt, "B2(10,20)+sw0.15+st0.088")
    print_info(info); results.append(info)

    # B2改进(5,20) + sw=0.15
    sig, wt = build_v13_b2_improved(mom_short=5, mom_long=20,
                                     stop_weight=0.15, stop_threshold=0.088)
    info = test_strategy("U72_B2改进5_最优", sig, wt, "B2(5,20)+sw0.15+st0.088")
    print_info(info); results.append(info)

    # 加权得分 + sw=0.15
    sig, wt = build_v13_score_based(t_weight=0.7, slope_weight=0.3,
                                     entry_score=0.8, exit_score=0.3,
                                     stop_weight=0.15, stop_threshold=0.088)
    info = test_strategy("U72_得分_最优", sig, wt, "得分(0.7/0.3)+sw0.15+st0.088")
    print_info(info); results.append(info)

    # T平滑3 + B2改进 + sw=0.15
    V_MOM10 = V_CLOSE.pct_change(10)
    sig, wt = build_v13_smooth_t(smooth_window=3, stop_weight=0.15,
                                  stop_threshold=0.088)
    # 重新构建带B2改进的版本
    info = test_strategy("U72_综合_v1", sig, wt, "综合v1")
    print_info(info); results.append(info)

    # ---- 第六组: 最终候选 ----
    print("\n--- 第六组: 最终候选 ---")
    final_configs = [
        ("U73_平滑3_sw0.15", lambda: build_v13_smooth_t(smooth_window=3, stop_weight=0.15, stop_threshold=0.088)),
        ("U73_平滑5_sw0.15", lambda: build_v13_smooth_t(smooth_window=5, stop_weight=0.15, stop_threshold=0.088)),
        ("U73_B2(10,20)_sw0.15", lambda: build_v13_b2_improved(mom_short=10, mom_long=20, stop_weight=0.15, stop_threshold=0.088)),
        ("U73_B2(5,20)_sw0.15", lambda: build_v13_b2_improved(mom_short=5, mom_long=20, stop_weight=0.15, stop_threshold=0.088)),
        ("U73_v9基准_sw0.15", lambda: build_v9(reduce_thresh=1.3, stop_threshold=0.088, stop_weight=0.15, e5_cooldown=5)),
        ("U73_v9基准_sw0.16", lambda: build_v9(reduce_thresh=1.3, stop_threshold=0.088, stop_weight=0.16, e5_cooldown=5)),
        ("U73_得分0.7_0.3", lambda: build_v13_score_based(t_weight=0.7, slope_weight=0.3, entry_score=0.8, exit_score=0.3, stop_weight=0.15, stop_threshold=0.088)),
        ("U73_得分0.5_0.5", lambda: build_v13_score_based(t_weight=0.5, slope_weight=0.5, entry_score=0.8, exit_score=0.3, stop_weight=0.15, stop_threshold=0.088)),
    ]
    for name, func in final_configs:
        sig, wt = func()
        info = test_strategy(name, sig, wt, name.replace("U73_", ""))
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
    print(f"  {'U62(v12最优)':<32} {'44.92%':>7} {'-21.93%':>7} {'1.509':>7} "
          f"{'2.048':>7} {'365':>5} {'42.35%':>8}")
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
