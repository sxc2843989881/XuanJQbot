"""unified_strategy_test_v10.py — 第十轮: st=0.09甜点细调 + 新逻辑探索
================================================================
第九轮发现: st=0.09让Calmar达1.930(年化44.87%/回撤-23.25%)
第十轮: 在st=0.09附近细调, 并尝试新逻辑改进
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

OUTPUT_DIR = Path(r'c:\XuanJLH\Qbot\custom\research\成长价值轮动\X12_偏离度空仓版')
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


def build_v10_multi_period_t(t_short=5, t_long=20, weights=(0.5, 0.5),
                              flat_thresh=0.3, reduce_thresh=1.3,
                              slope_thresh=0.002, reduce_weight=0.0,
                              dir_confirm=5, dir_cooldown=5,
                              use_b2=True, use_e5=True,
                              stop_threshold=0.09, stop_weight=0.30, e5_cooldown=5):
    """多周期T指标: 短期T和长期T加权平均, 平滑噪声"""
    ratio_ma_short = RATIO.rolling(t_short).mean()
    ratio_ma_long = RATIO.rolling(t_long).mean()
    dev_short = RATIO / ratio_ma_short - 1
    dev_long = RATIO / ratio_ma_long - 1
    std_short = dev_short.rolling(20).std()
    std_long = dev_long.rolling(20).std()
    z_short = dev_short / std_short
    z_long = dev_long / std_long
    w_s, w_l = weights
    T_multi = w_s * z_short + w_l * z_long

    raw_dir = (T_multi > 0).map({True: 'BULL', False: 'BEAR'})
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
    weak_t = T_multi.abs() < reduce_thresh
    is_weak = weak_t & weak_slope

    wt = pd.Series(1.0, index=T_multi.index)
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


def build_v10_dynamic_e5(vol_window=20, vol_thresh=0.03,
                          tight_stop=0.07, loose_stop=0.10,
                          flat_thresh=0.3, reduce_thresh=1.3,
                          slope_thresh=0.002, reduce_weight=0.0,
                          dir_confirm=5, dir_cooldown=5,
                          use_b2=True, stop_weight=0.30, e5_cooldown=5):
    """动态E5止损: 高波动期更严格(7%), 低波动期更宽松(10%)"""
    vol = G_CLOSE.pct_change().rolling(vol_window).std()
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
        wrong_value = (dir_raw == 'BEAR') & (V_MOM20 <= 0)
        dir_raw[wrong_value] = 'BULL'

    dir_s = dir_raw.map({'BULL': 'growth', 'BEAR': 'value'})
    # 动态阈值
    stop_threshold = pd.Series(loose_stop, index=T.index)
    stop_threshold[vol > vol_thresh] = tight_stop

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


def build_v10_hysteresis(entry_thresh=1.3, exit_thresh=0.8,
                          slope_thresh=0.002, reduce_weight=0.0,
                          dir_confirm=5, dir_cooldown=5,
                          use_b2=True, use_e5=True,
                          stop_threshold=0.09, stop_weight=0.30, e5_cooldown=5):
    """滞回带: 进入空仓需|T|<entry_thresh, 退出空仓需|T|>exit_thresh
    比单一阈值更稳定, 减少频繁切换"""
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

    # 滞回带仓位
    wt = pd.Series(1.0, index=T.index)
    is_flat = False
    for i in range(len(T)):
        if pd.isna(T.iloc[i]):
            wt.iloc[i] = 1.0
            continue
        t_val = abs(T.iloc[i])
        slope_weak = abs(SLOPE.iloc[i]) < slope_thresh
        if not is_flat:
            if t_val < entry_thresh and slope_weak:
                is_flat = True
                wt.iloc[i] = reduce_weight
            else:
                wt.iloc[i] = 1.0
        else:
            if t_val > exit_thresh:
                is_flat = False
                wt.iloc[i] = 1.0
            else:
                wt.iloc[i] = reduce_weight
    is_weak = wt == reduce_weight

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


if __name__ == '__main__':
    print("=" * 80)
    print("  统一策略第十轮: st=0.09甜点细调 + 新逻辑探索")
    print("=" * 80)

    results = []

    # ---- 第一组: st在0.085-0.095之间细调 ----
    print("\n--- 第一组: st甜点细调 (rt=1.3) ---")
    for st in [0.082, 0.085, 0.088, 0.090, 0.092, 0.095, 0.098]:
        sig, wt = build_v9(reduce_thresh=1.3, stop_threshold=st)
        info = test_strategy(f"U49_st{st}", sig, wt, f"st={st}")
        print_info(info); results.append(info)

    # ---- 第二组: rt在1.25-1.35细调 (st=0.09) ----
    print("\n--- 第二组: rt细调 (st=0.09) ---")
    for rt in [1.25, 1.28, 1.30, 1.32, 1.35, 1.38, 1.40]:
        sig, wt = build_v9(reduce_thresh=rt, stop_threshold=0.09)
        info = test_strategy(f"U50_rt{rt}", sig, wt, f"rt={rt}")
        print_info(info); results.append(info)

    # ---- 第三组: st+rt联合扫描 ----
    print("\n--- 第三组: st+rt联合扫描 ---")
    for rt, st in [(1.30, 0.090), (1.32, 0.090), (1.28, 0.090),
                    (1.30, 0.092), (1.32, 0.092), (1.28, 0.092),
                    (1.35, 0.090), (1.30, 0.088)]:
        sig, wt = build_v9(reduce_thresh=rt, stop_threshold=st)
        info = test_strategy(f"U51_rt{rt}_st{st}", sig, wt, f"rt={rt} st={st}")
        print_info(info); results.append(info)

    # ---- 第四组: E5冷却期+stop_weight组合 (st=0.09) ----
    print("\n--- 第四组: E5冷却+权重组合 (st=0.09) ---")
    for cd, sw in [(5, 0.3), (5, 0.2), (5, 0.4), (3, 0.3),
                    (7, 0.3), (5, 0.25), (4, 0.3), (6, 0.3)]:
        sig, wt = build_v9(reduce_thresh=1.3, stop_threshold=0.09,
                           stop_weight=sw, e5_cooldown=cd)
        info = test_strategy(f"U52_cd{cd}_sw{sw}", sig, wt, f"cd={cd} sw={sw}")
        print_info(info); results.append(info)

    # ---- 第五组: 多周期T指标 ----
    print("\n--- 第五组: 多周期T指标 ---")
    for t_s, t_l, w_s, w_l in [(5, 20, 0.5, 0.5), (5, 20, 0.3, 0.7),
                                (10, 20, 0.5, 0.5), (5, 30, 0.5, 0.5),
                                (5, 20, 0.7, 0.3), (3, 20, 0.5, 0.5)]:
        sig, wt = build_v10_multi_period_t(t_short=t_s, t_long=t_l,
                                            weights=(w_s, w_l),
                                            reduce_thresh=1.3, stop_threshold=0.09)
        info = test_strategy(f"U53_T{t_s}_{t_l}_w{w_s}", sig, wt,
                              f"T短{t_s}长{t_l}权重{w_s}")
        print_info(info); results.append(info)

    # ---- 第六组: 动态E5止损 ----
    print("\n--- 第六组: 动态E5止损 ---")
    for vol_w, vol_t, tight, loose in [(20, 0.03, 0.07, 0.10),
                                        (20, 0.025, 0.07, 0.10),
                                        (20, 0.035, 0.08, 0.11),
                                        (10, 0.03, 0.07, 0.10),
                                        (20, 0.03, 0.06, 0.09),
                                        (20, 0.03, 0.08, 0.11)]:
        sig, wt = build_v10_dynamic_e5(vol_window=vol_w, vol_thresh=vol_t,
                                        tight_stop=tight, loose_stop=loose,
                                        reduce_thresh=1.3)
        info = test_strategy(f"U54_vw{vol_w}_vt{vol_t}", sig, wt,
                              f"窗口{vol_w}阈值{vol_t}紧{tight}松{loose}")
        print_info(info); results.append(info)

    # ---- 第七组: 滞回带 ----
    print("\n--- 第七组: 滞回带 (st=0.09) ---")
    for et, xt in [(1.3, 0.8), (1.3, 1.0), (1.5, 0.8), (1.5, 1.0),
                    (1.3, 0.5), (1.4, 0.8), (1.3, 1.2), (1.2, 0.8)]:
        sig, wt = build_v10_hysteresis(entry_thresh=et, exit_thresh=xt,
                                        stop_threshold=0.09)
        info = test_strategy(f"U55_et{et}_xt{xt}", sig, wt, f"进{et}出{xt}")
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
    print(f"  {'U45_st0.09(v9最优)':<28} {'44.87%':>7} {'-23.25%':>7} {'1.497':>7} "
          f"{'1.930':>7} {'364':>5} {'42.20%':>8}")
    for r in ok[:20]:
        print(f"  {r['name']:<28} {r['ann']*100:>6.2f}% {r['dd']*100:>6.2f}% "
              f"{r['sharpe']:>7.3f} {r['calmar']:>7.3f} {r['n_trades']:>5} "
              f"{r['ann_sl']*100:>7.2f}%")

    print(f"\n  Top 10 Calmar:")
    results.sort(key=lambda x: -x['calmar'])
    for r in results[:10]:
        print(f"  {r['name']:<28} 年化{r['ann']*100:.2f}% 回撤{r['dd']*100:.2f}% "
              f"Calmar{r['calmar']:.3f} 交易{r['n_trades']}次 滑点{r['ann_sl']*100:.2f}%")
