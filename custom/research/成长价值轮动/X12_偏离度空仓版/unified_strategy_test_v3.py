"""unified_strategy_test_v3.py — 统一策略第三轮: 方向确认+更激进参数
================================================================
第二轮最优: U10_e0.5_x0.0_nw0.3 年化37.91% 回撤-31.39% 交易466次
问题: 方向切换204次太多, entry=0.5太激进导致频繁变向
第三轮: 加方向确认N天 + 调参 + 更长期std
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

# 不同窗口的T指标
T20 = RATIO_DEV_Z  # 20日std
T40 = RATIO_DEV / RATIO_DEV.rolling(40).std()  # 40日std
T60 = RATIO_DEV / RATIO_DEV.rolling(60).std()  # 60日std


def build_unified_v3(T_series, entry_thresh=0.5, exit_thresh=0.0,
                     neutral_weight=0.3, dir_confirm=3,
                     use_b2=True, use_e5=True,
                     stop_threshold=0.10, stop_weight=0.30, e5_cooldown=5):
    """统一三态状态机 v3: 加方向确认

    Args:
        T_series: T指标序列
        entry_thresh: 进入BULL/BEAR的|T|阈值
        exit_thresh: 退出阈值
        neutral_weight: NEUTRAL仓位
        dir_confirm: 方向确认天数(T符号连续N天一致才切换)
    """
    T = T_series
    # 方向: T的符号, 需连续dir_confirm天一致才确认
    raw_dir = (T > 0).map({True: 'BULL', False: 'BEAR'})
    mask = np.ones(len(raw_dir), dtype=bool)
    for k in range(1, dir_confirm):
        mask = mask & (raw_dir.values == raw_dir.shift(k).values)
    confirmed_dir = raw_dir.where(mask, np.nan).ffill().fillna('BULL')

    # 状态机: 基于confirmed_dir + |T|判断强度
    states = []
    state = 'NEUTRAL'
    for i in range(len(T)):
        t = T.iloc[i]
        d = confirmed_dir.iloc[i]
        if pd.isna(t):
            states.append(state)
            continue

        if state == 'BULL':
            if t < exit_thresh:
                state = 'BEAR' if (t < -entry_thresh and d == 'BEAR') else 'NEUTRAL'
        elif state == 'BEAR':
            if t > -exit_thresh:
                state = 'BULL' if (t > entry_thresh and d == 'BULL') else 'NEUTRAL'
        else:  # NEUTRAL
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
    print("  统一策略第三轮: 方向确认 + 不同std窗口")
    print("=" * 80)

    results = []

    # ---- 第一组: 方向确认天数扫描 (entry=0.5, nw=0.3) ----
    print("\n--- 第一组: 方向确认天数扫描 (T20, entry=0.5, exit=0, nw=0.3) ---")
    for dc in [2, 3, 4, 5, 7]:
        sig, wt = build_unified_v3(T20, entry_thresh=0.5, exit_thresh=0.0,
                                    neutral_weight=0.3, dir_confirm=dc)
        info = test_strategy(f"U11_dc{dc}", sig, wt, f"方向确认{dc}天")
        print_info(info); results.append(info)

    # ---- 第二组: entry扫描 + 方向确认5天 ----
    print("\n--- 第二组: entry扫描 (T20, dir_confirm=5, nw=0.3) ---")
    for et in [0.3, 0.5, 0.8, 1.0, 1.5]:
        sig, wt = build_unified_v3(T20, entry_thresh=et, exit_thresh=0.0,
                                    neutral_weight=0.3, dir_confirm=5)
        info = test_strategy(f"U12_e{et}_dc5", sig, wt, f"entry={et} dc=5")
        print_info(info); results.append(info)

    # ---- 第三组: 不同std窗口 ----
    print("\n--- 第三组: 不同std窗口 (entry=0.5, dc=5, nw=0.3) ---")
    for T_s, label in [(T20, "T20"), (T40, "T40"), (T60, "T60")]:
        sig, wt = build_unified_v3(T_s, entry_thresh=0.5, exit_thresh=0.0,
                                    neutral_weight=0.3, dir_confirm=5)
        info = test_strategy(f"U13_{label}", sig, wt, f"{label} std窗口")
        print_info(info); results.append(info)

    # ---- 第四组: nw扫描 + 方向确认5天 ----
    print("\n--- 第四组: nw扫描 (entry=0.5, dc=5) ---")
    for nw in [0.0, 0.2, 0.3, 0.5, 0.7]:
        sig, wt = build_unified_v3(T20, entry_thresh=0.5, exit_thresh=0.0,
                                    neutral_weight=nw, dir_confirm=5)
        info = test_strategy(f"U14_nw{nw}_dc5", sig, wt, f"nw={nw} dc=5")
        print_info(info); results.append(info)

    # ---- 第五组: 最优组合细调 ----
    print("\n--- 第五组: 最优组合细调 ---")
    for et, dc, nw in [(0.5, 4, 0.2), (0.5, 4, 0.3), (0.3, 5, 0.2),
                        (0.3, 7, 0.3), (0.8, 4, 0.3), (0.8, 5, 0.2),
                        (1.0, 5, 0.0), (0.5, 5, 0.0)]:
        sig, wt = build_unified_v3(T20, entry_thresh=et, exit_thresh=0.0,
                                    neutral_weight=nw, dir_confirm=dc)
        info = test_strategy(f"U15_e{et}_dc{dc}_nw{nw}", sig, wt,
                              f"entry={et} dc={dc} nw={nw}")
        print_info(info); results.append(info)

    # ---- 汇总 ----
    print("\n" + "=" * 80)
    print("  汇总 (按年化降序)")
    print("=" * 80)
    print(f"  {'名称':<26} {'年化':>7} {'回撤':>7} {'Sharpe':>7} {'Calmar':>7} "
          f"{'交易':>5} {'方向':>5} {'仓位':>5} {'滑点年化':>8}")
    print("  " + "-" * 94)
    print(f"  {'X61(基准)':<26} {'41.12%':>7} {'-30.21%':>7} {'1.357':>7} "
          f"{'1.361':>7} {'224':>5} {'179':>5} {'20':>5} {'39.04%':>8}")

    results.sort(key=lambda x: -x['ann'])
    for r in results[:15]:
        print(f"  {r['name']:<26} {r['ann']*100:>6.2f}% {r['dd']*100:>6.2f}% "
              f"{r['sharpe']:>7.3f} {r['calmar']:>7.3f} {r['n_trades']:>5} "
              f"{r['dir_sw']:>5} {r['cash_sw']:>5} {r['ann_sl']*100:>7.2f}%")

    ok = [r for r in results if r['ann'] > 0.40 and r['dd'] > -0.35]
    print(f"\n  满足约束(年化>40% 回撤<-35%): {len(ok)}个版本")
    for r in ok:
        print(f"    {r['name']}: 年化{r['ann']*100:.2f}% 回撤{r['dd']*100:.2f}% "
              f"交易{r['n_trades']}次 Calmar{r['calmar']:.3f}")
