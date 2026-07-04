"""unified_strategy_test_v2.py — 统一策略第二轮: NEUTRAL不空仓
================================================================
第一轮问题: NEUTRAL=空仓 → 踏空严重, 年化仅27.95%
第二轮方向: NEUTRAL时保持方向+降仓(不空仓)
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


def build_unified_v2(entry_thresh=1.5, exit_thresh=0.5,
                     neutral_weight=0.5,
                     use_b2=True, use_e5=True,
                     stop_threshold=0.10, stop_weight=0.30,
                     e5_cooldown=5):
    """统一三态状态机 v2: NEUTRAL保持方向+降仓

    Args:
        entry_thresh: 进入BULL/BEAR的|T|阈值
        exit_thresh: 退出BULL/BEAR的|T|阈值
        neutral_weight: NEUTRAL时的仓位(0=空仓, 0.5=半仓, 1=满仓)
    """
    states = []
    state = 'NEUTRAL'
    for i in range(len(T)):
        t = T.iloc[i]
        if pd.isna(t):
            states.append(state)
            continue
        if state == 'BULL':
            if t < exit_thresh:
                state = 'BEAR' if t < -entry_thresh else 'NEUTRAL'
        elif state == 'BEAR':
            if t > -exit_thresh:
                state = 'BULL' if t > entry_thresh else 'NEUTRAL'
        else:
            if t > entry_thresh:
                state = 'BULL'
            elif t < -entry_thresh:
                state = 'BEAR'
        states.append(state)

    states = pd.Series(states, index=T.index)

    # 方向: NEUTRAL时ffill上次方向
    dir_raw = states.where(states != 'NEUTRAL', np.nan).ffill().fillna('BULL')

    # 权重: BULL/BEAR=1, NEUTRAL=neutral_weight
    wt = pd.Series(1.0, index=T.index)
    wt[states == 'NEUTRAL'] = neutral_weight

    # B2
    if use_b2:
        wrong_value = (dir_raw == 'BEAR') & (V_MOM20 <= 0)
        dir_raw[wrong_value] = 'BULL'

    # E5
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


def build_continuous(entry_thresh=1.5, neutral_weight=0.5,
                     use_b2=True, use_e5=True,
                     stop_threshold=0.10, stop_weight=0.30, e5_cooldown=5):
    """连续仓位: weight = clip(|T|/entry, neutral_weight, 1)"""
    wt_raw = (T.abs() / entry_thresh).clip(0, 1)
    # 映射: |T|/entry < neutral_weight 时用neutral_weight
    wt = wt_raw.where(wt_raw > neutral_weight, neutral_weight)

    # 方向: T的符号
    dir_raw = (T > 0).map({True: 'BULL', False: 'BEAR'}).fillna('BULL')

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
                    # 不恢复到1,恢复到原始T对应的仓位
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
    print("  统一策略第二轮: NEUTRAL不空仓 + 连续仓位")
    print("=" * 80)

    results = []

    # ---- 第一组: NEUTRAL降仓扫描 ----
    print("\n--- 第一组: NEUTRAL仓位扫描 (entry=1.5 exit=0.5) ---")
    for nw in [0.0, 0.3, 0.5, 0.7, 1.0]:
        sig, wt = build_unified_v2(entry_thresh=1.5, exit_thresh=0.5,
                                    neutral_weight=nw)
        info = test_strategy(f"U7_nw{nw}", sig, wt, f"NEUTRAL权重={nw}")
        print_info(info); results.append(info)

    # ---- 第二组: NEUTRAL=0.5 + 滞回带扫描 ----
    print("\n--- 第二组: NEUTRAL=0.5 + 滞回带扫描 ---")
    for et, xt in [(1.5, 0.3), (1.5, 0.0), (1.2, 0.3), (1.0, 0.0), (2.0, 0.5)]:
        sig, wt = build_unified_v2(entry_thresh=et, exit_thresh=xt,
                                    neutral_weight=0.5)
        info = test_strategy(f"U8_e{et}_x{xt}_nw0.5", sig, wt,
                              f"entry={et} exit={xt} nw=0.5")
        print_info(info); results.append(info)

    # ---- 第三组: 连续仓位 ----
    print("\n--- 第三组: 连续仓位 weight=clip(|T|/entry, nw, 1) ---")
    for et, nw in [(1.5, 0.3), (1.5, 0.5), (1.0, 0.3), (1.0, 0.5), (2.0, 0.5)]:
        sig, wt = build_continuous(entry_thresh=et, neutral_weight=nw)
        info = test_strategy(f"U9_cont_e{et}_nw{nw}", sig, wt,
                              f"连续 entry={et} nw={nw}")
        print_info(info); results.append(info)

    # ---- 第四组: NEUTRAL=0.3 + 更激进参数 ----
    print("\n--- 第四组: NEUTRAL=0.3 + 激进参数 ---")
    for et, xt in [(1.0, 0.0), (0.8, 0.0), (1.2, 0.0), (0.5, 0.0)]:
        sig, wt = build_unified_v2(entry_thresh=et, exit_thresh=xt,
                                    neutral_weight=0.3)
        info = test_strategy(f"U10_e{et}_x{xt}_nw0.3", sig, wt,
                              f"entry={et} exit={xt} nw=0.3")
        print_info(info); results.append(info)

    # ---- 汇总 ----
    print("\n" + "=" * 80)
    print("  汇总 (按年化降序)")
    print("=" * 80)
    print(f"  {'名称':<24} {'年化':>7} {'回撤':>7} {'Sharpe':>7} {'Calmar':>7} "
          f"{'交易':>5} {'方向':>5} {'仓位':>5} {'滑点年化':>8}")
    print("  " + "-" * 92)
    print(f"  {'X61(基准)':<24} {'41.12%':>7} {'-30.21%':>7} {'1.357':>7} "
          f"{'1.361':>7} {'224':>5} {'179':>5} {'20':>5} {'39.04%':>8}")

    results.sort(key=lambda x: -x['ann'])
    for r in results:
        print(f"  {r['name']:<24} {r['ann']*100:>6.2f}% {r['dd']*100:>6.2f}% "
              f"{r['sharpe']:>7.3f} {r['calmar']:>7.3f} {r['n_trades']:>5} "
              f"{r['dir_sw']:>5} {r['cash_sw']:>5} {r['ann_sl']*100:>7.2f}%")

    ok = [r for r in results if r['ann'] > 0.40 and r['dd'] > -0.35]
    print(f"\n  满足约束(年化>40% 回撤<-35%): {len(ok)}个版本")
    for r in ok:
        print(f"    {r['name']}: 年化{r['ann']*100:.2f}% 回撤{r['dd']*100:.2f}% "
              f"交易{r['n_trades']}次 Calmar{r['calmar']:.3f}")
