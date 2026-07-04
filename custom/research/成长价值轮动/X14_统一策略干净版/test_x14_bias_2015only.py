"""BIAS仅保留2015-05-26空仓，其他全部忽略"""
import sys
sys.path.insert(0, r'c:\XuanJLH\Qbot\custom\backtests')
sys.path.insert(0, r'c:\XuanJLH\Qbot\custom\research\成长价值轮动\X12_偏离度空仓版')
sys.path.insert(0, r'c:\XuanJLH\Qbot\custom\research\成长价值轮动\X14_统一策略干净版')

import numpy as np
import pandas as pd
from optimize_runner import (
    G_CLOSE, V_CLOSE, RATIO, RATIO_MA20, RATIO_DEV, MA20_SLOPE,
    V_MOM20, G_DD20, V_DD20, BASE_DIR,
    run_backtest, calc_metrics, count_switches,
)
from run_x33_reduce_trades import RATIO_DEV_Z

T = RATIO_DEV_Z
SLOPE = MA20_SLOPE


def build_bias_2015only():
    """BIAS仅清仓2015-05-26 (硬编码日期)"""
    slope_thresh=0.002; sw=0.17; st=0.09; cd=8
    ms=10; ml=20; rt=1.3; dc=5; dcd=6
    
    V_MOM_S = V_CLOSE.pct_change(ms)
    V_MOM_L = V_CLOSE.pct_change(ml)
    
    raw_dir = (T > 0).map({True: 'BULL', False: 'BEAR'})
    mask = np.ones(len(raw_dir), dtype=bool)
    for k in range(1, dc):
        mask = mask & (raw_dir.values == raw_dir.shift(k).values)
    confirmed_dir = raw_dir.where(mask, np.nan).ffill().fillna('BULL')
    if dcd > 0:
        new_dir = confirmed_dir.copy()
        last_switch = -dcd - 1
        prev = confirmed_dir.iloc[0]
        for i in range(len(confirmed_dir)):
            if pd.isna(confirmed_dir.iloc[i]):
                new_dir.iloc[i] = prev
                continue
            if confirmed_dir.iloc[i] != prev:
                if i - last_switch >= dcd:
                    last_switch = i
                    prev = confirmed_dir.iloc[i]
                new_dir.iloc[i] = prev
            else:
                new_dir.iloc[i] = prev
        confirmed_dir = new_dir
    dir_raw = confirmed_dir
    weak_slope = SLOPE.abs() < slope_thresh
    weak_t = T.abs() < rt
    is_weak = weak_t & weak_slope
    wt = pd.Series(1.0, index=T.index)
    wt[is_weak] = 0.0
    
    wrong_value = (dir_raw == 'BEAR') & (V_MOM_S <= 0) & (V_MOM_L <= 0)
    dir_raw[wrong_value] = 'BULL'
    dir_s = dir_raw.map({'BULL': 'growth', 'BEAR': 'value'})
    
    # BIAS: 仅2015-05-26清仓
    bias_trigger_2015 = (wt.index == pd.Timestamp('2015-05-26'))
    wt[bias_trigger_2015] = 0.0
    
    # E5
    gs = (dir_s == 'growth') & (G_DD20 < -st)
    vs = (dir_s == 'value') & (V_DD20 < -st)
    e5_trigger = gs | vs
    in_cooldown = False; cooldown_count = 0
    for i in range(len(wt)):
        if pd.isna(wt.iloc[i]) or pd.isna(dir_s.iloc[i]): continue
        if e5_trigger.iloc[i] and not in_cooldown:
            in_cooldown = True; cooldown_count = 0
            wt.iloc[i] = wt.iloc[i] * sw
        elif in_cooldown:
            cooldown_count += 1
            if cooldown_count >= cd:
                if e5_trigger.iloc[i]:
                    cooldown_count = 0
                    wt.iloc[i] = wt.iloc[i] * sw
                else:
                    in_cooldown = False
                    wt.iloc[i] = 0.0 if is_weak.iloc[i] else 1.0
            else:
                if wt.iloc[i] > 0: wt.iloc[i] = sw
    signal = dir_raw.map({'BULL': 'growth', 'BEAR': 'value'})
    return signal, wt


def print_r(name, builder, desc=""):
    sig, wt = builder()
    result = run_backtest(sig, wt)
    m = calc_metrics(result)
    sw = count_switches(sig, wt)
    result_sl = run_backtest(sig, wt, impact_slippage=0.0005)
    m_sl = calc_metrics(result_sl)
    m['calmar_sl'] = m_sl['calmar']
    m['ann_sl'] = m_sl['ann']
    print(f"{name:35s} {m['ann']*100:>7.2f}% {m['dd']*100:>7.2f}% "
          f"{m['sharpe']:>7.3f} {m['calmar']:>7.3f} {m['n_trades']:>5d} "
          f"{m['calmar_sl']:>8.3f}  {desc}")
    return m


from backtest_x14_engine import build_core as build_x14

print("=" * 95)
print("  BIAS仅保留2015-05-26空仓 vs 完整版 vs 忽略版 对比")
print("=" * 95)
print(f"{'版本':35s} {'年化':>7s} {'回撤':>7s} {'Sharpe':>7s} {'Calmar':>7s} {'交易':>5s} {'滑点Calmar':>8s}  说明")
print("  " + "-" * 95)

m_clear = print_r("X14完整版(clear)", lambda: build_x14(bias_mode='clear'), "BIAS全开")
m_ignore = print_r("X14忽略版(ignore)", lambda: build_x14(bias_mode='ignore'), "无BIAS")
m_2015 = print_r("BIAS仅2015清仓", build_bias_2015only, "只清2015-05-26")

print(f"\n--- 差值分析 ---")
print(f"  完整版 vs 忽略版:   Calmar {m_clear['calmar']:.3f} vs {m_ignore['calmar']:.3f}  (+{m_clear['calmar']-m_ignore['calmar']:.3f})")
print(f"  仅2015 vs 忽略版:   Calmar {m_2015['calmar']:.3f} vs {m_ignore['calmar']:.3f}  (+{m_2015['calmar']-m_ignore['calmar']:.3f})")
print(f"  完整版 vs 仅2015:   Calmar {m_clear['calmar']:.3f} vs {m_2015['calmar']:.3f}  (+{m_clear['calmar']-m_2015['calmar']:.3f})")
print()
print(f"  完整版 vs 忽略版:   年化 {m_clear['ann']*100:.2f}% vs {m_ignore['ann']*100:.2f}%  (+{(m_clear['ann']-m_ignore['ann'])*100:.2f}pp)")
print(f"  仅2015 vs 忽略版:   年化 {m_2015['ann']*100:.2f}% vs {m_ignore['ann']*100:.2f}%  (+{(m_2015['ann']-m_ignore['ann'])*100:.2f}pp)")
