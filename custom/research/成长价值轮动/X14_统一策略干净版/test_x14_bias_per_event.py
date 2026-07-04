"""BIAS每次触发事件的独立贡献分析"""
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


def build_with_bias_dates(clear_dates=None):
    """
    BIAS仅清仓指定的日期列表
    clear_dates: list of date strings like ['2015-05-26', '2020-07-09']
    """
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
    
    # BIAS: 仅指定的日期清仓
    if clear_dates:
        for d in clear_dates:
            ts = pd.Timestamp(d)
            if ts in wt.index:
                wt.loc[ts] = 0.0
    
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


def run_and_return(name, builder, desc=""):
    sig, wt = builder()
    result = run_backtest(sig, wt)
    m = calc_metrics(result)
    sw = count_switches(sig, wt)
    result_sl = run_backtest(sig, wt, impact_slippage=0.0005)
    m_sl = calc_metrics(result_sl)
    m['calmar_sl'] = m_sl['calmar']
    m['ann_sl'] = m_sl['ann']
    return m


# 6个事件
events = {
    'E1: 2015-05-26': ['2015-05-26'],
    'E2: 2020-07-09': ['2020-07-09'],
    'E3: 2020-07-13~14': ['2020-07-13', '2020-07-14'],
    'E4: 2024-09-30': ['2024-09-30'],
    'E5: 2024-10-08': ['2024-10-08'],
    'E6: 2024-10-14': ['2024-10-14'],
}

# 基准: 忽略所有BIAS
m_baseline = run_and_return("忽略版", lambda: build_with_bias_dates(None))

print("=" * 95)
print("  每次BIAS触发事件的独立贡献分析")
print("=" * 95)
print(f"基准(无BIAS): 年化 {m_baseline['ann']*100:.2f}%  Calmar {m_baseline['calmar']:.3f}  交易 {m_baseline['n_trades']}\n")
print(f"{'事件':25s} {'年化':>7s} {'回撤':>7s} {'Calmar':>7s} {'交易':>5s} {'Δ年化':>8s} {'ΔCalmar':>9s} {'贡献方向':>8s}")
print("  " + "-" * 75)

results = []
for name, dates in events.items():
    m = run_and_return(f"仅{name}", lambda d=dates: build_with_bias_dates(d))
    delta_ann = (m['ann'] - m_baseline['ann']) * 100
    delta_cal = m['calmar'] - m_baseline['calmar']
    direction = "正贡献" if delta_cal > 0 else "负贡献"
    results.append((name, m, delta_ann, delta_cal, direction))
    print(f"  {name:25s} {m['ann']*100:>7.2f}% {m['dd']*100:>7.2f}% "
          f"{m['calmar']:>7.3f} {m['n_trades']:>5d} "
          f"{delta_ann:>+7.2f}pp {delta_cal:>+8.3f}  {direction:>8s}")

# 完整版
m_all = run_and_return("完整版(6次全开)", lambda: build_with_bias_dates([
    '2015-05-26', '2020-07-09', '2020-07-13', '2020-07-14',
    '2024-09-30', '2024-10-08', '2024-10-14'
]))
delta_ann_all = (m_all['ann'] - m_baseline['ann']) * 100
delta_cal_all = m_all['calmar'] - m_baseline['calmar']
print(f"  {'完整版(6次全开)':25s} {m_all['ann']*100:>7.2f}% {m_all['dd']*100:>7.2f}% "
      f"{m_all['calmar']:>7.3f} {m_all['n_trades']:>5d} "
      f"{delta_ann_all:>+7.2f}pp {delta_cal_all:>+8.3f}  {'':>8s}")

print(f"\n--- 汇总 ---")
print(f"  事件总贡献: 年化 +{delta_ann_all:.2f}pp  Calmar +{delta_cal_all:.3f}")
pos = [r for r in results if r[4] == '正贡献']
neg = [r for r in results if r[4] == '负贡献']
print(f"  正贡献事件: {len(pos)}次")
for n, m, da, dc, d in pos:
    print(f"    {n}: ΔCalmar {dc:+.3f}  Δ年化 {da:+.2f}pp")
print(f"  负贡献事件: {len(neg)}次")
for n, m, da, dc, d in neg:
    print(f"    {n}: ΔCalmar {dc:+.3f}  Δ年化 {da:+.2f}pp")
