"""BIAS层改为可配置: 清仓模式 vs 忽略模式"""
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


def build_x14_bias(bias_mode='clear'):
    """
    bias_mode:
      'clear'  -> BIAS>19%时清仓(原版Br=0.0逻辑)
      'ignore' -> 忽略BIAS, 永不触发
    """
    slope_thresh=0.002; sw=0.17; st=0.09; cd=8
    ms=10; ml=20; rt=1.3; dc=5; dcd=6
    bias_ma=20; bias_high=0.19
    
    G_MA = G_CLOSE.rolling(bias_ma).mean()
    V_MA = V_CLOSE.rolling(bias_ma).mean()
    G_BIAS = (G_CLOSE / G_MA - 1)
    V_BIAS = (V_CLOSE / V_MA - 1)
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
    
    # BIAS: 清仓模式 vs 忽略模式
    if bias_mode == 'clear':
        extreme_g = (dir_s == 'growth') & (G_BIAS > bias_high)
        extreme_v = (dir_s == 'value') & (V_BIAS > bias_high)
        extreme = extreme_g | extreme_v
        wt[extreme] = 0.0  # 直接清仓
    # ignore模式: 什么都不做
    
    # E5止损
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


def print_r(name, sig, wt, desc=""):
    result = run_backtest(sig, wt)
    m = calc_metrics(result)
    sw = count_switches(sig, wt)
    result_sl = run_backtest(sig, wt, impact_slippage=0.0005)
    m_sl = calc_metrics(result_sl)
    print(f"{name:35s} {m['ann']*100:>7.2f}% {m['dd']*100:>7.2f}% "
          f"{m['sharpe']:>7.3f} {m['calmar']:>7.3f} {m['n_trades']:>5d} "
          f"{m_sl['calmar']:>8.3f}  {desc}")
    # 合并返回
    m['calmar_sl'] = m_sl['calmar']
    m['ann_sl'] = m_sl['ann']
    return m


print("=" * 90)
print("  BIAS层: 清仓模式 vs 忽略模式 对比")
print("=" * 90)
print(f"{'版本':35s} {'年化':>7s} {'回撤':>7s} {'Sharpe':>7s} {'Calmar':>7s} {'交易':>5s} {'滑点Calmar':>8s}  说明")
print("  " + "-" * 90)

# 清仓模式
sig, wt = build_x14_bias(bias_mode='clear')
m_clear = print_r("BIAS清仓模式(clear)", sig, wt, "BIAS>19%→直接清仓")

# 忽略模式  
sig, wt = build_x14_bias(bias_mode='ignore')
m_ignore = print_r("BIAS忽略模式(ignore)", sig, wt, "BIAS触发了也不管")

# 再跑一个原版干净的对比
from backtest_x14_engine import build_core
sig, wt = build_core()
m_orig = print_r("X14原版(Br=0.05)", sig, wt, "原版bias_reduce=0.05降仓")

print("\n" + "=" * 90)
print("  对比总结")
print("=" * 90)
print(f"\n  {'模式':<20} {'年化':>8} {'回撤':>8} {'Calmar':>8} {'滑点Calmar':>10} {'交易':>6}")
print("  " + "-" * 60)
print(f"  {'清仓模式(clear)':<20} {m_clear['ann']*100:>7.2f}% {m_clear['dd']*100:>7.2f}% "
      f"{m_clear['calmar']:>8.3f} {m_clear['calmar_sl']:>10.3f} {m_clear['n_trades']:>6}")
print(f"  {'忽略模式(ignore)':<20} {m_ignore['ann']*100:>7.2f}% {m_ignore['dd']*100:>7.2f}% "
      f"{m_ignore['calmar']:>8.3f} {m_ignore['calmar_sl']:>10.3f} {m_ignore['n_trades']:>6}")
print(f"  {'X14原版(Br=0.05)':<20} {m_orig['ann']*100:>7.2f}% {m_orig['dd']*100:>7.2f}% "
      f"{m_orig['calmar']:>8.3f} {m_orig['calmar_sl']:>10.3f} {m_orig['n_trades']:>6}")

print(f"\n  用户选择权:")
print(f"    bias_mode='clear' → BIAS触发时自动清仓 (Calmar {m_clear['calmar']:.3f})")
print(f"    bias_mode='ignore' → 忽略BIAS信号 (Calmar {m_ignore['calmar']:.3f})")
print(f"    差值: Calmar {(m_clear['calmar']-m_ignore['calmar']):+.3f}, "
      f"年化 {(m_clear['ann']-m_ignore['ann'])*100:+.2f}pp")
print(f"\n  建议: 默认使用清仓模式，极端行情时用户可改为忽略模式。")
