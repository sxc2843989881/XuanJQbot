"""测试去掉第4层(B2)和第5层(BIAS)的效果"""
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

def print_r(name, sig, wt, desc=""):
    result = run_backtest(sig, wt)
    m = calc_metrics(result)
    sw = count_switches(sig, wt)
    result_sl = run_backtest(sig, wt, impact_slippage=0.0005)
    m_sl = calc_metrics(result_sl)
    print(f"{name:30s} {m['ann']*100:>7.2f}% {m['dd']*100:>7.2f}% "
          f"{m['sharpe']:>7.3f} {m['calmar']:>7.3f} {m['n_trades']:>5d} "
          f"{m_sl['calmar']:>8.3f}  {desc}")
    return m

# ====== 基准: X14 完整版 ======
def build_x14():
    from backtest_x14_engine import build_core
    return build_core()

# ====== 去B2: 去掉第4层价值动量过滤 ======
def build_no_b2(slope_thresh=0.002, sw=0.17, st=0.09, cd=8,
                ms=10, ml=20, rt=1.3, dc=5, dcd=6,
                bias_ma=20, bias_high=0.19, bias_reduce=0.05):
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
    
    # ★ 去掉B2: 不执行价值动量过滤
    
    dir_s = dir_raw.map({'BULL': 'growth', 'BEAR': 'value'})
    
    # BIAS
    extreme_g = (dir_s == 'growth') & (G_BIAS > bias_high)
    extreme_v = (dir_s == 'value') & (V_BIAS > bias_high)
    extreme = extreme_g | extreme_v
    wt[extreme] = wt[extreme] * bias_reduce
    
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

# ====== 去BIAS: 去掉第5层超买降仓 ======
def build_no_bias(slope_thresh=0.002, sw=0.17, st=0.09, cd=8,
                  ms=10, ml=20, rt=1.3, dc=5, dcd=6,
                  bias_ma=20, bias_high=0.19, bias_reduce=0.05):
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
    
    # B2
    wrong_value = (dir_raw == 'BEAR') & (V_MOM_S <= 0) & (V_MOM_L <= 0)
    dir_raw[wrong_value] = 'BULL'
    dir_s = dir_raw.map({'BULL': 'growth', 'BEAR': 'value'})
    
    # ★ 去掉BIAS: 不执行超买降仓
    
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

# ====== 去B2+去BIAS: 同时去掉这两层 ======
def build_no_b2_bias(slope_thresh=0.002, sw=0.17, st=0.09, cd=8,
                     ms=10, ml=20, rt=1.3, dc=5, dcd=6,
                     bias_ma=20, bias_high=0.19, bias_reduce=0.05):
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
    
    # ★ 去掉B2: 不执行价值动量过滤
    # ★ 去掉BIAS: 不执行超买降仓
    
    dir_s = dir_raw.map({'BULL': 'growth', 'BEAR': 'value'})
    
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


# ====== 跑 ======
print(f"{'版本':30s} {'年化':>7s} {'回撤':>7s} {'Sharpe':>7s} {'Calmar':>7s} {'交易':>5s} {'滑点Calmar':>8s}  说明")
print("  " + "-" * 90)

# 完整版
sig, wt = build_x14()
m_full = print_r("X14 完整版", sig, wt, "6层全开")

# 去B2
sig, wt = build_no_b2()
m_no_b2 = print_r("去掉B2(价值动量过滤)", sig, wt, "删第4层")

# 去BIAS
sig, wt = build_no_bias()
m_no_bias = print_r("去掉BIAS(超买降仓)", sig, wt, "删第5层")

# 去B2+BIAS
sig, wt = build_no_b2_bias()
m_no_both = print_r("去掉B2+BIAS", sig, wt, "删第4+5层")

# 只留前三层
def build_123_only(slope_thresh=0.002, sw=0.17, st=0.09, cd=8,
                   ms=10, ml=20, rt=1.3, dc=5, dcd=6):
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
    # 只保留E5（第3层是空仓、第6层是止损，两者是独立的）
    dir_s = dir_raw.map({'BULL': 'growth', 'BEAR': 'value'})
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

sig, wt = build_123_only()
m_123 = print_r("仅1+2+3+6层(E5)", sig, wt, "只剩方向确认+冷却+空仓+止损")

# 纯裸版: 只剩方向确认
def build_bare():
    raw_dir = (T > 0).map({True: 'BULL', False: 'BEAR'})
    wt = pd.Series(1.0, index=T.index)
    signal = raw_dir.map({'BULL': 'growth', 'BEAR': 'value'})
    return signal, wt

sig, wt = build_bare()
m_bare = print_r("纯裸版(方向确认仅dc=1)", sig, wt, "永远满仓，无任何保护")

print("\n" + "=" * 90)
print("  层贡献拆解")
print("=" * 90)
print(f"\n  完整版 Calmar:       {m_full['calmar']:.3f}")
print(f"  去B2:                {m_no_b2['calmar']:.3f}  (B2贡献: +{m_full['calmar']-m_no_b2['calmar']:.3f})")
print(f"  去BIAS:              {m_no_bias['calmar']:.3f}  (BIAS贡献: +{m_full['calmar']-m_no_bias['calmar']:.3f})")
print(f"  去B2+BIAS:           {m_no_both['calmar']:.3f}  (B2+BIAS合计贡献: +{m_full['calmar']-m_no_both['calmar']:.3f})")
print(f"  仅1+2+3+6层(E5):     {m_123['calmar']:.3f}")
print(f"  纯裸版:              {m_bare['calmar']:.3f}")
print()

# 年化贡献
print(f"  年化贡献拆解:")
print(f"    完整版:            {m_full['ann']*100:.2f}%")
print(f"    去B2:              {m_no_b2['ann']*100:.2f}%  (B2贡献年化: +{m_full['ann']-m_no_b2['ann']*100:.2f}pp)")
print(f"    去BIAS:            {m_no_bias['ann']*100:.2f}%  (BIAS贡献年化: +{m_full['ann']-m_no_bias['ann']*100:.2f}pp)")
print(f"    去B2+BIAS:         {m_no_both['ann']*100:.2f}%")
print(f"    仅1+2+3+6层(E5):   {m_123['ann']*100:.2f}%")
