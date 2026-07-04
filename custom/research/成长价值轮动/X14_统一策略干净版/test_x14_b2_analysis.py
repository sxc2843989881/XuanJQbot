"""测试B2逻辑改进: 当两者都跌时清仓而非反手"""
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

def print_r(name, builder, desc=""):
    sig, wt = builder()
    result = run_backtest(sig, wt)
    m = calc_metrics(result)
    sw = count_switches(sig, wt)
    result_sl = run_backtest(sig, wt, impact_slippage=0.0005)
    m_sl = calc_metrics(result_sl)
    print(f"{name:38s} {m['ann']*100:>7.2f}% {m['dd']*100:>7.2f}% "
          f"{m['sharpe']:>7.3f} {m['calmar']:>7.3f} {m['n_trades']:>5d} "
          f"{m_sl['calmar']:>8.3f}  {desc}")
    return m

# 基础构建函数（可配置B2行为）
def make_core(b2_mode='flip'):
    """
    b2_mode:
      'flip'   -> 原版B2: V在跌→反手买growth
      'cash'   -> 改进B2: V在跌→清仓(wt=0)
      'none'   -> 去掉B2
    """
    bias_ma=20; bias_high=0.19; bias_reduce=0.05
    slope_thresh=0.002; sw=0.17; st=0.09; cd=8
    ms=10; ml=20; rt=1.3; dc=5; dcd=6
    
    G_MA = G_CLOSE.rolling(bias_ma).mean()
    V_MA = V_CLOSE.rolling(bias_ma).mean()
    G_BIAS = (G_CLOSE / G_MA - 1)
    V_BIAS = (V_CLOSE / V_MA - 1)
    V_MOM_S = V_CLOSE.pct_change(ms)
    V_MOM_L = V_CLOSE.pct_change(ml)
    G_MOM_S = G_CLOSE.pct_change(ms)
    G_MOM_L = G_CLOSE.pct_change(ml)
    
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
    
    # B2 处理
    wrong_value = (dir_raw == 'BEAR') & (V_MOM_S <= 0) & (V_MOM_L <= 0)
    
    if b2_mode == 'flip':
        # 原版: 反手买growth
        dir_raw[wrong_value] = 'BULL'
    elif b2_mode == 'cash':
        # 改进: V在跌→清仓
        # 先不管方向怎么变，先把wt设0
        wt[wrong_value] = 0.0
    
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


# 额外测试: T<0时如果两者都跌→清仓
def make_dual_cash(b2_mode='flip'):
    """类似make_core但监控G是否也在跌"""
    bias_ma=20; bias_high=0.19; bias_reduce=0.05
    slope_thresh=0.002; sw=0.17; st=0.09; cd=8
    ms=10; ml=20; rt=1.3; dc=5; dcd=6
    
    G_MA = G_CLOSE.rolling(bias_ma).mean()
    V_MA = V_CLOSE.rolling(bias_ma).mean()
    G_BIAS = (G_CLOSE / G_MA - 1)
    V_BIAS = (V_CLOSE / V_MA - 1)
    V_MOM_S = V_CLOSE.pct_change(ms)
    V_MOM_L = V_CLOSE.pct_change(ml)
    G_MOM_S = G_CLOSE.pct_change(ms)
    G_MOM_L = G_CLOSE.pct_change(ml)
    
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
    
    # 当方向为BEAR时:
    #   V在跌 → 原版flip到growth
    #   G也在跌 → 清仓（新逻辑）
    v_falling = (dir_raw == 'BEAR') & (V_MOM_S <= 0) & (V_MOM_L <= 0)
    g_falling = (dir_raw == 'BEAR') & (G_MOM_S <= 0) & (G_MOM_L <= 0)
    
    if b2_mode == 'flip':
        # 只处理V跌的情况
        dir_raw[v_falling] = 'BULL'
    elif b2_mode == 'cash_v':
        # V跌→清仓
        wt[v_falling] = 0.0
    elif b2_mode == 'cash_both':
        # V跌+G跌→清仓，只有V跌→flip
        both = v_falling & g_falling
        v_only = v_falling & ~g_falling
        wt[both] = 0.0
        dir_raw[v_only] = 'BULL'
    elif b2_mode == 'cash_v_g':
        # V跌或G跌都清仓
        all_falling = v_falling | g_falling
        wt[all_falling] = 0.0
    
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


# ====== 跑 ======
print(f"{'版本':38s} {'年化':>7s} {'回撤':>7s} {'Sharpe':>7s} {'Calmar':>7s} {'交易':>5s} {'滑点Calmar':>8s}  说明")
print("  " + "-" * 100)

# 第一组: 简单B2模式对比
print("\n--- 第一组: B2基础模式对比 ---")
results = []
for mode, desc in [('flip', '原版: V跌→反手growth'), ('cash', 'V跌→清仓'), ('none', '去掉B2')]:
    m = print_r(f"B2_{mode}", lambda m=mode: make_core(m), desc)
    results.append((f"B2_{mode}", m))

# 第二组: 双下跌监控
print("\n--- 第二组: 双下跌监控 ---")
for mode, desc in [('flip', '原版B2'), ('cash_v', 'V跌→清仓'),
                    ('cash_both', 'V+G都跌→清仓, 仅V跌→flip'),
                    ('cash_v_g', 'V跌或G跌→清仓')]:
    m = print_r(f"双跌_{mode}", lambda m=mode: make_dual_cash(m), desc)
    results.append((f"双跌_{mode}", m))

# 第三组: B2+cash混合 → 当两者都跌时清仓
print("\n--- 第三组: 综合最佳模式 ---")
# 用make_core的cash模式，再加上V和G都跌时清仓
def make_hybrid():
    """混合模式: 原版B2 + 两者都跌时额外清仓"""
    bias_ma=20; bias_high=0.19; bias_reduce=0.05
    slope_thresh=0.002; sw=0.17; st=0.09; cd=8
    ms=10; ml=20; rt=1.3; dc=5; dcd=6
    
    G_MA = G_CLOSE.rolling(bias_ma).mean()
    V_MA = V_CLOSE.rolling(bias_ma).mean()
    G_BIAS = (G_CLOSE / G_MA - 1)
    V_BIAS = (V_CLOSE / V_MA - 1)
    V_MOM_S = V_CLOSE.pct_change(ms)
    V_MOM_L = V_CLOSE.pct_change(ml)
    G_MOM_S = G_CLOSE.pct_change(ms)
    G_MOM_L = G_CLOSE.pct_change(ml)
    
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
    
    # 原版B2 flip
    wrong_value = (dir_raw == 'BEAR') & (V_MOM_S <= 0) & (V_MOM_L <= 0)
    dir_raw[wrong_value] = 'BULL'
    
    dir_s = dir_raw.map({'BULL': 'growth', 'BEAR': 'value'})
    
    # 额外: 如果当前方向也在跌→清仓
    g_falling = (dir_s == 'growth') & (G_MOM_S <= 0) & (G_MOM_L <= 0)
    v_falling = (dir_s == 'value') & (V_MOM_S <= 0) & (V_MOM_L <= 0)
    both_falling = g_falling | v_falling
    wt[both_falling] = 0.0
    
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

m = print_r("混合_原版flip+双跌清仓", make_hybrid, "原版B2 + 持仓方向也在跌→清仓")
results.append(("混合_flip+双跌清仓", m))

# 再对比: B2 flip + 双跌清仓 但不再额外加方向动量过滤
def make_hybrid2():
    """只加G也在跌时清仓, 不改原版B2"""
    bias_ma=20; bias_high=0.19; bias_reduce=0.05
    slope_thresh=0.002; sw=0.17; st=0.09; cd=8
    ms=10; ml=20; rt=1.3; dc=5; dcd=6
    
    G_MA = G_CLOSE.rolling(bias_ma).mean()
    V_MA = V_CLOSE.rolling(bias_ma).mean()
    G_BIAS = (G_CLOSE / G_MA - 1)
    V_BIAS = (V_CLOSE / V_MA - 1)
    V_MOM_S = V_CLOSE.pct_change(ms)
    V_MOM_L = V_CLOSE.pct_change(ml)
    G_MOM_S = G_CLOSE.pct_change(ms)
    G_MOM_L = G_CLOSE.pct_change(ml)
    
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
    
    # 原版B2
    wrong_value = (dir_raw == 'BEAR') & (V_MOM_S <= 0) & (V_MOM_L <= 0)
    dir_raw[wrong_value] = 'BULL'
    
    dir_s = dir_raw.map({'BULL': 'growth', 'BEAR': 'value'})
    
    # 改进: BEAR时如果G也在跌→清仓
    v_falling = (dir_raw == 'BEAR') & (V_MOM_S <= 0) & (V_MOM_L <= 0)
    # B2已经把这部分flip了, 但我们额外检查: 如果flip之前的V也在跌, 且flip之后的growth也在跌
    # 实际上更容易: 不管方向, 持仓方向自己在下跌就清仓
    growth_bear = (dir_s == 'growth') & (G_MOM_S <= 0) & (G_MOM_L <= 0)
    value_bear = (dir_s == 'value') & (V_MOM_S <= 0) & (V_MOM_L <= 0)
    any_falling = growth_bear | value_bear
    wt[any_falling] = 0.0
    
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

m = print_r("混合2_flip+持仓跌清仓", make_hybrid2, "持仓方向动量<0→清仓(不拘泥于BEAR)")
results.append(("混合2_flip+持仓跌清仓", m))

# 汇总
print("\n" + "=" * 100)
print("  汇总 (按Calmar排序)")
print("=" * 100)
results.sort(key=lambda x: -x[1]['calmar'])
for name, m in results:
    cal_sl = m.get('calmar_sl', 0)
    print(f"  {name:38s} {m['ann']*100:>7.2f}% {m['dd']*100:>7.2f}% "
          f"{m['sharpe']:>7.3f} {m['calmar']:>7.3f} {m['n_trades']:>5d} "
          f"{cal_sl:>8.3f}")
