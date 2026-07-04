"""第3层(T+斜率空仓)优化：减少频繁跳仓"""
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


def build_v3_optimized(
    # 原参数
    sw=0.17, st=0.09, cd=8, ms=10, ml=20,
    rt=1.3, slope_thresh=0.002, dc=5, dcd=6,
    bias_ma=20, bias_high=0.19,
    # 第3层优化参数
    cash_confirm=0,          # 进入空仓需要连续N天weak (0=原版, 每天判断)
    cash_min_days=0,         # 空仓后最少持N天 (0=原版, 无限制)
    reentry_rt=0,            # 重新入场的T阈值 (0=用rt相同值, 即无滞后)
    reentry_slope=0,         # 重新入场的斜率阈值 (0=用slope_thresh相同值)
):
    """第3层优化版"""
    V_MOM_S = V_CLOSE.pct_change(ms)
    V_MOM_L = V_CLOSE.pct_change(ml)
    
    # 第1层：方向确认
    raw_dir = (T > 0).map({True: 'BULL', False: 'BEAR'})
    mask = np.ones(len(raw_dir), dtype=bool)
    for k in range(1, dc):
        mask = mask & (raw_dir.values == raw_dir.shift(k).values)
    confirmed_dir = raw_dir.where(mask, np.nan).ffill().fillna('BULL')
    
    # 第2层：方向冷却
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
    
    # 第3层：T+斜率双重确认（优化版）
    weak_slope = SLOPE.abs() < slope_thresh
    weak_t = T.abs() < rt
    is_weak = weak_t & weak_slope
    
    # 重新入场阈值（滞回带）
    r_rt = reentry_rt if reentry_rt > 0 else rt
    r_slope = reentry_slope if reentry_slope > 0 else slope_thresh
    strong_slope = SLOPE.abs() >= r_slope
    strong_t = T.abs() >= r_rt
    is_strong = strong_t | strong_slope  # OR: 任一条件满足即可重新入场
    
    wt = pd.Series(1.0, index=T.index)
    
    # 应用第3层逻辑
    cash_count = 0  # 已空仓天数计数
    weak_count = 0  # 连续weak天数计数
    
    for i in range(len(wt)):
        if pd.isna(is_weak.iloc[i]):
            continue
        
        # 连续weak计数
        if is_weak.iloc[i]:
            weak_count += 1
        else:
            weak_count = 0
        
        if cash_count > 0:
            # 已经空仓中
            if cash_min_days > 0 and cash_count < cash_min_days:
                # 还没到最短空仓期，继续空仓
                wt.iloc[i] = 0.0
                cash_count += 1
            elif is_strong.iloc[i]:
                # 满足重新入场条件
                wt.iloc[i] = 1.0
                cash_count = 0
            else:
                # 仍不满足，继续空仓
                wt.iloc[i] = 0.0
                cash_count += 1
        else:
            # 持仓中
            if cash_confirm > 0:
                # 需要连续N天weak才空仓
                if weak_count >= cash_confirm:
                    wt.iloc[i] = 0.0
                    cash_count = 1
            else:
                # 原版: 当天weak即空仓
                if is_weak.iloc[i]:
                    wt.iloc[i] = 0.0
                    cash_count = 1
    
    # 第4层：B2
    wrong_value = (dir_raw == 'BEAR') & (V_MOM_S <= 0) & (V_MOM_L <= 0)
    dir_raw[wrong_value] = 'BULL'
    dir_s = dir_raw.map({'BULL': 'growth', 'BEAR': 'value'})
    
    # 第5层：BIAS（清仓模式）
    G_BIAS = G_CLOSE / G_CLOSE.rolling(bias_ma).mean() - 1
    V_BIAS = V_CLOSE / V_CLOSE.rolling(bias_ma).mean() - 1
    extreme_g = (dir_s == 'growth') & (G_BIAS > bias_high)
    extreme_v = (dir_s == 'value') & (V_BIAS > bias_high)
    wt[extreme_g | extreme_v] = 0.0
    
    # 第6层：E5止损
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
                    if is_weak.iloc[i] and wt.iloc[i] > 0:
                        # 如果E5恢复时signal仍然weak，E5中已有恢复逻辑
                        pass
            else:
                if wt.iloc[i] > 0: wt.iloc[i] = sw
    
    signal = dir_raw.map({'BULL': 'growth', 'BEAR': 'value'})
    return signal, wt


def test(name, builder, desc=""):
    sig, wt = builder()
    result = run_backtest(sig, wt)
    m = calc_metrics(result)
    sw = count_switches(sig, wt)
    result_sl = run_backtest(sig, wt, impact_slippage=0.0005)
    m_sl = calc_metrics(result_sl)
    
    # 统计权重变动次数
    wt_changes = sum(1 for i in range(1, len(wt)) 
                     if not pd.isna(wt.iloc[i]) and not pd.isna(wt.iloc[i-1]) 
                     and wt.iloc[i] != wt.iloc[i-1])
    
    m['calmar_sl'] = m_sl['calmar']
    m['wt_changes'] = wt_changes
    print(f"{name:35s} {m['ann']*100:>7.2f}% {m['dd']*100:>7.2f}% "
          f"{m['sharpe']:>7.3f} {m['calmar']:>7.3f} {m['n_trades']:>5d} "
          f"{wt_changes:>5d} {m['calmar_sl']:>8.3f}  {desc}")
    return m


print("=" * 105)
print("  第3层优化: 减少频繁空仓切换")
print("=" * 105)
print(f"{'版本':35s} {'年化':>7s} {'回撤':>7s} {'Sharpe':>7s} {'Calmar':>7s} {'交易':>5s} {'权重变动':>6s} {'滑点Calmar':>8s}  说明")
print("  " + "-" * 105)

results = []

# 基准: 原版X14
m0 = test("X14原版(第3层默认)", 
          lambda: build_v3_optimized(cash_confirm=0, cash_min_days=0),
          "原版: 每天判断")
results.append(("原版", m0))

# 实验1: 连续确认 - 需要连续N天weak才空仓
print("\n--- 实验1: 连续确认 (连续N天weak才空仓) ---")
for confirm in [1, 2, 3, 5, 7, 10]:
    m = test(f"  cash_confirm={confirm}", 
             lambda c=confirm: build_v3_optimized(cash_confirm=c, cash_min_days=0),
             f"连续{confirm}天weak才空仓")
    results.append((f"确认{confirm}", m))

# 实验2: 最短空仓期 - 空仓后至少N天才能恢复
print("\n--- 实验2: 最短空仓期 ---")
for min_days in [1, 2, 3, 5, 7, 10]:
    m = test(f"  cash_min_days={min_days}",
             lambda d=min_days: build_v3_optimized(cash_confirm=0, cash_min_days=d),
             f"空仓最少{min_days}天")
    results.append((f"最短{min_days}", m))

# 实验3: 滞回带 - 用不同阈值重新入场
print("\n--- 实验3: 滞回带 (进入空仓 vs 重新入场用不同阈值) ---")
# 进入 rt=1.3, 重新入场需要用更大的T
for r_rt in [1.0, 1.5, 1.8, 2.0]:
    m = test(f"  reentry_T={r_rt}",
             lambda r=r_rt: build_v3_optimized(cash_confirm=0, cash_min_days=0, reentry_rt=r),
             f"进空仓|T|<1.3, 出场|T|>={r_rt}")
    results.append((f"滞回T{r_rt}", m))

# 实验4: 联合优化 - 连续确认+最短空仓
print("\n--- 实验4: 联合优化 ---")
combos = [
    (2, 2, "连续2天weak+最少空仓2天"),
    (2, 5, "连续2天weak+最少空仓5天"),
    (3, 3, "连续3天weak+最少空仓3天"),
    (3, 5, "连续3天weak+最少空仓5天"),
    (5, 5, "连续5天weak+最少空仓5天"),
]
for cc, md, desc in combos:
    m = test(f"  cf{cc}_md{md}",
             lambda c=cc, d=md: build_v3_optimized(cash_confirm=c, cash_min_days=d),
             desc)
    results.append((f"联合{cc}_{md}", m))

# 汇总
print("\n" + "=" * 105)
print("  汇总 (按Calmar排序)")
print("=" * 105)
results.sort(key=lambda x: -x[1]['calmar'])
print(f"  {'版本':35s} {'年化':>7s} {'回撤':>7s} {'Sharpe':>7s} {'Calmar':>7s} {'交易':>5s} {'权重变动':>6s} {'滑点Calmar':>8s}")
print("  " + "-" * 85)
best = results[0]
worst = results[-1]
for name, m in results:
    marker = " ★" if name == best[0] else ""
    print(f"  {name:35s} {m['ann']*100:>7.2f}% {m['dd']*100:>7.2f}% "
          f"{m['sharpe']:>7.3f} {m['calmar']:>7.3f} {m['n_trades']:>5d} "
          f"{m['wt_changes']:>6d} {m['calmar_sl']:>8.3f}{marker}")

print(f"\n  原版权重变动: {m0['wt_changes']}次")
print(f"  最优权重变动: {best[1]['wt_changes']}次 (减少{m0['wt_changes']-best[1]['wt_changes']}次)")
print(f"  最优Calmar: {best[1]['calmar']:.3f} (vs 原版{m0['calmar']:.3f})")
print(f"  最优滑点Calmar: {best[1]['calmar_sl']:.3f} (vs 原版{m0['calmar_sl']:.3f})")
