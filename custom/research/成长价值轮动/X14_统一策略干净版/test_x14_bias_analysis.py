"""BIAS层深度测试: 阈值扫描 + 贡献拆解"""
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

def build_custom(bias_high=0.19, bias_reduce=0.05):
    """X14 干净版核心, 可调BIAS参数"""
    slope_thresh=0.002; sw=0.17; st=0.09; cd=8
    ms=10; ml=20; rt=1.3; dc=5; dcd=6; bias_ma=20
    
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


def get_stats(sig, wt, bias_high):
    """统计BIAS触发情况"""
    bias_ma=20
    G_MA = G_CLOSE.rolling(bias_ma).mean()
    V_MA = V_CLOSE.rolling(bias_ma).mean()
    G_BIAS = G_CLOSE / G_MA - 1
    V_BIAS = V_CLOSE / V_MA - 1
    dir_s = sig
    
    g_over = (dir_s == 'growth') & (G_BIAS > bias_high)
    v_over = (dir_s == 'value') & (V_BIAS > bias_high)
    total_over = g_over | v_over
    trigger_days = total_over.sum()
    
    # 触发日的平均BIAS值
    if trigger_days > 0:
        g_bias_vals = G_BIAS[g_over]
        v_bias_vals = V_BIAS[v_over]
        all_bias = pd.concat([g_bias_vals, v_bias_vals])
        avg_bias = all_bias.mean()
    else:
        avg_bias = 0
    
    return trigger_days, avg_bias


def test(name, builder, desc=""):
    sig, wt = builder()
    result = run_backtest(sig, wt)
    m = calc_metrics(result)
    sw = count_switches(sig, wt)
    result_sl = run_backtest(sig, wt, impact_slippage=0.0005)
    m_sl = calc_metrics(result_sl)
    print(f"{name:35s} {m['ann']*100:>7.2f}% {m['dd']*100:>7.2f}% "
          f"{m['sharpe']:>7.3f} {m['calmar']:>7.3f} {m['n_trades']:>5d} "
          f"{m_sl['calmar']:>8.3f}  {desc}")
    return m


# ====== 第一组: bias_high 扫描 (0.05 到 0.50) ======
print("=" * 100)
print("  第一组: bias_high阈值扫描 (bias_reduce=0.05)")
print("=" * 100)
print(f"{'bias_high':35s} {'年化':>7s} {'回撤':>7s} {'Sharpe':>7s} {'Calmar':>7s} {'交易':>5s} {'滑点Calmar':>8s}  触发天数")
print("  " + "-" * 95 + "  " + "-" * 8)

thresholds = [0.05, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.19, 0.20, 0.22, 0.25, 0.30, 0.40, 1.0]
best_calmar = 0
best_th = 0

for th in thresholds:
    sig, wt = build_custom(bias_high=th, bias_reduce=0.05)
    trig_days, _ = get_stats(sig, wt, th)
    trigger_label = f"触发{trig_days}天"
    if th >= 0.99:
        trigger_label = "≈无BIAS"
    m = test(f"bias_high={th:.2f}", lambda t=th: build_custom(bias_high=t, bias_reduce=0.05), trigger_label)
    if m['calmar'] > best_calmar:
        best_calmar = m['calmar']
        best_th = th

print(f"\n  ★ 最优阈值: bias_high={best_th:.2f}, Calmar={best_calmar:.3f}")

# 看看极端情况
print(f"\n  范围: {min(thresholds):.2f}-{max(thresholds):.2f}, Calmar极差: {best_calmar - 2.164:.3f}")


# ====== 第二组: bias_reduce 敏感度 (固定bias_high=0.19) ======
print("\n" + "=" * 100)
print("  第二组: bias_reduce降仓比例扫描 (bias_high=0.19)")
print("=" * 100)
print(f"{'bias_reduce':35s} {'年化':>7s} {'回撤':>7s} {'Sharpe':>7s} {'Calmar':>7s} {'交易':>5s} {'滑点Calmar':>8s}  说明")
print("  " + "-" * 100)

reduces = [0.0, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0]
for br in reduces:
    label = f"全清仓" if br == 0 else f"完全不降仓" if br >= 0.99 else f""
    m = test(f"bias_reduce={br:.2f}", lambda b=br: build_custom(bias_high=0.19, bias_reduce=b), label)


# ====== 第三组: 实际贡献 ======
print("\n" + "=" * 100)
print("  第三组: BIAS实际贡献拆解")
print("=" * 100)

# 无BIAS
sig, wt = build_custom(bias_high=1.0, bias_reduce=1.0)
m_no = test("无BIAS层", lambda: build_custom(bias_high=1.0, bias_reduce=1.0), "bias_high=1.0 (永不触发)")

# X14干净版
m_full = test("X14干净版(BIAS全开)", lambda: build_custom(bias_high=0.19, bias_reduce=0.05), "bias_high=0.19, bias_reduce=0.05")

print(f"\n  BIAS层贡献:")
print(f"    无BIAS:    Calmar {m_no['calmar']:.3f}  年化 {m_no['ann']*100:.2f}%")
print(f"    有BIAS:    Calmar {m_full['calmar']:.3f}  年化 {m_full['ann']*100:.2f}%")
print(f"    贡献:      Calmar +{m_full['calmar']-m_no['calmar']:.3f} (+{(m_full['calmar']/m_no['calmar']-1)*100:.1f}%)")
print(f"              年化 +{(m_full['ann']-m_no['ann'])*100:.2f}pp")

# 看看无BIAS vs 去掉B2+BIAS的区别
print(f"\n  回顾B2贡献（以便对比）:")
print(f"    之前测试: B2层贡献约 +1.003 Calmar")
print(f"    BIAS层贡献: +0.119 Calmar")
print(f"    E5+空仓:     1.210 Calmar（基础保护）")
