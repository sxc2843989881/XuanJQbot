"""unified_strategy_test_v17.py — 第十七轮: 新因子探索
================================================================
v16最优: U87_sw0.17 年化45.62% 回撤-20.94% Calmar 2.179
当前框架已平台期, 尝试5个新方向:
  1. RSI辅助过滤(极端超买超卖反向)
  2. ATR自适应E5止损(波动率环境自适应)
  3. BIAS乖离率过滤(极端乖离反向)
  4. 布林带宽度过滤(低波动率减仓)
  5. T多周期组合(T_short + T_long加权)
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

T = RATIO_DEV_Z
SLOPE = MA20_SLOPE

# ============================================================
# 基础因子准备
# ============================================================
G_MA20 = G_CLOSE.rolling(20).mean()
V_MA20 = V_CLOSE.rolling(20).mean()
G_BIAS = (G_CLOSE / G_MA20 - 1)
V_BIAS = (V_CLOSE / V_MA20 - 1)

# RSI(14)
def calc_rsi(close, period=14):
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

G_RSI = calc_rsi(G_CLOSE, 14)
V_RSI = calc_rsi(V_CLOSE, 14)

# ATR(20)
def calc_atr(close, period=20):
    high = close.rolling(period).max()
    low = close.rolling(period).min()
    prev_close = close.shift(1)
    tr = pd.concat([(high - low).abs(),
                    (high - prev_close).abs(),
                    (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()

G_ATR = calc_atr(G_CLOSE, 20)
V_ATR = calc_atr(V_CLOSE, 20)
G_ATR_PCT = G_ATR / G_CLOSE  # ATR占比
V_ATR_PCT = V_ATR / V_CLOSE
ATR_PCT_AVG = (G_ATR_PCT + V_ATR_PCT) / 2

# 布林带宽度
RATIO_STD20 = RATIO.rolling(20).std()
RATIO_BB_WIDTH = RATIO_STD20 / RATIO_MA20  # 标准化宽度
BB_WIDTH_MA60 = RATIO_BB_WIDTH.rolling(60).mean()
BB_WIDTH_RATIO = RATIO_BB_WIDTH / BB_WIDTH_MA60  # 当前宽度/历史平均


# ============================================================
# 核心build函数(带可选因子)
# ============================================================
def build_core(slope_thresh=0.002, sw=0.17, st=0.088, cd=8,
               ms=10, ml=20, rt=1.3, dc=5, dcd=5,
               # 新因子开关
               use_rsi=False, rsi_high=70, rsi_low=30, rsi_reduce=0.5,
               use_atr_e5=False, atr_period=20,
               use_bias=False, bias_high=0.10, bias_reduce=0.5,
               use_bb=False, bb_low=0.7, bb_reduce=0.5,
               use_t_multi=False, t_short=10, t_long=30, t_long_weight=0.3):
    """v16最优基础上+新因子"""
    # --- T指标(可选多周期) ---
    if use_t_multi:
        RATIO_MA_S = RATIO.rolling(t_short).mean()
        RATIO_MA_L = RATIO.rolling(t_long).mean()
        RATIO_DEV_S = RATIO / RATIO_MA_S - 1
        RATIO_DEV_L = RATIO / RATIO_MA_L - 1
        RATIO_DEV_STD_S = RATIO_DEV_S.rolling(20).std()
        RATIO_DEV_STD_L = RATIO_DEV_L.rolling(20).std()
        T_S = RATIO_DEV_S / RATIO_DEV_STD_S
        T_L = RATIO_DEV_L / RATIO_DEV_STD_L
        T_eff = (1 - t_long_weight) * T_S + t_long_weight * T_L
    else:
        T_eff = T

    V_MOM_S = V_CLOSE.pct_change(ms)
    V_MOM_L = V_CLOSE.pct_change(ml)

    raw_dir = (T_eff > 0).map({True: 'BULL', False: 'BEAR'})
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
    weak_t = T_eff.abs() < rt
    is_weak = weak_t & weak_slope

    wt = pd.Series(1.0, index=T.index)
    wt[is_weak] = 0.0

    # B2改进
    wrong_value = (dir_raw == 'BEAR') & (V_MOM_S <= 0) & (V_MOM_L <= 0)
    dir_raw[wrong_value] = 'BULL'
    dir_s = dir_raw.map({'BULL': 'growth', 'BEAR': 'value'})

    # --- RSI辅助过滤 ---
    if use_rsi:
        rsi = pd.Series(np.nan, index=T.index)
        rsi[dir_s == 'growth'] = G_RSI
        rsi[dir_s == 'value'] = V_RSI
        # 极端超买(>rsi_high)减仓
        overbought = rsi > rsi_high
        wt[overbought] = wt[overbought] * rsi_reduce

    # --- BIAS过滤 ---
    if use_bias:
        bias = pd.Series(np.nan, index=T.index)
        bias[dir_s == 'growth'] = G_BIAS
        bias[dir_s == 'value'] = V_BIAS
        # 极端乖离(>bias_high)减仓
        extreme_bias = bias > bias_high
        wt[extreme_bias] = wt[extreme_bias] * bias_reduce

    # --- 布林带宽度过滤 ---
    if use_bb:
        # 当前宽度低于历史均值*bb_low → 减仓
        low_volatility = BB_WIDTH_RATIO < bb_low
        wt[low_volatility] = wt[low_volatility] * bb_reduce

    # --- E5止损(ATR自适应) ---
    if use_atr_e5:
        # 高波动率时止损阈值放宽
        atr_factor = (ATR_PCT_AVG / ATR_PCT_AVG.rolling(60).mean()).clip(0.5, 2.0)
        dynamic_st = st * atr_factor
    else:
        dynamic_st = pd.Series(st, index=T.index)

    gs = (dir_s == 'growth') & (G_DD20 < -dynamic_st)
    vs = (dir_s == 'value') & (V_DD20 < -dynamic_st)
    e5_trigger = gs | vs

    in_cooldown = False
    cooldown_count = 0
    for i in range(len(wt)):
        if pd.isna(wt.iloc[i]) or pd.isna(dir_s.iloc[i]):
            continue
        if e5_trigger.iloc[i] and not in_cooldown:
            in_cooldown = True
            cooldown_count = 0
            wt.iloc[i] = wt.iloc[i] * sw
        elif in_cooldown:
            cooldown_count += 1
            if cooldown_count >= cd:
                if e5_trigger.iloc[i]:
                    cooldown_count = 0
                    wt.iloc[i] = wt.iloc[i] * sw
                else:
                    in_cooldown = False
                    if is_weak.iloc[i]:
                        wt.iloc[i] = 0.0
                    else:
                        wt.iloc[i] = 1.0
            else:
                if wt.iloc[i] > 0:
                    wt.iloc[i] = sw

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
          f"交易={info['n_trades']}")
    print(f"    5/万滑点: 年化={info['ann_sl']*100:.2f}% Calmar={info['calmar_sl']:.3f}")


if __name__ == '__main__':
    print("=" * 80)
    print("  统一策略第十七轮: 新因子探索")
    print("=" * 80)

    results = []

    # ---- 基准: v16最优 ----
    print("\n--- 基准: v16最优(U87_sw0.17) ---")
    sig, wt = build_core()
    info = test_strategy("U87_基准", sig, wt, "v16最优")
    print_info(info); results.append(info)

    # ---- 第一组: RSI辅助过滤 ----
    print("\n--- 第一组: RSI辅助过滤 ---")
    for rh, rl, rr in [(70, 30, 0.5), (75, 25, 0.5), (80, 20, 0.3),
                        (70, 30, 0.3), (75, 25, 0.7), (65, 35, 0.5)]:
        sig, wt = build_core(use_rsi=True, rsi_high=rh, rsi_low=rl, rsi_reduce=rr)
        info = test_strategy(f"U94_RSI{rh}_{rl}_r{rr}", sig, wt,
                              f"RSI>{rh}减仓至{rr}")
        print_info(info); results.append(info)

    # ---- 第二组: ATR自适应E5止损 ----
    print("\n--- 第二组: ATR自适应E5止损 ---")
    sig, wt = build_core(use_atr_e5=True)
    info = test_strategy("U95_ATR_E5", sig, wt, "ATR自适应st")
    print_info(info); results.append(info)

    # ---- 第三组: BIAS过滤 ----
    print("\n--- 第三组: BIAS过滤 ---")
    for bh, br in [(0.08, 0.5), (0.10, 0.5), (0.12, 0.5),
                    (0.10, 0.3), (0.10, 0.7), (0.15, 0.5)]:
        sig, wt = build_core(use_bias=True, bias_high=bh, bias_reduce=br)
        info = test_strategy(f"U96_BIAS{bh}_r{br}", sig, wt,
                              f"BIAS>{bh}减仓至{br}")
        print_info(info); results.append(info)

    # ---- 第四组: 布林带宽度过滤 ----
    print("\n--- 第四组: 布林带宽度过滤 ---")
    for bl, br in [(0.7, 0.5), (0.6, 0.5), (0.8, 0.5),
                    (0.7, 0.3), (0.7, 0.7), (0.5, 0.5)]:
        sig, wt = build_core(use_bb=True, bb_low=bl, bb_reduce=br)
        info = test_strategy(f"U97_BB{bl}_r{br}", sig, wt,
                              f"BB宽度<{bl}减仓至{br}")
        print_info(info); results.append(info)

    # ---- 第五组: T多周期组合 ----
    print("\n--- 第五组: T多周期组合 ---")
    for ts, tl, tw in [(10, 30, 0.3), (10, 40, 0.3), (5, 30, 0.3),
                        (10, 30, 0.2), (10, 30, 0.4), (10, 30, 0.5),
                        (15, 45, 0.3), (10, 60, 0.3)]:
        sig, wt = build_core(use_t_multi=True, t_short=ts, t_long=tl, t_long_weight=tw)
        info = test_strategy(f"U98_T{ts}_{tl}_w{tw}", sig, wt,
                              f"T多周期短{ts}长{tl}权重{tw}")
        print_info(info); results.append(info)

    # ---- 第六组: 组合尝试(RSI+ATR, BIAS+ATR, T多周期+ATR) ----
    print("\n--- 第六组: 组合尝试 ---")
    combos = [
        ("U99_RSI+ATR", dict(use_rsi=True, rsi_high=75, rsi_low=25, rsi_reduce=0.5, use_atr_e5=True)),
        ("U99_BIAS+ATR", dict(use_bias=True, bias_high=0.10, bias_reduce=0.5, use_atr_e5=True)),
        ("U99_Tmulti+ATR", dict(use_t_multi=True, t_short=10, t_long=30, t_long_weight=0.3, use_atr_e5=True)),
        ("U99_RSI+BIAS", dict(use_rsi=True, rsi_high=75, rsi_reduce=0.5, use_bias=True, bias_high=0.10, bias_reduce=0.5)),
        ("U99_RSI+BB", dict(use_rsi=True, rsi_high=75, rsi_reduce=0.5, use_bb=True, bb_low=0.7, bb_reduce=0.5)),
        ("U99_ALL", dict(use_rsi=True, rsi_high=75, rsi_reduce=0.5,
                          use_bias=True, bias_high=0.10, bias_reduce=0.5,
                          use_bb=True, bb_low=0.7, bb_reduce=0.5,
                          use_atr_e5=True)),
    ]
    for name, params in combos:
        sig, wt = build_core(**params)
        info = test_strategy(name, sig, wt, name.replace("U99_", ""))
        print_info(info); results.append(info)

    # ---- 汇总 ----
    print("\n" + "=" * 80)
    print("  最终汇总")
    print("=" * 80)
    print(f"\n  {'名称':<32} {'年化':>7} {'回撤':>7} {'Sharpe':>7} {'Calmar':>7} "
          f"{'交易':>5} {'滑点年化':>8} {'滑点Calmar':>10}")
    print("  " + "-" * 92)
    print(f"  {'X61(基准)':<32} {'41.12%':>7} {'-30.21%':>7} {'1.357':>7} "
          f"{'1.361':>7} {'224':>5} {'39.04%':>8} {'1.214':>10}")
    print(f"  {'U87_v16(基准)':<32} {'45.62%':>7} {'-20.94%':>7} {'1.538':>7} "
          f"{'2.179':>7} {'388':>5} {'43.08%':>8} {'1.857':>10}")

    results.sort(key=lambda x: -x['calmar'])
    for r in results[:25]:
        if r['name'] in ['U87_基准']:
            continue
        print(f"  {r['name']:<32} {r['ann']*100:>6.2f}% {r['dd']*100:>6.2f}% "
              f"{r['sharpe']:>7.3f} {r['calmar']:>7.3f} {r['n_trades']:>5} "
              f"{r['ann_sl']*100:>7.2f}% {r['calmar_sl']:>10.3f}")

    print(f"\n  Top 15 Calmar:")
    for r in results[:15]:
        print(f"  {r['name']:<32} 年化{r['ann']*100:.2f}% 回撤{r['dd']*100:.2f}% "
              f"Calmar{r['calmar']:.3f} 滑点Calmar{r['calmar_sl']:.3f}")

    best = max(results, key=lambda x: x['calmar'])
    print(f"\n  ★最优: {best['name']} Calmar={best['calmar']:.3f} "
          f"年化={best['ann']*100:.2f}% 回撤={best['dd']*100:.2f}%")

    # 与v16基准对比
    v16_calmar = 2.179
    improved = [r for r in results if r['calmar'] > v16_calmar]
    print(f"\n  超越v16(Calmar>{v16_calmar})的版本: {len(improved)}个")
    for r in improved:
        print(f"    {r['name']}: Calmar={r['calmar']:.3f} (+{r['calmar']-v16_calmar:.3f})")
