"""unified_strategy_test_v24.py — 第二十四轮: 滑点Calmar>2.0甜点深度优化
================================================================
v23突破:
  - U135_MH92_Br0.15_MHr0.2: Calmar 2.343 (年化49.07% 回撤-20.94%)
  - 滑点Calmar首次突破2.0(2.001)
  - 23个版本超越v22(2.323)
v24: 围绕MH92+Br0.10-0.15+MHr0.2-0.3甜点深度优化
  1. MH(88-96) + Br(0.05-0.20) + MHr(0.2-0.4) 更细扫描
  2. dcd=6甜点联合(MH92+Br0.15+MHr0.2)
  3. 探索新过滤因子(波动率/RSI/累计涨幅)
  4. 双BIAS机制(短长联合)
  5. 持仓时间细调(88-96天)
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
from run_x33_reduce_trades import RATIO_DEV_Z

T = RATIO_DEV_Z
SLOPE = MA20_SLOPE


def build_core(slope_thresh=0.002, sw=0.17, st=0.088, cd=8,
               ms=10, ml=20, rt=1.3, dc=5, dcd=5,
               bias_ma=20, bias_high=0.19, bias_reduce=0.15,
               use_max_hold=True, max_hold_days=92, max_hold_reduce=0.3,
               hold_mode='reset_dir',
               use_vol_filter=False, vol_lookback=20, vol_high=0.025,
               use_rsi_filter=False, rsi_period=14, rsi_high=80, rsi_reduce=0.5,
               use_runup_filter=False, runup_days=60, runup_high=0.30, runup_reduce=0.5,
               use_double_bias=False, bias_high_long=0.30, bias_ma_long=60):
    """v23最优核心 + 新可选过滤因子"""
    G_MA = G_CLOSE.rolling(bias_ma).mean()
    V_MA = V_CLOSE.rolling(bias_ma).mean()
    G_BIAS = (G_CLOSE / G_MA - 1)
    V_BIAS = (V_CLOSE / V_MA - 1)

    # 双BIAS(长周期)
    if use_double_bias:
        G_MA_L = G_CLOSE.rolling(bias_ma_long).mean()
        V_MA_L = V_CLOSE.rolling(bias_ma_long).mean()
        G_BIAS_L = (G_CLOSE / G_MA_L - 1)
        V_BIAS_L = (V_CLOSE / V_MA_L - 1)

    # 波动率
    if use_vol_filter:
        G_VOL = G_CLOSE.pct_change().rolling(vol_lookback).std()
        V_VOL = V_CLOSE.pct_change().rolling(vol_lookback).std()

    # RSI
    if use_rsi_filter:
        def calc_rsi(series, period):
            delta = series.diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(period).mean()
            avg_loss = loss.rolling(period).mean()
            rs = avg_gain / avg_loss.replace(0, np.nan)
            return 100 - (100 / (1 + rs))
        G_RSI = calc_rsi(G_CLOSE, rsi_period)
        V_RSI = calc_rsi(V_CLOSE, rsi_period)

    # 累计涨幅
    if use_runup_filter:
        G_RUNUP = G_CLOSE.pct_change(runup_days)
        V_RUNUP = V_CLOSE.pct_change(runup_days)

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

    # BIAS过滤
    extreme_g = (dir_s == 'growth') & (G_BIAS > bias_high)
    extreme_v = (dir_s == 'value') & (V_BIAS > bias_high)
    extreme = extreme_g | extreme_v
    wt[extreme] = wt[extreme] * bias_reduce

    # 双BIAS过滤
    if use_double_bias:
        extreme_l_g = (dir_s == 'growth') & (G_BIAS_L > bias_high_long)
        extreme_l_v = (dir_s == 'value') & (V_BIAS_L > bias_high_long)
        extreme_l = extreme_l_g | extreme_l_v
        wt[extreme_l] = wt[extreme_l] * bias_reduce

    # 波动率过滤
    if use_vol_filter:
        high_vol_g = (dir_s == 'growth') & (G_VOL > vol_high)
        high_vol_v = (dir_s == 'value') & (V_VOL > vol_high)
        high_vol = high_vol_g | high_vol_v
        wt[high_vol] = wt[high_vol] * 0.5

    # RSI过滤
    if use_rsi_filter:
        overbought_g = (dir_s == 'growth') & (G_RSI > rsi_high)
        overbought_v = (dir_s == 'value') & (V_RSI > rsi_high)
        overbought = overbought_g | overbought_v
        wt[overbought] = wt[overbought] * rsi_reduce

    # 累计涨幅过滤
    if use_runup_filter:
        runup_g = (dir_s == 'growth') & (G_RUNUP > runup_high)
        runup_v = (dir_s == 'value') & (V_RUNUP > runup_high)
        runup = runup_g | runup_v
        wt[runup] = wt[runup] * runup_reduce

    # E5止损
    gs = (dir_s == 'growth') & (G_DD20 < -st)
    vs = (dir_s == 'value') & (V_DD20 < -st)
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

    # 持仓时间限制
    if use_max_hold:
        hold_count = 0
        prev_key = None
        for i in range(len(wt)):
            if pd.isna(wt.iloc[i]) or pd.isna(dir_s.iloc[i]):
                continue
            if hold_mode == 'reset_dir':
                key = dir_s.iloc[i]
            elif hold_mode == 'reset_pos':
                key = (dir_s.iloc[i], round(wt.iloc[i], 2))
            else:
                key = None

            if key != prev_key:
                hold_count = 0
                prev_key = key
            else:
                hold_count += 1
                if hold_count >= max_hold_days:
                    if wt.iloc[i] > 0:
                        wt.iloc[i] = wt.iloc[i] * max_hold_reduce

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
    print("  统一策略第二十四轮: 滑点Calmar>2.0甜点深度优化")
    print("=" * 80)

    results = []

    # ---- 基准 ----
    print("\n--- 基准: v23最优(U135_MH92_Br0.15_MHr0.2) ---")
    sig, wt = build_core(bias_high=0.19, bias_reduce=0.15,
                         max_hold_days=92, max_hold_reduce=0.2)
    info = test_strategy("U135_基准", sig, wt, "v23最优")
    print_info(info); results.append(info)

    # ---- 第一组: MH(88-96) + Br(0.05-0.20) + MHr(0.2) 细扫描 ----
    print("\n--- 第一组: MH+Br+MHr0.2更细扫描 ---")
    for mh in [88, 89, 90, 91, 92, 93, 94, 95, 96]:
        for br in [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20]:
            sig, wt = build_core(bias_high=0.19, bias_reduce=br,
                                 max_hold_days=mh, max_hold_reduce=0.2)
            info = test_strategy(f"U140_MH{mh}_Br{br}_MHr0.2", sig, wt,
                                 f"MH{mh} Br={br} MHr=0.2")
            print_info(info); results.append(info)

    # ---- 第二组: dcd=6甜点联合(MH92+Br0.15+MHr0.2) ----
    print("\n--- 第二组: dcd=6甜点联合 ---")
    for dcd in [4, 5, 6, 7, 8]:
        sig, wt = build_core(bias_high=0.19, bias_reduce=0.15,
                             max_hold_days=92, max_hold_reduce=0.2, dcd=dcd)
        info = test_strategy(f"U141_dcd{dcd}", sig, wt, f"dcd={dcd}")
        print_info(info); results.append(info)

    # ---- 第三组: 波动率过滤 ----
    print("\n--- 第三组: 波动率过滤 ---")
    for vol_lookback in [10, 20, 30]:
        for vol_high in [0.020, 0.025, 0.030, 0.035]:
            sig, wt = build_core(bias_high=0.19, bias_reduce=0.15,
                                 max_hold_days=92, max_hold_reduce=0.2,
                                 use_vol_filter=True,
                                 vol_lookback=vol_lookback, vol_high=vol_high)
            info = test_strategy(f"U142_VOL{vol_lookback}_{vol_high}", sig, wt,
                                 f"VOL{vol_lookback}高={vol_high}")
            print_info(info); results.append(info)

    # ---- 第四组: RSI过滤 ----
    print("\n--- 第四组: RSI过滤 ---")
    for rsi_high in [75, 78, 80, 82, 85]:
        for rsi_reduce in [0.3, 0.5, 0.7]:
            sig, wt = build_core(bias_high=0.19, bias_reduce=0.15,
                                 max_hold_days=92, max_hold_reduce=0.2,
                                 use_rsi_filter=True,
                                 rsi_high=rsi_high, rsi_reduce=rsi_reduce)
            info = test_strategy(f"U143_RSI{rsi_high}_r{rsi_reduce}", sig, wt,
                                 f"RSI>{rsi_high} 减仓{rsi_reduce}")
            print_info(info); results.append(info)

    # ---- 第五组: 累计涨幅过滤 ----
    print("\n--- 第五组: 累计涨幅过滤 ---")
    for runup_days in [40, 60, 90]:
        for runup_high in [0.25, 0.30, 0.35, 0.40]:
            sig, wt = build_core(bias_high=0.19, bias_reduce=0.15,
                                 max_hold_days=92, max_hold_reduce=0.2,
                                 use_runup_filter=True,
                                 runup_days=runup_days, runup_high=runup_high,
                                 runup_reduce=0.3)
            info = test_strategy(f"U144_RU{runup_days}_{runup_high}", sig, wt,
                                 f"RU{runup_days}高={runup_high}")
            print_info(info); results.append(info)

    # ---- 第六组: 双BIAS过滤 ----
    print("\n--- 第六组: 双BIAS过滤 ---")
    for bias_ma_long in [40, 60, 80]:
        for bias_high_long in [0.25, 0.30, 0.35, 0.40]:
            sig, wt = build_core(bias_high=0.19, bias_reduce=0.15,
                                 max_hold_days=92, max_hold_reduce=0.2,
                                 use_double_bias=True,
                                 bias_ma_long=bias_ma_long,
                                 bias_high_long=bias_high_long)
            info = test_strategy(f"U145_DBL{bias_ma_long}_{bias_high_long}", sig, wt,
                                 f"双BIAS长{bias_ma_long}高={bias_high_long}")
            print_info(info); results.append(info)

    # ---- 第七组: 极限组合最终验证 ----
    print("\n--- 第七组: 极限组合最终验证 ---")
    final_combos = [
        ("U146_最终_基准", dict(bias_high=0.19, bias_reduce=0.15,
                                max_hold_days=92, max_hold_reduce=0.2)),
        ("U146_最终_BH0.20", dict(bias_high=0.20, bias_reduce=0.15,
                                   max_hold_days=92, max_hold_reduce=0.2)),
        ("U146_最终_BH0.18", dict(bias_high=0.18, bias_reduce=0.15,
                                   max_hold_days=92, max_hold_reduce=0.2)),
        ("U146_最终_sw0.16", dict(bias_high=0.19, bias_reduce=0.15, sw=0.16,
                                   max_hold_days=92, max_hold_reduce=0.2)),
        ("U146_最终_dcd6", dict(bias_high=0.19, bias_reduce=0.15, dcd=6,
                                 max_hold_days=92, max_hold_reduce=0.2)),
    ]
    for name, params in final_combos:
        sig, wt = build_core(**params)
        info = test_strategy(name, sig, wt, name.replace("U146_最终_", ""))
        print_info(info); results.append(info)

    # ---- 汇总 ----
    print("\n" + "=" * 80)
    print("  最终汇总")
    print("=" * 80)
    print(f"\n  {'名称':<46} {'年化':>7} {'回撤':>7} {'Sharpe':>7} {'Calmar':>7} "
          f"{'交易':>5} {'滑点年化':>8} {'滑点Calmar':>10}")
    print("  " + "-" * 108)
    print(f"  {'X61(基准)':<46} {'41.12%':>7} {'-30.21%':>7} {'1.357':>7} "
          f"{'1.361':>7} {'224':>5} {'39.04%':>8} {'1.214':>10}")
    print(f"  {'U125_v22(基准)':<46} {'48.65%':>7} {'-20.94%':>7} {'1.650':>7} "
          f"{'2.323':>7} {'400':>5} {'45.99%':>8} {'1.982':>10}")
    print(f"  {'U135_v23(基准)':<46} {'49.07%':>7} {'-20.94%':>7} {'1.663':>7} "
          f"{'2.343':>7} {'400':>5} {'46.41%':>8} {'2.001':>10}")

    results.sort(key=lambda x: -x['calmar'])
    for r in results[:30]:
        if r['name'] == 'U135_基准':
            continue
        print(f"  {r['name']:<46} {r['ann']*100:>6.2f}% {r['dd']*100:>6.2f}% "
              f"{r['sharpe']:>7.3f} {r['calmar']:>7.3f} {r['n_trades']:>5} "
              f"{r['ann_sl']*100:>7.2f}% {r['calmar_sl']:>10.3f}")

    print(f"\n  Top 15 Calmar:")
    for r in results[:15]:
        print(f"  {r['name']:<46} 年化{r['ann']*100:.2f}% 回撤{r['dd']*100:.2f}% "
              f"Calmar{r['calmar']:.3f} 滑点Calmar{r['calmar_sl']:.3f}")

    best = max(results, key=lambda x: x['calmar'])
    print(f"\n  ★最优: {best['name']} Calmar={best['calmar']:.3f} "
          f"年化={best['ann']*100:.2f}% 回撤={best['dd']*100:.2f}%")

    best_sl = max(results, key=lambda x: x['calmar_sl'])
    print(f"  ★滑点最优: {best_sl['name']} 滑点Calmar={best_sl['calmar_sl']:.3f} "
          f"滑点年化={best_sl['ann_sl']*100:.2f}%")

    v23_calmar = 2.343
    improved = [r for r in results if r['calmar'] > v23_calmar]
    print(f"\n  超越v23(Calmar>{v23_calmar})的版本: {len(improved)}个")
    for r in improved:
        print(f"    {r['name']}: Calmar={r['calmar']:.3f} (+{r['calmar']-v23_calmar:.3f})")
