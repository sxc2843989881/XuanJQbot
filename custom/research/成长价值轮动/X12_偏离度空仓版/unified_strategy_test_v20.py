"""unified_strategy_test_v20.py — 第二十轮: BIAS甜点+新机制探索
================================================================
v19最优: BIAS0.19+r=0.3 Calmar 2.268 (+0.026 vs v18)
v20: 在v19最优基础上尝试5个新方向
  1. BIAS更细参数(0.185-0.195, r=0.25-0.35)
  2. 价格突破过滤(突破N日新高减仓)
  3. E5恢复确认机制(E5触发后N天才能恢复正常仓位)
  4. 波动率自适应BIAS阈值
  5. 持仓时间限制(持仓N天后强制减仓)
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

# ATR
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
G_ATR_PCT = G_ATR / G_CLOSE
V_ATR_PCT = V_ATR / V_CLOSE
ATR_PCT_AVG = (G_ATR_PCT + V_ATR_PCT) / 2


def build_core(slope_thresh=0.002, sw=0.17, st=0.088, cd=8,
               ms=10, ml=20, rt=1.3, dc=5, dcd=5,
               # BIAS过滤
               bias_ma=20, bias_high=0.19, bias_reduce=0.3,
               # 新机制
               use_breakout=False, breakout_days=60, breakout_reduce=0.5,
               use_e5_recovery=False, e5_recovery_days=3,
               use_atr_bias=False, atr_bias_factor=1.5,
               use_max_hold=False, max_hold_days=30, max_hold_reduce=0.5):
    """v19最优+新机制"""
    # BIAS计算
    if use_atr_bias:
        # 波动率自适应BIAS阈值
        atr_factor = (ATR_PCT_AVG / ATR_PCT_AVG.rolling(60).mean()).clip(0.5, 2.0)
        bias_thresh_dyn = bias_high * atr_factor * atr_bias_factor
        bias_thresh_dyn = pd.Series(bias_thresh_dyn, index=T.index)
    else:
        bias_thresh_dyn = pd.Series(bias_high, index=T.index)

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

    # BIAS过滤(动态阈值)
    extreme_g = (dir_s == 'growth') & (G_BIAS > bias_thresh_dyn)
    extreme_v = (dir_s == 'value') & (V_BIAS > bias_thresh_dyn)
    extreme = extreme_g | extreme_v
    wt[extreme] = wt[extreme] * bias_reduce

    # 价格突破过滤
    if use_breakout:
        # 当前价格创N日新高时减仓
        G_NEW_HIGH = G_CLOSE >= G_CLOSE.rolling(breakout_days).max().shift(1)
        V_NEW_HIGH = V_CLOSE >= V_CLOSE.rolling(breakout_days).max().shift(1)
        breakout_g = (dir_s == 'growth') & G_NEW_HIGH
        breakout_v = (dir_s == 'value') & V_NEW_HIGH
        wt[breakout_g | breakout_v] = wt[breakout_g | breakout_v] * breakout_reduce

    # E5止损
    gs = (dir_s == 'growth') & (G_DD20 < -st)
    vs = (dir_s == 'value') & (V_DD20 < -st)
    e5_trigger = gs | vs

    in_cooldown = False
    cooldown_count = 0
    recovery_count = 0  # E5恢复确认计数
    in_recovery = False  # E5恢复期

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
                    if use_e5_recovery:
                        # 进入恢复期
                        in_recovery = True
                        recovery_count = 0
                        if is_weak.iloc[i]:
                            wt.iloc[i] = 0.0
                        else:
                            wt.iloc[i] = sw  # 恢复期保持低仓位
                    else:
                        if is_weak.iloc[i]:
                            wt.iloc[i] = 0.0
                        else:
                            wt.iloc[i] = 1.0
            else:
                if wt.iloc[i] > 0:
                    wt.iloc[i] = sw
        elif in_recovery and use_e5_recovery:
            recovery_count += 1
            if recovery_count >= e5_recovery_days:
                in_recovery = False
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
        prev_dir = None
        for i in range(len(wt)):
            if pd.isna(wt.iloc[i]) or pd.isna(dir_s.iloc[i]):
                continue
            if dir_s.iloc[i] != prev_dir:
                hold_count = 0
                prev_dir = dir_s.iloc[i]
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
    print("  统一策略第二十轮: BIAS甜点+新机制探索")
    print("=" * 80)

    results = []

    # ---- 基准 ----
    print("\n--- 基准: v19最优(BIAS0.19+r=0.3) ---")
    sig, wt = build_core(bias_high=0.19, bias_reduce=0.3)
    info = test_strategy("U106_基准", sig, wt, "v19最优")
    print_info(info); results.append(info)

    # ---- 第一组: BIAS更细参数 ----
    print("\n--- 第一组: BIAS更细参数 ---")
    for bh in [0.185, 0.19, 0.195]:
        for br in [0.25, 0.30, 0.35]:
            sig, wt = build_core(bias_high=bh, bias_reduce=br)
            info = test_strategy(f"U112_BIAS{bh}_r{br}", sig, wt, f"BIAS>{bh} r={br}")
            print_info(info); results.append(info)

    # ---- 第二组: 价格突破过滤 ----
    print("\n--- 第二组: 价格突破过滤 ---")
    for bd, br in [(60, 0.5), (90, 0.5), (120, 0.5),
                    (60, 0.3), (60, 0.7), (90, 0.3),
                    (250, 0.5), (30, 0.5)]:
        sig, wt = build_core(bias_high=0.19, bias_reduce=0.3,
                              use_breakout=True, breakout_days=bd, breakout_reduce=br)
        info = test_strategy(f"U113_BO{bd}_r{br}", sig, wt, f"突破{bd}日新高减仓至{br}")
        print_info(info); results.append(info)

    # ---- 第三组: E5恢复确认机制 ----
    print("\n--- 第三组: E5恢复确认机制 ---")
    for rd in [0, 2, 3, 5, 7, 10]:
        sig, wt = build_core(bias_high=0.19, bias_reduce=0.3,
                              use_e5_recovery=True, e5_recovery_days=rd)
        info = test_strategy(f"U114_E5rec{rd}", sig, wt, f"E5恢复{rd}天")
        print_info(info); results.append(info)

    # ---- 第四组: 波动率自适应BIAS阈值 ----
    print("\n--- 第四组: 波动率自适应BIAS阈值 ---")
    for af in [0.8, 1.0, 1.2, 1.5, 2.0]:
        sig, wt = build_core(bias_high=0.19, bias_reduce=0.3,
                              use_atr_bias=True, atr_bias_factor=af)
        info = test_strategy(f"U115_ATRbias{af}", sig, wt, f"ATR因子{af}")
        print_info(info); results.append(info)

    # ---- 第五组: 持仓时间限制 ----
    print("\n--- 第五组: 持仓时间限制 ---")
    for mh, mr in [(30, 0.5), (45, 0.5), (60, 0.5),
                    (30, 0.3), (60, 0.7), (90, 0.5)]:
        sig, wt = build_core(bias_high=0.19, bias_reduce=0.3,
                              use_max_hold=True, max_hold_days=mh, max_hold_reduce=mr)
        info = test_strategy(f"U116_MH{mh}_r{mr}", sig, wt, f"持仓{mh}天减仓至{mr}")
        print_info(info); results.append(info)

    # ---- 第六组: 组合尝试 ----
    print("\n--- 第六组: 组合尝试 ---")
    combos = [
        ("U117_BIAS+BO", dict(bias_high=0.19, bias_reduce=0.3,
                                use_breakout=True, breakout_days=60, breakout_reduce=0.5)),
        ("U117_BIAS+E5rec", dict(bias_high=0.19, bias_reduce=0.3,
                                  use_e5_recovery=True, e5_recovery_days=3)),
        ("U117_BIAS+ATR", dict(bias_high=0.19, bias_reduce=0.3,
                                use_atr_bias=True, atr_bias_factor=1.0)),
        ("U117_BIAS+MH", dict(bias_high=0.19, bias_reduce=0.3,
                               use_max_hold=True, max_hold_days=60, max_hold_reduce=0.5)),
        ("U117_BO+E5rec", dict(bias_high=0.19, bias_reduce=0.3,
                                use_breakout=True, breakout_days=60, breakout_reduce=0.5,
                                use_e5_recovery=True, e5_recovery_days=3)),
        ("U117_ALL", dict(bias_high=0.19, bias_reduce=0.3,
                           use_breakout=True, breakout_days=60, breakout_reduce=0.5,
                           use_e5_recovery=True, e5_recovery_days=3,
                           use_max_hold=True, max_hold_days=60, max_hold_reduce=0.5)),
    ]
    for name, params in combos:
        sig, wt = build_core(**params)
        info = test_strategy(name, sig, wt, name.replace("U117_", ""))
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
    print(f"  {'U106_v19(基准)':<32} {'47.48%':>7} {'-20.94%':>7} {'1.605':>7} "
          f"{'2.268':>7} {'398':>5} {'44.83%':>8} {'1.933':>10}")

    results.sort(key=lambda x: -x['calmar'])
    for r in results[:25]:
        if r['name'] == 'U106_基准':
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

    v19_calmar = 2.268
    improved = [r for r in results if r['calmar'] > v19_calmar]
    print(f"\n  超越v19(Calmar>{v19_calmar})的版本: {len(improved)}个")
    for r in improved:
        print(f"    {r['name']}: Calmar={r['calmar']:.3f} (+{r['calmar']-v19_calmar:.3f})")
