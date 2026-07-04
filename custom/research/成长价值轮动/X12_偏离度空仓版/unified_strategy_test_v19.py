"""unified_strategy_test_v19.py — 第十九轮: BIAS双甜点联合+冷却机制
================================================================
v18发现两个甜点:
  - BIAS阈值0.19 (Calmar 2.242)
  - BIAS减仓比例0.3 (Calmar 2.242)
v19: 联合这两个甜点 + BIAS新机制探索
  - BIAS阈值0.17-0.22 + r=0.3-0.4 联合扫描
  - BIAS+冷却期机制(触发后N天不再触发)
  - G和V分别用不同BIAS阈值深度优化
  - BIAS不同周期组合
  - BIAS+T联合过滤
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
               # BIAS过滤
               bias_ma=20, bias_high=0.19, bias_reduce=0.3,
               g_bias_high=None, v_bias_high=None,
               # BIAS冷却期
               bias_cooldown=0,
               # BIAS+T联合
               use_bias_t=False, t_high=2.0):
    """v18最优+BIAS新机制"""
    G_MA = G_CLOSE.rolling(bias_ma).mean()
    V_MA = V_CLOSE.rolling(bias_ma).mean()
    G_BIAS = (G_CLOSE / G_MA - 1)
    V_BIAS = (V_CLOSE / V_MA - 1)

    g_thresh = g_bias_high if g_bias_high is not None else bias_high
    v_thresh = v_bias_high if v_bias_high is not None else bias_high

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

    # BIAS过滤(带冷却期)
    extreme_g = (dir_s == 'growth') & (G_BIAS > g_thresh)
    extreme_v = (dir_s == 'value') & (V_BIAS > v_thresh)

    if use_bias_t:
        # BIAS+T联合: 同时满足BIAS大和T值大才减仓
        extreme_g = extreme_g & (T > t_high)
        extreme_v = extreme_v & (T < -t_high)

    extreme = extreme_g | extreme_v

    if bias_cooldown > 0:
        # 带冷却期: 触发后N天内不再触发
        bias_trigger = extreme.copy()
        triggered_days = 0
        bias_active = pd.Series(False, index=T.index)
        for i in range(len(wt)):
            if pd.isna(wt.iloc[i]):
                continue
            if bias_trigger.iloc[i] and triggered_days == 0:
                bias_active.iloc[i] = True
                triggered_days = bias_cooldown
            else:
                if triggered_days > 0:
                    triggered_days -= 1
        wt[bias_active] = wt[bias_active] * bias_reduce
    else:
        wt[extreme] = wt[extreme] * bias_reduce

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
    print("  统一策略第十九轮: BIAS双甜点联合+新机制")
    print("=" * 80)

    results = []

    # ---- 基准 ----
    print("\n--- 基准: v18最优(U100_BIAS0.19) ---")
    sig, wt = build_core(bias_high=0.19, bias_reduce=0.3)
    info = test_strategy("U100_基准", sig, wt, "v18最优")
    print_info(info); results.append(info)

    # ---- 第一组: BIAS双甜点联合(0.17-0.22 + r=0.3-0.4) ----
    print("\n--- 第一组: BIAS双甜点联合扫描 ---")
    for bh in [0.17, 0.18, 0.19, 0.20, 0.21, 0.22]:
        for br in [0.3, 0.35, 0.4]:
            sig, wt = build_core(bias_high=bh, bias_reduce=br)
            info = test_strategy(f"U106_BIAS{bh}_r{br}", sig, wt, f"BIAS>{bh} r={br}")
            print_info(info); results.append(info)

    # ---- 第二组: BIAS+冷却期 ----
    print("\n--- 第二组: BIAS+冷却期 ---")
    for bcd in [0, 3, 5, 7, 10, 15, 20]:
        sig, wt = build_core(bias_high=0.19, bias_reduce=0.3, bias_cooldown=bcd)
        info = test_strategy(f"U107_BIAScd{bcd}", sig, wt, f"BIAS冷却{bcd}天")
        print_info(info); results.append(info)

    # ---- 第三组: G和V不同阈值深度优化 ----
    print("\n--- 第三组: G和V不同阈值深度优化 ---")
    for gh in [0.15, 0.18, 0.20, 0.22]:
        for vh in [0.15, 0.18, 0.20, 0.22]:
            sig, wt = build_core(bias_high=0.19, bias_reduce=0.3,
                                  g_bias_high=gh, v_bias_high=vh)
            info = test_strategy(f"U108_G{gh}_V{vh}", sig, wt, f"G阈值{gh} V阈值{vh}")
            print_info(info); results.append(info)

    # ---- 第四组: BIAS+T联合过滤 ----
    print("\n--- 第四组: BIAS+T联合过滤 ---")
    for th in [1.5, 1.8, 2.0, 2.5, 3.0]:
        sig, wt = build_core(bias_high=0.19, bias_reduce=0.3,
                              use_bias_t=True, t_high=th)
        info = test_strategy(f"U109_BIAST{th}", sig, wt, f"BIAS+T>{th}")
        print_info(info); results.append(info)

    # ---- 第五组: BIAS不同周期组合 ----
    print("\n--- 第五组: BIAS不同周期组合 ---")
    for ma in [15, 20, 25, 30]:
        sig, wt = build_core(bias_ma=ma, bias_high=0.19, bias_reduce=0.3)
        info = test_strategy(f"U110_BIASma{ma}", sig, wt, f"BIAS用MA{ma}")
        print_info(info); results.append(info)

    # ---- 第六组: 最优组合再联合v16参数 ----
    print("\n--- 第六组: 最优组合+v16参数联合 ---")
    combos = [
        ("U111_BIAS_sw0.16", dict(bias_high=0.19, bias_reduce=0.3, sw=0.16)),
        ("U111_BIAS_sw0.18", dict(bias_high=0.19, bias_reduce=0.3, sw=0.18)),
        ("U111_BIAS_st0.090", dict(bias_high=0.19, bias_reduce=0.3, st=0.090)),
        ("U111_BIAS_st0.085", dict(bias_high=0.19, bias_reduce=0.3, st=0.085)),
        ("U111_BIAS_cd7", dict(bias_high=0.19, bias_reduce=0.3, cd=7)),
        ("U111_BIAS_cd10", dict(bias_high=0.19, bias_reduce=0.3, cd=10)),
        ("U111_BIAS_rt1.2", dict(bias_high=0.19, bias_reduce=0.3, rt=1.2)),
        ("U111_BIAS_rt1.4", dict(bias_high=0.19, bias_reduce=0.3, rt=1.4)),
        ("U111_BIAS_sl0.0018", dict(bias_high=0.19, bias_reduce=0.3, slope_thresh=0.0018)),
    ]
    for name, params in combos:
        sig, wt = build_core(**params)
        info = test_strategy(name, sig, wt, name.replace("U111_BIAS_", ""))
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
    print(f"  {'U100_v18(基准)':<32} {'46.96%':>7} {'-20.94%':>7} {'1.588':>7} "
          f"{'2.242':>7} {'398':>5} {'44.31%':>8} {'1.910':>10}")

    results.sort(key=lambda x: -x['calmar'])
    for r in results[:25]:
        if r['name'] == 'U100_基准':
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

    v18_calmar = 2.242
    improved = [r for r in results if r['calmar'] > v18_calmar]
    print(f"\n  超越v18(Calmar>{v18_calmar})的版本: {len(improved)}个")
    for r in improved:
        print(f"    {r['name']}: Calmar={r['calmar']:.3f} (+{r['calmar']-v18_calmar:.3f})")
