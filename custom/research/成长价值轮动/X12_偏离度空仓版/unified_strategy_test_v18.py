"""unified_strategy_test_v18.py — 第十八轮: BIAS过滤深度优化
================================================================
v17突破: U96_BIAS0.15_r0.5 Calmar 2.224 (+0.045 vs v16)
v18: BIAS过滤参数深度优化
  - BIAS阈值细调 (0.13-0.20)
  - BIAS减仓比例细调 (0.3-0.7)
  - BIAS计算周期变体 (MA10/MA20/MA30/MA60)
  - G和V分别用不同阈值
  - 与v16其他参数联合优化
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
               use_bias=True, bias_ma=20, bias_high=0.15, bias_reduce=0.5,
               g_bias_high=None, v_bias_high=None):
    """v17 BIAS过滤+v16最优核心"""
    # BIAS计算
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

    # BIAS过滤
    if use_bias:
        extreme_g = (dir_s == 'growth') & (G_BIAS > g_thresh)
        extreme_v = (dir_s == 'value') & (V_BIAS > v_thresh)
        wt[extreme_g | extreme_v] = wt[extreme_g | extreme_v] * bias_reduce

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
    print("  统一策略第十八轮: BIAS过滤深度优化")
    print("=" * 80)

    results = []

    # ---- 基准 ----
    print("\n--- 基准: v17最优(U96_BIAS0.15_r0.5) ---")
    sig, wt = build_core(bias_high=0.15, bias_reduce=0.5)
    info = test_strategy("U96_基准", sig, wt, "v17最优")
    print_info(info); results.append(info)

    # ---- 第一组: BIAS阈值细调 (0.13-0.20) ----
    print("\n--- 第一组: BIAS阈值细调 ---")
    for bh in [0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20, 0.25, 0.30]:
        sig, wt = build_core(bias_high=bh, bias_reduce=0.5)
        info = test_strategy(f"U100_BIAS{bh}", sig, wt, f"BIAS>{bh}")
        print_info(info); results.append(info)

    # ---- 第二组: BIAS减仓比例细调 (0.3-0.7) ----
    print("\n--- 第二组: BIAS减仓比例细调 ---")
    for br in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        sig, wt = build_core(bias_high=0.15, bias_reduce=br)
        info = test_strategy(f"U101_BIAS0.15_r{br}", sig, wt, f"减仓至{br}")
        print_info(info); results.append(info)

    # ---- 第三组: BIAS计算周期变体 ----
    print("\n--- 第三组: BIAS计算周期变体 ---")
    for ma in [10, 15, 20, 25, 30, 40, 60]:
        sig, wt = build_core(bias_ma=ma, bias_high=0.15, bias_reduce=0.5)
        info = test_strategy(f"U102_BIASma{ma}", sig, wt, f"BIAS用MA{ma}")
        print_info(info); results.append(info)

    # ---- 第四组: G和V不同阈值 ----
    print("\n--- 第四组: G和V不同阈值 ---")
    for gh, vh in [(0.15, 0.15), (0.10, 0.20), (0.12, 0.18),
                    (0.15, 0.20), (0.20, 0.15), (0.18, 0.12),
                    (0.12, 0.15), (0.15, 0.12)]:
        sig, wt = build_core(bias_high=0.15, g_bias_high=gh, v_bias_high=vh)
        info = test_strategy(f"U103_G{gh}_V{vh}", sig, wt, f"G阈值{gh} V阈值{vh}")
        print_info(info); results.append(info)

    # ---- 第五组: 与v16其他参数联合 ----
    print("\n--- 第五组: 与v16其他参数联合 ---")
    combos = [
        # BIAS + sw细调
        ("U104_BIAS_sw0.15", dict(bias_high=0.15, bias_reduce=0.5, sw=0.15)),
        ("U104_BIAS_sw0.16", dict(bias_high=0.15, bias_reduce=0.5, sw=0.16)),
        ("U104_BIAS_sw0.18", dict(bias_high=0.15, bias_reduce=0.5, sw=0.18)),
        ("U104_BIAS_sw0.19", dict(bias_high=0.15, bias_reduce=0.5, sw=0.19)),
        # BIAS + st细调
        ("U104_BIAS_st0.085", dict(bias_high=0.15, bias_reduce=0.5, st=0.085)),
        ("U104_BIAS_st0.090", dict(bias_high=0.15, bias_reduce=0.5, st=0.090)),
        # BIAS + cd细调
        ("U104_BIAS_cd6", dict(bias_high=0.15, bias_reduce=0.5, cd=6)),
        ("U104_BIAS_cd7", dict(bias_high=0.15, bias_reduce=0.5, cd=7)),
        ("U104_BIAS_cd10", dict(bias_high=0.15, bias_reduce=0.5, cd=10)),
        # BIAS + rt细调
        ("U104_BIAS_rt1.2", dict(bias_high=0.15, bias_reduce=0.5, rt=1.2)),
        ("U104_BIAS_rt1.4", dict(bias_high=0.15, bias_reduce=0.5, rt=1.4)),
        # BIAS + slope细调
        ("U104_BIAS_sl0.0018", dict(bias_high=0.15, bias_reduce=0.5, slope_thresh=0.0018)),
        ("U104_BIAS_sl0.0022", dict(bias_high=0.15, bias_reduce=0.5, slope_thresh=0.0022)),
    ]
    for name, params in combos:
        sig, wt = build_core(**params)
        info = test_strategy(name, sig, wt, name.replace("U104_BIAS_", ""))
        print_info(info); results.append(info)

    # ---- 第六组: 双BIAS阈值(短+长) ----
    print("\n--- 第六组: 双BIAS阈值(短+长) ---")
    def build_dual_bias(bias_ma_short=10, bias_ma_long=30,
                        bias_high_s=0.10, bias_high_l=0.20,
                        bias_reduce=0.5, **kwargs):
        """短周期BIAS+长周期BIAS都触发才减仓"""
        G_MA_S = G_CLOSE.rolling(bias_ma_short).mean()
        V_MA_S = V_CLOSE.rolling(bias_ma_short).mean()
        G_MA_L = G_CLOSE.rolling(bias_ma_long).mean()
        V_MA_L = V_CLOSE.rolling(bias_ma_long).mean()
        G_BIAS_S = G_CLOSE / G_MA_S - 1
        V_BIAS_S = V_CLOSE / V_MA_S - 1
        G_BIAS_L = G_CLOSE / G_MA_L - 1
        V_BIAS_L = V_CLOSE / V_MA_L - 1

        sig, wt = build_core(bias_high=0.99, bias_reduce=1.0, **kwargs)  # 关闭单BIAS
        # 重新构建dir_s
        dir_s = sig
        extreme_g = (dir_s == 'growth') & (G_BIAS_S > bias_high_s) & (G_BIAS_L > bias_high_l)
        extreme_v = (dir_s == 'value') & (V_BIAS_S > bias_high_s) & (V_BIAS_L > bias_high_l)
        wt[extreme_g | extreme_v] = wt[extreme_g | extreme_v] * bias_reduce
        return sig, wt

    for bs, bl in [(0.08, 0.15), (0.10, 0.15), (0.10, 0.20),
                    (0.12, 0.18), (0.05, 0.15)]:
        sig, wt = build_dual_bias(bias_high_s=bs, bias_high_l=bl)
        info = test_strategy(f"U105_dualS{bs}_L{bl}", sig, wt, f"短BIAS>{bs}长BIAS>{bl}")
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
    print(f"  {'U96_v17(基准)':<32} {'46.56%':>7} {'-20.94%':>7} {'1.587':>7} "
          f"{'2.224':>7} {'410':>5} {'43.83%':>8} {'1.889':>10}")

    results.sort(key=lambda x: -x['calmar'])
    for r in results[:25]:
        if r['name'] == 'U96_基准':
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

    v17_calmar = 2.224
    improved = [r for r in results if r['calmar'] > v17_calmar]
    print(f"\n  超越v17(Calmar>{v17_calmar})的版本: {len(improved)}个")
    for r in improved:
        print(f"    {r['name']}: Calmar={r['calmar']:.3f} (+{r['calmar']-v17_calmar:.3f})")
