"""X13 纯裸跑 — 无任何附加逻辑
================================================================
去掉: dc方向确认=1天, dcd=0, B2去掉
只剩: T>0→growth, T<0→value, 永远满仓
================================================================
"""
import sys
sys.path.insert(0, r'c:\XuanJLH\Qbot\custom\backtests')
sys.path.insert(0, r'c:\XuanJLH\Qbot\custom\research\成长价值轮动\X12_偏离度空仓版')
sys.path.insert(0, r'c:\XuanJLH\Qbot\custom\research\成长价值轮动\X13_统一策略最优版')

import numpy as np
import pandas as pd
from optimize_runner import (
    G_CLOSE, V_CLOSE, RATIO, RATIO_MA20, RATIO_DEV, MA20_SLOPE,
    V_MOM20, G_DD20, V_DD20, BASE_DIR,
    run_backtest, calc_metrics, count_switches,
)
from run_x33_reduce_trades import RATIO_DEV_Z

T = RATIO_DEV_Z


def test(name, sig, wt, desc=""):
    result = run_backtest(sig, wt)
    m = calc_metrics(result)
    sw = count_switches(sig, wt)
    result_sl = run_backtest(sig, wt, impact_slippage=0.0005)
    m_sl = calc_metrics(result_sl)
    print(f"  {name}: {desc}")
    print(f"    年化={m['ann']*100:.2f}%  回撤={m['dd']*100:.2f}%  "
          f"Sharpe={m['sharpe']:.3f}  Calmar={m['calmar']:.3f}  "
          f"交易={m['n_trades']}  (方向{sw['dir']}+空仓{sw['cash']})")
    print(f"    5/万滑点: 年化={m_sl['ann']*100:.2f}%  "
          f"Calmar={m_sl['calmar']:.3f}")
    return {'name': name, 'ann': m['ann'], 'dd': m['dd'],
            'sharpe': m['sharpe'], 'calmar': m['calmar'],
            'calmar_sl': m_sl['calmar'], 'n_trades': m['n_trades']}


if __name__ == '__main__':
    print("=" * 80)
    print("  X13 纯裸跑 — 无方向确认/无冷却/无B2/无任何防回撤")
    print("=" * 80)

    results = []

    # ── X13 完整版（基准） ──
    print("\n--- 【对比】X13 完整版 v26 U164 ---")
    from backtest_x13_engine import build_core as x13_build
    sig, wt = x13_build()
    r = test("X13完整版", sig, wt, "全机制")
    results.append(r)

    # ── 纯裸版：T>0→growth, T<0→value，一秒都不等 ──
    print("\n--- ★ 纯裸版: T>0买成长, T<0买价值, 永远满仓 ---")
    signal = (T > 0).map({True: 'growth', False: 'value'})
    wt = pd.Series(1.0, index=T.index)
    r = test("X13_纯裸版_T-only", signal, wt, "无任何附加")
    results.append(r)

    # ── 还有更裸的吗？试试用RATIO_DEV代替T ──
    print("\n--- 纯裸版改用RATIO_DEV (不用z-score归一化) ---")
    ratio_dev = RATIO / RATIO.rolling(20).mean() - 1
    signal_dev = (ratio_dev > 0).map({True: 'growth', False: 'value'})
    wt = pd.Series(1.0, index=T.index)
    r = test("X13_纯裸版_RATIO_DEV", signal_dev, wt, "用偏离度代替T")
    results.append(r)

    # ── 甚至用原始RATIO (比价) ──
    print("\n--- 纯裸版用RATIO>1 (比价>1买成长) ---")
    signal_raw = (RATIO > 1).map({True: 'growth', False: 'value'})
    wt = pd.Series(1.0, index=T.index)
    r = test("X13_纯裸版_RATIO>1", signal_raw, wt, "原始比价")
    results.append(r)

    # ── 再试试随机方向（作为下限参考） ──
    print("\n--- 随机方向(上下限参考) ---")
    np.random.seed(42)
    rand_dir = np.random.choice(['growth', 'value'], size=len(T))
    signal_rand = pd.Series(rand_dir, index=T.index)
    wt = pd.Series(1.0, index=T.index)
    r = test("X13_随机方向", signal_rand, wt, "完全随机")
    results.append(r)

    # ── 一直买成长不动 ──
    print("\n--- 一直买成长(不动) ---")
    signal_bh = pd.Series('growth', index=T.index)
    wt = pd.Series(1.0, index=T.index)
    r = test("X13_恒定成长", signal_bh, wt, "满仓成长不动")
    results.append(r)

    # ── 一直买价值不动 ──
    print("\n--- 一直买价值(不动) ---")
    signal_bh = pd.Series('value', index=T.index)
    wt = pd.Series(1.0, index=T.index)
    r = test("X13_恒定价值", signal_bh, wt, "满仓价值不动")
    results.append(r)

    # ── X11-A 基准 ──
    print("\n--- X11-A 基准 ---")
    from optimize_runner import build_x11a
    sig, wt = build_x11a()
    r = test("X11-A", sig, wt, "历史基准")
    results.append(r)

    # ── 汇总 ──
    print("\n" + "=" * 80)
    print("  汇总")
    print("=" * 80)
    print(f"  {'名称':<24} {'年化':>8} {'回撤':>8} {'Sharpe':>8} {'Calmar':>8} "
          f"{'交易':>6} {'滑点Calmar':>12}")
    print("  " + "-" * 84)
    for r in results:
        print(f"  {r['name']:<24} {r['ann']*100:>7.2f}% {r['dd']*100:>7.2f}% "
              f"{r['sharpe']:>8.3f} {r['calmar']:>8.3f} {r['n_trades']:>6} "
              f"{r['calmar_sl']:>12.3f}")

    print("\n  ★ 纯裸跑完成!")
    print("=" * 80)
