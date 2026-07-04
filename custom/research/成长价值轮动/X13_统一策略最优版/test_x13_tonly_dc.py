"""X13 T-only + 方向确认3天
================================================================
只剩: T>0→growth, T<0→value + dc=3天确认
无冷却/无B2/无任何防回撤/永远满仓
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


def build_t_only(dc=1):
    """纯T-only: 方向确认后永远满仓"""
    raw_dir = (T > 0).map({True: 'BULL', False: 'BEAR'})
    mask = np.ones(len(raw_dir), dtype=bool)
    for k in range(1, dc):
        mask = mask & (raw_dir.values == raw_dir.shift(k).values)
    confirmed_dir = raw_dir.where(mask, np.nan).ffill().fillna('BULL')
    signal = confirmed_dir.map({'BULL': 'growth', 'BEAR': 'value'})
    wt = pd.Series(1.0, index=T.index)
    return signal, wt


if __name__ == '__main__':
    print("=" * 80)
    print("  T-only + dc扫描: 方向确认天数对纯裸版的影响")
    print("=" * 80)

    results = []

    # ── X13 完整版 ──
    print("\n--- 【对比】X13 完整版 v26 U164 ---")
    from backtest_x13_engine import build_core as x13_build
    sig, wt = x13_build()
    r = test("X13完整版", sig, wt, "全机制")
    results.append(r)

    # ── dc=1~10 扫描 ──
    print("\n--- dc=1~10 T-only 纯裸扫描 ---")
    for dc in range(1, 11):
        sig, wt = build_t_only(dc=dc)
        r = test(f"T-only_dc{dc}", sig, wt, f"确认{dc}天")
        results.append(r)

    # ── T-only + 冷却期组合 ──
    print("\n--- 最佳dc + dcd组合 ---")
    def build_t_with_dcd(dc=1, dcd=0):
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
        signal = confirmed_dir.map({'BULL': 'growth', 'BEAR': 'value'})
        wt = pd.Series(1.0, index=T.index)
        return signal, wt

    for dc in [3, 5, 7]:
        for dcd in [3, 5, 6, 10]:
            sig, wt = build_t_with_dcd(dc=dc, dcd=dcd)
            r = test(f"T-only_dc{dc}_dcd{dcd}", sig, wt, f"确认{dc}天 冷却{dcd}天")
            results.append(r)

    # ── 汇总 ──
    print("\n" + "=" * 80)
    print("  汇总")
    print("=" * 80)
    results.sort(key=lambda x: -x['calmar'])
    print(f"  {'名称':<24} {'年化':>8} {'回撤':>8} {'Sharpe':>8} {'Calmar':>8} "
          f"{'交易':>6} {'滑点Calmar':>12}")
    print("  " + "-" * 84)
    for r in results:
        print(f"  {r['name']:<24} {r['ann']*100:>7.2f}% {r['dd']*100:>7.2f}% "
              f"{r['sharpe']:>8.3f} {r['calmar']:>8.3f} {r['n_trades']:>6} "
              f"{r['calmar_sl']:>12.3f}")

    print(f"\n  纯裸最优: {max(results, key=lambda x: x['calmar'])['name']} "
          f"Calmar={max(results, key=lambda x: x['calmar'])['calmar']:.3f}")
    print("=" * 80)
