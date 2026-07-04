"""X13 去掉所有防回撤机制，裸跑
================================================================
去掉的层:
  - 第3层 T+斜率双重确认 (never空仓)
  - 第5层 BIAS过滤 (Br)
  - 第6层 E5止损 (st/sw/cd)
  - 第7层 持仓时间限制 (MH)

保留:
  - 第1层 方向确认 (dc)
  - 第2层 方向冷却 (dcd)
  - 第4层 B2价值动量过滤 (ms/ml)
================================================================
"""
import sys
sys.path.insert(0, r'c:\XuanJLH\Qbot\custom\backtests')
sys.path.insert(0, r'c:\XuanJLH\Qbot\custom\research\成长价值轮动\X12_偏离度空仓版')

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


def build_core_naked(dc=5, dcd=6, ms=10, ml=20):
    """X13 裸版：只有方向判断，没有任何防回撤"""
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

    # 第4层：B2价值动量过滤
    wrong_value = (dir_raw == 'BEAR') & (V_MOM_S <= 0) & (V_MOM_L <= 0)
    dir_raw[wrong_value] = 'BULL'

    # ★ 永远满仓（没有空仓、没有止损、没有BIAS过滤、没有持仓限制）
    wt = pd.Series(1.0, index=T.index)

    signal = dir_raw.map({'BULL': 'growth', 'BEAR': 'value'})
    return signal, wt


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


def build_core_full(**kw):
    """X13完整版（引用X13目录下的策略代码）"""
    from backtest_x13_engine import build_core as x13_build
    return x13_build(**kw)


if __name__ == '__main__':
    print("=" * 80)
    print("  X13 裸跑 — 去掉所有防回撤机制")
    print("=" * 80)

    results = []

    # ── X13 完整版（基准对比） ──
    print("\n--- 【对比基准】X13 完整版 v26 U164 ---")
    sys.path.insert(0, r'c:\XuanJLH\Qbot\custom\research\成长价值轮动\X13_统一策略最优版')
    from backtest_x13_engine import build_core as x13_build
    sig, wt = x13_build()
    r = test("X13完整版", sig, wt, "全机制")
    results.append(r)

    # ── X13 裸版 ──
    print("\n--- 【裸跑】X13 裸版(无空仓/无止损/无BIAS/无持仓限制) ---")
    sig, wt = build_core_naked()
    r = test("X13_裸版", sig, wt, "只有方向判断，永远满仓")
    results.append(r)

    # ── 裸版 + dc/dcd敏感性 ──
    print("\n--- 裸版方向参数敏感性 ---")
    for dc, dcd in [(5,6), (3,6), (7,6), (5,3), (5,10), (3,3), (7,10)]:
        sig, wt = build_core_naked(dc=dc, dcd=dcd)
        r = test(f"X13裸版_dc{dc}_dcd{dcd}", sig, wt,
                  f"dc={dc} dcd={dcd}")
        results.append(r)

    # ── 裸版 + B2有无对比 ──
    print("\n--- 裸版 B2有无对比 ---")
    # 无B2的裸版
    def naked_no_b2(dc=5, dcd=6):
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

    sig, wt = naked_no_b2()
    r = test("X13裸版_无B2", sig, wt, "无B2")
    results.append(r)

    # 裸版 + B2单周期(ms=10, ml=0 等价于只用短周期)
    def naked_b2_single(ms=10, dc=5, dcd=6):
        V_MOM_S = V_CLOSE.pct_change(ms)
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
        wrong = (dir_raw == 'BEAR') & (V_MOM_S <= 0)
        dir_raw[wrong] = 'BULL'
        signal = dir_raw.map({'BULL': 'growth', 'BEAR': 'value'})
        wt = pd.Series(1.0, index=T.index)
        return signal, wt

    sig, wt = naked_b2_single()
    r = test("X13裸版_B2单周期", sig, wt, "只用V_MOM10")
    results.append(r)

    # ── X11-A（历史基准） ──
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

    print("\n  ★ 裸跑完成!")
    print("=" * 80)
