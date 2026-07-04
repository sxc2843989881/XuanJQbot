"""X13 去掉不合理参数，重新跑
================================================================
去掉:
  - max_hold_days / max_hold_reduce（完全不合逻辑的固定持仓天数）
  - bias_reduce=0.0 → 改为 0.05（保留超买降仓逻辑，但不要太极端）
  - st=0.088 → 改为 0.09（取整，避免精度过拟合）
保留: 其余所有层
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
SLOPE = MA20_SLOPE


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
            'calmar_sl': m_sl['calmar'], 'n_trades': m['n_trades'],
            'dir_sw': sw['dir'], 'cash_sw': sw['cash']}


def build_clean(slope_thresh=0.002, sw=0.17, st=0.09, cd=8,
                ms=10, ml=20, rt=1.3, dc=5, dcd=6,
                bias_ma=20, bias_high=0.19, bias_reduce=0.05):
    """X13 干净版：去掉 max_hold，st取整0.09，bias_reduce=0.05"""
    G_MA = G_CLOSE.rolling(bias_ma).mean()
    V_MA = V_CLOSE.rolling(bias_ma).mean()
    G_BIAS = (G_CLOSE / G_MA - 1)
    V_BIAS = (V_CLOSE / V_MA - 1)

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

    # 第3层：T+斜率双重确认 → 空仓
    weak_slope = SLOPE.abs() < slope_thresh
    weak_t = T.abs() < rt
    is_weak = weak_t & weak_slope
    wt = pd.Series(1.0, index=T.index)
    wt[is_weak] = 0.0

    # 第4层：B2价值动量过滤
    wrong_value = (dir_raw == 'BEAR') & (V_MOM_S <= 0) & (V_MOM_L <= 0)
    dir_raw[wrong_value] = 'BULL'
    dir_s = dir_raw.map({'BULL': 'growth', 'BEAR': 'value'})

    # 第5层：BIAS过滤（bias_reduce=0.05，不减到0）
    extreme_g = (dir_s == 'growth') & (G_BIAS > bias_high)
    extreme_v = (dir_s == 'value') & (V_BIAS > bias_high)
    extreme = extreme_g | extreme_v
    wt[extreme] = wt[extreme] * bias_reduce

    # 第6层：E5止损
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

    # ★ 已移除：第7层 持仓时间限制（max_hold_days / max_hold_reduce）

    signal = dir_raw.map({'BULL': 'growth', 'BEAR': 'value'})
    return signal, wt


if __name__ == '__main__':
    print("=" * 80)
    print("  X13 干净版 — 去掉不合理参数")
    print("  改动:")
    print("    - 去掉 max_hold_days / max_hold_reduce（完全不合逻辑）")
    print("    - st: 0.088 → 0.09（取整，避免精度过拟合）")
    print("    - bias_reduce: 0.0 → 0.05（保留降仓逻辑，不要太极端）")
    print("=" * 80)

    results = []

    # X13 原版
    print("\n--- ★ X13 原版 v26 U164 ---")
    from backtest_x13_engine import build_core as x13_build
    sig, wt = x13_build()
    r = test("X13_原版", sig, wt, "全机制")
    results.append(r)

    # X13 干净版
    print("\n--- ★ X13 干净版（去不合理参数） ---")
    sig, wt = build_clean()
    r = test("X13_干净版", sig, wt, "去max_hold + st=0.09 + Br=0.05")
    results.append(r)

    # 只去掉max_hold（其他参数不动）
    print("\n--- 只去掉max_hold ---")
    def build_no_mh(slope_thresh=0.002, sw=0.17, st=0.088, cd=8,
                     ms=10, ml=20, rt=1.3, dc=5, dcd=6,
                     bias_ma=20, bias_high=0.19, bias_reduce=0.0):
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
        extreme_g = (dir_s == 'growth') & (G_BIAS > bias_high)
        extreme_v = (dir_s == 'value') & (V_BIAS > bias_high)
        extreme = extreme_g | extreme_v
        wt[extreme] = wt[extreme] * bias_reduce
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

    sig, wt = build_no_mh()
    r = test("X13_去MH_only", sig, wt, "仅去掉max_hold")
    results.append(r)

    # 干净版 + 敏感性测试：bias_reduce从0到0.20
    print("\n--- 干净版 Br敏感性 ---")
    for br in [0.0, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20]:
        sig, wt = build_clean(bias_reduce=br)
        r = test(f"X13干净_Br{br}", sig, wt, f"bias_reduce={br}")
        results.append(r)

    # 干净版 + st敏感性
    print("\n--- 干净版 st敏感性 ---")
    for st in [0.07, 0.08, 0.09, 0.10, 0.12]:
        sig, wt = build_clean(st=st)
        r = test(f"X13干净_st{st}", sig, wt, f"st={st}")
        results.append(r)

    # 干净版 + 有无BIAS对比
    print("\n--- 干净版 BIAS有无对比 ---")
    sig, wt = build_clean(bias_reduce=1.0)  # 不降仓 = 无BIAS过滤
    r = test("X13干净_无BIAS", sig, wt, "bias_reduce=1.0，完全去掉BIAS")
    results.append(r)

    # ── 汇总 ──
    print("\n" + "=" * 80)
    print("  汇总 (按Calmar排序)")
    print("=" * 80)
    results.sort(key=lambda x: -x['calmar'])
    print(f"  {'名称':<22} {'年化':>8} {'回撤':>8} {'Sharpe':>8} {'Calmar':>8} "
          f"{'交易':>6} {'滑点Calmar':>12}")
    print("  " + "-" * 84)
    for r in results:
        print(f"  {r['name']:<22} {r['ann']*100:>7.2f}% {r['dd']*100:>7.2f}% "
              f"{r['sharpe']:>8.3f} {r['calmar']:>8.3f} {r['n_trades']:>6} "
              f"{r['calmar_sl']:>12.3f}")

    print("\n  ★ 跑完了!")
    print("=" * 80)
