"""X14 v2.0 逐项消融测试 (修正版)
====================================================
用真实v1.0基线(45.27%/Calmar 1.951)对比各改进项的独立贡献。
====================================================
"""
import sys
sys.path.insert(0, r'c:\XuanJLH\Qbot\custom\backtests')
sys.path.insert(0, r'c:\XuanJLH\Qbot\custom\research\成长价值轮动\X12_偏离度空仓版')
sys.path.insert(0, r'c:\XuanJLH\Qbot\custom\research\成长价值轮动\X14_统一策略干净版')

import numpy as np
import pandas as pd
from optimize_runner import run_backtest, calc_metrics, count_switches
from backtest_x14_engine import build_core

# v1.0真实基线
V1 = dict(dcd=6, ms=10, ml=20, bias_mode='clear', dual_momentum=False,
          bias_t_constraint=False, sw_mid=0.17, sw_deep=0.17,
          rapid_decline=False, e5_reset=True)

def test(name, overrides):
    kwargs = {**V1, **overrides}
    sig, wt = build_core(**kwargs)
    result = run_backtest(sig, wt, impact_slippage=0.0005)
    m = calc_metrics(result)
    sw = count_switches(sig, wt)
    print(f"  {name:<48} {m['ann']*100:>7.2f}%  {m['dd']*100:>7.2f}%  "
          f"{m['sharpe']:>6.3f}  {m['calmar']:>6.3f}  {m['n_trades']:>5}")
    return m

print("=" * 105)
print("  X14 v2.0 逐项消融测试 (基线=v1.0真实原版)")
print("=" * 105)
print(f"\n  {'配置':<48} {'年化':>8} {'回撤':>8} {'Sharpe':>7} {'Calmar':>7} {'交易':>6}")
print("  " + "-" * 105)

# 0. 真实基线
m0 = test("0) v1.0真实原版 (45.27%/1.951)", {})

# 逐项测试
tests = [
    ("1) dcd=4", dict(dcd=4)),
    ("2) E5冷却期修复(延长3天)", dict(e5_reset=False)),
    ("3) 急跌加速判断(3日>7%)", dict(rapid_decline=True)),
    ("4) Dual Momentum", dict(dual_momentum=True)),
    ("5) B2 20/60", dict(ms=20, ml=60)),
    ("6) BIAS降仓50% (无T约束)", dict(bias_mode='half', bias_t_constraint=False)),
    ("7) BIAS降仓50%+T约束", dict(bias_mode='half', bias_t_constraint=True)),
    ("8) 权重漏斗(sw_mid=0.50)", dict(sw_mid=0.50)),
]

results = [m0]
for name, overrides in tests:
    m = test(name, overrides)
    results.append(m)

# 9. 全部v2.0
sig_all, wt_all = build_core()
r_all = run_backtest(sig_all, wt_all, impact_slippage=0.0005)
m_all = calc_metrics(r_all)
sw_all = count_switches(sig_all, wt_all)
print(f"  9) 全部v2.0默认                           {m_all['ann']*100:>7.2f}%  {m_all['dd']*100:>7.2f}%  "
      f"{m_all['sharpe']:>6.3f}  {m_all['calmar']:>6.3f}  {m_all['n_trades']:>5}")
results.append(m_all)

# 汇总
print("\n" + "=" * 105)
print("  汇总 (vs 真实v1.0基线)")
print("=" * 105)
print(f"\n  {'改进项':<48} {'年化':>8} {'回撤':>8} {'Sharpe':>8} {'Calmar':>8} {'Calmar变化':>12}")
print("  " + "-" * 92)
all_tests = [
    ("0) v1.0真实基线", m0),
    ("1) dcd=4", results[1]),
    ("2) E5冷却期修复", results[2]),
    ("3) 急跌加速判断", results[3]),
    ("4) Dual Momentum", results[4]),
    ("5) B2 20/60", results[5]),
    ("6) BIAS降仓50%(无T)", results[6]),
    ("7) BIAS降仓50%+T约束", results[7]),
    ("8) 权重漏斗两档", results[8]),
    ("9) 全部v2.0默认", results[9]),
]
for name, m in all_tests:
    delta = (m['calmar'] / m0['calmar'] - 1) * 100
    print(f"  {name:<48} {m['ann']*100:>7.2f}% {m['dd']*100:>7.2f}% "
          f"{m['sharpe']:>7.3f} {m['calmar']:>7.3f} {delta:>+10.1f}%")
