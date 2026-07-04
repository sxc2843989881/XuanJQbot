"""X14 v2.0 滑点敏感性扫描
====================================================
测试不同滑点水平对策略性能的影响。
档位: 0/2/5/10/20 bps (每边)
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

print("=" * 80)
print("  X14 v2.0 滑点敏感性扫描")
print("=" * 80)

sig, wt = build_core()
sw = count_switches(sig, wt)

# 滑点档位 (每边总滑点bps)
slippage_bps_list = [0, 2, 5, 10, 20]

print(f"\n  {'滑点(bps)':<15} {'年化':>10} {'回撤':>10} {'Sharpe':>8} {'Calmar':>8} {'交易':>8}")
print("  " + "-" * 65)

results = []
for sp_bps in slippage_bps_list:
    impact = sp_bps / 10000  # bps to decimal
    result = run_backtest(sig, wt, impact_slippage=impact)
    m = calc_metrics(result)
    results.append({
        'slippage_bps': sp_bps,
        'ann': m['ann'],
        'dd': m['dd'],
        'sharpe': m['sharpe'],
        'calmar': m['calmar'],
        'n_trades': m['n_trades'],
    })
    print(f"  {sp_bps:<15} {m['ann']*100:>9.2f}% {m['dd']*100:>9.2f}% "
          f"{m['sharpe']:>7.3f} {m['calmar']:>7.3f} {m['n_trades']:>8}")

# 汇总分析
print("\n" + "=" * 80)
print("  汇总分析")
print("=" * 80)

baseline = results[0]  # 0 bps
print(f"\n  基准(0bps): 年化={baseline['ann']*100:.2f}%  Calmar={baseline['calmar']:.3f}")
print()

for r in results[1:]:
    calmar_decay = (r['calmar'] / baseline['calmar'] - 1) * 100
    ann_decay = (r['ann'] / baseline['ann'] - 1) * 100
    print(f"  {r['slippage_bps']}bps: 年化={r['ann']*100:.2f}%({ann_decay:+.1f}%)  "
          f"Calmar={r['calmar']:.3f}({calmar_decay:+.1f}%)")

# 评估临界点
print(f"\n  滑点敏感性评估:")
for i in range(1, len(results)):
    calmar_drop = (results[i]['calmar'] / results[0]['calmar'] - 1) * 100
    if calmar_drop < -10:
        print(f"    ⚠ {results[i]['slippage_bps']}bps: Calmar下降{calmar_drop:.0f}% > 10% → 敏感!")
    elif calmar_drop < -5:
        print(f"    ~ {results[i]['slippage_bps']}bps: Calmar下降{calmar_drop:.0f}% (5-10%) → 中度敏感")
    else:
        print(f"    ✓ {results[i]['slippage_bps']}bps: Calmar下降{calmar_drop:.0f}% (<5%) → 不敏感")

print(f"\n  交易信号统计:")
print(f"    方向切换: {sw['dir']}次")
print(f"    空仓切换: {sw['cash']}次")
print(f"    总信号: {sw['dir'] + sw['cash']}次")
print(f"    每边滑点成本 = {sw['dir'] + sw['cash']}次 × 滑点bps")

print("\n" + "=" * 80)
print("  滑点扫描完成")
print("=" * 80)
