"""v6动态候选池版参数搜索：20只ETF + 13年回测

A/B共识：
- v6的5年优化参数(hold=4,switch=0.0,无止损,防御-8%)在13年+20只ETF场景下不适配
- 需要重新搜索最优参数
- 关键搜索维度：hold_count, switch_threshold, drawdown_threshold
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent / "v1_3etf"))

from data_generator_v3 import generate_simulation_data_v3, get_dynamic_universe
from factors_v6_dynamic import strategy_unified_v6_dynamic
from backtest import calc_metrics

print("生成13年模拟数据（20只ETF，动态上市）...")
close_df, ohlcv_dict, _, list_dates = generate_simulation_data_v3(n_years=13, seed=42)

# 动态等权基准
universe = get_dynamic_universe(close_df, list_dates)
daily_ret = close_df.pct_change().fillna(0)
n_available = universe.sum(axis=1)
bench_weights = universe.div(n_available, axis=0).fillna(0)
bench_ret = (daily_ret * bench_weights).sum(axis=1)
bench_metrics = calc_metrics(bench_ret)
TARGET = bench_metrics['sharpe']
print(f"动态等权基准: Sharpe={TARGET:.3f}, 年化={bench_metrics['annual_return']*100:.2f}%, 回撤={bench_metrics['max_drawdown']*100:.2f}%")
print()

print(f"{'配置':<40} {'Sharpe':>7} {'年化':>7} {'回撤':>7} {'换手':>5} {'防御':>5} {'vs基准':>8}")
print("-" * 90)

# 维度1: hold_count 搜索
print("\n=== 维度1: hold_count 搜索 ===")
for hc in [2, 3, 4, 5, 6, 8, 10]:
    eq, m, sig = strategy_unified_v6_dynamic(close_df, ohlcv_dict, list_dates, hold_count=hc)
    diff = m['sharpe'] - TARGET
    mark = "✓" if diff > 0 else " "
    print(f"  hold_count={hc:<28} {m['sharpe']:>7.3f} {m['annual_return']*100:>6.2f}% {m['max_drawdown']*100:>6.2f}% {m.get('n_switches',0):>5} {m.get('n_defense',0):>5} {diff:>+7.3f} {mark}")

# 维度2: switch_threshold 搜索
print("\n=== 维度2: switch_threshold 搜索 (hold=6) ===")
for st in [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]:
    eq, m, sig = strategy_unified_v6_dynamic(close_df, ohlcv_dict, list_dates, hold_count=6, switch_threshold=st)
    diff = m['sharpe'] - TARGET
    mark = "✓" if diff > 0 else " "
    print(f"  switch={st:<30} {m['sharpe']:>7.3f} {m['annual_return']*100:>6.2f}% {m['max_drawdown']*100:>6.2f}% {m.get('n_switches',0):>5} {m.get('n_defense',0):>5} {diff:>+7.3f} {mark}")

# 维度3: drawdown_threshold 搜索
print("\n=== 维度3: drawdown_threshold 搜索 (hold=6, switch=0.0) ===")
for dd in [-0.05, -0.08, -0.10, -0.12, -0.15, -0.20, -0.99]:
    use_def = (dd > -0.90)
    eq, m, sig = strategy_unified_v6_dynamic(close_df, ohlcv_dict, list_dates,
                                              hold_count=6, switch_threshold=0.0,
                                              drawdown_threshold=dd, use_defense=use_def)
    diff = m['sharpe'] - TARGET
    mark = "✓" if diff > 0 else " "
    print(f"  drawdown={dd:<28} {m['sharpe']:>7.3f} {m['annual_return']*100:>6.2f}% {m['max_drawdown']*100:>6.2f}% {m.get('n_switches',0):>5} {m.get('n_defense',0):>5} {diff:>+7.3f} {mark}")

# 维度4: 无防御 vs 有防御
print("\n=== 维度4: 无防御 vs 有防御 (hold=6, switch=0.0) ===")
for use_def, dd, name in [(False, -0.99, "无防御"), (True, -0.05, "防御-5%"), (True, -0.08, "防御-8%"),
                           (True, -0.10, "防御-10%"), (True, -0.15, "防御-15%")]:
    eq, m, sig = strategy_unified_v6_dynamic(close_df, ohlcv_dict, list_dates,
                                              hold_count=6, switch_threshold=0.0,
                                              use_defense=use_def, drawdown_threshold=dd)
    diff = m['sharpe'] - TARGET
    mark = "✓" if diff > 0 else " "
    print(f"  {name:<32} {m['sharpe']:>7.3f} {m['annual_return']*100:>6.2f}% {m['max_drawdown']*100:>6.2f}% {m.get('n_switches',0):>5} {m.get('n_defense',0):>5} {diff:>+7.3f} {mark}")

# 维度5: Top组合
print("\n=== 维度5: Top组合搜索 ===")
combos = [
    ("hold=6+switch=0+无防御",       6, 0.0, False, -0.99),
    ("hold=8+switch=0+无防御",       8, 0.0, False, -0.99),
    ("hold=6+switch=0.1+无防御",     6, 0.10, False, -0.99),
    ("hold=6+switch=0+防御-5%",      6, 0.0, True, -0.05),
    ("hold=8+switch=0+防御-5%",      8, 0.0, True, -0.05),
    ("hold=6+switch=0.1+防御-5%",    6, 0.10, True, -0.05),
    ("hold=8+switch=0.1+防御-8%",    8, 0.10, True, -0.08),
    ("hold=10+switch=0+无防御",      10, 0.0, False, -0.99),
]
best = []
for name, hc, st, ud, dd in combos:
    eq, m, sig = strategy_unified_v6_dynamic(close_df, ohlcv_dict, list_dates,
                                              hold_count=hc, switch_threshold=st,
                                              use_defense=ud, drawdown_threshold=dd)
    diff = m['sharpe'] - TARGET
    mark = "✓" if diff > 0 else " "
    print(f"  {name:<34} {m['sharpe']:>7.3f} {m['annual_return']*100:>6.2f}% {m['max_drawdown']*100:>6.2f}% {m.get('n_switches',0):>5} {m.get('n_defense',0):>5} {diff:>+7.3f} {mark}")
    best.append((name, m, diff))

print("\n=== Top 3 ===")
best.sort(key=lambda x: x[1]['sharpe'], reverse=True)
for i, (name, m, diff) in enumerate(best[:3]):
    print(f"  #{i+1} {name}: Sharpe={m['sharpe']:.3f} 年化={m['annual_return']*100:.2f}% 回撤={m['max_drawdown']*100:.2f}% vs基准={diff:+.3f}")
