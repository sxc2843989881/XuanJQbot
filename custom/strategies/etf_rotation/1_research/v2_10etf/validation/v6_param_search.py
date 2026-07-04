"""v3参数搜索：寻找打败等权基准的最优配置

A/B共识方向：
- v3框架(月频+4只+10%阈值+防御+止损)已最优，v4.x日频思路被证伪
- v3离打败基准只差vol降1个百分点(15.04%→14.04%)
- 搜索维度：hold_count, switch_threshold, drawdown_threshold, cost_stop_loss
"""
import sys
from pathlib import Path

HERE = Path(r'c:\XuanJLH\Qbot\custom\strategies\etf_rotation\1_research\v2_10etf')
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / 'validation'))
sys.path.insert(0, str(HERE.parent / 'v1_3etf'))

from data_generator_v2 import generate_simulation_data_v2
from factors_v3 import strategy_unified_v3
from benchmark_family import equal_weight_benchmark

print("生成5年模拟数据...")
close, ohlcv, _ = generate_simulation_data_v2(n_years=5, seed=42)

_, eq_m = equal_weight_benchmark(close)
TARGET = eq_m['sharpe']
print(f"等权基准: Sharpe={TARGET:.3f}, 年化={eq_m['annual_return']*100:.2f}%, 回撤={eq_m['max_drawdown']*100:.2f}%")
print()
print(f"{'配置':<30} {'Sharpe':>7} {'年化':>7} {'回撤':>7} {'vol':>6} {'换手':>5} {'防御':>5} {'止损':>5} {'vs基准':>8}")
print("-" * 95)

# 维度1: hold_count 搜索
print("\n=== 维度1: hold_count 搜索 (其他默认) ===")
for hc in [2, 3, 4, 5, 6, 8, 10]:
    eq, m, sig = strategy_unified_v3(close, ohlcv, hold_count=hc)
    vol = (m['annual_return'] - 0.02) / m['sharpe'] if m['sharpe'] > 0 else 0
    diff = m['sharpe'] - TARGET
    mark = "✓" if diff > 0 else " "
    print(f"  hold_count={hc:<23} {m['sharpe']:>7.3f} {m['annual_return']*100:>6.2f}% {m['max_drawdown']*100:>6.2f}% {vol*100:>5.2f}% {m.get('n_switches',0):>5} {m.get('n_defense',0):>5} {m.get('n_stop_loss',0):>5} {diff:>+7.3f} {mark}")

# 维度2: switch_threshold 搜索
print("\n=== 维度2: switch_threshold 搜索 (hold_count=4) ===")
for st in [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]:
    eq, m, sig = strategy_unified_v3(close, ohlcv, hold_count=4, switch_threshold=st)
    vol = (m['annual_return'] - 0.02) / m['sharpe'] if m['sharpe'] > 0 else 0
    diff = m['sharpe'] - TARGET
    mark = "✓" if diff > 0 else " "
    print(f"  switch_threshold={st:<18} {m['sharpe']:>7.3f} {m['annual_return']*100:>6.2f}% {m['max_drawdown']*100:>6.2f}% {vol*100:>5.2f}% {m.get('n_switches',0):>5} {m.get('n_defense',0):>5} {m.get('n_stop_loss',0):>5} {diff:>+7.3f} {mark}")

# 维度3: drawdown_threshold 搜索
print("\n=== 维度3: drawdown_threshold 搜索 (hold_count=4) ===")
for dd in [-0.05, -0.08, -0.10, -0.12, -0.15, -0.20, -0.99]:
    eq, m, sig = strategy_unified_v3(close, ohlcv, hold_count=4, drawdown_threshold=dd)
    vol = (m['annual_return'] - 0.02) / m['sharpe'] if m['sharpe'] > 0 else 0
    diff = m['sharpe'] - TARGET
    mark = "✓" if diff > 0 else " "
    print(f"  drawdown_threshold={dd:<14} {m['sharpe']:>7.3f} {m['annual_return']*100:>6.2f}% {m['max_drawdown']*100:>6.2f}% {vol*100:>5.2f}% {m.get('n_switches',0):>5} {m.get('n_defense',0):>5} {m.get('n_stop_loss',0):>5} {diff:>+7.3f} {mark}")

# 维度4: 止损参数搜索
print("\n=== 维度4: 止损参数搜索 (hold_count=4) ===")
stop_configs = [
    ("无止损",            0, 0),
    ("成本止损-8%",       -0.08, 0),
    ("成本止损-10%",      -0.10, 0),
    ("成本止损-12%(默认)", -0.12, 0),
    ("成本止损-15%",      -0.15, 0),
    ("跟踪止损-8%",       0, -0.08),
    ("跟踪止损-10%(默认)", 0, -0.10),
    ("跟踪止损-15%",      0, -0.15),
    ("12%+10%(默认)",     -0.12, -0.10),
    ("10%+8%",           -0.10, -0.08),
    ("15%+12%",          -0.15, -0.12),
]
for name, csl, tsl in stop_configs:
    use_sl = (csl != 0 or tsl != 0)
    eq, m, sig = strategy_unified_v3(close, ohlcv, hold_count=4, 
                                      use_stop_loss=use_sl,
                                      cost_stop_loss=csl if csl != 0 else -0.99,
                                      trailing_stop_loss=tsl if tsl != 0 else -0.99)
    vol = (m['annual_return'] - 0.02) / m['sharpe'] if m['sharpe'] > 0 else 0
    diff = m['sharpe'] - TARGET
    mark = "✓" if diff > 0 else " "
    print(f"  {name:<26} {m['sharpe']:>7.3f} {m['annual_return']*100:>6.2f}% {m['max_drawdown']*100:>6.2f}% {vol*100:>5.2f}% {m.get('n_switches',0):>5} {m.get('n_defense',0):>5} {m.get('n_stop_loss',0):>5} {diff:>+7.3f} {mark}")

# 维度5: 组合搜索
print("\n=== 维度5: 组合搜索 (Top候选组合) ===")
combos = [
    ("hold=5+阈值0.15+防御10%", 5, 0.15, -0.10, -0.12, -0.10),
    ("hold=6+阈值0.15+防御10%", 6, 0.15, -0.10, -0.12, -0.10),
    ("hold=5+阈值0.20+防御12%", 5, 0.20, -0.12, -0.12, -0.10),
    ("hold=6+无止损+防御10%",   6, 0.15, -0.10, -0.99, -0.99),
    ("hold=8+阈值0.20+防御15%", 8, 0.20, -0.15, -0.12, -0.10),
    ("hold=4+阈值0.15+防御10%+紧止损", 4, 0.15, -0.10, -0.10, -0.08),
    ("hold=5+阈值0.10+防御8%",  5, 0.10, -0.08, -0.12, -0.10),
]
for name, hc, st, dd, csl, tsl in combos:
    use_sl = (csl > -0.90)
    eq, m, sig = strategy_unified_v3(close, ohlcv, hold_count=hc, switch_threshold=st,
                                      drawdown_threshold=dd, 
                                      use_stop_loss=use_sl,
                                      cost_stop_loss=csl, trailing_stop_loss=tsl)
    vol = (m['annual_return'] - 0.02) / m['sharpe'] if m['sharpe'] > 0 else 0
    diff = m['sharpe'] - TARGET
    mark = "✓" if diff > 0 else " "
    print(f"  {name:<32} {m['sharpe']:>7.3f} {m['annual_return']*100:>6.2f}% {m['max_drawdown']*100:>6.2f}% {vol*100:>5.2f}% {m.get('n_switches',0):>5} {m.get('n_defense',0):>5} {m.get('n_stop_loss',0):>5} {diff:>+7.3f} {mark}")
