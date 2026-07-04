"""v6精细组合搜索：验证switch_threshold=0.0与其他参数的协同效应

A/B共识：
- switch_threshold=0.0 (Sharpe 2.169) 已打败基准
- 成本止损-10% (Sharpe 2.025) 接近打败基准
- 防御阈值-8% (Sharpe 1.991) 接近打败基准
- 目标：找到三者协同的最优组合
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
print(f"{'配置':<40} {'Sharpe':>7} {'年化':>7} {'回撤':>7} {'vol':>6} {'换手':>5} {'防御':>5} {'止损':>5} {'vs基准':>8}")
print("-" * 105)

# 组合1: switch=0.0 + 不同hold_count
print("\n=== 组合1: switch=0.0 + 不同hold_count ===")
for hc in [2, 3, 4, 5, 6]:
    eq, m, sig = strategy_unified_v3(close, ohlcv, hold_count=hc, switch_threshold=0.0)
    vol = (m['annual_return'] - 0.02) / m['sharpe'] if m['sharpe'] > 0 else 0
    diff = m['sharpe'] - TARGET
    mark = "✓" if diff > 0 else " "
    print(f"  hold={hc}+switch=0.0{' '*24} {m['sharpe']:>7.3f} {m['annual_return']*100:>6.2f}% {m['max_drawdown']*100:>6.2f}% {vol*100:>5.2f}% {m.get('n_switches',0):>5} {m.get('n_defense',0):>5} {m.get('n_stop_loss',0):>5} {diff:>+7.3f} {mark}")

# 组合2: switch=0.0 + 不同止损
print("\n=== 组合2: switch=0.0 + hold=4 + 不同止损 ===")
stop_configs = [
    ("无止损",            -0.99, -0.99),
    ("成本止损-8%",       -0.08, -0.99),
    ("成本止损-10%",      -0.10, -0.99),
    ("成本止损-12%",      -0.12, -0.99),
    ("跟踪止损-10%",      -0.99, -0.10),
    ("跟踪止损-12%",      -0.99, -0.12),
    ("成本-10%+跟踪-10%", -0.10, -0.10),
    ("成本-12%+跟踪-10%", -0.12, -0.10),
    ("成本-10%+跟踪-8%",  -0.10, -0.08),
]
for name, csl, tsl in stop_configs:
    use_sl = (csl > -0.90 or tsl > -0.90)
    eq, m, sig = strategy_unified_v3(close, ohlcv, hold_count=4, switch_threshold=0.0,
                                      use_stop_loss=use_sl,
                                      cost_stop_loss=csl, trailing_stop_loss=tsl)
    vol = (m['annual_return'] - 0.02) / m['sharpe'] if m['sharpe'] > 0 else 0
    diff = m['sharpe'] - TARGET
    mark = "✓" if diff > 0 else " "
    print(f"  {name:<36} {m['sharpe']:>7.3f} {m['annual_return']*100:>6.2f}% {m['max_drawdown']*100:>6.2f}% {vol*100:>5.2f}% {m.get('n_switches',0):>5} {m.get('n_defense',0):>5} {m.get('n_stop_loss',0):>5} {diff:>+7.3f} {mark}")

# 组合3: switch=0.0 + 不同防御阈值
print("\n=== 组合3: switch=0.0 + hold=4 + 不同防御阈值 ===")
for dd in [-0.05, -0.08, -0.10, -0.12, -0.15, -0.20, -0.99]:
    eq, m, sig = strategy_unified_v3(close, ohlcv, hold_count=4, switch_threshold=0.0,
                                      drawdown_threshold=dd)
    vol = (m['annual_return'] - 0.02) / m['sharpe'] if m['sharpe'] > 0 else 0
    diff = m['sharpe'] - TARGET
    mark = "✓" if diff > 0 else " "
    print(f"  防御阈值={dd:<28} {m['sharpe']:>7.3f} {m['annual_return']*100:>6.2f}% {m['max_drawdown']*100:>6.2f}% {vol*100:>5.2f}% {m.get('n_switches',0):>5} {m.get('n_defense',0):>5} {m.get('n_stop_loss',0):>5} {diff:>+7.3f} {mark}")

# 组合4: 三维联合搜索 (switch=0.0 + hold + 止损)
print("\n=== 组合4: 三维联合搜索 ===")
best_configs = []
for hc in [3, 4, 5]:
    for csl, tsl in [(-0.10, -0.10), (-0.12, -0.10), (-0.10, -0.08), (-0.99, -0.99)]:
        for dd in [-0.08, -0.10, -0.15]:
            use_sl = (csl > -0.90 or tsl > -0.90)
            eq, m, sig = strategy_unified_v3(close, ohlcv, hold_count=hc, switch_threshold=0.0,
                                              use_stop_loss=use_sl,
                                              cost_stop_loss=csl, trailing_stop_loss=tsl,
                                              drawdown_threshold=dd)
            vol = (m['annual_return'] - 0.02) / m['sharpe'] if m['sharpe'] > 0 else 0
            diff = m['sharpe'] - TARGET
            mark = "✓" if diff > 0 else " "
            name = f"h={hc},sl={csl},tl={tsl},dd={dd}"
            best_configs.append((name, m, vol, diff))
            print(f"  {name:<38} {m['sharpe']:>7.3f} {m['annual_return']*100:>6.2f}% {m['max_drawdown']*100:>6.2f}% {vol*100:>5.2f}% {m.get('n_switches',0):>5} {m.get('n_defense',0):>5} {m.get('n_stop_loss',0):>5} {diff:>+7.3f} {mark}")

# 排序找Top5
print("\n" + "=" * 105)
print("=== Top 5 最优配置 ===")
best_configs.sort(key=lambda x: x[1]['sharpe'], reverse=True)
for i, (name, m, vol, diff) in enumerate(best_configs[:5]):
    mark = "✓" if diff > 0 else " "
    print(f"  #{i+1} {name:<38} Sharpe={m['sharpe']:.3f} 年化={m['annual_return']*100:.2f}% 回撤={m['max_drawdown']*100:.2f}% vol={vol*100:.2f}% vs基准={diff:+.3f} {mark}")
