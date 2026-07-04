"""快速测试优化版v3策略"""
import sys
from pathlib import Path

HERE = Path(r'c:\XuanJLH\Qbot\custom\strategies\etf_rotation\1_research\v2_10etf')
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / 'validation'))
sys.path.insert(0, str(HERE.parent / 'v1_3etf'))

from data_generator_v2 import generate_simulation_data_v2
from factors_v3 import strategy_unified_v3
from benchmark_family import calc_metrics, equal_weight_benchmark

print("生成5年模拟数据...")
close, ohlcv, _ = generate_simulation_data_v2(n_years=5, seed=42)
print(f"  数据: {close.shape[0]}交易日, {close.shape[1]}ETF")

print("\n跑优化版v3策略（4只持仓+防御+止损+阈值）...")
eq, m, sig = strategy_unified_v3(close, ohlcv)
print(f"  v3策略: Sharpe={m['sharpe']:.3f}, 年化={m['annual_return']*100:.2f}%, 回撤={m['max_drawdown']*100:.2f}%")
print(f"  换手={m.get('n_switches','?')}次, 防御触发={m.get('n_defense','?')}次, 止损={m.get('n_stop_loss','?')}次")

print("\n等权10只ETF基准...")
_, eq_m = equal_weight_benchmark(close)
print(f"  等权基准: Sharpe={eq_m['sharpe']:.3f}, 年化={eq_m['annual_return']*100:.2f}%, 回撤={eq_m['max_drawdown']*100:.2f}%")

result = "✓ 打败" if m['sharpe'] > eq_m['sharpe'] else "✗ 跑输"
print(f"\n对比: v3 Sharpe {m['sharpe']:.3f} vs 基准 {eq_m['sharpe']:.3f} → {result}基准")

# v2对比
print("\n跑原版v2策略对比...")
from run_validation import strategy_unified
eq2, m2, sig2 = strategy_unified(close, ohlcv)
print(f"  v2策略: Sharpe={m2['sharpe']:.3f}, 年化={m2['annual_return']*100:.2f}%, 回撤={m2['max_drawdown']*100:.2f}%")

print(f"\n=== 优化效果 ===")
print(f"  Sharpe: v2 {m2['sharpe']:.3f} → v3 {m['sharpe']:.3f} ({(m['sharpe']-m2['sharpe'])/abs(m2['sharpe'])*100:+.1f}%)")
print(f"  回撤:   v2 {m2['max_drawdown']*100:.2f}% → v3 {m['max_drawdown']*100:.2f}%")
