"""快速测试v4策略：上涨区间识别+目标收益止盈"""
import sys
from pathlib import Path

HERE = Path(r'c:\XuanJLH\Qbot\custom\strategies\etf_rotation\1_research\v2_10etf')
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / 'validation'))
sys.path.insert(0, str(HERE.parent / 'v1_3etf'))

from data_generator_v2 import generate_simulation_data_v2
from factors_v4 import strategy_unified_v4
from factors_v3 import strategy_unified_v3
from benchmark_family import calc_metrics, equal_weight_benchmark
from run_validation import strategy_unified

print("=" * 70)
print("v4策略测试：上涨区间识别 + 目标收益止盈")
print("=" * 70)

print("\n生成5年模拟数据...")
close, ohlcv, _ = generate_simulation_data_v2(n_years=5, seed=42)
print(f"  数据: {close.shape[0]}交易日, {close.shape[1]}ETF")

print("\n[v2] 原版策略（2只持仓+月频）...")
eq2, m2, _ = strategy_unified(close, ohlcv)
print(f"  Sharpe={m2['sharpe']:.3f}, 年化={m2['annual_return']*100:.2f}%, 回撤={m2['max_drawdown']*100:.2f}%")

print("\n[v3] 优化版（4只+防御+止损+阈值）...")
eq3, m3, _ = strategy_unified_v3(close, ohlcv)
print(f"  Sharpe={m3['sharpe']:.3f}, 年化={m3['annual_return']*100:.2f}%, 回撤={m3['max_drawdown']*100:.2f}%")

print("\n[v4] 上涨区间+止盈（4只+日频扫描+10%止盈）...")
eq4, m4, sig4 = strategy_unified_v4(close, ohlcv)
print(f"  Sharpe={m4['sharpe']:.3f}, 年化={m4['annual_return']*100:.2f}%, 回撤={m4['max_drawdown']*100:.2f}%")
print(f"  换手={m4.get('n_switches','?')}次, 止损={m4.get('n_stop_loss','?')}次, "
      f"止盈={m4.get('n_take_profit','?')}次, 动量衰减={m4.get('n_momentum_decay','?')}次, "
      f"防御={m4.get('n_defense','?')}次")

print("\n[基准] 等权10只ETF买入持有...")
_, eq_m = equal_weight_benchmark(close)
print(f"  Sharpe={eq_m['sharpe']:.3f}, 年化={eq_m['annual_return']*100:.2f}%, 回撤={eq_m['max_drawdown']*100:.2f}%")

print("\n" + "=" * 70)
print("版本对比")
print("=" * 70)
print(f"{'版本':<20} {'Sharpe':>8} {'年化':>8} {'回撤':>8} {'vs基准':>10}")
print("-" * 60)
print(f"{'v2 原版':<20} {m2['sharpe']:>8.3f} {m2['annual_return']*100:>7.2f}% {m2['max_drawdown']*100:>7.2f}% {m2['sharpe']-eq_m['sharpe']:>+9.3f}")
print(f"{'v3 优化版':<20} {m3['sharpe']:>8.3f} {m3['annual_return']*100:>7.2f}% {m3['max_drawdown']*100:>7.2f}% {m3['sharpe']-eq_m['sharpe']:>+9.3f}")
print(f"{'v4 上涨区间+止盈':<20} {m4['sharpe']:>8.3f} {m4['annual_return']*100:>7.2f}% {m4['max_drawdown']*100:>7.2f}% {m4['sharpe']-eq_m['sharpe']:>+9.3f}")
print(f"{'等权基准':<20} {eq_m['sharpe']:>8.3f} {eq_m['annual_return']*100:>7.2f}% {eq_m['max_drawdown']*100:>7.2f}% {0.0:>+9.3f}")

beat = m4['sharpe'] > eq_m['sharpe']
print(f"\nv4 vs 基准: {'✓ 打败基准！' if beat else '✗ 仍跑输基准'}")
