"""快速重跑公平基准族（修复bug后）"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent / "v1_3etf"))

from data_generator_v2 import generate_simulation_data_v2
from run_validation import strategy_unified
from benchmark_family import run_all_benchmarks

print("生成5年模拟数据...")
close_df, ohlcv_dict, _ = generate_simulation_data_v2(n_years=5, seed=42)
print(f"  数据: {close_df.shape[0]}交易日, {close_df.shape[1]}ETF")

print("\n跑原始策略...")
eq, metrics, sig = strategy_unified(close_df, ohlcv_dict)
print(f"  原始: Sharpe={metrics['sharpe']:.3f}, 年化={metrics['annual_return']*100:.2f}%")

print("\n跑公平基准族（修复bug后，蒙特卡洛100次）...")
result = run_all_benchmarks(close_df, metrics, n_mc_runs=100)

print("\n=== 公平基准族结果（修复后）===")
print(f"策略Sharpe:       {result['strategy']['sharpe']:.3f}")
print(f"等权全池Sharpe:   {result['equal_weight']['sharpe']:.3f}")
print(f"动量Top2买入持有: {result['momentum_buyhold']['sharpe']:.3f}")
mc = result['monte_carlo']
print(f"随机Top2均值:     {mc['random_sharpe_mean']:.3f}")
print(f"随机Top2 P95:     {mc['random_sharpe_p95']:.3f}")
print(f"策略分位:         {mc['percentile']:.1f}%")
print(f"Z-score:          {mc['z_score']:.2f}")
print(f"alpha显著:        {mc['alpha_significant']}")

print("\n=== 判定 ===")
for check in result['verdict']['checks']:
    print(f"  {check}")
print(f"\n总结: {result['verdict']['verdict_summary']}")
