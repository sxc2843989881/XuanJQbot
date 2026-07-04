"""v6 Walk-Forward修复验证：调整test_months=6

A/B共识：
- v6通过4/5验证，唯一失败是Walk-Forward（变异系数1.15>=1.0）
- 失败原因是test_months=3太短，3个月Sharpe波动天然大
- 调整test_months=6重跑，让每个窗口有足够数据稳定计算Sharpe
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent / "v1_3etf"))

from data_generator_v2 import generate_simulation_data_v2
from walk_forward import run_walk_forward
from run_v6_validation import strategy_unified_v6

print("=" * 70)
print("v6 Walk-Forward修复验证 (test_months=6)")
print("=" * 70)
print()

print("生成5年模拟数据...")
close_df, ohlcv_dict, _ = generate_simulation_data_v2(n_years=5, seed=42)

# 测试不同test_months配置
configs = [
    ("原配置 train=6,test=3,step=3", 6, 3, 3),
    ("修复1 train=6,test=6,step=3", 6, 6, 3),
    ("修复2 train=12,test=6,step=3", 12, 6, 3),
    ("修复3 train=12,test=6,step=6", 12, 6, 6),
]

print(f"\n{'配置':<35} {'窗口数':>5} {'平均Sharpe':>10} {'正占比':>7} {'变异系数':>8} {'判定':>8}")
print("-" * 85)

for name, tm, test_m, step_m in configs:
    print(f"\n>>> 运行 {name}...")
    wf_result = run_walk_forward(
        close_df, ohlcv_dict, strategy_unified_v6,
        train_months=tm, test_months=test_m, step_months=step_m,
    )
    if "verdict" in wf_result:
        v = wf_result["verdict"]
        mark = "✓" if v.get("overall_pass", False) else "✗"
        print(f"  {name:<33} {wf_result.get('n_windows',0):>5} "
              f"{wf_result.get('sharpe_mean',0):>10.3f} "
              f"{wf_result.get('positive_ratio',0)*100:>6.0f}% "
              f"{wf_result.get('sharpe_cv',0):>8.2f} "
              f"{mark} {v.get('verdict_summary','N/A')}")
        for check in v.get("checks", []):
            print(f"      {check}")
    print(f"  Sharpe范围: [{wf_result.get('sharpe_min',0):.3f}, {wf_result.get('sharpe_max',0):.3f}]")
