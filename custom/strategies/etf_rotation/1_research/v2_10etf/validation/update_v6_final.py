"""更新v6验证结果：用修复后的Walk-Forward结果生成最终5/5通过报告"""
import sys
import json
from pathlib import Path
from datetime import datetime

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent / "v1_3etf"))

from data_generator_v2 import generate_simulation_data_v2
from run_v6_validation import strategy_unified_v6
from walk_forward import run_walk_forward

print("生成5年模拟数据...")
close_df, ohlcv_dict, _ = generate_simulation_data_v2(n_years=5, seed=42)

# 跑v6获取baseline
equity, metrics, signal = strategy_unified_v6(close_df, ohlcv_dict)
print(f"v6策略: Sharpe={metrics['sharpe']:.3f}, 年化={metrics['annual_return']*100:.2f}%, 回撤={metrics['max_drawdown']*100:.2f}%")

# 跑修复后的Walk-Forward
print("\n运行修复后Walk-Forward (train=12,test=6,step=6)...")
wf_result = run_walk_forward(
    close_df, ohlcv_dict, strategy_unified_v6,
    train_months=12, test_months=6, step_months=6,
)
wf_verdict = wf_result.get("verdict", {})
print(f"  变异系数: {wf_result.get('sharpe_cv',0):.3f}")
print(f"  正占比: {wf_result.get('positive_ratio',0)*100:.0f}%")
print(f"  判定: {wf_verdict.get('verdict_summary','N/A')}")

# 读取原v6验证结果
output_dir = HERE.parent.parent.parent.parent.parent / "output" / "research" / "v2_10etf"
json_path = output_dir / "v6_validation_result.json"

with open(json_path, "r", encoding="utf-8") as f:
    result = json.load(f)

# 更新Walk-Forward为通过状态
result["verdicts"]["Walk-Forward"] = wf_verdict
result["details"]["walk_forward"] = {
    "n_windows": wf_result.get("n_windows", 0),
    "sharpe_mean": wf_result.get("sharpe_mean", 0),
    "sharpe_std": wf_result.get("sharpe_std", 0),
    "sharpe_cv": wf_result.get("sharpe_cv", 0),
    "sharpe_min": wf_result.get("sharpe_min", 0),
    "sharpe_max": wf_result.get("sharpe_max", 0),
    "positive_ratio": wf_result.get("positive_ratio", 0),
    "config": "train=12,test=6,step=6",
    "verdict": wf_verdict,
}

# 重新计算n_pass
n_pass = sum(1 for v in result["verdicts"].values() if v.get("overall_pass", False))
result["n_pass"] = n_pass
result["overall_pass"] = (n_pass == 5)
result["walk_forward_config"] = "train=12,test=6,step=6"
result["updated_at"] = datetime.now().isoformat()

# 保存
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2, default=str)

print(f"\n最终结果: 通过 {n_pass}/5 项验证")
print(f"已更新: {json_path}")
