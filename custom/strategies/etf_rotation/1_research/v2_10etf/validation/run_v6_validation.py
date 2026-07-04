"""v6策略5项验证：用A/B共识最优配置跑完整验证

v6最优配置（A/B共识）：
- hold_count=4
- switch_threshold=0.0 (无调仓阈值，让动量自然轮换)
- use_stop_loss=False (无止损，防御机制已足够)
- drawdown_threshold=-0.08 (更早防御)
- 其他保持v3默认

参数搜索结果：Sharpe 2.284 (+0.232 vs 基准2.052)
"""
import sys
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent / "v1_3etf"))

from data_generator_v2 import generate_simulation_data_v2
from factors_v3 import strategy_unified_v3
from backtest import run_backtest, calc_metrics as bt_calc_metrics

from benchmark_family import run_all_benchmarks
from walk_forward import run_walk_forward
from param_sensitivity import run_param_sensitivity
from slippage_stress import run_slippage_stress
from random_perturbation import run_random_perturbation


# ============================================================
# v6 策略统一接口
# ============================================================

def strategy_unified_v6(close, ohlcv_dict, **kwargs):
    """v6统一策略接口

    A/B共识最优配置：
    - hold_count=4
    - switch_threshold=0.0
    - use_stop_loss=False
    - drawdown_threshold=-0.08
    """
    defaults = {
        'hold_count': 4,
        'switch_threshold': 0.0,
        'use_stop_loss': False,
        'drawdown_threshold': -0.08,
        'use_defense': True,
        'cost_stop_loss': -0.12,
        'trailing_stop_loss': -0.10,
    }
    defaults.update(kwargs)
    return strategy_unified_v3(close, ohlcv_dict, **defaults)


# ============================================================
# 主流程
# ============================================================

def main():
    print("=" * 70)
    print("v6策略 5项验证 (A/B共识最优配置)")
    print("=" * 70)
    print("配置: hold=4, switch=0.0, 无止损, 防御-8%")
    print()

    # 1. 加载数据
    print("[1/6] 生成模拟数据...")
    close_df, ohlcv_dict, _ = generate_simulation_data_v2(n_years=5, seed=42)
    print(f"  数据: {close_df.shape[0]}交易日, {close_df.shape[1]}ETF")

    # 2. 跑v6策略获取baseline
    print("\n[2/6] 跑v6策略获取baseline...")
    equity, metrics, signal = strategy_unified_v6(close_df, ohlcv_dict)
    print(f"  v6策略: 年化={metrics['annual_return']*100:.2f}%, "
          f"Sharpe={metrics['sharpe']:.3f}, 回撤={metrics['max_drawdown']*100:.2f}%")
    print(f"  换手={metrics.get('n_switches',0)}, 防御={metrics.get('n_defense',0)}, 止损={metrics.get('n_stop_loss',0)}")

    # 3. 验证1：公平基准族
    print("\n[3/6] 验证1: 公平基准族 (蒙特卡洛500次)")
    benchmark_result = run_all_benchmarks(close_df, metrics, n_mc_runs=500)
    _print_verdict("公平基准族", benchmark_result["verdict"])

    # 4. 验证2：Walk-Forward
    print("\n[4/6] 验证2: Walk-Forward 滚动样本外")
    wf_result = run_walk_forward(
        close_df, ohlcv_dict, strategy_unified_v6,
        train_months=6, test_months=3, step_months=3,
    )
    if "verdict" in wf_result:
        _print_verdict("Walk-Forward", wf_result["verdict"])

    # 5. 验证3：参数敏感性
    print("\n[5/6] 验证3: 参数敏感性")
    params_to_scan = {
        "momentum_window": 25,
        "rsrs_window": 18,
        "ma_short": 20,
        "ma_long": 60,
    }
    ps_result = run_param_sensitivity(
        strategy_unified_v6, close_df, ohlcv_dict, params_to_scan,
        fixed_kwargs={}
    )
    _print_verdict("参数敏感性", ps_result["verdict"])

    # 6. 验证4：滑点压力测试
    print("\n[6/6] 验证4 & 5: 滑点压力 + 随机扰动")
    slip_result = run_slippage_stress(
        strategy_unified_v6, close_df, ohlcv_dict,
        base_kwargs={}
    )
    _print_verdict("滑点压力", slip_result["verdict"])

    # 7. 验证5：随机扰动
    print("\n  随机扰动测试 (80次)...")
    perturb_result = run_random_perturbation(
        strategy_unified_v6, close_df, ohlcv_dict,
        base_kwargs={},
        n_runs=80,
    )
    if "verdict" in perturb_result:
        _print_verdict("随机扰动", perturb_result["verdict"])

    # 8. 汇总
    print("\n" + "=" * 70)
    print("v6策略 5项验证汇总")
    print("=" * 70)

    all_verdicts = {
        "公平基准族": benchmark_result["verdict"],
        "Walk-Forward": wf_result.get("verdict", {"verdict_summary": "N/A", "overall_pass": False}),
        "参数敏感性": ps_result["verdict"],
        "滑点压力": slip_result["verdict"],
        "随机扰动": perturb_result.get("verdict", {"verdict_summary": "N/A", "overall_pass": False}),
    }
    n_pass = sum(1 for v in all_verdicts.values() if v.get("overall_pass", False))
    print(f"\n通过 {n_pass}/5 项验证")
    for name, v in all_verdicts.items():
        mark = "✓" if v.get("overall_pass", False) else "✗"
        print(f"  {mark} {name}: {v.get('verdict_summary', 'N/A')}")

    overall_pass = n_pass == 5
    print(f"\n{'='*70}")
    if overall_pass:
        print("最终判定: v6策略通过全部5项验证 — alpha显著且稳健")
    else:
        print(f"最终判定: v6策略未通过全部验证 ({n_pass}/5)")
    print(f"{'='*70}")

    # 9. 输出JSON结果
    output_dir = HERE.parent.parent.parent.parent.parent / "output" / "research" / "v2_10etf"
    output_dir.mkdir(parents=True, exist_ok=True)

    result_json = {
        "timestamp": datetime.now().isoformat(),
        "version": "v6",
        "config": {
            "hold_count": 4,
            "switch_threshold": 0.0,
            "use_stop_loss": False,
            "drawdown_threshold": -0.08,
        },
        "strategy_metrics": metrics,
        "verdicts": all_verdicts,
        "n_pass": n_pass,
        "overall_pass": overall_pass,
        "details": {
            "benchmark": _safe_dict(benchmark_result),
            "walk_forward": _safe_dict(wf_result),
            "param_sensitivity": _safe_dict(ps_result),
            "slippage": _safe_dict(slip_result),
            "perturbation": _safe_dict(perturb_result),
        },
    }

    json_path = output_dir / "v6_validation_result.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result_json, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n结果已保存: {json_path}")

    return overall_pass


def _print_verdict(name, verdict):
    """打印单项验证结论"""
    if not verdict:
        return
    mark = "✓" if verdict.get("overall_pass", False) else "✗"
    print(f"  {mark} {name}: {verdict.get('verdict_summary', 'N/A')}")
    for check in verdict.get("checks", []):
        print(f"      {check}")


def _safe_dict(d):
    """移除不可序列化的字段"""
    if not isinstance(d, dict):
        return d
    safe = {}
    for k, v in d.items():
        if isinstance(v, (np.ndarray, list)) and k in ["sharpes_raw", "sharpes"]:
            safe[k] = [float(x) for x in v] if isinstance(v, list) else v.tolist()
        elif isinstance(v, pd.DataFrame):
            continue
        elif isinstance(v, pd.Series):
            continue
        elif isinstance(v, dict):
            safe[k] = _safe_dict(v)
        elif isinstance(v, (np.integer, np.floating)):
            safe[k] = float(v)
        elif isinstance(v, (str, int, float, bool, list)) or v is None:
            safe[k] = v
        else:
            try:
                json.dumps(v, default=str)
                safe[k] = v
            except Exception:
                safe[k] = str(v)
    return safe


if __name__ == "__main__":
    main()
