"""验证器4：滑点压力测试

A/B共识第2点扩展：策略换手37次/年，交易成本敏感

本模块用不同滑点成本跑策略，观察Sharpe衰减：
- 0.05% （宽松）
- 0.10% （标准）
- 0.15% （基准，与回测一致）
- 0.20% （保守）
- 0.30% （压力）

判定规则：
- Sharpe从0.05% → 0.30%衰减 < 30% → 策略对成本不敏感，稳健
- 衰减 30%-60% → 中等敏感
- 衰减 > 60% → 高度敏感，实盘风险大
"""
import numpy as np
import pandas as pd
from typing import Dict, Callable
from benchmark_family import calc_metrics


def run_slippage_stress(strategy_fn: Callable, close: pd.DataFrame, ohlcv_dict: dict,
                        slippage_levels=(0.0005, 0.001, 0.0015, 0.002, 0.003),
                        base_kwargs: Dict = None) -> Dict:
    """滑点压力测试

    Args:
        strategy_fn: 策略函数 (close, ohlcv_dict, cost=, **kwargs) -> (equity, metrics, signal)
        close: 收盘价
        ohlcv_dict: OHLCV
        slippage_levels: 滑点级别
        base_kwargs: 其他固定参数

    Returns:
        各滑点下的表现 + 衰减率 + 判定
    """
    base_kwargs = base_kwargs or {}
    results = []

    for slip in slippage_levels:
        kwargs = {**base_kwargs, "cost": slip}
        try:
            equity, metrics, _ = strategy_fn(close, ohlcv_dict, **kwargs)
            results.append({
                "slippage": slip,
                **metrics,
            })
        except Exception as e:
            results.append({"slippage": slip, "error": str(e)})

    valid = [r for r in results if "sharpe" in r]
    if len(valid) < 2:
        return {"results": results, "verdict": {"error": "有效点数不足"}}

    base_sharpe = valid[0]["sharpe"]
    stress_sharpe = valid[-1]["sharpe"]
    decay_rate = (base_sharpe - stress_sharpe) / (abs(base_sharpe) + 1e-9)

    if decay_rate < 0.3:
        sensitivity = "LOW"
        pass_judge = True
    elif decay_rate < 0.6:
        sensitivity = "MEDIUM"
        pass_judge = True
    else:
        sensitivity = "HIGH"
        pass_judge = False

    return {
        "results": results,
        "base_sharpe": base_sharpe,
        "stress_sharpe": stress_sharpe,
        "decay_rate": float(decay_rate),
        "sensitivity": sensitivity,
        "verdict": {
            "sensitivity": sensitivity,
            "decay_rate": decay_rate,
            "verdict_summary": f"Sharpe衰减率 {decay_rate*100:.1f}% ({sensitivity})",
            "overall_pass": pass_judge,
        },
    }
