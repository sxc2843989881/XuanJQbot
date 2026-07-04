"""验证器3：参数敏感性（plateau vs spike）

A/B共识第5点：单点参数取值，没做plateau检验

wu.run 5项验证流程中的核心一项：
- plateau（平台）：参数周围Sharpe稳定 → 策略稳健
- spike（尖峰）：只有单点最优、稍偏即崩溃 → 过拟合

本模块扫描关键参数周围±50%范围，绘制Sharpe曲线判定形状。
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Callable, Tuple
from benchmark_family import calc_metrics


def scan_parameter(strategy_fn: Callable, close: pd.DataFrame, ohlcv_dict: dict,
                   param_name: str, base_value, scan_range=(-0.5, 0.5),
                   n_points=11, fixed_kwargs: Dict = None) -> Dict:
    """扫描单个参数

    Args:
        strategy_fn: 策略函数 (close, ohlcv_dict, **kwargs) -> (equity, metrics, signal)
        close: 收盘价
        ohlcv_dict: OHLCV
        param_name: 要扫描的参数名
        base_value: 基准参数值
        scan_range: 扫描范围（相对base_value的偏移比例）
        n_points: 扫描点数
        fixed_kwargs: 其他固定参数

    Returns:
        扫描结果，含values列表、sharpes列表、判定
    """
    fixed_kwargs = fixed_kwargs or {}

    # 生成扫描值
    if isinstance(base_value, int) or base_value == int(base_value):
        # 整数参数（窗口类）
        base_int = int(base_value)
        delta = max(1, int(base_int * scan_range[1]))
        values = list(range(base_int - delta, base_int + delta + 1,
                            max(1, (2 * delta) // (n_points - 1))))
        values = sorted(set(values))
    else:
        # 浮点参数
        lo = base_value * (1 + scan_range[0])
        hi = base_value * (1 + scan_range[1])
        values = np.linspace(lo, hi, n_points).tolist()

    sharpes = []
    ann_rets = []
    max_dds = []
    scan_details = []

    for v in values:
        kwargs = {**fixed_kwargs, param_name: v}
        try:
            equity, metrics, _ = strategy_fn(close, ohlcv_dict, **kwargs)
            sharpes.append(metrics["sharpe"])
            ann_rets.append(metrics["annual_return"])
            max_dds.append(metrics["max_drawdown"])
            scan_details.append({"value": v, **metrics})
        except Exception as e:
            sharpes.append(np.nan)
            ann_rets.append(np.nan)
            max_dds.append(np.nan)
            scan_details.append({"value": v, "error": str(e)})

    # 判定plateau vs spike
    valid_sharpes = [s for s in sharpes if not np.isnan(s)]
    if len(valid_sharpes) < 3:
        return {
            "param_name": param_name,
            "base_value": base_value,
            "values": values,
            "sharpes": sharpes,
            "verdict": {"error": "有效点数不足"},
        }

    base_idx = _find_nearest_idx(values, base_value)
    base_sharpe = sharpes[base_idx] if base_idx < len(sharpes) else valid_sharpes[len(valid_sharpes)//2]

    # 计算周围5个点的Sharpe标准差（衡量稳定性）
    surround_start = max(0, base_idx - 2)
    surround_end = min(len(sharpes), base_idx + 3)
    surround_sharpes = [s for s in sharpes[surround_start:surround_end] if not np.isnan(s)]
    surround_std = float(np.std(surround_sharpes)) if len(surround_sharpes) > 1 else 0

    # 判定：周围标准差 / |基准Sharpe| < 0.3 → plateau
    if abs(base_sharpe) < 0.05:
        shape = "FLAT"  # 都接近0
        is_plateau = False
    elif surround_std / abs(base_sharpe) < 0.3:
        shape = "PLATEAU"
        is_plateau = True
    elif surround_std / abs(base_sharpe) > 0.8:
        shape = "SPIKE"
        is_plateau = False
    else:
        shape = "MODERATE"
        is_plateau = True  # 中等波动可接受

    return {
        "param_name": param_name,
        "base_value": base_value,
        "values": values,
        "sharpes": sharpes,
        "ann_rets": ann_rets,
        "max_dds": max_dds,
        "details": scan_details,
        "base_sharpe": base_sharpe,
        "surround_std": surround_std,
        "shape": shape,
        "is_plateau": is_plateau,
        "verdict": {
            "shape": shape,
            "verdict_summary": f"{shape}: 参数{param_name}={base_value}周围Sharpe标准差={surround_std:.3f}",
            "overall_pass": is_plateau,
        },
    }


def run_param_sensitivity(strategy_fn: Callable, close: pd.DataFrame, ohlcv_dict: dict,
                          params_to_scan: Dict, fixed_kwargs: Dict = None) -> Dict:
    """运行多参数敏感性扫描

    Args:
        strategy_fn: 策略函数
        close: 收盘价
        ohlcv_dict: OHLCV
        params_to_scan: {param_name: base_value} 待扫描参数
        fixed_kwargs: 其他固定参数

    Returns:
        各参数扫描结果 + 综合判定
    """
    results = {}
    for param_name, base_value in params_to_scan.items():
        print(f"  扫描参数 {param_name} (基准值={base_value})...")
        results[param_name] = scan_parameter(
            strategy_fn, close, ohlcv_dict, param_name, base_value,
            fixed_kwargs=fixed_kwargs
        )

    # 综合判定
    all_pass = all(r.get("is_plateau", False) for r in results.values())
    shapes = [r.get("shape", "UNKNOWN") for r in results.values()]
    n_spike = sum(1 for s in shapes if s == "SPIKE")

    return {
        "params": results,
        "all_pass": all_pass,
        "n_spike": n_spike,
        "verdict": {
            "shapes": shapes,
            "verdict_summary": (
                f"全部参数呈plateau — 策略稳健" if all_pass else
                f"{n_spike}个参数呈spike — 过拟合风险"
            ),
            "overall_pass": all_pass,
        },
    }


def _find_nearest_idx(values: list, target) -> int:
    """找到values中离target最近的索引"""
    if not values:
        return 0
    diffs = [abs(v - target) for v in values]
    return int(np.argmin(diffs))
