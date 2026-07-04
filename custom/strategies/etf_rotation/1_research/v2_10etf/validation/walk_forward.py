"""验证器2：Walk-Forward 滚动样本外检验

A/B共识第4点：5年数据全用在了训练+回测上，没有walk-forward

本模块实现滚动窗口样本外检验：
- 把数据切成多个窗口
- 每个窗口：前N个月作为"预热期"计算因子，后M个月作为"测试期"应用规则
- 滚动推进，汇总所有测试期表现

判定规则：
- 各测试期Sharpe稳定为正 → 策略稳健
- 仅1-2段好、其余差 → 运气成分大
- 各段Sharpe标准差 / 均值 > 1.0 → 策略不稳定
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Callable
from benchmark_family import calc_metrics


def walk_forward_split(close: pd.DataFrame, ohlcv_dict: dict,
                       train_months=6, test_months=3,
                       step_months=3) -> List[Dict]:
    """生成walk-forward窗口列表

    Args:
        close: 收盘价
        ohlcv_dict: OHLCV字典（每个ETF一个DataFrame）
        train_months: 训练（预热）月数
        test_months: 测试月数
        step_months: 滚动步长月数

    Returns:
        窗口列表，每个窗口含 train_start, train_end, test_start, test_end
    """
    start_date = close.index[0]
    end_date = close.index[-1]

    # 按月生成日期序列
    month_ends = close.index.to_series().groupby(
        [close.index.year, close.index.month]
    ).last().sort_values().tolist()

    windows = []
    i = 0
    while True:
        train_start_idx = i
        train_end_idx = i + train_months - 1
        test_start_idx = train_end_idx + 1
        test_end_idx = test_start_idx + test_months - 1

        if test_end_idx >= len(month_ends):
            break

        train_start = month_ends[train_start_idx] if train_start_idx > 0 else start_date
        # 训练起点实际取该月第一个交易日
        train_start = close.index[close.index >= month_ends[train_start_idx].replace(day=1)][0]
        train_end = month_ends[train_end_idx]
        test_start = close.index[close.index > train_end][0]
        test_end = month_ends[test_end_idx]

        windows.append({
            "window_id": len(windows),
            "train_start": train_start,
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
        })
        i += step_months

    return windows


def run_walk_forward(close: pd.DataFrame, ohlcv_dict: dict,
                     strategy_fn: Callable, strategy_kwargs: Dict = None,
                     train_months=6, test_months=3, step_months=3) -> Dict:
    """运行walk-forward检验

    Args:
        close: 完整收盘价
        ohlcv_dict: OHLCV字典
        strategy_fn: 策略函数，签名 (close, ohlcv_dict, **kwargs) -> (equity, metrics, signal)
        strategy_kwargs: 策略参数
        train_months/test_months/step_months: 切分参数

    Returns:
        各窗口结果 + 汇总判定
    """
    strategy_kwargs = strategy_kwargs or {}
    windows = walk_forward_split(close, ohlcv_dict, train_months, test_months, step_months)
    print(f"  共{len(windows)}个walk-forward窗口")

    window_results = []
    for w in windows:
        # 切片测试期数据
        test_mask = (close.index >= w["test_start"]) & (close.index <= w["test_end"])
        test_close = close.loc[test_mask].copy()
        test_ohlcv = {k: v.loc[test_mask].copy() for k, v in ohlcv_dict.items()}

        if len(test_close) < 20:  # 数据太少跳过
            continue

        try:
            equity, metrics, _ = strategy_fn(test_close, test_ohlcv, **strategy_kwargs)
            # 归一化equity从1开始
            if equity.iloc[0] != 0:
                equity = equity / equity.iloc[0]
            window_results.append({
                "window_id": w["window_id"],
                "test_start": w["test_start"],
                "test_end": w["test_end"],
                "n_days": len(test_close),
                **metrics,
            })
        except Exception as e:
            window_results.append({
                "window_id": w["window_id"],
                "test_start": w["test_start"],
                "test_end": w["test_end"],
                "error": str(e),
            })

    # 汇总
    valid_results = [r for r in window_results if "sharpe" in r]
    if not valid_results:
        return {"windows": window_results, "verdict": {"error": "无有效窗口"}}

    sharpes = [r["sharpe"] for r in valid_results]
    ann_rets = [r["annual_return"] for r in valid_results]
    max_dds = [r["max_drawdown"] for r in valid_results]

    sharpe_mean = float(np.mean(sharpes))
    sharpe_std = float(np.std(sharpes))
    sharpe_cv = sharpe_std / (abs(sharpe_mean) + 1e-9)  # 变异系数

    # 正Sharpe窗口占比
    positive_ratio = float(np.mean([1 if s > 0 else 0 for s in sharpes]))

    # 判定
    checks = []
    if sharpe_mean > 0:
        checks.append(f"✓ 平均Sharpe = {sharpe_mean:.3f} (正)")
    else:
        checks.append(f"✗ 平均Sharpe = {sharpe_mean:.3f} (非正)")

    if positive_ratio >= 0.6:
        checks.append(f"✓ 正Sharpe窗口占比 {positive_ratio*100:.0f}% (>=60%)")
    else:
        checks.append(f"✗ 正Sharpe窗口占比 {positive_ratio*100:.0f}% (<60%) — 不稳定")

    if sharpe_cv < 1.0:
        checks.append(f"✓ Sharpe变异系数 {sharpe_cv:.2f} (<1.0) — 各期稳定")
    else:
        checks.append(f"✗ Sharpe变异系数 {sharpe_cv:.2f} (>=1.0) — 各期波动大")

    overall_pass = (sharpe_mean > 0 and positive_ratio >= 0.6 and sharpe_cv < 1.0)

    return {
        "windows": window_results,
        "n_windows": len(valid_results),
        "sharpe_mean": sharpe_mean,
        "sharpe_std": sharpe_std,
        "sharpe_cv": sharpe_cv,
        "sharpe_min": float(np.min(sharpes)),
        "sharpe_max": float(np.max(sharpes)),
        "positive_ratio": positive_ratio,
        "ann_ret_mean": float(np.mean(ann_rets)),
        "max_dd_mean": float(np.mean(max_dds)),
        "verdict": {
            "checks": checks,
            "overall_pass": overall_pass,
            "verdict_summary": "策略通过Walk-Forward检验" if overall_pass else "策略未通过Walk-Forward检验",
        },
    }
