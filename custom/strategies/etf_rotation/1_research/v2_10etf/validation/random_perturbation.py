"""验证器5：随机扰动测试（wu.run 80次）

wu.run 5项验证流程中的"杀手锏"：
- 保留策略的实际换手时点和次数
- 但在每次换手时随机选择持仓标的
- 重复80次（wu.run建议80次，足够稳定分布）
- 看策略Sharpe落在分布的哪个分位

判定规则：
- 策略Sharpe > 95%分位 → 非偶然alpha
- 70%-95%分位 → 弱alpha，需更多证据
- < 70%分位 → 无alpha，策略选择不优于随机

与基准族random_top2的区别：
- random_top2：每个调仓日都随机选，等价"完全无信息"
- 随机扰动：保留策略的换手时点和次数，只替换选择
  → 更严格，控制了"换手频率"变量
"""
import numpy as np
import pandas as pd
from typing import Dict, Callable
from benchmark_family import calc_metrics


def run_random_perturbation(strategy_fn: Callable, close: pd.DataFrame, ohlcv_dict: dict,
                            base_kwargs: Dict = None, n_runs=80, seed=42) -> Dict:
    """随机扰动测试

    Args:
        strategy_fn: 策略函数 (close, ohlcv_dict, **kwargs) -> (equity, metrics, signal)
        close: 收盘价
        ohlcv_dict: OHLCV
        base_kwargs: 策略参数
        n_runs: 随机扰动次数
        seed: 随机种子

    Returns:
        扰动分布 + 策略分位 + 判定
    """
    base_kwargs = base_kwargs or {}

    # 1. 跑原始策略，获取换手时点
    print(f"  [1/{n_runs+1}] 跑原始策略获取换手时点...")
    try:
        strategy_equity, strategy_metrics, strategy_signal = strategy_fn(
            close, ohlcv_dict, **base_kwargs
        )
    except Exception as e:
        return {"error": f"原始策略执行失败: {e}"}

    # 提取换手时点（持仓变化的日子）
    weight_change = strategy_signal.diff().abs().sum(axis=1)
    rebalance_dates = weight_change[weight_change > 0].index.tolist()
    n_rebalances = len(rebalance_dates)
    print(f"  原始策略换手{n_rebalances}次")

    if n_rebalances < 2:
        return {"error": "策略换手次数过少，无法做扰动测试"}

    # 2. 在每个换手时点随机选择持仓，跑n_runs次
    rng = np.random.default_rng(seed)
    daily_ret = close.pct_change().fillna(0)
    hold_count = int(strategy_signal.sum(axis=1).max())  # 推断持仓数
    hold_count = max(1, hold_count)

    sharpes_perturbed = []
    ann_rets_perturbed = []
    max_dds_perturbed = []

    for run_i in range(n_runs):
        # 每次run生成随机持仓序列
        random_signal = pd.DataFrame(0.0, index=close.index, columns=close.columns)
        current_holdings = None

        for date in close.index:
            if date in rebalance_dates:
                # 随机选hold_count只
                current_holdings = rng.choice(close.columns, size=hold_count, replace=False)
            if current_holdings is not None:
                for col in current_holdings:
                    random_signal.loc[date, col] = 1.0 / hold_count

        # shift避免未来函数
        random_signal = random_signal.shift(1).fillna(0)

        # 计算收益（用与原策略相同的成本）
        cost = base_kwargs.get("cost", 0.0015)
        portfolio_ret = (daily_ret * random_signal).sum(axis=1)
        weight_change_random = random_signal.diff().abs().sum(axis=1).fillna(0)
        portfolio_ret = portfolio_ret - weight_change_random * cost

        equity = (1 + portfolio_ret).cumprod()
        m = calc_metrics(equity)
        sharpes_perturbed.append(m["sharpe"])
        ann_rets_perturbed.append(m["annual_return"])
        max_dds_perturbed.append(m["max_drawdown"])

        if (run_i + 1) % 10 == 0:
            print(f"  [{run_i+1}/{n_runs}] 完成")

    # 3. 计算策略Sharpe在扰动分布中的分位
    strategy_sharpe = strategy_metrics["sharpe"]
    sharpes_arr = np.array(sharpes_perturbed)
    percentile = float((sharpes_arr < strategy_sharpe).mean() * 100)
    z_score = float((strategy_sharpe - sharpes_arr.mean()) / (sharpes_arr.std() + 1e-9))

    if percentile >= 95:
        verdict_summary = f"策略Sharpe位于{percentile:.1f}分位 (>=95%) — 非偶然alpha"
        overall_pass = True
    elif percentile >= 70:
        verdict_summary = f"策略Sharpe位于{percentile:.1f}分位 (70-95%) — 弱alpha，需更多证据"
        overall_pass = False
    else:
        verdict_summary = f"策略Sharpe位于{percentile:.1f}分位 (<70%) — 无alpha，策略选择不优于随机"
        overall_pass = False

    return {
        "strategy_sharpe": strategy_sharpe,
        "strategy_metrics": strategy_metrics,
        "n_runs": n_runs,
        "n_rebalances": n_rebalances,
        "perturbation_stats": {
            "sharpe_mean": float(sharpes_arr.mean()),
            "sharpe_std": float(sharpes_arr.std()),
            "sharpe_p5": float(np.percentile(sharpes_arr, 5)),
            "sharpe_p50": float(np.percentile(sharpes_arr, 50)),
            "sharpe_p95": float(np.percentile(sharpes_arr, 95)),
            "ann_ret_mean": float(np.mean(ann_rets_perturbed)),
            "max_dd_mean": float(np.mean(max_dds_perturbed)),
        },
        "percentile": percentile,
        "z_score": z_score,
        "sharpes_raw": sharpes_perturbed,
        "verdict": {
            "percentile": percentile,
            "z_score": z_score,
            "verdict_summary": verdict_summary,
            "overall_pass": overall_pass,
        },
    }
