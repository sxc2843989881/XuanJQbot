"""阶段1：回测引擎

最小化回测引擎，支持：
- 持仓信号输入（已 shift(1)）
- 交易成本扣除（单边0.15%）
- 资金曲线计算
- 换手率统计

知识库依据：01_ETF轮动策略/03_经典策略实现.md 入门版回测函数
"""
import pandas as pd
import numpy as np


# 默认交易成本（单边）
DEFAULT_FEE = 0.00025       # 佣金万2.5
DEFAULT_SLIPPAGE = 0.0003   # 滑点万3
DEFAULT_TOTAL_COST = DEFAULT_FEE + DEFAULT_SLIPPAGE  # 单边0.15%


def run_backtest(
    close: pd.DataFrame,
    signal: pd.DataFrame,
    cost: float = DEFAULT_TOTAL_COST,
) -> dict:
    """运行回测

    Args:
        close: 收盘价 DataFrame
        signal: 持仓信号 DataFrame（已 shift(1)，1=持有, 0=空仓）
        cost: 单边交易成本（佣金+滑点）

    Returns:
        dict: {
            'equity': 资金曲线,
            'returns': 日收益率序列,
            'turnover': 每日换手率,
            'cost': 每日成本,
            'positions': 每日持仓明细,
        }
    """
    # 日收益率
    daily_ret = close.pct_change().fillna(0)

    # 组合日收益率 = 持仓权重 × 个股收益率
    # signal 已 shift(1)，所以当日的 signal 是基于昨日信息决定的今日持仓
    portfolio_ret = (signal * daily_ret).sum(axis=1)

    # 换手率 = 每日持仓变化绝对值之和 / 2
    turnover = signal.diff().abs().sum(axis=1).fillna(0) / 2

    # 交易成本
    daily_cost = turnover * cost

    # 扣除成本后的净收益率
    net_ret = portfolio_ret - daily_cost

    # 资金曲线
    equity = (1 + net_ret).cumprod()

    return {
        "equity": equity,
        "returns": net_ret,
        "turnover": turnover,
        "cost": daily_cost,
        "positions": signal,
        "gross_returns": portfolio_ret,
    }


def calc_metrics(returns: pd.Series, risk_free_rate: float = 0.02) -> dict:
    """计算业绩指标

    知识库依据：03_经典策略实现.md calc_metrics 函数

    Args:
        returns: 日收益率序列（已扣成本）
        risk_free_rate: 年化无风险利率

    Returns:
        dict: 业绩指标
    """
    equity = (1 + returns).cumprod()
    n_days = len(returns)
    years = n_days / 252

    # 总收益与年化
    total_return = equity.iloc[-1] - 1
    annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

    # 年化波动率
    annual_vol = returns.std() * np.sqrt(252)

    # Sharpe
    excess_ret = returns - risk_free_rate / 252
    sharpe = np.sqrt(252) * excess_ret.mean() / excess_ret.std() if excess_ret.std() > 0 else 0

    # 最大回撤（完整周期，不按年切分）
    rolling_max = equity.cummax()
    drawdown = (equity - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    # Calmar
    calmar = annual_return / abs(max_drawdown) if max_drawdown < 0 else 0

    # 胜率
    nonzero_ret = returns[returns != 0]
    win_rate = (nonzero_ret > 0).sum() / len(nonzero_ret) if len(nonzero_ret) > 0 else 0

    # 盈亏比
    wins = nonzero_ret[nonzero_ret > 0]
    losses = nonzero_ret[nonzero_ret < 0]
    avg_win = wins.mean() if len(wins) > 0 else 0
    avg_loss = abs(losses.mean()) if len(losses) > 0 else 0
    profit_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0

    # 年化换手率（估算）
    # turnover 是每日换手，年化 = 平均日换手 × 252 × 2（双边）
    # 注意：这里 turnover 已经是单边的，×2 得到双边

    return {
        "total_return": float(total_return),
        "annual_return": float(annual_return),
        "annual_volatility": float(annual_vol),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_drawdown),
        "calmar": float(calmar),
        "win_rate": float(win_rate),
        "profit_loss_ratio": float(profit_loss_ratio),
        "years": float(years),
        "n_days": int(n_days),
    }


def calc_benchmark_metrics(close: pd.DataFrame, risk_free_rate: float = 0.02) -> dict:
    """计算等权基准的业绩指标（作为对比）

    Args:
        close: 收盘价 DataFrame
        risk_free_rate: 年化无风险利率

    Returns:
        dict: 等权基准业绩指标
    """
    # 等权持有所有ETF
    daily_ret = close.pct_change().fillna(0)
    bench_ret = daily_ret.mean(axis=1)
    return calc_metrics(bench_ret, risk_free_rate)


if __name__ == "__main__":
    # 快速自测
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))

    from data_generator import generate_simulation_data
    from momentum_roc import generate_signal

    close_df, _, _ = generate_simulation_data(n_years=3, seed=42)
    signal = generate_signal(close_df, window=60, rebalance_freq="M")

    result = run_backtest(close_df, signal)
    metrics = calc_metrics(result["returns"])

    bench_metrics = calc_benchmark_metrics(close_df)

    print("=== 策略绩效 ===")
    for k, v in metrics.items():
        if isinstance(v, float):
            if k in ["total_return", "annual_return", "annual_volatility", "max_drawdown", "win_rate"]:
                print(f"  {k}: {v*100:.2f}%")
            else:
                print(f"  {k}: {v:.3f}")
        else:
            print(f"  {k}: {v}")

    print("\n=== 等权基准 ===")
    for k, v in bench_metrics.items():
        if isinstance(v, float):
            if k in ["total_return", "annual_return", "annual_volatility", "max_drawdown", "win_rate"]:
                print(f"  {k}: {v*100:.2f}%")
            else:
                print(f"  {k}: {v:.3f}")
        else:
            print(f"  {k}: {v}")
