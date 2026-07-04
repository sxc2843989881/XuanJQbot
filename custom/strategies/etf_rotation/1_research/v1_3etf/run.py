"""阶段1主入口：3个ETF最小可运行版本

完整流程：
1. 生成模拟数据（3个ETF，5年日线）
2. 计算60日ROC动量
3. 生成月频调仓信号（持有最强1只，全池<0空仓）
4. 运行回测（含交易成本）
5. 计算业绩指标
6. 生成可视化报告

知识库依据：01_ETF轮动策略 系列6篇文档
"""
import sys
from pathlib import Path

# 添加当前目录到path，方便 import 同级模块
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_generator import generate_simulation_data, SIMULATED_ETFS
from momentum_roc import calculate_roc_momentum, generate_signal
from backtest import run_backtest, calc_metrics, calc_benchmark_metrics
from report import generate_report


# ============================================================
# 策略参数（阶段1配置）
# ============================================================
STRATEGY_PARAMS = {
    "n_years": 5,              # 模拟5年数据
    "seed": 42,                # 随机种子（可复现）
    "window": 60,              # 动量计算窗口（60日，知识库24%玩家用60日）
    "rebalance_freq": "M",     # 月频调仓（知识库新手推荐）
    "cost": 0.0015,            # 单边交易成本0.15%
}


def main():
    print("=" * 60)
    print("ETF轮动策略 阶段1：3个ETF最小可运行版本")
    print("=" * 60)
    print(f"\n策略参数:")
    for k, v in STRATEGY_PARAMS.items():
        print(f"  {k}: {v}")

    # 1. 生成模拟数据
    print(f"\n[1/6] 生成模拟数据（{STRATEGY_PARAMS['n_years']}年, 3个ETF）...")
    close_df, ohlcv_df, market_states = generate_simulation_data(
        n_years=STRATEGY_PARAMS["n_years"],
        seed=STRATEGY_PARAMS["seed"],
    )
    print(f"  数据: {close_df.shape[0]} 个交易日, {close_df.shape[1]} 个ETF")
    print(f"  日期范围: {close_df.index[0].date()} ~ {close_df.index[-1].date()}")

    # 各ETF基础统计
    print(f"\n  各模拟ETF基础统计:")
    for code in close_df.columns:
        close = close_df[code]
        total_ret = close.iloc[-1] / close.iloc[0] - 1
        years = len(close) / 252
        annual_ret = (1 + total_ret) ** (1 / years) - 1
        daily_ret = close.pct_change().dropna()
        annual_vol = daily_ret.std() * (252 ** 0.5)
        print(f"    {code} ({SIMULATED_ETFS[code]['name']}): "
              f"年化={annual_ret*100:.2f}%, 波动={annual_vol*100:.2f}%, "
              f"首={close.iloc[0]:.3f}, 末={close.iloc[-1]:.3f}")

    # 2. 计算动量
    print(f"\n[2/6] 计算{STRATEGY_PARAMS['window']}日ROC动量...")
    momentum = calculate_roc_momentum(close_df, window=STRATEGY_PARAMS["window"])
    print(f"  动量计算完成, 最后5日:")
    print(momentum.tail().to_string())

    # 3. 生成信号
    print(f"\n[3/6] 生成月频调仓信号...")
    signal = generate_signal(
        close_df,
        window=STRATEGY_PARAMS["window"],
        rebalance_freq=STRATEGY_PARAMS["rebalance_freq"],
    )

    # 信号统计
    print(f"\n  各ETF持仓天数占比:")
    for col in signal.columns:
        hold_pct = (signal[col] == 1).sum() / len(signal) * 100
        print(f"    {col}: {hold_pct:.1f}%")
    empty_pct = (signal.sum(axis=1) == 0).sum() / len(signal) * 100
    print(f"    空仓: {empty_pct:.1f}%")

    # 4. 运行回测
    print(f"\n[4/6] 运行回测（单边成本={STRATEGY_PARAMS['cost']*100:.2f}%）...")
    result = run_backtest(close_df, signal, cost=STRATEGY_PARAMS["cost"])

    # 5. 计算指标
    print(f"\n[5/6] 计算业绩指标...")
    strategy_metrics = calc_metrics(result["returns"])
    benchmark_metrics = calc_benchmark_metrics(close_df)

    print(f"\n  === 策略绩效 ===")
    print(f"    总收益:     {strategy_metrics['total_return']*100:.2f}%")
    print(f"    年化收益:   {strategy_metrics['annual_return']*100:.2f}%")
    print(f"    年化波动:   {strategy_metrics['annual_volatility']*100:.2f}%")
    print(f"    Sharpe:     {strategy_metrics['sharpe']:.3f}")
    print(f"    最大回撤:   {strategy_metrics['max_drawdown']*100:.2f}%")
    print(f"    Calmar:     {strategy_metrics['calmar']:.3f}")
    print(f"    胜率:       {strategy_metrics['win_rate']*100:.2f}%")
    print(f"    盈亏比:     {strategy_metrics['profit_loss_ratio']:.3f}")

    print(f"\n  === 等权基准 ===")
    print(f"    总收益:     {benchmark_metrics['total_return']*100:.2f}%")
    print(f"    年化收益:   {benchmark_metrics['annual_return']*100:.2f}%")
    print(f"    年化波动:   {benchmark_metrics['annual_volatility']*100:.2f}%")
    print(f"    Sharpe:     {benchmark_metrics['sharpe']:.3f}")
    print(f"    最大回撤:   {benchmark_metrics['max_drawdown']*100:.2f}%")
    print(f"    Calmar:     {benchmark_metrics['calmar']:.3f}")

    # 超额收益
    excess_annual = strategy_metrics["annual_return"] - benchmark_metrics["annual_return"]
    print(f"\n  === 超额收益 ===")
    print(f"    年化超额:   {excess_annual*100:.2f}%")
    print(f"    回撤改善:   {(strategy_metrics['max_drawdown'] - benchmark_metrics['max_drawdown'])*100:.2f}%")
    print(f"    Sharpe差:   {strategy_metrics['sharpe'] - benchmark_metrics['sharpe']:.3f}")

    # 6. 生成报告
    print(f"\n[6/6] 生成可视化报告...")
    # run.py 在 custom/strategies/etf_rotation/1_research/v1_3etf/run.py
    # 需要回到 custom/ 目录，然后进入 output/research/v1_3etf/
    output_dir = Path(__file__).resolve().parent.parent.parent.parent.parent / "output" / "research" / "v1_3etf"
    html_path = generate_report(
        close_df, signal, result, strategy_metrics, benchmark_metrics,
        momentum, output_dir, STRATEGY_PARAMS,
    )
    print(f"  HTML报告: {html_path}")
    print(f"  图表目录: {output_dir}")

    print(f"\n{'='*60}")
    print(f"阶段1完成！")
    print(f"{'='*60}")
    print(f"\n验证结论:")
    if strategy_metrics["annual_return"] > benchmark_metrics["annual_return"]:
        print(f"  [OK] 策略年化({strategy_metrics['annual_return']*100:.2f}%) > 基准({benchmark_metrics['annual_return']*100:.2f}%), 轮动有效")
    else:
        print(f"  [WARN] 策略年化({strategy_metrics['annual_return']*100:.2f}%) <= 基准({benchmark_metrics['annual_return']*100:.2f}%), 需检查")

    if strategy_metrics["max_drawdown"] > benchmark_metrics["max_drawdown"]:
        print(f"  [OK] 策略回撤({strategy_metrics['max_drawdown']*100:.2f}%) < 基准({benchmark_metrics['max_drawdown']*100:.2f}%), 风控有效")
    else:
        print(f"  [WARN] 策略回撤({strategy_metrics['max_drawdown']*100:.2f}%) >= 基准({benchmark_metrics['max_drawdown']*100:.2f}%), 需改进")

    if strategy_metrics["sharpe"] > 0.5:
        print(f"  [OK] Sharpe={strategy_metrics['sharpe']:.3f} > 0.5, 风险调整收益可接受")
    else:
        print(f"  [WARN] Sharpe={strategy_metrics['sharpe']:.3f} <= 0.5, 需优化")


if __name__ == "__main__":
    main()
