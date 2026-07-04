"""阶段1：报告生成器

生成回测报告，包括：
- 指标摘要表
- 资金曲线图（策略 vs 等权基准）
- 回撤曲线图（灰色阴影）
- 持仓信号图
- 动量因子曲线图
- 交易明细表

知识库依据：项目约定 custom/reports/ 模块，但阶段1用简化版自建matplotlib图
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime


def setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def generate_report(
    close: pd.DataFrame,
    signal: pd.DataFrame,
    backtest_result: dict,
    strategy_metrics: dict,
    benchmark_metrics: dict,
    momentum: pd.DataFrame,
    output_dir: Path,
    params: dict,
) -> Path:
    """生成阶段1回测报告

    Args:
        close: 收盘价
        signal: 持仓信号
        backtest_result: 回测结果
        strategy_metrics: 策略指标
        benchmark_metrics: 基准指标
        momentum: 动量值
        output_dir: 输出目录
        params: 策略参数

    Returns:
        Path: HTML报告路径
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    plt = setup_matplotlib()

    equity = backtest_result["equity"]
    returns = backtest_result["returns"]
    positions = backtest_result["positions"]

    # 等权基准
    bench_ret = close.pct_change().fillna(0).mean(axis=1)
    bench_equity = (1 + bench_ret).cumprod()

    # ===== 图1: 资金曲线 + 回撤阴影 =====
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [3, 1]},
                              constrained_layout=True)

    # 上图: 资金曲线
    ax1 = axes[0]
    ax1.plot(equity.index, equity.values, label="轮动策略", color="steelblue", linewidth=1.5)
    ax1.plot(bench_equity.index, bench_equity.values, label="等权基准", color="gray", linewidth=1, alpha=0.7)
    ax1.axhline(1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax1.set_title(f"阶段1: 3ETF轮动策略资金曲线 (动量窗口={params['window']}日, 调仓={params['rebalance_freq']})",
                  fontsize=13)
    ax1.set_ylabel("净值")
    ax1.legend(loc="best")
    ax1.grid(True, alpha=0.3)

    # 回撤阴影
    rolling_max = equity.cummax()
    drawdown = (equity - rolling_max) / rolling_max
    ax1.fill_between(equity.index, 1.0, equity.values, where=(drawdown < 0),
                     color="red", alpha=0.15, label="回撤区间")

    # 下图: 回撤曲线
    ax2 = axes[1]
    ax2.fill_between(drawdown.index, drawdown.values * 100, 0, color="red", alpha=0.4)
    ax2.set_ylabel("回撤 (%)")
    ax2.set_xlabel("日期")
    ax2.grid(True, alpha=0.3)

    fig.savefig(output_dir / "01_equity_drawdown.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # ===== 图2: 持仓信号 + 动量因子 =====
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), constrained_layout=True)

    # 上图: 持仓信号（堆叠面积图）
    ax1 = axes[0]
    # 转换为持仓ETF名称
    hold_names = []
    for date in positions.index:
        held = positions.columns[positions.loc[date] == 1].tolist()
        hold_names.append(held[0] if held else "空仓")
    hold_series = pd.Series(hold_names, index=positions.index)

    # 用数值编码画持仓
    etf_codes = list(positions.columns)
    for i, code in enumerate(etf_codes):
        ax1.fill_between(positions.index, 0, (positions[code] * (i + 1)).values,
                         alpha=0.5, label=f"{code}")
    ax1.axhline(0, color="black", linewidth=0.5)
    ax1.set_yticks(range(len(etf_codes) + 1))
    ax1.set_yticklabels(["空仓"] + etf_codes)
    ax1.set_title("持仓信号（每月调仓）", fontsize=12)
    ax1.set_ylabel("持仓标的")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, alpha=0.3)

    # 下图: 动量因子曲线
    ax2 = axes[1]
    for code in momentum.columns:
        ax2.plot(momentum.index, momentum[code].values * 100, label=code, linewidth=1, alpha=0.8)
    ax2.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax2.set_title(f"{params['window']}日ROC动量因子", fontsize=12)
    ax2.set_ylabel("动量 (%)")
    ax2.set_xlabel("日期")
    ax2.legend(loc="best", fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.savefig(output_dir / "02_signal_momentum.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # ===== 图3: 各ETF净值走势（归一化） =====
    fig, ax = plt.subplots(figsize=(14, 6))
    for code in close.columns:
        nav = close[code] / close[code].iloc[0]
        ax.plot(close.index, nav.values, label=code, linewidth=1.2, alpha=0.85)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_title("3个模拟ETF净值走势（归一化）", fontsize=13)
    ax.set_ylabel("归一化净值")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "03_etf_nav.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # ===== HTML 报告 =====
    html_path = generate_html_report(
        strategy_metrics, benchmark_metrics, backtest_result, params, output_dir
    )
    return html_path


def generate_html_report(
    strategy_metrics: dict,
    benchmark_metrics: dict,
    backtest_result: dict,
    params: dict,
    output_dir: Path,
) -> Path:
    """生成HTML报告"""
    # 指标对比表
    metrics_compare = pd.DataFrame({
        "轮动策略": strategy_metrics,
        "等权基准": benchmark_metrics,
    }).T

    # 格式化
    pct_cols = ["total_return", "annual_return", "annual_volatility", "max_drawdown", "win_rate"]
    for col in pct_cols:
        if col in metrics_compare.columns:
            metrics_compare[col] = (metrics_compare[col] * 100).round(2).astype(str) + "%"
    float_cols = ["sharpe", "calmar", "profit_loss_ratio", "years"]
    for col in float_cols:
        if col in metrics_compare.columns:
            metrics_compare[col] = metrics_compare[col].round(3)

    metrics_compare = metrics_compare.rename(columns={
        "total_return": "总收益", "annual_return": "年化收益", "annual_volatility": "年化波动",
        "sharpe": "Sharpe", "max_drawdown": "最大回撤", "calmar": "Calmar",
        "win_rate": "胜率", "profit_loss_ratio": "盈亏比", "years": "年数", "n_days": "交易日数",
    })

    # 总换手成本
    total_cost = backtest_result["cost"].sum()
    total_turnover = backtest_result["turnover"].sum()

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>ETF轮动策略阶段1回测报告</title>
<style>
body {{ font-family: "Microsoft YaHei", sans-serif; margin: 20px; background: #f5f5f5; }}
h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
h2 {{ color: #34495e; margin-top: 30px; }}
table {{ border-collapse: collapse; width: 100%; background: white; margin: 10px 0; font-size: 13px; }}
th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: center; }}
th {{ background: #3498db; color: white; }}
tr:nth-child(even) {{ background: #f9f9f9; }}
tr:hover {{ background: #e3f2fd; }}
img {{ max-width: 100%; border: 1px solid #ddd; margin: 10px 0; }}
.summary-box {{ background: white; padding: 15px; border-left: 4px solid #3498db; margin: 15px 0; }}
.note {{ background: #fff9c4; padding: 10px; border-radius: 4px; margin: 10px 0; }}
.params {{ background: #e8f5e9; padding: 10px; border-radius: 4px; margin: 10px 0; font-family: monospace; }}
</style>
</head>
<body>
<h1>ETF轮动策略阶段1回测报告</h1>
<p>生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>

<div class="summary-box">
<b>阶段1目标</b>：验证策略框架正确性，跑通"数据→信号→回测→报告"全流程。<br>
<b>数据</b>：3个模拟ETF（GBM+动量持续性+市场状态切换），5年日线数据。<br>
<b>策略</b>：60日ROC动量，月频调仓，持有动量最强1只，全池动量<0空仓。
</div>

<div class="params">
<b>策略参数</b><br>
  动量窗口: {params['window']}日<br>
  调仓频率: {params['rebalance_freq']} ({'月频' if params['rebalance_freq']=='M' else '周频' if params['rebalance_freq']=='W' else '季频'})<br>
  持仓数量: 1只（动量最强）<br>
  空仓条件: 最强动量 < 0<br>
  交易成本: 单边0.15%（佣金万2.5+滑点万3）<br>
  信号处理: shift(1)后移一天，避免未来函数
</div>

<h2>一、绩效指标对比</h2>
{metrics_compare.to_html(escape=False)}

<div class="note">
<b>交易成本统计</b>：总换手={total_turnover:.2f}, 总成本={total_cost*100:.2f}%（占初始资金比例）
</div>

<h2>二、资金曲线与回撤</h2>
<img src="01_equity_drawdown.png" alt="资金曲线与回撤">
<p>说明：上图蓝线为策略净值，灰线为等权基准净值，红色阴影为回撤区间；下图为完整周期回撤曲线。</p>

<h2>三、持仓信号与动量因子</h2>
<img src="02_signal_momentum.png" alt="持仓信号与动量因子">
<p>说明：上图为每日持仓标的（堆叠面积），下图为{params['window']}日ROC动量因子曲线，虚线为0轴。</p>

<h2>四、模拟ETF净值走势</h2>
<img src="03_etf_nav.png" alt="ETF净值走势">
<p>说明：3个模拟ETF归一化净值，起始=1.0。</p>

<h2>五、阶段1结论与下一步</h2>
<div class="summary-box">
<b>验证要点</b>：
1. 策略框架是否正确跑通（数据→信号→回测→报告）
2. 轮动策略相对等权基准是否有超额收益
3. 动量因子是否能识别强弱势ETF
4. 回撤控制是否合理（全池动量<0空仓机制）

<b>下一步（阶段2）</b>：
- 扩展到10个模拟ETF
- 因子升级：回归动量（年化收益×R²，过滤妖基）
- 加入L1双均线过滤 + L2双止损
- 信号过滤：调仓阈值10%
</div>

</body>
</html>
"""
    html_path = output_dir / "report.html"
    html_path.write_text(html, encoding="utf-8")
    return html_path
