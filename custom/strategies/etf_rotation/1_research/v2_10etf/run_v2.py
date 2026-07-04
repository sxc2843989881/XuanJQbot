"""阶段2主入口：10个ETF + 多因子 + 消融分析

完整流程：
1. 生成10个模拟ETF数据（5年）
2. 消融分析：分别测试各模块组合的绩效
   - A: 纯回归动量（基线）
   - B: 回归动量 + RSRS
   - C: 回归动量 + 双均线过滤
   - D: 回归动量 + 趋势反转风控
   - E: 回归动量 + RSRS + 双均线（无反转风控）
   - F: 全策略（回归动量+RSRS+双均线+反转风控）
3. 对比等权基准
4. 生成对比报告

知识库依据：06_失败教训/wu.run 5项验证之"消融分析"
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "v1_3etf"))

from data_generator_v2 import generate_simulation_data_v2, SIMULATED_ETFS_V2
from factors_v2 import calculate_composite_score, generate_signal_v2
from backtest import run_backtest, calc_metrics, calc_benchmark_metrics


# ============================================================
# 策略参数
# ============================================================
STRATEGY_PARAMS = {
    "n_years": 5,
    "seed": 42,
    "momentum_window": 25,    # 回归动量窗口（同花顺默认）
    "rsrs_window": 18,        # RSRS窗口（光大默认）
    "ma_short": 20,           # 短期均线
    "ma_long": 60,            # 长期均线
    "rebalance_freq": "M",    # 月频调仓
    "hold_count": 2,          # 持仓2只（适度分散）
    "cost": 0.0015,           # 单边成本0.15%
    "empty_threshold": 0.0,   # 最高分<0空仓
}

# 消融分析配置：6种组合
ABLATION_CONFIGS = {
    "A_纯回归动量": {"use_rsrs": False, "use_ma_filter": False, "use_reversal_filter": False},
    "B_动量+RSRS": {"use_rsrs": True, "use_ma_filter": False, "use_reversal_filter": False},
    "C_动量+双均线": {"use_rsrs": False, "use_ma_filter": True, "use_reversal_filter": False},
    "D_动量+反转风控": {"use_rsrs": False, "use_ma_filter": False, "use_reversal_filter": True},
    "E_动量+RSRS+双均线": {"use_rsrs": True, "use_ma_filter": True, "use_reversal_filter": False},
    "F_全策略": {"use_rsrs": True, "use_ma_filter": True, "use_reversal_filter": True},
}


def setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def run_ablation_analysis(close_df, ohlcv_dict, params):
    """运行消融分析：6种组合对比"""
    results = {}

    for name, config in ABLATION_CONFIGS.items():
        print(f"\n  [{name}] 配置: {config}")

        # 计算综合评分
        score, momentum, rsrs, ma_filter, reversal_filter = calculate_composite_score(
            close_df, ohlcv_dict,
            momentum_window=params["momentum_window"],
            rsrs_window=params["rsrs_window"],
            ma_short=params["ma_short"],
            ma_long=params["ma_long"],
            **config,
        )

        # 生成信号
        signal = generate_signal_v2(
            close_df, score,
            rebalance_freq=params["rebalance_freq"],
            hold_count=params["hold_count"],
            empty_threshold=params["empty_threshold"],
        )

        # 回测
        bt_result = run_backtest(close_df, signal, cost=params["cost"])
        metrics = calc_metrics(bt_result["returns"])

        # 持仓统计
        hold_pct = {}
        for col in signal.columns:
            hold_pct[col] = (signal[col] > 0).sum() / len(signal) * 100
        empty_pct = (signal.sum(axis=1) == 0).sum() / len(signal) * 100
        turnover_total = bt_result["turnover"].sum()

        results[name] = {
            "metrics": metrics,
            "equity": bt_result["equity"],
            "signal": signal,
            "hold_pct": hold_pct,
            "empty_pct": empty_pct,
            "turnover_total": turnover_total,
            "config": config,
        }

        print(f"    年化={metrics['annual_return']*100:.2f}%, "
              f"Sharpe={metrics['sharpe']:.3f}, "
              f"回撤={metrics['max_drawdown']*100:.2f}%, "
              f"空仓={empty_pct:.1f}%, 换手={turnover_total:.1f}")

    return results


def generate_ablation_report(close_df, ablation_results, benchmark_metrics, params, output_dir):
    """生成消融分析对比报告"""
    output_dir.mkdir(parents=True, exist_ok=True)
    plt = setup_matplotlib()

    # ===== 图1: 资金曲线对比 =====
    fig, ax = plt.subplots(figsize=(14, 7))
    colors = ["#e74c3c", "#e67e22", "#f1c40f", "#2ecc71", "#3498db", "#9b59b6"]

    for i, (name, result) in enumerate(ablation_results.items()):
        ax.plot(result["equity"].index, result["equity"].values,
                label=name, color=colors[i], linewidth=1.3, alpha=0.85)

    # 等权基准
    bench_ret = close_df.pct_change().fillna(0).mean(axis=1)
    bench_equity = (1 + bench_ret).cumprod()
    ax.plot(bench_equity.index, bench_equity.values,
            label="等权基准", color="gray", linewidth=1, linestyle="--", alpha=0.7)

    ax.axhline(1.0, color="black", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.set_title("阶段2消融分析：各组合资金曲线对比", fontsize=13)
    ax.set_ylabel("净值")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "01_ablation_equity.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # ===== 图2: 指标对比柱状图 =====
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    names = list(ablation_results.keys()) + ["等权基准"]
    annual_returns = [r["metrics"]["annual_return"] * 100 for r in ablation_results.values()] + [benchmark_metrics["annual_return"] * 100]
    sharpes = [r["metrics"]["sharpe"] for r in ablation_results.values()] + [benchmark_metrics["sharpe"]]
    max_dds = [r["metrics"]["max_drawdown"] * 100 for r in ablation_results.values()] + [benchmark_metrics["max_drawdown"] * 100]
    calmars = [r["metrics"]["calmar"] for r in ablation_results.values()] + [benchmark_metrics["calmar"]]

    short_names = [n.split("_")[0] for n in names]

    axes[0, 0].bar(range(len(names)), annual_returns, color=colors[:6] + ["gray"])
    axes[0, 0].set_xticks(range(len(names)))
    axes[0, 0].set_xticklabels(short_names, rotation=45, ha="right", fontsize=9)
    axes[0, 0].set_title("年化收益率 (%)", fontsize=12)
    axes[0, 0].axhline(0, color="black", linewidth=0.8)
    axes[0, 0].grid(True, axis="y", alpha=0.3)

    axes[0, 1].bar(range(len(names)), sharpes, color=colors[:6] + ["gray"])
    axes[0, 1].set_xticks(range(len(names)))
    axes[0, 1].set_xticklabels(short_names, rotation=45, ha="right", fontsize=9)
    axes[0, 1].set_title("Sharpe比率", fontsize=12)
    axes[0, 1].axhline(0, color="black", linewidth=0.8)
    axes[0, 1].grid(True, axis="y", alpha=0.3)

    axes[1, 0].bar(range(len(names)), max_dds, color=colors[:6] + ["gray"])
    axes[1, 0].set_xticks(range(len(names)))
    axes[1, 0].set_xticklabels(short_names, rotation=45, ha="right", fontsize=9)
    axes[1, 0].set_title("最大回撤 (%)", fontsize=12)
    axes[1, 0].grid(True, axis="y", alpha=0.3)

    axes[1, 1].bar(range(len(names)), calmars, color=colors[:6] + ["gray"])
    axes[1, 1].set_xticks(range(len(names)))
    axes[1, 1].set_xticklabels(short_names, rotation=45, ha="right", fontsize=9)
    axes[1, 1].set_title("Calmar比率", fontsize=12)
    axes[1, 1].axhline(0, color="black", linewidth=0.8)
    axes[1, 1].grid(True, axis="y", alpha=0.3)

    fig.suptitle("阶段2消融分析：各组合指标对比", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / "02_ablation_metrics.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # ===== 图3: 全策略F的持仓+回撤 =====
    full_result = ablation_results["F_全策略"]
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [3, 1]},
                              constrained_layout=True)

    ax1 = axes[0]
    ax1.plot(full_result["equity"].index, full_result["equity"].values,
             label="全策略F", color="#9b59b6", linewidth=1.5)
    ax1.plot(bench_equity.index, bench_equity.values,
             label="等权基准", color="gray", linewidth=1, linestyle="--", alpha=0.7)
    ax1.axhline(1.0, color="black", linestyle=":", linewidth=0.8, alpha=0.5)

    # 回撤阴影
    rolling_max = full_result["equity"].cummax()
    drawdown = (full_result["equity"] - rolling_max) / rolling_max
    ax1.fill_between(full_result["equity"].index, 1.0, full_result["equity"].values,
                     where=(drawdown < 0), color="red", alpha=0.15)
    ax1.set_title(f"阶段2全策略F资金曲线 (年化={full_result['metrics']['annual_return']*100:.2f}%, "
                  f"Sharpe={full_result['metrics']['sharpe']:.3f}, "
                  f"回撤={full_result['metrics']['max_drawdown']*100:.2f}%)", fontsize=12)
    ax1.set_ylabel("净值")
    ax1.legend(loc="best")
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.fill_between(drawdown.index, drawdown.values * 100, 0, color="red", alpha=0.4)
    ax2.set_ylabel("回撤 (%)")
    ax2.set_xlabel("日期")
    ax2.grid(True, alpha=0.3)

    fig.savefig(output_dir / "03_full_strategy_equity.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # ===== HTML报告 =====
    generate_html_report(close_df, ablation_results, benchmark_metrics, params, output_dir)


def generate_html_report(close_df, ablation_results, benchmark_metrics, params, output_dir):
    """生成HTML对比报告"""
    # 指标对比表
    rows = []
    for name, result in ablation_results.items():
        m = result["metrics"]
        rows.append({
            "组合": name,
            "年化收益": f"{m['annual_return']*100:.2f}%",
            "年化波动": f"{m['annual_volatility']*100:.2f}%",
            "Sharpe": f"{m['sharpe']:.3f}",
            "最大回撤": f"{m['max_drawdown']*100:.2f}%",
            "Calmar": f"{m['calmar']:.3f}",
            "胜率": f"{m['win_rate']*100:.1f}%",
            "空仓占比": f"{result['empty_pct']:.1f}%",
            "总换手": f"{result['turnover_total']:.1f}",
        })
    rows.append({
        "组合": "等权基准",
        "年化收益": f"{benchmark_metrics['annual_return']*100:.2f}%",
        "年化波动": f"{benchmark_metrics['annual_volatility']*100:.2f}%",
        "Sharpe": f"{benchmark_metrics['sharpe']:.3f}",
        "最大回撤": f"{benchmark_metrics['max_drawdown']*100:.2f}%",
        "Calmar": f"{benchmark_metrics['calmar']:.3f}",
        "胜率": f"{benchmark_metrics['win_rate']*100:.1f}%",
        "空仓占比": "0.0%",
        "总换手": "—",
    })
    compare_df = pd.DataFrame(rows)

    # 全策略F的持仓分布
    full_result = ablation_results["F_全策略"]
    hold_dist = pd.DataFrame({
        "ETF": list(full_result["hold_pct"].keys()),
        "持仓天数占比(%)": [f"{v:.1f}" for v in full_result["hold_pct"].values()],
    })

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>ETF轮动策略阶段2消融分析报告</title>
<style>
body {{ font-family: "Microsoft YaHei", sans-serif; margin: 20px; background: #f5f5f5; }}
h1 {{ color: #2c3e50; border-bottom: 3px solid #9b59b6; padding-bottom: 10px; }}
h2 {{ color: #34495e; margin-top: 30px; }}
table {{ border-collapse: collapse; width: 100%; background: white; margin: 10px 0; font-size: 12px; }}
th, td {{ border: 1px solid #ddd; padding: 6px 8px; text-align: center; }}
th {{ background: #9b59b6; color: white; }}
tr:nth-child(even) {{ background: #f9f9f9; }}
tr:hover {{ background: #f3e5f5; }}
img {{ max-width: 100%; border: 1px solid #ddd; margin: 10px 0; }}
.box {{ background: white; padding: 15px; border-left: 4px solid #9b59b6; margin: 15px 0; }}
.note {{ background: #fff9c4; padding: 10px; border-radius: 4px; margin: 10px 0; }}
.params {{ background: #e8f5e9; padding: 10px; border-radius: 4px; font-family: monospace; font-size: 12px; }}
</style>
</head>
<body>
<h1>ETF轮动策略阶段2消融分析报告</h1>
<p>生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>

<div class="box">
<b>阶段2目标</b>：验证回归动量+RSRS+双均线+反转风控的改进效果，通过消融分析定位各模块贡献。<br>
<b>数据</b>：10个模拟ETF（5类资产×2风格），5年日线OHLCV数据。<br>
<b>方法</b>：6种组合对比（纯动量 → 逐步加模块 → 全策略），等权基准参照。
</div>

<div class="params">
<b>策略参数</b><br>
  动量窗口: {params['momentum_window']}日（回归动量，年化×R²）<br>
  RSRS窗口: {params['rsrs_window']}日（High/Low回归斜率×R²）<br>
  双均线: MA{params['ma_short']}/MA{params['ma_long']}（价>MA短且MA短>MA长）<br>
  调仓频率: {params['rebalance_freq']}（月频）<br>
  持仓数量: {params['hold_count']}只（等权）<br>
  交易成本: 单边0.15%<br>
  信号处理: shift(1)后移一天
</div>

<h2>一、消融分析指标对比</h2>
{compare_df.to_html(index=False, escape=False)}

<div class="note">
<b>消融分析解读要点</b>：<br>
1. A→B（加RSRS）：Sharpe是否提升？（知识库实证：0.21→1.33）<br>
2. A→C（加双均线）：回撤是否降低？（知识库：回撤-45%→-34%）<br>
3. A→D（加反转风控）：胜率是否提升？<br>
4. E→F（加反转风控）：换手是否降低？<br>
5. F vs 等权基准：全策略是否有超额？
</div>

<h2>二、资金曲线对比</h2>
<img src="01_ablation_equity.png" alt="资金曲线对比">
<p>说明：6种组合+等权基准的资金曲线，紫色为全策略F。</p>

<h2>三、指标对比柱状图</h2>
<img src="02_ablation_metrics.png" alt="指标对比">
<p>说明：4个子图分别为年化收益、Sharpe、最大回撤、Calmar。灰色为等权基准。</p>

<h2>四、全策略F资金曲线与回撤</h2>
<img src="03_full_strategy_equity.png" alt="全策略F">
<p>说明：上图紫色为全策略F净值，灰虚线为等权基准，红色阴影为回撤区间；下图为回撤曲线。</p>

<h2>五、全策略F持仓分布</h2>
{hold_dist.to_html(index=False, escape=False)}
<p>说明：全策略F各ETF持仓天数占比，空仓占比={full_result['empty_pct']:.1f}%。</p>

<h2>六、阶段1 vs 阶段2对比</h2>
<div class="box">
<b>阶段1</b>（3ETF + ROC60 + 月频 + 1只持仓）：年化-7.32%, Sharpe-0.34, 回撤-47.09%<br>
<b>阶段2全策略F</b>（10ETF + 回归动量25日 + RSRS + 双均线 + 反转风控 + 月频 + 2只持仓）：
年化{full_result['metrics']['annual_return']*100:.2f}%, Sharpe{full_result['metrics']['sharpe']:.3f}, 
回撤{full_result['metrics']['max_drawdown']*100:.2f}%<br>
<b>等权基准</b>：年化{benchmark_metrics['annual_return']*100:.2f}%, Sharpe{benchmark_metrics['sharpe']:.3f}, 
回撤{benchmark_metrics['max_drawdown']*100:.2f}%
</div>

</body>
</html>
"""
    html_path = output_dir / "report.html"
    html_path.write_text(html, encoding="utf-8")
    return html_path


def main():
    print("=" * 60)
    print("ETF轮动策略 阶段2：10个ETF + 多因子 + 消融分析")
    print("=" * 60)

    # 1. 生成数据
    print(f"\n[1/4] 生成{STRATEGY_PARAMS['n_years']}年模拟数据（10个ETF）...")
    close_df, ohlcv_dict, _ = generate_simulation_data_v2(
        n_years=STRATEGY_PARAMS["n_years"], seed=STRATEGY_PARAMS["seed"],
    )
    print(f"  数据: {close_df.shape[0]}交易日, {close_df.shape[1]}ETF")

    # 各ETF统计
    print(f"\n  各ETF年化收益:")
    for code in close_df.columns:
        close = close_df[code]
        ret = (close.iloc[-1] / close.iloc[0]) ** (252 / len(close)) - 1
        print(f"    {code}: {ret*100:.2f}%")

    # 2. 等权基准
    print(f"\n[2/4] 计算等权基准...")
    benchmark_metrics = calc_benchmark_metrics(close_df)
    print(f"  等权基准: 年化={benchmark_metrics['annual_return']*100:.2f}%, "
          f"Sharpe={benchmark_metrics['sharpe']:.3f}, "
          f"回撤={benchmark_metrics['max_drawdown']*100:.2f}%")

    # 3. 消融分析
    print(f"\n[3/4] 运行消融分析（6种组合）...")
    ablation_results = run_ablation_analysis(close_df, ohlcv_dict, STRATEGY_PARAMS)

    # 4. 生成报告
    print(f"\n[4/4] 生成报告...")
    output_dir = Path(__file__).resolve().parent.parent.parent.parent.parent / "output" / "research" / "v2_10etf"
    generate_ablation_report(close_df, ablation_results, benchmark_metrics, STRATEGY_PARAMS, output_dir)
    print(f"  输出目录: {output_dir}")

    # 汇总
    print(f"\n{'='*60}")
    print(f"阶段2消融分析完成！")
    print(f"{'='*60}")
    print(f"\n=== 各组合绩效汇总 ===")
    print(f"{'组合':<20s} {'年化':>8s} {'Sharpe':>8s} {'回撤':>8s} {'Calmar':>8s} {'空仓%':>6s}")
    print("-" * 60)
    for name, result in ablation_results.items():
        m = result["metrics"]
        print(f"{name:<20s} {m['annual_return']*100:>7.2f}% {m['sharpe']:>8.3f} "
              f"{m['max_drawdown']*100:>7.2f}% {m['calmar']:>8.3f} {result['empty_pct']:>5.1f}%")
    print(f"{'等权基准':<20s} {benchmark_metrics['annual_return']*100:>7.2f}% "
          f"{benchmark_metrics['sharpe']:>8.3f} {benchmark_metrics['max_drawdown']*100:>7.2f}% "
          f"{benchmark_metrics['calmar']:>8.3f} {'0.0%':>6s}")

    # 阶段1对比
    full = ablation_results["F_全策略"]["metrics"]
    print(f"\n=== 阶段1 vs 阶段2(全策略F) ===")
    print(f"  阶段1: 年化=-7.32%, Sharpe=-0.336, 回撤=-47.09%")
    print(f"  阶段2: 年化={full['annual_return']*100:.2f}%, Sharpe={full['sharpe']:.3f}, 回撤={full['max_drawdown']*100:.2f}%")
    print(f"  改善:  年化+{(full['annual_return']+0.0732)*100:.2f}%, "
          f"Sharpe+{full['sharpe']+0.336:.3f}, "
          f"回撤+{(full['max_drawdown']+0.4709)*100:.2f}%")


if __name__ == "__main__":
    main()
