"""验证框架统一入口

跑5项验证：
1. 公平基准族（蒙特卡洛500次）
2. Walk-Forward 滚动样本外
3. 参数敏感性（plateau vs spike）
4. 滑点压力测试
5. 随机扰动测试（80次）

输出：
- custom/output/research/v2_10etf/validation_report.html
- custom/output/research/v2_10etf/validation_result.json

使用方法：
    python run_validation.py
    python run_validation.py --quick  # 快速版（蒙特卡洛100次，扰动20次）
"""
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

# 路径设置
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # validation 目录自身（让子模块的绝对导入可工作）
sys.path.insert(0, str(HERE.parent))  # v2_10etf 目录
sys.path.insert(0, str(HERE.parent.parent / "v1_3etf"))  # v1_3etf 目录 (for backtest.py)

from data_generator_v2 import generate_simulation_data_v2
from factors_v2 import calculate_composite_score, generate_signal_v2
from backtest import run_backtest, calc_metrics as bt_calc_metrics

from benchmark_family import run_all_benchmarks
from walk_forward import run_walk_forward
from param_sensitivity import run_param_sensitivity
from slippage_stress import run_slippage_stress
from random_perturbation import run_random_perturbation


# ============================================================
# 策略统一接口
# ============================================================

def strategy_unified(close, ohlcv_dict,
                     momentum_window=25, rsrs_window=18,
                     ma_short=20, ma_long=60,
                     use_rsrs=True, use_ma_filter=True, use_reversal_filter=True,
                     rebalance_freq="M", hold_count=2,
                     empty_threshold=0.0, cost=0.0015):
    """统一策略接口

    签名: (close, ohlcv_dict, **kwargs) -> (equity, metrics, signal)
    """
    score, _, _, _, _ = calculate_composite_score(
        close, ohlcv_dict,
        momentum_window=momentum_window,
        rsrs_window=rsrs_window,
        ma_short=ma_short,
        ma_long=ma_long,
        use_rsrs=use_rsrs,
        use_ma_filter=use_ma_filter,
        use_reversal_filter=use_reversal_filter,
    )
    signal = generate_signal_v2(
        close, score,
        rebalance_freq=rebalance_freq,
        hold_count=hold_count,
        empty_threshold=empty_threshold,
    )
    bt = run_backtest(close, signal, cost=cost)
    metrics = bt_calc_metrics(bt["returns"])
    return bt["equity"], metrics, signal


# ============================================================
# 主流程
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="快速版（减少蒙特卡洛/扰动次数）")
    parser.add_argument("--n_years", type=int, default=5, help="模拟数据年数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    n_mc = 100 if args.quick else 500
    n_perturb = 20 if args.quick else 80

    print("=" * 70)
    print("ETF轮动策略阶段2 — 5项验证框架")
    print("=" * 70)
    print(f"配置: n_years={args.n_years}, seed={args.seed}, "
          f"蒙特卡洛{n_mc}次, 随机扰动{n_perturb}次")
    print()

    # 1. 加载数据
    print("[1/6] 生成模拟数据...")
    close_df, ohlcv_dict, _ = generate_simulation_data_v2(
        n_years=args.n_years, seed=args.seed
    )
    print(f"  数据: {close_df.shape[0]}交易日, {close_df.shape[1]}ETF")

    # 2. 跑原始策略获取baseline
    print("\n[2/6] 跑原始策略获取baseline...")
    equity, metrics, signal = strategy_unified(close_df, ohlcv_dict)
    print(f"  原始策略: 年化={metrics['annual_return']*100:.2f}%, "
          f"Sharpe={metrics['sharpe']:.3f}, 回撤={metrics['max_drawdown']*100:.2f}%")

    # 3. 验证1：公平基准族
    print("\n[3/6] 验证1: 公平基准族")
    benchmark_result = run_all_benchmarks(close_df, metrics, n_mc_runs=n_mc)
    _print_verdict("公平基准族", benchmark_result["verdict"])

    # 4. 验证2：Walk-Forward
    print("\n[4/6] 验证2: Walk-Forward 滚动样本外")
    wf_result = run_walk_forward(
        close_df, ohlcv_dict, strategy_unified,
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
        strategy_unified, close_df, ohlcv_dict, params_to_scan,
        fixed_kwargs={"use_rsrs": True, "use_ma_filter": True, "use_reversal_filter": True}
    )
    _print_verdict("参数敏感性", ps_result["verdict"])

    # 6. 验证4：滑点压力测试
    print("\n[6/6] 验证4 & 5: 滑点压力 + 随机扰动")
    slip_result = run_slippage_stress(
        strategy_unified, close_df, ohlcv_dict,
        base_kwargs={"use_rsrs": True, "use_ma_filter": True, "use_reversal_filter": True}
    )
    _print_verdict("滑点压力", slip_result["verdict"])

    # 7. 验证5：随机扰动
    print("\n  随机扰动测试...")
    perturb_result = run_random_perturbation(
        strategy_unified, close_df, ohlcv_dict,
        base_kwargs={"use_rsrs": True, "use_ma_filter": True, "use_reversal_filter": True},
        n_runs=n_perturb,
    )
    if "verdict" in perturb_result:
        _print_verdict("随机扰动", perturb_result["verdict"])

    # 8. 汇总
    print("\n" + "=" * 70)
    print("5项验证汇总")
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
        print("最终判定: 策略通过全部5项验证 — alpha显著且稳健")
    else:
        print(f"最终判定: 策略未通过全部验证 ({n_pass}/5) — 不可进入阶段3")
    print(f"{'='*70}")

    # 9. 输出JSON结果
    output_dir = HERE.parent.parent.parent.parent.parent / "output" / "research" / "v2_10etf"
    output_dir.mkdir(parents=True, exist_ok=True)

    result_json = {
        "timestamp": datetime.now().isoformat(),
        "config": {"n_years": args.n_years, "seed": args.seed,
                   "n_mc": n_mc, "n_perturb": n_perturb},
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

    json_path = output_dir / "validation_result.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result_json, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n结果已保存: {json_path}")

    # 10. 生成HTML报告
    html_path = output_dir / "validation_report.html"
    _generate_html_report(html_path, result_json)
    print(f"HTML报告: {html_path}")

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


def _generate_html_report(html_path, result):
    """生成简洁HTML报告"""
    summary_class = "summary-pass" if result["overall_pass"] else "summary-fail"
    if result["overall_pass"]:
        summary_text = "✓ 通过全部5项验证"
    else:
        summary_text = f"✗ 未通过全部验证 ({result['n_pass']}/5)"

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>ETF轮动策略阶段2 验证报告</title>
<style>
body {{ font-family: "Microsoft YaHei", sans-serif; max-width: 1100px; margin: 0 auto; padding: 20px; background: #f8f9fa; }}
h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
h2 {{ color: #34495e; margin-top: 30px; }}
.card {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin: 15px 0; }}
.metric {{ display: inline-block; min-width: 180px; margin: 5px 15px 5px 0; }}
.metric-label {{ color: #7f8c8d; font-size: 13px; }}
.metric-value {{ font-size: 18px; font-weight: bold; color: #2c3e50; }}
.pass {{ color: #27ae60; }}
.fail {{ color: #e74c3c; }}
.check-list {{ list-style: none; padding-left: 0; }}
.check-list li {{ padding: 5px 0; border-bottom: 1px solid #ecf0f1; }}
table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
th {{ background: #3498db; color: white; }}
tr:nth-child(even) {{ background: #f2f2f2; }}
.summary-box {{ padding: 20px; border-radius: 8px; text-align: center; font-size: 20px; font-weight: bold; margin: 20px 0; }}
.summary-pass {{ background: #d4edda; color: #155724; border: 2px solid #c3e6cb; }}
.summary-fail {{ background: #f8d7da; color: #721c24; border: 2px solid #f5c6cb; }}
</style>
</head>
<body>
<h1>ETF轮动策略阶段2 — 5项验证报告</h1>
<p style="color:#7f8c8d;">生成时间: {result['timestamp']}</p>

<div class="card">
<h2>策略原始表现</h2>
<div class="metric"><div class="metric-label">年化收益</div><div class="metric-value">{result['strategy_metrics']['annual_return']*100:.2f}%</div></div>
<div class="metric"><div class="metric-label">Sharpe</div><div class="metric-value">{result['strategy_metrics']['sharpe']:.3f}</div></div>
<div class="metric"><div class="metric-label">最大回撤</div><div class="metric-value">{result['strategy_metrics']['max_drawdown']*100:.2f}%</div></div>
<div class="metric"><div class="metric-label">Calmar</div><div class="metric-value">{result['strategy_metrics']['calmar']:.3f}</div></div>
</div>

<div class="summary-box {summary_class}">
{summary_text}
</div>

<h2>5项验证结论</h2>
<table>
<tr><th>验证项</th><th>结论</th><th>是否通过</th></tr>
"""

    for name, v in result["verdicts"].items():
        mark = '<span class="pass">✓ 通过</span>' if v.get("overall_pass") else '<span class="fail">✗ 未通过</span>'
        html += f'<tr><td>{name}</td><td>{v.get("verdict_summary", "N/A")}</td><td>{mark}</td></tr>'

    html += "</table>"

    # 各项详细checks
    html += "<h2>详细判定</h2>"
    for name, v in result["verdicts"].items():
        html += f'<div class="card"><h3>{name}</h3><ul class="check-list">'
        for check in v.get("checks", []):
            html += f"<li>{check}</li>"
        if not v.get("checks"):
            html += f"<li>{v.get('verdict_summary', 'N/A')}</li>"
        html += "</ul></div>"

    html += """
<h2>验证框架说明</h2>
<div class="card">
<p><strong>5项验证流程</strong>（基于 wu.run 黄金标准 + A/B角色共识）：</p>
<ol>
<li><strong>公平基准族</strong>：策略必须打败等权全池 + 动量Top2买入持有 + 随机Top2蒙特卡洛95%分位</li>
<li><strong>Walk-Forward</strong>：滚动样本外，平均Sharpe为正 + 正Sharpe窗口占比≥60% + 变异系数<1.0</li>
<li><strong>参数敏感性</strong>：参数周围±50%扫描，呈plateau而非spike</li>
<li><strong>滑点压力</strong>：滑点0.05%→0.30%衰减率<60%</li>
<li><strong>随机扰动</strong>：保留换手时点随机选股，策略Sharpe须≥95%分位</li>
</ol>
<p style="color:#7f8c8d;">参考：知识库 07_开源项目与社区策略/02_ETF轮动策略实例合集.md (wu.run 5项验证)</p>
</div>
</body>
</html>
"""
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)


if __name__ == "__main__":
    main()
