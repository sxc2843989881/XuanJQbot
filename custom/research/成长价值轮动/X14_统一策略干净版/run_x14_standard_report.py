"""用标准报告模块跑 X14 回测分析"""
import sys
sys.path.insert(0, r'c:\XuanJLH\Qbot\custom\backtests')
sys.path.insert(0, r'c:\XuanJLH\Qbot\custom\research\成长价值轮动\X12_偏离度空仓版')
sys.path.insert(0, r'c:\XuanJLH\Qbot\custom\research\成长价值轮动\X14_统一策略干净版')

from pathlib import Path
from backtest_x14_engine import build_core
from optimize_runner import run_backtest, calc_metrics, count_switches
from backtest_report import BacktestReport

OUTPUT = Path(r'c:\XuanJLH\Qbot\custom\research\成长价值轮动\X14_统一策略干净版\标准报告输出')
OUTPUT.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("  X14 标准回测报告生成")
print("=" * 80)

# 1. 跑回测
print("\n运行回测...")
sig, wt = build_core(bias_mode='clear')
result = run_backtest(sig, wt, impact_slippage=0.0005)
m = calc_metrics(result)
sw = count_switches(sig, wt)

print(f"  年化={m['ann']*100:.2f}%  Calmar={m['calmar']:.3f}  交易={m['n_trades']}")

# 2. 生成报告
print("\n生成报告...")
report = BacktestReport(
    output_dir=str(OUTPUT),
    title='X14 统一策略干净版 — 完整回测分析'
)
report.add_from_backtest('X14', sig, wt, result, m, sw)

paths = report.generate_all(skip_slippage=True)  # 跳过滑点敏感性(需import specific)
big = report.merge_big_chart()

print("\n" + "=" * 80)
print("  完成! 输出目录:", OUTPUT)
print("=" * 80)

# 列出所有生成文件
for f in sorted(OUTPUT.iterdir()):
    size = f.stat().st_size
    print(f"  {f.name}  ({size/1024:.1f} KB)")
