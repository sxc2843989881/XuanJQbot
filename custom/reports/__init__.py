"""custom.reports - 通用回测报告生成模块

与策略解耦的报告生成流水线。任何策略只要按数据契约（BacktestData）传入
回测结果，即可生成基础版（4合1 PNG + 简版 HTML）和详细版（扩展图 +
完整 HTML）报告。

核心组件：
- contract: 数据契约（BacktestData / TradeRecord / BacktestMetrics）
- metrics: 通用指标计算（含完整周期最大回撤、滚动 Sharpe 等）
- plotter: 基础版 4合1 matplotlib 绘图
- detailed_plotter: 详细版扩展图（月度热力图/滚动 Sharpe/回撤期/收益分布）
- html_reporter: HTML 报告生成
- pipeline: 报告生成流水线编排
- factor_plugins: 因子子图可插拔插件

典型用法：
    from custom.reports import BacktestData, BacktestMetrics, ReportPipeline
    from custom.reports.factor_plugins.sma_plugin import SmaFactorPlugin

    data = BacktestData(dates=..., closes=..., values=..., positions=...,
                        trades=..., strategy_name="SMA", code="sh.600519")
    metrics = BacktestMetrics(...)
    pipeline = ReportPipeline(data, metrics, factor_plugin=SmaFactorPlugin(pfast=10, pslow=30))
    result = pipeline.generate(output_dir, name_prefix="sma_cross_sh_600519")
    # result.chart_path / result.html_path / result.detailed_chart_path / result.detailed_html_path
"""
from custom.reports.contract import BacktestData, BacktestMetrics, TradeRecord, ReportResult
from custom.reports.metrics import calc_max_drawdown, calc_metrics_from_equity
from custom.reports.pipeline import ReportPipeline

__all__ = [
    "BacktestData",
    "BacktestMetrics",
    "TradeRecord",
    "ReportResult",
    "ReportPipeline",
    "calc_max_drawdown",
    "calc_metrics_from_equity",
]
