"""custom.reports.pipeline - 报告生成流水线编排

将数据契约 → 指标计算 → 因子插件填充 → 绘图 → HTML 报告 串成完整流水线。
支持生成基础版和详细版报告。
"""
from datetime import datetime
from pathlib import Path
from typing import Optional

from custom.reports.contract import BacktestData, BacktestMetrics, ReportResult
from custom.reports.factor_plugins.base import FactorPlugin
from custom.reports.plotter import plot_basic_report
from custom.reports.detailed_plotter import plot_detailed_report
from custom.reports.html_reporter import generate_basic_html_report, generate_detailed_html_report


class ReportPipeline:
    """报告生成流水线

    用法：
        pipeline = ReportPipeline(data, metrics, factor_plugin=SmaFactorPlugin(10, 30))
        result = pipeline.generate(output_dir, name_prefix="sma_cross_sh_600519")
    """

    def __init__(self,
                 data: BacktestData,
                 metrics: BacktestMetrics,
                 factor_plugin: Optional[FactorPlugin] = None):
        self.data = data
        self.metrics = metrics
        self.factor_plugin = factor_plugin

    def generate(self,
                 output_dir,
                 name_prefix: str = "backtest",
                 include_basic: bool = True,
                 include_detailed: bool = True) -> ReportResult:
        """生成报告

        Args:
            output_dir: 输出目录（Path 或 str）
            name_prefix: 文件名前缀（如 "sma_cross_sh_600519"）
            include_basic: 是否生成基础版
            include_detailed: 是否生成详细版

        Returns:
            ReportResult: 包含所有生成文件路径
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 校验数据契约
        self.data.validate()

        result = ReportResult(generated_at=datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

        # ===== 基础版 =====
        if include_basic:
            chart_path = output_dir / f'{name_prefix}_chart.png'
            html_path = output_dir / f'{name_prefix}_report.html'

            print(f"[报告生成] 基础版 4合1 图表...")
            plot_basic_report(self.data, self.metrics, self.factor_plugin, chart_path)

            print(f"[报告生成] 基础版 HTML 报告...")
            generate_basic_html_report(self.data, self.metrics, chart_path, html_path)

            result.chart_path = str(chart_path)
            result.html_path = str(html_path)

        # ===== 详细版 =====
        if include_detailed:
            detailed_chart_path = output_dir / f'{name_prefix}_detailed_chart.png'
            detailed_html_path = output_dir / f'{name_prefix}_detailed_report.html'

            print(f"[报告生成] 详细版扩展分析图表...")
            plot_detailed_report(self.data, self.metrics, detailed_chart_path)

            # 详细版 HTML 需要基础版图表，若未生成则先补一张
            if not include_basic:
                basic_chart_for_detailed = output_dir / f'{name_prefix}_chart.png'
                print(f"[报告生成] 为详细版生成基础 4合1 图表...")
                plot_basic_report(self.data, self.metrics, self.factor_plugin,
                                  basic_chart_for_detailed)
            else:
                basic_chart_for_detailed = chart_path

            print(f"[报告生成] 详细版 HTML 报告...")
            generate_detailed_html_report(self.data, self.metrics,
                                          basic_chart_for_detailed,
                                          detailed_chart_path,
                                          detailed_html_path)

            result.detailed_chart_path = str(detailed_chart_path)
            result.detailed_html_path = str(detailed_html_path)

        return result
