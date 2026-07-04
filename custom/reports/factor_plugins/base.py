"""custom.reports.factor_plugins.base - 因子插件抽象基类

设计目标：
- 解耦"策略特定的因子计算"与"通用绘图层"
- 策略只需实现本接口，绘图层即可渲染因子子图
- 插件可同时填充 BacktestData.factors / signals / cross_points

接口契约：
- prepare(data: BacktestData) -> BacktestData:
    接收原始 BacktestData（可能 factors/signals 为空），
    由插件计算并填充因子序列、信号、金叉死叉点，返回更新后的 data。
- factor_plot_specs() -> List[FactorPlotSpec]:
    返回因子子图中需要绘制的因子线规格（名称、颜色、线宽、标签）。
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

from custom.reports.contract import BacktestData


@dataclass
class FactorPlotSpec:
    """单条因子线的绘图规格"""
    name: str               # 因子名（与 BacktestData.factors 的 key 对应）
    label: str              # 图例标签
    color: str              # 颜色（hex 或 matplotlib 颜色名）
    linewidth: float = 1.3
    zorder: int = 3


class FactorPlugin(ABC):
    """因子子图插件抽象基类

    子类必须实现：
    - prepare(data): 计算并填充 data.factors / data.signals / data.cross_points
    - factor_plot_specs(): 返回要在因子子图绘制的因子线规格列表
    - plugin_name(): 返回插件名（用于报告标题）

    可选实现：
    - signal_yticklabels(): 因子信号子图的 y 轴标签（默认 ["空头(-1)", "空仓(0)", "多头(+1)"]）
    """

    @abstractmethod
    def plugin_name(self) -> str:
        """插件名（如 "SMA" / "MACD"）"""
        ...

    @abstractmethod
    def prepare(self, data: BacktestData) -> BacktestData:
        """计算因子序列、信号、金叉死叉点，填充到 data 并返回

        实现要点：
        - 若 data.factors/signals/cross_points 已由策略填充，可跳过计算
        - 否则用 data.closes 计算 SMA/MACD/RSI 等，填入 data.factors
        - 信号序列：1=多头 / 0=空仓 / -1=空头
        - cross_points: (金叉日期列表, 死叉日期列表)
        """
        ...

    @abstractmethod
    def factor_plot_specs(self) -> List[FactorPlotSpec]:
        """返回因子子图要绘制的因子线规格"""
        ...

    def signal_yticklabels(self) -> Optional[List[str]]:
        """因子信号子图 y 轴标签（默认 None 用通用标签）"""
        return None

    def factor_subtitle(self) -> str:
        """因子子图副标题（默认空，子类可覆盖）"""
        return ""
