"""custom.reports.contract - 通用回测报告模块的数据契约

定义与策略解耦的数据结构。任何策略（SMA/MACD/RSI/Boll 等）回测后，
按本文件定义的 dataclass 组装数据，即可调用 ReportPipeline 生成报告。

设计原则：
1. 字段全部为基础 Python 类型或 pandas/numpy 对象，不依赖任何具体策略
2. factors/signals 是可选字段，策略可自定义填充
3. 通过 FactorPlugin 插件机制解耦因子子图绘制（详见 factor_plugins/base.py）
"""
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


@dataclass
class TradeRecord:
    """单笔交易记录（与策略解耦）"""
    date: Any              # date 或 datetime
    type: str              # "BUY" / "SELL" / 其他自定义
    price: float
    size: float            # 交易数量（正数）
    commission: float = 0.0  # 手续费（可选）


@dataclass
class BacktestData:
    """回测时间序列数据契约

    所有 Series 的 index 必须是同一 DatetimeIndex（dates 的副本或同长度）。
    """
    # 必填：时间序列（长度必须一致）
    dates: pd.DatetimeIndex
    closes: pd.Series         # 收盘价序列
    values: pd.Series         # 资产净值序列（每日）
    positions: pd.Series      # 持仓数量序列（每日）

    # 必填：交易记录
    trades: List[TradeRecord]

    # 策略元信息
    strategy_name: str = ""          # 策略名（如 "SMA" / "MACD"）
    strategy_params: Dict[str, Any] = field(default_factory=dict)  # 策略参数
    code: str = ""                   # 标的代码

    # 可选：策略自定义因子序列（用于因子子图）
    # key=因子名（如 "SMA10"），value=与 dates 同长度的 Series
    factors: Dict[str, pd.Series] = field(default_factory=dict)

    # 可选：因子信号序列（1=多头 / 0=空仓 / -1=空头），用于因子信号子图
    signals: Optional[pd.Series] = None

    # 可选：金叉/死叉点（因子信号变化点），由 FactorPlugin 计算或策略直接提供
    # (金叉日期列表, 死叉日期列表)
    cross_points: Optional[Tuple[List[Any], List[Any]]] = None

    def validate(self) -> None:
        """校验数据契约一致性"""
        n = len(self.dates)
        if len(self.closes) != n:
            raise ValueError(f"closes 长度 {len(self.closes)} != dates 长度 {n}")
        if len(self.values) != n:
            raise ValueError(f"values 长度 {len(self.values)} != dates 长度 {n}")
        if len(self.positions) != n:
            raise ValueError(f"positions 长度 {len(self.positions)} != dates 长度 {n}")
        for name, s in self.factors.items():
            if len(s) != n:
                raise ValueError(f"factor '{name}' 长度 {len(s)} != dates 长度 {n}")
        if self.signals is not None and len(self.signals) != n:
            raise ValueError(f"signals 长度 {len(self.signals)} != dates 长度 {n}")


@dataclass
class BacktestMetrics:
    """回测指标摘要（与策略解耦）

    所有指标在 metrics.py 中由 BacktestData + analyzers 输出计算得出。
    收益率/回撤使用小数表示（如 0.15 表示 15%，-0.12 表示 -12%）。
    """
    code: str
    start: str               # YYYY-MM-DD
    end: str
    days: int                # 实际天数
    cash: float              # 初始资金
    final_value: float       # 最终资金

    # 收益指标
    total_return: float
    annual_return: float

    # 风险指标
    max_drawdown: float      # 完整周期最大回撤（负数小数）
    sharpe: Optional[float]  # Sharpe Ratio（可能为 None）
    calmar: float

    # 交易指标
    total_trades: int
    won_trades: int
    lost_trades: int
    win_rate: float          # 百分比 0-100

    # 策略元信息（用于报告展示）
    strategy_name: str = ""
    strategy_params: Dict[str, Any] = field(default_factory=dict)

    # 可选扩展指标（详细版报告用）
    sortino: Optional[float] = None
    volatility: Optional[float] = None     # 年化波动率
    max_dd_duration: Optional[int] = None  # 最大回撤持续天数


@dataclass
class ReportResult:
    """报告生成结果（pipeline.generate 返回）"""
    # 基础版
    chart_path: Optional[str] = None       # 4合1 PNG
    html_path: Optional[str] = None        # 基础版 HTML

    # 详细版
    detailed_chart_path: Optional[str] = None  # 扩展图 PNG
    detailed_html_path: Optional[str] = None   # 详细版 HTML

    # 生成时间戳
    generated_at: Optional[str] = None

    def __repr__(self) -> str:
        parts = ["ReportResult("]
        if self.chart_path:
            parts.append(f"  chart={self.chart_path}")
        if self.html_path:
            parts.append(f"  html={self.html_path}")
        if self.detailed_chart_path:
            parts.append(f"  detailed_chart={self.detailed_chart_path}")
        if self.detailed_html_path:
            parts.append(f"  detailed_html={self.detailed_html_path}")
        parts.append(")")
        return "\n".join(parts)
