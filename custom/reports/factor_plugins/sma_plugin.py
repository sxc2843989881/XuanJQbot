"""custom.reports.factor_plugins.sma_plugin - SMA 双均线策略因子插件

参考实现：演示如何为具体策略编写因子插件。
任何 SMA 系策略（SMA 交叉、SMA 趋势等）都可直接复用本插件。
"""
from typing import List, Optional

import pandas as pd

from custom.reports.contract import BacktestData
from custom.reports.factor_plugins.base import FactorPlugin, FactorPlotSpec


class SmaFactorPlugin(FactorPlugin):
    """SMA 双均线因子插件

    计算 SMA 快线/慢线、金叉/死叉信号、买卖点。
    与 custom/factors/factor_sma.py 的逻辑等价，但解耦为独立插件。
    """

    def __init__(self, pfast: int = 10, pslow: int = 30):
        self.pfast = pfast
        self.pslow = pslow

    def plugin_name(self) -> str:
        return "SMA"

    def _calc_sma(self, series: pd.Series, period: int) -> pd.Series:
        """简单移动平均"""
        return series.rolling(window=period, min_periods=1).mean()

    def prepare(self, data: BacktestData) -> BacktestData:
        """计算 SMA 快/慢线、信号、金叉死叉点"""
        # 若策略已填充 factors，跳过计算
        fast_key = f"SMA{self.pfast}"
        slow_key = f"SMA{self.pslow}"

        if fast_key not in data.factors or slow_key not in data.factors:
            sma_fast = self._calc_sma(data.closes, self.pfast)
            sma_slow = self._calc_sma(data.closes, self.pslow)
            data.factors[fast_key] = sma_fast
            data.factors[slow_key] = sma_slow

        # 信号：快线 > 慢线 → +1，快线 < 慢线 → -1，相等 → 0
        if data.signals is None:
            sma_fast = data.factors[fast_key]
            sma_slow = data.factors[slow_key]
            signal = pd.Series(0, index=data.dates, dtype=int)
            signal[sma_fast > sma_slow] = 1
            signal[sma_fast < sma_slow] = -1
            data.signals = signal

        # 金叉/死叉点（信号变化点）
        if data.cross_points is None and data.signals is not None:
            sig_diff = data.signals.diff().fillna(0)
            golden_cross = list(data.dates[sig_diff > 0])    # 0/-1 → +1
            death_cross = list(data.dates[sig_diff < 0])     # +1/0 → -1
            data.cross_points = (golden_cross, death_cross)

        return data

    def factor_plot_specs(self) -> List[FactorPlotSpec]:
        """返回 SMA 快/慢线的绘图规格"""
        return [
            FactorPlotSpec(
                name=f"SMA{self.pfast}",
                label=f"SMA{self.pfast}（快线因子）",
                color='#FF6B35',
                linewidth=1.3,
                zorder=3,
            ),
            FactorPlotSpec(
                name=f"SMA{self.pslow}",
                label=f"SMA{self.pslow}（慢线因子）",
                color='#1E88E5',
                linewidth=1.3,
                zorder=3,
            ),
        ]

    def signal_yticklabels(self) -> Optional[List[str]]:
        return ['空头(-1)', '空仓(0)', '多头(+1)']

    def factor_subtitle(self) -> str:
        return f"双均线交叉：快线 SMA{self.pfast} / 慢线 SMA{self.pslow}"
