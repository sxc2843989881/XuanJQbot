"""SMA 双均线交叉策略（backtrader）

基于 custom/factors/factor_sma.py 的因子逻辑实现。
策略只负责下单逻辑，因子计算由 backtrader 内置 SMA 指标完成（与因子逻辑一致）。
"""
import backtrader as bt


class SmaCrossStrategy(bt.Strategy):
    """SMA 双均线交叉策略

    Params:
        pfast: 快线周期（默认 10）
        pslow: 慢线周期（默认 30）
    """

    params = (('pfast', 10), ('pslow', 30),)

    def __init__(self):
        sma_fast = bt.indicators.SMA(period=self.p.pfast)
        sma_slow = bt.indicators.SMA(period=self.p.pslow)
        self.crossover = bt.indicators.CrossOver(sma_fast, sma_slow)

    def next(self):
        if self.position.size == 0:
            if self.crossover > 0:
                self.buy()
        elif self.position.size > 0:
            if self.crossover < 0:
                self.close()
