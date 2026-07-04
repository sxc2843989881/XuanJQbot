"""custom.reports.factor_plugins - 因子子图可插拔插件

不同策略的因子展示不同（SMA 是快慢线、MACD 是 DIF/DEA、RSI 是 RSI 值）。
通过 FactorPlugin 抽象基类解耦：策略实现插件，绘图层调用插件接口。

参考实现：
- sma_plugin.SmaFactorPlugin: SMA 双均线策略的因子插件
"""
