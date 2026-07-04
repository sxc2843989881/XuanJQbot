# 示例

> 源文档：https://pmorissette.github.io/bt/examples.html

## SMA 策略

选择价格在 50 日均线之上的证券，等权重配置。

```python
import bt
import pandas as pd

# 下载数据
data = bt.get('aapl,msft,c,gs,ge', start='2010-01-01')
sma = data.rolling(50).mean()

# 自定义选券 Algo
class SelectWhere(bt.Algo):
    """根据信号 DataFrame 选择证券"""
    def __init__(self, signal):
        self.signal = signal

    def __call__(self, target):
        if target.now in self.signal.index:
            sig = self.signal.loc[target.now]
            selected = list(sig.index[sig])
            target.temp['selected'] = selected
        return True

# 创建策略
s = bt.Strategy('above50sma', [
    SelectWhere(data > sma),
    bt.algos.WeighEqually(),
    bt.algos.Rebalance()
])

t = bt.Backtest(s, data)
res = bt.run(t)
res.plot()
res.display()
```

### 可复用的封装函数

```python
def above_sma(tickers, sma_per=50, start='2010-01-01', name='above_sma'):
    data = bt.get(tickers, start=start)
    sma = data.rolling(sma_per).mean()
    s = bt.Strategy(name, [
        SelectWhere(data > sma),
        bt.algos.WeighEqually(),
        bt.algos.Rebalance()
    ])
    return bt.Backtest(s, data)
```

## SMA 交叉策略

50日均线上穿200日均线时做多，下穿时做空。

```python
# 计算目标权重
data = bt.get('spy', start='2010-01-01')
sma50 = data.rolling(50).mean()
sma200 = data.rolling(200).mean()

tw = sma200.copy()
tw[sma50 > sma200] = 1.0
tw[sma50 <= sma200] = -1.0
tw[sma200.isnull()] = 0.0

class WeighTarget(bt.Algo):
    def __init__(self, target_weights):
        self.tw = target_weights

    def __call__(self, target):
        if target.now in self.tw.index:
            w = self.tw.loc[target.now]
            target.temp['weights'] = w.dropna()
        return True

ma_cross = bt.Strategy('ma_cross', [
    WeighTarget(tw),
    bt.algos.Rebalance()
])
t = bt.Backtest(ma_cross, data)
res = bt.run(t)
```

## 树结构探索

使用子策略的净值曲线创建"合成证券"来进行策略组合。

## 买入持有策略

```python
def long_only_ew(tickers, start='2010-01-01', name='long_only_ew'):
    s = bt.Strategy(name, [
        bt.algos.RunOnce(),
        bt.algos.SelectAll(),
        bt.algos.WeighEqually(),
        bt.algos.Rebalance()
    ])
    data = bt.get(tickers, start=start)
    return bt.Backtest(s, data)
```

## 更多示例

完整示例列表包括：

- 趋势策略示例 1 和 2
- 策略组合
- 等风险贡献组合 (Equal Risk Contribution)
- 预测跟踪误差再平衡组合
- 固定收益示例

详见源文档：https://pmorissette.github.io/bt/examples.html
