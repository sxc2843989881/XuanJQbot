# 树结构 (Tree Structure)

> 源文档：https://pmorissette.github.io/bt/tree.html

## 概述

除了 [`Algos`](https://pmorissette.github.io/bt/bt.html#bt.core.Algo) 和 [`AlgoStacks`](https://pmorissette.github.io/bt/bt.html#bt.core.AlgoStack) 的概念，树结构是框架的核心。它允许你混合搭配证券和策略，以表达复杂的交易想法。

一个简单的 `Strategy` 可以包含多个子节点（证券或其他策略）。子节点不一定是证券，也可以是策略。这允许你：

- 组合多个策略
- 随时间动态分配资金
- 构建类似对冲基金的策略组合

### 示例：用子策略替换证券节点

```python
import bt

# 创建动量子策略
mom_s = bt.Strategy('mom_s', [
    bt.algos.RunMonthly(),
    bt.algos.SelectAll(),
    bt.algos.SelectMomentum(1),
    bt.algos.WeighEqually(),
    bt.algos.Rebalance()
], ['spy', 'eem'])

# 创建父策略 — 一个子节点是策略，另一个是证券
parent = bt.Strategy('parent', [
    bt.algos.RunMonthly(),
    bt.algos.SelectAll(),
    bt.algos.WeighEqually(),
    bt.algos.Rebalance()
], [mom_s, 'agg'])

# 运行回测
t = bt.Backtest(parent, data)
r = bt.run(t)
```

### 动态创建子策略

通过 `parent` 参数，可以在构建父策略后动态添加子策略：

```python
parent = bt.Strategy('parent', [...], ['agg'])

mom_s = bt.Strategy('mom_s', [...], ['spy', 'eem'], parent=parent)
```

## 节点类型

### 基类

- **`Node`** — 树中所有节点的基类

### 策略类型 (StrategyBase)

- **`Strategy`** — 基于市值的权重策略（使用 Algos）
- **`FixedIncomeStrategy`** — 基于名义本金的权重策略

### 证券类型 (SecurityBase)

- **`Security`** — 标准证券，市值为名义权重
- **`CouponPayingSecurity`** — 含息证券，支付定期/不定期现金流，可含持有成本
- **`FixedIncomeSecurity`** — 用持仓量作为名义权重（如零息债券）
- **`HedgeSecurity`** — 名义权重为零的证券（如 ETF 对冲）
- **`CouponPayingHedgeSecurity`** — 名义权重为零的含息对冲证券

### 节点接口

每个节点提供以下 **当前值** 的接口：

- 价格 (price)
- 价值 (value)
- 权重 (weight)

以及这些量的 **历史数据**，用于构建路径依赖的算法和回测后分析。
