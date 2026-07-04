# bt - Flexible Backtesting for Python

> 源文档：https://pmorissette.github.io/bt/index.html

## 什么是 bt？

**bt** 是一个灵活的 Python 回测框架，用于测试量化交易策略。**回测**是在给定数据集上测试策略的过程。该框架允许你通过组合不同的 [`Algos`](https://pmorissette.github.io/bt/bt.html#bt.core.Algo) 轻松创建策略。它旨在促进可测试、可重用和灵活的策略逻辑模块的创建，以加速复杂交易策略的开发。

目标：让 **量化研究员** 不必重复造轮子，专注于策略开发这一核心工作。

**bt** 使用 **Python** 编写，融入了一个充满活力的数据分析生态系统。它可以利用现有的机器学习、信号处理和统计库，避免重复造轮子。

bt 构建在 [ffn](https://github.com/pmorissette/ffn)（一个 Python 金融函数库）之上。

## 快速示例

### 简单策略回测

创建一个月度再平衡、等权重、只做多的策略：

```python
import bt

# 获取数据
data = bt.get('spy,agg', start='2010-01-01')

# 创建策略
s = bt.Strategy('s1', [
    bt.algos.RunMonthly(),
    bt.algos.SelectAll(),
    bt.algos.WeighEqually(),
    bt.algos.Rebalance()
])

# 创建回测并运行
test = bt.Backtest(s, data)
res = bt.run(test)

# 查看结果
res.plot()
res.display()
res.plot_security_weights()
```

### 修改策略

如果要改为周频再平衡 + 风险平价（逆波动率加权）：

```python
s2 = bt.Strategy('s2', [
    bt.algos.RunWeekly(),
    bt.algos.SelectAll(),
    bt.algos.WeighInvVol(),
    bt.algos.Rebalance()
])

test2 = bt.Backtest(s2, data)
res2 = bt.run(test, test2)
```

## 功能特性

- **树结构 (Tree Structure)** — 支持复杂策略的组合，每个树节点都有自己的价格指数
- **算法栈 (AlgoStacks)** — 模块化、可重用的策略逻辑构建块
- **交易成本建模** — 通过佣金函数和特定品种的买卖价差
- **固定收益支持** — 含息证券、未融资工具、持有成本、名义权重等
- **图表和报告** — 丰富的可视化回测结果功能
- **详细统计** — 跨多个回测的统计对比

## 路线图

- **速度** — 在保持易用性的前提下提升性能
- **算法** — 持续开发更多内置算法
- **图表和报告** — 持续改进报告功能
