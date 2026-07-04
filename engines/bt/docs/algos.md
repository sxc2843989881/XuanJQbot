# 算法 (Algos)

> 源文档：https://pmorissette.github.io/bt/algos.html

## 概述

bt 的核心构建块是 [`Algo`](https://pmorissette.github.io/bt/bt.html#bt.core.Algo) 和 [`AlgoStack`](https://pmorissette.github.io/bt/bt.html#bt.core.AlgoStack)。

### Algo（算法）

Algo 本质上是一个返回 `True` 或 `False` 的函数。它接收一个参数——被测试的 [`Strategy`](https://pmorissette.github.io/bt/bt.html#bt.core.Strategy) 对象。一个 Algo 应只服务于一个特定目的：控制执行流、控制选券、控制仓位分配等。

### AlgoStack（算法栈）

`AlgoStack` 将多个 Algo 组合在一起，依次执行，只要每个 Algo 返回 `True`。一旦有 Algo 返回 `False`，整个栈停止执行并返回 `False`。这使得我们可以组合不同的 Algo 并通过返回值控制执行流。

## 数据传递

Strategy 有两个字典属性用于在 Algo 之间传递数据：

- **temp** — 临时数据，每次数据变化时刷新
- **perm** — 永久数据，不会被清除

典型的数据流：**选择 (Select) → 权重 (Weight) → 再平衡 (Rebalance)**

```python
# 选择阶段 — 设置 temp['selected']
bt.algos.SelectAll()
bt.algos.SelectThese(['spy', 'agg'])
bt.algos.SelectWhere(signal)
bt.algos.SelectN(5)

# 权重阶段 — 读取 temp['selected']，设置 temp['weights']
bt.algos.WeighEqually()
bt.algos.WeighInvVol()
bt.algos.WeighRandomly()
bt.algos.WeighSpecified(weights)
bt.algos.WeighTarget(target_weights)
bt.algos.WeighMeanVar()

# 再平衡阶段 — 读取 temp['weights']，执行交易
bt.algos.Rebalance()
bt.algos.RebalanceOverTime(n=10)
```

## 实现自定义 Algo

### 类方式（推荐，可保存状态）

```python
class MyAlgo(bt.Algo):
    def __init__(self, arg1, arg2):
        self.arg1 = arg1
        self.arg2 = arg2

    def __call__(self, target):
        # 通过 target.temp['key'] 访问/存储变量
        # 访问数据: target.get_data('signal_name')
        return True  # 记得返回 bool 值
```

### 函数方式（无状态）

```python
def MyAlgo2(target):
    # 所有逻辑
    return True
```

## 最佳实践

### 可重用性

Algo 应跨不同回测可重用。当需要额外数据时，使用 **数据名称** 构建 Algo，然后在 `Backtest` 中传入 `additional_data`：

```python
s = bt.Strategy('s1', [MyAlgo('my_signal')])
test = bt.Backtest(s, data, additional_data={'my_signal': signal_df})
```

### 调试

在 Algo 栈中插入以下 Algo 进行调试：

- `bt.algos.Debug()` — 触发 pdb 断点
- `bt.algos.PrintTempData()` — 打印临时数据
- `bt.algos.PrintInfo('{name}:{now}')` — 打印策略信息
- `bt.algos.PrintRisk()` — 打印风险数据

### 分支与控制流

使用 `Or` Algo 实现分支逻辑：

```python
logging_stack = bt.AlgoStack(
    bt.algos.RunWeekly(),
    bt.algos.PrintInfo('{name}:{now}, Value:{_value:0.0f}')
)

trading_stack = bt.AlgoStack(
    bt.algos.RunMonthly(),
    bt.algos.SelectAll(),
    bt.algos.WeighEqually(),
    bt.algos.Rebalance()
)

branch_stack = bt.AlgoStack(
    bt.algos.Or([logging_stack, trading_stack])
)
```
