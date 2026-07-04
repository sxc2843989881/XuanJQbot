# API 参考

> 源文档：https://pmorissette.github.io/bt/bt.html

## core 模块

### Node

所有树节点的基类，提供价格、价值、权重的当前值和历史数据接口。

### SecurityBase

证券节点的基类。
- **`Security`** — 标准证券
- **`CouponPayingSecurity`** — 含息证券（债券、互换等）

### StrategyBase

策略节点的基类。
- **`Strategy`** — 市值加权策略
- **`FixedIncomeStrategy`** — 名义本金加权策略

### Algo

所有算法的基类。`__call__(self, target)` 返回 True/False。

### AlgoStack

算法栈，按顺序执行算法，遇到 False 停止。

## algos 模块

### 执行流控制

| Algo | 说明 |
|------|------|
| `RunDaily()` | 每日返回 True |
| `RunWeekly()` | 每周返回 True |
| `RunMonthly()` | 每月返回 True |
| `RunOnDate(dates)` | 在指定日期返回 True |
| `RunAfterDate(date)` | 在指定日期之后返回 True |
| `RunAfterDays(days)` | 经过指定交易日数后返回 True |
| `RunOnce()` | 只在第一次调用时返回 True |
| `Or(algos)` | 任意 Algo 返回 True 则返回 True |
| `Not(algo)` | 反转 Algo 的返回值 |
| `Require(pred, item)` | 根据 temp 条目的谓词控制流 |

### 证券选择

| Algo | 说明 |
|------|------|
| `SelectAll()` | 选择所有可用证券 |
| `SelectThese(tickers)` | 选择指定证券列表 |
| `SelectWhere(signal)` | 根据信号 DataFrame 选择 |
| `SelectN(n)` | 选择前 N 个证券 |
| `SelectMomentum(n)` | 选择过去 n 个月动量最强的 |
| `ResolveOnTheRun(data)` | 解析 on-the-run 证券别名 |
| `CloseDead()` | 关闭价格为 0 的持仓 |
| `ClosePositionsAfterDates(dates)` | 在指定日期后关闭持仓 |
| `RollPositionsAfterDates(data)` | 根据映射滚动证券持仓 |

### 权重设置

| Algo | 说明 |
|------|------|
| `WeighEqually()` | 等权重 |
| `WeighInvVol()` | 逆波动率加权 |
| `WeighRandomly()` | 随机权重 |
| `WeighSpecified(weights)` | 指定权重 |
| `WeighTarget(target_weights)` | 目标权重 DataFrame |
| `WeighMeanVar()` | 均值-方差优化 |
| `LimitWeights(limit)` | 限制单个证券权重上限 |
| `LimitDeltas(limit)` | 限制权重单次变化幅度 |

### 再平衡

| Algo | 说明 |
|------|------|
| `Rebalance()` | 按 temp['weights'] 再平衡 |
| `RebalanceOverTime(n)` | 在 n 个周期内逐步再平衡 |
| `PTE_Rebalance(vol_cap, tw, ...)` | 当预测跟踪误差超过阈值时触发再平衡 |

### 风险

| Algo | 说明 |
|------|------|
| `UpdateRisk(...)` | 更新风险度量 |
| `HedgeRisks(measures, ...)` | 对冲指定风险敞口 |

### 现金流

| Algo | 说明 |
|------|------|
| `CapitalFlow(amount)` | 模拟资金流入/流出 |

### 交易回放

| Algo | 说明 |
|------|------|
| `ReplayTransactions(transactions)` | 回放实际交易记录 |

### 调试

| Algo | 说明 |
|------|------|
| `Debug()` | 触发 pdb 断点 |
| `PrintDate()` | 打印当前日期 |
| `PrintInfo(fmt)` | 打印策略信息 |
| `PrintRisk(fmt)` | 打印风险数据 |
| `PrintTempData(fmt)` | 打印临时数据 |

## backtest 模块

### Backtest

回测对象，组合 Strategy + 数据。

```python
Backtest(strategy, data, initial_capital=1e6, commission=None,
         bidoffer=None, additional_data=None, name=None)
```

参数：
- `strategy` — Strategy 对象
- `data` — 价格 DataFrame
- `initial_capital` — 初始资金（默认 1e6）
- `commission` — 佣金函数
- `bidoffer` — 买卖价差数据
- `additional_data` — 额外数据字典

### Result

回测结果对象，封装了 ffn.GroupStats 并添加了辅助方法。

主要方法：
- `plot()` — 净值曲线图
- `plot_histogram()` — 收益分布直方图
- `plot_security_weights()` — 证券权重变化图
- `display()` — 显示详细统计指标
- `get_transactions()` — 获取交易记录
- `get_security_weights()` — 获取权重历史

### 便捷函数

```python
bt.run(*backtests)     # 运行一个或多个回测
bt.get(*tickers, **kwargs)  # 下载数据（ffn.get 的别名）
bt.merge(*args)        # 合并数据
```
