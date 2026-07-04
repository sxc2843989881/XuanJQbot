# bt_engine — bt (Flexible Backtesting for Python) 集成目录

## 概述

本目录集成 **bt** 回测框架（https://github.com/pmorissette/bt），用于 Qbot 项目中的事件驱动型策略回测。

bt 是一个灵活的 Python 回测框架，专为组合/多资产策略设计，原生支持资产轮动、再平衡和组合管理。

## 安装

```bash
pip install bt
```

已安装版本：**0.2.9**

## 核心概念

bt 的核心是 **Algo（算法）** 组合。每个 Algo 是一个返回 True/False 的函数，多个 Algo 组成 AlgoStack，按顺序执行。

标准流程：
1. **Run时机** — `RunMonthly()` / `RunWeekly()` / `RunDaily()` 控制调仓频率
2. **选资产** — `SelectAll()` / `SelectWhere()` 选择持仓标的
3. **定权重** — `WeighEqually()` / `WeighInvVol()` / `WeighTarget()` 设定目标权重
4. **再平衡** — `Rebalance()` 执行调仓

## 文档

- `docs_index.md` — bt 官方文档首页
- `docs_api.md` — API 参考（algos / backtest / core 模块）
- `docs_install.md` — 安装指南
- `docs_examples.md` — 示例代码
- `docs_algos.md` — Algo 设计原理

## 用法示例

```python
import bt

# 加载数据（DataFrame，列为资产名，行为日期）
data = bt.get('spy,agg', start='2010-01-01')

# 创建策略
s = bt.Strategy('s1', [
    bt.algos.RunMonthly(),
    bt.algos.SelectAll(),
    bt.algos.WeighEqually(),
    bt.algos.Rebalance()
])

# 运行回测
test = bt.Backtest(s, data)
res = bt.run(test)
res.display()
```
