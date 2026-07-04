# Custom — 个人定制策略工作区

> 本目录是用户私有策略与产出的根目录，与 Qbot 开源代码完全隔离。
> AI 协作时，所有新增策略、因子、回测、研究笔记均放在此目录下，**不要修改** `qbot/`、`pytrader/`、`pyfunds/` 等原项目代码。

## 策略研究 4 阶段生命周期

策略研究必须按 4 阶段顺序推进，**不允许跳阶段**。详见 `.ai/workflows/strategy_lifecycle.md`。

| 阶段 | 目录 | 入口脚本 | 产物目录 | 模块文档 |
|---|---|---|---|---|
| 1. 研究 | `custom/research/` | `research_<name>.py` | `custom/output/research/` | `.ai/modules/custom_research.md` |
| 2. 回测 | `custom/backtests/` | `backtest_<name>.py` | `custom/output/reports/` | `.ai/modules/custom_reports.md` |
| 3. 验证 | `custom/validation/` | `validate_<name>.py` | `custom/output/validation/` | `.ai/modules/custom_validation.md` |
| 4. 实盘 | `custom/live/` | `live_<name>.py` / `sim_<name>.py` | `custom/output/live/` | `.ai/modules/custom_live.md` |

**阶段衔接**：研究 → 回测 → 验证 → 实盘，每阶段通过门槛才能进入下一阶段，失败回退到上一阶段。

## 目录结构

```
custom/
├── README.md              ← 本文件
├── __init__.py
│
├── research/              ← 【阶段 1】研究阶段：因子有效性分析、信号探索、假设检验
├── backtests/             ← 【阶段 2】回测阶段：完整策略历史回测（命名 backtest_<name>.py）
│   ├── backtest_sma_cross.py       ← 基础回测（baostock + backtrader）
│   ├── backtest_sma_cross_viz.py   ← 旧版可视化脚本（保留作历史参考）
│   ├── backtest_v72.py             ← V72 成长价值轮动策略（双ETF，周频）
│   ├── backtest_sailboat_v2.py     ← 帆船V2 策略回测
│   └── _debug_index_codes.py       ← 指数代码调试脚本（临时）
├── validation/            ← 【阶段 3】验证阶段：OOS、Walk-Forward、参数敏感性、稳健性检验
├── live/                  ← 【阶段 4】实盘阶段：模拟盘、实盘下单、监控、风控
│
├── strategies/            ← 策略代码（贯穿 4 阶段共享，命名 <name>_strategy.py）
│   └── sma_cross_strategy.py
├── factors/               ← 因子逻辑（抽离自策略，便于复用与回测）
│   └── factor_sma.py
├── reports/               ← 通用回测报告模块（与策略解耦，详见 .ai/modules/custom_reports.md）
│   ├── contract.py                ← 数据契约（BacktestData/BacktestMetrics/TradeRecord）
│   ├── metrics.py                 ← 通用指标计算（完整周期回撤/Sharpe/Sortino/滚动Sharpe）
│   ├── plotter.py                 ← 基础版 4合1 matplotlib 绘图
│   ├── detailed_plotter.py        ← 详细版扩展图（月度热力图/滚动Sharpe/回撤期/收益分布）
│   ├── html_reporter.py           ← HTML 报告生成（基础版+详细版）
│   ├── pipeline.py                ← 报告生成流水线（ReportPipeline 类）
│   └── factor_plugins/            ← 因子子图可插拔插件
│       ├── base.py                ← FactorPlugin 抽象基类
│       └── sma_plugin.py          ← SMA 因子插件（参考实现）
├── tests/                 ← 测试脚本（命名 test_<module>.py）
│   ├── test_reports_module.py          ← custom.reports 端到端测试
│   └── test_data_fetch_etf_index.py    ← 数据获取测试（baostock 拉 ETF/指数列表）
├── notebooks/             ← 研究笔记、因子分析、可视化
├── data/                  ← 本地缓存（如有；默认应从网络获取最新数据）
├── output/                ← 统一输出目录（运行时生成）
│   ├── research/          ← 研究阶段产物（IC/分层/研究报告）
│   ├── reports/           ← 回测阶段产物（PNG/HTML）
│   ├── validation/        ← 验证阶段产物（OOS/Walk-Forward/参数敏感性）
│   ├── live/              ← 实盘阶段产物（日志/监控日报/风控记录）
│   └── data/              ← 数据获取输出（ETF/指数列表 CSV 等）
└── logs/                  ← 运行日志
```

## 使用约定

1. **4 阶段顺序推进**：研究 → 回测 → 验证 → 实盘，不允许跳阶段。每阶段通过门槛才能进入下一阶段。
2. **数据来源**：默认从网络获取最新数据（tushare pro_api / baostock / akshare），禁止依赖本地旧数据。
3. **因子抽离**：因子逻辑放 `factors/`，策略只调用因子，便于复用。
4. **回测引擎**：使用 `qbot/engine/backtest/` 新引擎或 backtrader，不用 `pyfunds/backtest/xalpha/` 老引擎。
5. **回撤定义**：从上一轮历史高点到下一轮低点的完整周期，不按年切分。
6. **报告生成**：所有回测报告统一用 `custom/reports/` 通用模块，**不要在每个策略里重写绘图代码**。新策略只需写 `FactorPlugin` 插件。
7. **命名规范**：
   - 研究脚本 `research_<name>.py`，主函数 `run_research()`
   - 策略文件 `<name>_strategy.py`，类名 `<Name>Strategy`
   - 因子文件 `factor_<name>.py`
   - 回测脚本 `backtest_<name>.py`，主函数 `run_backtest()`
   - 验证脚本 `validate_<name>.py` / `walkforward_<name>.py` / `oos_<name>.py`
   - 实盘脚本 `live_<name>.py` / `sim_<name>.py` / `monitor_<name>.py` / `risk_<name>.py`
   - 测试脚本 `test_<module>.py`
   - 因子插件 `<name>_plugin.py`，类名 `<Name>FactorPlugin`
8. **沟通语言**：所有注释、文档、commit message 使用中文，量化术语保留英文（Sharpe、Calmar、IC、IR 等）。

## 环境激活

> 项目使用 Python 3.10.20（conda 环境名 `Qbot`，路径 `C:\Users\28439\AppData\Local\conda\conda\envs\Qbot`）。
> 关键依赖版本：pandas 2.0.3 / numpy 1.26.4 / matplotlib 3.5.3 / akshare 1.18.64 / quantstats 0.0.77 / wxPython 4.2.1 / TA-Lib 0.6.8。

```powershell
conda activate Qbot
cd c:\XuanJLH\Qbot
python main.py   # 启动 GUI
```

## 常用命令

```powershell
# 跑最小回测（验证环境）
python -m custom.backtests.backtest_sma_cross

# 启动 GUI
$env:PYTHONPATH = "c:\XuanJLH\Qbot\pytrader"
python main.py

# 跑通用报告模块端到端测试
python -m custom.tests.test_reports_module

# 跑数据获取测试（ETF/指数列表）
python -m custom.tests.test_data_fetch_etf_index
```

## 与 Qbot 原项目的关系

| 项目原目录 | 本目录 | 关系 |
|-----------|--------|------|
| `qbot/strategies/` | `custom/strategies/` | 原项目示例策略 vs 用户私有策略 |
| `qbot/engine/backtest/` | `custom/backtests/` | 原项目回测引擎（调用） vs 用户回测脚本 |
| — | `custom/research/` | 研究阶段工作区（原项目无） |
| — | `custom/validation/` | 验证阶段工作区（原项目无） |
| `pytrader/easytrader/` | `custom/live/` | 原项目交易系统（调用） vs 用户实盘脚本 |
| — | `custom/factors/` | 用户因子库（原项目无独立因子目录） |
| — | `custom/reports/` | 通用回测报告模块（原项目无统一报告生成器） |
| — | `custom/tests/` | 测试脚本（原项目无独立测试目录） |
| — | `custom/output/` | 统一输出目录（原项目无统一输出规范） |

## AI 协作提示

AI 接手本目录任务时，应先读取：
1. `c:\XuanJLH\Qbot\.ai\conventions.md` — 代码约定
2. `c:\XuanJLH\Qbot\.ai\custom_directory.md` — custom 目录协作规范
3. `c:\XuanJLH\Qbot\.ai\workflows\strategy_lifecycle.md` — **策略研究 4 阶段工作流（必读）**
4. `c:\XuanJLH\Qbot\.ai\workflows\add_strategy.md` — 新增策略流程
5. `c:\XuanJLH\Qbot\.ai\workflows\run_backtest.md` — 回测执行流程
6. `c:\XuanJLH\Qbot\.ai\modules\custom_reports.md` — 通用报告模块说明（新增策略接入报告必读）
7. `c:\XuanJLH\Qbot\.ai\modules\custom_research.md` — 研究阶段模块说明
8. `c:\XuanJLH\Qbot\.ai\modules\custom_validation.md` — 验证阶段模块说明
9. `c:\XuanJLH\Qbot\.ai\modules\custom_live.md` — 实盘阶段模块说明
10. `c:\XuanJLH\Qbot\.ai\modules\synthetic_data.md` — 合成样本数据生成器模块说明（样本不足时模拟扩展）
