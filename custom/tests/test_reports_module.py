"""custom.reports 模块端到端测试

测试链路：
  [1] baostock 取数据 →
  [2] backtrader 跑 SMA 策略 →
  [3] 组装 BacktestData + BacktestMetrics 数据契约 →
  [4] 调用 ReportPipeline 生成基础版 + 详细版报告 →
  [5] 验证所有输出文件存在且非空

运行：
    conda activate Qbot  (或直接用 Qbot 环境的 python)
    cd c:\\XuanJLH\\Qbot
    python -m custom.tests.test_reports_module
"""
import sys
from datetime import datetime, timedelta
from pathlib import Path

# 确保从项目根目录运行时能 import custom 包
# 本文件位于 custom/tests/，需上溯 2 层到项目根
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import baostock as bs  # noqa: F401  (baostock import 副作用：初始化)
import backtrader as bt
import pandas as pd

# 复用已有的 SMA 策略实现（不重新造轮子）
from custom.strategies.sma_cross_strategy import SmaCrossStrategy
from custom.backtests.backtest_sma_cross import fetch_data, calc_max_drawdown

# 被测模块
from custom.reports import BacktestData, BacktestMetrics, TradeRecord, ReportPipeline
from custom.reports.factor_plugins.sma_plugin import SmaFactorPlugin


# ===== 第 1 步：带记录的 SMA 策略 =====
class SmaCrossStrategyRecord(SmaCrossStrategy):
    """继承基础 SMA 策略，额外记录每日持仓/资金与交易点"""
    params = (('pfast', 10), ('pslow', 30))

    def __init__(self):
        super().__init__()
        self.position_records = []
        self.trade_records = []

    def next(self):
        super().next()
        self.position_records.append({
            'date': self.data.datetime.date(0),
            'position': self.position.size,
            'value': self.broker.getvalue(),
            'close': self.data.close[0],
        })
        if len(self.position_records) >= 2:
            prev_pos = self.position_records[-2]['position']
            curr_pos = self.position_records[-1]['position']
            if curr_pos > prev_pos:
                self.trade_records.append(TradeRecord(
                    date=self.data.datetime.date(0),
                    type='BUY',
                    price=float(self.data.close[0]),
                    size=float(curr_pos - prev_pos),
                ))
            elif curr_pos < prev_pos:
                self.trade_records.append(TradeRecord(
                    date=self.data.datetime.date(0),
                    type='SELL',
                    price=float(self.data.close[0]),
                    size=float(prev_pos - curr_pos),
                ))


# ===== 第 2 步：回测编排 =====
def run_backtest(code="sh.600519", days=730, cash=100000.0,
                 commission=0.001, pfast=10, pslow=30):
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

    print(f"[1/5] 从 baostock 获取数据：{code}，{start_date} ~ {end_date}")
    df = fetch_data(code, start_date, end_date)
    print(f"      获取到 {len(df)} 条数据，区间 {df.index[0].date()} ~ {df.index[-1].date()}")

    print(f"[2/5] 运行 backtrader 回测（SMA {pfast}/{pslow}）")
    cerebro = bt.Cerebro()
    cerebro.addstrategy(SmaCrossStrategyRecord, pfast=pfast, pslow=pslow)
    cerebro.adddata(bt.feeds.PandasData(dataname=df))
    cerebro.broker.setcash(cash)
    cerebro.broker.setcommission(commission=commission)

    # 分析器
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', riskfreerate=0.03)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='timereturn')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')

    print(f"      初始资金：{cash:.2f}")
    results = cerebro.run()
    strat = results[0]
    final_value = cerebro.broker.getvalue()
    print(f"      最终资金：{final_value:.2f}")

    return df, strat, final_value, cash


# ===== 第 3 步：组装数据契约 =====
def build_contract(df, strat, final_value, cash, code, pfast, pslow):
    """把 backtrader 回测结果组装成 BacktestData + BacktestMetrics"""
    print(f"[3/5] 组装通用数据契约")

    # 时间序列
    dates = pd.to_datetime([r['date'] for r in strat.position_records])
    closes = pd.Series([r['close'] for r in strat.position_records], index=dates)
    values = pd.Series([r['value'] for r in strat.position_records], index=dates)
    positions = pd.Series([r['position'] for r in strat.position_records], index=dates)

    # 指标计算（复用 custom.reports.metrics 的完整周期回撤）
    total_return = (final_value - cash) / cash
    actual_days = (df.index[-1] - df.index[0]).days
    annual_return = (1 + total_return) ** (365 / actual_days) - 1 if actual_days > 0 else 0.0

    # 用 timereturn 构建等权资金曲线算回撤
    timereturn = strat.analyzers.timereturn.get_analysis()
    equity = pd.Series(timereturn)
    if not equity.empty:
        equity_curve = (1 + equity).cumprod() * cash
        max_dd = calc_max_drawdown(equity_curve)
    else:
        max_dd = 0.0

    sharpe = strat.analyzers.sharpe.get_analysis().get('sharperatio', None)
    calmar = annual_return / abs(max_dd) if max_dd < 0 else 0.0

    # 交易统计
    trade_analysis = strat.analyzers.trades.get_analysis()
    total_trades = trade_analysis.get('total', {}).get('total', 0)
    won_trades = trade_analysis.get('won', {}).get('total', 0)
    lost_trades = trade_analysis.get('lost', {}).get('total', 0)
    win_rate = won_trades / total_trades * 100 if total_trades > 0 else 0

    # 组装数据契约（注意：factors/signals 留给 FactorPlugin 填充）
    data = BacktestData(
        dates=dates,
        closes=closes,
        values=values,
        positions=positions,
        trades=strat.trade_records,
        strategy_name="SMA",
        strategy_params={'pfast': pfast, 'pslow': pslow},
        code=code,
        # factors / signals / cross_points 留空，由 SmaFactorPlugin.prepare() 填充
    )

    metrics = BacktestMetrics(
        code=code,
        start=str(df.index[0].date()),
        end=str(df.index[-1].date()),
        days=actual_days,
        cash=cash,
        final_value=final_value,
        total_return=total_return,
        annual_return=annual_return,
        max_drawdown=max_dd,
        sharpe=sharpe,
        calmar=calmar,
        total_trades=total_trades,
        won_trades=won_trades,
        lost_trades=lost_trades,
        win_rate=win_rate,
        strategy_name="SMA",
        strategy_params={'pfast': pfast, 'pslow': pslow},
    )

    print(f"      数据契约：{len(dates)} 个交易日，{len(strat.trade_records)} 笔交易")
    print(f"      指标：总收益 {total_return*100:.2f}% / 最大回撤 {max_dd*100:.2f}% / Sharpe {sharpe}")
    return data, metrics


# ===== 第 4 步：调用通用报告模块 =====
def generate_reports(data, metrics, pfast, pslow, code):
    print(f"[4/5] 调用 custom.reports 通用模块生成报告")
    # 输出统一到 custom/output/reports/
    # 本文件位于 custom/tests/，上溯 1 层到 custom/，再到 output/reports/
    output_dir = Path(__file__).resolve().parents[1] / 'output' / 'reports'
    code_safe = code.replace('.', '_')
    name_prefix = f'sma_cross_{code_safe}'

    # 实例化因子插件
    factor_plugin = SmaFactorPlugin(pfast=pfast, pslow=pslow)

    # 调用流水线生成基础版 + 详细版
    pipeline = ReportPipeline(data, metrics, factor_plugin=factor_plugin)
    result = pipeline.generate(output_dir, name_prefix=name_prefix,
                               include_basic=True, include_detailed=True)
    return result


# ===== 第 5 步：验证输出 =====
def verify_outputs(result):
    print(f"[5/5] 验证输出文件")
    expected_files = [
        ('基础版 PNG', result.chart_path),
        ('基础版 HTML', result.html_path),
        ('详细版 PNG', result.detailed_chart_path),
        ('详细版 HTML', result.detailed_html_path),
    ]
    all_ok = True
    for label, path in expected_files:
        if path is None:
            print(f"  [FAIL] {label}: 未生成（路径为 None）")
            all_ok = False
            continue
        p = Path(path)
        if not p.exists():
            print(f"  [FAIL] {label}: 文件不存在 - {path}")
            all_ok = False
        elif p.stat().st_size == 0:
            print(f"  [FAIL] {label}: 文件为空 - {path}")
            all_ok = False
        else:
            print(f"  [OK]   {label}: {p.stat().st_size:>10,} 字节 - {path}")
    return all_ok


def main():
    print("=" * 70)
    print("custom.reports 通用回测报告模块 - 端到端测试")
    print("=" * 70)

    code = "sh.600519"
    pfast, pslow = 10, 30

    # 1+2: 取数据 + 回测
    df, strat, final_value, cash = run_backtest(code=code, days=730,
                                                pfast=pfast, pslow=pslow)

    # 3: 组装契约
    data, metrics = build_contract(df, strat, final_value, cash, code, pfast, pslow)

    # 4: 生成报告
    result = generate_reports(data, metrics, pfast, pslow, code)

    # 5: 验证
    print()
    all_ok = verify_outputs(result)

    print()
    print("=" * 70)
    if all_ok:
        print("[OK] 端到端测试通过：custom.reports 模块走通")
    else:
        print("[FAIL] 端到端测试失败：部分文件未生成或为空")
    print("=" * 70)
    print(result)
    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main())
