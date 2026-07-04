"""custom.reports.plotter - 基础版 4合1 matplotlib 绘图

子图1: 价格 + 因子（来自 FactorPlugin）+ 买卖点
子图2: 资金曲线 + 回撤灰色区间 + 最大回撤高亮
子图3: 持仓变化（跳仓点标记）
子图4: 因子信号（1/0/-1，金叉/死叉区域）

兼容性：matplotlib 3.2.2 + pandas 2.0（所有 Series 转 numpy array 后绘图）
"""
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use('Agg')  # 非交互后端，保存图片
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

from custom.reports.contract import BacktestData, BacktestMetrics
from custom.reports.factor_plugins.base import FactorPlugin
from custom.reports.metrics import calc_drawdown_series, calc_max_drawdown_period

# 中文字体（Windows 优先 SimHei/Microsoft YaHei）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def plot_basic_report(data: BacktestData,
                      metrics: BacktestMetrics,
                      factor_plugin: Optional[FactorPlugin],
                      save_path) -> None:
    """生成基础版 4合1 可视化报告

    Args:
        data: 回测数据契约
        metrics: 回测指标
        factor_plugin: 因子插件（可为 None，则跳过因子子图相关绘制）
        save_path: PNG 保存路径
    """
    data.validate()

    # 若提供因子插件，先让插件填充 factors/signals/cross_points
    if factor_plugin is not None:
        data = factor_plugin.prepare(data)

    # 准备数据（Series 用于索引操作，numpy array 用于绘图）
    dates = data.dates
    closes = data.closes
    values = data.values
    positions = data.positions

    # 回撤计算（完整周期）
    drawdown = calc_drawdown_series(values)
    running_max = values.cummax()

    # 转 numpy array 用于 matplotlib 绘图（兼容 matplotlib 3.2.2 + pandas 2.0）
    nd = dates.to_numpy()
    nc = closes.to_numpy()
    nv = values.to_numpy()
    npos = positions.to_numpy()
    nrm = running_max.to_numpy()
    ndd = drawdown.to_numpy()

    # 交易点
    buy_dates = [pd.to_datetime(t.date) for t in data.trades if t.type.upper() == 'BUY']
    buy_prices = [t.price for t in data.trades if t.type.upper() == 'BUY']
    sell_dates = [pd.to_datetime(t.date) for t in data.trades if t.type.upper() == 'SELL']
    sell_prices = [t.price for t in data.trades if t.type.upper() == 'SELL']

    # 最大回撤区间
    max_dd_start, max_dd_end, _ = calc_max_drawdown_period(values)

    # 因子信号
    nsig = data.signals.to_numpy() if data.signals is not None else None
    cross_points = data.cross_points if data.cross_points is not None else ([], [])
    golden_cross, death_cross = cross_points

    # 子图高度比例：价格/资金曲线/持仓/信号
    fig, axes = plt.subplots(4, 1, figsize=(16, 18), sharex=True,
                             gridspec_kw={'height_ratios': [3, 3, 1.5, 1.5]})

    strategy_label = f"{data.strategy_name} " if data.strategy_name else ""
    title = f'{data.code} {strategy_label}回测报告'

    # ===== 子图1: 价格 + 因子 + 买卖点 =====
    ax1 = axes[0]
    ax1.plot(nd, nc, color='#333333', linewidth=1, label='收盘价', zorder=2)
    # 因子线（来自插件规格）
    if factor_plugin is not None:
        for spec in factor_plugin.factor_plot_specs():
            if spec.name in data.factors:
                factor_arr = data.factors[spec.name].to_numpy()
                ax1.plot(nd, factor_arr, color=spec.color,
                         linewidth=spec.linewidth, label=spec.label, zorder=spec.zorder)
    if buy_dates:
        ax1.scatter(buy_dates, buy_prices, marker='^', color='#E53935',
                    s=150, zorder=5, label='买入信号')
    if sell_dates:
        ax1.scatter(sell_dates, sell_prices, marker='v', color='#43A047',
                    s=150, zorder=5, label='卖出信号')
    sub_title1 = f'{title} - 价格走势与因子变化'
    if factor_plugin and factor_plugin.factor_subtitle():
        sub_title1 += f'\n{factor_plugin.factor_subtitle()}'
    ax1.set_title(sub_title1, fontsize=14, fontweight='bold')
    ax1.set_ylabel('价格', fontsize=11)
    ax1.legend(loc='upper left', ncol=5, fontsize=9)
    ax1.grid(True, alpha=0.3)

    # ===== 子图2: 资金曲线 + 回撤灰色区间 =====
    ax2 = axes[1]
    ax2.plot(nd, nv, color='#1E88E5', linewidth=1.5, label='资产净值', zorder=3)
    # 所有回撤区间灰色阴影
    ax2.fill_between(nd, nv, nrm, where=(nv < nrm),
                     color='#9E9E9E', alpha=0.4, label='回撤区间', zorder=2)
    # 最大回撤区间高亮
    if max_dd_start is not None and max_dd_end is not None:
        ax2.axvspan(max_dd_start, max_dd_end, color='#FFCDD2', alpha=0.5, zorder=1,
                    label=f'最大回撤区间 ({metrics.max_drawdown*100:.2f}%)')
    ax2.set_title('资金曲线与回撤区间（灰色为回撤过程）', fontsize=14, fontweight='bold')
    ax2.set_ylabel('资产净值', fontsize=11)
    ax2.legend(loc='upper left', ncol=3, fontsize=9)
    ax2.grid(True, alpha=0.3)

    # ===== 子图3: 持仓变化（跳仓）=====
    ax3 = axes[2]
    ax3.step(nd, npos, where='post', color='#6A1B9A', linewidth=1.5, label='持仓数量')
    ax3.fill_between(nd, npos, step='post', color='#6A1B9A', alpha=0.2)
    # 标注跳仓点（仓位变化的点）
    pos_changes = positions.diff().fillna(0)
    jump_mask = pos_changes != 0
    if jump_mask.any():
        jump_dates = dates[jump_mask]
        jump_values = positions[jump_mask]
        ax3.scatter(jump_dates, jump_values, color='#E53935', s=80, zorder=5, label='跳仓点')
    ax3.set_title('持仓变化（跳仓过程）', fontsize=14, fontweight='bold')
    ax3.set_ylabel('持仓数量', fontsize=11)
    ax3.legend(loc='upper right', fontsize=9)
    ax3.grid(True, alpha=0.3)

    # ===== 子图4: 因子信号 =====
    ax4 = axes[3]
    if nsig is not None:
        ax4.plot(nd, nsig, color='#FF8F00', linewidth=1.2, label='因子信号')
        ax4.fill_between(nd, nsig, 0, where=(nsig > 0), color='#43A047',
                         alpha=0.3, label='多头区间')
        ax4.fill_between(nd, nsig, 0, where=(nsig < 0), color='#E53935',
                         alpha=0.3, label='空头区间')
        ax4.axhline(y=0, color='#333333', linewidth=0.8)
        # 标注金叉/死叉点
        if len(golden_cross) > 0:
            ax4.scatter(golden_cross, [1] * len(golden_cross), marker='^',
                        color='#43A047', s=100, zorder=5, label='金叉')
        if len(death_cross) > 0:
            ax4.scatter(death_cross, [-1] * len(death_cross), marker='v',
                        color='#E53935', s=100, zorder=5, label='死叉')
        # y 轴标签（来自插件或默认）
        ytick_labels = (factor_plugin.signal_yticklabels() if factor_plugin
                        else ['空头(-1)', '空仓(0)', '多头(+1)'])
        ax4.set_yticks([-1, 0, 1])
        ax4.set_yticklabels(ytick_labels)
        ax4.set_title('因子信号变化（金叉/死叉）', fontsize=14, fontweight='bold')
        ax4.legend(loc='upper right', ncol=5, fontsize=9)
    else:
        ax4.text(0.5, 0.5, '策略未提供因子信号（data.signals 为空）',
                 ha='center', va='center', transform=ax4.transAxes, fontsize=12)
        ax4.set_title('因子信号（无）', fontsize=14, fontweight='bold')
    ax4.set_ylabel('信号值', fontsize=11)
    ax4.set_xlabel('日期', fontsize=11)
    ax4.grid(True, alpha=0.3)

    # 日期轴格式
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax4.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    fig.autofmt_xdate(rotation=30)

    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  基础版图表已保存：{save_path}")
