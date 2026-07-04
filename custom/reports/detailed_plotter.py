"""custom.reports.detailed_plotter - 详细版扩展分析图表

4 合 1 扩展图：
- 子图1: 月度收益热力图（年×月，颜色表示收益正负）
- 子图2: 滚动 Sharpe Ratio 曲线（252 日窗口）
- 子图3: 回撤期分析（回撤序列 + Top5 回撤期高亮）
- 子图4: 日收益分布直方图

兼容性：matplotlib 3.2.2 + pandas 2.0（所有 Series 转 numpy array）
"""
from typing import Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd

from custom.reports.contract import BacktestData, BacktestMetrics
from custom.reports.metrics import (
    calc_drawdown_series,
    calc_max_drawdown_period,
    calc_monthly_returns_heatmap,
    calc_rolling_sharpe,
)

# 中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def plot_detailed_report(data: BacktestData,
                         metrics: BacktestMetrics,
                         save_path) -> None:
    """生成详细版扩展分析图表

    Args:
        data: 回测数据契约
        metrics: 回测指标
        save_path: PNG 保存路径
    """
    data.validate()

    # 准备数据
    values = data.values
    dates = data.dates
    returns = values.pct_change().dropna()

    # 子图1: 月度收益热力图数据
    monthly_heatmap = calc_monthly_returns_heatmap(returns)

    # 子图2: 滚动 Sharpe
    rolling_sharpe = calc_rolling_sharpe(returns, window=252)

    # 子图3: 回撤序列 + Top5 回撤期
    drawdown = calc_drawdown_series(values)
    top5_periods = _find_top_drawdown_periods(values, top_n=5)

    # 子图4: 日收益分布
    daily_returns = returns.dropna()

    fig = plt.figure(figsize=(16, 20))

    # 用 GridSpec 控制子图布局（热力图宽，其他子图正常）
    gs = fig.add_gridspec(4, 1, hspace=0.4, height_ratios=[2, 2, 2, 2])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])
    ax4 = fig.add_subplot(gs[3])

    # ===== 子图1: 月度收益热力图 =====
    _plot_monthly_heatmap(ax1, monthly_heatmap, data.code)

    # ===== 子图2: 滚动 Sharpe Ratio =====
    _plot_rolling_sharpe(ax2, rolling_sharpe)

    # ===== 子图3: 回撤期分析 =====
    _plot_drawdown_periods(ax3, dates, drawdown, top5_periods, metrics.max_drawdown)

    # ===== 子图4: 日收益分布 =====
    _plot_returns_distribution(ax4, daily_returns)

    plt.savefig(str(save_path), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  详细版扩展图已保存：{save_path}")


def _plot_monthly_heatmap(ax, heatmap_df: pd.DataFrame, code: str) -> None:
    """月度收益热力图"""
    if heatmap_df.empty:
        ax.text(0.5, 0.5, '月度收益数据不足', ha='center', va='center',
                transform=ax.transAxes, fontsize=12)
        ax.set_title('月度收益热力图（数据不足）', fontsize=13, fontweight='bold')
        return

    # 转 numpy array（兼容 matplotlib 3.2.2 + pandas 2.0）
    arr = heatmap_df.values
    # 用 TwoSlopeNorm 以 0 为中心（正绿负红）
    vmax = max(abs(np.nanmin(arr)), abs(np.nanmax(arr)))
    if vmax == 0:
        vmax = 0.01
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    im = ax.imshow(arr, aspect='auto', cmap='RdYlGn', norm=norm)

    # 坐标轴
    ax.set_xticks(range(len(heatmap_df.columns)))
    ax.set_xticklabels([f'{m}月' for m in heatmap_df.columns])
    ax.set_yticks(range(len(heatmap_df.index)))
    ax.set_yticklabels(heatmap_df.index)

    # 在每个格子上标注数值
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            v = arr[i, j]
            if not np.isnan(v):
                ax.text(j, i, f'{v*100:.1f}%', ha='center', va='center',
                        color='black' if abs(v) < vmax * 0.5 else 'white',
                        fontsize=8)

    ax.set_title(f'{code} 月度收益热力图', fontsize=13, fontweight='bold')
    ax.set_xlabel('月份', fontsize=10)
    ax.set_ylabel('年份', fontsize=10)
    plt.colorbar(im, ax=ax, label='月度收益率')


def _plot_rolling_sharpe(ax, rolling_sharpe: pd.Series) -> None:
    """滚动 Sharpe Ratio 曲线"""
    if rolling_sharpe.empty or rolling_sharpe.dropna().empty:
        ax.text(0.5, 0.5, '滚动 Sharpe 数据不足（需要至少 252 个交易日）',
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title('滚动 Sharpe Ratio（数据不足）', fontsize=13, fontweight='bold')
        return

    # 转 numpy array
    nd = rolling_sharpe.index.to_numpy()
    nv = rolling_sharpe.to_numpy()

    ax.plot(nd, nv, color='#1E88E5', linewidth=1.3, label='滚动 Sharpe (252日)')
    ax.fill_between(nd, nv, 0, where=(nv > 0), color='#43A047', alpha=0.3)
    ax.fill_between(nd, nv, 0, where=(nv < 0), color='#E53935', alpha=0.3)
    ax.axhline(y=0, color='#333333', linewidth=0.8)
    ax.axhline(y=1, color='#888888', linewidth=0.6, linestyle='--', label='Sharpe=1')
    ax.set_title('滚动 Sharpe Ratio（252 日窗口）', fontsize=13, fontweight='bold')
    ax.set_ylabel('Sharpe Ratio', fontsize=10)
    ax.set_xlabel('日期', fontsize=10)
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))


def _find_top_drawdown_periods(equity: pd.Series, top_n: int = 5) -> list:
    """找出 Top N 回撤期（起止日期 + 谷底回撤）

    Returns:
        List[dict]: 每个元素 {peak_date, trough_date, recovery_date, max_dd, duration_days}
    """
    if equity.empty:
        return []
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max

    periods = []
    in_dd = False
    peak_date = None
    trough_date = None
    max_dd_in_period = 0

    for i in range(len(equity)):
        dd = drawdown.iloc[i]
        if dd < 0:
            if not in_dd:
                # 进入回撤
                in_dd = True
                peak_date = equity.index[i - 1] if i > 0 else equity.index[i]
                trough_date = equity.index[i]
                max_dd_in_period = dd
            else:
                if dd < max_dd_in_period:
                    max_dd_in_period = dd
                    trough_date = equity.index[i]
        else:
            if in_dd:
                # 退出回撤
                recovery_date = equity.index[i]
                duration = (trough_date - peak_date).days if peak_date else 0
                periods.append({
                    'peak_date': peak_date,
                    'trough_date': trough_date,
                    'recovery_date': recovery_date,
                    'max_dd': max_dd_in_period,
                    'duration_days': duration,
                })
                in_dd = False
                peak_date = None
                trough_date = None
                max_dd_in_period = 0

    # 若回测结束时仍在回撤中
    if in_dd:
        recovery_date = None
        duration = (trough_date - peak_date).days if peak_date else 0
        periods.append({
            'peak_date': peak_date,
            'trough_date': trough_date,
            'recovery_date': recovery_date,
            'max_dd': max_dd_in_period,
            'duration_days': duration,
        })

    # 按 max_dd 排序，取 Top N
    periods.sort(key=lambda x: x['max_dd'])
    return periods[:top_n]


def _plot_drawdown_periods(ax, dates, drawdown: pd.Series,
                            top5_periods: list, overall_max_dd: float) -> None:
    """回撤期分析图"""
    if drawdown.empty:
        ax.text(0.5, 0.5, '无回撤数据', ha='center', va='center',
                transform=ax.transAxes, fontsize=12)
        ax.set_title('回撤期分析（无数据）', fontsize=13, fontweight='bold')
        return

    # 转 numpy array
    nd = dates.to_numpy()
    ndd = drawdown.to_numpy()

    # 绘制回撤曲线（百分比）
    ax.fill_between(nd, ndd * 100, 0, color='#9E9E9E', alpha=0.5, label='回撤区间')
    ax.plot(nd, ndd * 100, color='#424242', linewidth=1)

    # 高亮 Top5 回撤期
    colors = ['#E53935', '#FF6B35', '#FFA726', '#FFEE58', '#D4E157']
    for i, period in enumerate(top5_periods):
        color = colors[i % len(colors)]
        peak = period['peak_date']
        recovery = period['recovery_date'] or dates[-1]
        ax.axvspan(peak, recovery, color=color, alpha=0.25,
                   label=f"Top{i+1}: {period['max_dd']*100:.2f}% ({period['duration_days']}天)")

    ax.set_title(f'回撤期分析（最大回撤 {overall_max_dd*100:.2f}%，Top5 高亮）',
                 fontsize=13, fontweight='bold')
    ax.set_ylabel('回撤 (%)', fontsize=10)
    ax.set_xlabel('日期', fontsize=10)
    ax.legend(loc='lower left', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))


def _plot_returns_distribution(ax, daily_returns: pd.Series) -> None:
    """日收益分布直方图"""
    if daily_returns.empty:
        ax.text(0.5, 0.5, '无日收益数据', ha='center', va='center',
                transform=ax.transAxes, fontsize=12)
        ax.set_title('日收益分布（无数据）', fontsize=13, fontweight='bold')
        return

    returns_arr = daily_returns.to_numpy() * 100  # 转百分比
    mean_v = float(np.mean(returns_arr))
    std_v = float(np.std(returns_arr))

    ax.hist(returns_arr, bins=50, color='#1E88E5', alpha=0.7,
            edgecolor='#1565C0', linewidth=0.5)
    ax.axvline(x=0, color='#333333', linewidth=1, linestyle='--', label='0%')
    ax.axvline(x=mean_v, color='#E53935', linewidth=1.5, label=f'均值 {mean_v:.3f}%')
    ax.axvline(x=mean_v + std_v, color='#FFA726', linewidth=1, linestyle=':',
               label=f'+1σ {mean_v+std_v:.3f}%')
    ax.axvline(x=mean_v - std_v, color='#FFA726', linewidth=1, linestyle=':',
               label=f'-1σ {mean_v-std_v:.3f}%')

    ax.set_title(f'日收益分布（均值 {mean_v:.3f}% / 标准差 {std_v:.3f}%）',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('日收益率 (%)', fontsize=10)
    ax.set_ylabel('频数', fontsize=10)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
