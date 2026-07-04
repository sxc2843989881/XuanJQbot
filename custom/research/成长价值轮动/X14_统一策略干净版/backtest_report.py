"""backtest_report.py — 标准回测分析报告生成器
================================================================
生成完整的回测分析图表集，支持单策略/多策略对比。

用法:
  from backtest_report import BacktestReport
  report = BacktestReport(output_dir='回测结果')
  report.add_strategy('X14', sig, wt, result, metrics)
  report.generate_all()
  report.merge_big_chart()

图表清单 (8种):
  1. 净值+回撤组合图 (必组合, 3:1) 
  2. 年度收益对比图 (独立)
  3. 持仓信号+相对强弱 (组合, 2:1)
  4. 月度收益率热力图 (独立)
  5. 交易收益+持有天数 (并排, 1:1)
  6. 交易类型分布三子图 (一行)
  7. 核心指标仪表盘 (2x4)
  8. 滑点敏感性分析 (可选, 1:1)
================================================================
"""
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

# ========== 全局样式 ==========
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150

COLORS = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6', '#1ABC9C']
STYLES = ['-', '--', ':', '-.']
DD_COLORS = ['#E74C3C', '#E67E22', '#F1C40F', '#2ECC71']


@dataclass
class StrategyResult:
    """策略结果容器"""
    name: str
    signal: pd.Series
    weight: pd.Series
    nav: np.ndarray
    dates: np.ndarray
    metrics: dict
    trades: Optional[pd.DataFrame] = None  # 引擎实际逐笔交易记录
    position_series: Optional[pd.Series] = None  # 引擎实际每日持仓
    switches: Optional[dict] = None
    color: str = ''
    linestyle: str = '-'
    
    def __post_init__(self):
        if not self.color:
            color_idx = StrategyResult._color_idx
            StrategyResult._color_idx += 1
            self.color = COLORS[color_idx % len(COLORS)]
    
    @property
    def nav_series(self) -> pd.Series:
        return pd.Series(self.nav, index=pd.to_datetime(self.dates))
    
    @property
    def daily_ret(self) -> pd.Series:
        return self.nav_series.pct_change().dropna()
    
    @property
    def drawdown(self) -> pd.Series:
        return self.nav_series / self.nav_series.cummax() - 1


StrategyResult._color_idx = 0  # class-level counter


class BacktestReport:
    """回测报告生成器"""
    
    def __init__(self, output_dir: str, title: str = '策略回测分析报告'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.title = title
        self.strategies: List[StrategyResult] = []
        self._generated_paths: Dict[str, Path] = {}
    
    def add_strategy(self, name: str, signal: pd.Series, weight: pd.Series,
                     nav: np.ndarray, dates: np.ndarray, metrics: dict,
                     switches: Optional[dict] = None, color: str = '',
                     trades: Optional[pd.DataFrame] = None,
                     position_series: Optional[pd.Series] = None):
        """添加一个策略到报告中"""
        self.strategies.append(StrategyResult(
            name=name, signal=signal, weight=weight,
            nav=nav, dates=dates, metrics=metrics,
            trades=trades, position_series=position_series,
            switches=switches, color=color,
        ))
    
    def add_from_backtest(self, name: str, sig, wt, result, metrics,
                          switches=None, color=''):
        """从回测结果添加策略"""
        trades = result.trades_to_dataframe() if hasattr(result, 'trades_to_dataframe') else None
        # 从引擎结果提取实际持仓序列（考虑T+1执行）
        df_r = result.to_dataframe()
        position_series = pd.Series(
            df_r['position'].values,
            index=pd.to_datetime(df_r['date'].values)
        )
        self.add_strategy(
            name=name, signal=sig, weight=wt,
            nav=result.nav, dates=result.dates,
            metrics=metrics, trades=trades,
            position_series=position_series,
            switches=switches, color=color,
        )
    
    # ─────────────── 1. 净值+回撤组合图 ───────────────
    def chart_nav_drawdown(self) -> Path:
        """净值曲线(上3) + 回撤曲线(下1) 组合图"""
        fig, axes = plt.subplots(2, 1, figsize=(16, 10),
                                 gridspec_kw={'height_ratios': [3, 1]})
        
        _POS_COLORS = {'growth': '#F1C40F', 'value': '#2ECC71', 'cash': '#95A5A6'}
        _POS_LABELS = {'growth': '成长', 'value': '价值', 'cash': '空仓'}
        
        # 上: 净值曲线（按持仓着色）
        ax = axes[0]
        for s in self.strategies:
            nav = s.nav_series
            sig = s.signal
            
            if sig is not None and not sig.empty:
                # 对齐信号索引到净值日期
                sig_aligned = sig.reindex(nav.index).ffill()
                # 找到连续相同信号的分组
                groups = (sig_aligned != sig_aligned.shift()).cumsum()
                for _, idx in groups.groupby(groups):
                    seg = nav.loc[idx.index]
                    pos = sig_aligned.loc[idx.index[0]]
                    color = _POS_COLORS.get(pos, s.color)
                    ax.plot(seg.index, seg.values, color=color,
                            linewidth=2.0, solid_joinstyle='round')
            else:
                ax.plot(nav.index, nav.values, color=s.color,
                        linewidth=2.0)
            
            # 策略名称标注（右上角）
            label = (f"{s.name}  年化{s.metrics['ann']*100:.1f}%  "
                     f"Calmar{s.metrics['calmar']:.3f}")
            ax.text(0.99, 0.01, label, transform=ax.transAxes,
                    fontsize=10, va='bottom', ha='right',
                    color=s.color, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                              edgecolor=s.color, alpha=0.8))
        
        # 持仓颜色图例
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=c, label=_POS_LABELS[p], edgecolor='gray', lw=0.5)
            for p, c in _POS_COLORS.items()
        ]
        ax.legend(handles=legend_elements, fontsize=10,
                  loc='upper left', title='持仓方向')
        
        # 线性坐标 + 人性化格式
        ax.set_ylabel('累计净值', fontsize=11)
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f'{x:.0f}x'))
        ax.grid(True, alpha=0.3)
        ax.set_title(f'{self.title} — 净值曲线（按持仓着色）',
                     fontsize=13, fontweight='bold')
        ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
        
        # 下: 回撤曲线
        ax = axes[1]
        for s in self.strategies:
            dd = s.drawdown
            ax.fill_between(dd.index, dd.values * 100, 0,
                            color=s.color, alpha=0.25, label=f'{s.name}')
            ax.plot(dd.index, dd.values * 100, color=s.color,
                    linestyle=s.linestyle, linewidth=1.2)
            # 标注最大回撤
            min_idx = dd.idxmin()
            min_val = dd.min()
            ax.annotate(f'MaxDD={min_val*100:.1f}%',
                        xy=(min_idx, min_val * 100),
                        xytext=(min_idx, min_val * 100 - 5),
                        fontsize=8, color=s.color,
                        arrowprops=dict(arrowstyle='->', color=s.color, lw=0.8))
        ax.set_ylabel('回撤(%)', fontsize=11)
        ax.set_xlabel('日期', fontsize=11)
        ax.legend(fontsize=9, loc='lower left')
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='black', lw=0.5)
        
        plt.tight_layout()
        path = self.output_dir / '1_净值回撤图.png'
        plt.savefig(path, bbox_inches='tight')
        plt.close()
        self._generated_paths['nav_dd'] = path
        return path
    
    # ─────────────── 2. 年度收益对比图 ───────────────
    def chart_yearly_returns(self) -> Path:
        """年度收益柱状图（独立）"""
        fig, ax = plt.subplots(figsize=(16, 7))
        
        # 收集所有年份
        all_years = set()
        yearly_data = {}
        for s in self.strategies:
            nav = s.nav_series
            yearly = nav.resample('Y').last().pct_change().dropna() * 100
            yearly_data[s.name] = yearly
            for y in yearly.index:
                all_years.add(y.year)
        
        years = sorted(all_years)
        x = np.arange(len(years))
        n = len(self.strategies)
        w = 0.7 / max(n, 1)
        
        for i, s in enumerate(self.strategies):
            vals = []
            for y in years:
                ydata = yearly_data[s.name]
                mask = ydata.index.year == y
                if mask.any():
                    vals.append(ydata[mask].values[0])
                else:
                    vals.append(0)
            offset = (i - (n - 1) / 2) * w
            bars = ax.bar(x + offset, vals, w, color=s.color, alpha=0.85, label=s.name)
            for bar, v in zip(bars, vals):
                if abs(v) > 3:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + (1 if v >= 0 else -5),
                            f'{v:.1f}%', ha='center', fontsize=7)
        
        ax.set_xticks(x)
        ax.set_xticklabels(years, fontsize=10)
        ax.set_ylabel('年收益率(%)', fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, axis='y', alpha=0.3)
        ax.axhline(0, color='black', lw=0.8)
        ax.set_title('年度收益对比', fontsize=13, fontweight='bold')
        
        plt.tight_layout()
        path = self.output_dir / '2_年度收益图.png'
        plt.savefig(path, bbox_inches='tight')
        plt.close()
        self._generated_paths['yearly'] = path
        return path
    
    # ─────────────── 3. 持仓信号+相对强弱 ───────────────
    def chart_signal_relative(self) -> Path:
        """持仓信号区域(上) + 相对强弱(下) 组合图"""
        fig, axes = plt.subplots(2, 1, figsize=(16, 9),
                                 gridspec_kw={'height_ratios': [1.5, 1]})
        
        # 上: 持仓区域
        ax = axes[0]
        for s in self.strategies:
            if s.signal is None:
                continue
            sig_map = s.signal.map({'growth': 1, 'value': -1, 'cash': 0}).fillna(0)
            wt_plot = s.weight * sig_map
            ax.fill_between(s.signal.index, wt_plot.values, 0,
                            color=s.color, alpha=0.3, label=f'{s.name} 持仓方向')
            ax.plot(s.signal.index, wt_plot.values, color=s.color,
                    linewidth=0.5, alpha=0.6)
            
            # 统计切换次数
            if s.switches:
                ax.text(0.02, 0.95 - 0.08 * self.strategies.index(s),
                        f"{s.name}: 方向{s.switches.get('dir',0)}次 "
                        f"空仓{s.switches.get('cash',0)}次",
                        transform=ax.transAxes, fontsize=8, color=s.color,
                        verticalalignment='top')
        
        ax.set_ylabel('方向(+成长/0现金/-价值)', fontsize=11)
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_title('持仓信号时间轴', fontsize=12, fontweight='bold')
        ax.set_ylim(-1.5, 1.5)
        
        # 下: 相对强弱 (策略/基准比值)
        ax = axes[1]
        if len(self.strategies) >= 2:
            base = self.strategies[0]
            for s in self.strategies[1:]:
                ratio = s.nav_series / base.nav_series
                ax.plot(ratio.index, ratio.values, color=s.color,
                        linewidth=1.5, label=f'{s.name}/{base.name}')
            ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
        else:
            # 单策略模式: 指数增长示意
            nav = self.strategies[0].nav_series
            ax.plot(nav.index, nav.values / nav.iloc[0],
                    color=self.strategies[0].color, linewidth=1.5)
        
        ax.set_ylabel('相对强弱比率', fontsize=11)
        ax.set_xlabel('日期', fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_title('相对强弱对比', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        path = self.output_dir / '3_信号强弱图.png'
        plt.savefig(path, bbox_inches='tight')
        plt.close()
        self._generated_paths['signal'] = path
        return path
    
    # ─────────────── 4. 月度收益率热力图 ───────────────
    def chart_monthly_heatmap(self) -> Path:
        """月度收益率热力图（独立）"""
        n = len(self.strategies)
        fig, axes = plt.subplots(1, n, figsize=(8 * n, 7))
        if n == 1:
            axes = [axes]
        
        for idx, s in enumerate(self.strategies):
            ax = axes[idx]
            nav = s.nav_series
            monthly = nav.resample('M').last().pct_change().dropna() * 100
            
            years = sorted(set(d.year for d in monthly.index))
            months = list(range(1, 13))
            heat_data = pd.DataFrame(index=years, columns=months, data=np.nan)
            for d, v in monthly.items():
                if d.year in years and d.month in months:
                    heat_data.loc[d.year, d.month] = v
            
            vmax = max(abs(heat_data.min().min()), abs(heat_data.max().max()))
            cmap = sns.diverging_palette(10, 130, s=80, l=55, as_cmap=True)
            sns.heatmap(heat_data, annot=True, fmt='.1f', cmap=cmap,
                        center=0, vmin=-vmax, vmax=vmax,
                        ax=ax, linewidths=0.5,
                        cbar_kws={'label': '月收益率(%)'},
                        annot_kws={'fontsize': 7})
            ax.set_title(f'{s.name}\n年化{s.metrics["ann"]*100:.1f}% '
                         f'Calmar{s.metrics["calmar"]:.3f}',
                         fontsize=11, fontweight='bold')
            ax.set_ylabel('年份')
            ax.set_xlabel('月份')
        
        plt.suptitle('月度收益率热力图', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        path = self.output_dir / '4_月度热力图.png'
        plt.savefig(path, bbox_inches='tight')
        plt.close()
        self._generated_paths['heatmap'] = path
        return path
    
    # ─────────────── 5. 交易收益分布+持有天数 ───────────────
    def chart_trade_distribution(self) -> Path:
        """交易收益分布直方图 + 持有天数分布（左右并排）
        
        每笔交易 = 相邻两次信号变化之间的持仓区间。
        用信号变化点捕捉所有调仓（~397次），而非仅现金→持仓→现金周期（~100次）。
        """
        n = len(self.strategies)
        fig, axes = plt.subplots(n, 2, figsize=(16, 5 * n))
        
        for idx, s in enumerate(self.strategies):
            if n > 1:
                ax_ret, ax_hold = axes[idx]
            else:
                ax_ret, ax_hold = axes
            
            if s.signal is not None:
                sig = s.signal
                # 找到所有信号变化点（不区分方向还是现金，每个变化都是一次调仓）
                changed = sig != sig.shift(1)
                change_dates = sig[changed].index
                
                trade_rets = []
                hold_days = []
                for i in range(1, len(change_dates)):
                    entry_date = change_dates[i - 1]
                    exit_date = change_dates[i]
                    ret = s.nav_series.loc[exit_date] / s.nav_series.loc[entry_date] - 1
                    trade_rets.append(ret * 100)
                    hold_days.append((exit_date - entry_date).days)
                
                # 收益分布
                if trade_rets:
                    colors_bar = ['#2ECC71' if r >= 0 else '#E74C3C' for r in trade_rets]
                    ax_ret.bar(range(len(trade_rets)), trade_rets, color=colors_bar, width=0.7)
                    ax_ret.axhline(0, color='gray', lw=0.5)
                    mean_ret = np.mean(trade_rets)
                    win_rate = sum(1 for r in trade_rets if r > 0) / len(trade_rets) * 100
                    ax_ret.axhline(mean_ret, color='blue', linestyle='--', lw=1.0)
                    ax_ret.text(len(trade_rets) * 0.7, mean_ret,
                                f'均值={mean_ret:.2f}%  胜率={win_rate:.1f}%',
                                fontsize=9, color='blue')
                    # 明确设置 x 轴范围以匹配实际交易数
                    ax_ret.set_xlim(-1, len(trade_rets))
                ax_ret.set_title(f'{s.name} 逐笔交易收益 ({len(trade_rets)}笔)',
                                 fontsize=11, fontweight='bold')
                ax_ret.set_ylabel('收益率(%)')
                ax_ret.set_xlabel('交易序号')
                ax_ret.grid(True, alpha=0.3)
                
                # 持有天数分布
                if hold_days:
                    ax_hold.hist(hold_days, bins=min(30, len(set(hold_days))),
                                 color=s.color, alpha=0.7, edgecolor='white')
                    mean_hold = np.mean(hold_days)
                    ax_hold.axvline(mean_hold, color='blue', linestyle='--', lw=1.0)
                    ax_hold.text(mean_hold + 1, ax_hold.get_ylim()[1] * 0.9,
                                 f'平均{mean_hold:.0f}天', fontsize=9, color='blue')
                ax_hold.set_title(f'{s.name} 持有天数分布', fontsize=11, fontweight='bold')
                ax_hold.set_ylabel('次数')
                ax_hold.set_xlabel('持有天数')
                ax_hold.grid(True, alpha=0.3)
        
        plt.suptitle('交易分析', fontsize=14, fontweight='bold')
        plt.tight_layout()
        path = self.output_dir / '5_交易分布图.png'
        plt.savefig(path, bbox_inches='tight')
        plt.close()
        self._generated_paths['trades'] = path
        return path
    
    # ─────────────── 6. 交易类型分布三子图 ───────────────
    def chart_trade_type(self) -> Path:
        """交易类型分布三子图（饼图+胜率+平均收益）"""
        n = len(self.strategies)
        fig, axes = plt.subplots(n, 3, figsize=(18, 5 * n))
        
        for idx, s in enumerate(self.strategies):
            if n > 1:
                row = axes[idx]
            else:
                row = axes
            
            if s.signal is None or s.weight is None:
                continue
            
            # 按信号分类
            sig = s.signal
            wt = s.weight
            
            # 方向天数
            growth_days = (sig == 'growth').sum()
            value_days = (sig == 'value').sum()
            cash_days = (sig == 'cash').sum() if 'cash' in sig.values else (wt <= 0).sum()
            
            # 饼图
            ax = row[0]
            labels = ['成长', '价值', '空仓']
            sizes = [growth_days, value_days, cash_days]
            colors_pie = ['#E74C3C', '#3498DB', '#95A5A6']
            ax.pie([s for s in sizes if s > 0],
                   labels=[l for l, s in zip(labels, sizes) if s > 0],
                   autopct='%1.0f%%', colors=[c for c, s in zip(colors_pie, sizes) if s > 0],
                   startangle=90)
            ax.set_title(f'{s.name} 方向占比', fontsize=11, fontweight='bold')
            
            # 胜率(按日) — 分别计算成长/价值持仓日胜率
            ax = row[1]
            daily_ret = s.daily_ret
            overall_win = (daily_ret > 0).mean() * 100
            
            # 使用引擎实际持仓区分品种
            if s.position_series is not None:
                pos = s.position_series.reindex(daily_ret.index).ffill()
            elif s.weight is not None:
                wt_aligned = s.weight.reindex(daily_ret.index, method='ffill')
                pos = pd.Series('cash', index=daily_ret.index)
                pos[wt_aligned > 0] = 'growth'
            else:
                pos = pd.Series('cash', index=daily_ret.index)
            
            # 分别计算成长/价值的胜率（空仓日收益≈0，跳过）
            win_data = {}
            for label, mask_val in [('成长', 'growth'), ('价值', 'value')]:
                mask = pos == mask_val
                if mask.sum() > 0:
                    ret_in_pos = daily_ret[mask].dropna()
                    win_data[label] = (ret_in_pos > 0).mean() * 100 if len(ret_in_pos) > 0 else 0
            
            if win_data:
                names = list(win_data.keys())
                values = list(win_data.values())
                colors_bar = ['#E74C3C', '#3498DB']  # 红=成长, 蓝=价值
                bars = ax.bar(names, values, color=colors_bar[:len(names)], width=0.5)
                ax.axhline(overall_win, color='red', linestyle='--', lw=1.0,
                           label=f'总胜率{overall_win:.1f}%')
                for bar, v in zip(bars, values):
                    ax.text(bar.get_x() + bar.get_width() / 2, v + 1,
                            f'{v:.1f}%', ha='center', fontsize=9)
            ax.set_title(f'{s.name} 按品种日胜率', fontsize=11, fontweight='bold')
            ax.set_ylabel('胜率(%)')
            ax.legend(fontsize=8)
            ax.set_ylim(0, 100)
            ax.grid(True, axis='y', alpha=0.3)
            
            # 平均日收益（分别计算成长/价值）
            ax = row[2]
            avg_data = {}
            for label, mask_val in [('成长', 'growth'), ('价值', 'value')]:
                mask = pos == mask_val
                if mask.sum() > 0:
                    ret_in_pos = daily_ret[mask].dropna()
                    avg_data[label] = ret_in_pos.mean() * 100 if len(ret_in_pos) > 0 else 0
            
            if avg_data:
                names = list(avg_data.keys())
                values = list(avg_data.values())
                colors_bar = ['#E74C3C', '#3498DB']
                bars = ax.bar(names, values, color=colors_bar[:len(names)], width=0.5)
                overall_avg = daily_ret.mean() * 100
                ax.axhline(overall_avg, color='red', linestyle='--', lw=1.0,
                           label=f'总均值{overall_avg:.3f}%')
                for bar, v in zip(bars, values):
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            v + 0.002 if v >= 0 else v - 0.005,
                            f'{v:.3f}%', ha='center', fontsize=9)
            ax.set_title(f'{s.name} 按品种日平均收益', fontsize=11, fontweight='bold')
            ax.set_ylabel('平均日收益(%)')
            ax.legend(fontsize=8)
            ax.grid(True, axis='y', alpha=0.3)
        
        plt.suptitle('交易类型分析', fontsize=14, fontweight='bold')
        plt.tight_layout()
        path = self.output_dir / '6_交易类型图.png'
        plt.savefig(path, bbox_inches='tight')
        plt.close()
        self._generated_paths['trade_type'] = path
        return path
    
    # ─────────────── 7. 核心指标仪表盘 ───────────────
    def chart_dashboard(self) -> Path:
        """核心指标仪表盘 (2x4)"""
        n = len(self.strategies)
        fig, axes = plt.subplots(n, 8, figsize=(20, 4 * n), squeeze=False)
        
        metrics_display = [
            ('年化收益', lambda m: f"{m['ann']*100:.1f}%", '#2ECC71'),
            ('Sharpe', lambda m: f"{m['sharpe']:.2f}", '#3498DB'),
            ('Calmar', lambda m: f"{m['calmar']:.2f}", '#E74C3C'),
            ('最大回撤', lambda m: f"{m['dd']*100:.1f}%", '#E67E22'),
            ('总收益率', lambda m: f"{m['total_return']*100:.0f}%", '#9B59B6'),
            ('交易次数', lambda m: f"{m['n_trades']}", '#1ABC9C'),
            ('日胜率', lambda m: f"{m.get('win_rate',0)*100:.1f}%", '#F39C12'),
            ('年化波动', lambda m: f"{m['vol']*100:.2f}%", '#E74C4A'),
        ]
        
        for idx, s in enumerate(self.strategies):
            row = axes[idx]
            m = s.metrics
            
            # 计算日胜率
            daily_ret = s.daily_ret
            m['win_rate'] = (daily_ret > 0).mean()
            m['total_return'] = s.nav[-1] / s.nav[0] - 1
            m['vol'] = daily_ret.std() * np.sqrt(252)
            
            for col, (label, fmt_func, color) in enumerate(metrics_display):
                ax = row[col]
                ax.text(0.5, 0.6, fmt_func(m), ha='center', va='center',
                        fontsize=24, fontweight='bold', color=color)
                ax.text(0.5, 0.15, label, ha='center', va='center',
                        fontsize=11, color='gray')
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.axis('off')
                # 顶部色条
                ax.axhline(0.85, 0.1, 0.9, color=color, linewidth=4)
            
            # 策略名称
            fig.text(0.02, 0.98 - idx * (0.85 / n), s.name,
                     fontsize=12, fontweight='bold', color=s.color,
                     transform=fig.transFigure)
        
        plt.suptitle('核心指标仪表盘', fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0.05, 0, 1, 0.95])
        path = self.output_dir / '7_指标仪表盘.png'
        plt.savefig(path, bbox_inches='tight')
        plt.close()
        self._generated_paths['dashboard'] = path
        return path
    
    # ─────────────── 8. 滑点敏感性分析 ───────────────
    def chart_slippage_sensitivity(self, base_slippage=0.0005,
                                   slippage_range: Optional[List[float]] = None) -> Path:
        """滑点敏感性分析（左右并排）"""
        if len(self.strategies) == 0:
            return None
        
        if slippage_range is None:
            slippage_range = [0.0, 0.0003, 0.0005, 0.0008, 0.0010, 0.0015, 0.0020]
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # 对每个策略跑不同滑点
        from optimize_runner import run_backtest, calc_metrics
        
        for s in self.strategies:
            anns = []
            calmars = []
            for slp in slippage_range:
                try:
                    r = run_backtest(s.signal, s.weight, impact_slippage=slp)
                    m = calc_metrics(r)
                    anns.append(m['ann'] * 100)
                    calmars.append(m['calmar'])
                except:
                    anns.append(0)
                    calmars.append(0)
            
            slp_bps = [x * 10000 for x in slippage_range]
            
            ax = axes[0]
            ax.plot(slp_bps, anns, 'o-', color=s.color, linewidth=2,
                    markersize=6, label=s.name)
            ax.set_xlabel('冲击滑点(bps)', fontsize=11)
            ax.set_ylabel('年化收益率(%)', fontsize=11)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_title('滑点敏感性 — 年化', fontsize=12, fontweight='bold')
            
            # 标注当前滑点
            base_idx = slippage_range.index(base_slippage) if base_slippage in slippage_range else 0
            ax.axvline(base_slippage * 10000, color='gray', linestyle=':', alpha=0.5)
            ax.annotate(f'当前{base_slippage*10000:.0f}bps',
                        xy=(base_slippage * 10000, anns[base_idx]),
                        fontsize=8, color='gray')
            
            ax = axes[1]
            ax.plot(slp_bps, calmars, 's-', color=s.color, linewidth=2,
                    markersize=6, label=s.name)
            ax.set_xlabel('冲击滑点(bps)', fontsize=11)
            ax.set_ylabel('Calmar', fontsize=11)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_title('滑点敏感性 — Calmar', fontsize=12, fontweight='bold')
            ax.axvline(base_slippage * 10000, color='gray', linestyle=':', alpha=0.5)
        
        plt.suptitle('滑点敏感性分析', fontsize=14, fontweight='bold')
        plt.tight_layout()
        path = self.output_dir / '8_滑点敏感性.png'
        plt.savefig(path, bbox_inches='tight')
        plt.close()
        self._generated_paths['slippage'] = path
        return path
    
    # ─────────────── 9. 大图合成 ───────────────
    def merge_big_chart(self, chart_order: Optional[List[str]] = None,
                        output_name: str = '0_完整报告图.png') -> Path:
        """将所有图表合成为一张大图"""
        if chart_order is None:
            chart_order = ['nav_dd', 'yearly', 'signal', 'heatmap',
                          'trades', 'trade_type', 'dashboard']
        
        valid_paths = []
        for key in chart_order:
            if key in self._generated_paths and self._generated_paths[key].exists():
                valid_paths.append(self._generated_paths[key])
        
        if not valid_paths:
            return None
        
        from PIL import Image
        import io
        
        images = []
        total_height = 0
        max_width = 0
        
        for path in valid_paths:
            img = Image.open(path)
            # 统一宽度
            target_w = 2400
            ratio = target_w / img.width
            target_h = int(img.height * ratio)
            img_resized = img.resize((target_w, target_h), Image.LANCZOS)
            images.append(img_resized)
            total_height += target_h + 30  # 30px gap
            max_width = max(max_width, target_w)
        
        # 创建大画布
        big_img = Image.new('RGB', (max_width, total_height), color='white')
        y_offset = 0
        for img in images:
            x_offset = (max_width - img.width) // 2
            big_img.paste(img, (x_offset, y_offset))
            y_offset += img.height + 30
        
        path = self.output_dir / output_name
        big_img.save(path, quality=95)
        self._generated_paths['big_chart'] = path
        return path
    
    # ─────────────── 10. 生成TXT报告 ───────────────
    def generate_report_txt(self) -> Path:
        """生成文本报告"""
        path = self.output_dir / '回测报告.txt'
        with open(path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write(f"  {self.title}\n")
            f.write(f"  生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("一、核心指标对比\n")
            f.write("-" * 50 + "\n")
            f.write(f"  {'策略':<20} {'年化':>8} {'回撤':>8} {'Sharpe':>8} "
                    f"{'Calmar':>8} {'交易':>6} {'总收益':>10}\n")
            f.write("  " + "-" * 68 + "\n")
            for s in self.strategies:
                m = s.metrics
                tr = s.nav[-1] / s.nav[0] - 1
                f.write(f"  {s.name:<20} {m['ann']*100:>7.2f}% {m['dd']*100:>7.2f}% "
                        f"{m['sharpe']:>8.3f} {m['calmar']:>8.3f} {m['n_trades']:>6} "
                        f"{tr*100:>9.1f}%\n")
            f.write("\n")
            
            f.write("二、参数设置\n")
            f.write("-" * 50 + "\n")
            f.write("  滑点: 手续费 1bps + 冲击滑点 5bps = 6bps/边\n")
            f.write("  跳空滑点: 不计（高开低开期望值为0）\n")
            f.write("  调仓时序: T日收盘信号 → T+1开盘执行\n\n")
            
            f.write("三、生成图表清单\n")
            f.write("-" * 50 + "\n")
            chart_names = {
                'nav_dd': '1_净值回撤图.png — 净值曲线+回撤曲线组合',
                'yearly': '2_年度收益图.png — 逐年收益对比柱状图',
                'signal': '3_信号强弱图.png — 持仓信号+相对强弱',
                'heatmap': '4_月度热力图.png — 月度收益率热力图',
                'trades': '5_交易分布图.png — 逐笔收益+持有天数',
                'trade_type': '6_交易类型图.png — 方向占比/胜率/收益',
                'dashboard': '7_指标仪表盘.png — 核心指标2x8展示',
                'slippage': '8_滑点敏感性.png — 滑点vs年化/Calmar',
                'big_chart': '0_完整报告图.png — 所有图表合成大图',
            }
            for key, desc in chart_names.items():
                if key in self._generated_paths:
                    f.write(f"  ✓ {desc}\n")
                else:
                    f.write(f"  ✗ {desc}\n")
            f.write("\n")
            
            f.write("四、策略列表\n")
            for s in self.strategies:
                f.write(f"  {s.name}\n")
            f.write("\n")
        
        self._generated_paths['report_txt'] = path
        return path
    
    # ─────────────── 全部生成 ───────────────
    def generate_all(self, skip_slippage=False) -> Dict[str, Path]:
        """生成所有图表"""
        print("生成图表...")
        self.chart_nav_drawdown()
        print("  ✓ 1/7 净值回撤图")
        self.chart_yearly_returns()
        print("  ✓ 2/7 年度收益图")
        self.chart_signal_relative()
        print("  ✓ 3/7 信号强弱图")
        self.chart_monthly_heatmap()
        print("  ✓ 4/7 月度热力图")
        self.chart_trade_distribution()
        print("  ✓ 5/7 交易分布图")
        self.chart_trade_type()
        print("  ✓ 6/7 交易类型图")
        self.chart_dashboard()
        print("  ✓ 7/7 指标仪表盘")
        
        if not skip_slippage:
            try:
                self.chart_slippage_sensitivity()
                print("  ✓ 8/8 滑点敏感性")
            except Exception as e:
                print(f"  ✗ 滑点敏感性跳过: {e}")
        
        self.generate_report_txt()
        print("  ✓ 报告文本")
        
        return dict(self._generated_paths)
