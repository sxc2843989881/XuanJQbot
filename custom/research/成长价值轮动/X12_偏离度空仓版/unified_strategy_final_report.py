"""unified_strategy_final_report.py — 统一策略最终报告生成
================================================================
为统一策略29轮优化的最优版本生成完整回测报告:
  - v26 U164 (无滑点最优): Calmar 2.378 年化49.79% 回撤-20.94%
  - v27 U175_cd9_sw0.16 (滑点最优): 滑点Calmar 2.059
  - X61 (基准对比): Calmar 1.361 年化41.12% 回撤-30.21%

输出:
  - 日收益率序列 CSV
  - 信号/权重序列 CSV
  - 交易记录 CSV
  - 月度收益率矩阵 CSV
  - 年度收益率 CSV
  - 指数行情+因子 CSV
  - 完整报告 JSON/TXT
  - 可视化图表 PNG (资金曲线/回撤/因子信号/月度热力图)
================================================================
"""
import sys
sys.path.insert(0, r'c:\XuanJLH\Qbot\custom\backtests')
sys.path.insert(0, r'c:\XuanJLH\Qbot\custom\research\成长价值轮动\X12_偏离度空仓版')

from pathlib import Path
import numpy as np
import pandas as pd
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec

from optimize_runner import (
    G_CLOSE, V_CLOSE, RATIO, RATIO_MA20, RATIO_DEV, MA20_SLOPE,
    V_MOM20, G_DD20, V_DD20, BASE_DIR,
    run_backtest, calc_metrics, count_switches,
)
from run_x33_reduce_trades import RATIO_DEV_STD20, RATIO_DEV_Z

# 中文字体设置
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = Path(r'c:\XuanJLH\Qbot\custom\research\成长价值轮动\X12_偏离度空仓版\回测结果\unified_strategy')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

T = RATIO_DEV_Z
SLOPE = MA20_SLOPE


# ============================================================
# 统一策略核心构建函数 (v27版本, 包含所有最新逻辑)
# ============================================================
def build_core(slope_thresh=0.002, sw=0.17, st=0.088, cd=8,
               ms=10, ml=20, rt=1.3, dc=5, dcd=6,
               bias_ma=20, bias_high=0.19, bias_reduce=0.0,
               use_max_hold=True, max_hold_days=92, max_hold_reduce=0.0,
               hold_mode='reset_dir'):
    """统一策略最优核心: Br=0完全空仓 + MHr=0完全空仓 + dcd=6"""
    G_MA = G_CLOSE.rolling(bias_ma).mean()
    V_MA = V_CLOSE.rolling(bias_ma).mean()
    G_BIAS = (G_CLOSE / G_MA - 1)
    V_BIAS = (V_CLOSE / V_MA - 1)

    V_MOM_S = V_CLOSE.pct_change(ms)
    V_MOM_L = V_CLOSE.pct_change(ml)

    raw_dir = (T > 0).map({True: 'BULL', False: 'BEAR'})
    mask = np.ones(len(raw_dir), dtype=bool)
    for k in range(1, dc):
        mask = mask & (raw_dir.values == raw_dir.shift(k).values)
    confirmed_dir = raw_dir.where(mask, np.nan).ffill().fillna('BULL')

    if dcd > 0:
        new_dir = confirmed_dir.copy()
        last_switch = -dcd - 1
        prev = confirmed_dir.iloc[0]
        for i in range(len(confirmed_dir)):
            if pd.isna(confirmed_dir.iloc[i]):
                new_dir.iloc[i] = prev
                continue
            if confirmed_dir.iloc[i] != prev:
                if i - last_switch >= dcd:
                    last_switch = i
                    prev = confirmed_dir.iloc[i]
                new_dir.iloc[i] = prev
            else:
                new_dir.iloc[i] = prev
        confirmed_dir = new_dir

    dir_raw = confirmed_dir
    weak_slope = SLOPE.abs() < slope_thresh
    weak_t = T.abs() < rt
    is_weak = weak_t & weak_slope

    wt = pd.Series(1.0, index=T.index)
    wt[is_weak] = 0.0

    wrong_value = (dir_raw == 'BEAR') & (V_MOM_S <= 0) & (V_MOM_L <= 0)
    dir_raw[wrong_value] = 'BULL'
    dir_s = dir_raw.map({'BULL': 'growth', 'BEAR': 'value'})

    # BIAS过滤(Br=0直接空仓)
    extreme_g = (dir_s == 'growth') & (G_BIAS > bias_high)
    extreme_v = (dir_s == 'value') & (V_BIAS > bias_high)
    extreme = extreme_g | extreme_v
    wt[extreme] = wt[extreme] * bias_reduce

    # E5止损
    gs = (dir_s == 'growth') & (G_DD20 < -st)
    vs = (dir_s == 'value') & (V_DD20 < -st)
    e5_trigger = gs | vs

    in_cooldown = False
    cooldown_count = 0
    for i in range(len(wt)):
        if pd.isna(wt.iloc[i]) or pd.isna(dir_s.iloc[i]):
            continue
        if e5_trigger.iloc[i] and not in_cooldown:
            in_cooldown = True
            cooldown_count = 0
            wt.iloc[i] = wt.iloc[i] * sw
        elif in_cooldown:
            cooldown_count += 1
            if cooldown_count >= cd:
                if e5_trigger.iloc[i]:
                    cooldown_count = 0
                    wt.iloc[i] = wt.iloc[i] * sw
                else:
                    in_cooldown = False
                    if is_weak.iloc[i]:
                        wt.iloc[i] = 0.0
                    else:
                        wt.iloc[i] = 1.0
            else:
                if wt.iloc[i] > 0:
                    wt.iloc[i] = sw

    # 持仓时间限制(MHr=0直接空仓)
    if use_max_hold:
        hold_count = 0
        prev_key = None
        for i in range(len(wt)):
            if pd.isna(wt.iloc[i]) or pd.isna(dir_s.iloc[i]):
                continue
            if hold_mode == 'reset_dir':
                key = dir_s.iloc[i]
            elif hold_mode == 'reset_pos':
                key = (dir_s.iloc[i], round(wt.iloc[i], 2))
            else:
                key = None

            if key != prev_key:
                hold_count = 0
                prev_key = key
            else:
                hold_count += 1
                if hold_count >= max_hold_days:
                    if wt.iloc[i] > 0:
                        wt.iloc[i] = wt.iloc[i] * max_hold_reduce

    signal = dir_raw.map({'BULL': 'growth', 'BEAR': 'value'})
    return signal, wt


# ============================================================
# 1. 运行回测并收集数据
# ============================================================
def run_detailed_backtest(name, signal, weight):
    """运行完整回测并收集所有数据"""
    common_idx = signal.index.intersection(G_CLOSE.index)
    sig = signal.loc[common_idx]
    wt = weight.loc[common_idx]
    g_c = G_CLOSE.loc[common_idx]
    v_c = V_CLOSE.loc[common_idx]
    mask = ~(sig.isna() | wt.isna())
    sig = sig[mask].astype(str)
    wt = wt[mask].astype(float)
    g_c = g_c[mask]
    v_c = v_c[mask]

    result = run_backtest(signal, weight)
    result_sl = run_backtest(signal, weight, impact_slippage=0.0005)

    m = calc_metrics(result)
    m_sl = calc_metrics(result_sl)
    sw = count_switches(signal, weight)

    df_r = result.to_dataframe()
    df_r_sl = result_sl.to_dataframe()

    return {
        'name': name,
        'signal': sig, 'weight': wt,
        'g_close': g_c, 'v_close': v_c,
        'result': result, 'result_sl': result_sl,
        'metrics': m, 'metrics_sl': m_sl,
        'switches': sw,
        'daily_returns': df_r,
        'daily_returns_sl': df_r_sl,
    }


# ============================================================
# 2. 交易级别分析
# ============================================================
def analyze_trades(sig, wt, g_close, v_close):
    """分析每次调仓的类型、盈亏、持有天数"""
    df = pd.DataFrame({'signal': sig.astype(str), 'weight': wt.astype(float),
                       'g_close': g_close, 'v_close': v_close})

    df['prev_sig'] = df['signal'].shift(1)
    df['prev_wt'] = df['weight'].shift(1)
    df['prev_g'] = df['g_close'].shift(1)
    df['prev_v'] = df['v_close'].shift(1)

    df['daily_ret'] = 0.0
    g_ret = df['g_close'] / df['prev_g'] - 1
    v_ret = df['v_close'] / df['prev_v'] - 1
    is_growth = df['signal'] == 'growth'
    df.loc[is_growth, 'daily_ret'] = g_ret[is_growth] * df.loc[is_growth, 'weight']
    df.loc[~is_growth, 'daily_ret'] = v_ret[~is_growth] * df.loc[~is_growth, 'weight']
    df['daily_ret'] = df['daily_ret'].fillna(0)
    df['equity'] = (1 + df['daily_ret']).cumprod()

    df['wt_changed'] = (df['weight'] != df['prev_wt']) | (df['signal'] != df['prev_sig'])

    trades = []
    start_idx = None
    prev_sig_val = None
    prev_wt_val = None

    for i, row in df.iterrows():
        if start_idx is None:
            if row['wt_changed']:
                start_idx = i
                prev_sig_val = row['signal']
                prev_wt_val = row['weight']
            continue

        if row['wt_changed']:
            end_idx = i
            curr_sig = df.loc[end_idx, 'signal']
            curr_wt = df.loc[end_idx, 'weight']
            prev_sig = prev_sig_val
            prev_wt = prev_wt_val

            if pd.isna(prev_wt) or pd.isna(curr_wt):
                start_idx = end_idx
                prev_sig_val = curr_sig
                prev_wt_val = curr_wt
                continue

            if prev_wt == 0 and curr_wt > 0:
                ttype = '空仓恢复'
            elif prev_wt > 0 and curr_wt == 0:
                ttype = '进入空仓'
            elif prev_sig != curr_sig and prev_wt > 0 and curr_wt > 0:
                ttype = '方向切换'
            elif prev_wt != curr_wt and prev_sig == curr_sig:
                ttype = 'E5调整'
            else:
                ttype = '其他'

            hold_days = (end_idx - start_idx).days if hasattr(end_idx - start_idx, 'days') else 0
            eq_chunk = df.loc[start_idx:end_idx, 'equity']
            ret = (eq_chunk.iloc[-1] / eq_chunk.iloc[0] - 1) if len(eq_chunk) > 1 else 0

            trades.append({
                'start_date': start_idx,
                'end_date': end_idx,
                'type': ttype,
                'return': ret,
                'hold_days': hold_days,
                'start_dir': df.loc[start_idx, 'signal'],
                'end_dir': curr_sig,
                'start_wt': prev_wt,
                'end_wt': curr_wt,
            })

            start_idx = end_idx
            prev_sig_val = curr_sig
            prev_wt_val = curr_wt

    trades_df = pd.DataFrame(trades)
    return trades_df, df


# ============================================================
# 3. 生成报告数据
# ============================================================
def generate_report(data, strategy_name, strategy_rules, strategy_params):
    """生成完整回测报告"""
    m = data['metrics']
    m_sl = data['metrics_sl']
    sw = data['switches']
    df_r = data['daily_returns']

    dates = data['signal'].index
    r = df_r['daily_ret'].values
    r_series = pd.Series(r, index=dates[:len(r)])

    win_rate = (r_series > 0).mean()
    avg_win = r_series[r_series > 0].mean() if (r_series > 0).any() else 0
    avg_loss = r_series[r_series < 0].mean() if (r_series < 0).any() else 0
    profit_factor = (r_series[r_series > 0].sum() / abs(r_series[r_series < 0].sum())) if (r_series < 0).any() and r_series[r_series < 0].sum() != 0 else np.inf

    df_monthly = pd.DataFrame({'daily_ret': r_series})
    df_monthly['year'] = r_series.index.year
    df_monthly['month'] = r_series.index.month
    monthly_ret = df_monthly.groupby(['year', 'month'])['daily_ret'].apply(lambda x: (1 + x).prod() - 1)
    monthly_win_rate = (monthly_ret > 0).mean()
    monthly_mean = monthly_ret.mean()
    monthly_std = monthly_ret.std()
    monthly_sharpe = monthly_mean / monthly_std * np.sqrt(12) if monthly_std > 0 else 0

    yearly_ret = df_monthly.groupby('year')['daily_ret'].apply(lambda x: (1 + x).prod() - 1)
    yearly_win_rate = (yearly_ret > 0).mean()

    report = {
        '策略名称': strategy_name,
        '策略规则': strategy_rules,
        '核心参数': strategy_params,
        '核心指标': {
            '年化收益': f"{m['ann']*100:.2f}%",
            '最大回撤': f"{m['dd']*100:.2f}%",
            'Sharpe比率': round(m['sharpe'], 3),
            'Calmar比率': round(m['calmar'], 3),
            '总收益率': f"{m['total_return']*100:.2f}%",
            '交易次数': m['n_trades'],
            '方向切换次数': sw['dir'],
            '空仓切换次数': sw['cash'],
            '数据起始日': str(r_series.index[0].date()),
            '数据结束日': str(r_series.index[-1].date()),
            '交易日数': len(r_series),
        },
        '滑点测试(5/万)': {
            '年化收益': f"{m_sl['ann']*100:.2f}%",
            '最大回撤': f"{m_sl['dd']*100:.2f}%",
            'Sharpe比率': round(m_sl['sharpe'], 3),
            'Calmar比率': round(m_sl['calmar'], 3),
            '总收益率': f"{m_sl['total_return']*100:.2f}%",
        },
        '日收益率统计': {
            '日收益率均值': f"{r_series.mean()*100:.4f}%",
            '日收益率标准差': f"{r_series.std()*100:.4f}%",
            '日收益率偏度': round(r_series.skew(), 4),
            '日收益率峰度': round(r_series.kurtosis(), 4),
            '日胜率': f"{win_rate*100:.2f}%",
            '平均盈利': f"{avg_win*100:.4f}%",
            '平均亏损': f"{avg_loss*100:.4f}%",
            '盈亏比': f"{abs(avg_win/avg_loss):.4f}" if avg_loss != 0 else "N/A",
            '利润因子': round(profit_factor, 4),
            '最大单日涨幅': f"{r_series.max()*100:.4f}%",
            '最大单日跌幅': f"{r_series.min()*100:.4f}%",
        },
        '月度统计': {
            '月胜率': f"{monthly_win_rate*100:.2f}%",
            '月均收益': f"{monthly_mean*100:.4f}%",
            '月收益标准差': f"{monthly_std*100:.4f}%",
            '月化Sharpe': round(monthly_sharpe, 4),
            '最佳月份': f"{monthly_ret.max()*100:.4f}%",
            '最差月份': f"{monthly_ret.min()*100:.4f}%",
        },
        '年度统计': {
            '年胜率': f"{yearly_win_rate*100:.2f}%",
            '最佳年份': f"{yearly_ret.max()*100:.2f}%",
            '最差年份': f"{yearly_ret.min()*100:.2f}%",
        },
        '年度收益率明细': {str(yr): f"{r*100:.2f}%" for yr, r in yearly_ret.items()},
    }

    return report, r_series, yearly_ret, monthly_ret


# ============================================================
# 4. 导出数据到CSV
# ============================================================
def export_data(name_prefix, data, report, trades_df, r_series):
    """导出所有回测数据到CSV文件"""
    dates = data['signal'].index[:len(r_series)]
    r_values = r_series.values.astype(float)

    # 1. 日收益率序列
    eq = (1 + r_values).cumprod()
    peak = np.maximum.accumulate(eq)
    dd = eq / peak - 1
    df_export = pd.DataFrame({
        'date': dates.strftime('%Y-%m-%d'),
        'daily_return': r_values,
        'equity': eq,
        'drawdown': dd,
    })
    df_export.to_csv(str(OUTPUT_DIR / f'{name_prefix}_daily_returns.csv'), index=False, encoding='utf-8')

    # 2. 信号和权重序列
    sig_df = pd.DataFrame({
        'date': data['signal'].index.strftime('%Y-%m-%d'),
        'signal': data['signal'].values,
        'weight': data['weight'].values,
    })
    sig_df.to_csv(str(OUTPUT_DIR / f'{name_prefix}_signal_weight.csv'), index=False, encoding='utf-8')

    # 3. 交易记录
    if len(trades_df) > 0:
        trades_out = trades_df.copy()
        trades_out['start_date'] = trades_out['start_date'].dt.strftime('%Y-%m-%d')
        trades_out['end_date'] = trades_out['end_date'].dt.strftime('%Y-%m-%d')
        trades_out.to_csv(str(OUTPUT_DIR / f'{name_prefix}_trades.csv'), index=False, encoding='utf-8')

    # 4. 月度收益率矩阵
    r_s = pd.Series(r_values, index=dates)
    df_m = pd.DataFrame({'daily_ret': r_s})
    df_m['year'] = r_s.index.year
    df_m['month'] = r_s.index.month
    monthly_matrix = df_m.groupby(['year', 'month'])['daily_ret'].apply(
        lambda x: (1 + x).prod() - 1
    ).unstack()
    monthly_matrix.to_csv(str(OUTPUT_DIR / f'{name_prefix}_monthly_returns.csv'), encoding='utf-8')

    # 5. 年度收益率
    yearly_ret = df_m.groupby('year')['daily_ret'].apply(lambda x: (1 + x).prod() - 1)
    yearly_ret.to_csv(str(OUTPUT_DIR / f'{name_prefix}_yearly_returns.csv'), encoding='utf-8')

    # 6. 指数行情+因子数据
    idx = data['signal'].index.intersection(G_CLOSE.index).intersection(V_CLOSE.index)
    G_MA = G_CLOSE.rolling(20).mean()
    V_MA = V_CLOSE.rolling(20).mean()
    G_BIAS = (G_CLOSE / G_MA - 1)
    V_BIAS = (V_CLOSE / V_MA - 1)
    idx_df = pd.DataFrame({
        'date': idx.strftime('%Y-%m-%d'),
        'growth_close': G_CLOSE.loc[idx].values,
        'value_close': V_CLOSE.loc[idx].values,
        'ratio': (G_CLOSE / V_CLOSE).loc[idx].values,
        'ratio_ma20': RATIO_MA20.loc[idx].values,
        'ratio_dev': RATIO_DEV.loc[idx].values,
        'ratio_dev_z': RATIO_DEV_Z.loc[idx].values,
        'ma20_slope': MA20_SLOPE.loc[idx].values,
        'g_bias': G_BIAS.loc[idx].values,
        'v_bias': V_BIAS.loc[idx].values,
        'g_dd20': G_DD20.loc[idx].values,
        'v_dd20': V_DD20.loc[idx].values,
    })
    idx_df.to_csv(str(OUTPUT_DIR / f'{name_prefix}_index_data.csv'), index=False, encoding='utf-8')

    # 7. 完整报告JSON
    with open(str(OUTPUT_DIR / f'{name_prefix}_report.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # 8. 交易类型统计CSV
    if len(trades_df) > 0:
        type_stats = trades_df.groupby('type').agg({
            'return': ['count', 'mean', 'sum'],
            'hold_days': 'mean'
        })
        type_stats.columns = ['count', 'mean_return', 'sum_return', 'avg_hold_days']
        type_stats = type_stats.reset_index()
        type_stats['win_rate'] = trades_df.groupby('type')['return'].apply(
            lambda x: (x > 0).mean()
        ).values * 100
        type_stats.to_csv(str(OUTPUT_DIR / f'{name_prefix}_trade_type_stats.csv'), index=False, encoding='utf-8')

    print(f"  [{name_prefix}] 数据导出完成")


# ============================================================
# 5. 可视化图表生成
# ============================================================
def plot_strategy_report(name_prefix, data, r_series, trades_df):
    """生成完整可视化报告 (资金曲线+回撤+持仓+因子信号)"""
    dates = data['signal'].index[:len(r_series)]
    r_values = r_series.values
    eq = (1 + r_values).cumprod()
    peak = np.maximum.accumulate(eq)
    dd = eq / peak - 1

    sig = data['signal']
    wt = data['weight']

    # 调仓点标记
    sig_aligned = sig.reindex(dates)
    wt_aligned = wt.reindex(dates)
    pos_changed = (sig_aligned != sig_aligned.shift(1)) | (wt_aligned != wt_aligned.shift(1))
    switch_dates = dates[pos_changed.values]

    # 因子数据
    g_close = G_CLOSE.reindex(dates)
    v_close = V_CLOSE.reindex(dates)
    ratio = (g_close / v_close).values
    ratio_ma20 = RATIO_MA20.reindex(dates).values
    t_value = RATIO_DEV_Z.reindex(dates).values
    slope = MA20_SLOPE.reindex(dates).values

    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(5, 1, figure=fig, height_ratios=[2.5, 1, 1, 1, 1], hspace=0.35)

    # ---- 子图1: 资金曲线 + 回撤阴影 ----
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(dates, eq, color='#2196F3', linewidth=1.5, label='资金曲线', zorder=3)
    ax1.fill_between(dates, 1, eq, where=(eq >= 1), alpha=0.1, color='#2196F3')
    ax1.fill_between(dates, eq, peak, where=(eq < peak), alpha=0.3, color='gray', label='回撤区间')
    ax1.set_title(f'{name_prefix} 资金曲线与回撤', fontsize=14, fontweight='bold')
    ax1.set_ylabel('净值')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    # ---- 子图2: 回撤曲线 ----
    ax2 = fig.add_subplot(gs[1])
    ax2.fill_between(dates, dd * 100, 0, color='#F44336', alpha=0.5)
    ax2.plot(dates, dd * 100, color='#D32F2F', linewidth=0.8)
    ax2.set_title('回撤百分比', fontsize=12)
    ax2.set_ylabel('回撤 (%)')
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    # ---- 子图3: 持仓方向+权重 ----
    ax3 = fig.add_subplot(gs[2])
    wt_vals = wt_aligned.values
    sig_vals = sig_aligned.values
    # growth=1, value=0.5, 空仓=0 (用权重*方向编码)
    pos_encoded = np.where(sig_vals == 'growth', 1.0, 0.5) * wt_vals
    colors_pos = np.where(sig_vals == 'growth', '#4CAF50', '#FF9800')
    for i in range(len(dates) - 1):
        ax3.bar(dates[i:i+2], pos_encoded[i:i+1], width=2, color=colors_pos[i], alpha=0.7)
    ax3.set_title('持仓方向与权重 (绿=growth 橙=value 0=空仓)', fontsize=12)
    ax3.set_ylabel('持仓编码')
    ax3.set_yticks([0, 0.5, 1.0])
    ax3.set_yticklabels(['空仓', 'value', 'growth'])
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_locator(mdates.YearLocator())
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    # ---- 子图4: 比价 + MA20 + 调仓标记 ----
    ax4 = fig.add_subplot(gs[3])
    ax4.plot(dates, ratio, color='#9C27B0', linewidth=1, label='G/V比价', alpha=0.8)
    ax4.plot(dates, ratio_ma20, color='#FF5722', linewidth=1, label='MA20', alpha=0.8)
    # 调仓标记
    for sd in switch_dates:
        ax4.axvline(x=sd, color='gray', alpha=0.15, linewidth=0.5)
    ax4.set_title('G/V比价 vs MA20 (灰线=调仓)', fontsize=12)
    ax4.set_ylabel('比价')
    ax4.legend(loc='upper left')
    ax4.grid(True, alpha=0.3)
    ax4.xaxis.set_major_locator(mdates.YearLocator())
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    # ---- 子图5: T指标(z-score) + 斜率 ----
    ax5 = fig.add_subplot(gs[4])
    ax5.plot(dates, t_value, color='#3F51B5', linewidth=1, label='T (RATIO_DEV_Z)', alpha=0.8)
    ax5.axhline(y=0, color='black', linewidth=0.5)
    ax5.axhline(y=1.3, color='red', linewidth=0.5, linestyle='--', alpha=0.5, label='rt=±1.3')
    ax5.axhline(y=-1.3, color='red', linewidth=0.5, linestyle='--', alpha=0.5)
    ax5.set_title('T指标 (z-score偏离度)', fontsize=12)
    ax5.set_ylabel('T值')
    ax5.legend(loc='upper left')
    ax5.grid(True, alpha=0.3)
    ax5.xaxis.set_major_locator(mdates.YearLocator())
    ax5.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    plt.savefig(str(OUTPUT_DIR / f'{name_prefix}_chart.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [{name_prefix}] 主图表生成完成")


def plot_monthly_heatmap(name_prefix, r_series):
    """生成月度收益率热力图"""
    dates = r_series.index
    df_m = pd.DataFrame({'daily_ret': r_series.values})
    df_m['year'] = dates.year
    df_m['month'] = dates.month
    monthly_matrix = df_m.groupby(['year', 'month'])['daily_ret'].apply(
        lambda x: (1 + x).prod() - 1
    ).unstack() * 100

    fig, ax = plt.subplots(figsize=(14, 8))
    im = ax.imshow(monthly_matrix.values, cmap='RdYlGn', aspect='auto',
                   vmin=-15, vmax=15)

    ax.set_xticks(range(12))
    ax.set_xticklabels(['1月', '2月', '3月', '4月', '5月', '6月',
                         '7月', '8月', '9月', '10月', '11月', '12月'])
    ax.set_yticks(range(len(monthly_matrix)))
    ax.set_yticklabels(monthly_matrix.index)

    for i in range(len(monthly_matrix)):
        for j in range(12):
            val = monthly_matrix.iloc[i, j]
            if pd.notna(val):
                color = 'white' if abs(val) > 10 else 'black'
                ax.text(j, i, f'{val:.1f}', ha='center', va='center',
                        color=color, fontsize=8)

    plt.colorbar(im, ax=ax, label='月度收益率 (%)')
    ax.set_title(f'{name_prefix} 月度收益率热力图', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(str(OUTPUT_DIR / f'{name_prefix}_monthly_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [{name_prefix}] 月度热力图生成完成")


def plot_comparison(v26_data, v27_data, x61_data):
    """生成三策略对比图"""
    fig, axes = plt.subplots(3, 1, figsize=(18, 14))

    # ---- 资金曲线对比 ----
    ax1 = axes[0]
    for data, name, color in [(v26_data, 'v26 U164 (无滑点最优)', '#2196F3'),
                                (v27_data, 'v27 U175 (滑点最优)', '#4CAF50'),
                                (x61_data, 'X61 (基准)', '#FF9800')]:
        dates = data['signal'].index[:len(data['daily_returns'])]
        r = data['daily_returns']['daily_ret'].values
        eq = (1 + r).cumprod()
        ax1.plot(dates, eq, label=name, color=color, linewidth=1.5)
    ax1.set_title('资金曲线对比: 统一策略 vs X61', fontsize=14, fontweight='bold')
    ax1.set_ylabel('净值')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    # ---- 回撤对比 ----
    ax2 = axes[1]
    for data, name, color in [(v26_data, 'v26 U164', '#2196F3'),
                                (v27_data, 'v27 U175', '#4CAF50'),
                                (x61_data, 'X61', '#FF9800')]:
        dates = data['signal'].index[:len(data['daily_returns'])]
        r = data['daily_returns']['daily_ret'].values
        eq = (1 + r).cumprod()
        peak = np.maximum.accumulate(eq)
        dd = (eq / peak - 1) * 100
        ax2.fill_between(dates, dd, 0, alpha=0.3, color=color, label=name)
    ax2.set_title('回撤对比', fontsize=14, fontweight='bold')
    ax2.set_ylabel('回撤 (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    # ---- 年度收益对比 ----
    ax3 = axes[2]
    width = 0.25
    all_years = set()
    bar_data = {}
    for data, name in [(v26_data, 'v26'), (v27_data, 'v27'), (x61_data, 'X61')]:
        dates = data['signal'].index[:len(data['daily_returns'])]
        r = pd.Series(data['daily_returns']['daily_ret'].values, index=dates)
        yearly = r.groupby(r.index.year).apply(lambda x: (1 + x).prod() - 1) * 100
        bar_data[name] = yearly
        all_years.update(yearly.index.tolist())
    years = sorted(all_years)

    x = np.arange(len(years))
    colors = {'v26': '#2196F3', 'v27': '#4CAF50', 'X61': '#FF9800'}
    for i, (name, yearly) in enumerate(bar_data.items()):
        vals = [yearly.get(y, 0) for y in years]
        ax3.bar(x + i * width, vals, width, label=name, color=colors[name], alpha=0.8)
    ax3.set_xticks(x + width)
    ax3.set_xticklabels(years)
    ax3.set_title('年度收益率对比', fontsize=14, fontweight='bold')
    ax3.set_ylabel('年化收益率 (%)')
    ax3.axhline(y=0, color='black', linewidth=0.5)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(str(OUTPUT_DIR / 'comparison_v26_v27_x61.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [对比图] 三策略对比图生成完成")


def plot_trade_analysis(name_prefix, trades_df):
    """生成交易分析图"""
    if len(trades_df) == 0:
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # 1. 交易类型分布
    ax1 = axes[0, 0]
    type_counts = trades_df['type'].value_counts()
    colors = ['#4CAF50', '#FF9800', '#2196F3', '#F44336', '#9C27B0']
    ax1.bar(type_counts.index, type_counts.values, color=colors[:len(type_counts)])
    ax1.set_title('交易类型分布', fontsize=12, fontweight='bold')
    ax1.set_ylabel('次数')
    for i, v in enumerate(type_counts.values):
        ax1.text(i, v + 0.5, str(v), ha='center', va='bottom')

    # 2. 持有天数分布
    ax2 = axes[0, 1]
    ax2.hist(trades_df['hold_days'], bins=30, color='#3F51B5', alpha=0.7, edgecolor='black')
    ax2.set_title('持有天数分布', fontsize=12, fontweight='bold')
    ax2.set_xlabel('持有天数')
    ax2.set_ylabel('频次')
    ax2.axvline(x=trades_df['hold_days'].mean(), color='red', linewidth=2,
                label=f'均值={trades_df["hold_days"].mean():.1f}天')
    ax2.legend()

    # 3. 各类型交易收益率
    ax3 = axes[1, 0]
    types = trades_df['type'].unique()
    type_returns = [trades_df[trades_df['type'] == t]['return'].values * 100 for t in types]
    bp = ax3.boxplot(type_returns, labels=types, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors[:len(types)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax3.set_title('各类型交易收益率分布', fontsize=12, fontweight='bold')
    ax3.set_ylabel('收益率 (%)')
    ax3.axhline(y=0, color='black', linewidth=0.5)
    ax3.grid(True, alpha=0.3, axis='y')

    # 4. 累计交易收益
    ax4 = axes[1, 1]
    cum_returns = trades_df['return'].cumsum() * 100
    ax4.plot(range(len(cum_returns)), cum_returns, color='#2196F3', linewidth=1.5)
    ax4.set_title('累计交易收益', fontsize=12, fontweight='bold')
    ax4.set_xlabel('交易序号')
    ax4.set_ylabel('累计收益率 (%)')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(OUTPUT_DIR / f'{name_prefix}_trade_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [{name_prefix}] 交易分析图生成完成")


# ============================================================
# 6. 导出TXT报告
# ============================================================
def export_txt_report(name_prefix, report, trades_df, yearly_ret):
    """导出TXT格式完整报告"""
    with open(str(OUTPUT_DIR / f'{name_prefix}_final_report.txt'), 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(f"  {report['策略名称']} 完整策略回测报告\n")
        f.write("=" * 80 + "\n\n")

        f.write("一、策略逻辑\n")
        f.write("-" * 40 + "\n")
        for layer, desc in report['策略规则'].items():
            f.write(f"  {layer}: {desc}\n")
        f.write("\n")

        f.write("二、核心参数\n")
        f.write("-" * 40 + "\n")
        for param, val in report['核心参数'].items():
            f.write(f"  {param} = {val}\n")
        f.write("\n")

        f.write("三、核心指标\n")
        f.write("-" * 40 + "\n")
        for k, v in report['核心指标'].items():
            f.write(f"  {k}: {v}\n")
        f.write("\n")

        f.write("四、滑点测试(5/万)\n")
        f.write("-" * 40 + "\n")
        for k, v in report['滑点测试(5/万)'].items():
            f.write(f"  {k}: {v}\n")
        f.write("\n")

        f.write("五、日收益率统计\n")
        f.write("-" * 40 + "\n")
        for k, v in report['日收益率统计'].items():
            f.write(f"  {k}: {v}\n")
        f.write("\n")

        f.write("六、月度统计\n")
        f.write("-" * 40 + "\n")
        for k, v in report['月度统计'].items():
            f.write(f"  {k}: {v}\n")
        f.write("\n")

        f.write("七、年度统计\n")
        f.write("-" * 40 + "\n")
        for k, v in report['年度统计'].items():
            f.write(f"  {k}: {v}\n")
        f.write("\n")

        f.write("八、年度收益率明细\n")
        f.write("-" * 40 + "\n")
        for yr, r in yearly_ret.items():
            f.write(f"  {yr}: {r*100:.2f}%\n")
        f.write("\n")

        f.write("九、交易分析\n")
        f.write("-" * 40 + "\n")
        if len(trades_df) > 0:
            f.write(f"  总交易次数: {len(trades_df)}\n")
            f.write(f"  正收益交易: {(trades_df['return']>0).sum()} ({(trades_df['return']>0).mean()*100:.1f}%)\n")
            f.write(f"  负收益交易: {(trades_df['return']<=0).sum()} ({(trades_df['return']<=0).mean()*100:.1f}%)\n")
            f.write(f"  平均持有天数: {trades_df['hold_days'].mean():.1f}\n\n")

            f.write("  交易类型分布:\n")
            for ttype, grp in trades_df.groupby('type'):
                f.write(f"  {ttype}: {len(grp)}次 "
                        f"(正收益{(grp['return']>0).sum()}/{len(grp)} "
                        f"均收益{grp['return'].mean()*100:.2f}% "
                        f"均持有{grp['hold_days'].mean():.1f}天)\n")
        f.write("\n" + "=" * 80 + "\n")
        f.write(f"  {report['策略名称']} 报告生成完毕\n")
        f.write("=" * 80 + "\n")
    print(f"  [{name_prefix}] TXT报告生成完成")


# ============================================================
# 主流程
# ============================================================
if __name__ == '__main__':
    print("=" * 80)
    print("  统一策略最终报告生成 (v26 + v27 + X61对比)")
    print("=" * 80)

    # ---- 策略规则定义 ----
    unified_rules = {
        'F1-基础信号': 'T=RATIO_DEV_Z>0 → BULL(growth), T<0 → BEAR(value)',
        'A1-方向确认': '连续dc=5天F1信号一致才确认方向',
        '方向冷却期': 'dcd=6天, 方向切换后6天内不再切换',
        'T+斜率双重确认': '|T|<rt=1.3 且 |斜率|<0.002 → both_weak → 空仓',
        'B2-价值动量过滤': 'value方向且V_MOM10<=0 AND V_MOM20<=0 → 改growth',
        'BIAS过滤(Br=0)': 'G_BIAS>0.19或V_BIAS>0.19 → 完全空仓(Br=0)',
        'E5-止损降仓': '20日跌幅>8.8% → 降仓至sw(17%或16%)',
        'E5冷却期': 'cd=8或9天, 降仓后冷却期内不恢复',
        '持仓时间限制(MHr=0)': '持仓>=92天 → 完全空仓(MHr=0)',
    }

    # ---- v26 U164 参数 (无滑点最优) ----
    v26_params = {
        'slope_thresh': 0.002, 'sw': 0.17, 'st': 0.088, 'cd': 8,
        'ms': 10, 'ml': 20, 'rt': 1.3, 'dc': 5, 'dcd': 6,
        'bias_ma': 20, 'bias_high': 0.19, 'bias_reduce': 0.0,
        'max_hold_days': 92, 'max_hold_reduce': 0.0,
    }

    # ---- v27 U175 参数 (滑点最优) ----
    v27_params = {
        'slope_thresh': 0.002, 'sw': 0.16, 'st': 0.088, 'cd': 9,
        'ms': 10, 'ml': 20, 'rt': 1.3, 'dc': 5, 'dcd': 6,
        'bias_ma': 20, 'bias_high': 0.19, 'bias_reduce': 0.0,
        'max_hold_days': 92, 'max_hold_reduce': 0.0,
    }

    # ---- X61 参数 (基准) ----
    x61_rules = {
        'F1-基础信号': '比价 vs MA20 → growth/value',
        'A1-方向确认': '连续4天方向一致',
        '方向确认': '连续5天F1信号一致才切换方向',
        'F0+斜率双重确认': '|偏离度z|<1.5 且 |斜率|<0.2% → both_weak',
        '空仓进入确认6天': '连续6天both_weak=True才进入空仓',
        '空仓退出确认2天': '连续2天both_weak=False才恢复满仓',
        'B2-价值动量过滤': 'value方向且V_MOM20≤0 → 改growth',
        'E5-止损降仓': '20日跌幅>10% → 降仓30%',
        'E5冷却5天': '降仓后5天内不恢复',
    }
    x61_params = {
        'Z_THRESH': 1.5, 'SLOPE_THRESH': 0.002, 'N_CONFIRM': 4,
        'DIR_CONFIRM_DAYS': 5, 'ENTRY_CONFIRM_DAYS': 6, 'EXIT_CONFIRM_DAYS': 2,
        'STOP_THRESHOLD': 0.10, 'STOP_WEIGHT': 0.30, 'E5_COOLDOWN_DAYS': 5,
    }

    # ============================================================
    # 1. v26 U164 (无滑点最优)
    # ============================================================
    print("\n[1/3] 构建 v26 U164 (无滑点最优)...")
    sig_v26, wt_v26 = build_core(**v26_params)
    data_v26 = run_detailed_backtest('v26_U164', sig_v26, wt_v26)
    trades_v26, _ = analyze_trades(data_v26['signal'], data_v26['weight'],
                                    data_v26['g_close'], data_v26['v_close'])
    report_v26, r_v26, yearly_v26, monthly_v26 = generate_report(
        data_v26, 'v26 U164 (无滑点最优, Calmar 2.378)',
        unified_rules, {k: v for k, v in v26_params.items()})
    m_v26 = data_v26['metrics']
    print(f"  年化={m_v26['ann']*100:.2f}% 回撤={m_v26['dd']*100:.2f}% "
          f"Sharpe={m_v26['sharpe']:.3f} Calmar={m_v26['calmar']:.3f} "
          f"交易={m_v26['n_trades']}")

    # ============================================================
    # 2. v27 U175 (滑点最优)
    # ============================================================
    print("\n[2/3] 构建 v27 U175 (滑点最优)...")
    sig_v27, wt_v27 = build_core(**v27_params)
    data_v27 = run_detailed_backtest('v27_U175', sig_v27, wt_v27)
    trades_v27, _ = analyze_trades(data_v27['signal'], data_v27['weight'],
                                    data_v27['g_close'], data_v27['v_close'])
    report_v27, r_v27, yearly_v27, monthly_v27 = generate_report(
        data_v27, 'v27 U175 (滑点最优, 滑点Calmar 2.059)',
        unified_rules, {k: v for k, v in v27_params.items()})
    m_v27 = data_v27['metrics']
    m_v27_sl = data_v27['metrics_sl']
    print(f"  年化={m_v27['ann']*100:.2f}% 回撤={m_v27['dd']*100:.2f}% "
          f"Sharpe={m_v27['sharpe']:.3f} Calmar={m_v27['calmar']:.3f} "
          f"交易={m_v27['n_trades']}")
    print(f"  滑点: 年化={m_v27_sl['ann']*100:.2f}% Calmar={m_v27_sl['calmar']:.3f}")

    # ============================================================
    # 3. X61 (基准)
    # ============================================================
    print("\n[3/3] 构建 X61 (基准)...")
    from run_x49_x54 import build_x51
    sig_x61, wt_x61 = build_x51(
        entry_confirm_days=6, exit_confirm_days=2, e5_cooldown_days=5,
        dir_confirm_days=5, z_thresh=1.5, slope_thresh=0.002,
        n_confirm=4, stop_threshold=0.10, stop_weight=0.30,
    )
    data_x61 = run_detailed_backtest('X61', sig_x61, wt_x61)
    trades_x61, _ = analyze_trades(data_x61['signal'], data_x61['weight'],
                                    data_x61['g_close'], data_x61['v_close'])
    report_x61, r_x61, yearly_x61, monthly_x61 = generate_report(
        data_x61, 'X61 (基准, Calmar 1.361)',
        x61_rules, x61_params)
    m_x61 = data_x61['metrics']
    print(f"  年化={m_x61['ann']*100:.2f}% 回撤={m_x61['dd']*100:.2f}% "
          f"Sharpe={m_x61['sharpe']:.3f} Calmar={m_x61['calmar']:.3f} "
          f"交易={m_x61['n_trades']}")

    # ============================================================
    # 导出所有数据
    # ============================================================
    print("\n" + "=" * 80)
    print("  导出数据...")
    print("=" * 80)

    export_data('v26_U164', data_v26, report_v26, trades_v26, r_v26)
    export_data('v27_U175', data_v27, report_v27, trades_v27, r_v27)
    export_data('X61', data_x61, report_x61, trades_x61, r_x61)

    export_txt_report('v26_U164', report_v26, trades_v26, yearly_v26)
    export_txt_report('v27_U175', report_v27, trades_v27, yearly_v27)
    export_txt_report('X61', report_x61, trades_x61, yearly_x61)

    # ============================================================
    # 生成图表
    # ============================================================
    print("\n" + "=" * 80)
    print("  生成可视化图表...")
    print("=" * 80)

    plot_strategy_report('v26_U164', data_v26, r_v26, trades_v26)
    plot_strategy_report('v27_U175', data_v27, r_v27, trades_v27)
    plot_strategy_report('X61', data_x61, r_x61, trades_x61)

    plot_monthly_heatmap('v26_U164', r_v26)
    plot_monthly_heatmap('v27_U175', r_v27)
    plot_monthly_heatmap('X61', r_x61)

    plot_trade_analysis('v26_U164', trades_v26)
    plot_trade_analysis('v27_U175', trades_v27)
    plot_trade_analysis('X61', trades_x61)

    plot_comparison(data_v26, data_v27, data_x61)

    # ============================================================
    # 三策略对比汇总
    # ============================================================
    print("\n" + "=" * 80)
    print("  三策略对比汇总")
    print("=" * 80)
    print(f"\n  {'策略':<28} {'年化':>8} {'回撤':>8} {'Sharpe':>8} {'Calmar':>8} "
          f"{'滑点年化':>10} {'滑点Calmar':>12} {'交易':>6}")
    print("  " + "-" * 100)
    for data, name in [(data_v26, 'v26 U164 (无滑点最优)'),
                        (data_v27, 'v27 U175 (滑点最优)'),
                        (data_x61, 'X61 (基准)')]:
        m = data['metrics']
        m_sl = data['metrics_sl']
        print(f"  {name:<28} {m['ann']*100:>7.2f}% {m['dd']*100:>7.2f}% "
              f"{m['sharpe']:>8.3f} {m['calmar']:>8.3f} "
              f"{m_sl['ann']*100:>9.2f}% {m_sl['calmar']:>12.3f} "
              f"{m['n_trades']:>6}")

    print(f"\n  统一策略 vs X61 提升:")
    print(f"    v26 vs X61: Calmar +{(m_v26['calmar']-m_x61['calmar']):.3f} "
          f"(+{(m_v26['calmar']/m_x61['calmar']-1)*100:.1f}%)")
    print(f"    v27 vs X61: 滑点Calmar +{(m_v27_sl['calmar']-data_x61['metrics_sl']['calmar']):.3f} "
          f"(+{(m_v27_sl['calmar']/data_x61['metrics_sl']['calmar']-1)*100:.1f}%)")

    # 对比报告JSON
    comparison = {
        '生成时间': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'v26_U164_无滑点最优': report_v26,
        'v27_U175_滑点最优': report_v27,
        'X61_基准': report_x61,
        '对比汇总': {
            'v26_vs_X61_Calmar提升': f"+{(m_v26['calmar']-m_x61['calmar']):.3f} (+{(m_v26['calmar']/m_x61['calmar']-1)*100:.1f}%)",
            'v27_vs_X61_滑点Calmar提升': f"+{(m_v27_sl['calmar']-data_x61['metrics_sl']['calmar']):.3f} (+{(m_v27_sl['calmar']/data_x61['metrics_sl']['calmar']-1)*100:.1f}%)",
            'v26_vs_X61_年化提升': f"+{(m_v26['ann']-m_x61['ann'])*100:.2f}pp",
            'v26_vs_X61_回撤改善': f"{(m_v26['dd']-m_x61['dd'])*100:.2f}pp",
            'v26_vs_X61_交易减少': f"-{m_x61['n_trades']-m_v26['n_trades']}次",
        }
    }
    with open(str(OUTPUT_DIR / 'comparison_report.json'), 'w', encoding='utf-8') as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)

    print(f"\n  所有报告已生成到: {OUTPUT_DIR}")
    print("=" * 80)
