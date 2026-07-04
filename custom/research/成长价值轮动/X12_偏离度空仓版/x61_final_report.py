"""x61_final_report.py — X61完整策略回测报告生成
================================================================
生成X61(en6/ex2/e55/dc5)的完整回测数据用于图表生成
================================================================
"""
import sys
sys.path.insert(0, r'c:\XuanJLH\Qbot\custom\backtests')
sys.path.insert(0, r'c:\XuanJLH\Qbot\custom\research\成长价值轮动\X12_偏离度空仓版')

from pathlib import Path
import numpy as np
import pandas as pd
import json

from optimize_runner import (
    G_CLOSE, V_CLOSE, RATIO, RATIO_MA20, RATIO_DEV, MA20_SLOPE,
    V_MOM20, G_DD20, V_DD20, BASE_DIR,
    run_backtest, calc_metrics, count_switches,
)
from run_x33_reduce_trades import (
    Z_THRESH, SLOPE_THRESH, N_CONFIRM, STOP_THRESHOLD, STOP_WEIGHT,
    RATIO_DEV_STD20, RATIO_DEV_Z,
)
from run_x49_x54 import build_x51

OUTPUT_DIR = Path(r'c:\XuanJLH\Qbot\custom\research\成长价值轮动\X12_偏离度空仓版\回测结果')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# X61完整策略
# ============================================================
def build_x61():
    """X61(en6/ex2/e55/dc5)完整策略"""
    return build_x51(
        entry_confirm_days=6, exit_confirm_days=2, e5_cooldown_days=5,
        dir_confirm_days=5, z_thresh=1.5, slope_thresh=0.002,
        n_confirm=4, stop_threshold=0.10, stop_weight=0.30,
    )


# ============================================================
# 1. 基础回测
# ============================================================
def run_detailed_backtest():
    """运行完整回测并收集所有数据"""
    print("正在构建X61策略...")
    signal, weight = build_x61()

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

    print("正在运行回测引擎...")
    result = run_backtest(signal, weight)
    result_sl = run_backtest(signal, weight, impact_slippage=0.0005)

    m = calc_metrics(result)
    m_sl = calc_metrics(result_sl)
    sw = count_switches(signal, weight)

    df_r = result.to_dataframe()
    df_r_sl = result_sl.to_dataframe()

    return {
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
def analyze_trades(sig, wt):
    """分析每次调仓的类型、盈亏、持有天数"""
    df = pd.DataFrame({'signal': sig.astype(str), 'weight': wt.astype(float),
                       'g_close': G_CLOSE[sig.index], 'v_close': V_CLOSE[sig.index]})

    # 计算每日收益
    df['prev_sig'] = df['signal'].shift(1)
    df['prev_wt'] = df['weight'].shift(1)
    df['prev_g'] = df['g_close'].shift(1)
    df['prev_v'] = df['v_close'].shift(1)

    # 持仓组合收益
    df['daily_ret'] = 0.0
    g_ret = df['g_close'] / df['prev_g'] - 1
    v_ret = df['v_close'] / df['prev_v'] - 1
    is_growth = df['signal'] == 'growth'
    df.loc[is_growth, 'daily_ret'] = g_ret[is_growth] * df.loc[is_growth, 'weight']
    df.loc[~is_growth, 'daily_ret'] = v_ret[~is_growth] * df.loc[~is_growth, 'weight']
    df['daily_ret'] = df['daily_ret'].fillna(0)
    df['equity'] = (1 + df['daily_ret']).cumprod()

    # 识别调仓点
    df['wt_changed'] = (df['weight'] != df['prev_wt']) | (df['signal'] != df['prev_sig'])
    df['trade_id'] = df['wt_changed'].cumsum()

    # 重置方向信号(处理混乱)
    # 直接在逐笔交易层面计算
    trades = []
    start_idx = None
    prev_sig_val = None
    prev_wt_val = None
    start_eq = 1.0

    for i, row in df.iterrows():
        if start_idx is None:
            # 第一行
            if row['wt_changed']:
                start_idx = i
                prev_sig_val = row['signal']
                prev_wt_val = row['weight']
                start_eq = row['equity']
            continue

        if row['wt_changed']:
            # 交易结束
            end_idx = i
            end_eq = df.loc[end_idx, 'equity'] if end_idx in df.index else df['equity'].iloc[-1]

            # 确定交易类型
            curr_sig = df.loc[end_idx, 'signal']
            curr_wt = df.loc[end_idx, 'weight']
            prev_sig = prev_sig_val
            prev_wt = prev_wt_val

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

            # 计算持有天数
            hold_days = (end_idx - start_idx).days if hasattr(end_idx - start_idx, 'days') else 0

            # 计算期间收益(使用实际序列)
            eq_chunk = df.loc[start_idx:end_idx, 'equity']
            ret = (eq_chunk.iloc[-1] / eq_chunk.iloc[0] - 1) if len(eq_chunk) > 1 else 0

            # 计算方向(跳过空仓)
            dir_at_start = df.loc[start_idx, 'signal']
            dir_at_end = df.loc[end_idx, 'signal']

            trades.append({
                'start_date': start_idx,
                'end_date': end_idx,
                'type': ttype,
                'return': ret,
                'hold_days': hold_days,
                'start_dir': dir_at_start,
                'end_dir': dir_at_end,
                'start_wt': prev_wt,
                'end_wt': curr_wt,
            })

            # 更新新交易起点
            start_idx = end_idx
            prev_sig_val = curr_sig
            prev_wt_val = curr_wt
            start_eq = row['equity']

    trades_df = pd.DataFrame(trades)
    return trades_df, df


# ============================================================
# 3. 输出完整报告
# ============================================================
def generate_report(data):
    """生成完整回测报告"""
    m = data['metrics']
    m_sl = data['metrics_sl']
    sw = data['switches']
    df_r = data['daily_returns']

    # 从signal索引获取日期
    dates = data['signal'].index
    # 确保长度匹配
    r = df_r['daily_ret'].values
    r_series = pd.Series(r, index=dates[:len(r)])

    # 正收益比例
    win_rate = (r_series > 0).mean()
    avg_win = r_series[r_series > 0].mean() if (r_series > 0).any() else 0
    avg_loss = r_series[r_series < 0].mean() if (r_series < 0).any() else 0
    profit_factor = (r_series[r_series > 0].sum() / abs(r_series[r_series < 0].sum())) if (r_series < 0).any() else np.inf

    # 月度统计
    df_monthly = pd.DataFrame({'daily_ret': r_series})
    df_monthly['year'] = r_series.index.year
    df_monthly['month'] = r_series.index.month
    monthly_ret = df_monthly.groupby(['year', 'month'])['daily_ret'].apply(lambda x: (1 + x).prod() - 1)
    monthly_win_rate = (monthly_ret > 0).mean()
    monthly_mean = monthly_ret.mean()
    monthly_std = monthly_ret.std()
    monthly_sharpe = monthly_mean / monthly_std * np.sqrt(12) if monthly_std > 0 else 0

    # 年度统计
    yearly_ret = df_monthly.groupby('year')['daily_ret'].apply(lambda x: (1 + x).prod() - 1)
    yearly_win_rate = (yearly_ret > 0).mean()

    report = {
        '策略名称': 'X61(en6/ex2/e55/dc5)',
        '策略规则': {
            'F1-基础信号': '比价 vs MA20 → growth/value',
            'A1-方向确认': '连续4天方向一致',
            '方向确认': '连续5天F1信号一致才切换方向',
            'F0+斜率双重确认': '|偏离度z|<1.5 且 |斜率|<0.2% → both_weak',
            '空仓进入确认6天': '连续6天both_weak=True才进入空仓',
            '空仓退出确认2天': '连续2天both_weak=False才恢复满仓',
            'B2-价值动量过滤': 'value方向且V_MOM20≤0 → 改growth',
            'E5-止损降仓': '20日跌幅>10% → 降仓30%',
            'E5冷却5天': '降仓后5天内不恢复',
        },
        '核心参数': {
            'Z_THRESH': 1.5,
            'SLOPE_THRESH': 0.002,
            'N_CONFIRM': 4,
            'DIR_CONFIRM_DAYS': 5,
            'ENTRY_CONFIRM_DAYS': 6,
            'EXIT_CONFIRM_DAYS': 2,
            'STOP_THRESHOLD': 0.10,
            'STOP_WEIGHT': 0.30,
            'E5_COOLDOWN_DAYS': 5,
        },
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
    }

    return report, r_series, yearly_ret, monthly_ret


# ============================================================
# 4. 导出数据到CSV
# ============================================================
def export_data(data, report, trades_df, r_series):
    """导出所有回测数据到CSV文件"""
    m = data['metrics']

    # 从signal索引获取日期(确保长度匹配)
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
    df_export.to_csv(str(OUTPUT_DIR / 'x61_daily_returns.csv'), index=False, encoding='utf-8')
    print(f"  → 日收益率已导出: {OUTPUT_DIR / 'x61_daily_returns.csv'}")

    # 2. 信号和权重序列
    sig_df = pd.DataFrame({
        'date': data['signal'].index.strftime('%Y-%m-%d'),
        'signal': data['signal'].values,
        'weight': data['weight'].values,
    })
    sig_df.to_csv(str(OUTPUT_DIR / 'x61_signal_weight.csv'), index=False, encoding='utf-8')
    print(f"  → 信号/权重已导出: {OUTPUT_DIR / 'x61_signal_weight.csv'}")

    # 3. 交易记录
    if len(trades_df) > 0:
        trades_df.to_csv(str(OUTPUT_DIR / 'x61_trades.csv'), index=False, encoding='utf-8')
        print(f"  → 交易记录已导出: {OUTPUT_DIR / 'x61_trades.csv'}")

    # 4. 月度收益率矩阵
    r_series = pd.Series(r_values, index=dates)
    df_monthly = pd.DataFrame({'daily_ret': r_series})
    df_monthly['year'] = r_series.index.year
    df_monthly['month'] = r_series.index.month
    monthly_matrix = df_monthly.groupby(['year', 'month'])['daily_ret'].apply(
        lambda x: (1 + x).prod() - 1
    ).unstack()
    monthly_matrix.to_csv(str(OUTPUT_DIR / 'x61_monthly_returns.csv'), encoding='utf-8')
    print(f"  → 月度收益率已导出: {OUTPUT_DIR / 'x61_monthly_returns.csv'}")

    # 5. 年度收益率
    yearly_ret = df_monthly.groupby('year')['daily_ret'].apply(lambda x: (1 + x).prod() - 1)
    yearly_ret.to_csv(str(OUTPUT_DIR / 'x61_yearly_returns.csv'), encoding='utf-8')
    print(f"  → 年度收益率已导出: {OUTPUT_DIR / 'x61_yearly_returns.csv'}")

    # 6. 指数行情数据
    idx = data['signal'].index.intersection(G_CLOSE.index).intersection(V_CLOSE.index)
    idx_df = pd.DataFrame({
        'date': idx.strftime('%Y-%m-%d'),
        'growth_close': G_CLOSE.loc[idx].values,
        'value_close': V_CLOSE.loc[idx].values,
        'ratio': (G_CLOSE / V_CLOSE).loc[idx].values,
        'ratio_ma20': RATIO_MA20.loc[idx].values,
        'ratio_dev_z': RATIO_DEV_Z.loc[idx].values,
    })
    idx_df.to_csv(str(OUTPUT_DIR / 'x61_index_data.csv'), index=False, encoding='utf-8')
    print(f"  → 指数行情已导出: {OUTPUT_DIR / 'x61_index_data.csv'}")

    # 7. 完整报告JSON
    with open(str(OUTPUT_DIR / 'x61_report.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  → 报告JSON已导出: {OUTPUT_DIR / 'x61_report.json'}")

    # 8. 报告TXT
    with open(str(OUTPUT_DIR / 'x61_final_report.txt'), 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("  X61(en6/ex2/e55/dc5) 完整策略回测报告\n")
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

        # 交易分析
        f.write("八、交易分析\n")
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
        f.write("\n")

        f.write("九、年度收益率明细\n")
        f.write("-" * 40 + "\n")
        r_series = pd.Series(data['daily_returns']['daily_ret'].values,
                             index=data['signal'].index[:len(data['daily_returns'])])
        for yr, r in r_series.groupby(r_series.index.year).apply(lambda x: (1+x).prod()-1).items():
            f.write(f"  {yr}: {r*100:.2f}%\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("  X61报告生成完毕\n")
        f.write("=" * 80 + "\n")
    print(f"  → 完整报告已导出: {OUTPUT_DIR / 'x61_final_report.txt'}")

    # 9. 交易类型统计CSV
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
        type_stats.to_csv(str(OUTPUT_DIR / 'x61_trade_type_stats.csv'), index=False, encoding='utf-8')
        print(f"  → 交易类型统计已导出: {OUTPUT_DIR / 'x61_trade_type_stats.csv'}")


# ============================================================
# 主流程
# ============================================================
if __name__ == '__main__':
    print("=" * 80)
    print("  X61(en6/ex2/e55/dc5) 完整策略回测报告")
    print("=" * 80)

    # 运行回测
    data = run_detailed_backtest()

    # 交易分析
    print("正在分析交易记录...")
    trades_df, df_daily = analyze_trades(data['signal'], data['weight'])

    # 生成报告
    print("正在生成报告...")
    report, df_r, yearly_ret, monthly_ret = generate_report(data)

    # 导出数据
    r_series_for_export = data['daily_returns']['daily_ret']
    print("\n正在导出数据...")
    export_data(data, report, trades_df, r_series_for_export)

    print("\n" + "=" * 80)
    print("  完成! 所有回测数据已导出到:")
    print(f"  {OUTPUT_DIR}")
    print("=" * 80)
