"""backtest_x6_engine.py — X6 策略：60日ROC四状态，都涨超配成长
================================================================
X6 = X2-A（分段降仓基线）+ 都涨状态覆盖层

  信号层（保持X2-A不变）:
    F1 = tanh(ratio_dev × 30) × 0.5        # 比价MA20方向
    F2 = accel_diff.clip(±0.02) × 5.0      # 动量加速度
    style_score = F1 + F2
    candidate_g = style_score > 0           # 二元方向（相对动量）

  状态覆盖层（X6新增，仅都涨时生效）:
    roc_g60 = g_close.pct_change(60).shift(1)
    roc_v60 = v_close.pct_change(60).shift(1)
    both_up = (roc_g60 > 0) & (roc_v60 > 0)
    final_g = candidate_g | both_up         # 都涨时强制选成长

  择时层（保持X2-A不变）:
    MA50双跌破 → weight=0.50
    MA75双跌破 → weight=0.25

  消融分析5方案:
    X2-baseline      : X2-A原版（无覆盖层）
    X6-full          : X2-A + 都涨覆盖(60日ROC)
    X6-window-20     : X2-A + 都涨覆盖(20日ROC)
    X6-window-120    : X2-A + 都涨覆盖(120日ROC)
    X6-both-down-cash: X2-A + 都涨选成长 + 都跌选cash（weight=0）

  设计原则:
    1) 不改变F1/F2和X2-A任何参数（因子要少）
    2) 状态覆盖层只在都涨时生效，其他状态保持X2-A原版
    3) 60日ROC是A 股动量有效窗口（1-3月上界）
    4) 符号阈值法（threshold=0，最简形式，禁止拟合）
    5) shift(1)避免前视，T日决策T+1执行

数据:
  成长100: c:\\temp_v72_data\\index_480080.csv
  价值100: c:\\temp_v72_data\\index_480081.csv
  区间:    2012-12-31 ~ 2026-07-01

回测引擎: backtest_engine.py (run_backtest_engine_weighted)
  指数 open=close，佣金万1，无冲击/跳空滑点，周频W-FRI调仓
  Sharpe 用 rf=2.5%/年（与X1/X2/X5口径一致）
================================================================
"""
import sys
sys.path.insert(0, r'c:\XuanJLH\Qbot\custom\backtests')

from pathlib import Path
import numpy as np
import pandas as pd
from backtest_engine import (
    BacktestInput, BacktestConfig,
    run_backtest_engine_weighted,
)

# ============================================================
# 1. 参数
# ============================================================
# F1/F2/MA参数（与X2-A完全一致，不优化）
F1 = 0.5
F2 = 5.0
MA_PCT = 0.97

# 状态覆盖层参数（X6新增）
ROC_WINDOW_DEFAULT = 60    # 60日ROC窗口（A 股动量有效窗口上界）
ROC_THRESHOLD = 0.0        # 符号阈值法（>0即上涨）

DATA_DIR = Path(r'c:\temp_v72_data')
REPORT_PATH = DATA_DIR / 'x6_ablation_report.txt'

# ============================================================
# 2. 加载数据
# ============================================================
print("=" * 70)
print("  X6 策略 — 60日ROC四状态，都涨超配成长")
print("  X2-A基线 + 都涨时强制选成长（覆盖F1/F2）")
print("=" * 70)

g_raw = pd.read_csv(str(DATA_DIR / 'index_480080.csv'))
v_raw = pd.read_csv(str(DATA_DIR / 'index_480081.csv'))

for d in (g_raw, v_raw):
    d['date'] = pd.to_datetime(d['date'])
    d['close'] = pd.to_numeric(d['close'], errors='coerce')

g_close = g_raw.set_index('date')['close'].astype(float).sort_index().dropna()
v_close = v_raw.set_index('date')['close'].astype(float).sort_index().dropna()
common = g_close.index.intersection(v_close.index)
g_close = g_close[common].sort_index()
v_close = v_close[common].sort_index()

print(f"数据区间: {g_close.index[0]:%Y-%m-%d} ~ {g_close.index[-1]:%Y-%m-%d}")
print(f"交易日数: {len(g_close)}")

# ============================================================
# 3. 信号层（与X2-A完全一致）
# ============================================================
print("\n[信号层] F1=比价MA20方向(0.5) + F2=动量加速度(5.0)")

# F1: 比价MA20方向
ratio = (g_close / v_close).shift(1)
ratio_ma20 = ratio.rolling(20).mean()
ratio_dev = ratio / ratio_ma20 - 1
f1_signal = np.tanh(ratio_dev * 30) * F1
style_score = f1_signal.copy()

# F2: 动量加速度
g_roc21 = g_close.pct_change(21).shift(1)
v_roc21 = v_close.pct_change(21).shift(1)
g_accel = g_roc21 - g_roc21.shift(10)
v_accel = v_roc21 - v_roc21.shift(10)
accel_diff = (g_accel - v_accel).clip(-0.02, 0.02)
f2_signal = accel_diff * F2
style_score = style_score + f2_signal

# 二元方向（相对动量，X2-A原版）
candidate_g = style_score > 0

# ============================================================
# 4. 状态覆盖层（X6新增）
# ============================================================
print("[状态覆盖层] 60日ROC四状态识别")

# 计算多个ROC窗口，供消融分析使用
roc_g_60 = g_close.pct_change(60).shift(1)
roc_v_60 = v_close.pct_change(60).shift(1)
roc_g_20 = g_close.pct_change(20).shift(1)
roc_v_20 = v_close.pct_change(20).shift(1)
roc_g_120 = g_close.pct_change(120).shift(1)
roc_v_120 = v_close.pct_change(120).shift(1)

# 四状态识别（60日）
both_up_60 = (roc_g_60 > ROC_THRESHOLD) & (roc_v_60 > ROC_THRESHOLD)
both_down_60 = (roc_g_60 < ROC_THRESHOLD) & (roc_v_60 < ROC_THRESHOLD)
g_up_v_down_60 = (roc_g_60 > ROC_THRESHOLD) & (roc_v_60 < ROC_THRESHOLD)
g_down_v_up_60 = (roc_g_60 < ROC_THRESHOLD) & (roc_v_60 > ROC_THRESHOLD)

# 四状态识别（20日/120日，供消融）
both_up_20 = (roc_g_20 > ROC_THRESHOLD) & (roc_v_20 > ROC_THRESHOLD)
both_down_20 = (roc_g_20 < ROC_THRESHOLD) & (roc_v_20 < ROC_THRESHOLD)
both_up_120 = (roc_g_120 > ROC_THRESHOLD) & (roc_v_120 > ROC_THRESHOLD)
both_down_120 = (roc_g_120 < ROC_THRESHOLD) & (roc_v_120 < ROC_THRESHOLD)

# 都涨时强制选成长（X6核心）
final_g_60 = candidate_g | both_up_60
final_g_20 = candidate_g | both_up_20
final_g_120 = candidate_g | both_up_120

# ============================================================
# 5. 择时层：MA50/MA75（X2-A原版）
# ============================================================
print("[择时层] MA50双跌破 + MA75双跌破（X2-A原版）")

g_ma50 = g_close.shift(1).rolling(50).mean()
v_ma50 = v_close.shift(1).rolling(50).mean()
both_below_ma50 = (g_close.shift(1) < g_ma50 * MA_PCT) & (v_close.shift(1) < v_ma50 * MA_PCT)

g_ma75 = g_close.shift(1).rolling(75).mean()
v_ma75 = v_close.shift(1).rolling(75).mean()
both_below_ma75 = (g_close.shift(1) < g_ma75 * MA_PCT) & (v_close.shift(1) < v_ma75 * MA_PCT)

# ============================================================
# 6. 周频采样（W-FRI，与X2-A一致）
# ============================================================
df = pd.DataFrame(index=g_close.index)
df['g_close'] = g_close
df['v_close'] = v_close
df['candidate_g'] = candidate_g
df['both_below_ma50'] = both_below_ma50
df['both_below_ma75'] = both_below_ma75
# 状态覆盖层
df['final_g_60'] = final_g_60
df['final_g_20'] = final_g_20
df['final_g_120'] = final_g_120
df['both_up_60'] = both_up_60
df['both_down_60'] = both_down_60
df['both_up_20'] = both_up_20
df['both_down_20'] = both_down_20
df['both_up_120'] = both_up_120
df['both_down_120'] = both_down_120
df['roc_g_60'] = roc_g_60
df['roc_v_60'] = roc_v_60
df['style_score'] = style_score

weekly = df.resample('W-FRI').last().dropna(subset=['candidate_g']).iloc[1:]
# 修复：resample('W-FRI').last() 会把 boolean 列转成 float64（pandas bug）
bool_cols = ['candidate_g', 'both_below_ma50', 'both_below_ma75',
             'final_g_60', 'final_g_20', 'final_g_120',
             'both_up_60', 'both_down_60', 'both_up_20', 'both_down_20',
             'both_up_120', 'both_down_120']
for col in bool_cols:
    weekly[col] = weekly[col].astype(bool)

print(f"周数: {len(weekly)}")
print(f"  MA50双跌破周数: {int(weekly['both_below_ma50'].sum())}")
print(f"  MA75浅跌破周数: {int(weekly['both_below_ma75'].sum())}")
print(f"\n  [四状态统计 - 60日ROC]")
print(f"  都涨周数:   {int(weekly['both_up_60'].sum())} ({weekly['both_up_60'].sum()/len(weekly)*100:.1f}%)")
print(f"  都跌周数:   {int(weekly['both_down_60'].sum())} ({weekly['both_down_60'].sum()/len(weekly)*100:.1f}%)")
print(f"  成涨价跌:   {int(weekly['both_up_60'].sum() + weekly['both_down_60'].sum())}")
# 计算剩余两种状态
g_up_v_down = (weekly['roc_g_60'] > 0) & (weekly['roc_v_60'] < 0)
g_down_v_up = (weekly['roc_g_60'] < 0) & (weekly['roc_v_60'] > 0)
print(f"  成长涨价值跌: {int(g_up_v_down.sum())} ({g_up_v_down.sum()/len(weekly)*100:.1f}%)")
print(f"  成长跌价值涨: {int(g_down_v_up.sum())} ({g_down_v_up.sum()/len(weekly)*100:.1f}%)")
# 覆盖层影响
override_count = (weekly['candidate_g'] != weekly['final_g_60']).sum()
print(f"\n  [覆盖层影响] 60日ROC都涨覆盖F1/F2方向: {int(override_count)} 周 ({override_count/len(weekly)*100:.1f}%)")


# ============================================================
# 7. 方案构建函数
# ============================================================
def build_x2_baseline(weekly_df):
    """X2-A基线 — 分段降仓（无覆盖层）
    MA50→50%, MA75→25%, 方向由style_score决定
    """
    n = len(weekly_df)
    signal = pd.Series(['value'] * n, index=weekly_df.index, dtype=object)
    weight = pd.Series(1.0, index=weekly_df.index)

    for i in range(n):
        row = weekly_df.iloc[i]
        signal.iloc[i] = 'growth' if row['candidate_g'] else 'value'
        if row['both_below_ma75']:
            weight.iloc[i] = 0.25
        elif row['both_below_ma50']:
            weight.iloc[i] = 0.5
        else:
            weight.iloc[i] = 1.0
    return signal, weight


def build_x6(weekly_df, final_g_col='final_g_60', both_down_col=None, cash_on_both_down=False):
    """X6 — X2-A + 都涨覆盖层
    final_g_col: 使用哪个final_g列（60/20/120日ROC）
    both_down_col: 都跌时使用的列（None表示不特殊处理）
    cash_on_both_down: True时都跌清仓（weight=0）
    """
    n = len(weekly_df)
    signal = pd.Series(['value'] * n, index=weekly_df.index, dtype=object)
    weight = pd.Series(1.0, index=weekly_df.index)

    for i in range(n):
        row = weekly_df.iloc[i]
        # 方向由final_g决定（都涨时覆盖F1/F2）
        signal.iloc[i] = 'growth' if row[final_g_col] else 'value'

        # 仓位权重（X2-A分段降仓）
        if row['both_below_ma75']:
            w = 0.25
        elif row['both_below_ma50']:
            w = 0.5
        else:
            w = 1.0

        # 激进版：都跌清仓
        if cash_on_both_down and both_down_col is not None:
            if row[both_down_col]:
                w = 0.0

        weight.iloc[i] = w
    return signal, weight


def expand_to_daily(weekly_series, daily_index):
    """周频信号前向填充到日频（与X2引擎一致）"""
    s = weekly_series.copy()
    idx_pos = daily_index.searchsorted(s.index, side='right') - 1
    valid = idx_pos >= 0
    s = s[valid]
    idx_pos = idx_pos[valid]
    s.index = daily_index[idx_pos]
    s = s[~s.index.duplicated(keep='last')]
    daily = pd.Series(np.nan, index=daily_index, dtype=s.dtype)
    daily.loc[s.index] = s
    return daily.ffill()


def run_backtest(daily_signal, daily_weight, g_close_s, v_close_s):
    """运行回测引擎"""
    mask = ~(daily_signal.isna() | daily_weight.isna())
    daily_signal = daily_signal[mask].astype(str)
    daily_weight = daily_weight[mask].astype(float)
    g_align = g_close_s.loc[daily_signal.index]
    v_align = v_close_s.loc[daily_signal.index]

    n = len(daily_signal)
    dates = daily_signal.index.strftime('%Y-%m-%d').values
    signal = daily_signal.values.astype(str)
    position_weight = daily_weight.values.astype(np.float64)

    v_open = v_align.values.astype(np.float64)
    v_close_arr = v_align.values.astype(np.float64)
    g_open = g_align.values.astype(np.float64)
    g_close_arr = g_align.values.astype(np.float64)

    bt_input = BacktestInput(
        dates=dates,
        value_open=v_open, value_close=v_close_arr,
        growth_open=g_open, growth_close=g_close_arr,
        signal=signal,
    )
    config = BacktestConfig(
        start_cash=1_000_000.0,
        commission=0.0001,
        impact_slippage=0.0,
        apply_gap_slippage=False,
    )
    return run_backtest_engine_weighted(bt_input, config, position_weight)


def calc_metrics_v72(result, freq=252, rf_annual=0.025):
    """V72风格指标计算（与X2引擎完全一致）"""
    df_r = result.to_dataframe()
    r = df_r['daily_ret']
    eq = (1 + r).cumprod()
    n = len(r)
    years = n / freq
    total = eq.iloc[-1] - 1
    ann = (1 + total) ** (1 / years) - 1 if years > 0 else 0
    vol = r.std() * np.sqrt(freq)
    rf_p = rf_annual / freq
    sharpe = (r.mean() - rf_p) / r.std() * np.sqrt(freq) if r.std() > 0 else 0
    peak = eq.cummax()
    dd = (eq - peak) / peak
    max_dd = dd.min()
    calmar = ann / abs(max_dd) if max_dd < 0 else 0
    wr = (r > 0).sum() / (r != 0).sum() if (r != 0).sum() > 0 else 0
    return {
        'ann': ann, 'dd': max_dd, 'sharpe': sharpe, 'calmar': calmar,
        'wr': wr, 'total': total,
        'n_trades': result.metrics['num_trades'],
        'final_nav': result.metrics['final_nav'],
        'final_multiple': result.metrics['final_multiple'],
        'num_days': result.metrics['num_days'],
    }


# ============================================================
# 8. 公平起始日对齐（60日热身期）
# ============================================================
# 60日ROC需要前60个交易日作为热身期
first_valid_roc = weekly['roc_g_60'].first_valid_index()
print(f"\n[公平对比] 60日ROC首次有效日: {first_valid_roc:%Y-%m-%d}")
weekly_fair = weekly.loc[first_valid_roc:].copy()
print(f"  公平对比周数: {len(weekly_fair)}")
print(f"  公平对比区间: {weekly_fair.index[0]:%Y-%m-%d} ~ {weekly_fair.index[-1]:%Y-%m-%d}")


# ============================================================
# 9. 运行5个消融方案
# ============================================================
print("\n" + "=" * 70)
print("  消融分析：5方案对比（公平起始日对齐）")
print("=" * 70)

schemes = [
    ('X2-baseline',       'X2-A原版（无覆盖层）',                    None),
    ('X6-full',           'X2-A+都涨覆盖(60日ROC)',                  ('final_g_60', None, False)),
    ('X6-window-20',      'X2-A+都涨覆盖(20日ROC)',                  ('final_g_20', None, False)),
    ('X6-window-120',     'X2-A+都涨覆盖(120日ROC)',                 ('final_g_120', None, False)),
    ('X6-both-down-cash', 'X2-A+都涨覆盖(60日)+都跌清仓',            ('final_g_60', 'both_down_60', True)),
]

results = {}
for name, desc, config in schemes:
    print(f"\n[运行] {name}: {desc}")
    if config is None:
        wsig, ww = build_x2_baseline(weekly_fair)
    else:
        final_g_col, both_down_col, cash_on_both_down = config
        wsig, ww = build_x6(weekly_fair, final_g_col=final_g_col,
                            both_down_col=both_down_col, cash_on_both_down=cash_on_both_down)

    dsig = expand_to_daily(wsig, g_close.index)
    dw = expand_to_daily(ww, g_close.index)
    dsig = dsig.loc[first_valid_roc:]
    dw = dw.loc[first_valid_roc:]

    result = run_backtest(dsig, dw, g_close, v_close)
    m = calc_metrics_v72(result)
    results[name] = (result, m, ww)
    print(f"  年化={m['ann']*100:6.2f}%  回撤={m['dd']*100:6.2f}%  "
          f"Sharpe={m['sharpe']:.3f}  Calmar={m['calmar']:.3f}  "
          f"调仓={m['n_trades']}")


# ============================================================
# 10. 主对比表
# ============================================================
print("\n" + "=" * 70)
print("  X6 消融分析 — 主对比表（公平起始日对齐）")
print("=" * 70)
header = f"  {'方案':<22}{'年化':>10}{'回撤':>10}{'Sharpe':>9}{'Calmar':>9}{'调仓':>8}"
print(header)
print("  " + "-" * 64)
for name, _, _ in schemes:
    r, m, _ = results[name]
    print(f"  {name:<22}{m['ann']*100:>9.2f}%{m['dd']*100:>9.2f}%"
          f"{m['sharpe']:>9.3f}{m['calmar']:>9.3f}{m['n_trades']:>8}")
print("=" * 70)

# X6-full vs X2基线 改进幅度
x2_m = results['X2-baseline'][1]
x6_m = results['X6-full'][1]
d_ann = (x6_m['ann'] - x2_m['ann']) * 100
d_dd = (x6_m['dd'] - x2_m['dd']) * 100
d_sharpe = x6_m['sharpe'] - x2_m['sharpe']
d_calmar = x6_m['calmar'] - x2_m['calmar']

print(f"\n  X6-full vs X2-baseline 改进幅度:")
print(f"    年化:    {x2_m['ann']*100:6.2f}% → {x6_m['ann']*100:6.2f}%  ({d_ann:+.2f}pp)")
print(f"    回撤:    {x2_m['dd']*100:6.2f}% → {x6_m['dd']*100:6.2f}%  ({d_dd:+.2f}pp, 正=改善)")
print(f"    Sharpe:  {x2_m['sharpe']:6.3f} → {x6_m['sharpe']:6.3f}  ({d_sharpe:+.3f})")
print(f"    Calmar:  {x2_m['calmar']:6.3f} → {x6_m['calmar']:6.3f}  ({d_calmar:+.3f})")


# ============================================================
# 11. 通过标准检查
# ============================================================
print("\n" + "=" * 70)
print("  X6-full 通过标准检查（B研究员硬约束：年化≥37.49%）")
print("=" * 70)
checks = [
    ('年化收益 ≥ X2基线 (37.49%)',  x6_m['ann']    >= x2_m['ann']),
    ('最大回撤 ≤ X2基线 (-37.66%)', x6_m['dd']     >= x2_m['dd']),
    ('Sharpe ≥ X2基线 (1.298)',     x6_m['sharpe'] >= x2_m['sharpe']),
    ('Calmar ≥ X2基线 (0.995)',     x6_m['calmar'] >= x2_m['calmar']),
]
all_pass = True
for label, ok in checks:
    flag = '✓' if ok else '✗'
    print(f"    [{flag}] {label}")
    if not ok:
        all_pass = False

# 都涨状态识别频率
both_up_rate = weekly_fair['both_up_60'].sum() / len(weekly_fair) * 100
freq_ok = 10 <= both_up_rate <= 60
print(f"    [{'✓' if freq_ok else '✗'}] 都涨状态识别合理 ({both_up_rate:.1f}%, 期望10-60%)")
if not freq_ok:
    all_pass = False

print(f"\n  >>> 总体结论: {'通过 ✓' if all_pass else '失败 ✗'}")
if not all_pass and x6_m['ann'] < x2_m['ann']:
    print(f"  >>> B研究员红线触发：年化低于X2基线，方向A失败")
print("=" * 70)


# ============================================================
# 12. 都涨状态时段分析
# ============================================================
print("\n" + "=" * 70)
print("  都涨状态时段分析（60日ROC）")
print("=" * 70)

weekly_fair_copy = weekly_fair.copy()
weekly_fair_copy['year'] = weekly_fair_copy.index.year
weekly_fair_copy['both_up'] = weekly_fair_copy['both_up_60'].astype(bool)

yearly_up = weekly_fair_copy.groupby('year')['both_up'].sum()
print(f"\n  年度都涨周数统计:")
print(f"  {'年份':<8}{'都涨周数':>10}{'总周数':>10}{'都涨率':>10}")
print("  " + "-" * 38)
for year, cnt in yearly_up.items():
    total = (weekly_fair_copy['year'] == year).sum()
    rate = cnt / total * 100 if total > 0 else 0
    flag = '***' if cnt > 0 else ''
    print(f"  {year:<8}{cnt:>10}{total:>10}{rate:>9.1f}% {flag}")

# 关键时段都涨情况
print(f"\n  关键时段都涨情况:")
key_periods = [
    ('2014H2牛市启动',  '2014-07', '2014-12'),
    ('2015股灾前牛市',  '2015-01', '2015-06'),
    ('2017蓝筹行情',    '2017-01', '2017-12'),
    ('2019Q1反弹',      '2019-01', '2019-04'),
    ('2020-2021牛市',   '2020-04', '2021-12'),
    ('2022熊市',        '2022-01', '2022-10'),
    ('2024-2025反弹',   '2024-01', '2025-06'),
]
for name, start, end in key_periods:
    mask = (weekly_fair_copy.index >= pd.Timestamp(start)) & (weekly_fair_copy.index <= pd.Timestamp(end))
    period_data = weekly_fair_copy[mask]
    if len(period_data) > 0:
        up_count = period_data['both_up'].sum()
        total = len(period_data)
        print(f"    {name} ({start}~{end}): 都涨 {up_count}/{total} 周 ({up_count/total*100:.1f}%)")


# ============================================================
# 13. 保存详细报告
# ============================================================
report_lines = []
def w(line=""):
    report_lines.append(line)

w("=" * 70)
w("  X6 策略消融分析报告 — 60日ROC四状态，都涨超配成长")
w("=" * 70)
w()
w(f"数据区间: {g_close.index[0]:%Y-%m-%d} ~ {g_close.index[-1]:%Y-%m-%d}")
w(f"公平对比区间: {weekly_fair.index[0]:%Y-%m-%d} ~ {weekly_fair.index[-1]:%Y-%m-%d}")
w(f"交易日数: {len(g_close)}, 周数: {len(weekly_fair)}")
w()
w("------------------------------------------------------------------------------")
w("一、策略设计")
w("------------------------------------------------------------------------------")
w()
w("  X6 = X2-A（分段降仓基线）+ 都涨状态覆盖层")
w()
w("  信号层（保持X2-A不变）:")
w("    F1 = tanh(ratio_dev × 30) × 0.5        # 比价MA20方向")
w("    F2 = accel_diff.clip(±0.02) × 5.0      # 动量加速度")
w("    style_score = F1 + F2")
w("    candidate_g = style_score > 0           # 二元方向（相对动量）")
w()
w("  状态覆盖层（X6新增，仅都涨时生效）:")
w("    roc_g60 = g_close.pct_change(60).shift(1)")
w("    roc_v60 = v_close.pct_change(60).shift(1)")
w("    both_up = (roc_g60 > 0) & (roc_v60 > 0)")
w("    final_g = candidate_g | both_up         # 都涨时强制选成长")
w()
w("  择时层（保持X2-A不变）:")
w("    MA50双跌破 → weight=0.50")
w("    MA75双跌破 → weight=0.25")
w()
w("  设计原则:")
w("    1) 不改变F1/F2和X2-A任何参数（因子要少）")
w("    2) 状态覆盖层只在都涨时生效，其他状态保持X2-A原版")
w("    3) 60日ROC是A 股动量有效窗口（1-3月上界）")
w("    4) 符号阈值法（threshold=0，最简形式）")
w("    5) shift(1)避免前视，T日决策T+1执行")
w()
w("------------------------------------------------------------------------------")
w("二、消融分析方案")
w("------------------------------------------------------------------------------")
w()
w(f"  {'方案':<22}{'配置':<55}")
w("  " + "-" * 77)
w(f"  {'X2-baseline':<22}{'X2-A原版（无覆盖层）':<55}")
w(f"  {'X6-full':<22}{'X2-A+都涨覆盖(60日ROC)':<55}")
w(f"  {'X6-window-20':<22}{'X2-A+都涨覆盖(20日ROC)':<55}")
w(f"  {'X6-window-120':<22}{'X2-A+都涨覆盖(120日ROC)':<55}")
w(f"  {'X6-both-down-cash':<22}{'X2-A+都涨覆盖(60日)+都跌清仓':<55}")
w()
w("------------------------------------------------------------------------------")
w("三、主对比表（公平起始日对齐）")
w("------------------------------------------------------------------------------")
w()
w(f"  {'方案':<22}{'年化':>10}{'回撤':>10}{'Sharpe':>9}{'Calmar':>9}{'调仓':>8}")
w("  " + "-" * 64)
for name, _, _ in schemes:
    r, m, _ = results[name]
    w(f"  {name:<22}{m['ann']*100:>9.2f}%{m['dd']*100:>9.2f}%"
      f"{m['sharpe']:>9.3f}{m['calmar']:>9.3f}{m['n_trades']:>8}")
w()
w(f"  X6-full vs X2-baseline 改进幅度:")
w(f"    年化:    {x2_m['ann']*100:6.2f}% → {x6_m['ann']*100:6.2f}%  ({d_ann:+.2f}pp)")
w(f"    回撤:    {x2_m['dd']*100:6.2f}% → {x6_m['dd']*100:6.2f}%  ({d_dd:+.2f}pp, 正=改善)")
w(f"    Sharpe:  {x2_m['sharpe']:6.3f} → {x6_m['sharpe']:6.3f}  ({d_sharpe:+.3f})")
w(f"    Calmar:  {x2_m['calmar']:6.3f} → {x6_m['calmar']:6.3f}  ({d_calmar:+.3f})")
w()
w("------------------------------------------------------------------------------")
w("四、通过标准检查")
w("------------------------------------------------------------------------------")
w()
for label, ok in checks:
    flag = '✓' if ok else '✗'
    w(f"  [{flag}] {label}")
w(f"  [{ '✓' if freq_ok else '✗' }] 都涨状态识别合理 ({both_up_rate:.1f}%, 期望10-60%)")
w()
w(f"  >>> 总体结论: {'通过 ✓' if all_pass else '失败 ✗'}")
if not all_pass and x6_m['ann'] < x2_m['ann']:
    w(f"  >>> B研究员红线触发：年化低于X2基线，方向A失败")
w()
w("------------------------------------------------------------------------------")
w("五、都涨状态时段分析")
w("------------------------------------------------------------------------------")
w()
w(f"  年度都涨周数统计:")
w(f"  {'年份':<8}{'都涨周数':>10}{'总周数':>10}{'都涨率':>10}")
w("  " + "-" * 38)
for year, cnt in yearly_up.items():
    total = (weekly_fair_copy['year'] == year).sum()
    rate = cnt / total * 100 if total > 0 else 0
    flag = '***' if cnt > 0 else ''
    w(f"  {year:<8}{cnt:>10}{total:>10}{rate:>9.1f}% {flag}")
w()
w(f"  关键时段都涨情况:")
for name, start, end in key_periods:
    mask = (weekly_fair_copy.index >= pd.Timestamp(start)) & (weekly_fair_copy.index <= pd.Timestamp(end))
    period_data = weekly_fair_copy[mask]
    if len(period_data) > 0:
        up_count = period_data['both_up'].sum()
        total = len(period_data)
        w(f"    {name} ({start}~{end}): 都涨 {up_count}/{total} 周 ({up_count/total*100:.1f}%)")
w()
w("------------------------------------------------------------------------------")
w("六、各方案详细 summary")
w("------------------------------------------------------------------------------")
for name, _, _ in schemes:
    r, _, _ = results[name]
    w()
    w(r.summary(name))
w()
w("=" * 70)
w("  报告生成完毕")
w("=" * 70)

with open(REPORT_PATH, 'w', encoding='utf-8') as f:
    f.write("\n".join(report_lines))

print(f"\n[报告已保存] {REPORT_PATH}")
print("=" * 70)
print("[完成] X6 策略消融分析执行结束")
