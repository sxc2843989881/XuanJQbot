"""backtest_x5_engine.py — X5 策略：Dual Momentum 绝对动量过滤
================================================================
X5 = X2-A（分段降仓基线）+ Dual Momentum绝对动量过滤层

  信号层（与X2-A完全一致）:
    F1 = tanh(ratio_dev × 30) × 0.5   # 比价MA20方向
    F2 = accel_diff.clip(±0.02) × 5.0 # 动量加速度
    style_score = F1 + F2
    candidate_g = style_score > 0     # 二元方向（相对动量）

  择时层（X2-A分段降仓 + Dual Momentum绝对动量）:
    MA50双跌破 → weight=0.50
    MA75双跌破 → weight=0.25
    双双负动量(252日) → weight=min(weight, ABS_MOM_CAP=0.25)

  消融分析5方案:
    X2-baseline    : X2-A原版（无绝对动量）
    X5-full        : X2-A + Dual Momentum(252日, cap=0.25, threshold=0)
    X5-thr-5pct    : X2-A + Dual Momentum(252日, cap=0.25, threshold=-0.05)
    X5-lookback-126: X2-A + Dual Momentum(126日, cap=0.25, threshold=0)
    X5-cap-0       : X2-A + Dual Momentum(252日, cap=0.0,  threshold=0)

  设计原则:
    1) 不改变F1/F2和X2-A的任何参数（因子要少，避免X3多组件干扰教训）
    2) Dual Momentum是独立择时层，不干扰信号层
    3) 双双负动量而非单一标的负动量（符合Antonacci原始定义）
    4) ABS_MOM_CAP=0.25与MA75降仓一致（避免双重降仓冲突）
    5) 252日回看期（Antonacci原始论文，避免X4 RSRS过于敏感教训）

数据:
  成长100: c:\\temp_v72_data\\index_480080.csv
  价值100: c:\\temp_v72_data\\index_480081.csv
  区间:    2012-12-31 ~ 2026-07-01

回测引擎: backtest_engine.py (run_backtest_engine_weighted)
  指数 open=close，佣金万1，无冲击/跳空滑点，周频W-FRI调仓
  Sharpe 用 rf=2.5%/年（与X1/X2口径一致）
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
MA_PCT = 0.97       # MA跌破阈值

# Dual Momentum参数（X5新增）
MOM_LOOKBACK_DEFAULT = 252   # 12月动量回看期
ABS_MOM_CAP_DEFAULT = 0.25   # 绝对动量降仓上限（与MA75一致）
MOM_THRESHOLD_DEFAULT = 0.0  # 动量阈值（<0即负动量）

DATA_DIR = Path(r'c:\temp_v72_data')
REPORT_PATH = DATA_DIR / 'x5_ablation_report.txt'

# ============================================================
# 2. 加载数据
# ============================================================
print("=" * 70)
print("  X5 策略 — Dual Momentum 绝对动量过滤")
print("  X2-A基线 + 双双负动量(252日) → 降仓到0.25")
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

# 二元方向（相对动量）
candidate_g = style_score > 0

# ============================================================
# 4. 择时层：MA50/MA75（X2-A原版）+ Dual Momentum（X5新增）
# ============================================================
print("[择时层] MA50双跌破 + MA75双跌破 + Dual Momentum(252日双双负动量)")

# 趋势防御：MA50/MA75双跌破（X2-A原版）
g_ma50 = g_close.shift(1).rolling(50).mean()
v_ma50 = v_close.shift(1).rolling(50).mean()
both_below_ma50 = (g_close.shift(1) < g_ma50 * MA_PCT) & (v_close.shift(1) < v_ma50 * MA_PCT)

g_ma75 = g_close.shift(1).rolling(75).mean()
v_ma75 = v_close.shift(1).rolling(75).mean()
both_below_ma75 = (g_close.shift(1) < g_ma75 * MA_PCT) & (v_close.shift(1) < v_ma75 * MA_PCT)

# 动量防御：Dual Momentum绝对动量（X5新增）
# 计算多个回看期，供消融分析使用
g_mom_252 = g_close.pct_change(252).shift(1)
v_mom_252 = v_close.pct_change(252).shift(1)
g_mom_126 = g_close.pct_change(126).shift(1)
v_mom_126 = v_close.pct_change(126).shift(1)

# 双双负动量标志
both_neg_mom_252 = (g_mom_252 < MOM_THRESHOLD_DEFAULT) & (v_mom_252 < MOM_THRESHOLD_DEFAULT)
both_neg_mom_252_thr5 = (g_mom_252 < -0.05) & (v_mom_252 < -0.05)  # 宽松阈值
both_neg_mom_126 = (g_mom_126 < MOM_THRESHOLD_DEFAULT) & (v_mom_126 < MOM_THRESHOLD_DEFAULT)

# ============================================================
# 5. 周频采样（W-FRI，与X2-A一致）
# ============================================================
df = pd.DataFrame(index=g_close.index)
df['g_close'] = g_close
df['v_close'] = v_close
df['candidate_g'] = candidate_g
df['both_below_ma50'] = both_below_ma50
df['both_below_ma75'] = both_below_ma75
df['both_neg_mom_252'] = both_neg_mom_252
df['both_neg_mom_252_thr5'] = both_neg_mom_252_thr5
df['both_neg_mom_126'] = both_neg_mom_126
df['g_mom_252'] = g_mom_252
df['v_mom_252'] = v_mom_252
df['style_score'] = style_score

weekly = df.resample('W-FRI').last().dropna(subset=['candidate_g']).iloc[1:]
# 修复：resample('W-FRI').last() 会把 boolean 列转成 float64（pandas bug）
# 强制还原 boolean 列，避免后续 & 操作报错
bool_cols = ['candidate_g', 'both_below_ma50', 'both_below_ma75',
             'both_neg_mom_252', 'both_neg_mom_252_thr5', 'both_neg_mom_126']
for col in bool_cols:
    weekly[col] = weekly[col].astype(bool)
print(f"周数: {len(weekly)}")
print(f"  MA50双跌破周数:        {int(weekly['both_below_ma50'].sum())}")
print(f"  MA75浅跌破周数:        {int(weekly['both_below_ma75'].sum())}")
print(f"  双双负动量(252日)周数: {int(weekly['both_neg_mom_252'].sum())}")
print(f"  双双负动量(252,-5%)周数: {int(weekly['both_neg_mom_252_thr5'].sum())}")
print(f"  双双负动量(126日)周数: {int(weekly['both_neg_mom_126'].sum())}")

# 叠加分析：Dual Momentum vs MA75
overlap_ma75_mom = int((weekly['both_below_ma75'] & weekly['both_neg_mom_252']).sum())
mom_only = int((~weekly['both_below_ma75'] & weekly['both_neg_mom_252']).sum())
ma75_only = int((weekly['both_below_ma75'] & ~weekly['both_neg_mom_252']).sum())
print(f"\n  [叠加分析] MA75 vs Dual Momentum(252日):")
print(f"    两者同时触发: {overlap_ma75_mom} 周")
print(f"    仅Dual Momentum触发: {mom_only} 周")
print(f"    仅MA75触发: {ma75_only} 周")


# ============================================================
# 6. 方案构建函数
# ============================================================
def build_x2_baseline(weekly_df):
    """X2-A基线 — 分段降仓（无绝对动量）
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


def build_x5(weekly_df, mom_col='both_neg_mom_252', abs_mom_cap=0.25):
    """X5 — X2-A + Dual Momentum绝对动量过滤
    分段降仓(MA50→50%, MA75→25%) + 双双负动量→min(weight, abs_mom_cap)
    """
    n = len(weekly_df)
    signal = pd.Series(['value'] * n, index=weekly_df.index, dtype=object)
    weight = pd.Series(1.0, index=weekly_df.index)

    for i in range(n):
        row = weekly_df.iloc[i]
        signal.iloc[i] = 'growth' if row['candidate_g'] else 'value'

        # X2-A分段降仓
        if row['both_below_ma75']:
            w = 0.25
        elif row['both_below_ma50']:
            w = 0.5
        else:
            w = 1.0

        # Dual Momentum绝对动量过滤（X5新增）
        if row[mom_col]:
            w = min(w, abs_mom_cap)

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
# 7. 公平起始日对齐（252日热身期）
# ============================================================
# 252日动量需要前252个交易日作为热身期
# 数据从2012-12-31开始，252个交易日后约为2013-12-xx
# 为公平对比，X2基线和X5都从252日动量首次有效的日期开始
first_valid_mom = weekly['g_mom_252'].first_valid_index()
print(f"\n[公平对比] 252日动量首次有效日: {first_valid_mom:%Y-%m-%d}")
# 截取到公平起始日
weekly_fair = weekly.loc[first_valid_mom:].copy()
print(f"  公平对比周数: {len(weekly_fair)}")
print(f"  公平对比区间: {weekly_fair.index[0]:%Y-%m-%d} ~ {weekly_fair.index[-1]:%Y-%m-%d}")


# ============================================================
# 8. 运行5个消融方案
# ============================================================
print("\n" + "=" * 70)
print("  消融分析：5方案对比（公平起始日对齐）")
print("=" * 70)

schemes = [
    ('X2-baseline',     'X2-A原版（无绝对动量）',                 None),
    ('X5-full',         'X2-A+DualMom(252日, cap=0.25, thr=0)',  ('both_neg_mom_252', 0.25)),
    ('X5-thr-5pct',     'X2-A+DualMom(252日, cap=0.25, thr=-5%)',('both_neg_mom_252_thr5', 0.25)),
    ('X5-lookback-126', 'X2-A+DualMom(126日, cap=0.25, thr=0)',  ('both_neg_mom_126', 0.25)),
    ('X5-cap-0',        'X2-A+DualMom(252日, cap=0.0, thr=0)',   ('both_neg_mom_252', 0.0)),
]

results = {}
for name, desc, config in schemes:
    print(f"\n[运行] {name}: {desc}")
    if config is None:
        # X2基线
        wsig, ww = build_x2_baseline(weekly_fair)
    else:
        mom_col, cap = config
        wsig, ww = build_x5(weekly_fair, mom_col=mom_col, abs_mom_cap=cap)

    dsig = expand_to_daily(wsig, g_close.index)
    dw = expand_to_daily(ww, g_close.index)
    # 截取到公平起始日之后
    dsig = dsig.loc[first_valid_mom:]
    dw = dw.loc[first_valid_mom:]

    result = run_backtest(dsig, dw, g_close, v_close)
    m = calc_metrics_v72(result)
    results[name] = (result, m, ww)
    print(f"  年化={m['ann']*100:6.2f}%  回撤={m['dd']*100:6.2f}%  "
          f"Sharpe={m['sharpe']:.3f}  Calmar={m['calmar']:.3f}  "
          f"调仓={m['n_trades']}")


# ============================================================
# 9. 主对比表
# ============================================================
print("\n" + "=" * 70)
print("  X5 消融分析 — 主对比表（公平起始日对齐）")
print("=" * 70)
header = f"  {'方案':<20}{'年化':>10}{'回撤':>10}{'Sharpe':>9}{'Calmar':>9}{'调仓':>8}"
print(header)
print("  " + "-" * 62)
for name, _, _ in schemes:
    r, m, _ = results[name]
    print(f"  {name:<20}{m['ann']*100:>9.2f}%{m['dd']*100:>9.2f}%"
          f"{m['sharpe']:>9.3f}{m['calmar']:>9.3f}{m['n_trades']:>8}")
print("=" * 70)

# X5-full vs X2基线 改进幅度
x2_m = results['X2-baseline'][1]
x5_m = results['X5-full'][1]
d_ann = (x5_m['ann'] - x2_m['ann']) * 100
d_dd = (x5_m['dd'] - x2_m['dd']) * 100
d_sharpe = x5_m['sharpe'] - x2_m['sharpe']
d_calmar = x5_m['calmar'] - x2_m['calmar']

print(f"\n  X5-full vs X2-baseline 改进幅度:")
print(f"    年化:    {x2_m['ann']*100:6.2f}% → {x5_m['ann']*100:6.2f}%  ({d_ann:+.2f}pp)")
print(f"    回撤:    {x2_m['dd']*100:6.2f}% → {x5_m['dd']*100:6.2f}%  ({d_dd:+.2f}pp, 正=改善)")
print(f"    Sharpe:  {x2_m['sharpe']:6.3f} → {x5_m['sharpe']:6.3f}  ({d_sharpe:+.3f})")
print(f"    Calmar:  {x2_m['calmar']:6.3f} → {x5_m['calmar']:6.3f}  ({d_calmar:+.3f})")


# ============================================================
# 10. 通过标准检查
# ============================================================
print("\n" + "=" * 70)
print("  X5-full 通过标准检查")
print("=" * 70)
checks = [
    ('年化收益 ≥ X2基线',  x5_m['ann']    >= x2_m['ann']),
    ('最大回撤 ≤ X2基线',  x5_m['dd']     >= x2_m['dd']),  # dd为负，>=表示回撤更小
    ('Sharpe ≥ X2基线',    x5_m['sharpe'] >= x2_m['sharpe']),
    ('Calmar ≥ X2基线',    x5_m['calmar'] >= x2_m['calmar']),
]
all_pass = True
for label, ok in checks:
    flag = '✓' if ok else '✗'
    print(f"    [{flag}] {label}")
    if not ok:
        all_pass = False

# 触发频率检查
x5_full_weekly_weight = results['X5-full'][2]
# Dual Momentum单独触发的周数（weight从1.0或0.5降到0.25，但MA75未触发）
# 这里用更精确的统计：both_neg_mom_252为True的周数
mom_trigger_count = int(weekly_fair['both_neg_mom_252'].sum())
freq_ok = 10 <= mom_trigger_count <= 200  # 放宽到200，因为13年可以有较多触发
print(f"    [{'✓' if freq_ok else '✗'}] Dual Momentum触发次数合理 ({mom_trigger_count}次, 期望10-200)")
if not freq_ok:
    all_pass = False

print(f"\n  >>> 总体结论: {'通过 ✓' if all_pass else '失败 ✗'}")
print("=" * 70)


# ============================================================
# 11. Dual Momentum触发时段分析
# ============================================================
print("\n" + "=" * 70)
print("  Dual Momentum触发时段分析（252日双双负动量）")
print("=" * 70)

# 找出连续触发时段
mom_active = weekly_fair['both_neg_mom_252'].astype(bool)
# 计算每个周次的年份
weekly_fair_copy = weekly_fair.copy()
weekly_fair_copy['year'] = weekly_fair_copy.index.year
weekly_fair_copy['mom_active'] = mom_active

yearly_triggers = weekly_fair_copy.groupby('year')['mom_active'].sum()
print(f"\n  年度触发周数统计:")
print(f"  {'年份':<8}{'触发周数':>10}{'总周数':>10}{'触发率':>10}")
print("  " + "-" * 38)
for year, cnt in yearly_triggers.items():
    total = (weekly_fair_copy['year'] == year).sum()
    rate = cnt / total * 100 if total > 0 else 0
    flag = '***' if cnt > 0 else ''
    print(f"  {year:<8}{cnt:>10}{total:>10}{rate:>9.1f}% {flag}")

# 关键熊市时段触发情况
print(f"\n  关键熊市时段触发情况:")
bear_periods = [
    ('2015股灾',     '2015-06', '2015-09'),
    ('2018熊市',     '2018-01', '2018-12'),
    ('2022熊市',     '2022-01', '2022-10'),
    ('2025调整',     '2025-01', '2025-12'),
]
for name, start, end in bear_periods:
    mask = (weekly_fair_copy.index >= pd.Timestamp(start)) & (weekly_fair_copy.index <= pd.Timestamp(end))
    period_data = weekly_fair_copy[mask]
    if len(period_data) > 0:
        triggered = period_data['mom_active'].sum()
        total = len(period_data)
        print(f"    {name} ({start}~{end}): 触发 {triggered}/{total} 周")


# ============================================================
# 12. 保存详细报告
# ============================================================
report_lines = []
def w(line=""):
    report_lines.append(line)

w("=" * 70)
w("  X5 策略消融分析报告 — Dual Momentum 绝对动量过滤")
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
w("  X5 = X2-A（分段降仓基线）+ Dual Momentum绝对动量过滤层")
w()
w("  信号层（与X2-A完全一致）:")
w("    F1 = tanh(ratio_dev × 30) × 0.5   # 比价MA20方向")
w("    F2 = accel_diff.clip(±0.02) × 5.0 # 动量加速度")
w("    style_score = F1 + F2")
w("    candidate_g = style_score > 0     # 二元方向（相对动量）")
w()
w("  择时层:")
w("    趋势防御（X2-A原版）:")
w("      MA50双跌破 → weight=0.50")
w("      MA75双跌破 → weight=0.25")
w("    动量防御（X5新增）:")
w("      g_12m_mom = g_close.pct_change(252).shift(1)")
w("      v_12m_mom = v_close.pct_change(252).shift(1)")
w("      both_neg_mom = (g_12m_mom < 0) & (v_12m_mom < 0)")
w("      if both_neg_mom: weight = min(weight, 0.25)")
w()
w("  设计原则:")
w("    1) 不改变F1/F2和X2-A任何参数（因子要少）")
w("    2) Dual Momentum独立择时层，不干扰信号层")
w("    3) 双双负动量（符合Antonacci原始定义）")
w("    4) ABS_MOM_CAP=0.25与MA75降仓一致（避免双重降仓冲突）")
w("    5) 252日回看期（Antonacci原始论文）")
w()
w("------------------------------------------------------------------------------")
w("二、消融分析方案")
w("------------------------------------------------------------------------------")
w()
w(f"  {'方案':<20}{'配置':<55}")
w("  " + "-" * 75)
w(f"  {'X2-baseline':<20}{'X2-A原版（无绝对动量）':<55}")
w(f"  {'X5-full':<20}{'X2-A+DualMom(252日, cap=0.25, thr=0)':<55}")
w(f"  {'X5-thr-5pct':<20}{'X2-A+DualMom(252日, cap=0.25, thr=-5%)':<55}")
w(f"  {'X5-lookback-126':<20}{'X2-A+DualMom(126日, cap=0.25, thr=0)':<55}")
w(f"  {'X5-cap-0':<20}{'X2-A+DualMom(252日, cap=0.0, thr=0)':<55}")
w()
w("------------------------------------------------------------------------------")
w("三、主对比表（公平起始日对齐）")
w("------------------------------------------------------------------------------")
w()
w(f"  {'方案':<20}{'年化':>10}{'回撤':>10}{'Sharpe':>9}{'Calmar':>9}{'调仓':>8}")
w("  " + "-" * 62)
for name, _, _ in schemes:
    r, m, _ = results[name]
    w(f"  {name:<20}{m['ann']*100:>9.2f}%{m['dd']*100:>9.2f}%"
      f"{m['sharpe']:>9.3f}{m['calmar']:>9.3f}{m['n_trades']:>8}")
w()
w(f"  X5-full vs X2-baseline 改进幅度:")
w(f"    年化:    {x2_m['ann']*100:6.2f}% → {x5_m['ann']*100:6.2f}%  ({d_ann:+.2f}pp)")
w(f"    回撤:    {x2_m['dd']*100:6.2f}% → {x5_m['dd']*100:6.2f}%  ({d_dd:+.2f}pp, 正=改善)")
w(f"    Sharpe:  {x2_m['sharpe']:6.3f} → {x5_m['sharpe']:6.3f}  ({d_sharpe:+.3f})")
w(f"    Calmar:  {x2_m['calmar']:6.3f} → {x5_m['calmar']:6.3f}  ({d_calmar:+.3f})")
w()
w("------------------------------------------------------------------------------")
w("四、通过标准检查")
w("------------------------------------------------------------------------------")
w()
for label, ok in checks:
    flag = '✓' if ok else '✗'
    w(f"  [{flag}] {label}")
w(f"  [{ '✓' if freq_ok else '✗' }] Dual Momentum触发次数合理 ({mom_trigger_count}次, 期望10-200)")
w()
w(f"  >>> 总体结论: {'通过 ✓' if all_pass else '失败 ✗'}")
w()
w("------------------------------------------------------------------------------")
w("五、Dual Momentum触发时段分析")
w("------------------------------------------------------------------------------")
w()
w(f"  年度触发周数统计:")
w(f"  {'年份':<8}{'触发周数':>10}{'总周数':>10}{'触发率':>10}")
w("  " + "-" * 38)
for year, cnt in yearly_triggers.items():
    total = (weekly_fair_copy['year'] == year).sum()
    rate = cnt / total * 100 if total > 0 else 0
    flag = '***' if cnt > 0 else ''
    w(f"  {year:<8}{cnt:>10}{total:>10}{rate:>9.1f}% {flag}")
w()
w(f"  关键熊市时段触发情况:")
for name, start, end in bear_periods:
    mask = (weekly_fair_copy.index >= pd.Timestamp(start)) & (weekly_fair_copy.index <= pd.Timestamp(end))
    period_data = weekly_fair_copy[mask]
    if len(period_data) > 0:
        triggered = period_data['mom_active'].sum()
        total = len(period_data)
        w(f"    {name} ({start}~{end}): 触发 {triggered}/{total} 周")
w()
w("------------------------------------------------------------------------------")
w("六、MA75 vs Dual Momentum 叠加分析")
w("------------------------------------------------------------------------------")
w()
w(f"  两者同时触发: {overlap_ma75_mom} 周")
w(f"  仅Dual Momentum触发: {mom_only} 周")
w(f"  仅MA75触发: {ma75_only} 周")
w()
overlap_rate = overlap_ma75_mom / max(mom_trigger_count, 1) * 100
w(f"  Dual Momentum触发中与MA75重叠的比例: {overlap_rate:.1f}%")
w(f"  → {'Dual Momentum多为独立触发，有独立价值' if overlap_rate < 50 else 'Dual Momentum多与MA75重叠，独立价值有限'}")
w()
w("------------------------------------------------------------------------------")
w("七、各方案详细 summary")
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
print("[完成] X5 策略消融分析执行结束")
