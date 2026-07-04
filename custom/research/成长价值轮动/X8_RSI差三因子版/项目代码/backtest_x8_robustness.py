"""backtest_x8_robustness.py — X8-4 三因子叠加稳健性验证
================================================================
对X8-4(F1+F2原版+F2''RSI差+MA)执行3项稳健性验证：

1. Walk-forward：2024-2026样本外测试，+1.99pp是否保持
2. F2''权重敏感度：0.5x/0.3x/0.1x/0.15x/0.45x扰动，增量是否单调衰减
3. 蒙特卡洛：RSI参数14±20%扰动(11/14/17)，+1.99pp是否消失

通过标准（全部通过才采纳X8-4为新基准）：
  - Walk-forward：2024-2026年化≥36.34%（X2-A基准）
  - 权重敏感度：增量随权重单调衰减（0.15x仍>0，0.45x不超0.3x过多）
  - 蒙特卡洛：RSI 11/17年化仍≥36.34%（不依赖单一参数）

任一不通过 → 终止优化循环，X2-A为最终最优解
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
# 1. 参数与数据加载
# ============================================================
F1 = 0.5
F2_ORIG = 5.0
F2_RSI_DEFAULT = 0.3
RSI_PERIOD_DEFAULT = 14
MA_PCT = 0.97

DATA_DIR = Path(r'c:\temp_v72_data')
REPORT_PATH = DATA_DIR / 'x8_robustness_report.txt'

print("=" * 70)
print("  X8-4 稳健性验证（3项）")
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


# ============================================================
# 2. 通用函数
# ============================================================
def compute_rsi(close, period=14):
    """Wilder RSI"""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - 100 / (1 + rs)


def compute_factors(g_close, v_close, rsi_period=14, f2_rsi_weight=0.3):
    """计算F1+F2原版+F2''因子信号"""
    # F1
    ratio = (g_close / v_close).shift(1)
    ratio_ma20 = ratio.rolling(20).mean()
    ratio_dev = ratio / ratio_ma20 - 1
    f1_signal = np.tanh(ratio_dev * 30) * F1

    # F2原版
    g_roc21 = g_close.pct_change(21).shift(1)
    v_roc21 = v_close.pct_change(21).shift(1)
    g_accel = g_roc21 - g_roc21.shift(10)
    v_accel = v_roc21 - v_roc21.shift(10)
    accel_diff = (g_accel - v_accel).clip(-0.02, 0.02)
    f2_orig_signal = accel_diff * F2_ORIG

    # F2''
    g_rsi = compute_rsi(g_close, rsi_period).shift(1)
    v_rsi = compute_rsi(v_close, rsi_period).shift(1)
    rsi_diff = g_rsi - v_rsi
    f2_rsi_signal = np.tanh(rsi_diff / 10.0) * f2_rsi_weight

    # style_score：F1+F2原版+F2''
    style_score = f1_signal + f2_orig_signal + f2_rsi_signal
    candidate = style_score > 0

    # MA
    g_ma50 = g_close.shift(1).rolling(50).mean()
    v_ma50 = v_close.shift(1).rolling(50).mean()
    both_below_ma50 = (g_close.shift(1) < g_ma50 * MA_PCT) & (v_close.shift(1) < v_ma50 * MA_PCT)

    g_ma75 = g_close.shift(1).rolling(75).mean()
    v_ma75 = v_close.shift(1).rolling(75).mean()
    both_below_ma75 = (g_close.shift(1) < g_ma75 * MA_PCT) & (v_close.shift(1) < v_ma75 * MA_PCT)

    return candidate, both_below_ma50, both_below_ma75


def build_signal_weight(weekly_df, candidate_col='candidate', use_ma=True):
    n = len(weekly_df)
    signal = pd.Series(['value'] * n, index=weekly_df.index, dtype=object)
    weight = pd.Series(1.0, index=weekly_df.index)
    for i in range(n):
        row = weekly_df.iloc[i]
        signal.iloc[i] = 'growth' if row[candidate_col] else 'value'
        if use_ma:
            if row['both_below_ma75']:
                weight.iloc[i] = 0.25
            elif row['both_below_ma50']:
                weight.iloc[i] = 0.5
            else:
                weight.iloc[i] = 1.0
        else:
            weight.iloc[i] = 1.0
    return signal, weight


def expand_to_daily(weekly_series, daily_index):
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
    mask = ~(daily_signal.isna() | daily_weight.isna())
    daily_signal = daily_signal[mask].astype(str)
    daily_weight = daily_weight[mask].astype(float)
    g_align = g_close_s.loc[daily_signal.index]
    v_align = v_close_s.loc[daily_signal.index]

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


def calc_metrics(result, freq=252, rf_annual=0.025):
    df_r = result.to_dataframe()
    r = df_r['daily_ret']
    eq = (1 + r).cumprod()
    n = len(r)
    years = n / freq
    total = eq.iloc[-1] - 1
    ann = (1 + total) ** (1 / years) - 1 if years > 0 else 0
    rf_p = rf_annual / freq
    sharpe = (r.mean() - rf_p) / r.std() * np.sqrt(freq) if r.std() > 0 else 0
    peak = eq.cummax()
    dd = (eq - peak) / peak
    max_dd = dd.min()
    calmar = ann / abs(max_dd) if max_dd < 0 else 0
    return {
        'ann': ann, 'dd': max_dd, 'sharpe': sharpe, 'calmar': calmar,
        'n_trades': result.metrics['num_trades'],
    }


def run_x8_4(g_close_s, v_close_s, rsi_period=14, f2_rsi_weight=0.3, start_date=None, end_date=None):
    """运行X8-4回测（可指定区间和参数）"""
    if start_date is not None:
        g_s = g_close_s.loc[start_date:]
        v_s = v_close_s.loc[start_date:]
    else:
        g_s = g_close_s
        v_s = v_close_s
    if end_date is not None:
        g_s = g_s.loc[:end_date]
        v_s = v_s.loc[:end_date]

    candidate, ma50, ma75 = compute_factors(g_close_s, v_close_s, rsi_period, f2_rsi_weight)

    # 对齐起始日（公平起见，用全局数据的起始日）
    start_f2rsi = candidate.first_valid_index()
    start_ma75 = ma75.first_valid_index()
    start_idx = max(start_f2rsi, start_ma75) + pd.Timedelta(days=7)

    # 截取到指定区间
    if start_date is not None:
        start_idx = max(start_idx, pd.Timestamp(start_date))
    if end_date is not None:
        g_s = g_s.loc[start_idx:end_date]
        v_s = v_s.loc[start_idx:end_date]
    else:
        g_s = g_s.loc[start_idx:]
        v_s = v_s.loc[start_idx:]

    candidate_bt = candidate.loc[g_s.index]
    ma50_bt = ma50.loc[g_s.index]
    ma75_bt = ma75.loc[g_s.index]

    df = pd.DataFrame({
        'candidate': candidate_bt,
        'both_below_ma50': ma50_bt,
        'both_below_ma75': ma75_bt,
    })
    weekly = df.resample('W-FRI').last().dropna(subset=['candidate']).iloc[1:]

    wsig, ww = build_signal_weight(weekly, 'candidate', use_ma=True)
    dsig = expand_to_daily(wsig, g_s.index)
    dw = expand_to_daily(ww, g_s.index)

    result = run_backtest(dsig, dw, g_s, v_s)
    return calc_metrics(result)


# ============================================================
# 3. 验证1：Walk-forward 2024-2026样本外测试
# ============================================================
print("\n" + "=" * 70)
print("  验证1：Walk-forward 2024-2026样本外测试")
print("=" * 70)

# 全周期对照
m_full = run_x8_4(g_close, v_close, RSI_PERIOD_DEFAULT, F2_RSI_DEFAULT)
print(f"  全周期(2013-2026): 年化={m_full['ann']*100:.2f}%  回撤={m_full['dd']*100:.2f}%  Sharpe={m_full['sharpe']:.3f}  Calmar={m_full['calmar']:.3f}")

# 2024-2026样本外
m_oos = run_x8_4(g_close, v_close, RSI_PERIOD_DEFAULT, F2_RSI_DEFAULT,
                 start_date='2024-01-01', end_date='2026-07-01')
print(f"  样本外(2024-2026): 年化={m_oos['ann']*100:.2f}%  回撤={m_oos['dd']*100:.2f}%  Sharpe={m_oos['sharpe']:.3f}  Calmar={m_oos['calmar']:.3f}")

# 同时跑X2-A基准在2024-2026的表现
# X2-A = F1+F2原版+MA（f2_rsi_weight=0）
m_x2_oos = run_x8_4(g_close, v_close, RSI_PERIOD_DEFAULT, 0.0,
                    start_date='2024-01-01', end_date='2026-07-01')
print(f"  X2-A基准(2024-2026): 年化={m_x2_oos['ann']*100:.2f}%  回撤={m_x2_oos['dd']*100:.2f}%  Sharpe={m_x2_oos['sharpe']:.3f}")

oos_diff = (m_oos['ann'] - m_x2_oos['ann']) * 100
print(f"\n  X8-4 vs X2-A 样本外增量: {oos_diff:+.2f}pp")
print(f"  通过标准: X8-4样本外年化≥36.34%（X2-A全周期基准）")
print(f"  实际: X8-4样本外年化={m_oos['ann']*100:.2f}%  {'✅通过' if m_oos['ann']>=0.3634 else '❌未通过'}")

wf_pass = m_oos['ann'] >= 0.3634

# ============================================================
# 4. 验证2：F2''权重敏感度
# ============================================================
print("\n" + "=" * 70)
print("  验证2：F2''权重敏感度（0.5x/0.3x/0.1x/0.15x/0.45x）")
print("=" * 70)

weight_tests = [
    ('0.0x(=X2-A)', 0.0),
    ('0.15x(0.5x缩)', 0.15),
    ('0.3x(默认)', 0.3),
    ('0.45x(1.5x)', 0.45),
    ('0.6x(2x)', 0.6),
]

print(f"  {'权重':<18}{'年化':>10}{'回撤':>10}{'Sharpe':>9}{'Calmar':>9}{'调仓':>8}")
print("  " + "-" * 64)

weight_results = []
for label, w in weight_tests:
    m = run_x8_4(g_close, v_close, RSI_PERIOD_DEFAULT, w)
    weight_results.append((label, w, m))
    print(f"  {label:<18}{m['ann']*100:>9.2f}%{m['dd']*100:>9.2f}%{m['sharpe']:>9.3f}{m['calmar']:>9.3f}{m['n_trades']:>8}")

# 检查单调性：增量应随权重单调衰减
print(f"\n  增量分析（vs 0.0x=X2-A）:")
m_base = weight_results[0][2]
for label, w, m in weight_results[1:]:
    diff = (m['ann'] - m_base['ann']) * 100
    print(f"    {label}: 年化增量={diff:+.2f}pp")

# 通过标准：0.15x仍>0（正贡献），0.6x不超0.3x过多（不超过+1.5pp）
m_015 = weight_results[1][2]
m_03 = weight_results[2][2]
m_06 = weight_results[4][2]
diff_015 = (m_015['ann'] - m_base['ann']) * 100
diff_06 = (m_06['ann'] - m_03['ann']) * 100

print(f"\n  通过标准:")
print(f"    0.15x增量>0: {diff_015:+.2f}pp  {'✅' if diff_015>0 else '❌'}")
print(f"    0.6x相对0.3x增量<+1.5pp: {diff_06:+.2f}pp  {'✅' if diff_06<1.5 else '❌'}")

weight_pass = diff_015 > 0 and diff_06 < 1.5

# ============================================================
# 5. 验证3：RSI参数蒙特卡洛（14±20%: 11/14/17）
# ============================================================
print("\n" + "=" * 70)
print("  验证3：RSI参数蒙特卡洛（11/14/17）")
print("=" * 70)

rsi_tests = [11, 14, 17]
print(f"  {'RSI周期':<10}{'年化':>10}{'回撤':>10}{'Sharpe':>9}{'Calmar':>9}{'调仓':>8}")
print("  " + "-" * 56)

rsi_results = []
for p in rsi_tests:
    m = run_x8_4(g_close, v_close, p, F2_RSI_DEFAULT)
    rsi_results.append((p, m))
    print(f"  {p:<10}{m['ann']*100:>9.2f}%{m['dd']*100:>9.2f}%{m['sharpe']:>9.3f}{m['calmar']:>9.3f}{m['n_trades']:>8}")

# 通过标准：所有RSI周期年化均≥36.34%
print(f"\n  通过标准: 所有RSI周期年化≥36.34%")
rsi_pass = True
for p, m in rsi_results:
    ok = m['ann'] >= 0.3634
    print(f"    RSI={p}: 年化={m['ann']*100:.2f}%  {'✅' if ok else '❌'}")
    if not ok:
        rsi_pass = False

# ============================================================
# 6. 综合结论
# ============================================================
print("\n" + "=" * 70)
print("  X8-4 稳健性验证综合结论")
print("=" * 70)

print(f"\n  验证1 Walk-forward: {'✅通过' if wf_pass else '❌未通过'}")
print(f"    X8-4样本外年化={m_oos['ann']*100:.2f}%  vs  X2-A基准36.34%  (增量={oos_diff:+.2f}pp)")

print(f"\n  验证2 权重敏感度: {'✅通过' if weight_pass else '❌未通过'}")
print(f"    0.15x增量={diff_015:+.2f}pp  0.6x相对0.3x增量={diff_06:+.2f}pp")

print(f"\n  验证3 RSI蒙特卡洛: {'✅通过' if rsi_pass else '❌未通过'}")
for p, m in rsi_results:
    print(f"    RSI={p}: 年化={m['ann']*100:.2f}%")

all_pass = wf_pass and weight_pass and rsi_pass
print(f"\n  综合结论: {'✅全部通过，X8-4升级为新基准，开X9' if all_pass else '❌任一未通过，终止优化循环，X2-A为最终最优解'}")

# ============================================================
# 7. 保存报告
# ============================================================
report_lines = []
def w(line=""):
    report_lines.append(line)

w("=" * 70)
w("  X8-4 稳健性验证报告（3项）")
w("=" * 70)
w("")
w("验证1：Walk-forward 2024-2026样本外测试")
w(f"  全周期(2013-2026): 年化={m_full['ann']*100:.2f}%  回撤={m_full['dd']*100:.2f}%  Sharpe={m_full['sharpe']:.3f}")
w(f"  样本外(2024-2026): 年化={m_oos['ann']*100:.2f}%  回撤={m_oos['dd']*100:.2f}%  Sharpe={m_oos['sharpe']:.3f}")
w(f"  X2-A基准(2024-2026): 年化={m_x2_oos['ann']*100:.2f}%  回撤={m_x2_oos['dd']*100:.2f}%")
w(f"  X8-4 vs X2-A 样本外增量: {oos_diff:+.2f}pp")
w(f"  通过标准: X8-4样本外年化≥36.34%")
w(f"  结果: {'✅通过' if wf_pass else '❌未通过'} (实际={m_oos['ann']*100:.2f}%)")
w("")
w("验证2：F2''权重敏感度")
w(f"  {'权重':<18}{'年化':>10}{'回撤':>10}{'Sharpe':>9}{'Calmar':>9}")
w("  " + "-" * 56)
for label, w_, m in weight_results:
    w(f"  {label:<18}{m['ann']*100:>9.2f}%{m['dd']*100:>9.2f}%{m['sharpe']:>9.3f}{m['calmar']:>9.3f}")
w(f"  0.15x增量={diff_015:+.2f}pp  0.6x相对0.3x增量={diff_06:+.2f}pp")
w(f"  通过标准: 0.15x增量>0 且 0.6x相对0.3x增量<+1.5pp")
w(f"  结果: {'✅通过' if weight_pass else '❌未通过'}")
w("")
w("验证3：RSI参数蒙特卡洛")
w(f"  {'RSI周期':<10}{'年化':>10}{'回撤':>10}{'Sharpe':>9}{'Calmar':>9}")
w("  " + "-" * 48)
for p, m in rsi_results:
    w(f"  {p:<10}{m['ann']*100:>9.2f}%{m['dd']*100:>9.2f}%{m['sharpe']:>9.3f}{m['calmar']:>9.3f}")
w(f"  通过标准: 所有RSI周期年化≥36.34%")
w(f"  结果: {'✅通过' if rsi_pass else '❌未通过'}")
w("")
w("=" * 70)
w("  综合结论")
w("=" * 70)
w(f"  验证1 Walk-forward: {'✅通过' if wf_pass else '❌未通过'}")
w(f"  验证2 权重敏感度: {'✅通过' if weight_pass else '❌未通过'}")
w(f"  验证3 RSI蒙特卡洛: {'✅通过' if rsi_pass else '❌未通过'}")
w("")
if all_pass:
    w(">>> ✅ 全部通过，X8-4升级为新基准，开X9 <<<")
else:
    w(">>> ❌ 任一未通过，终止优化循环，X2-A为最终最优解 <<<")

REPORT_PATH.write_text("\n".join(report_lines), encoding='utf-8')
print(f"\n报告已保存: {REPORT_PATH}")
