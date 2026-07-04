"""backtest_x9_engine.py — X9 策略回测（F2''-F1正交化诊断）
================================================================
X9 = F1原版 + F2原版 + F2''正交化(RSI差去除F1重叠) + MA50/MA75分段降仓

  目标：验证F2''的+1.99pp边际增量是否真实独立（vs F1重复计数）
  方法：对F2''做F1正交化（F2''_perp = F2'' - β×F1），测独立alpha贡献

消融分析5方案：
  X9-0: F1 + F2原版 + MA（=X2-A基准复现）
  X9-1: F1 + F2原版 + F2''原始 + MA（=X8-4基准复现）
  X9-2: F1 + F2原版 + F2''正交化 + MA（★X9完整版）
  X9-3: F1 + F2''正交化 + MA（无F2原版，验证F2''_perp独立贡献）
  X9-4: F1 + F2''正交化 + F2原版 + MA（顺序差异）

判断逻辑：
  - X9-2年化≥38.18% → F2''_perp有效，X8-4确认，继续X10
  - X9-2年化36.34%~38.18% → 研究员评价
  - X9-2年化<36.34% → X8-4降级，X2-A维持最终最优解，终止优化循环

数据:
  成长100: c:\\temp_v72_data\\index_480080.csv
  价值100: c:\\temp_v72_data\\index_480081.csv
  区间:    2012-12-31 ~ 2026-07-01

回测引擎: backtest_engine.py
================================================================
"""
import sys
sys.path.insert(0, r'c:\XuanJLH\Qbot\custom\backtests')

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from backtest_engine import (
    BacktestInput, BacktestConfig,
    run_backtest_engine_weighted,
)

# ============================================================
# 1. 参数
# ============================================================
F1 = 0.5
F2_ORIG = 5.0
F2_RSI = 0.3
RSI_PERIOD = 14
MA_PCT = 0.97

DATA_DIR = Path(r'c:\temp_v72_data')
REPORT_PATH = DATA_DIR / 'x9_ablation_report.txt'

# ============================================================
# 2. 加载数据
# ============================================================
print("=" * 70)
print("  X9 策略回测 — F2''-F1正交化诊断")
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
# 3. 因子计算
# ============================================================
print("\n[因子计算] F1=比价MA20(0.5) + F2原版=动量加速度(5.0) + F2''=RSI差(0.3) + F2''_perp正交化")

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

# F2''原始
def compute_rsi(close, period=14):
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - 100 / (1 + rs)

g_rsi = compute_rsi(g_close, RSI_PERIOD).shift(1)
v_rsi = compute_rsi(v_close, RSI_PERIOD).shift(1)
rsi_diff = g_rsi - v_rsi
f2_rsi_signal = np.tanh(rsi_diff / 10.0) * F2_RSI

# F2''正交化（去除F1线性影响）
print("\n[正交化] F2''对F1做OLS回归，取残差作为F2''_perp")
valid = ~(f1_signal.isna() | f2_rsi_signal.isna())
reg = LinearRegression().fit(
    f1_signal[valid].values.reshape(-1, 1),
    f2_rsi_signal[valid].values
)
beta = reg.coef_[0]
intercept = reg.intercept_
print(f"  回归系数 β = {beta:.6f}")
print(f"  截距 = {intercept:.6f}")
print(f"  R² = {reg.score(f1_signal[valid].values.reshape(-1,1), f2_rsi_signal[valid].values):.4f}")

# 正交化：F2''_perp = F2'' - (intercept + β×F1)
f2_rsi_perp = f2_rsi_signal - (intercept + beta * f1_signal)

# 验证正交化效果
corr_orig, _ = stats.pearsonr(f1_signal[valid].values, f2_rsi_signal[valid].values)
corr_perp, _ = stats.pearsonr(f1_signal[valid].values, f2_rsi_perp[valid].values)
print(f"  F1 vs F2''原始   Pearson: {corr_orig:+.6f}")
print(f"  F1 vs F2''_perp  Pearson: {corr_perp:+.6f}  (应≈0)")
print(f"  F2''原始 std: {f2_rsi_signal[valid].std():.6f}")
print(f"  F2''_perp std: {f2_rsi_perp[valid].std():.6f}  (剩余独立信息)")

# style_score组合
style_x9_0 = f1_signal + f2_orig_signal                              # X9-0 = X2-A
style_x9_1 = f1_signal + f2_orig_signal + f2_rsi_signal              # X9-1 = X8-4
style_x9_2 = f1_signal + f2_orig_signal + f2_rsi_perp                # X9-2 ★正交版
style_x9_3 = f1_signal + f2_rsi_perp                                  # X9-3 无F2原版
style_x9_4 = f1_signal + f2_rsi_perp + f2_orig_signal                # X9-4 顺序差异

cand_x9_0 = style_x9_0 > 0
cand_x9_1 = style_x9_1 > 0
cand_x9_2 = style_x9_2 > 0
cand_x9_3 = style_x9_3 > 0
cand_x9_4 = style_x9_4 > 0

# ============================================================
# 4. 共线性检测
# ============================================================
print("\n[共线性检测]")

# F2原版 vs F2''_perp
corr_f2orig_perp, _ = stats.pearsonr(f2_orig_signal[valid].values, f2_rsi_perp[valid].values)
print(f"  F2原版 vs F2''_perp Pearson: {corr_f2orig_perp:+.4f}")

# 触发率1：F2''_perp改变X2-A决策（F1+F2原版）的比例
f2perp_override_x2a = (cand_x9_0[valid] != cand_x9_2[valid]).mean()
# 触发率2：F2''_perp改变F1单独决策的比例
cand_f1_only = (f1_signal > 0)
f2perp_override_f1 = (cand_f1_only[valid] != cand_x9_3[valid]).mean()

# 对照：F2''原始的触发率
f2rsi_override_x2a = (cand_x9_0[valid] != cand_x9_1[valid]).mean()

print(f"  F2''_perp改变X2-A决策频率: {f2perp_override_x2a*100:.2f}%")
print(f"  F2''原始改变X2-A决策频率: {f2rsi_override_x2a*100:.2f}%  (X8-4参照)")
print(f"  F2''_perp改变F1决策频率: {f2perp_override_f1*100:.2f}%")

# ============================================================
# 5. 择时信号
# ============================================================
print("\n[择时计算] MA50 + MA75")

g_ma50 = g_close.shift(1).rolling(50).mean()
v_ma50 = v_close.shift(1).rolling(50).mean()
both_below_ma50 = (g_close.shift(1) < g_ma50 * MA_PCT) & (v_close.shift(1) < v_ma50 * MA_PCT)

g_ma75 = g_close.shift(1).rolling(75).mean()
v_ma75 = v_close.shift(1).rolling(75).mean()
both_below_ma75 = (g_close.shift(1) < g_ma75 * MA_PCT) & (v_close.shift(1) < v_ma75 * MA_PCT)

# ============================================================
# 6. 通用函数
# ============================================================
def build_signal_weight(weekly_df, candidate_col, use_ma=True):
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


# ============================================================
# 7. 对齐起始日
# ============================================================
start_f2perp = f2_rsi_perp.first_valid_index()
start_ma75 = both_below_ma75.first_valid_index()
start_idx = max(start_f2perp, start_ma75) + pd.Timedelta(days=7)
end_idx = g_close.index[-1]

print(f"\n公平起始日: {start_idx:%Y-%m-%d} ~ {end_idx:%Y-%m-%d}")

# ============================================================
# 8. 周频采样
# ============================================================
df = pd.DataFrame(index=g_close.index)
df['cand_x9_0'] = cand_x9_0
df['cand_x9_1'] = cand_x9_1
df['cand_x9_2'] = cand_x9_2
df['cand_x9_3'] = cand_x9_3
df['cand_x9_4'] = cand_x9_4
df['both_below_ma50'] = both_below_ma50
df['both_below_ma75'] = both_below_ma75

df_bt = df.loc[start_idx:end_idx]
weekly = df_bt.resample('W-FRI').last().dropna(subset=['cand_x9_0']).iloc[1:]
print(f"周数: {len(weekly)}")

# ============================================================
# 9. 运行5方案消融分析
# ============================================================
results = {}
g_close_bt = g_close.loc[start_idx:end_idx]
v_close_bt = v_close.loc[start_idx:end_idx]

schemes = [
    ('X9-0(F1+F2原版+MA)=X2-A', 'cand_x9_0'),
    ('X9-1(F1+F2原版+F2rsi+MA)=X8-4', 'cand_x9_1'),
    ('X9-2(F1+F2原版+F2perp+MA)★', 'cand_x9_2'),
    ('X9-3(F1+F2perp+MA无F2原版)', 'cand_x9_3'),
    ('X9-4(F1+F2perp+F2原版+MA)', 'cand_x9_4'),
]

for name, col in schemes:
    print(f"\n[运行] {name}")
    wsig, ww = build_signal_weight(weekly, col, use_ma=True)
    dsig = expand_to_daily(wsig, g_close_bt.index)
    dw = expand_to_daily(ww, g_close_bt.index)
    result = run_backtest(dsig, dw, g_close_bt, v_close_bt)
    m = calc_metrics(result)
    results[name] = m
    print(f"  年化={m['ann']*100:6.2f}%  回撤={m['dd']*100:6.2f}%  Sharpe={m['sharpe']:.3f}  Calmar={m['calmar']:.3f}  调仓={m['n_trades']}")

# ============================================================
# 10. 主对比表输出
# ============================================================
print("\n" + "=" * 70)
print("  X9 消融分析报告 — F2''-F1正交化诊断")
print("=" * 70)

print(f"\n公平起始日: {start_idx:%Y-%m-%d} ~ {end_idx:%Y-%m-%d}")

print(f"\n正交化诊断:")
print(f"  F1 vs F2''原始   Pearson: {corr_orig:+.4f}")
print(f"  F1 vs F2''_perp  Pearson: {corr_perp:+.4f}  (应≈0)")
print(f"  F2原版 vs F2''_perp Pearson: {corr_f2orig_perp:+.4f}")
print(f"  F2''_perp std: {f2_rsi_perp[valid].std():.6f}  (剩余独立信息)")
print(f"  F2''_perp改变X2-A决策频率: {f2perp_override_x2a*100:.2f}%")
print(f"  F2''原始改变X2-A决策频率: {f2rsi_override_x2a*100:.2f}%  (X8-4参照)")

print(f"\n{'方案':<40}{'年化':>10}{'回撤':>10}{'Sharpe':>9}{'Calmar':>9}{'调仓':>8}")
print("-" * 86)
for name, _ in schemes:
    m = results[name]
    print(f"{name:<40}{m['ann']*100:>9.2f}%{m['dd']*100:>9.2f}%{m['sharpe']:>9.3f}{m['calmar']:>9.3f}{m['n_trades']:>8}")

print(f"\n{'X2-A基准(参照)':<40}{'36.34':>9}%{'-37.66':>9}%{'1.271':>9}{'0.965':>9}{'~144':>8}")
print(f"{'X8-4基准(参照)':<40}{'38.18':>9}%{'-38.54':>9}%{'1.327':>9}{'0.991':>9}{'~176':>8}")

# ============================================================
# 11. 结论判断
# ============================================================
print("\n" + "=" * 70)
print("  X9 结论判断")
print("=" * 70)

x9_0 = results['X9-0(F1+F2原版+MA)=X2-A']
x9_1 = results['X9-1(F1+F2原版+F2rsi+MA)=X8-4']
x9_2 = results['X9-2(F1+F2原版+F2perp+MA)★']
x9_3 = results['X9-3(F1+F2perp+MA无F2原版)']

print(f"\nX9-2（正交版）vs X8-4基准:")
print(f"  年化: {x9_2['ann']*100:.2f}% vs 38.18%  ({'✅' if x9_2['ann']>=0.3818 else '❌'})")
print(f"  回撤: {x9_2['dd']*100:.2f}% vs -38.54%  ({'✅' if x9_2['dd']>=-0.3854 else '❌'})")
print(f"  Sharpe: {x9_2['sharpe']:.3f} vs 1.327  ({'✅' if x9_2['sharpe']>=1.327 else '❌'})")

print(f"\nX9-2 vs X2-A基准:")
print(f"  年化: {x9_2['ann']*100:.2f}% vs 36.34%  ({'✅' if x9_2['ann']>=0.3634 else '❌'})")

# F2''_perp独立贡献
perp_contrib = (x9_2['ann'] - x9_0['ann']) * 100
orig_contrib = (x9_1['ann'] - x9_0['ann']) * 100
overlap = orig_contrib - perp_contrib

print(f"\nF2''独立alpha分析:")
print(f"  F2''原始贡献（X9-1 - X9-0）: {orig_contrib:+.2f}pp")
print(f"  F2''_perp独立贡献（X9-2 - X9-0）: {perp_contrib:+.2f}pp")
print(f"  F1重叠部分（原始 - 正交）: {overlap:+.2f}pp")
print(f"  独立贡献占比: {perp_contrib/orig_contrib*100:.1f}%" if orig_contrib != 0 else "  独立贡献占比: N/A")

print(f"\n判断:")
if x9_2['ann'] >= 0.3818:
    print(f"  ✅ X9-2年化≥38.18% → F2''_perp有真实独立alpha，X8-4确认有效，继续X10")
elif x9_2['ann'] >= 0.3634:
    print(f"  ⚠️ X9-2年化36.34%~38.18% → F2''_perp部分有效，调用量化研究员评价")
else:
    print(f"  ❌ X9-2年化<36.34% → F2''_perp无独立alpha，X8-4是F1重复计数")
    print(f"  >>> X8-4降级，X2-A维持最终最优解，终止优化循环 <<<")

# ============================================================
# 12. 保存报告
# ============================================================
report_lines = []
def w(line=""):
    report_lines.append(line)

w("=" * 70)
w("  X9 策略回测 — F2''-F1正交化诊断")
w("=" * 70)
w(f"数据区间: {g_close.index[0]:%Y-%m-%d} ~ {g_close.index[-1]:%Y-%m-%d}")
w(f"交易日数: {len(g_close)}")
w("")
w("[正交化] F2''对F1做OLS回归，取残差作为F2''_perp")
w(f"  回归系数 β = {beta:.6f}")
w(f"  截距 = {intercept:.6f}")
w(f"  R² = {reg.score(f1_signal[valid].values.reshape(-1,1), f2_rsi_signal[valid].values):.4f}")
w(f"  F1 vs F2''原始   Pearson: {corr_orig:+.6f}")
w(f"  F1 vs F2''_perp  Pearson: {corr_perp:+.6f}  (应≈0)")
w(f"  F2''原始 std: {f2_rsi_signal[valid].std():.6f}")
w(f"  F2''_perp std: {f2_rsi_perp[valid].std():.6f}")
w("")
w("[共线性检测]")
w(f"  F2原版 vs F2''_perp Pearson: {corr_f2orig_perp:+.4f}")
w(f"  F2''_perp改变X2-A决策频率: {f2perp_override_x2a*100:.2f}%")
w(f"  F2''原始改变X2-A决策频率: {f2rsi_override_x2a*100:.2f}%  (X8-4参照)")
w("")
w(f"公平起始日: {start_idx:%Y-%m-%d} ~ {end_idx:%Y-%m-%d}")
w(f"周数: {len(weekly)}")
w("")
w("=" * 70)
w("  X9 消融分析报告")
w("=" * 70)
w("")
w(f"{'方案':<40}{'年化':>10}{'回撤':>10}{'Sharpe':>9}{'Calmar':>9}{'调仓':>8}")
w("-" * 86)
for name, _ in schemes:
    m = results[name]
    w(f"{name:<40}{m['ann']*100:>9.2f}%{m['dd']*100:>9.2f}%{m['sharpe']:>9.3f}{m['calmar']:>9.3f}{m['n_trades']:>8}")
w("")
w(f"{'X2-A基准(参照)':<40}{'36.34':>9}%{'-37.66':>9}%{'1.271':>9}{'0.965':>9}{'~144':>8}")
w(f"{'X8-4基准(参照)':<40}{'38.18':>9}%{'-38.54':>9}%{'1.327':>9}{'0.991':>9}{'~176':>8}")
w("")
w("=" * 70)
w("  X9 结论判断")
w("=" * 70)
w("")
w(f"X9-2（正交版）vs X8-4基准:")
w(f"  年化: {x9_2['ann']*100:.2f}% vs 38.18%  ({'✅' if x9_2['ann']>=0.3818 else '❌'})")
w(f"  回撤: {x9_2['dd']*100:.2f}% vs -38.54%  ({'✅' if x9_2['dd']>=-0.3854 else '❌'})")
w(f"  Sharpe: {x9_2['sharpe']:.3f} vs 1.327  ({'✅' if x9_2['sharpe']>=1.327 else '❌'})")
w("")
w(f"X9-2 vs X2-A基准:")
w(f"  年化: {x9_2['ann']*100:.2f}% vs 36.34%  ({'✅' if x9_2['ann']>=0.3634 else '❌'})")
w("")
w(f"F2''独立alpha分析:")
w(f"  F2''原始贡献（X9-1 - X9-0）: {orig_contrib:+.2f}pp")
w(f"  F2''_perp独立贡献（X9-2 - X9-0）: {perp_contrib:+.2f}pp")
w(f"  F1重叠部分（原始 - 正交）: {overlap:+.2f}pp")
if orig_contrib != 0:
    w(f"  独立贡献占比: {perp_contrib/orig_contrib*100:.1f}%")
w("")
if x9_2['ann'] >= 0.3818:
    w(">>> ✅ X9-2年化≥38.18% → F2''_perp有真实独立alpha，X8-4确认有效，继续X10 <<<")
elif x9_2['ann'] >= 0.3634:
    w(">>> ⚠️ X9-2年化36.34%~38.18% → F2''_perp部分有效，调用量化研究员评价 <<<")
else:
    w(">>> ❌ X9-2年化<36.34% → F2''_perp无独立alpha，X8-4是F1重复计数 <<<")
    w(">>> X8-4降级，X2-A维持最终最优解，终止优化循环 <<<")

REPORT_PATH.write_text("\n".join(report_lines), encoding='utf-8')
print(f"\n报告已保存: {REPORT_PATH}")
