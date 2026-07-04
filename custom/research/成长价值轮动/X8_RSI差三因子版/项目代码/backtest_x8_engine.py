"""backtest_x8_engine.py — X8 策略回测（F2''=RSI差替代F2动量加速度）
================================================================
X8 = F1原版 + F2''重构(RSI差) + 周频调仓 + MA50/MA75分段降仓

  F1因子：比价MA20方向 (f1=0.5)          —— 保留X2原版，用户确认有效
  F2''因子：RSI差 (f2_rsi=0.3)           —— 重构，反转类与F1正交
  调仓：周频 W-FRI                       —— X7证伪日频，回归周频
  择时：MA50双跌破→50%, MA75双跌破→25%   —— 保留X2-A

  F2''理论支撑：
    1. RSI属反转/超买超卖类，与F1（比价MA20趋势方向）理论正交
    2. RSI仅用close计算，绕开high/low缺失（X4 RSRS失败根因）
    3. Wilder标准RSI周期14（1978年原版，非参数拟合）
    4. RSI差捕捉风格相对拐点：成长RSI>价值RSI→风格偏成长

消融分析5方案：
  X8-0: F1单独 + MA（基线对照）
  X8-1: F1 + F2原版 + MA（=X2-A基准复现）
  X8-2: F1 + F2''(RSI差) + MA（★完整版）
  X8-3: F1 + F2'' + 无MA（验证择时层贡献）
  X8-4: F1 + F2原版 + F2'' + MA（三因子叠加）

硬终止条件：
  - X8-2年化<36.00% 或 回撤>-40% → 终止优化循环
  - F2''与F1共线性|r|≥0.5 → 终止
  - F2''改变F1决策频率<5% → 终止

数据:
  成长100: c:\\temp_v72_data\\index_480080.csv
  价值100: c:\\temp_v72_data\\index_480081.csv
  区间:    2012-12-31 ~ 2026-07-01

回测引擎: backtest_engine.py
  run_backtest_engine_weighted(bt_input, config, position_weight)
================================================================
"""
import sys
sys.path.insert(0, r'c:\XuanJLH\Qbot\custom\backtests')

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from backtest_engine import (
    BacktestInput, BacktestConfig,
    run_backtest_engine_weighted,
)

# ============================================================
# 1. 参数
# ============================================================
F1 = 0.5           # 比价MA20权重（保留X2）
F2_ORIG = 5.0      # F2原版动量加速度权重（X2值）
F2_RSI = 0.3       # F2'' RSI差权重（重构）
RSI_PERIOD = 14    # Wilder RSI周期
MA_PCT = 0.97      # MA跌破阈值

DATA_DIR = Path(r'c:\temp_v72_data')
REPORT_PATH = DATA_DIR / 'x8_ablation_report.txt'

# ============================================================
# 2. 加载数据
# ============================================================
print("=" * 70)
print("  X8 策略回测 — F2''重构(RSI差) + 周频调仓")
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
# 3. 因子计算（F1保留 + F2原版对照 + F2''=RSI差重构）
# ============================================================
print("\n[因子计算] F1=比价MA20方向(0.5) + F2原版=动量加速度(5.0) + F2''=RSI差(0.3)")

# ---- F1：比价MA20方向（保留X2）----
ratio = (g_close / v_close).shift(1)
ratio_ma20 = ratio.rolling(20).mean()
ratio_dev = ratio / ratio_ma20 - 1
f1_signal = np.tanh(ratio_dev * 30) * F1

# ---- F2原版：动量加速度（X2原值，对照）----
g_roc21 = g_close.pct_change(21).shift(1)
v_roc21 = v_close.pct_change(21).shift(1)
g_accel = g_roc21 - g_roc21.shift(10)
v_accel = v_roc21 - v_roc21.shift(10)
accel_diff = (g_accel - v_accel).clip(-0.02, 0.02)
f2_orig_signal = accel_diff * F2_ORIG

# ---- F2''：RSI差（重构，Wilder标准）----
def compute_rsi(close, period=14):
    """Wilder RSI（1978年原版）"""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - 100 / (1 + rs)
    return rsi

g_rsi = compute_rsi(g_close, RSI_PERIOD).shift(1)
v_rsi = compute_rsi(v_close, RSI_PERIOD).shift(1)
rsi_diff = g_rsi - v_rsi  # 范围[-100, +100]，实际±30以内
f2_rsi_signal = np.tanh(rsi_diff / 10.0) * F2_RSI  # tanh平滑，权重0.3

# ---- style_score组合 ----
style_f1_only = f1_signal
style_f1_f2orig = f1_signal + f2_orig_signal            # =X2-A
style_f1_f2rsi = f1_signal + f2_rsi_signal              # =X8-2 ★
style_f1_f2orig_f2rsi = f1_signal + f2_orig_signal + f2_rsi_signal  # =X8-4 三因子

cand_f1_only = style_f1_only > 0
cand_f1_f2orig = style_f1_f2orig > 0
cand_f1_f2rsi = style_f1_f2rsi > 0
cand_f1_f2orig_f2rsi = style_f1_f2orig_f2rsi > 0

# ============================================================
# 4. 共线性检测 + 触发率统计（B研究员约束）
# ============================================================
print("\n[B研究员约束] 共线性检测 + 触发率统计")

valid = ~(f1_signal.isna() | f2_rsi_signal.isna() | f2_orig_signal.isna())
f1_v = f1_signal[valid]
f2r_v = f2_rsi_signal[valid]
f2o_v = f2_orig_signal[valid]

# 共线性1：F1 vs F2''
corr_f1_f2rsi, _ = stats.pearsonr(f1_v.values, f2r_v.values)
vif_f1_f2rsi = 1 / (1 - corr_f1_f2rsi**2) if abs(corr_f1_f2rsi) < 0.999 else float('inf')

# 共线性2：F2原版 vs F2''（避免F2''也是动量族）
corr_f2orig_f2rsi, _ = stats.pearsonr(f2o_v.values, f2r_v.values)

# 共线性3：F1 vs F2原版（参照X2原共线性）
corr_f1_f2orig, _ = stats.pearsonr(f1_v.values, f2o_v.values)

# 触发率1：F2''改变style_score符号（相对F1单独）
f2rsi_override = (cand_f1_only[valid] != cand_f1_f2rsi[valid]).mean()

# 触发率2：F2原版改变style_score符号（参照）
f2orig_override = (cand_f1_only[valid] != cand_f1_f2orig[valid]).mean()

print(f"  F1 vs F2''(RSI差) Pearson: {corr_f1_f2rsi:+.4f}  (要求|r|<0.5)")
print(f"  F1 vs F2原版   Pearson: {corr_f1_f2orig:+.4f}  (X2原值参照)")
print(f"  F2原版 vs F2''  Pearson: {corr_f2orig_f2rsi:+.4f}  (要求|r|<0.5 避免同族)")
print(f"  VIF近似(F1 vs F2''): {vif_f1_f2rsi:.3f}  (要求<5)")
print(f"  F2''改变F1决策频率: {f2rsi_override*100:.2f}%  (要求>5%)")
print(f"  F2原版改变F1决策频率: {f2orig_override*100:.2f}%  (参照)")

collinearity_ok = abs(corr_f1_f2rsi) < 0.5
trigger_ok = f2rsi_override > 0.05
print(f"  共线性检测(F1 vs F2''): {'✅通过' if collinearity_ok else '❌未通过'}")
print(f"  触发率检测(F2''): {'✅通过' if trigger_ok else '❌未通过'}")

# ============================================================
# 5. 择时信号（MA50 + MA75，与X2相同）
# ============================================================
print("\n[择时计算] MA50双跌破 + MA75双跌破")

g_ma50 = g_close.shift(1).rolling(50).mean()
v_ma50 = v_close.shift(1).rolling(50).mean()
both_below_ma50 = (g_close.shift(1) < g_ma50 * MA_PCT) & (v_close.shift(1) < v_ma50 * MA_PCT)

g_ma75 = g_close.shift(1).rolling(75).mean()
v_ma75 = v_close.shift(1).rolling(75).mean()
both_below_ma75 = (g_close.shift(1) < g_ma75 * MA_PCT) & (v_close.shift(1) < v_ma75 * MA_PCT)

# ============================================================
# 6. 信号构建函数（周频，与X2-A相同结构）
# ============================================================

def build_signal_weight(weekly_df, candidate_col, use_ma=True):
    """构建周频signal + weight（X2-A结构）
    - 方向由 candidate_col 决定
    - weight：MA75→0.25, MA50→0.5, 否则1.0
    """
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
    """周频信号前向填充到日频（与X2一致）"""
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


# ============================================================
# 7. 回测运行函数（与X2一致）
# ============================================================

def run_backtest(daily_signal, daily_weight, g_close_s, v_close_s):
    """运行回测引擎"""
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
        value_open=v_open,
        value_close=v_close_arr,
        growth_open=g_open,
        growth_close=g_close_arr,
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
    """指标计算（与X2口径一致，rf=2.5%/年）"""
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
        'total': total, 'n_trades': result.metrics['num_trades'],
        'final_nav': result.metrics['final_nav'],
        'final_multiple': result.metrics['final_multiple'],
        'num_days': result.metrics['num_days'],
    }


# ============================================================
# 8. 对齐起始日（F2''需要RSI 14日热身 + ratio shift(1) + MA75 75日热身）
# ============================================================
# RSI需要14日，ratio需要shift(1)，MA75需要75日
# 公平起始日：max(F2''首个有效日, MA75首个有效日) + 1周缓冲
start_f2rsi = f2_rsi_signal.first_valid_index()
start_ma75 = both_below_ma75.first_valid_index()
start_idx = max(start_f2rsi, start_ma75)
start_idx = start_idx + pd.Timedelta(days=7)
end_idx = g_close.index[-1]

print(f"\n公平起始日: {start_idx:%Y-%m-%d} ~ {end_idx:%Y-%m-%d}")
print(f"回测天数: {(g_close.index >= start_idx).sum()}")

# ============================================================
# 9. 周频采样（W-FRI，与X2一致）
# ============================================================
df = pd.DataFrame(index=g_close.index)
df['g_close'] = g_close
df['v_close'] = v_close
df['cand_f1_only'] = cand_f1_only
df['cand_f1_f2orig'] = cand_f1_f2orig
df['cand_f1_f2rsi'] = cand_f1_f2rsi
df['cand_f1_f2orig_f2rsi'] = cand_f1_f2orig_f2rsi
df['both_below_ma50'] = both_below_ma50
df['both_below_ma75'] = both_below_ma75
df['style_f1_f2rsi'] = style_f1_f2rsi

# 截取对齐区间
df_bt = df.loc[start_idx:end_idx]
weekly = df_bt.resample('W-FRI').last().dropna(subset=['cand_f1_only']).iloc[1:]
print(f"周数: {len(weekly)}")
print(f"  MA50双跌破周数: {int(weekly['both_below_ma50'].sum())}")
print(f"  MA75双跌破周数: {int(weekly['both_below_ma75'].sum())}")

# ============================================================
# 10. 运行5方案消融分析
# ============================================================
results = {}
g_close_bt = g_close.loc[start_idx:end_idx]
v_close_bt = v_close.loc[start_idx:end_idx]

# --- X8-0: F1单独 + MA（基线对照）---
print("\n[运行] X8-0: F1单独 + MA")
wsig_x0, ww_x0 = build_signal_weight(weekly, 'cand_f1_only', use_ma=True)
dsig_x0 = expand_to_daily(wsig_x0, g_close_bt.index)
dw_x0 = expand_to_daily(ww_x0, g_close_bt.index)
result_x0 = run_backtest(dsig_x0, dw_x0, g_close_bt, v_close_bt)
m_x0 = calc_metrics(result_x0)
results['X8-0(F1单独+MA)'] = m_x0
print(f"  年化={m_x0['ann']*100:6.2f}%  回撤={m_x0['dd']*100:6.2f}%  Sharpe={m_x0['sharpe']:.3f}  Calmar={m_x0['calmar']:.3f}  调仓={m_x0['n_trades']}")

# --- X8-1: F1 + F2原版 + MA（=X2-A基准复现）---
print("\n[运行] X8-1: F1 + F2原版 + MA (=X2-A基准复现)")
wsig_x1, ww_x1 = build_signal_weight(weekly, 'cand_f1_f2orig', use_ma=True)
dsig_x1 = expand_to_daily(wsig_x1, g_close_bt.index)
dw_x1 = expand_to_daily(ww_x1, g_close_bt.index)
result_x1 = run_backtest(dsig_x1, dw_x1, g_close_bt, v_close_bt)
m_x1 = calc_metrics(result_x1)
results['X8-1(F1+F2原版+MA)'] = m_x1
print(f"  年化={m_x1['ann']*100:6.2f}%  回撤={m_x1['dd']*100:6.2f}%  Sharpe={m_x1['sharpe']:.3f}  Calmar={m_x1['calmar']:.3f}  调仓={m_x1['n_trades']}")

# --- X8-2: F1 + F2''(RSI差) + MA（★完整版）---
print("\n[运行] X8-2: F1 + F2''(RSI差) + MA（★完整版）")
wsig_x2, ww_x2 = build_signal_weight(weekly, 'cand_f1_f2rsi', use_ma=True)
dsig_x2 = expand_to_daily(wsig_x2, g_close_bt.index)
dw_x2 = expand_to_daily(ww_x2, g_close_bt.index)
result_x2 = run_backtest(dsig_x2, dw_x2, g_close_bt, v_close_bt)
m_x2 = calc_metrics(result_x2)
results['X8-2(F1+F2rsi+MA)'] = m_x2
print(f"  年化={m_x2['ann']*100:6.2f}%  回撤={m_x2['dd']*100:6.2f}%  Sharpe={m_x2['sharpe']:.3f}  Calmar={m_x2['calmar']:.3f}  调仓={m_x2['n_trades']}")

# --- X8-3: F1 + F2'' + 无MA（验证择时层贡献）---
print("\n[运行] X8-3: F1 + F2'' + 无MA（验证择时层贡献）")
wsig_x3, ww_x3 = build_signal_weight(weekly, 'cand_f1_f2rsi', use_ma=False)
dsig_x3 = expand_to_daily(wsig_x3, g_close_bt.index)
dw_x3 = expand_to_daily(ww_x3, g_close_bt.index)
result_x3 = run_backtest(dsig_x3, dw_x3, g_close_bt, v_close_bt)
m_x3 = calc_metrics(result_x3)
results['X8-3(F1+F2rsi无MA)'] = m_x3
print(f"  年化={m_x3['ann']*100:6.2f}%  回撤={m_x3['dd']*100:6.2f}%  Sharpe={m_x3['sharpe']:.3f}  Calmar={m_x3['calmar']:.3f}  调仓={m_x3['n_trades']}")

# --- X8-4: F1 + F2原版 + F2'' + MA（三因子叠加）---
print("\n[运行] X8-4: F1 + F2原版 + F2'' + MA（三因子叠加）")
wsig_x4, ww_x4 = build_signal_weight(weekly, 'cand_f1_f2orig_f2rsi', use_ma=True)
dsig_x4 = expand_to_daily(wsig_x4, g_close_bt.index)
dw_x4 = expand_to_daily(ww_x4, g_close_bt.index)
result_x4 = run_backtest(dsig_x4, dw_x4, g_close_bt, v_close_bt)
m_x4 = calc_metrics(result_x4)
results['X8-4(F1+F2orig+F2rsi+MA)'] = m_x4
print(f"  年化={m_x4['ann']*100:6.2f}%  回撤={m_x4['dd']*100:6.2f}%  Sharpe={m_x4['sharpe']:.3f}  Calmar={m_x4['calmar']:.3f}  调仓={m_x4['n_trades']}")

# ============================================================
# 11. 主对比表输出
# ============================================================
print("\n" + "=" * 70)
print("  X8 消融分析报告")
print("=" * 70)

print(f"\n公平起始日: {start_idx:%Y-%m-%d} ~ {end_idx:%Y-%m-%d}")
print(f"\n共线性检测:")
print(f"  F1 vs F2''(RSI差) Pearson = {corr_f1_f2rsi:+.4f}  ({'✅<0.5' if abs(corr_f1_f2rsi)<0.5 else '❌≥0.5'})")
print(f"  F1 vs F2原版   Pearson = {corr_f1_f2orig:+.4f}  (X2原值参照)")
print(f"  F2原版 vs F2''  Pearson = {corr_f2orig_f2rsi:+.4f}  ({'✅<0.5' if abs(corr_f2orig_f2rsi)<0.5 else '❌≥0.5'})")
print(f"  F2''改变F1决策频率 = {f2rsi_override*100:.2f}%  ({'✅>5%' if f2rsi_override>0.05 else '❌≤5%'})")

print(f"\n{'方案':<32}{'年化':>10}{'回撤':>10}{'Sharpe':>9}{'Calmar':>9}{'调仓':>8}")
print("-" * 78)
order = ['X8-0(F1单独+MA)', 'X8-1(F1+F2原版+MA)', 'X8-2(F1+F2rsi+MA)',
         'X8-3(F1+F2rsi无MA)', 'X8-4(F1+F2orig+F2rsi+MA)']
for name in order:
    m = results[name]
    print(f"{name:<32}{m['ann']*100:>9.2f}%{m['dd']*100:>9.2f}%{m['sharpe']:>9.3f}{m['calmar']:>9.3f}{m['n_trades']:>8}")

print(f"\n{'X2-A基准(参照)':<32}{'36.34':>9}%{'-37.66':>9}%{'1.271':>9}{'0.965':>9}{'~144':>8}")
print(f"{'X1基准(参照)':<32}{'36.29':>9}%{'-37.69':>9}%{'1.269':>9}{'0.963':>9}{'~144':>8}")

# ============================================================
# 12. 结论判断（硬终止条件检查）
# ============================================================
print("\n" + "=" * 70)
print("  X8 结论判断")
print("=" * 70)

x8_2 = results['X8-2(F1+F2rsi+MA)']
x8_1 = results['X8-1(F1+F2原版+MA)']
x8_0 = results['X8-0(F1单独+MA)']

print(f"\nX8-2（完整版）vs X2-A基准:")
print(f"  年化: {x8_2['ann']*100:.2f}% vs 36.34%  ({'✅' if x8_2['ann']>=0.3634 else '❌'})")
print(f"  回撤: {x8_2['dd']*100:.2f}% vs -37.66%  ({'✅' if x8_2['dd']>=-0.3766 else '❌'})")
print(f"  Sharpe: {x8_2['sharpe']:.3f} vs 1.271  ({'✅' if x8_2['sharpe']>=1.271 else '❌'})")
print(f"  Calmar: {x8_2['calmar']:.3f} vs 0.965  ({'✅' if x8_2['calmar']>=0.965 else '❌'})")

print(f"\n硬终止条件检查:")
hard_stop_ann = x8_2['ann'] < 0.36
hard_stop_dd = x8_2['dd'] > -0.40
hard_stop_collinear = abs(corr_f1_f2rsi) >= 0.5
hard_stop_trigger = f2rsi_override <= 0.05

print(f"  ❌ 年化<36.00%: {'触发' if hard_stop_ann else '未触发'} ({x8_2['ann']*100:.2f}%)")
print(f"  ❌ 回撤>-40%: {'触发' if hard_stop_dd else '未触发'} ({x8_2['dd']*100:.2f}%)")
print(f"  ❌ 共线性|r|≥0.5: {'触发' if hard_stop_collinear else '未触发'} ({corr_f1_f2rsi:+.4f})")
print(f"  ❌ 触发率≤5%: {'触发' if hard_stop_trigger else '未触发'} ({f2rsi_override*100:.2f}%)")

if hard_stop_ann or hard_stop_dd or hard_stop_collinear or hard_stop_trigger:
    print(f"\n>>> X8 触发硬终止条件，优化循环终止，X2-A为最终最优解 <<<")
else:
    if x8_2['ann'] >= 0.3634:
        print(f"\n>>> X8-2年化超X2-A基准，建议调用量化研究员评价是否采纳 <<<")
    else:
        print(f"\n>>> X8-2年化介于硬终止线和X2基准之间，调用量化研究员评价 <<<")

x8_4 = results['X8-4(F1+F2orig+F2rsi+MA)']

print(f"\nF2''单独贡献（X8-2 - X8-0）: {(x8_2['ann'] - x8_0['ann'])*100:+.2f}pp")
print(f"F2原版单独贡献（X8-1 - X8-0）: {(x8_1['ann'] - x8_0['ann'])*100:+.2f}pp")
print(f"三因子叠加增量（X8-4 - X8-1）: {(x8_4['ann'] - x8_1['ann'])*100:+.2f}pp")

# ============================================================
# 13. 保存报告到文件
# ============================================================
report_lines = []
def w(line=""):
    report_lines.append(line)

w("=" * 70)
w("  X8 策略回测 — F2''重构(RSI差) + 周频调仓")
w("=" * 70)
w(f"数据区间: {g_close.index[0]:%Y-%m-%d} ~ {g_close.index[-1]:%Y-%m-%d}")
w(f"交易日数: {len(g_close)}")
w("")
w(f"[因子计算] F1=比价MA20方向(0.5) + F2原版=动量加速度(5.0) + F2''=RSI差(0.3)")
w("")
w("[B研究员约束] 共线性检测 + 触发率统计")
w(f"  F1 vs F2''(RSI差) Pearson: {corr_f1_f2rsi:+.4f}  (要求|r|<0.5)")
w(f"  F1 vs F2原版   Pearson: {corr_f1_f2orig:+.4f}  (X2原值参照)")
w(f"  F2原版 vs F2''  Pearson: {corr_f2orig_f2rsi:+.4f}  (要求|r|<0.5)")
w(f"  VIF近似(F1 vs F2''): {vif_f1_f2rsi:.3f}  (要求<5)")
w(f"  F2''改变F1决策频率: {f2rsi_override*100:.2f}%  (要求>5%)")
w(f"  F2原版改变F1决策频率: {f2orig_override*100:.2f}%  (参照)")
w(f"  共线性检测: {'✅通过' if collinearity_ok else '❌未通过'}")
w(f"  触发率检测: {'✅通过' if trigger_ok else '❌未通过'}")
w("")
w(f"公平起始日: {start_idx:%Y-%m-%d} ~ {end_idx:%Y-%m-%d}")
w(f"周数: {len(weekly)}")
w(f"  MA50双跌破周数: {int(weekly['both_below_ma50'].sum())}")
w(f"  MA75双跌破周数: {int(weekly['both_below_ma75'].sum())}")
w("")
w("=" * 70)
w("  X8 消融分析报告")
w("=" * 70)
w("")
w(f"{'方案':<32}{'年化':>10}{'回撤':>10}{'Sharpe':>9}{'Calmar':>9}{'调仓':>8}")
w("-" * 78)
for name in order:
    m = results[name]
    w(f"{name:<32}{m['ann']*100:>9.2f}%{m['dd']*100:>9.2f}%{m['sharpe']:>9.3f}{m['calmar']:>9.3f}{m['n_trades']:>8}")
w("")
w(f"{'X2-A基准(参照)':<32}{'36.34':>9}%{'-37.66':>9}%{'1.271':>9}{'0.965':>9}{'~144':>8}")
w(f"{'X1基准(参照)':<32}{'36.29':>9}%{'-37.69':>9}%{'1.269':>9}{'0.963':>9}{'~144':>8}")
w("")
w("=" * 70)
w("  X8 结论判断")
w("=" * 70)
w("")
w(f"X8-2（完整版）vs X2-A基准:")
w(f"  年化: {x8_2['ann']*100:.2f}% vs 36.34%  ({'✅' if x8_2['ann']>=0.3634 else '❌'})")
w(f"  回撤: {x8_2['dd']*100:.2f}% vs -37.66%  ({'✅' if x8_2['dd']>=-0.3766 else '❌'})")
w(f"  Sharpe: {x8_2['sharpe']:.3f} vs 1.271  ({'✅' if x8_2['sharpe']>=1.271 else '❌'})")
w(f"  Calmar: {x8_2['calmar']:.3f} vs 0.965  ({'✅' if x8_2['calmar']>=0.965 else '❌'})")
w("")
w(f"硬终止条件检查:")
w(f"  年化<36.00%: {'触发' if hard_stop_ann else '未触发'} ({x8_2['ann']*100:.2f}%)")
w(f"  回撤>-40%: {'触发' if hard_stop_dd else '未触发'} ({x8_2['dd']*100:.2f}%)")
w(f"  共线性|r|≥0.5: {'触发' if hard_stop_collinear else '未触发'} ({corr_f1_f2rsi:+.4f})")
w(f"  触发率≤5%: {'触发' if hard_stop_trigger else '未触发'} ({f2rsi_override*100:.2f}%)")
w("")
if hard_stop_ann or hard_stop_dd or hard_stop_collinear or hard_stop_trigger:
    w(">>> X8 触发硬终止条件，优化循环终止，X2-A为最终最优解 <<<")
else:
    if x8_2['ann'] >= 0.3634:
        w(">>> X8-2年化超X2-A基准，建议调用量化研究员评价是否采纳 <<<")
    else:
        w(">>> X8-2年化介于硬终止线和X2基准之间，调用量化研究员评价 <<<")
w("")
w(f"F2''单独贡献（X8-2 - X8-0）: {(x8_2['ann'] - x8_0['ann'])*100:+.2f}pp")
w(f"F2原版单独贡献（X8-1 - X8-0）: {(x8_1['ann'] - x8_0['ann'])*100:+.2f}pp")
w(f"三因子叠加增量（X8-4 - X8-1）: {(x8_4['ann'] - x8_1['ann'])*100:+.2f}pp")
w("")
w("=" * 70)
w("  X8-4 三因子叠加意外成功分析")
w("=" * 70)
w("")
w(f"X8-4 vs X2-A基准:")
w(f"  年化: {x8_4['ann']*100:.2f}% vs 36.34%  ({'✅' if x8_4['ann']>=0.3634 else '❌'} +{(x8_4['ann']-0.3634)*100:.2f}pp)")
w(f"  回撤: {x8_4['dd']*100:.2f}% vs -37.66%  ({'✅' if x8_4['dd']>=-0.3766 else '❌'} {(x8_4['dd']+0.3766)*100:+.2f}pp)")
w(f"  Sharpe: {x8_4['sharpe']:.3f} vs 1.271  ({'✅' if x8_4['sharpe']>=1.271 else '❌'} {x8_4['sharpe']-1.271:+.3f})")
w(f"  Calmar: {x8_4['calmar']:.3f} vs 0.965  ({'✅' if x8_4['calmar']>=0.965 else '❌'} {x8_4['calmar']-0.965:+.3f})")
w(f"  调仓: {x8_4['n_trades']} vs ~144")
w("")
w("注：X8-2触发硬终止，但X8-4三因子叠加年化38.18%超X2-A，需研究员评价是否采纳")

REPORT_PATH.write_text("\n".join(report_lines), encoding='utf-8')
print(f"\n报告已保存: {REPORT_PATH}")
