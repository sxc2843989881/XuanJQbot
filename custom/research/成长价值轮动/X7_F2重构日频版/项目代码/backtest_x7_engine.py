"""backtest_x7_engine.py — X7 策略回测（F2重构 + 日频T+1调仓）
================================================================
X7 = F1原版 + F2'重构（比价rolling Sharpe） + 日频T+1调仓 + 切换阈值±0.05 + MA50/MA75择时

  F1因子：比价MA20方向 (f1=0.5)          —— 保留X2原版，用户确认有效
  F2'因子：比价rolling Sharpe (f2=0.3)   —— 重构，风险调整动量，与F1正交
  调仓：日频T+1 + 迟滞带±0.05            —— 周频→日频，防whipsaw
  择时：MA50双跌破→50%, MA75双跌破→25%   —— 保留X2-A

  F2'三重证据：
    1. mathandmarkets案例：rolling Sharpe门控ROI 343%→361%, Sharpe 0.96→1.06
    2. 标普500动量指数编制法：RiskAdjustedMomentum = 动量/波动率
    3. 本地知识库已收录公式（01_ETF轮动策略/02_动量因子计算方法.md 第五节）

  F2'与F1正交性：
    - F1看比价水平位置（vs均线高低）
    - F2'看比价趋势质量（收益/波动率）
    - 高位置≠高质量（假突破时F1>0但F2'<0）

消融分析6方案：
  X7-0: F1单独 + 周频（基线对照）
  X7-1: F1 + F2' + 周频（验证F2'单独贡献）
  X7-2: F1 + F2' + 日频 + 无阈值（验证日频whipsaw）
  X7-3: F1 + F2' + 日频 + 阈值0.05（★完整版）
  X7-4: X7-3去MA75（验证择时层贡献）
  X7-5: X7-3 + 0.3%单边成本（实盘压力测试）

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
F2_NEW = 0.3       # F2'比价rolling Sharpe权重（重构）
SWITCH_THRESHOLD = 0.05  # 日频切换阈值（迟滞带±0.05）
MA_PCT = 0.97      # MA跌破阈值
F2_LOOKBACK = 20   # F2'滚动窗口

DATA_DIR = Path(r'c:\temp_v72_data')
REPORT_PATH = DATA_DIR / 'x7_ablation_report.txt'

# ============================================================
# 2. 加载数据
# ============================================================
print("=" * 70)
print("  X7 策略回测 — F2重构(rolling Sharpe) + 日频T+1调仓")
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
# 3. 因子计算（F1保留 + F2'重构）
# ============================================================
print("\n[因子计算] F1=比价MA20方向(0.5) + F2'=比价rolling Sharpe(0.3)")

# ---- F1：比价MA20方向（保留X2）----
ratio = (g_close / v_close).shift(1)
ratio_ma20 = ratio.rolling(20).mean()
ratio_dev = ratio / ratio_ma20 - 1
f1_signal = np.tanh(ratio_dev * 30) * F1

# ---- F2'：比价rolling Sharpe（重构）----
ratio_ret = ratio.pct_change()
ratio_mom = ratio_ret.rolling(F2_LOOKBACK).sum()       # 20日比价累计收益
ratio_vol = ratio_ret.rolling(F2_LOOKBACK).std()       # 20日比价波动率
ratio_sharpe = ratio_mom / (ratio_vol * np.sqrt(F2_LOOKBACK))  # rolling Sharpe
f2_new_signal = np.tanh(ratio_sharpe * 1.0) * F2_NEW   # tanh平滑，权重0.3

# ---- style_score（F1+F2'）----
style_score_new = f1_signal + f2_new_signal
candidate_g_new = style_score_new > 0  # 无阈值版（用于X7-2）

# ---- 原F2（对照，用于X7-0）----
g_roc21 = g_close.pct_change(21).shift(1)
v_roc21 = v_close.pct_change(21).shift(1)
g_accel = g_roc21 - g_roc21.shift(10)
v_accel = v_roc21 - v_roc21.shift(10)
accel_diff = (g_accel - v_accel).clip(-0.02, 0.02)
f2_old_signal = accel_diff * 5.0

# ============================================================
# 4. 共线性检测 + 触发率统计（B研究员约束）
# ============================================================
print("\n[B研究员约束] 共线性检测 + 触发率统计")

valid = ~(f1_signal.isna() | f2_new_signal.isna())
f1_v = f1_signal[valid]
f2_v = f2_new_signal[valid]
style_v = style_score_new[valid]
f1_only_cand = (f1_v > 0)
full_cand = (style_v > 0)

# 共线性：F1 vs F2'相关系数
corr_f1_f2, _ = stats.pearsonr(f1_v.values, f2_v.values)
# VIF近似：1/(1-R²)，R²=corr²
vif_approx = 1 / (1 - corr_f1_f2**2) if abs(corr_f1_f2) < 0.999 else float('inf')

# 触发率：F2'改变style_score符号（相对F1单独）的比例
f2_override = (f1_only_cand != full_cand).mean()

print(f"  F1 vs F2' Pearson相关: {corr_f1_f2:+.4f}  (要求|r|<0.5)")
print(f"  VIF近似: {vif_approx:.3f}  (要求<5)")
print(f"  F2'改变F1决策频率: {f2_override*100:.2f}%  (要求>5%)")

collinearity_ok = abs(corr_f1_f2) < 0.5
trigger_ok = f2_override > 0.05
print(f"  共线性检测: {'✅通过' if collinearity_ok else '❌未通过'}")
print(f"  触发率检测: {'✅通过' if trigger_ok else '❌未通过'}")

# ============================================================
# 5. 择时信号（MA50 + MA75，日频计算）
# ============================================================
print("\n[择时计算] MA50双跌破 + MA75双跌破")

g_ma50 = g_close.shift(1).rolling(50).mean()
v_ma50 = v_close.shift(1).rolling(50).mean()
both_below_ma50 = (g_close.shift(1) < g_ma50 * MA_PCT) & (v_close.shift(1) < v_ma50 * MA_PCT)

g_ma75 = g_close.shift(1).rolling(75).mean()
v_ma75 = v_close.shift(1).rolling(75).mean()
both_below_ma75 = (g_close.shift(1) < g_ma75 * MA_PCT) & (v_close.shift(1) < v_ma75 * MA_PCT)

# ============================================================
# 6. 信号构建函数
# ============================================================

def build_daily_signal_with_hysteresis(style_score_series, threshold=SWITCH_THRESHOLD):
    """日频信号 + 迟滞带（防whipsaw）
    style_score > threshold → 'growth'
    style_score < -threshold → 'value'
    -threshold ≤ score ≤ threshold → 维持上次仓位（迟滞带）
    """
    n = len(style_score_series)
    signal = ['value'] * n  # 默认value
    current_style = None
    for i in range(n):
        score = style_score_series.iloc[i]
        if pd.isna(score):
            signal[i] = 'value' if current_style is None else current_style
            continue
        if score > threshold:
            current_style = 'growth'
        elif score < -threshold:
            current_style = 'value'
        # else: 维持current_style（迟滞带）
        if current_style is None:
            current_style = 'growth' if score > 0 else 'value'
        signal[i] = current_style
    return pd.Series(signal, index=style_score_series.index)


def build_daily_weight(both_below_ma50_s, both_below_ma75_s, use_ma=True):
    """日频仓位权重（MA50→50%, MA75→25%，否则100%）"""
    n = len(both_below_ma50_s)
    weight = pd.Series(1.0, index=both_below_ma50_s.index)
    for i in range(n):
        if use_ma:
            if both_below_ma75_s.iloc[i]:
                weight.iloc[i] = 0.25
            elif both_below_ma50_s.iloc[i]:
                weight.iloc[i] = 0.5
            else:
                weight.iloc[i] = 1.0
        else:
            weight.iloc[i] = 1.0
    return weight


def build_weekly_signal(style_score_series, threshold=0.0):
    """周频信号（resample W-FRI，无迟滞带，threshold=0即简单>0判断）"""
    df = pd.DataFrame({'score': style_score_series})
    weekly = df.resample('W-FRI').last().dropna(subset=['score']).iloc[1:]
    signal = (weekly['score'] > threshold).astype(bool)
    return signal, weekly.index


def expand_to_daily(weekly_series, daily_index):
    """周频信号前向填充到日频"""
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
# 7. 回测运行函数
# ============================================================

def run_backtest(daily_signal, daily_weight, g_close_s, v_close_s, commission=0.0001):
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
        commission=commission,
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
    sharpe = (r.mean() - rf_annual/freq) / r.std() * np.sqrt(freq) if r.std() > 0 else 0
    peak = eq.cummax()
    dd = ((eq - peak) / peak).min()
    calmar = ann / abs(dd) if dd < 0 else 0
    return {
        'ann': ann, 'dd': dd, 'sharpe': sharpe, 'calmar': calmar,
        'total': total, 'n_trades': result.metrics['num_trades'],
        'final_nav': result.metrics['final_nav'],
        'final_multiple': result.metrics['final_multiple'],
        'num_days': result.metrics['num_days'],
    }


# ============================================================
# 8. 对齐起始日（F2'需要20日热身，MA75需要75日热身）
# ============================================================
# F2'的ratio_vol需要20日，ratio_mom需要20日，ratio需要shift(1)
# MA75需要75日
# 公平起始日：max(F2'首个有效日, MA75首个有效日) + 1日缓冲
start_f2 = f2_new_signal.first_valid_index()
start_ma75 = both_below_ma75.first_valid_index()
start_idx = max(start_f2, start_ma75)
# 再往后推1周确保稳定
start_idx = start_idx + pd.Timedelta(days=7)
end_idx = g_close.index[-1]

print(f"\n公平起始日: {start_idx:%Y-%m-%d} ~ {end_idx:%Y-%m-%d}")
print(f"回测天数: {(g_close.index >= start_idx).sum()}")

# 截取对齐区间
g_close_bt = g_close.loc[start_idx:end_idx]
v_close_bt = v_close.loc[start_idx:end_idx]
f1_bt = f1_signal.loc[start_idx:end_idx]
f2_new_bt = f2_new_signal.loc[start_idx:end_idx]
f2_old_bt = f2_old_signal.loc[start_idx:end_idx]
style_new_bt = style_score_new.loc[start_idx:end_idx]
ma50_bt = both_below_ma50.loc[start_idx:end_idx]
ma75_bt = both_below_ma75.loc[start_idx:end_idx]

# ============================================================
# 9. 运行6方案消融分析
# ============================================================
results = {}

# --- X7-0: F1单独 + 周频（基线对照）---
print("\n[运行] X7-0: F1单独 + 周频")
weekly_cand_f1, weekly_idx = build_weekly_signal(f1_bt, threshold=0.0)
weekly_sig_f1 = pd.Series(['growth' if c else 'value' for c in weekly_cand_f1], index=weekly_idx)
weekly_w_full = pd.Series(1.0, index=weekly_idx)  # 先用全仓，下面再算MA
# 周频MA采样
df_ma = pd.DataFrame({'ma50': ma50_bt, 'ma75': ma75_bt})
weekly_ma = df_ma.resample('W-FRI').last().dropna().iloc[1:]
weekly_ma50 = weekly_ma['ma50'].astype(bool)
weekly_ma75 = weekly_ma['ma75'].astype(bool)
weekly_w_ma = pd.Series(1.0, index=weekly_idx)
for i in range(len(weekly_idx)):
    if weekly_ma75.iloc[i]:
        weekly_w_ma.iloc[i] = 0.25
    elif weekly_ma50.iloc[i]:
        weekly_w_ma.iloc[i] = 0.5
# 对齐
common_idx = weekly_sig_f1.index.intersection(weekly_w_ma.index)
weekly_sig_f1 = weekly_sig_f1.loc[common_idx]
weekly_w_ma = weekly_w_ma.loc[common_idx]
dsig_x0 = expand_to_daily(weekly_sig_f1, g_close_bt.index)
dw_x0 = expand_to_daily(weekly_w_ma, g_close_bt.index)
result_x0 = run_backtest(dsig_x0, dw_x0, g_close_bt, v_close_bt)
m_x0 = calc_metrics(result_x0)
results['X7-0(F1单独+周频)'] = m_x0
print(f"  年化={m_x0['ann']*100:6.2f}%  回撤={m_x0['dd']*100:6.2f}%  Sharpe={m_x0['sharpe']:.3f}  Calmar={m_x0['calmar']:.3f}  调仓={m_x0['n_trades']}")

# --- X7-1: F1 + F2' + 周频（验证F2'单独贡献）---
print("\n[运行] X7-1: F1 + F2' + 周频")
weekly_cand_full, weekly_idx2 = build_weekly_signal(style_new_bt, threshold=0.0)
weekly_sig_full = pd.Series(['growth' if c else 'value' for c in weekly_cand_full], index=weekly_idx2)
common_idx2 = weekly_sig_full.index.intersection(weekly_w_ma.index)
weekly_sig_full = weekly_sig_full.loc[common_idx2]
weekly_w_ma2 = weekly_w_ma.loc[common_idx2]
dsig_x1 = expand_to_daily(weekly_sig_full, g_close_bt.index)
dw_x1 = expand_to_daily(weekly_w_ma2, g_close_bt.index)
result_x1 = run_backtest(dsig_x1, dw_x1, g_close_bt, v_close_bt)
m_x1 = calc_metrics(result_x1)
results['X7-1(F1+F2new+周频)'] = m_x1
print(f"  年化={m_x1['ann']*100:6.2f}%  回撤={m_x1['dd']*100:6.2f}%  Sharpe={m_x1['sharpe']:.3f}  Calmar={m_x1['calmar']:.3f}  调仓={m_x1['n_trades']}")

# --- X7-2: F1 + F2' + 日频 + 无阈值（验证whipsaw）---
print("\n[运行] X7-2: F1 + F2' + 日频 + 无阈值")
dsig_x2 = pd.Series(['growth' if s > 0 else 'value' for s in style_new_bt], index=style_new_bt.index)
dw_x2 = build_daily_weight(ma50_bt, ma75_bt, use_ma=True)
result_x2 = run_backtest(dsig_x2, dw_x2, g_close_bt, v_close_bt)
m_x2 = calc_metrics(result_x2)
results['X7-2(F1+F2new+日频无阈值)'] = m_x2
print(f"  年化={m_x2['ann']*100:6.2f}%  回撤={m_x2['dd']*100:6.2f}%  Sharpe={m_x2['sharpe']:.3f}  Calmar={m_x2['calmar']:.3f}  调仓={m_x2['n_trades']}")

# --- X7-3: F1 + F2' + 日频 + 阈值0.05（★完整版）---
print("\n[运行] X7-3: F1 + F2' + 日频 + 阈值0.05（★完整版）")
dsig_x3 = build_daily_signal_with_hysteresis(style_new_bt, threshold=SWITCH_THRESHOLD)
dw_x3 = build_daily_weight(ma50_bt, ma75_bt, use_ma=True)
result_x3 = run_backtest(dsig_x3, dw_x3, g_close_bt, v_close_bt)
m_x3 = calc_metrics(result_x3)
results['X7-3(F1+F2new+日频阈值0.05)★'] = m_x3
print(f"  年化={m_x3['ann']*100:6.2f}%  回撤={m_x3['dd']*100:6.2f}%  Sharpe={m_x3['sharpe']:.3f}  Calmar={m_x3['calmar']:.3f}  调仓={m_x3['n_trades']}")

# --- X7-4: X7-3去MA75（验证择时层贡献）---
print("\n[运行] X7-4: X7-3去MA75（无择时层）")
dw_x4 = pd.Series(1.0, index=style_new_bt.index)  # 无MA降仓，全仓
result_x4 = run_backtest(dsig_x3, dw_x4, g_close_bt, v_close_bt)
m_x4 = calc_metrics(result_x4)
results['X7-4(X7-3去MA75)'] = m_x4
print(f"  年化={m_x4['ann']*100:6.2f}%  回撤={m_x4['dd']*100:6.2f}%  Sharpe={m_x4['sharpe']:.3f}  Calmar={m_x4['calmar']:.3f}  调仓={m_x4['n_trades']}")

# --- X7-5: X7-3 + 0.3%单边成本（实盘压力测试）---
print("\n[运行] X7-5: X7-3 + 0.3%单边成本（commission=0.006）")
result_x5 = run_backtest(dsig_x3, dw_x3, g_close_bt, v_close_bt, commission=0.006)
m_x5 = calc_metrics(result_x5)
results['X7-5(X7-3+0.3%成本)'] = m_x5
print(f"  年化={m_x5['ann']*100:6.2f}%  回撤={m_x5['dd']*100:6.2f}%  Sharpe={m_x5['sharpe']:.3f}  Calmar={m_x5['calmar']:.3f}  调仓={m_x5['n_trades']}")

# ============================================================
# 10. 输出对比报告
# ============================================================
print("\n" + "=" * 70)
print("  X7 消融分析报告")
print("=" * 70)

print(f"\n公平起始日: {start_idx:%Y-%m-%d} ~ {end_idx:%Y-%m-%d}")
print(f"\n共线性检测: F1 vs F2' Pearson = {corr_f1_f2:+.4f} ({'✅<0.5' if collinearity_ok else '❌≥0.5'})")
print(f"触发率检测: F2'改变F1决策 = {f2_override*100:.2f}% ({'✅>5%' if trigger_ok else '❌≤5%'})")

print(f"\n{'方案':<32} {'年化':>8} {'回撤':>8} {'Sharpe':>8} {'Calmar':>8} {'调仓':>6}")
print("-" * 80)
for name, m in results.items():
    print(f"{name:<32} {m['ann']*100:>+7.2f}% {m['dd']*100:>+7.2f}% {m['sharpe']:>8.3f} {m['calmar']:>8.3f} {m['n_trades']:>6}")

# X2基准对照（已知值）
print(f"\n{'X2基准(参照)':<32} {'36.34':>7}% {'-37.66':>7}% {'1.271':>8} {'0.965':>8} {'~144':>6}")
print(f"{'X1基准(参照)':<32} {'36.29':>7}% {'-37.69':>7}% {'1.269':>8} {'0.963':>8} {'~144':>6}")

# ============================================================
# 11. 结论判断
# ============================================================
print("\n" + "=" * 70)
print("  X7 结论判断")
print("=" * 70)

x7_3 = results['X7-3(F1+F2new+日频阈值0.05)★']
x7_0 = results['X7-0(F1单独+周频)']
x7_1 = results['X7-1(F1+F2new+周频)']
x7_2 = results['X7-2(F1+F2new+日频无阈值)']
x7_4 = results['X7-4(X7-3去MA75)']
x7_5 = results['X7-5(X7-3+0.3%成本)']

print(f"\nX7-3（完整版）vs X2基准:")
print(f"  年化: {x7_3['ann']*100:.2f}% vs 36.34% ({'✅' if x7_3['ann'] >= 0.3634 else '❌'}{'超X2' if x7_3['ann'] >= 0.3634 else '低于X2'})")
print(f"  回撤: {x7_3['dd']*100:.2f}% vs -37.66% ({'✅改善' if x7_3['dd'] > -0.3766 else '❌恶化'})")
print(f"  Sharpe: {x7_3['sharpe']:.3f} vs 1.271 ({'✅' if x7_3['sharpe'] >= 1.271 else '❌'})")
print(f"  Calmar: {x7_3['calmar']:.3f} vs 0.965 ({'✅' if x7_3['calmar'] >= 0.965 else '❌'})")

# 失败红线检查
print(f"\n失败红线检查:")
red_lines = []
if x7_3['ann'] < 0.3629:
    red_lines.append(f"❌ 年化{x7_3['ann']*100:.2f}% < 36.29%（X1基准）")
if abs(corr_f1_f2) > 0.7:
    red_lines.append(f"❌ F1与F2'相关{corr_f1_f2:.4f} > 0.7（共线性）")
if f2_override < 0.05:
    red_lines.append(f"❌ F2'触发率{f2_override*100:.2f}% < 5%（装饰性）")
if x7_3['dd'] < -0.42:
    red_lines.append(f"❌ 回撤{x7_3['dd']*100:.2f}% < -42%（超V72）")

if red_lines:
    print("  " + "\n  ".join(red_lines))
    print(f"\n>>> X7 触发失败红线，建议回退X2 <<<")
else:
    print("  ✅ 所有失败红线通过")

# F2'单独贡献
f2_contribution = x7_1['ann'] - x7_0['ann']
print(f"\nF2'单独贡献（X7-1 - X7-0）: {f2_contribution*100:+.2f}pp ({'✅正贡献' if f2_contribution > 0 else '❌负贡献'})")

# 日频vs周频
daily_vs_weekly = x7_3['ann'] - x7_1['ann']
print(f"日频vs周频（X7-3 - X7-1）: {daily_vs_weekly*100:+.2f}pp ({'✅日频增益' if daily_vs_weekly > 0 else '❌日频拖累'})")

# 阈值vs无阈值
threshold_vs_none = x7_3['sharpe'] - x7_2['sharpe']
print(f"阈值vs无阈值（X7-3 Sharpe - X7-2）: {threshold_vs_none:+.3f} ({'✅阈值改善Sharpe' if threshold_vs_none > 0 else '❌阈值无效'})")

# MA75贡献
ma75_contribution = x7_3['dd'] - x7_4['dd']
print(f"MA75择时贡献（X7-3回撤 - X7-4回撤）: {ma75_contribution*100:+.2f}pp ({'✅MA75改善回撤' if ma75_contribution > 0 else '❌MA75无效'})")

# 成本压力
cost_impact = x7_3['ann'] - x7_5['ann']
print(f"成本压力（X7-3 - X7-5）: {cost_impact*100:+.2f}pp ({'✅成本可控' if x7_5['ann'] >= 0.3629 else '❌成本压垮收益'})")

# 保存报告
with open(REPORT_PATH, 'w', encoding='utf-8') as f:
    import io
    import contextlib
    # 简化版：直接写关键结论
    f.write(f"X7 消融分析报告\n")
    f.write(f"公平起始日: {start_idx:%Y-%m-%d} ~ {end_idx:%Y-%m-%d}\n")
    f.write(f"F1 vs F2' 相关: {corr_f1_f2:+.4f}\n")
    f.write(f"F2'触发率: {f2_override*100:.2f}%\n\n")
    for name, m in results.items():
        f.write(f"{name}: 年化{m['ann']*100:.2f}% 回撤{m['dd']*100:.2f}% Sharpe{m['sharpe']:.3f} Calmar{m['calmar']:.3f} 调仓{m['n_trades']}\n")
    f.write(f"\nX2基准: 年化36.34% 回撤-37.66% Sharpe1.271 Calmar0.965\n")

print(f"\n报告已保存: {REPORT_PATH}")
print("=" * 70)
