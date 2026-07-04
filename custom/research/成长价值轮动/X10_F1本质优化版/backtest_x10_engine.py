"""backtest_x10_engine.py — X10策略：F1本质优化版
================================================================
版本：X10（F1本质优化版）
路径：F1本质研究 → A1+B2收益增强 → E5持仓止损防回撤 → F3参数优化

设计哲学：
  F1单独日频 = 34.12%（核心alpha来源）
  F1的34%→36%差距是回撤控制不足导致的
  F1负责进攻，找不降收益的防回撤方法才是正确的路

三层逻辑（2个因子）：
  1. 收益增强层（F1+A1+B2）：
     - F1: 比价MA20方向信号
     - A1: MA20斜率>0.3% + 2天确认（过滤震荡市假信号）
     - B2: value方向要求价值指数20日动量>0（过滤value方向弱信号）
  2. 风险防御层（E5/F3）：
     - 持仓方向20日跌幅>10% → 降仓30%（趋势反转止损）

关键参数：
  F1 = 0.5（比价MA20方向强度）
  斜率阈值 = 0.003（0.3%）
  确认天数 = 2天
  value动量窗口 = 20日
  止损跌幅阈值 = 10%
  止损降仓比例 = 30%

回测结果（2013-02-07 ~ 2026-07-01，日频T+1）：
  年化 = 37.26%
  回撤 = -34.18%
  Sharpe = 1.238
  Calmar = 1.090（历史最高）

对比基准：
  X2-A:  36.34% / -37.66% / 1.271 / 0.965
  X9-2:  37.60% / -37.66% / 1.313 / 0.999
  X10:   37.26% / -34.18% / 1.238 / 1.090 ★

X10优势：
  - 年化+0.92pp vs X2-A
  - 回撤好3.48pp vs X2-A
  - Calmar高0.125 vs X2-A（首破1.0）
  - 逻辑更简洁：2个因子（F1增强 + 止损），无F2/F2''复杂变换
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
# 1. 加载数据
# ============================================================
DATA_DIR = Path(r'c:\temp_v72_data')

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
# 2. X10策略信号生成
# ============================================================
F1 = 0.5
SLOPE_THRESHOLD = 0.003
STOP_LOSS_THRESHOLD = 0.10
STOP_LOSS_WEIGHT = 0.30

# F1信号：比价MA20方向
ratio = g_close / v_close
ratio_ma20 = ratio.rolling(20).mean()
ratio_dev = ratio / ratio_ma20 - 1
f1_signal = np.tanh(ratio_dev * 30) * F1
base_dir = (f1_signal > 0).map({True: 'growth', False: 'value'})

# A1: MA20斜率>0.3% + 2天确认
ma20_slope = (ratio_ma20 - ratio_ma20.shift(5)) / ratio_ma20.shift(5)
slope_ok = ma20_slope.abs() > SLOPE_THRESHOLD
dir_confirmed = base_dir.where(base_dir == base_dir.shift(1), np.nan)
a1_dir = dir_confirmed.where(slope_ok, np.nan).ffill()

# B2: value方向要求价值指数20日动量>0
v_mom20 = v_close.pct_change(20)
wrong_value = (a1_dir == 'value') & (v_mom20 <= 0)
x10_dir = a1_dir.copy()
x10_dir[wrong_value] = 'growth'

# E5/F3: 持仓方向20日跌幅>10% → 降仓30%
g_dd20 = g_close / g_close.shift(20) - 1
v_dd20 = v_close / v_close.shift(20) - 1
growth_stop = (x10_dir == 'growth') & (g_dd20 < -STOP_LOSS_THRESHOLD)
value_stop = (x10_dir == 'value') & (v_dd20 < -STOP_LOSS_THRESHOLD)

x10_wt = pd.Series(1.0, index=x10_dir.index)
x10_wt[growth_stop | value_stop] = STOP_LOSS_WEIGHT

print(f"\n信号统计:")
print(f"  总交易日: {len(x10_dir)}")
print(f"  growth持仓天数: {(x10_dir == 'growth').sum()} ({(x10_dir == 'growth').mean()*100:.1f}%)")
print(f"  value持仓天数: {(x10_dir == 'value').sum()} ({(x10_dir == 'value').mean()*100:.1f}%)")
print(f"  止损降仓天数: {(x10_wt == STOP_LOSS_WEIGHT).sum()} ({(x10_wt == STOP_LOSS_WEIGHT).mean()*100:.1f}%)")

# ============================================================
# 3. 回测
# ============================================================
def run_daily(signal_series, weight_series):
    common_idx = signal_series.index.intersection(g_close.index)
    sig = signal_series.loc[common_idx]
    wt = weight_series.loc[common_idx]
    g_a = g_close.loc[common_idx]
    v_a = v_close.loc[common_idx]
    mask = ~(sig.isna() | wt.isna())
    sig = sig[mask].astype(str)
    wt = wt[mask].astype(float)
    g_a = g_a[mask]
    v_a = v_a[mask]
    bt_input = BacktestInput(
        dates=sig.index.strftime('%Y-%m-%d').values,
        value_open=v_a.values.astype(np.float64),
        value_close=v_a.values.astype(np.float64),
        growth_open=g_a.values.astype(np.float64),
        growth_close=g_a.values.astype(np.float64),
        signal=sig.values,
    )
    config = BacktestConfig(start_cash=1_000_000.0, commission=0.0001,
                            impact_slippage=0.0, apply_gap_slippage=False)
    return run_backtest_engine_weighted(bt_input, config, wt.values)

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
    return {'ann': ann, 'dd': max_dd, 'sharpe': sharpe, 'calmar': calmar,
            'n_trades': result.metrics['num_trades'], 'total': total,
            'final_nav': eq.iloc[-1] * 1_000_000}

print("\n[X10] F1本质优化版回测")
r_x10 = run_daily(x10_dir, x10_wt)
m_x10 = calc_metrics(r_x10)
print(f"  年化={m_x10['ann']*100:.2f}%  回撤={m_x10['dd']*100:.2f}%  "
      f"Sharpe={m_x10['sharpe']:.3f}  Calmar={m_x10['calmar']:.3f}")
print(f"  交易次数={m_x10['n_trades']}  最终净值={m_x10['final_nav']:,.0f}")

# ============================================================
# 4. 对比基准
# ============================================================
print("\n" + "=" * 70)
print("  X10 vs 基准对比")
print("=" * 70)
print(f"\n{'版本':<25} {'年化':>8} {'回撤':>8} {'Sharpe':>8} {'Calmar':>8} {'达标':>6}")
print("-" * 70)
print(f"{'F1纯日频':<25} {'34.47%':>7} {'-45.21%':>7} {'1.156':>8} {'0.762':>8} {'❌':>6}")
print(f"{'X2-A':<25} {'36.34%':>7} {'-37.66%':>7} {'1.271':>8} {'0.965':>8} {'✅':>6}")
print(f"{'X9-2':<25} {'37.60%':>7} {'-37.66%':>7} {'1.313':>8} {'0.999':>8} {'✅':>6}")
print(f"{'X10 ★':<25} {m_x10['ann']*100:>6.2f}% {m_x10['dd']*100:>6.2f}% "
      f"{m_x10['sharpe']:>8.3f} {m_x10['calmar']:>8.3f} {'✅':>6}")
print("-" * 70)

# 保存详细报告
report_lines = []
report_lines.append("=" * 70)
report_lines.append("  X10策略回测报告 — F1本质优化版")
report_lines.append("=" * 70)
report_lines.append("")
report_lines.append("【策略逻辑】")
report_lines.append("  收益增强层（F1+A1+B2）:")
report_lines.append("    F1: 比价MA20方向信号 (f1_signal = np.tanh(ratio_dev * 30) * 0.5)")
report_lines.append("    A1: MA20斜率>0.3% + 2天确认 (过滤震荡市假信号)")
report_lines.append("    B2: value方向要求价值指数20日动量>0 (过滤value方向弱信号)")
report_lines.append("  风险防御层 (E5/F3):")
report_lines.append("    持仓方向20日跌幅>10% → 降仓30% (趋势反转止损)")
report_lines.append("")
report_lines.append("【回测结果】")
report_lines.append(f"  年化收益率: {m_x10['ann']*100:.2f}%")
report_lines.append(f"  最大回撤:   {m_x10['dd']*100:.2f}%")
report_lines.append(f"  Sharpe:     {m_x10['sharpe']:.3f}")
report_lines.append(f"  Calmar:     {m_x10['calmar']:.3f}")
report_lines.append(f"  交易次数:   {m_x10['n_trades']}")
report_lines.append(f"  最终净值:   {m_x10['final_nav']:,.0f}")
report_lines.append("")
report_lines.append("【对比基准】")
report_lines.append(f"  {'版本':<25} {'年化':>8} {'回撤':>8} {'Sharpe':>8} {'Calmar':>8}")
report_lines.append(f"  {'-'*65}")
report_lines.append(f"  {'F1纯日频':<25} {'34.47%':>7} {'-45.21%':>7} {'1.156':>8} {'0.762':>8}")
report_lines.append(f"  {'X2-A':<25} {'36.34%':>7} {'-37.66%':>7} {'1.271':>8} {'0.965':>8}")
report_lines.append(f"  {'X9-2':<25} {'37.60%':>7} {'-37.66%':>7} {'1.313':>8} {'0.999':>8}")
report_lines.append(f"  {'X10 ★':<25} {m_x10['ann']*100:>6.2f}% {m_x10['dd']*100:>6.2f}% {m_x10['sharpe']:>8.3f} {m_x10['calmar']:>8.3f}")
report_lines.append("")
report_lines.append("【X10优势】")
report_lines.append(f"  vs X2-A: 年化+{(m_x10['ann']*100-36.34):.2f}pp, 回撤{(m_x10['dd']*100+37.66):+.2f}pp, Calmar+{(m_x10['calmar']-0.965):.3f}")
report_lines.append(f"  vs X9-2: 年化{(m_x10['ann']*100-37.60):+.2f}pp, 回撤{(m_x10['dd']*100+37.66):+.2f}pp, Calmar+{(m_x10['calmar']-0.999):.3f}")
report_lines.append("  逻辑更简洁: 2个因子(F1增强+止损), 无F2/F2''复杂变换")
report_lines.append("  符合'因子要少'原则")

report_path = Path(r'c:\temp_v72_data\x10_backtest_report.txt')
report_path.write_text("\n".join(report_lines), encoding='utf-8')
print(f"\n报告已保存: {report_path}")

# 保存X10为最终版本报告
final_report = Path(r'c:\XuanJLH\Qbot\custom\research\成长价值轮动\X10_F1本质优化版\X10_回测报告.txt')
final_report.write_text("\n".join(report_lines), encoding='utf-8')
print(f"最终报告已保存: {final_report}")
