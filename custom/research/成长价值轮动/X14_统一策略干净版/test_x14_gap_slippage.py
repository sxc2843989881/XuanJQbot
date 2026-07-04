"""X14加固定5bps跳空滑点"""
import sys
sys.path.insert(0, r'c:\XuanJLH\Qbot\custom\backtests')
sys.path.insert(0, r'c:\XuanJLH\Qbot\custom\research\成长价值轮动\X12_偏离度空仓版')
sys.path.insert(0, r'c:\XuanJLH\Qbot\custom\research\成长价值轮动\X14_统一策略干净版')

import numpy as np
import pandas as pd
from dataclasses import dataclass
from backtest_x14_engine import build_core

# 从optimize_runner导入需要的东西
from optimize_runner import (
    G_CLOSE, V_CLOSE, BASE_DIR, run_backtest, calc_metrics, count_switches
)


def run_backtest_with_gap(sig, wt, impact_slippage=0.0005, gap_slippage=0.0005):
    """回测+固定跳空滑点"""
    from backtest_engine import run_backtest_engine_weighted, BacktestInput, BacktestConfig
    
    # 取共同日期
    common_idx = sig.index.intersection(wt.index)
    sig = sig.loc[common_idx]
    wt = wt.loc[common_idx]
    
    g_a = G_CLOSE.reindex(common_idx).ffill()
    v_a = V_CLOSE.reindex(common_idx).ffill()
    
    bt_input = BacktestInput(
        dates=sig.index.strftime('%Y-%m-%d').values,
        value_open=v_a.values.astype(np.float64),
        value_close=v_a.values.astype(np.float64),
        growth_open=g_a.values.astype(np.float64),
        growth_close=g_a.values.astype(np.float64),
        signal=sig.values,
    )
    
    config = BacktestConfig(
        start_cash=1_000_000.0,
        commission=0.0001,
        impact_slippage=impact_slippage,
        apply_gap_slippage=False
    )
    
    result = run_backtest_engine_weighted(bt_input, config, wt.values)
    
    # 后处理：加固定跳空滑点（5bps × 交易金额）
    df = result.to_dataframe()
    trades_df = result.trades_to_dataframe()
    
    total_gap_cost = 0.0
    for _, t in trades_df.iterrows():
        amount = t.get('amount', 0)
        gap_penalty = amount * gap_slippage  # 固定5bps
        total_gap_cost += gap_penalty
    
    # 修改累收收益率：减去跳空成本
    gap_ratio = total_gap_cost / 1_000_000.0
    
    # 从总收益率扣减（近似）
    df['nav'] = df['nav'] * (1 - gap_ratio * (df.index / len(df)))
    
    # 重新计算metrics
    from optimize_runner import calc_metrics as cm
    daily_ret = df['nav'].pct_change().dropna()
    m = cm(df)
    
    return df, m, total_gap_cost


# 跑对比
print("=" * 70)
print("  X14 加入固定5bps跳空滑点")
print("=" * 70)

sig, wt = build_core(bias_mode='clear')

# 基准：无跳空滑点
result_base = run_backtest(sig, wt, impact_slippage=0.0005)
m_base = calc_metrics(result_base)
sw_base = count_switches(sig, wt)

# 加跳空滑点
df_gap, m_gap, total_gap = run_backtest_with_gap(sig, wt, impact_slippage=0.0005, gap_slippage=0.0005)

# 对比
print(f"\n  {'指标':<20s} {'无跳空滑点':>14s} {'+跳空5bps':>14s} {'差异':>10s}")
print("  " + "-" * 60)
print(f"  {'年化收益率':<20s} {m_base['ann']*100:>12.2f}% {m_gap['ann']*100:>12.2f}% "
      f"{m_gap['ann']-m_base['ann']:>+10.2%}")
print(f"  {'最大回撤':<20s} {m_base['dd']*100:>12.2f}% {m_gap['dd']*100:>12.2f}% "
      f"{(m_gap['dd']-m_base['dd'])*100:>+9.2f}pp")
print(f"  {'年化波动率':<20s} {m_base['vol']*100:>12.2f}% {m_gap['vol']*100:>12.2f}%")
print(f"  {'Sharpe':<20s} {m_base['sharpe']:>14.3f} {m_gap['sharpe']:>14.3f}")
print(f"  {'Calmar':<20s} {m_base['calmar']:>14.3f} {m_gap['calmar']:>14.3f}")
print(f"  {'总交易次数':<20s} {sw_base['total']:>14d}")
print(f"  {'总跳空成本':<20s} {'-':>14s} {total_gap:>12,.0f}元")

# 滑点对比
result_sl = run_backtest(sig, wt, impact_slippage=0.0005)
m_sl = calc_metrics(result_sl)

gross_gap = total_gap / 1_000_000 * 100
print(f"\n  跳空成本占总资产比例: {gross_gap:.4f}%")
print(f"  单次交易平均跳空成本: {total_gap/sw_base['total']:,.0f}元")

# 实际滑点总成本(冲击+跳空)
total_slippage = m_base['ann'] - m_gap['ann']
print(f"\n  冲击滑点(5bps)已在前置计算")
print(f"  跳空滑点(5bps/笔)扣除后:")
print(f"    年化: {m_base['ann']*100:.2f}% → {m_gap['ann']*100:.2f}% (降{abs(total_slippage)*100:.2f}pp)")
print(f"    Calmar: {m_base['calmar']:.3f} → {m_gap['calmar']:.3f}")
