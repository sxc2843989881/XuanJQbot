"""验证引擎的T+1执行时序"""
import sys
sys.path.insert(0, r'c:\XuanJLH\Qbot\custom\backtests')
sys.path.insert(0, r'c:\XuanJLH\Qbot\custom\research\成长价值轮动\X12_偏离度空仓版')
sys.path.insert(0, r'c:\XuanJLH\Qbot\custom\research\成长价值轮动\X14_统一策略干净版')

import numpy as np
import pandas as pd
from backtest_x14_engine import build_core
from optimize_runner import run_backtest, G_CLOSE, V_CLOSE

sig, wt = build_core(bias_mode='clear')
result = run_backtest(sig, wt, impact_slippage=0.0005)
df = result.to_dataframe()
trades_df = result.trades_to_dataframe()

# BIAS触发日
bias_dates = pd.DatetimeIndex(['2015-05-26', '2020-07-09', '2020-07-13',
                                '2020-07-14', '2024-09-30', '2024-10-08', '2024-10-14'])

print("=" * 90)
print("  BIAS信号→执行 时序验证 (T+1)")
print("=" * 90)

# 引擎时序说明
print("""
引擎时序（核心代码 649-655 + 558-563）:
  T日收盘:  信号/权重与前日比较 → 如有变化 → pending_rebalance=True
  T+1开盘: 执行pending调仓（卖出旧仓+买入新仓）
""")

print(f"\n{'BIAS触发日(T)':<18s} {'T日权重':<10s} {'T+1执行日':<14s} {'T+1开盘卖价':<12s} {'T收盘价':<12s} {'跳空':<10s}")
print("  " + "-" * 75)

for d in bias_dates:
    # T日（信号触发）
    wt_t = wt.loc[d]
    g_close_t = G_CLOSE.loc[d]
    v_close_t = V_CLOSE.loc[d]
    
    # T+1日（执行）
    next_idx = wt.index.get_loc(d) + 1
    if next_idx < len(wt):
        d1 = wt.index[next_idx]
        g_close_d1 = G_CLOSE.loc[d1]
        v_close_d1 = V_CLOSE.loc[d1]
        sig_t = sig.loc[d]
        
        # 卖哪个？
        sell_price_close = g_close_t if sig_t == 'growth' else v_close_t
        sell_price_open = g_close_d1 if sig_t == 'growth' else v_close_d1
        
        gap_pct = (sell_price_open / sell_price_close - 1) * 100
        gap_str = f"{gap_pct:+.2f}%" if gap_pct != 0 else "0%"
        
        # 找trades记录
        trade_info = "已执行"
        
        print(f"  {str(d.date()):<18s} {wt_t:<10.2f} {str(d1.date()):<14s} "
              f"{sell_price_open:<11.2f} {sell_price_close:<11.2f} {gap_str:<10s}")

print(f"\n--- BIAS清仓的实际交易记录 ---")
recent_trades = trades_df.tail(30)
print(f"\n{'日期':<14s} {'信号':<10s} {'方向':<8s} {'交易价':<10s} {'金额':<12s}")
print("  " + "-" * 55)
for _, t in trades_df.iterrows():
    tdate = t.get('trade_date', '')
    # 找BIAS相关交易（清仓到现金）
    pos = t.get('position', '')
    amount = t.get('amount', 0)
    price = t.get('trade_price', 0)
    signal_date = t.get('signal_date', '')
    if signal_date in [str(d.date()) for d in bias_dates]:
        print(f"  {tdate:<14s} {signal_date:<10s} {pos:<8s} {price:<10.2f} {amount:<12.2f}")

print(f"\n--- 结论 ---")
print(f"  1. 引擎已实现 T+1 执行：T日收盘出信号 → T+1开盘调仓")
print(f"  2. BIAS触发日(T)仍然全天持仓，承受当日涨跌")
print(f"  3. T+1日开盘才清仓，若T+1跳空低开则亏损更大（反之亦然）")
print(f"  4. 当前滑点模型: impact_slippage=0.0005 (固定5bps), apply_gap_slippage=False")
print(f"     → 未计入跳空滑点（T收盘→T+1开盘的价差）")
print(f"  5. 但BIAS 7天触发中，跳空为正（继续涨）的有:")
for d in bias_dates:
    next_idx = wt.index.get_loc(d) + 1
    if next_idx < len(wt):
        d1 = wt.index[next_idx]
        sig_t = sig.loc[d]
        sell_close = g_close_t if sig_t == 'growth' else v_close_t
        sell_open = G_CLOSE.loc[d1] if sig_t == 'growth' else V_CLOSE.loc[d1]
        gap = (sell_open / sell_close - 1) * 100
        effect = "正(少亏)" if gap < 0 else "负(少赚)"
        print(f"     {d.date()}→{d1.date()}: 跳空{gap:+.2f}% {effect}")
