"""
Ma20_GV_Rotation 测试基线回测
-------------------------------
成长/价值比价均线轮动策略
使用通用 bt 引擎进行回测
"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, r'c:\XuanJLH\Qbot')
sys.path.insert(0, r'c:\XuanJLH\Qbot\engines\bt')
sys.path.insert(0, r'c:\XuanJLH\Qbot\custom\tests\Ma20_GV_Rotation')
sys.path.insert(0, r'c:\XuanJLH\Qbot\custom\research\成长价值轮动\X12_偏离度空仓版')

from engine import run_bt_backtest, BTBacktestInput, calc_metrics
from optimize_runner import run_backtest as run_lw, calc_metrics as calc_lw_metrics, G_CLOSE, V_CLOSE
from strategy import build_core, set_ma_period

# ===== 数据准备 =====
G, V = G_CLOSE, V_CLOSE
common = G.index.intersection(V.index)
G, V = G[common], V[common]
idx = common

# ===== 信号生成 =====
sig, wt = build_core(bias_mode='clear', dcd=6, bias_t_constraint=False, e5_reset=True)
sig, wt = sig[idx], wt[idx]

# ===== 轻量化引擎（对照） =====
res_lw = run_lw(sig, wt, impact_slippage=0.0005)
m_lw = calc_lw_metrics(res_lw)

# ===== bt 引擎 =====
# engine.py 内部自动做 shift(1) + 首日填充
# 传入原始信号即可
price_df = pd.DataFrame({'growth': G.values, 'value': V.values}, index=idx)
weight_df = pd.DataFrame({
    'growth': np.where(sig == 'growth', wt, 0.0),
    'value': np.where(sig == 'value', wt, 0.0),
}, index=idx)

bt_input = BTBacktestInput(
    price_df=price_df,
    weight_df=weight_df,
    initial_capital=1_000_000,
)
res_bt = run_bt_backtest(
    bt_input,
    commission_rate=0.0001,
    impact_rate=0.0005,
    t1_delay=True,  # 与轻量化引擎保持一致的 T+1 延迟
)
m_bt = calc_metrics(res_bt)

# ===== 对比输出 =====
out_lines = []
def p(msg): print(msg); out_lines.append(msg)

p("=" * 80)
p("  X14 策略 — bt 事件驱动引擎 vs 轻量化引擎")
p("  (bt 引擎内部自动处理 T+1 延迟)")
p("=" * 80)
p(f"{'指标':>12} {'轻量化':>12} {'bt':>12} {'差异':>12}")
p("-" * 48)
lw_total = res_lw.nav[-1]/res_lw.nav[0]*100-100
p(f"{'总收益':>12} {lw_total:>11.2f}% {res_bt.total_return*100:>11.2f}% "
  f"{(res_bt.total_return-res_lw.nav[-1]/res_lw.nav[0]+1)*100:>+11.2f}%")
p(f"{'CAGR':>12} {m_lw['ann']*100:>11.2f}% {m_bt['ann']*100:>11.2f}% "
  f"{(m_bt['ann']-m_lw['ann'])*100:>+10.2f}%")
p(f"{'回撤':>12} {m_lw['dd']*100:>11.2f}% {m_bt['dd']*100:>11.2f}% "
  f"{(m_bt['dd']-m_lw['dd'])*100:>+10.2f}%")
p(f"{'Sharpe':>12} {m_lw['sharpe']:>11.3f} {m_bt['sharpe']:>11.3f} "
  f"{m_bt['sharpe']-m_lw['sharpe']:>+10.3f}")
p(f"{'Calmar':>12} {m_lw['calmar']:>11.3f} {m_bt['calmar']:>11.3f} "
  f"{m_bt['calmar']-m_lw['calmar']:>+10.3f}")
p(f"{'调仓次数':>12} {m_lw['n_trades']:>11} {m_bt['n_trades']:>11} "
  f"{m_bt['n_trades']-m_lw['n_trades']:>+11}")

p(f"\n调仓记录 (前10笔):")
p(f"{'执行日期':>12} {'信号日期':>12} {'动作':>18} {'权重':>8}")
for t in res_bt.trades[:10]:
    p(f"{t.trade_date:>12} {t.signal_date:>12} {t.action:>18} {t.weight:>8.2f}")
p(f"  ... 共 {len(res_bt.trades)} 笔调仓")

with open(r'c:\XuanJLH\Qbot\bt_engine\out.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(out_lines))
