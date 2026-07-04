"""X14完整滑点对比: 冲击5bps + 跳空5bps = 总滑点10bps"""
import sys
sys.path.insert(0, r'c:\XuanJLH\Qbot\custom\backtests')
sys.path.insert(0, r'c:\XuanJLH\Qbot\custom\research\成长价值轮动\X12_偏离度空仓版')
sys.path.insert(0, r'c:\XuanJLH\Qbot\custom\research\成长价值轮动\X14_统一策略干净版')

from backtest_x14_engine import build_core
from optimize_runner import run_backtest, calc_metrics, count_switches

sig, wt = build_core(bias_mode='clear')

print("=" * 80)
print("  X14 滑点对比: 冲击5bps vs 冲击5bps+跳空5bps")
print("=" * 80)

# 1. 无滑点
r0 = run_backtest(sig, wt, impact_slippage=0.0)
m0 = calc_metrics(r0)

# 2. 只有冲击滑点5bps（现有基准）
r1 = run_backtest(sig, wt, impact_slippage=0.0005)
m1 = calc_metrics(r1)

# 3. 冲击5bps + 跳空5bps = 总滑点10bps
r2 = run_backtest(sig, wt, impact_slippage=0.0010)  # 10bps总滑点
m2 = calc_metrics(r2)

# 4. 再极端一点: 总滑点15bps
r3 = run_backtest(sig, wt, impact_slippage=0.0015)
m3 = calc_metrics(r3)

sw = count_switches(sig, wt)

print(f"\n  {'滑点设置':<24s} {'年化':>8s} {'回撤':>8s} {'Sharpe':>8s} {'Calmar':>8s} {'Calmar降幅':>10s}")
print("  " + "-" * 70)
print(f"  {'0 无滑点':<24s} {m0['ann']*100:>7.2f}% {m0['dd']*100:>7.2f}% {m0['sharpe']:>8.3f} {m0['calmar']:>8.3f} {'-':>10s}")
print(f"  {'5bps 冲击(现有)':<24s} {m1['ann']*100:>7.2f}% {m1['dd']*100:>7.2f}% {m1['sharpe']:>8.3f} {m1['calmar']:>8.3f} {(m1['calmar']-m0['calmar'])/m0['calmar']*100:>+9.2f}%")
print(f"  {'★ 10bps 冲击+跳空':<24s} {m2['ann']*100:>7.2f}% {m2['dd']*100:>7.2f}% {m2['sharpe']:>8.3f} {m2['calmar']:>8.3f} {(m2['calmar']-m0['calmar'])/m0['calmar']*100:>+9.2f}%")
print(f"  {'15bps 冲击+跳空':<24s} {m3['ann']*100:>7.2f}% {m3['dd']*100:>7.2f}% {m3['sharpe']:>8.3f} {m3['calmar']:>8.3f} {(m3['calmar']-m0['calmar'])/m0['calmar']*100:>+9.2f}%")

# 年化差异
print(f"\n  年化差异:")
print(f"    无滑点 → 5bps冲击: -{(m0['ann']-m1['ann'])*100:.2f}pp")
print(f"    5bps冲击 → 10bps总: -{(m1['ann']-m2['ann'])*100:.2f}pp")
print(f"    总滑点成本(0→10bps): -{(m0['ann']-m2['ann'])*100:.2f}pp")

print(f"\n  总交易次数: {sw['total']}次")
print(f"  方向切换: {sw['dir']}次")
print(f"  空仓切换: {sw['cash']}次")
print(f"  平均每笔滑点成本(10bps): {0.001 * 10000:.1f}元/万元")

# 原始3版本对比(对用户最直观)
print(f"\n{'='*80}")
print(f"  三版本最终对比(用户最关心的)")
print(f"{'='*80}")
print(f"  {'版本':<20s} {'年化':>8s} {'回撤':>8s} {'Calmar':>8s} {'滑点Calmar':>10s}")
print("  " + "-" * 58)

# 现有5bps滑点结果
result_sl = run_backtest(sig, wt, impact_slippage=0.0005)
m_sl = calc_metrics(result_sl)
print(f"  {'X14(5bps冲击)':<20s} {m1['ann']*100:>7.2f}% {m1['dd']*100:>7.2f}% {m1['calmar']:>8.3f} {m_sl['calmar']:>10.3f}")

# 10bps
r2_sl = run_backtest(sig, wt, impact_slippage=0.0010)
m2_sl = calc_metrics(r2_sl)
print(f"  {'X14(10bps总)':<20s} {m2['ann']*100:>7.2f}% {m2['dd']*100:>7.2f}% {m2['calmar']:>8.3f} {m2_sl['calmar']:>10.3f}")

print(f"\n  结论: 5bps冲击+5bps跳空(总10bps)后:")
print(f"    年化: {m1['ann']*100:.2f}% → {m2['ann']*100:.2f}% (降{(m1['ann']-m2['ann'])*100:.2f}pp)")
print(f"    Calmar: {m1['calmar']:.3f} → {m2['calmar']:.3f} (降{(m1['calmar']-m2['calmar']):.3f})")
