"""v4参数搜索：找最优组合"""
import sys
from pathlib import Path

HERE = Path(r'c:\XuanJLH\Qbot\custom\strategies\etf_rotation\1_research\v2_10etf')
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / 'validation'))
sys.path.insert(0, str(HERE.parent / 'v1_3etf'))

from data_generator_v2 import generate_simulation_data_v2
from factors_v4 import strategy_unified_v4
from benchmark_family import equal_weight_benchmark

print("生成5年模拟数据...")
close, ohlcv, _ = generate_simulation_data_v2(n_years=5, seed=42)

_, eq_m = equal_weight_benchmark(close)
print(f"等权基准: Sharpe={eq_m['sharpe']:.3f}")
print()

# 参数搜索
configs = [
    # (name, take_profit, hard_tp, mom_rank, mom_confirm, tp_confirm, weekly)
    ("v4.1 默认",          0.10, 0.25, 6, 3, 2, True),
    ("止盈15%+禁动量衰减",   0.15, 0.30, 99, 3, 3, True),  # mom_rank=99 等于禁用
    ("止盈20%+禁动量衰减",   0.20, 0.30, 99, 3, 3, True),
    ("止盈15%+月频补仓",    0.15, 0.30, 99, 3, 3, False),  # weekly=False 等于月频？不对
    ("止盈10%+禁动量衰减",   0.10, 0.25, 99, 3, 2, True),
    ("无止盈+禁动量衰减",    0.99, 0.99, 99, 3, 2, True),  # 等于禁用止盈
]

print(f"{'配置':<25} {'Sharpe':>7} {'年化':>7} {'回撤':>7} {'换手':>5} {'止盈':>5} {'止损':>5} {'vs基准':>8}")
print("-" * 80)

for name, tp, htp, mr, mc, tc, wk in configs:
    try:
        eq, m, sig = strategy_unified_v4(
            close, ohlcv,
            take_profit_threshold=tp,
            hard_take_profit=htp,
            momentum_rank_threshold=mr,
            momentum_decay_confirm_days=mc,
            take_profit_confirm_days=tc,
            rebalance_weekly=wk,
        )
        diff = m['sharpe'] - eq_m['sharpe']
        mark = "✓" if diff > 0 else " "
        print(f"{name:<25} {m['sharpe']:>7.3f} {m['annual_return']*100:>6.2f}% {m['max_drawdown']*100:>6.2f}% "
              f"{m.get('n_switches',0):>5} {m.get('n_take_profit',0):>5} {m.get('n_stop_loss',0):>5} {diff:>+7.3f} {mark}")
    except Exception as e:
        print(f"{name:<25} ERROR: {e}")
