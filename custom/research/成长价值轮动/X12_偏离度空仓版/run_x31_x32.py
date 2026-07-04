"""run_x31_x32.py — 第19-20轮优化(参数敏感性+消融+流水线验证)
================================================================
X31: X23(z=1.5)参数敏感性扫描+消融实验
X32: X23(z=1.5)7大检测流水线验证

最终最优版: X23(z=1.5) 年化41.10% 回撤-26.66% Calmar1.542
================================================================
"""
import sys
sys.path.insert(0, r'c:\XuanJLH\Qbot\custom\backtests')
sys.path.insert(0, r'c:\XuanJLH\Qbot\custom\research\成长价值轮动\X12_偏离度空仓版')
sys.path.insert(0, r'c:\XuanJLH\Qbot\custom\validation')

from pathlib import Path
import numpy as np
import pandas as pd
from optimize_runner import (
    G_CLOSE, V_CLOSE, RATIO, RATIO_MA20, RATIO_DEV, MA20_SLOPE, SLOPE_OK,
    V_MOM20, G_DD20, V_DD20, BASE_DIR,
    run_backtest, calc_metrics, count_switches, test_strategy, print_result,
    build_x11a, build_x12,
)
from run_x27_x30 import build_x23

# ============================================================
# X31: 参数敏感性扫描
# ============================================================
print("=" * 78)
print("  第19轮 X31: X23(z=1.5)参数敏感性扫描")
print("=" * 78)

# 1. z阈值敏感性(已扫,这里复现关键点)
print("\n  [1/5] z阈值敏感性:")
print(f"  {'z阈值':>8} {'年化':>8} {'回撤':>8} {'Sharpe':>8} {'Calmar':>8} {'交易':>6}")
print(f"  {'-'*55}")
for z in [0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5]:
    info = test_strategy(f"z={z}", build_x23, {'z_thresh': z}, f"z={z}")
    print(f"  {z:>8} {info['ann']*100:>7.2f}% {info['dd']*100:>7.2f}% "
          f"{info['sharpe']:>8.3f} {info['calmar']:>8.3f} {info['n_trades']:>6}")

# 2. 斜率阈值敏感性
print("\n  [2/5] 斜率阈值敏感性(z=1.5):")
print(f"  {'斜率阈值':>10} {'年化':>8} {'回撤':>8} {'Sharpe':>8} {'Calmar':>8} {'交易':>6}")
print(f"  {'-'*58}")
for s in [0.001, 0.002, 0.003, 0.005, 0.008, 0.010]:
    info = test_strategy(f"slope={s}", build_x23,
                         {'z_thresh': 1.5, 'slope_thresh': s}, f"slope={s}")
    print(f"  {s:>10} {info['ann']*100:>7.2f}% {info['dd']*100:>7.2f}% "
          f"{info['sharpe']:>8.3f} {info['calmar']:>8.3f} {info['n_trades']:>6}")

# 3. 确认天数敏感性
print("\n  [3/5] 确认天数敏感性(z=1.5):")
print(f"  {'天数':>8} {'年化':>8} {'回撤':>8} {'Sharpe':>8} {'Calmar':>8} {'交易':>6}")
print(f"  {'-'*55}")
for n in [2, 3, 4, 5, 6]:
    info = test_strategy(f"n={n}", build_x23,
                         {'z_thresh': 1.5, 'n_confirm': n}, f"n={n}")
    print(f"  {n:>8} {info['ann']*100:>7.2f}% {info['dd']*100:>7.2f}% "
          f"{info['sharpe']:>8.3f} {info['calmar']:>8.3f} {info['n_trades']:>6}")

# 4. E5止损参数敏感性
print("\n  [4/5] E5止损参数敏感性(z=1.5):")
print(f"  {'阈值/降仓':>12} {'年化':>8} {'回撤':>8} {'Sharpe':>8} {'Calmar':>8} {'交易':>6}")
print(f"  {'-'*60}")
for thr, w in [(0.08, 0.20), (0.10, 0.20), (0.10, 0.30), (0.12, 0.20),
               (0.12, 0.30), (0.15, 0.30), (0.10, 0.40)]:
    info = test_strategy(f"thr={thr},w={w}", build_x23,
                         {'z_thresh': 1.5, 'stop_threshold': thr, 'stop_weight': w},
                         f"thr={thr},w={w}")
    print(f"  {f'{thr}/{w}':>12} {info['ann']*100:>7.2f}% {info['dd']*100:>7.2f}% "
          f"{info['sharpe']:>8.3f} {info['calmar']:>8.3f} {info['n_trades']:>6}")


# ============================================================
# 消融实验
# ============================================================
print("\n  [5/5] 消融实验(逐个去掉各机制):")
print(f"  {'配置':>30} {'年化':>8} {'回撤':>8} {'Calmar':>8} {'交易':>6}")
print(f"  {'-'*68}")

def build_x23_no_f0():
    """去掉F0双确认空仓"""
    dir_s = BASE_DIR.copy()
    mask = np.ones(len(dir_s), dtype=bool)
    for k in range(1, 4):
        mask = mask & (dir_s.values == dir_s.shift(k).values)
    confirmed = dir_s.where(mask, np.nan)
    wt = pd.Series(1.0, index=dir_s.index)
    # 无F0空仓
    dir_s = confirmed.ffill()
    wrong_value = (dir_s == 'value') & (V_MOM20 <= 0)
    dir_s[wrong_value] = 'growth'
    gs = (dir_s == 'growth') & (G_DD20 < -0.10)
    vs = (dir_s == 'value') & (V_DD20 < -0.10)
    wt[gs | vs] = 0.30
    return dir_s, wt

def build_x23_no_b2():
    """去掉B2过滤"""
    dir_s = BASE_DIR.copy()
    mask = np.ones(len(dir_s), dtype=bool)
    for k in range(1, 4):
        mask = mask & (dir_s.values == dir_s.shift(k).values)
    confirmed = dir_s.where(mask, np.nan)
    RATIO_DEV_STD20 = RATIO_DEV.rolling(20).std()
    RATIO_DEV_Z = RATIO_DEV / RATIO_DEV_STD20
    wt = pd.Series(1.0, index=dir_s.index)
    low_dev_z = RATIO_DEV_Z.abs() < 1.5
    low_slope = MA20_SLOPE.abs() < 0.003
    both_weak = low_dev_z & low_slope
    wt[both_weak] = 0.0
    dir_s = confirmed.ffill()
    # 无B2
    gs = (dir_s == 'growth') & (G_DD20 < -0.10)
    vs = (dir_s == 'value') & (V_DD20 < -0.10)
    wt[gs | vs] = wt[gs | vs] * 0.30
    return dir_s, wt

def build_x23_no_e5():
    """去掉E5止损"""
    dir_s = BASE_DIR.copy()
    mask = np.ones(len(dir_s), dtype=bool)
    for k in range(1, 4):
        mask = mask & (dir_s.values == dir_s.shift(k).values)
    confirmed = dir_s.where(mask, np.nan)
    RATIO_DEV_STD20 = RATIO_DEV.rolling(20).std()
    RATIO_DEV_Z = RATIO_DEV / RATIO_DEV_STD20
    wt = pd.Series(1.0, index=dir_s.index)
    low_dev_z = RATIO_DEV_Z.abs() < 1.5
    low_slope = MA20_SLOPE.abs() < 0.003
    both_weak = low_dev_z & low_slope
    wt[both_weak] = 0.0
    dir_s = confirmed.ffill()
    wrong_value = (dir_s == 'value') & (V_MOM20 <= 0)
    dir_s[wrong_value] = 'growth'
    # 无E5
    return dir_s, wt

# 完整版
info = test_strategy("X23完整版", build_x23, {'z_thresh': 1.5}, "完整")
print(f"  {'X23完整版':>30} {info['ann']*100:>7.2f}% {info['dd']*100:>7.2f}% "
      f"{info['calmar']:>8.3f} {info['n_trades']:>6}")

# 去掉各机制
for name, func, desc in [
    ('去掉F0双确认', build_x23_no_f0, '无F0'),
    ('去掉B2过滤', build_x23_no_b2, '无B2'),
    ('去掉E5止损', build_x23_no_e5, '无E5'),
]:
    info = test_strategy(name, func, desc=desc)
    print(f"  {name:>30} {info['ann']*100:>7.2f}% {info['dd']*100:>7.2f}% "
          f"{info['calmar']:>8.3f} {info['n_trades']:>6}")


# ============================================================
# X32: 7大检测流水线验证
# ============================================================
print("\n" + "=" * 78)
print("  第20轮 X32: X23(z=1.5)流水线验证")
print("=" * 78)

from backtest_engine import BacktestInput, BacktestConfig, run_backtest_engine_weighted
from strategy_validation_pipeline import StrategyValidator, ValidationConfig, quick_validate

def x23_strategy(params: dict) -> tuple:
    """X23策略函数(适配流水线)"""
    z_thresh = params.get('z_thresh', 1.5)
    slope_thresh = params.get('slope_thresh', 0.003)
    n_confirm = params.get('n_confirm', 4)
    stop_threshold = params.get('stop_threshold', 0.10)
    stop_weight = params.get('stop_weight', 0.30)

    dir_s = BASE_DIR.copy()
    mask = np.ones(len(dir_s), dtype=bool)
    for k in range(1, n_confirm):
        mask = mask & (dir_s.values == dir_s.shift(k).values)
    confirmed = dir_s.where(mask, np.nan)

    RATIO_DEV_STD20 = RATIO_DEV.rolling(20).std()
    RATIO_DEV_Z = RATIO_DEV / RATIO_DEV_STD20
    wt = pd.Series(1.0, index=dir_s.index)
    low_dev_z = RATIO_DEV_Z.abs() < z_thresh
    low_slope = MA20_SLOPE.abs() < slope_thresh
    both_weak = low_dev_z & low_slope
    wt[both_weak] = 0.0

    dir_s = confirmed.ffill()
    wrong_value = (dir_s == 'value') & (V_MOM20 <= 0)
    dir_s[wrong_value] = 'growth'
    gs = (dir_s == 'growth') & (G_DD20 < -stop_threshold)
    vs = (dir_s == 'value') & (V_DD20 < -stop_threshold)
    wt[gs | vs] = wt[gs | vs] * stop_weight
    return dir_s, wt


def backtest_adapter(signal: pd.Series, weight: pd.Series,
                     g: pd.Series, v: pd.Series) -> pd.Series:
    """适配通用流水线的回测函数"""
    common_idx = signal.index.intersection(g.index)
    sig = signal.loc[common_idx]
    wt = weight.loc[common_idx]
    g_a = g.loc[common_idx]
    v_a = v.loc[common_idx]
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
    result = run_backtest_engine_weighted(bt_input, config, wt.values)
    df = result.to_dataframe()
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    return df['daily_ret']


config = ValidationConfig(
    benchmark_annual=0.3634,      # X2-A基准
    benchmark_sharpe=1.271,
    benchmark_dd=-0.3766,
    n_bootstrap=500,
    n_walkforward=5,
    walkforward_train_years=3,
    walkforward_test_years=1,
    n_random_splits=10,
    strategy_type="trend_following",
)

param_grid = {
    'z_thresh': [0.8, 1.0, 1.2, 1.5, 1.8, 2.0],
}

OUTPUT_DIR = Path(r'c:\temp_v72_data\validation')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("\n运行X23 7大维度检测...")
report = quick_validate(
    strategy_func=x23_strategy,
    param_grid=param_grid,
    g_close=G_CLOSE,
    v_close=V_CLOSE,
    strategy_name="X23(z-score1.5)",
    config=config,
    backtest_engine_func=backtest_adapter,
    output_dir=str(OUTPUT_DIR),
)

print("\n" + report.print_report())

print("\n" + "=" * 78)
print("  X23-X32 全部优化完成!")
print("=" * 78)
print(f"  最终最优版: X23(z=1.5)")
print(f"    年化41.10% 回撤-26.66% Sharpe1.393 Calmar1.542 交易384次")
print(f"  vs X12(0.5%): 年化+0pp 回撤+1.74pp Calmar+0.092 交易-253次")
print("=" * 78)
