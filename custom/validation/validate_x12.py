"""validate_x12.py — 使用通用检测流水线检测X12策略
================================================================
调用 strategy_validation_pipeline 对X12执行7大维度检测
X12 = X11-A + F0偏离度空仓过滤(0.5%阈值)
================================================================
"""
import sys
sys.path.insert(0, r'c:\XuanJLH\Qbot\custom\backtests')
sys.path.insert(0, r'c:\XuanJLH\Qbot\custom\validation')

from pathlib import Path
import numpy as np
import pandas as pd
from backtest_engine import (
    BacktestInput, BacktestConfig,
    run_backtest_engine_weighted,
)
from strategy_validation_pipeline import (
    StrategyValidator, ValidationConfig, quick_validate,
)

DATA_DIR = Path(r'c:\temp_v72_data')
OUTPUT_DIR = Path(r'c:\temp_v72_data\validation')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 1. 加载数据
# ============================================================
print("=" * 70)
print("  X12策略通用检测流水线")
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

# ============================================================
# 2. 预计算因子
# ============================================================
ratio = g_close / v_close
ratio_ma20 = ratio.rolling(20).mean()
ratio_dev = ratio / ratio_ma20 - 1
ma20_slope = (ratio_ma20 - ratio_ma20.shift(5)) / ratio_ma20.shift(5)
slope_ok = ma20_slope.abs() > 0.003
v_mom20 = v_close.pct_change(20)
g_dd20 = g_close / g_close.shift(20) - 1
v_dd20 = v_close / v_close.shift(20) - 1
base_dir = (ratio > ratio_ma20).map({True: 'growth', False: 'value'})

# ============================================================
# 3. X12策略函数
# ============================================================
def x12_strategy(params: dict) -> tuple:
    """X12策略函数

    6层结构:
      F1基础信号 → F0偏离度空仓 → A1四天确认 → 斜率确认 → B2动量过滤 → E5止损

    Args:
        params: {
            'dev_threshold': 0.005,  # F0偏离度空仓阈值
            'n_confirm': 4,          # A1确认天数
            'stop_threshold': 0.10,  # E5止损阈值
            'stop_weight': 0.30,     # E5止损后仓位
        }
    Returns:
        (signal_series, weight_series)
    """
    dev_threshold = params.get('dev_threshold', 0.005)
    n_confirm = params.get('n_confirm', 4)
    stop_threshold = params.get('stop_threshold', 0.10)
    stop_weight = params.get('stop_weight', 0.30)

    # 第1层 F1基础信号
    dir_s = base_dir.copy()

    # 第2层 F0空仓过滤: 偏离度<阈值空仓
    wt = pd.Series(1.0, index=dir_s.index)
    low_dev = ratio_dev.abs() < dev_threshold
    wt[low_dev] = 0.0

    # 第3层 A1确认: n天方向一致
    mask = np.ones(len(dir_s), dtype=bool)
    for k in range(1, n_confirm):
        mask = mask & (dir_s.values == dir_s.shift(k).values)
    confirmed = dir_s.where(mask, np.nan)

    # 第4层 斜率确认
    dir_s = confirmed.where(slope_ok, np.nan).ffill()

    # 第5层 B2过滤
    wrong_value = (dir_s == 'value') & (v_mom20 <= 0)
    dir_s[wrong_value] = 'growth'

    # 第6层 E5止损 (在F0权重基础上叠加)
    gs = (dir_s == 'growth') & (g_dd20 < -stop_threshold)
    vs = (dir_s == 'value') & (v_dd20 < -stop_threshold)
    wt[gs | vs] = wt[gs | vs] * stop_weight

    return dir_s, wt

# ============================================================
# 4. 回测引擎适配器
# ============================================================
def backtest_adapter(signal: pd.Series, weight: pd.Series,
                     g: pd.Series, v: pd.Series) -> pd.Series:
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

# ============================================================
# 5. 运行检测
# ============================================================
config = ValidationConfig(
    benchmark_annual=0.3996,      # X11-A基准(更新为X12的对比基准)
    benchmark_sharpe=1.307,
    benchmark_dd=-0.3471,
    n_bootstrap=500,
    n_walkforward=5,
    walkforward_train_years=3,
    walkforward_test_years=1,
    n_random_splits=10,
    strategy_type="trend_following",
)

# 参数网格: 偏离度阈值扫描
param_grid = {
    'dev_threshold': [0.000, 0.003, 0.005, 0.008, 0.010],
}

print("\n运行X12 7大维度检测...")
report = quick_validate(
    strategy_func=x12_strategy,
    param_grid=param_grid,
    g_close=g_close,
    v_close=v_close,
    strategy_name="X12(偏离度空仓0.5%)",
    config=config,
    backtest_engine_func=backtest_adapter,
    output_dir=str(OUTPUT_DIR),
)

# ============================================================
# 6. 打印报告
# ============================================================
print("\n" + report.print_report())

print("\n" + "=" * 70)
print("  X12检测全部完成！")
print(f"  报告: {OUTPUT_DIR}/X12(偏离度空仓0.5%)_validation_report.txt")
print(f"  图表: {OUTPUT_DIR}/X12(偏离度空仓0.5%)_validation_chart.png")
print("=" * 70)
