"""validate_x11a.py — 使用通用检测流水线检测X11-A策略
================================================================
调用 strategy_validation_pipeline 对X11-A执行7大维度检测
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
print("  X11-A策略通用检测流水线")
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
# 2. 定义X11-A策略函数（接收参数返回signal, weight）
# ============================================================
# 预计算所有需要的数据
ratio = g_close / v_close
ratio_ma20 = ratio.rolling(20).mean()
ratio_dev = ratio / ratio_ma20 - 1
f1_signal = np.tanh(ratio_dev * 30) * 0.5
base_dir = (f1_signal > 0).map({True: 'growth', False: 'value'})

ma20_slope = (ratio_ma20 - ratio_ma20.shift(5)) / ratio_ma20.shift(5)
slope_ok = ma20_slope.abs() > 0.003
v_mom20 = v_close.pct_change(20)
g_dd20 = g_close / g_close.shift(20) - 1
v_dd20 = v_close / v_close.shift(20) - 1

def x11a_strategy(params: dict) -> tuple:
    """X11-A策略函数
    
    Args:
        params: {'n_confirm': 4, 'stop_threshold': 0.10, 'stop_weight': 0.3}
    
    Returns:
        (signal_series, weight_series)
    """
    n_confirm = params.get('n_confirm', 4)
    stop_threshold = params.get('stop_threshold', 0.10)
    stop_weight = params.get('stop_weight', 0.3)
    
    # n天确认
    dir_s = base_dir.copy()
    if n_confirm <= 1:
        confirmed = dir_s
    else:
        mask = np.ones(len(dir_s), dtype=bool)
        for k in range(1, n_confirm):
            mask = mask & (dir_s.values == dir_s.shift(k).values)
        confirmed = dir_s.where(mask, np.nan)
    
    dir_s = confirmed.where(slope_ok, np.nan).ffill()
    
    # B2: value方向要求价值指数20日动量>0
    wrong_value = (dir_s == 'value') & (v_mom20 <= 0)
    dir_s[wrong_value] = 'growth'
    
    # E5止损
    wt = pd.Series(1.0, index=dir_s.index)
    gs = (dir_s == 'growth') & (g_dd20 < -stop_threshold)
    vs = (dir_s == 'value') & (v_dd20 < -stop_threshold)
    wt[gs | vs] = stop_weight
    
    return dir_s, wt

# ============================================================
# 3. 定义回测引擎适配器
# ============================================================
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

# ============================================================
# 4. 运行检测
# ============================================================
config = ValidationConfig(
    benchmark_annual=0.3634,      # X2-A基准
    benchmark_sharpe=1.271,
    benchmark_dd=-0.3766,
    n_bootstrap=500,              # 减少Bootstrap次数加速
    n_walkforward=5,
    walkforward_train_years=3,
    walkforward_test_years=1,
    n_random_splits=10,
    strategy_type="trend_following",  # X11-A是趋势跟踪策略
)

param_grid = {
    'n_confirm': [2, 3, 4, 5, 6],  # 确认天数扫描
}

print("\n运行X11-A 7大维度检测...")
report = quick_validate(
    strategy_func=x11a_strategy,
    param_grid=param_grid,
    g_close=g_close,
    v_close=v_close,
    strategy_name="X11-A(4天确认)",
    config=config,
    backtest_engine_func=backtest_adapter,
    output_dir=str(OUTPUT_DIR),
)

# ============================================================
# 5. 打印报告
# ============================================================
print("\n" + report.print_report())

# ============================================================
# 6. 额外：分段参数最优vs总体最优对比
# ============================================================
print("\n" + "=" * 70)
print("  分段参数最优 vs 总体最优 对比")
print("=" * 70)

def calc_metrics_from_ret(ret, cfg):
    """从日收益序列计算指标"""
    r = ret.dropna()
    n = len(r)
    if n < 10:
        return {'ann': 0, 'dd': 0, 'sharpe': 0, 'calmar': 0}
    years = n / cfg.freq
    eq = (1 + r).cumprod()
    total = eq.iloc[-1] - 1
    ann = (1 + total) ** (1 / years) - 1 if years > 0 else 0
    rf_p = cfg.rf_annual / cfg.freq
    sharpe = (r.mean() - rf_p) / r.std() * np.sqrt(cfg.freq) if r.std() > 0 else 0
    peak = eq.cummax()
    dd = ((eq - peak) / peak).min()
    calmar = ann / abs(dd) if dd < 0 else 0
    return {'ann': ann, 'dd': dd, 'sharpe': sharpe, 'calmar': calmar}

# 总体最优：4天确认
sig_total, wt_total = x11a_strategy({'n_confirm': 4})
ret_total = backtest_adapter(sig_total, wt_total, g_close, v_close)
m_total = calc_metrics_from_ret(ret_total, config)

# 分段最优：每年找最优参数
years = sorted(set(ret_total.index.year))
print(f"\n{'年份':>6} {'2天':>10} {'3天':>10} {'4天':>10} {'5天':>10} {'6天':>10} {'最优':>10}")
print("-" * 70)

yearly_best = []
for year in years:
    year_mask = ret_total.index.year == year
    if year_mask.sum() < 100:
        continue
    
    best_ann = -999
    best_n = 0
    row_data = []
    for n in [2, 3, 4, 5, 6]:
        sig_n, wt_n = x11a_strategy({'n_confirm': n})
        ret_n = backtest_adapter(sig_n, wt_n, g_close, v_close)
        ret_year = ret_n.loc[ret_n.index.year == year]
        if len(ret_year) > 0:
            year_total = (1 + ret_year).prod() - 1
            row_data.append(f"{year_total*100:.1f}%")
            if year_total > best_ann:
                best_ann = year_total
                best_n = n
        else:
            row_data.append("N/A")
    
    yearly_best.append({'year': year, 'best_n': best_n, 'best_ret': best_ann})
    row_str = " ".join(f"{d:>10}" for d in row_data)
    print(f"{year:>6} {row_str} {best_n}天({best_ann*100:.1f}%)")

# 统计：总体最优(4天)在各年排名
print(f"\n总体最优: 4天确认")
print(f"分段最优分布:")
best_n_count = {}
for r in yearly_best:
    best_n_count[r['best_n']] = best_n_count.get(r['best_n'], 0) + 1
for n, c in sorted(best_n_count.items()):
    print(f"  {n}天: {c}年")

# 一致性：如果4天在各年都排名前2，说明稳健
print(f"\n一致性分析:")
consistent_count = 0
for year_info in yearly_best:
    year = year_info['year']
    # 计算各参数在该年的排名
    rets = {}
    for n in [2, 3, 4, 5, 6]:
        sig_n, wt_n = x11a_strategy({'n_confirm': n})
        ret_n = backtest_adapter(sig_n, wt_n, g_close, v_close)
        ret_year = ret_n.loc[ret_n.index.year == year]
        if len(ret_year) > 0:
            rets[n] = (1 + ret_year).prod() - 1
    if rets:
        sorted_n = sorted(rets.items(), key=lambda x: -x[1])
        rank_4 = [i for i, (n, _) in enumerate(sorted_n) if n == 4][0] + 1
        if rank_4 <= 2:
            consistent_count += 1
        print(f"  {year}: 4天排名第{rank_4}/{len(sorted_n)}")

print(f"\n4天确认在前2名的年份: {consistent_count}/{len(yearly_best)}")
if consistent_count >= len(yearly_best) * 0.6:
    print("✅ 参数选择稳健（多数年份排名前2）")
else:
    print("❌ 参数选择不稳定")

print("\n" + "=" * 70)
print("  X11-A检测全部完成！")
print(f"  报告: {OUTPUT_DIR}/X11-A(4天确认)_validation_report.txt")
print(f"  图表: {OUTPUT_DIR}/X11-A(4天确认)_validation_chart.png")
print("=" * 70)
