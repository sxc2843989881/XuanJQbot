"""test_staged_drawdown.py — X2 Task3 分段降仓机制测试
================================================================
对比 4 种降仓方案（F1+F2因子与X1完全一致，仅改降仓逻辑）：

  X1基线 : MA75双跌破(close<MA75*0.97) → 10%仓位(direction='value')
           *复现 backtest_v72_engine.py 的行为*
  方案A  : MA50双跌破→50%, MA75双跌破→25%
           方向始终由 style_score 决定
  方案B  : MA75双跌破 → 0% (cash)
           方向始终由 style_score 决定（cash时方向无意义）
  方案C  : MA50双跌破→50%, MA75浅跌破(<0.97)→25%, MA75深跌破(<0.94)→0%
           方向始终由 style_score 决定

数据: c:\\temp_v72_data\\index_480080.csv (成长100)
      c:\\temp_v72_data\\index_480081.csv (价值100)
回测引擎: c:\\XuanJLH\\Qbot\\custom\\backtests\\backtest_engine.py
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
# 1. 公共参数（与X1完全一致）
# ============================================================
F1 = 0.5       # 比价MA20权重
F2 = 5.0       # 动量加速度权重
MA_PCT = 0.97       # MA浅跌破阈值
MA_DEEP_PCT = 0.94  # 方案C深跌破阈值
CUT_X1 = 0.1        # X1基线降仓后总仓位

DATA_DIR = Path(r'c:\temp_v72_data')

# ============================================================
# 2. 加载数据
# ============================================================
print("=" * 70)
print("  X2 Task3 分段降仓机制测试 — F1+F2因子（与X1一致）")
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
# 3. 公共因子计算（F1+F2，与X1完全一致）
# ============================================================
print("\n[因子计算] F1=比价MA20方向(0.5) + F2=动量加速度(5.0)")

# ---- 因子1：比价MA20方向 ----
ratio = (g_close / v_close).shift(1)
ratio_ma20 = ratio.rolling(20).mean()
ratio_dev = ratio / ratio_ma20 - 1
f1_signal = np.tanh(ratio_dev * 30) * F1
style_score = f1_signal.copy()

# ---- 因子2：动量加速度 ----
g_roc21 = g_close.pct_change(21).shift(1)
v_roc21 = v_close.pct_change(21).shift(1)
g_accel = g_roc21 - g_roc21.shift(10)
v_accel = v_roc21 - v_roc21.shift(10)
accel_diff = (g_accel - v_accel).clip(-0.02, 0.02)
f2_signal = accel_diff * F2
style_score = style_score + f2_signal

# 二元候选方向
candidate_g = style_score > 0

# ============================================================
# 4. 公共择时信号（MA50 + MA75浅 + MA75深）
# ============================================================
print("[择时计算] MA50双跌破 + MA75浅跌破 + MA75深跌破")

g_ma50 = g_close.shift(1).rolling(50).mean()
v_ma50 = v_close.shift(1).rolling(50).mean()
both_below_ma50 = (g_close.shift(1) < g_ma50 * MA_PCT) & (v_close.shift(1) < v_ma50 * MA_PCT)

g_ma75 = g_close.shift(1).rolling(75).mean()
v_ma75 = v_close.shift(1).rolling(75).mean()
both_below_ma75 = (g_close.shift(1) < g_ma75 * MA_PCT) & (v_close.shift(1) < v_ma75 * MA_PCT)

# 方案C深跌破
both_below_ma75_deep = (g_close.shift(1) < g_ma75 * MA_DEEP_PCT) & (v_close.shift(1) < v_ma75 * MA_DEEP_PCT)

# ============================================================
# 5. 周频采样（W-FRI，与X1一致）
# ============================================================
df = pd.DataFrame(index=g_close.index)
df['g_close'] = g_close
df['v_close'] = v_close
df['candidate_g'] = candidate_g
df['both_below_ma50'] = both_below_ma50
df['both_below_ma75'] = both_below_ma75
df['both_below_ma75_deep'] = both_below_ma75_deep
df['style_score'] = style_score

weekly = df.resample('W-FRI').last().dropna(subset=['candidate_g']).iloc[1:]
print(f"周数: {len(weekly)}")
print(f"  MA50双跌破周数: {int(weekly['both_below_ma50'].sum())}")
print(f"  MA75浅跌破周数: {int(weekly['both_below_ma75'].sum())}")
print(f"  MA75深跌破周数: {int(weekly['both_below_ma75_deep'].sum())}")


# ============================================================
# 6. 方案构建函数
# ============================================================

def build_x1_baseline(weekly_df):
    """X1基线 — 严格复现 backtest_v72_engine.py 的状态机
    - MA75双跌破 → position=0.1, signal='value', weight=0.1
    - 否则 → 二元全仓(growth/value), weight=1.0
    - 降仓时 current_pos 不更新（emerging后用旧方向）
    """
    position = pd.Series(np.nan, index=weekly_df.index)
    current_pos = None  # 1.0=growth, 0.0=value
    n_switches = 0

    for i in range(len(weekly_df)):
        row = weekly_df.iloc[i]
        if row['both_below_ma75']:
            position.iloc[i] = CUT_X1
            continue
        if current_pos is None:
            current_pos = 1.0 if row['candidate_g'] else 0.0
            position.iloc[i] = current_pos
            n_switches += 1
            continue
        target = 1.0 if row['candidate_g'] else 0.0
        if target == current_pos:
            position.iloc[i] = current_pos
        else:
            current_pos = target
            position.iloc[i] = current_pos
            n_switches += 1

    # 映射到 signal + weight
    signal = pd.Series(
        ['growth' if p == 1.0 else 'value' for p in position.values],
        index=position.index
    )
    weight = pd.Series(
        [CUT_X1 if p == CUT_X1 else 1.0 for p in position.values],
        index=position.index
    )
    return signal, weight, n_switches


def build_staged(weekly_df, scheme='A'):
    """分段降仓 — 方向始终由 style_score 决定
    scheme='A': MA50→50%, MA75→25%
    scheme='B': MA75→0% (cash)
    scheme='C': MA50→50%, MA75浅→25%, MA75深→0%
    """
    n = len(weekly_df)
    signal = pd.Series(['value'] * n, index=weekly_df.index, dtype=object)
    weight = pd.Series(1.0, index=weekly_df.index)

    for i in range(n):
        row = weekly_df.iloc[i]
        # 方向始终由 style_score 决定（不被降仓覆盖）
        signal.iloc[i] = 'growth' if row['candidate_g'] else 'value'

        if scheme == 'A':
            if row['both_below_ma75']:
                weight.iloc[i] = 0.25
            elif row['both_below_ma50']:
                weight.iloc[i] = 0.5
            else:
                weight.iloc[i] = 1.0
        elif scheme == 'B':
            if row['both_below_ma75']:
                weight.iloc[i] = 0.0
            else:
                weight.iloc[i] = 1.0
        elif scheme == 'C':
            if row['both_below_ma75_deep']:
                weight.iloc[i] = 0.0
            elif row['both_below_ma75']:
                weight.iloc[i] = 0.25
            elif row['both_below_ma50']:
                weight.iloc[i] = 0.5
            else:
                weight.iloc[i] = 1.0

    return signal, weight


def expand_to_daily(weekly_series, daily_index):
    """周频信号前向填充到日频（与V72引擎一致）"""
    s = weekly_series.copy()
    # 把每个周末映射到 <= 该周末的最后交易日
    idx_pos = daily_index.searchsorted(s.index, side='right') - 1
    valid = idx_pos >= 0
    s = s[valid]
    idx_pos = idx_pos[valid]
    s.index = daily_index[idx_pos]
    # 防止多个周末映射到同一日
    s = s[~s.index.duplicated(keep='last')]
    daily = pd.Series(np.nan, index=daily_index, dtype=s.dtype)
    daily.loc[s.index] = s
    return daily.ffill()


# ============================================================
# 7. 通用回测运行
# ============================================================

def run_backtest(daily_signal, daily_weight, g_close_s, v_close_s):
    """运行回测引擎，返回 BacktestResult"""
    # 对齐 & 删除起始NaN
    mask = ~(daily_signal.isna() | daily_weight.isna())
    daily_signal = daily_signal[mask].astype(str)
    daily_weight = daily_weight[mask].astype(float)
    g_align = g_close_s.loc[daily_signal.index]
    v_align = v_close_s.loc[daily_signal.index]

    n = len(daily_signal)
    dates = daily_signal.index.strftime('%Y-%m-%d').values
    signal = daily_signal.values.astype(str)
    position_weight = daily_weight.values.astype(np.float64)

    # 指数 open=close
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
        commission=0.0001,        # 万1佣金
        impact_slippage=0.0,      # 指数无冲击滑点
        apply_gap_slippage=False, # 指数无跳空（开=收）
    )

    return run_backtest_engine_weighted(bt_input, config, position_weight)


def calc_metrics_v72(result, freq=252, rf_annual=0.025):
    """V72 风格的指标计算 — 与 backtest_v72_engine.py 中 calc_metrics_r 完全一致
    Sharpe 使用 rf=2.5%/年（rf_p = 0.025/freq）
    """
    df_r = result.to_dataframe()
    r = df_r['daily_ret']
    eq = (1 + r).cumprod()
    n = len(r)
    years = n / freq
    total = eq.iloc[-1] - 1
    ann = (1 + total) ** (1 / years) - 1 if years > 0 else 0
    vol = r.std() * np.sqrt(freq)
    rf_p = rf_annual / freq
    sharpe = (r.mean() - rf_p) / r.std() * np.sqrt(freq) if r.std() > 0 else 0
    peak = eq.cummax()
    dd = (eq - peak) / peak
    max_dd = dd.min()
    calmar = ann / abs(max_dd) if max_dd < 0 else 0
    return {
        'ann': ann,
        'dd': max_dd,
        'sharpe': sharpe,
        'calmar': calmar,
        'n_trades': result.metrics['num_trades'],
    }


# ============================================================
# 8. 运行四种方案
# ============================================================
results = {}      # name -> (result, metrics_v72)
weekly_weights = {}  # name -> weekly weight series (for降仓统计)

# --- X1 基线 ---
print("\n[运行] X1基线 (MA75→10%, direction='value')")
wsig_x1, ww_x1, n_sw_x1 = build_x1_baseline(weekly)
dsig_x1 = expand_to_daily(wsig_x1, g_close.index)
dw_x1 = expand_to_daily(ww_x1, g_close.index)
result_x1 = run_backtest(dsig_x1, dw_x1, g_close, v_close)
m_x1 = calc_metrics_v72(result_x1)
results['X1基线(各持5%)'] = (result_x1, m_x1)
weekly_weights['X1基线(各持5%)'] = ww_x1
print(f"  年化={m_x1['ann']*100:6.2f}%  回撤={m_x1['dd']*100:6.2f}%  "
      f"Sharpe={m_x1['sharpe']:.3f}  Calmar={m_x1['calmar']:.3f}  "
      f"调仓={m_x1['n_trades']}")

# --- 方案A ---
print("\n[运行] 方案A (MA50→50%, MA75→25%)")
wsig_a, ww_a = build_staged(weekly, scheme='A')
dsig_a = expand_to_daily(wsig_a, g_close.index)
dw_a = expand_to_daily(ww_a, g_close.index)
result_a = run_backtest(dsig_a, dw_a, g_close, v_close)
m_a = calc_metrics_v72(result_a)
results['方案A(分段降仓)'] = (result_a, m_a)
weekly_weights['方案A(分段降仓)'] = ww_a
print(f"  年化={m_a['ann']*100:6.2f}%  回撤={m_a['dd']*100:6.2f}%  "
      f"Sharpe={m_a['sharpe']:.3f}  Calmar={m_a['calmar']:.3f}  "
      f"调仓={m_a['n_trades']}")

# --- 方案B ---
print("\n[运行] 方案B (MA75→0%)")
wsig_b, ww_b = build_staged(weekly, scheme='B')
dsig_b = expand_to_daily(wsig_b, g_close.index)
dw_b = expand_to_daily(ww_b, g_close.index)
result_b = run_backtest(dsig_b, dw_b, g_close, v_close)
m_b = calc_metrics_v72(result_b)
results['方案B(直接0%)'] = (result_b, m_b)
weekly_weights['方案B(直接0%)'] = ww_b
print(f"  年化={m_b['ann']*100:6.2f}%  回撤={m_b['dd']*100:6.2f}%  "
      f"Sharpe={m_b['sharpe']:.3f}  Calmar={m_b['calmar']:.3f}  "
      f"调仓={m_b['n_trades']}")

# --- 方案C ---
print("\n[运行] 方案C (MA50→50%, MA75浅→25%, MA75深→0%)")
wsig_c, ww_c = build_staged(weekly, scheme='C')
dsig_c = expand_to_daily(wsig_c, g_close.index)
dw_c = expand_to_daily(ww_c, g_close.index)
result_c = run_backtest(dsig_c, dw_c, g_close, v_close)
m_c = calc_metrics_v72(result_c)
results['方案C(三档降仓)'] = (result_c, m_c)
weekly_weights['方案C(三档降仓)'] = ww_c
print(f"  年化={m_c['ann']*100:6.2f}%  回撤={m_c['dd']*100:6.2f}%  "
      f"Sharpe={m_c['sharpe']:.3f}  Calmar={m_c['calmar']:.3f}  "
      f"调仓={m_c['n_trades']}")

# ============================================================
# 9. 对比表输出
# ============================================================
print("\n" + "=" * 70)
print("  降仓方案 A/B/C 测试结果对比")
print("=" * 70)
header = f"  {'方案':<18}{'年化':>10}{'回撤':>10}{'Sharpe':>9}{'Calmar':>9}{'调仓次数':>10}"
print(header)
print("  " + "-" * 60)
order = ['X1基线(各持5%)', '方案A(分段降仓)', '方案B(直接0%)', '方案C(三档降仓)']
for name in order:
    r, m = results[name]
    print(f"  {name:<18}{m['ann']*100:>9.2f}%{m['dd']*100:>9.2f}%"
          f"{m['sharpe']:>9.3f}{m['calmar']:>9.3f}{m['n_trades']:>10}")
print("=" * 70)

# ============================================================
# 10. 补充：各方案降仓周数统计
# ============================================================
print("\n[降仓统计]")
print(f"  {'方案':<18}{'正常周数':>10}{'50%周数':>10}{'25%周数':>10}{'0%周数':>10}")
print("  " + "-" * 60)
for name in order:
    ww = weekly_weights[name]
    n_100 = int((ww == 1.0).sum())
    n_50 = int((ww == 0.5).sum())
    n_25 = int((ww == 0.25).sum())
    n_10 = int((ww == CUT_X1).sum())
    n_0 = int((ww == 0.0).sum())
    # X1用10%列，其它方案用25%列
    if name.startswith('X1'):
        print(f"  {name:<18}{n_100:>10}{'-':>10}{'-':>10}{n_10:>10}")
    else:
        n_25_total = n_25 + n_10  # X1的10%归到25%列不合适，单独显示
        print(f"  {name:<18}{n_100:>10}{n_50:>10}{n_25:>10}{n_0:>10}")
print("=" * 70)

# ============================================================
# 11. 各方案详细 summary 输出
# ============================================================
for name in order:
    r, _ = results[name]
    print()
    print(r.summary(name))

print("\n[完成] 测试脚本执行结束")
