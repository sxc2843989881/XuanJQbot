"""backtest_x2_engine.py — X2 策略集成回测（Task 5 / Task 7 归档版）
================================================================
X2 策略 = F1原版 + F2原版 + 新降仓逻辑（修复X1方向硬编码瑕疵）

  F1因子：比价MA20方向 (f1=0.5)   —— 与X1完全相同
  F2因子：动量加速度   (f2=5.0)   —— 与X1完全相同
  降仓逻辑：默认采用方案A（分段降仓），保留方案B/C代码用于对比
    X2-A (默认/正式版): F2原版 + 降仓方案A（分段降仓：MA50→50%, MA75→25%）+ style_score方向
    X2-B (对比):        F2原版 + 降仓方案B（直接0%：MA75触发即清仓）      + style_score方向
    X2-C (对比):        F2原版 + 降仓方案C（三档降仓：MA50→50%, MA75浅→25%, MA75深→0%）+ style_score方向

  X1基线（参照）: MA75双跌破 → 10%仓位（direction='value'，实现瑕疵）

  关键修复点：
    X1在降仓时 direction 硬编码为 'value'，是已知的实现瑕疵。
    X2 所有方案在降仓时 direction 由 style_score>0 决定（growth/value），
    即使权重降到 0.25/0.0，方向仍跟随因子信号。

  X2正式版选择（Task 7，2026-07-02）：
    量化研究员 2v1 评价结论：X2 正式版采用方案A（分段降仓），原因：
      - 方案B（直接0%）年化36.14% 低于 X1基准36.29%，违反"收益为先"原则
      - 方案A 年化36.34%（+0.05pp vs X1），回撤-37.66%（+0.03pp改善），
        Sharpe 1.271（+0.002），Calmar 0.965（+0.002），均小幅优于X1
      - 方案A 在"收益为先"原则下综合最优，符合Alpha派主张

数据:
  成长100: c:\\temp_v72_data\\index_480080.csv
  价值100: c:\\temp_v72_data\\index_480081.csv
  区间:    2012-12-31 ~ 2026-07-01

回测引擎: c:\\XuanJLH\\Qbot\\custom\\backtests\\backtest_engine.py
  run_backtest_engine_weighted(bt_input, config, position_weight)
  BacktestInput(dates, value_open, value_close, growth_open, growth_close, signal)
  BacktestConfig(start_cash=1e6, commission=1e-4, impact_slippage=0, apply_gap_slippage=False)

约束：
  - 指数 open=close
  - 佣金万1，无冲击滑点，无跳空滑点
  - 周频调仓（W-FRI）
  - 初始资金100万
  - Sharpe 用 rf=2.5%/年（与X1口径一致）
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
# 1. 参数（F1/F2与X1完全一致）
# ============================================================
F1 = 0.5       # 比价MA20权重
F2 = 5.0       # 动量加速度权重
MA_PCT = 0.97       # MA浅跌破阈值
MA_DEEP_PCT = 0.94  # 方案C深跌破阈值
CUT_X1 = 0.1        # X1基线降仓后总仓位（各持5%）

DATA_DIR = Path(r'c:\temp_v72_data')
REPORT_PATH = DATA_DIR / 'x2_integration_report.txt'

# ============================================================
# 2. 加载数据
# ============================================================
print("=" * 70)
print("  X2 策略集成回测 — Task 5")
print("  F1原版 + F2原版 + 新降仓逻辑（style_score方向）")
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
# 3. 公共因子计算（F1+F2，与X1完全一致，不优化因子）
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
    - 降仓时 direction 硬编码为 'value'（X1的实现瑕疵）
    - 降仓时 current_pos 不更新（恢复后用旧方向）
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
    # X1降仓时 direction 硬编码为 'value'（这是已知的实现瑕疵）
    signal = pd.Series(
        ['growth' if p == 1.0 else 'value' for p in position.values],
        index=position.index
    )
    weight = pd.Series(
        [CUT_X1 if p == CUT_X1 else 1.0 for p in position.values],
        index=position.index
    )
    return signal, weight, n_switches


def build_x2(weekly_df, scheme='A'):
    """X2方案 — F1+F2原版 + 新降仓逻辑（style_score决定方向）

    scheme='A': MA50→50%, MA75→25%
    scheme='B': MA75→0% (cash)
    scheme='C': MA50→50%, MA75浅→25%, MA75深→0%

    与X1基线的核心差异：
      1) 降仓时方向由 style_score>0 决定（不再硬编码 'value'）
      2) 方向始终跟随 style_score，即使降仓权重很低
      3) 无状态机延迟（X1降仓时冻结方向，X2始终跟随因子）
    """
    n = len(weekly_df)
    signal = pd.Series(['value'] * n, index=weekly_df.index, dtype=object)
    weight = pd.Series(1.0, index=weekly_df.index)

    for i in range(n):
        row = weekly_df.iloc[i]
        # 方向始终由 style_score 决定（不被降仓覆盖，修复X1的实现瑕疵）
        signal.iloc[i] = 'growth' if row['candidate_g'] else 'value'

        if scheme == 'A':
            # 分段降仓：MA50→50%, MA75→25%
            if row['both_below_ma75']:
                weight.iloc[i] = 0.25
            elif row['both_below_ma50']:
                weight.iloc[i] = 0.5
            else:
                weight.iloc[i] = 1.0
        elif scheme == 'B':
            # 直接0%：MA75触发即清仓
            if row['both_below_ma75']:
                weight.iloc[i] = 0.0
            else:
                weight.iloc[i] = 1.0
        elif scheme == 'C':
            # 三档降仓：MA50→50%, MA75浅→25%, MA75深→0%
            if row['both_below_ma75_deep']:
                weight.iloc[i] = 0.0
            elif row['both_below_ma75']:
                weight.iloc[i] = 0.25
            elif row['both_below_ma50']:
                weight.iloc[i] = 0.5
            else:
                weight.iloc[i] = 1.0
        else:
            raise ValueError(f"未知方案: {scheme}")

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
    wr = (r > 0).sum() / (r != 0).sum() if (r != 0).sum() > 0 else 0
    return {
        'ann': ann,
        'dd': max_dd,
        'sharpe': sharpe,
        'calmar': calmar,
        'wr': wr,
        'total': total,
        'n_trades': result.metrics['num_trades'],
        'final_nav': result.metrics['final_nav'],
        'final_multiple': result.metrics['final_multiple'],
        'num_days': result.metrics['num_days'],
    }


# ============================================================
# 8. 运行 X1基线 + X2-A/B/C
# ============================================================
results = {}        # name -> (result, metrics_v72)
weekly_weights = {}  # name -> weekly weight series (for降仓统计)

# --- X1 基线 ---
print("\n[运行] X1基线 (MA75→10%, direction='value' 硬编码)")
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

# --- X2-A ---
print("\n[运行] X2-A (F2原版 + 分段降仓：MA50→50%, MA75→25% + style_score方向)")
wsig_a, ww_a = build_x2(weekly, scheme='A')
dsig_a = expand_to_daily(wsig_a, g_close.index)
dw_a = expand_to_daily(ww_a, g_close.index)
result_a = run_backtest(dsig_a, dw_a, g_close, v_close)
m_a = calc_metrics_v72(result_a)
results['X2-A(分段降仓)'] = (result_a, m_a)
weekly_weights['X2-A(分段降仓)'] = ww_a
print(f"  年化={m_a['ann']*100:6.2f}%  回撤={m_a['dd']*100:6.2f}%  "
      f"Sharpe={m_a['sharpe']:.3f}  Calmar={m_a['calmar']:.3f}  "
      f"调仓={m_a['n_trades']}")

# --- X2-B ---
print("\n[运行] X2-B (F2原版 + 直接0%：MA75触发即清仓 + style_score方向)")
wsig_b, ww_b = build_x2(weekly, scheme='B')
dsig_b = expand_to_daily(wsig_b, g_close.index)
dw_b = expand_to_daily(ww_b, g_close.index)
result_b = run_backtest(dsig_b, dw_b, g_close, v_close)
m_b = calc_metrics_v72(result_b)
results['X2-B(直接0%)'] = (result_b, m_b)
weekly_weights['X2-B(直接0%)'] = ww_b
print(f"  年化={m_b['ann']*100:6.2f}%  回撤={m_b['dd']*100:6.2f}%  "
      f"Sharpe={m_b['sharpe']:.3f}  Calmar={m_b['calmar']:.3f}  "
      f"调仓={m_b['n_trades']}")

# --- X2-C ---
print("\n[运行] X2-C (F2原版 + 三档降仓：MA50→50%, MA75浅→25%, MA75深→0% + style_score方向)")
wsig_c, ww_c = build_x2(weekly, scheme='C')
dsig_c = expand_to_daily(wsig_c, g_close.index)
dw_c = expand_to_daily(ww_c, g_close.index)
result_c = run_backtest(dsig_c, dw_c, g_close, v_close)
m_c = calc_metrics_v72(result_c)
results['X2-C(三档降仓)'] = (result_c, m_c)
weekly_weights['X2-C(三档降仓)'] = ww_c
print(f"  年化={m_c['ann']*100:6.2f}%  回撤={m_c['dd']*100:6.2f}%  "
      f"Sharpe={m_c['sharpe']:.3f}  Calmar={m_c['calmar']:.3f}  "
      f"调仓={m_c['n_trades']}")

# ============================================================
# 9. 主对比表输出
# ============================================================
print("\n" + "=" * 70)
print("  X2 策略方案对比（vs X1基准）")
print("=" * 70)
header = f"  {'方案':<22}{'年化':>10}{'回撤':>10}{'Sharpe':>9}{'Calmar':>9}{'调仓次数':>10}"
print(header)
print("  " + "-" * 64)
order = ['X1基线(各持5%)', 'X2-A(分段降仓)', 'X2-B(直接0%)', 'X2-C(三档降仓)']
for name in order:
    r, m = results[name]
    print(f"  {name:<22}{m['ann']*100:>9.2f}%{m['dd']*100:>9.2f}%"
          f"{m['sharpe']:>9.3f}{m['calmar']:>9.3f}{m['n_trades']:>10}")
print("=" * 70)

# ============================================================
# 10. 选择 X2 正式版方案（Task 7：方案A 分段降仓为默认）
# ============================================================
# 选择标准（Task 7，2026-07-02 调整）：
#   量化研究员 2v1 评价结论：以"收益为先"为最高原则
#     - 方案B（直接0%）年化36.14% 低于X1基准36.29%（-0.15pp），违反收益为先
#     - 方案A 年化36.34%（+0.05pp vs X1），同时回撤/Sharpe/Calmar均小幅改善
#   因此 X2 正式版固定采用方案A（分段降仓），方案B/C仅保留作对比展示。
#   下方保留 pick_best_x2_by_calmar() 函数体（不再默认调用），便于后续若需
#   重新按风控优先切换方案时直接复用。

DEFAULT_X2_SCHEME = 'X2-A(分段降仓)'  # Task 7 固定默认值


def pick_best_x2_by_calmar(results_dict, x1_metrics):
    """[保留对比用] 按 Calmar 风控优先选择 X2 方案
    Task 5 旧规则（Task 7 起不再默认调用）：
      1) 排除年化显著低于X1基线（>1.5pp）的方案
      2) 在剩余方案中按 Calmar 排序（风控优先）
      3) 若 Calmar 差异 <2%，取回撤更小者
    注意：Task 7 已将正式版固定为方案A，此函数仅供对比/复用。
    """
    candidates = []
    x1_ann = x1_metrics['ann']
    for name in ['X2-A(分段降仓)', 'X2-B(直接0%)', 'X2-C(三档降仓)']:
        _, m = results_dict[name]
        ann_ok = m['ann'] >= x1_ann - 0.015  # 年化不低于X1超过1.5pp
        candidates.append((name, m, ann_ok))

    # 过滤掉年化显著劣化的方案
    valid = [c for c in candidates if c[2]]
    if not valid:
        # 全部劣化，仍按 Calmar 选最优
        valid = candidates

    # 按 Calmar 降序排序
    valid_sorted = sorted(valid, key=lambda x: x[1]['calmar'], reverse=True)
    best_name, best_m, _ = valid_sorted[0]

    # 若与第二名 Calmar 差异 <2%，取回撤更小者
    if len(valid_sorted) >= 2:
        second_name, second_m, _ = valid_sorted[1]
        if abs(best_m['calmar'] - second_m['calmar']) / max(best_m['calmar'], 1e-9) < 0.02:
            if abs(second_m['dd']) < abs(best_m['dd']):
                best_name, best_m = second_name, second_m

    return best_name, best_m


# Task 7：X2正式版固定采用方案A（分段降仓），不再按Calmar自动选择
best_name = DEFAULT_X2_SCHEME
best_m = results[best_name][1]
print(f"\n[选择] X2 正式版方案: {best_name}（Task 7 默认：收益为先，方案A）")
print(f"  [对比] 旧Calmar自动选择结果: {pick_best_x2_by_calmar(results, m_x1)[0]}（仅供对比）")

# 改进幅度对比
x1 = m_x1
xb = best_m
d_ann = (xb['ann'] - x1['ann']) * 100
d_dd = (xb['dd'] - x1['dd']) * 100  # 回撤为负，差值正=回撤变大(恶化)
d_sharpe = xb['sharpe'] - x1['sharpe']
d_calmar = xb['calmar'] - x1['calmar']
d_trades = xb['n_trades'] - x1['n_trades']

print(f"\n  X2正式版 vs X1基线 改进幅度:")
print(f"    年化:    {x1['ann']*100:6.2f}% → {xb['ann']*100:6.2f}%  ({d_ann:+.2f}pp)")
print(f"    回撤:    {x1['dd']*100:6.2f}% → {xb['dd']*100:6.2f}%  ({d_dd:+.2f}pp, 正=回撤变小/改善)")
print(f"    Sharpe:  {x1['sharpe']:6.3f} → {xb['sharpe']:6.3f}  ({d_sharpe:+.3f})")
print(f"    Calmar:  {x1['calmar']:6.3f} → {xb['calmar']:6.3f}  ({d_calmar:+.3f})")
print(f"    调仓:    {x1['n_trades']:6d} → {xb['n_trades']:6d}  ({d_trades:+d})")

# ============================================================
# 11. 降仓周数统计
# ============================================================
print("\n[降仓统计]")
print(f"  {'方案':<22}{'100%周数':>10}{'50%周数':>10}{'25%周数':>10}{'10%周数':>10}{'0%周数':>10}")
print("  " + "-" * 70)
for name in order:
    ww = weekly_weights[name]
    n_100 = int((ww == 1.0).sum())
    n_50 = int((ww == 0.5).sum())
    n_25 = int((ww == 0.25).sum())
    n_10 = int((ww == CUT_X1).sum())
    n_0 = int((ww == 0.0).sum())
    print(f"  {name:<22}{n_100:>10}{n_50:>10}{n_25:>10}{n_10:>10}{n_0:>10}")
print("=" * 70)

# ============================================================
# 12. X2 正式版完整性能指标输出
# ============================================================
print("\n" + "=" * 70)
print(f"  X2 正式版 — {best_name} 完整性能指标")
print("=" * 70)
print(results[best_name][0].summary(f"X2正式版 ({best_name})"))

print(f"\n  V72 风格指标（Sharpe用rf=2.5%/年）:")
print(f"    年化收益:  {xb['ann']*100:8.2f}%")
print(f"    最大回撤:  {xb['dd']*100:8.2f}%")
print(f"    Sharpe:    {xb['sharpe']:8.3f}")
print(f"    Calmar:    {xb['calmar']:8.3f}")
print(f"    胜率:      {xb['wr']*100:8.2f}%")
print(f"    总收益:    {xb['total']*100:8.2f}%")
print(f"    最终净值:  {xb['final_nav']:>12,.0f}")
print(f"    倍率:      {xb['final_multiple']:8.2f}x")
print(f"    交易日数:  {xb['num_days']:8d}")
print(f"    调仓次数:  {xb['n_trades']:8d}")

# ============================================================
# 13. 所有方案详细 summary 输出
# ============================================================
print("\n" + "=" * 70)
print("  各方案详细 summary")
print("=" * 70)
for name in order:
    r, _ = results[name]
    print()
    print(r.summary(name))

# ============================================================
# 14. 保存报告到文件
# ============================================================
report_lines = []

def w(line=""):
    report_lines.append(line)


w("=" * 70)
w("  X2 策略集成回测报告（Task 5）")
w("  F1原版 + F2原版 + 新降仓逻辑（style_score方向）")
w("=" * 70)
w()
w(f"数据区间: {g_close.index[0]:%Y-%m-%d} ~ {g_close.index[-1]:%Y-%m-%d}")
w(f"交易日数: {len(g_close)}")
w(f"周频样本(W-FRI): {len(weekly)}")
w(f"  MA50双跌破周数: {int(weekly['both_below_ma50'].sum())}")
w(f"  MA75浅跌破周数: {int(weekly['both_below_ma75'].sum())}")
w(f"  MA75深跌破周数: {int(weekly['both_below_ma75_deep'].sum())}")
w()
w("------------------------------------------------------------------------------")
w("一、策略配置")
w("------------------------------------------------------------------------------")
w()
w("  F1因子: 比价MA20方向 (f1=0.5) — 与X1完全相同")
w("    ratio = (g_close/v_close).shift(1)")
w("    ratio_ma20 = ratio.rolling(20).mean()")
w("    ratio_dev = ratio / ratio_ma20 - 1")
w("    f1_signal = tanh(ratio_dev * 30) * 0.5")
w()
w("  F2因子: 动量加速度 (f2=5.0) — 与X1完全相同")
w("    g_roc21 = g_close.pct_change(21).shift(1); v_roc21 同理")
w("    g_accel = g_roc21 - g_roc21.shift(10); v_accel 同理")
w("    accel_diff = (g_accel - v_accel).clip(-0.02, 0.02)")
w("    f2_signal = accel_diff * 5.0")
w("    style_score = f1_signal + f2_signal")
w()
w("  降仓逻辑（X2三种方案 + X1基线）：")
w("    X1基线:  MA75双跌破(close<MA75*0.97) → 10%仓位(direction='value'硬编码)")
w("    X2-A:    MA50→50%, MA75→25%, 方向由 style_score 决定")
w("    X2-B:    MA75→0% (cash),     方向由 style_score 决定")
w("    X2-C:    MA50→50%, MA75浅→25%, MA75深→0%, 方向由 style_score 决定")
w()
w("  关键修复点：")
w("    X1在降仓时 direction 硬编码为 'value'，是已知的实现瑕疵。")
w("    X2 所有方案在降仓时 direction 由 style_score>0 决定（growth/value），")
w("    即使权重降到 0.25/0.0，方向仍跟随因子信号。")
w()
w("  回测参数：")
w("    初始资金: 1,000,000")
w("    佣金: 万1（双边）")
w("    冲击滑点: 0（指数无冲击）")
w("    跳空滑点: 关闭（指数 open=close）")
w("    调仓频率: 周频 W-FRI")
w("    Sharpe rf: 2.5%/年（与X1口径一致）")
w()
w("------------------------------------------------------------------------------")
w("二、方案对比表")
w("------------------------------------------------------------------------------")
w()
w("=" * 70)
w("  X2 策略方案对比（vs X1基准）")
w("=" * 70)
w(f"  {'方案':<22}{'年化':>10}{'回撤':>10}{'Sharpe':>9}{'Calmar':>9}{'调仓次数':>10}")
w("  " + "-" * 64)
for name in order:
    r, m = results[name]
    w(f"  {name:<22}{m['ann']*100:>9.2f}%{m['dd']*100:>9.2f}%"
      f"{m['sharpe']:>9.3f}{m['calmar']:>9.3f}{m['n_trades']:>10}")
w("=" * 70)
w()
w("------------------------------------------------------------------------------")
w("三、X2 正式版选择")
w("------------------------------------------------------------------------------")
w()
w(f"  选择标准（Task 7，2026-07-02 调整）：")
w(f"    量化研究员 2v1 评价结论：以'收益为先'为最高原则")
w(f"      1) 排除年化低于X1基线的方案（方案B 36.14% < X1 36.29%，违反收益为先）")
w(f"      2) 在年化不低于X1的方案中，选择综合指标最优者")
w(f"      3) 方案A 年化+0.05pp 同时回撤/Sharpe/Calmar均小幅改善，符合Alpha派主张")
w(f"    [对比] 旧规则（Task 5）：按 Calmar 风控优先 → 会选方案B，但已被否决")
w()
w(f"  >>> 选择 X2 正式版方案: {best_name}")
w()
w("  选择理由：")
# 从方案名中提取方案字母（A/B/C）
scheme_letter = None
for letter in ('A', 'B', 'C'):
    if f'X2-{letter}(' in best_name:
        scheme_letter = letter
        break
if scheme_letter == 'B':
    w("    - X2-B 在 MA75 双跌破时直接清仓（0%仓位），最大程度规避深度下跌")
    w("    - Calmar 风控指标最优（0.978），符合 Task 3-4 '风控最优' 结论")
    w("    - 调仓次数与 X1 一致（183次），交易摩擦成本无增加")
    w("    - 回撤从 -37.69% 改善至 -36.94%（减少 0.74pp）")
    w("    - 年化微降 0.16pp（36.29% → 36.14%），属于风控换收益的合理权衡")
elif scheme_letter == 'C':
    w("    - X2-C 三档降仓机制对市场状态响应更精细")
    w("    - 在 MA75 浅跌破阶段保留 25% 仓位避免踏空反弹，深跌破时清仓")
    w("    - Calmar 与回撤均优于 X1 基线")
elif scheme_letter == 'A':
    w("    - X2-A 分段降仓（MA50→50%, MA75→25%）保留部分仓位暴露")
    w("    - 在 MA50 跌破时即开始减仓，提前规避部分下跌")
    w("    - 年化 36.34% 高于 X1 基线 36.29%（+0.05pp），符合'收益为先'原则")
    w("    - 同时 Sharpe(1.271)、Calmar(0.965)、回撤(-37.66%) 均小幅优于 X1")
    w("    - 相比方案B：年化高 0.20pp（36.34% vs 36.14%），且不违反收益为先")
w("    - 所有 X2 方案均修复了 X1 降仓时方向硬编码为 'value' 的实现瑕疵")
w()
w("------------------------------------------------------------------------------")
w("四、X2 正式版完整性能指标")
w("------------------------------------------------------------------------------")
w()
w(f"  方案: {best_name}")
w()
w("  V72 风格指标（Sharpe用rf=2.5%/年）:")
w(f"    年化收益:  {xb['ann']*100:8.2f}%")
w(f"    最大回撤:  {xb['dd']*100:8.2f}%")
w(f"    Sharpe:    {xb['sharpe']:8.3f}")
w(f"    Calmar:    {xb['calmar']:8.3f}")
w(f"    胜率:      {xb['wr']*100:8.2f}%")
w(f"    总收益:    {xb['total']*100:8.2f}%")
w(f"    最终净值:  {xb['final_nav']:>12,.0f}")
w(f"    倍率:      {xb['final_multiple']:8.2f}x")
w(f"    交易日数:  {xb['num_days']:8d}")
w(f"    调仓次数:  {xb['n_trades']:8d}")
w()
w("  引擎原始指标:")
for k, v in results[best_name][0].metrics.items():
    if isinstance(v, float):
        w(f"    {k:<28}: {v:>14,.4f}")
    else:
        w(f"    {k:<28}: {v:>14}")
w()
w("------------------------------------------------------------------------------")
w("五、X2 正式版 vs X1基线 对比分析")
w("------------------------------------------------------------------------------")
w()
w(f"  {'指标':<14}{'X1基线':>14}{'X2正式版':>14}{'差异':>14}{'评价':>10}")
w("  " + "-" * 64)
def fmt_pct(x): return f"{x*100:.2f}%"
def fmt_num(x): return f"{x:.3f}"
diff_ann = (xb['ann'] - x1['ann']) * 100
diff_dd = (xb['dd'] - x1['dd']) * 100
diff_sharpe = xb['sharpe'] - x1['sharpe']
diff_calmar = xb['calmar'] - x1['calmar']
diff_trades = xb['n_trades'] - x1['n_trades']
w(f"  {'年化收益':<14}{x1['ann']*100:>13.2f}%{xb['ann']*100:>13.2f}%{diff_ann:>+13.2f}pp{'✓' if diff_ann>=-0.5 else '⚠':>10}")
w(f"  {'最大回撤':<14}{x1['dd']*100:>13.2f}%{xb['dd']*100:>13.2f}%{diff_dd:>+13.2f}pp{'✓' if diff_dd>=0 else '⚠':>10}")
w(f"  {'Sharpe':<14}{x1['sharpe']:>14.3f}{xb['sharpe']:>14.3f}{diff_sharpe:>+14.3f}{'✓' if diff_sharpe>=0 else '⚠':>10}")
w(f"  {'Calmar':<14}{x1['calmar']:>14.3f}{xb['calmar']:>14.3f}{diff_calmar:>+14.3f}{'✓' if diff_calmar>=0 else '⚠':>10}")
w(f"  {'调仓次数':<14}{x1['n_trades']:>14d}{xb['n_trades']:>14d}{diff_trades:>+14d}{'✓' if diff_trades<=0 else '⚠':>10}")
w()
w("  改进点分析：")
w(f"    1) 降仓方向修复: X1 降仓时硬编码 direction='value'，X2 改为 style_score 决定方向")
w(f"       → 修复了 X1 的实现瑕疵，降仓时方向更符合因子信号")
if scheme_letter == 'B':
    w(f"    2) 降仓机制优化: X1 降仓到10%（各持5%），X2-B 直接清仓0%")
    w(f"       → 在深度下跌市场（MA75双跌破）中风险暴露更低")
    w(f"    3) 调仓次数: {'不变' if diff_trades==0 else ('减少' if diff_trades<0 else '增加') + ' ' + str(abs(diff_trades)) + '次'}")
elif scheme_letter == 'C':
    w(f"    2) 降仓机制优化: X1 降仓到10%（各持5%），X2-C 三档降仓")
    w(f"       → MA50→50% 提前预警, MA75浅→25% 部分减仓, MA75深→0% 清仓")
    w(f"    3) 调仓次数: {'不变' if diff_trades==0 else ('减少' if diff_trades<0 else '增加') + ' ' + str(abs(diff_trades)) + '次'}")
elif scheme_letter == 'A':
    w(f"    2) 降仓机制优化: X1 降仓到10%（各持5%），X2-A 分段降仓")
    w(f"       → MA50→50% 提前减仓, MA75→25% 进一步降仓")
    w(f"    3) 调仓次数: {'不变' if diff_trades==0 else ('减少' if diff_trades<0 else '增加') + ' ' + str(abs(diff_trades)) + '次'}")
w(f"    4) F1/F2因子: 完全保留X1原版（Task 1-2 结论：F2当前用法已接近最优）")
w()
w("------------------------------------------------------------------------------")
w("六、降仓周数统计")
w("------------------------------------------------------------------------------")
w()
w(f"  {'方案':<22}{'100%周数':>10}{'50%周数':>10}{'25%周数':>10}{'10%周数':>10}{'0%周数':>10}")
w("  " + "-" * 70)
for name in order:
    ww = weekly_weights[name]
    n_100 = int((ww == 1.0).sum())
    n_50 = int((ww == 0.5).sum())
    n_25 = int((ww == 0.25).sum())
    n_10 = int((ww == CUT_X1).sum())
    n_0 = int((ww == 0.0).sum())
    w(f"  {name:<22}{n_100:>10}{n_50:>10}{n_25:>10}{n_10:>10}{n_0:>10}")
w()
w("------------------------------------------------------------------------------")
w("七、各方案详细 summary")
w("------------------------------------------------------------------------------")
for name in order:
    r, _ = results[name]
    w()
    w(r.summary(name))
w()
w("=" * 70)
w("  报告生成完毕")
w("=" * 70)

# 写入文件
with open(REPORT_PATH, 'w', encoding='utf-8') as f:
    f.write("\n".join(report_lines))

print(f"\n[报告已保存] {REPORT_PATH}")
print("=" * 70)
print("[完成] X2 策略集成回测（Task 5）执行结束")
