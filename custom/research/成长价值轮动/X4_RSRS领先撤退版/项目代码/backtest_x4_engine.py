"""backtest_x4_engine.py — X4 策略集成回测（X2基线 + RSRS领先撤退层）
================================================================
X4 策略 = X2基线（F1+F2，clip放宽到±0.025）+ RSRS修正版β×R²领先撤退层

背景:
  X3"进攻升级"失败（连续加权+KST+ADX三重错配）
  X4 回退 X2 基线，叠加 RSRS 领先撤退层（择时层，不参与方向判断）

策略配置:
  F1：比价MA20方向 = tanh(ratio_dev × 30) × 0.5   （与X1/X2完全相同）
  F2：动量加速度 = accel_diff.clip(±0.025) × 5.0  （clip从±0.02放宽到±0.025）
  style_score = f1 + f2，二元切换（>0→成长，≤0→价值）

  分段降仓（X2方案A）:
    MA50双跌破 → ma50_cut = 0.5
    MA75双跌破 → ma75_cut = 0.5（在ma50基础上再降50%）
    ma_position = ma50_cut × ma75_cut  (1.0 / 0.5 / 0.25)

  RSRS领先撤退层（修正版RSRS = β × R²，标准化z-score）:
    z < -0.7 → rsrs_cut = 0.5
    z < -1.0 → rsrs_cut = 0.2
    否则     → rsrs_cut = 1.0
    RSRS信号NaN（早期N+M日） → rsrs_cut = 1.0（不触发降仓）

  多信号取最保守仓位:
    final_position = min(ma_position, rsrs_cut)

  RSRS对成长100和价值100分别计算，取两者z-score的min（更保守）作为信号。

数据:
  成长100: c:\\temp_v72_data\\index_480080.csv
  价值100: c:\\temp_v72_data\\index_480081.csv
  数据 high/low 列存在但2012~2024-10-28段为空，用 max/min(close_t, close_{t-1}) 近似

回测引擎: c:\\XuanJLH\\Qbot\\custom\\backtests\\backtest_engine.py
  run_backtest_engine_weighted(bt_input, config, position_weight)
  Sharpe 用 rf=2.5%/年（与X1/X2口径一致）

约束:
  - 指数 open=close
  - 佣金万1，无冲击滑点，无跳空滑点
  - 周频调仓（W-FRI）
  - 初始资金100万
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
# 1. 参数
# ============================================================
F1 = 0.5          # 比价MA20权重
F2 = 5.0          # 动量加速度权重
MA_PCT = 0.97     # MA跌破阈值

CLIP_X4 = 0.025   # X4放宽（消融默认）
CLIP_X2 = 0.02    # X2原版

RSRS_N_DEFAULT = 18
RSRS_M_DEFAULT = 600
RSRS_CUT_MODERATE = -0.7   # 降仓到50%
RSRS_CUT_SEVERE = -1.0     # 降仓到20%
RSRS_POS_MODERATE = 0.5
RSRS_POS_SEVERE = 0.2

DATA_DIR = Path(r'c:\temp_v72_data')
REPORT_PATH = DATA_DIR / 'x4_ablation_report.txt'

# ============================================================
# 2. 加载数据
# ============================================================
print("=" * 70)
print("  X4 策略集成回测 — X2基线 + RSRS领先撤退层")
print("=" * 70)

g_raw = pd.read_csv(str(DATA_DIR / 'index_480080.csv'))
v_raw = pd.read_csv(str(DATA_DIR / 'index_480081.csv'))

for d in (g_raw, v_raw):
    d['date'] = pd.to_datetime(d['date'])
    d['close'] = pd.to_numeric(d['close'], errors='coerce')
    d['high'] = pd.to_numeric(d['high'], errors='coerce')
    d['low'] = pd.to_numeric(d['low'], errors='coerce')

g_close = g_raw.set_index('date')['close'].astype(float).sort_index().dropna()
v_close = v_raw.set_index('date')['close'].astype(float).sort_index().dropna()
common = g_close.index.intersection(v_close.index)
g_close = g_close[common].sort_index()
v_close = v_close[common].sort_index()

print(f"数据区间: {g_close.index[0]:%Y-%m-%d} ~ {g_close.index[-1]:%Y-%m-%d}")
print(f"交易日数: {len(g_close)}")

# ============================================================
# 3. 准备 high/low（真实 + 伪值近似）
# ============================================================
# 数据 high/low 列存在但 2012~2024-10-28 段为空，用 max/min(close_t, close_{t-1}) 近似
g_high_real = g_raw.set_index('date')['high'].reindex(g_close.index)
g_low_real = g_raw.set_index('date')['low'].reindex(g_close.index)
v_high_real = v_raw.set_index('date')['high'].reindex(v_close.index)
v_low_real = v_raw.set_index('date')['low'].reindex(v_close.index)

n_real_g_hl = int(g_high_real.notna().sum())
n_real_v_hl = int(v_high_real.notna().sum())
print(f"真实high/low非空: 成长={n_real_g_hl}, 价值={n_real_v_hl} (其余用伪值近似)")


def make_pseudo_hl(close):
    """用 max/min(close_t, close_{t-1}) 作为伪 high/low"""
    prev = close.shift(1)
    df = pd.concat([close, prev], axis=1)
    pseudo_high = df.max(axis=1)
    pseudo_low = df.min(axis=1)
    return pseudo_high, pseudo_low


g_ph, g_pl = make_pseudo_hl(g_close)
v_ph, v_pl = make_pseudo_hl(v_close)

# 拼接：真实值优先，缺失用伪值（保持口径尽可能贴近真实）
g_high = g_high_real.fillna(g_ph)
g_low = g_low_real.fillna(g_pl)
v_high = v_high_real.fillna(v_ph)
v_low = v_low_real.fillna(v_pl)

# ============================================================
# 4. RSRS 指标计算（修正版 = β × R²，标准化z-score）
# ============================================================
def calc_rsrs_z(high, low, N=18, M=600):
    """
    修正版 RSRS = β × R²
    - N: OLS回归窗口（近N日 high 对 low 回归）
    - M: 标准化窗口（滚动M日均值/标准差）
    返回 z-score，shift(1) 防未来函数
    """
    n = len(high)
    beta = np.full(n, np.nan)
    r_squared = np.full(n, np.nan)

    h_arr = high.values.astype(np.float64)
    l_arr = low.values.astype(np.float64)

    for t in range(N - 1, n):
        x = l_arr[t - N + 1:t + 1]
        y = h_arr[t - N + 1:t + 1]
        # 跳过含NaN的窗口
        if np.isnan(x).any() or np.isnan(y).any():
            continue
        x_mean = x.mean()
        y_mean = y.mean()
        denom = ((x - x_mean) ** 2).sum()
        if denom <= 1e-12:
            continue
        b = ((x - x_mean) * (y - y_mean)).sum() / denom
        a = y_mean - b * x_mean
        y_pred = a + b * x
        ss_res = ((y - y_pred) ** 2).sum()
        ss_tot = ((y - y_mean) ** 2).sum()
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
        beta[t] = b
        r_squared[t] = r2

    rsrs = pd.Series(beta * r_squared, index=high.index)
    rsrs_mean = rsrs.rolling(M).mean()
    rsrs_std = rsrs.rolling(M).std()
    z = (rsrs - rsrs_mean) / rsrs_std
    # 防未来函数：决策时点 t 用 t-1 的 z
    return z.shift(1)


def calc_rsrs_z_fast(high, low, N=18, M=600):
    """向量化版 RSRS（性能优化）— 与 calc_rsrs_z 等价但更快"""
    n = len(high)
    h_arr = high.values.astype(np.float64)
    l_arr = low.values.astype(np.float64)

    # 滚动窗口的统计量
    # 使用 pandas rolling + 自定义聚合
    s_h = pd.Series(h_arr, index=high.index)
    s_l = pd.Series(l_arr, index=low.index)

    # 滚动均值
    h_mean = s_h.rolling(N).mean()
    l_mean = s_l.rolling(N).mean()
    # 滚动方差/协方差（ddof=0，与np默认一致）
    l_var = s_l.rolling(N).var(ddof=0)
    hl_cov = (s_h * s_l).rolling(N).mean() - h_mean * l_mean

    beta = hl_cov / l_var.replace(0, np.nan)
    # R² = cov(h,l)^2 / (var(h)*var(l))
    h_var = s_h.rolling(N).var(ddof=0)
    r_squared = (hl_cov ** 2) / (h_var * l_var).replace(0, np.nan)

    rsrs = beta * r_squared
    rsrs_mean = rsrs.rolling(M).mean()
    rsrs_std = rsrs.rolling(M).std()
    z = (rsrs - rsrs_mean) / rsrs_std
    return z.shift(1)


# ============================================================
# 5. X2 基线因子（F1+F2，clip可调）
# ============================================================
def calc_style_score(g_close, v_close, clip_abs):
    """计算 style_score = F1 + F2
    F1 = tanh(ratio_dev × 30) × 0.5
    F2 = accel_diff.clip(±clip_abs) × 5.0
    """
    ratio = (g_close / v_close).shift(1)
    ratio_ma20 = ratio.rolling(20).mean()
    ratio_dev = ratio / ratio_ma20 - 1
    f1_signal = np.tanh(ratio_dev * 30) * F1

    g_roc21 = g_close.pct_change(21).shift(1)
    v_roc21 = v_close.pct_change(21).shift(1)
    g_accel = g_roc21 - g_roc21.shift(10)
    v_accel = v_roc21 - v_roc21.shift(10)
    accel_diff = (g_accel - v_accel).clip(-clip_abs, clip_abs)
    f2_signal = accel_diff * F2

    return f1_signal + f2_signal


# ============================================================
# 6. MA 择时信号
# ============================================================
def calc_ma_signals(g_close, v_close):
    """计算 MA50 / MA75 双跌破信号（与X2一致）"""
    g_ma50 = g_close.shift(1).rolling(50).mean()
    v_ma50 = v_close.shift(1).rolling(50).mean()
    both_below_ma50 = (g_close.shift(1) < g_ma50 * MA_PCT) & (v_close.shift(1) < v_ma50 * MA_PCT)

    g_ma75 = g_close.shift(1).rolling(75).mean()
    v_ma75 = v_close.shift(1).rolling(75).mean()
    both_below_ma75 = (g_close.shift(1) < g_ma75 * MA_PCT) & (v_close.shift(1) < v_ma75 * MA_PCT)

    return both_below_ma50, both_below_ma75


# ============================================================
# 7. 周频采样与回测
# ============================================================
def expand_to_daily(weekly_series, daily_index):
    """周频信号前向填充到日频（与X2引擎一致）"""
    s = weekly_series.copy()
    idx_pos = daily_index.searchsorted(s.index, side='right') - 1
    valid = idx_pos >= 0
    s = s[valid]
    idx_pos = idx_pos[valid]
    s.index = daily_index[idx_pos]
    s = s[~s.index.duplicated(keep='last')]
    daily = pd.Series(np.nan, index=daily_index, dtype=s.dtype)
    daily.loc[s.index] = s
    return daily.ffill()


def run_backtest(daily_signal, daily_weight, g_close_s, v_close_s):
    """运行回测引擎，返回 BacktestResult"""
    mask = ~(daily_signal.isna() | daily_weight.isna())
    daily_signal = daily_signal[mask].astype(str)
    daily_weight = daily_weight[mask].astype(float)
    g_align = g_close_s.loc[daily_signal.index]
    v_align = v_close_s.loc[daily_signal.index]

    dates = daily_signal.index.strftime('%Y-%m-%d').values
    signal = daily_signal.values.astype(str)
    position_weight = daily_weight.values.astype(np.float64)

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
        commission=0.0001,
        impact_slippage=0.0,
        apply_gap_slippage=False,
    )
    return run_backtest_engine_weighted(bt_input, config, position_weight)


def calc_metrics_v72(result, freq=252, rf_annual=0.025):
    """V72 风格的指标计算 — 与 backtest_v72_engine.py 完全一致
    Sharpe 使用 rf=2.5%/年
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
        'ann': ann, 'dd': max_dd, 'sharpe': sharpe, 'calmar': calmar,
        'wr': wr, 'total': total,
        'n_trades': result.metrics['num_trades'],
        'final_nav': result.metrics['final_nav'],
        'final_multiple': result.metrics['final_multiple'],
        'num_days': result.metrics['num_days'],
    }


# ============================================================
# 8. X4 仓位构建函数（核心）
# ============================================================
def build_x4_position(weekly_df, rsrs_z, mode='min', with_rsrs=True):
    """
    构建 X4 仓位（signal + weight）
    - rsrs_z: 周频RSRS z-score序列
    - mode: 'min' = 取成长/价值RSRS的min（更保守）; 'mean' = 取均值
    - with_rsrs: True=启用RSRS层; False=纯X2基线（消融用）
    """
    n = len(weekly_df)
    signal = pd.Series(['value'] * n, index=weekly_df.index, dtype=object)
    weight = pd.Series(1.0, index=weekly_df.index)
    n_rsrs_moderate = 0
    n_rsrs_severe = 0
    n_ma50_cut = 0
    n_ma75_cut = 0

    for i in range(n):
        row = weekly_df.iloc[i]
        # 方向始终由 style_score 决定（X2基线，修复X1瑕疵）
        # 强制 bool 转换：resample 后 candidate_g 可能是 float64(0.0/1.0)
        cg_val = row['candidate_g']
        is_growth = bool(cg_val) if not (isinstance(cg_val, float) and np.isnan(cg_val)) else False
        signal.iloc[i] = 'growth' if is_growth else 'value'

        # MA 分段降仓（X2方案A：MA75→25%, MA50→50%，MA75优先）
        if row['both_below_ma75']:
            ma_position = 0.25
            n_ma75_cut += 1
        elif row['both_below_ma50']:
            ma_position = 0.5
            n_ma50_cut += 1
        else:
            ma_position = 1.0

        # RSRS 撤退层
        if with_rsrs:
            z_val = row['rsrs_z']
            if pd.isna(z_val):
                rsrs_cut = 1.0  # 早期段不触发降仓
            elif z_val < RSRS_CUT_SEVERE:
                rsrs_cut = RSRS_POS_SEVERE
                n_rsrs_severe += 1
            elif z_val < RSRS_CUT_MODERATE:
                rsrs_cut = RSRS_POS_MODERATE
                n_rsrs_moderate += 1
            else:
                rsrs_cut = 1.0
        else:
            rsrs_cut = 1.0

        # 多信号取最保守仓位
        weight.iloc[i] = min(ma_position, rsrs_cut)

    stats = {
        'n_ma50_cut': n_ma50_cut,
        'n_ma75_cut': n_ma75_cut,
        'n_rsrs_moderate': n_rsrs_moderate,
        'n_rsrs_severe': n_rsrs_severe,
    }
    return signal, weight, stats


def build_x1_baseline(weekly_df):
    """X1基线（与X2引擎一致）— MA75双跌破→10%, direction='value'硬编码"""
    position = pd.Series(np.nan, index=weekly_df.index)
    current_pos = None
    n_switches = 0
    CUT_X1 = 0.1

    for i in range(len(weekly_df)):
        row = weekly_df.iloc[i]
        # 强制 bool 转换：resample 后 candidate_g 可能是 float64(0.0/1.0)
        cg_val = row['candidate_g']
        is_growth = bool(cg_val) if not (isinstance(cg_val, float) and np.isnan(cg_val)) else False
        if row['both_below_ma75']:
            position.iloc[i] = CUT_X1
            continue
        if current_pos is None:
            current_pos = 1.0 if is_growth else 0.0
            position.iloc[i] = current_pos
            n_switches += 1
            continue
        target = 1.0 if is_growth else 0.0
        if target == current_pos:
            position.iloc[i] = current_pos
        else:
            current_pos = target
            position.iloc[i] = current_pos
            n_switches += 1

    signal = pd.Series(
        ['growth' if p == 1.0 else 'value' for p in position.values],
        index=position.index
    )
    weight = pd.Series(
        [CUT_X1 if p == CUT_X1 else 1.0 for p in position.values],
        index=position.index
    )
    return signal, weight, n_switches


# ============================================================
# 9. 计算因子与信号
# ============================================================
print("\n[因子计算] F1=比价MA20方向(0.5) + F2=动量加速度(5.0, clip±0.025)")

style_score_x4 = calc_style_score(g_close, v_close, CLIP_X4)
style_score_x2 = calc_style_score(g_close, v_close, CLIP_X2)
candidate_g_x4 = style_score_x4 > 0
candidate_g_x2 = style_score_x2 > 0

both_below_ma50, both_below_ma75 = calc_ma_signals(g_close, v_close)

# ============================================================
# 10. RSRS 计算（默认 N=18, M=600）
# ============================================================
print(f"\n[RSRS计算] N={RSRS_N_DEFAULT}, M={RSRS_M_DEFAULT}")
g_rsrs_z = calc_rsrs_z_fast(g_high, g_low, RSRS_N_DEFAULT, RSRS_M_DEFAULT)
v_rsrs_z = calc_rsrs_z_fast(v_high, v_low, RSRS_N_DEFAULT, RSRS_M_DEFAULT)

rsrs_z_min = pd.concat([g_rsrs_z, v_rsrs_z], axis=1).min(axis=1)
rsrs_z_mean = pd.concat([g_rsrs_z, v_rsrs_z], axis=1).mean(axis=1)

print(f"  Growth RSRS z 非NaN: {g_rsrs_z.notna().sum()}/{len(g_rsrs_z)}")
print(f"  Value  RSRS z 非NaN: {v_rsrs_z.notna().sum()}/{len(v_rsrs_z)}")
print(f"  Min z 非NaN: {rsrs_z_min.notna().sum()}/{len(rsrs_z_min)}")
if rsrs_z_min.notna().sum() > 0:
    first_valid = rsrs_z_min[rsrs_z_min.notna()].index[0]
    print(f"  Min z 首个非NaN日: {first_valid:%Y-%m-%d}")


# ============================================================
# 11. 周频采样
# ============================================================
df_weekly = pd.DataFrame(index=g_close.index)
df_weekly['g_close'] = g_close
df_weekly['v_close'] = v_close
df_weekly['candidate_g_x4'] = candidate_g_x4
df_weekly['candidate_g_x2'] = candidate_g_x2
df_weekly['both_below_ma50'] = both_below_ma50
df_weekly['both_below_ma75'] = both_below_ma75
df_weekly['rsrs_z_min'] = rsrs_z_min
df_weekly['rsrs_z_mean'] = rsrs_z_mean

weekly = df_weekly.resample('W-FRI').last().dropna(subset=['candidate_g_x4']).iloc[1:]
print(f"\n周数: {len(weekly)}")
print(f"  MA50双跌破周数: {int(weekly['both_below_ma50'].sum())}")
print(f"  MA75双跌破周数: {int(weekly['both_below_ma75'].sum())}")
n_rsrs_mod_w = int(((weekly['rsrs_z_min'] < RSRS_CUT_MODERATE) & (weekly['rsrs_z_min'] >= RSRS_CUT_SEVERE)).sum())
n_rsrs_sev_w = int((weekly['rsrs_z_min'] < RSRS_CUT_SEVERE).sum())
print(f"  RSRS中度降仓周数 (z<-0.7): {n_rsrs_mod_w}")
print(f"  RSRS重度降仓周数 (z<-1.0): {n_rsrs_sev_w}")
print(f"  RSRS有效周数 (非NaN): {weekly['rsrs_z_min'].notna().sum()}")


# ============================================================
# 12. 运行消融实验
# ============================================================
print("\n" + "=" * 70)
print("  消融实验: X1基线 / X2基线 / X4-full / X4-no-RSRS / X4-no-clip")
print("=" * 70)

results = {}        # name -> (result, metrics_v72)
weekly_stats = {}   # name -> stats dict

# --- X1 基线 ---
print("\n[运行] X1基线 (MA75→10%, direction='value' 硬编码)")
# X1 用 X2 的 candidate_g（X1 原版 clip=0.02）
weekly_x1 = weekly.copy()
weekly_x1['candidate_g'] = weekly_x1['candidate_g_x2'].astype(bool)
wsig_x1, ww_x1, _ = build_x1_baseline(weekly_x1)
dsig_x1 = expand_to_daily(wsig_x1, g_close.index)
dw_x1 = expand_to_daily(ww_x1, g_close.index)
result_x1 = run_backtest(dsig_x1, dw_x1, g_close, v_close)
m_x1 = calc_metrics_v72(result_x1)
results['X1基线'] = (result_x1, m_x1)
print(f"  年化={m_x1['ann']*100:6.2f}%  回撤={m_x1['dd']*100:6.2f}%  "
      f"Sharpe={m_x1['sharpe']:.3f}  Calmar={m_x1['calmar']:.3f}  调仓={m_x1['n_trades']}")

# --- X2 基线 (clip=0.02, 无RSRS) ---
print("\n[运行] X2基线 (clip±0.02, MA50→50%/MA75→25%, 无RSRS)")
weekly_x2 = weekly.copy()
weekly_x2['candidate_g'] = weekly_x2['candidate_g_x2'].astype(bool)
wsig_x2, ww_x2, stats_x2 = build_x4_position(weekly_x2, weekly_x2['rsrs_z_min'],
                                              mode='min', with_rsrs=False)
dsig_x2 = expand_to_daily(wsig_x2, g_close.index)
dw_x2 = expand_to_daily(ww_x2, g_close.index)
result_x2 = run_backtest(dsig_x2, dw_x2, g_close, v_close)
m_x2 = calc_metrics_v72(result_x2)
results['X2基线'] = (result_x2, m_x2)
weekly_stats['X2基线'] = stats_x2
print(f"  年化={m_x2['ann']*100:6.2f}%  回撤={m_x2['dd']*100:6.2f}%  "
      f"Sharpe={m_x2['sharpe']:.3f}  Calmar={m_x2['calmar']:.3f}  调仓={m_x2['n_trades']}")

# --- X4-full (clip=0.025, RSRS=min模式) ---
print("\n[运行] X4-full (clip±0.025 + RSRS z-min, MA分段降仓 + RSRS撤退层)")
weekly_full = weekly.copy()
weekly_full['candidate_g'] = weekly_full['candidate_g_x4'].astype(bool)
weekly_full['rsrs_z'] = weekly_full['rsrs_z_min']
wsig_full, ww_full, stats_full = build_x4_position(weekly_full, weekly_full['rsrs_z_min'],
                                                    mode='min', with_rsrs=True)
dsig_full = expand_to_daily(wsig_full, g_close.index)
dw_full = expand_to_daily(ww_full, g_close.index)
result_full = run_backtest(dsig_full, dw_full, g_close, v_close)
m_full = calc_metrics_v72(result_full)
results['X4-full'] = (result_full, m_full)
weekly_stats['X4-full'] = stats_full
print(f"  年化={m_full['ann']*100:6.2f}%  回撤={m_full['dd']*100:6.2f}%  "
      f"Sharpe={m_full['sharpe']:.3f}  Calmar={m_full['calmar']:.3f}  调仓={m_full['n_trades']}")
print(f"  RSRS中度降仓周数(z<-0.7): {stats_full['n_rsrs_moderate']}, "
      f"重度降仓周数(z<-1.0): {stats_full['n_rsrs_severe']}")

# --- X4-no-RSRS (clip=0.025, 无RSRS) ---
print("\n[运行] X4-no-RSRS (clip±0.025, 无RSRS) — 测试clip放宽的独立贡献")
weekly_no_rsrs = weekly.copy()
weekly_no_rsrs['candidate_g'] = weekly_no_rsrs['candidate_g_x4'].astype(bool)
wsig_no_rsrs, ww_no_rsrs, stats_no_rsrs = build_x4_position(
    weekly_no_rsrs, weekly_no_rsrs['rsrs_z_min'],
    mode='min', with_rsrs=False)
dsig_no_rsrs = expand_to_daily(wsig_no_rsrs, g_close.index)
dw_no_rsrs = expand_to_daily(ww_no_rsrs, g_close.index)
result_no_rsrs = run_backtest(dsig_no_rsrs, dw_no_rsrs, g_close, v_close)
m_no_rsrs = calc_metrics_v72(result_no_rsrs)
results['X4-no-RSRS'] = (result_no_rsrs, m_no_rsrs)
weekly_stats['X4-no-RSRS'] = stats_no_rsrs
print(f"  年化={m_no_rsrs['ann']*100:6.2f}%  回撤={m_no_rsrs['dd']*100:6.2f}%  "
      f"Sharpe={m_no_rsrs['sharpe']:.3f}  Calmar={m_no_rsrs['calmar']:.3f}  "
      f"调仓={m_no_rsrs['n_trades']}")

# --- X4-no-clip (clip=0.02原版 + RSRS) ---
print("\n[运行] X4-no-clip (clip±0.02原版 + RSRS) — 测试RSRS的独立贡献")
weekly_no_clip = weekly.copy()
weekly_no_clip['candidate_g'] = weekly_no_clip['candidate_g_x2'].astype(bool)
weekly_no_clip['rsrs_z'] = weekly_no_clip['rsrs_z_min']
wsig_no_clip, ww_no_clip, stats_no_clip = build_x4_position(
    weekly_no_clip, weekly_no_clip['rsrs_z_min'],
    mode='min', with_rsrs=True)
dsig_no_clip = expand_to_daily(wsig_no_clip, g_close.index)
dw_no_clip = expand_to_daily(ww_no_clip, g_close.index)
result_no_clip = run_backtest(dsig_no_clip, dw_no_clip, g_close, v_close)
m_no_clip = calc_metrics_v72(result_no_clip)
results['X4-no-clip'] = (result_no_clip, m_no_clip)
weekly_stats['X4-no-clip'] = stats_no_clip
print(f"  年化={m_no_clip['ann']*100:6.2f}%  回撤={m_no_clip['dd']*100:6.2f}%  "
      f"Sharpe={m_no_clip['sharpe']:.3f}  Calmar={m_no_clip['calmar']:.3f}  "
      f"调仓={m_no_clip['n_trades']}")

# --- X4-mean (RSRS z-mean 模式) ---
print("\n[运行] X4-mean (clip±0.025 + RSRS z-mean) — 对比 min vs mean 信号聚合")
weekly_mean = weekly.copy()
weekly_mean['candidate_g'] = weekly_mean['candidate_g_x4'].astype(bool)
weekly_mean['rsrs_z'] = weekly_mean['rsrs_z_mean']
wsig_mean, ww_mean, stats_mean = build_x4_position(
    weekly_mean, weekly_mean['rsrs_z_mean'],
    mode='mean', with_rsrs=True)
dsig_mean = expand_to_daily(wsig_mean, g_close.index)
dw_mean = expand_to_daily(ww_mean, g_close.index)
result_mean = run_backtest(dsig_mean, dw_mean, g_close, v_close)
m_mean = calc_metrics_v72(result_mean)
results['X4-mean'] = (result_mean, m_mean)
weekly_stats['X4-mean'] = stats_mean
print(f"  年化={m_mean['ann']*100:6.2f}%  回撤={m_mean['dd']*100:6.2f}%  "
      f"Sharpe={m_mean['sharpe']:.3f}  Calmar={m_mean['calmar']:.3f}  "
      f"调仓={m_mean['n_trades']}")


# ============================================================
# 13. RSRS 参数敏感性扫描 (N=10/18/24/36)
# ============================================================
print("\n" + "=" * 70)
print("  RSRS 参数敏感性扫描 (N=10/18/24/36, M=600)")
print("=" * 70)

sensitivity_rows = []
for n_val in [10, 18, 24, 36]:
    print(f"\n[扫描] N={n_val}")
    g_z = calc_rsrs_z_fast(g_high, g_low, n_val, RSRS_M_DEFAULT)
    v_z = calc_rsrs_z_fast(v_high, v_low, n_val, RSRS_M_DEFAULT)
    z_min = pd.concat([g_z, v_z], axis=1).min(axis=1)

    # 周频采样
    df_sens = pd.DataFrame(index=g_close.index)
    df_sens['candidate_g'] = candidate_g_x4.astype(bool)
    df_sens['both_below_ma50'] = both_below_ma50
    df_sens['both_below_ma75'] = both_below_ma75
    df_sens['rsrs_z'] = z_min
    wk_sens = df_sens.resample('W-FRI').last().dropna(subset=['candidate_g']).iloc[1:]

    n_valid = int(wk_sens['rsrs_z'].notna().sum())
    n_trig_mod = int(((wk_sens['rsrs_z'] < RSRS_CUT_MODERATE) &
                      (wk_sens['rsrs_z'] >= RSRS_CUT_SEVERE)).sum())
    n_trig_sev = int((wk_sens['rsrs_z'] < RSRS_CUT_SEVERE).sum())
    n_total = len(wk_sens)
    trig_freq = (n_trig_mod + n_trig_sev) / max(n_valid, 1) * 100

    wsig, ww, _ = build_x4_position(wk_sens, wk_sens['rsrs_z'],
                                     mode='min', with_rsrs=True)
    dsig = expand_to_daily(wsig, g_close.index)
    dw = expand_to_daily(ww, g_close.index)
    res = run_backtest(dsig, dw, g_close, v_close)
    m = calc_metrics_v72(res)
    print(f"  RSRS触发: 中度={n_trig_mod}周, 重度={n_trig_sev}周, "
          f"频率={trig_freq:.1f}% (基于{n_valid}个有效周)")
    print(f"  年化={m['ann']*100:6.2f}%  回撤={m['dd']*100:6.2f}%  "
          f"Sharpe={m['sharpe']:.3f}  Calmar={m['calmar']:.3f}  调仓={m['n_trades']}")

    sensitivity_rows.append({
        'N': n_val, 'n_valid': n_valid,
        'n_trig_mod': n_trig_mod, 'n_trig_sev': n_trig_sev,
        'trig_freq': trig_freq,
        'ann': m['ann'], 'dd': m['dd'], 'sharpe': m['sharpe'],
        'calmar': m['calmar'], 'n_trades': m['n_trades'],
    })

sens_df = pd.DataFrame(sensitivity_rows)


# ============================================================
# 14. 主对比表输出
# ============================================================
# X3 文档基准（失败策略，全周期）
x3_doc = {'ann': 0.1994, 'dd': -0.3420, 'sharpe': 0.957, 'calmar': 0.583, 'n_trades': 305}

print("\n" + "=" * 78)
print("  X4 策略消融分析（vs X1/X2/X3基准）")
print("=" * 78)
header = f"  {'方案':<22}{'年化':>10}{'回撤':>10}{'Sharpe':>9}{'Calmar':>9}{'调仓次数':>10}"
print(header)
print("  " + "-" * 72)
# X3 文档基准
print(f"  {'X1基线':<22}{m_x1['ann']*100:>9.2f}%{m_x1['dd']*100:>9.2f}%"
      f"{m_x1['sharpe']:>9.3f}{m_x1['calmar']:>9.3f}{m_x1['n_trades']:>10}")
print(f"  {'X2基线':<22}{m_x2['ann']*100:>9.2f}%{m_x2['dd']*100:>9.2f}%"
      f"{m_x2['sharpe']:>9.3f}{m_x2['calmar']:>9.3f}{m_x2['n_trades']:>10}")
print(f"  {'X3-full(失败)':<22}{x3_doc['ann']*100:>9.2f}%{x3_doc['dd']*100:>9.2f}%"
      f"{x3_doc['sharpe']:>9.3f}{x3_doc['calmar']:>9.3f}{x3_doc['n_trades']:>10}")
order = ['X4-full', 'X4-no-RSRS', 'X4-no-clip', 'X4-mean']
for name in order:
    r, m = results[name]
    print(f"  {name:<22}{m['ann']*100:>9.2f}%{m['dd']*100:>9.2f}%"
          f"{m['sharpe']:>9.3f}{m['calmar']:>9.3f}{m['n_trades']:>10}")
print("=" * 78)

# RSRS参数敏感性表
print("\n" + "=" * 78)
print("  RSRS 参数敏感性（X4-full 配置: clip±0.025 + RSRS z-min）")
print("=" * 78)
print(f"  {'N值':<8}{'有效周数':>10}{'中度触发':>10}{'重度触发':>10}"
      f"{'触发频率':>10}{'年化':>10}{'回撤':>10}{'Sharpe':>9}{'Calmar':>9}")
print("  " + "-" * 76)
for _, row in sens_df.iterrows():
    print(f"  {int(row['N']):<8}{int(row['n_valid']):>10}{int(row['n_trig_mod']):>10}"
          f"{int(row['n_trig_sev']):>10}{row['trig_freq']:>9.1f}%"
          f"{row['ann']*100:>9.2f}%{row['dd']*100:>9.2f}%"
          f"{row['sharpe']:>9.3f}{row['calmar']:>9.3f}")
print("=" * 78)

# ============================================================
# 14b. RSRS 是否压收益分析（任务Step 6）
# ============================================================
print("\n" + "=" * 78)
print("  关键问题：RSRS叠加后是否压收益？")
print("=" * 78)

x4_full_ann = results['X4-full'][1]['ann']
x4_no_rsrs_ann = results['X4-no-RSRS'][1]['ann']
x4_full_dd = results['X4-full'][1]['dd']
x4_no_rsrs_dd = results['X4-no-RSRS'][1]['dd']

print(f"\n  X2基线:        年化={m_x2['ann']*100:.2f}%  回撤={m_x2['dd']*100:.2f}%")
print(f"  X4-no-RSRS:    年化={x4_no_rsrs_ann*100:.2f}%  回撤={x4_no_rsrs_dd*100:.2f}%  (X2基线+clip±0.025)")
print(f"  X4-full:       年化={x4_full_ann*100:.2f}%  回撤={x4_full_dd*100:.2f}%  (X2基线+clip±0.025+RSRS)")

# RSRS增量贡献
rsrs_delta_ann = (x4_full_ann - x4_no_rsrs_ann) * 100
rsrs_delta_dd = (x4_full_dd - x4_no_rsrs_dd) * 100  # 正值=回撤变小=改善
rsrs_delta_sharpe = results['X4-full'][1]['sharpe'] - results['X4-no-RSRS'][1]['sharpe']
rsrs_delta_calmar = results['X4-full'][1]['calmar'] - results['X4-no-RSRS'][1]['calmar']

print(f"\n  RSRS增量贡献 (X4-full vs X4-no-RSRS):")
print(f"    年化:    {rsrs_delta_ann:+.2f}pp")
print(f"    回撤:    {rsrs_delta_dd:+.2f}pp  (正=回撤变小/改善)")
print(f"    Sharpe:  {rsrs_delta_sharpe:+.3f}")
print(f"    Calmar:  {rsrs_delta_calmar:+.3f}")

# 判断结论（严格标准：回撤改善>=2pp才算显著）
dd_improvement_pp = rsrs_delta_dd  # 正值=改善
ann_loss_pp = -rsrs_delta_ann  # 正值=损失
tradeoff_ratio = ann_loss_pp / dd_improvement_pp if dd_improvement_pp > 0.01 else float('inf')

print("\n  [结论]")
if x4_full_ann >= m_x2['ann'] - 0.005:
    print(f"    X4-full年化({x4_full_ann*100:.2f}%) ≥ X2基线({m_x2['ann']*100:.2f}%)")
    print(f"    → RSRS有效，不压收益，可作为正式版")
elif dd_improvement_pp >= 2.0 and tradeoff_ratio < 1.5:
    print(f"    X4-full年化({x4_full_ann*100:.2f}%) < X2基线({m_x2['ann']*100:.2f}%)")
    print(f"    但回撤显著改善 {dd_improvement_pp:.2f}pp（>=2pp阈值）")
    print(f"    权衡性价比: 年化损失{ann_loss_pp:.2f}pp / 回撤改善{dd_improvement_pp:.2f}pp = {tradeoff_ratio:.2f}（<1.5合理）")
    print(f"    → RSRS是'风控换收益'的合理权衡，可考虑采用")
elif dd_improvement_pp >= 2.0:
    print(f"    X4-full年化({x4_full_ann*100:.2f}%) < X2基线({m_x2['ann']*100:.2f}%)")
    print(f"    回撤改善 {dd_improvement_pp:.2f}pp，但权衡性价比差: {tradeoff_ratio:.2f}（>=1.5）")
    print(f"    → RSRS是'风控换收益'但不划算，不建议采用")
else:
    print(f"    X4-full年化({x4_full_ann*100:.2f}%) < X2基线({m_x2['ann']*100:.2f}%)")
    print(f"    且回撤改善仅 {dd_improvement_pp:.2f}pp（<2pp显著阈值）")
    print(f"    权衡性价比: 年化损失{ann_loss_pp:.2f}pp / 回撤改善{dd_improvement_pp:.2f}pp = {tradeoff_ratio:.2f}")
    print(f"    → RSRS无效，应移除")


# ============================================================
# 15. 选择最优 X4 配置
# ============================================================
# 选择标准（与Task 7一致）：收益为先 + 风控为辅
# 1) 年化不低于X2基线超过0.5pp
# 2) 在剩余方案中按 Sharpe 排序
# 3) 若Sharpe差异<1%，取回撤更小者
print("\n[选择] 最优 X4 配置（收益为先 + 风控为辅）")

x2_ann = m_x2['ann']
candidates_x4 = []
for name in ['X4-full', 'X4-mean']:
    _, m = results[name]
    ann_ok = m['ann'] >= x2_ann - 0.005  # 年化不低于X2超过0.5pp
    candidates_x4.append((name, m, ann_ok))
    print(f"  {name}: 年化={m['ann']*100:.2f}% vs X2 {x2_ann*100:.2f}% "
          f"(差异{(m['ann']-x2_ann)*100:+.2f}pp) | "
          f"Sharpe={m['sharpe']:.3f} | 回撤={m['dd']*100:.2f}% | "
          f"通过收益门槛: {ann_ok}")

# RSRS参数敏感性中也找最优（按Sharpe）
best_sens = sens_df.loc[sens_df['sharpe'].idxmax()]
print(f"\n  RSRS参数敏感性最优: N={int(best_sens['N'])} "
      f"(Sharpe={best_sens['sharpe']:.3f}, 年化={best_sens['ann']*100:.2f}%, "
      f"回撤={best_sens['dd']*100:.2f}%)")

# 综合: 从 X4-full/X4-mean + 最优N 中选最优
# 候选列表（放宽门槛：即使没通过收益门槛也加入，用于对比）
final_candidates = []
for name in ['X4-full', 'X4-mean']:
    _, m = results[name]
    final_candidates.append((name, m, 18))  # 默认N=18

# 加入敏感性中最优N的X4-full结果
best_n = int(best_sens['N'])
if best_n != 18:
    # 重新跑 X4-full with best_n
    g_z_best = calc_rsrs_z_fast(g_high, g_low, best_n, RSRS_M_DEFAULT)
    v_z_best = calc_rsrs_z_fast(v_high, v_low, best_n, RSRS_M_DEFAULT)
    z_min_best = pd.concat([g_z_best, v_z_best], axis=1).min(axis=1)
    df_best = pd.DataFrame(index=g_close.index)
    df_best['candidate_g'] = candidate_g_x4.astype(bool)
    df_best['both_below_ma50'] = both_below_ma50
    df_best['both_below_ma75'] = both_below_ma75
    df_best['rsrs_z'] = z_min_best
    wk_best = df_best.resample('W-FRI').last().dropna(subset=['candidate_g']).iloc[1:]
    wsig_b, ww_b, _ = build_x4_position(wk_best, wk_best['rsrs_z'],
                                          mode='min', with_rsrs=True)
    dsig_b = expand_to_daily(wsig_b, g_close.index)
    dw_b = expand_to_daily(ww_b, g_close.index)
    res_b = run_backtest(dsig_b, dw_b, g_close, v_close)
    m_b = calc_metrics_v72(res_b)
    final_candidates.append((f'X4-full(N={best_n})', m_b, best_n))
    print(f"  X4-full(N={best_n}): 年化={m_b['ann']*100:.2f}% Sharpe={m_b['sharpe']:.3f} 回撤={m_b['dd']*100:.2f}%")

# 按Sharpe降序排序（如果为空，用X4-full默认）
if final_candidates:
    final_sorted = sorted(final_candidates, key=lambda x: x[1]['sharpe'], reverse=True)
    best_x4_name, best_x4_m, best_x4_n = final_sorted[0]
else:
    best_x4_name = 'X4-full'
    best_x4_m = results['X4-full'][1]
    best_x4_n = 18
print(f"\n  >>> 最优 X4 配置: {best_x4_name} (N={best_x4_n})")
print(f"      年化={best_x4_m['ann']*100:.2f}%  回撤={best_x4_m['dd']*100:.2f}%  "
      f"Sharpe={best_x4_m['sharpe']:.3f}  Calmar={best_x4_m['calmar']:.3f}")

# 改进幅度对比 X4 vs X2 vs X1
print("\n  X4最优 vs X2基线 vs X1基线 改进幅度:")
print(f"    {'指标':<10}{'X1基线':>12}{'X2基线':>12}{'X4最优':>12}{'X4vsX2':>12}")
print("    " + "-" * 56)
def fmt_pct(x): return f"{x*100:.2f}%"
def fmt_num(x): return f"{x:.3f}"
print(f"    {'年化':<10}{m_x1['ann']*100:>11.2f}%{m_x2['ann']*100:>11.2f}%"
      f"{best_x4_m['ann']*100:>11.2f}%{(best_x4_m['ann']-m_x2['ann'])*100:>+11.2f}pp")
print(f"    {'回撤':<10}{m_x1['dd']*100:>11.2f}%{m_x2['dd']*100:>11.2f}%"
      f"{best_x4_m['dd']*100:>11.2f}%{(best_x4_m['dd']-m_x2['dd'])*100:>+11.2f}pp")
print(f"    {'Sharpe':<10}{m_x1['sharpe']:>12.3f}{m_x2['sharpe']:>12.3f}"
      f"{best_x4_m['sharpe']:>12.3f}{best_x4_m['sharpe']-m_x2['sharpe']:>+12.3f}")
print(f"    {'Calmar':<10}{m_x1['calmar']:>12.3f}{m_x2['calmar']:>12.3f}"
      f"{best_x4_m['calmar']:>12.3f}{best_x4_m['calmar']-m_x2['calmar']:>+12.3f}")
print(f"    {'调仓次数':<10}{m_x1['n_trades']:>12d}{m_x2['n_trades']:>12d}"
      f"{best_x4_m['n_trades']:>12d}{best_x4_m['n_trades']-m_x2['n_trades']:>+12d}")


# ============================================================
# 16. 详细 summary 输出
# ============================================================
print("\n" + "=" * 78)
print("  各方案详细 summary")
print("=" * 78)
for name in order:
    r, _ = results[name]
    print()
    print(r.summary(name))

print("\n" + "=" * 78)
print(f"  最优 X4 配置: {best_x4_name}")
print("=" * 78)
# 若最优为X4-full或X4-mean，直接复用 results；否则重新打印
if best_x4_name in results:
    print(results[best_x4_name][0].summary(best_x4_name))
    final_metrics = results[best_x4_name][1]
else:
    # 重新跑一次打印
    print(f"  N={best_x4_n} (从敏感性扫描中复现)")
    final_metrics = best_x4_m

# V72风格指标
print(f"\n  V72 风格指标（Sharpe用rf=2.5%/年）:")
print(f"    年化收益:  {final_metrics['ann']*100:8.2f}%")
print(f"    最大回撤:  {final_metrics['dd']*100:8.2f}%")
print(f"    Sharpe:    {final_metrics['sharpe']:8.3f}")
print(f"    Calmar:    {final_metrics['calmar']:8.3f}")
print(f"    胜率:      {final_metrics['wr']*100:8.2f}%")
print(f"    总收益:    {final_metrics['total']*100:8.2f}%")
print(f"    最终净值:  {final_metrics['final_nav']:>12,.0f}")
print(f"    倍率:      {final_metrics['final_multiple']:8.2f}x")
print(f"    交易日数:  {final_metrics['num_days']:8d}")
print(f"    调仓次数:  {final_metrics['n_trades']:8d}")


# ============================================================
# 17. 降仓统计
# ============================================================
print("\n" + "=" * 78)
print("  降仓统计")
print("=" * 78)
print(f"  {'方案':<22}{'MA50触发':>10}{'MA75触发':>10}{'RSRS中度':>10}{'RSRS重度':>10}")
print("  " + "-" * 70)
for name in order:
    if name in weekly_stats:
        s = weekly_stats[name]
        print(f"  {name:<22}{s['n_ma50_cut']:>10}{s['n_ma75_cut']:>10}"
              f"{s['n_rsrs_moderate']:>10}{s['n_rsrs_severe']:>10}")
    else:
        # X1基线 无 stats
        print(f"  {name:<22}{'-':>10}{'-':>10}{'-':>10}{'-':>10}")
print("=" * 78)


# ============================================================
# 18. 保存报告到文件
# ============================================================
report_lines = []

def w(line=""):
    report_lines.append(line)


w("=" * 78)
w("  X4 策略集成回测报告")
w("  X2基线 + RSRS领先撤退层（修正版β×R²）")
w("=" * 78)
w()
w(f"数据区间: {g_close.index[0]:%Y-%m-%d} ~ {g_close.index[-1]:%Y-%m-%d}")
w(f"交易日数: {len(g_close)}")
w(f"真实high/low非空: 成长={n_real_g_hl}, 价值={n_real_v_hl} (其余用 max/min(close_t, close_{{t-1}}) 近似)")
w(f"周频样本(W-FRI): {len(weekly)}")
w(f"  MA50双跌破周数: {int(weekly['both_below_ma50'].sum())}")
w(f"  MA75双跌破周数: {int(weekly['both_below_ma75'].sum())}")
w(f"  RSRS中度降仓周数 (z<-0.7): {n_rsrs_mod_w}")
w(f"  RSRS重度降仓周数 (z<-1.0): {n_rsrs_sev_w}")
w(f"  RSRS有效周数 (非NaN): {weekly['rsrs_z_min'].notna().sum()}")
w()
w("-" * 78)
w("一、策略配置")
w("-" * 78)
w()
w("  F1因子: 比价MA20方向 (f1=0.5) — 与X1/X2完全相同")
w("    ratio = (g_close/v_close).shift(1)")
w("    ratio_ma20 = ratio.rolling(20).mean()")
w("    ratio_dev = ratio / ratio_ma20 - 1")
w("    f1_signal = tanh(ratio_dev * 30) * 0.5")
w()
w("  F2因子: 动量加速度 (f2=5.0)")
w("    X2原版: clip(-0.02, 0.02)")
w("    X4版:   clip(-0.025, 0.025)  ← 放宽")
w("    accel_diff = (g_accel - v_accel).clip(±clip_abs)")
w("    f2_signal = accel_diff * 5.0")
w("    style_score = f1_signal + f2_signal")
w()
w("  MA分段降仓 (X2方案A):")
w("    MA75双跌破 → 25% (MA75优先)")
w("    MA50双跌破 → 50%")
w("    否则       → 100%")
w()
w("  RSRS领先撤退层 (修正版RSRS = β × R²):")
w(f"    N={RSRS_N_DEFAULT} (OLS窗口), M={RSRS_M_DEFAULT} (标准化窗口)")
w("    对每个时点用最近N日 (high, low) 做 OLS: high = α + β×low + ε")
w("    rsrs = β × R²")
w("    z = (rsrs - rolling_mean(M)) / rolling_std(M)")
w("    z < -0.7 → rsrs_cut = 0.5 (中度降仓)")
w("    z < -1.0 → rsrs_cut = 0.2 (重度降仓)")
w("    z = NaN  → rsrs_cut = 1.0 (早期段不触发)")
w("    信号 shift(1) 防未来函数")
w("    对成长/价值分别计算，取 min(z_g, z_v) 作为更保守信号")
w()
w("  多信号取最保守仓位:")
w("    final_position = min(ma_position, rsrs_cut)")
w()
w("  high/low 处理:")
w("    数据 2012~2024-10-28 段 high/low 为空，用 max/min(close_t, close_{t-1}) 近似")
w("    2024-10-29 后使用真实 high/low")
w()
w("  回测参数:")
w("    初始资金: 1,000,000")
w("    佣金: 万1（双边）")
w("    冲击滑点: 0（指数无冲击）")
w("    跳空滑点: 关闭（指数 open=close）")
w("    调仓频率: 周频 W-FRI")
w("    Sharpe rf: 2.5%/年（与X1/X2口径一致）")
w()
w("-" * 78)
w("二、消融分析对比表")
w("-" * 78)
w()
w("=" * 78)
w("  X4 策略消融分析（vs X1/X2/X3基准）")
w("=" * 78)
w(f"  {'方案':<22}{'年化':>10}{'回撤':>10}{'Sharpe':>9}{'Calmar':>9}{'调仓次数':>10}")
w("  " + "-" * 72)
# X1/X2基线
w(f"  {'X1基线':<22}{m_x1['ann']*100:>9.2f}%{m_x1['dd']*100:>9.2f}%"
  f"{m_x1['sharpe']:>9.3f}{m_x1['calmar']:>9.3f}{m_x1['n_trades']:>10}")
w(f"  {'X2基线':<22}{m_x2['ann']*100:>9.2f}%{m_x2['dd']*100:>9.2f}%"
  f"{m_x2['sharpe']:>9.3f}{m_x2['calmar']:>9.3f}{m_x2['n_trades']:>10}")
# X3 文档基准
w(f"  {'X3-full(失败)':<22}{x3_doc['ann']*100:>9.2f}%{x3_doc['dd']*100:>9.2f}%"
  f"{x3_doc['sharpe']:>9.3f}{x3_doc['calmar']:>9.3f}{x3_doc['n_trades']:>10}")
for name in order:
    r, m = results[name]
    w(f"  {name:<22}{m['ann']*100:>9.2f}%{m['dd']*100:>9.2f}%"
      f"{m['sharpe']:>9.3f}{m['calmar']:>9.3f}{m['n_trades']:>10}")
w("=" * 78)
w()
w("  消融分析说明:")
w("    X1基线:    MA75双跌破→10%, direction='value'硬编码 (实现瑕疵)")
w("    X2基线:    clip±0.02 + MA分段降仓 + style_score方向 (无RSRS)")
w("    X3-full:   KST+连续加权+ADX+clip±0.025 (失败策略，文档基准)")
w("    X4-full:   clip±0.025 + MA分段降仓 + RSRS撤退层 (z<-0.7→50%, z<-1.0→20%)")
w("    X4-no-RSRS: clip±0.025 + MA分段降仓 (无RSRS) — 测试clip放宽独立贡献")
w("    X4-no-clip: clip±0.02原版 + MA分段降仓 + RSRS — 测试RSRS独立贡献")
w("    X4-mean:    clip±0.025 + RSRS z-mean聚合 (对比min vs mean)")
w()

# RSRS 压收益分析（任务Step 6）
w("-" * 78)
w("二(b)、关键问题：RSRS是否压收益？")
w("-" * 78)
w()
w(f"  X2基线:        年化={m_x2['ann']*100:.2f}%  回撤={m_x2['dd']*100:.2f}%")
w(f"  X4-no-RSRS:    年化={x4_no_rsrs_ann*100:.2f}%  回撤={x4_no_rsrs_dd*100:.2f}%  (X2基线+clip±0.025)")
w(f"  X4-full:       年化={x4_full_ann*100:.2f}%  回撤={x4_full_dd*100:.2f}%  (X2基线+clip±0.025+RSRS)")
w()
w(f"  RSRS增量贡献 (X4-full vs X4-no-RSRS):")
w(f"    年化:    {rsrs_delta_ann:+.2f}pp")
w(f"    回撤:    {rsrs_delta_dd:+.2f}pp  (正=回撤变小/改善)")
w(f"    Sharpe:  {rsrs_delta_sharpe:+.3f}")
w(f"    Calmar:  {rsrs_delta_calmar:+.3f}")
w()
w("  [结论]")
if x4_full_ann >= m_x2['ann'] - 0.005:
    w(f"    X4-full年化({x4_full_ann*100:.2f}%) ≥ X2基线({m_x2['ann']*100:.2f}%)")
    w(f"    → RSRS有效，不压收益，可作为正式版")
elif dd_improvement_pp >= 2.0 and tradeoff_ratio < 1.5:
    w(f"    X4-full年化({x4_full_ann*100:.2f}%) < X2基线({m_x2['ann']*100:.2f}%)")
    w(f"    但回撤显著改善 {dd_improvement_pp:.2f}pp（>=2pp阈值）")
    w(f"    权衡性价比: 年化损失{ann_loss_pp:.2f}pp / 回撤改善{dd_improvement_pp:.2f}pp = {tradeoff_ratio:.2f}（<1.5合理）")
    w(f"    → RSRS是'风控换收益'的合理权衡，可考虑采用")
elif dd_improvement_pp >= 2.0:
    w(f"    X4-full年化({x4_full_ann*100:.2f}%) < X2基线({m_x2['ann']*100:.2f}%)")
    w(f"    回撤改善 {dd_improvement_pp:.2f}pp，但权衡性价比差: {tradeoff_ratio:.2f}（>=1.5）")
    w(f"    → RSRS是'风控换收益'但不划算，不建议采用")
else:
    w(f"    X4-full年化({x4_full_ann*100:.2f}%) < X2基线({m_x2['ann']*100:.2f}%)")
    w(f"    且回撤改善仅 {dd_improvement_pp:.2f}pp（<2pp显著阈值）")
    w(f"    权衡性价比: 年化损失{ann_loss_pp:.2f}pp / 回撤改善{dd_improvement_pp:.2f}pp = {tradeoff_ratio:.2f}")
    w(f"    → RSRS无效，应移除")
w()
w("-" * 78)
w("三、RSRS 参数敏感性")
w("-" * 78)
w()
w("=" * 78)
w("  RSRS 参数敏感性（X4-full 配置: clip±0.025 + RSRS z-min, M=600）")
w("=" * 78)
w(f"  {'N值':<8}{'有效周数':>10}{'中度触发':>10}{'重度触发':>10}"
  f"{'触发频率':>10}{'年化':>10}{'回撤':>10}{'Sharpe':>9}{'Calmar':>9}")
w("  " + "-" * 76)
for _, row in sens_df.iterrows():
    w(f"  {int(row['N']):<8}{int(row['n_valid']):>10}{int(row['n_trig_mod']):>10}"
      f"{int(row['n_trig_sev']):>10}{row['trig_freq']:>9.1f}%"
      f"{row['ann']*100:>9.2f}%{row['dd']*100:>9.2f}%"
      f"{row['sharpe']:>9.3f}{row['calmar']:>9.3f}")
w("=" * 78)
w()
w("-" * 78)
w("四、最优 X4 配置选择")
w("-" * 78)
w()
w(f"  选择标准: 收益为先 + 风控为辅")
w(f"    1) 年化不低于X2基线超过0.5pp")
w(f"    2) 在剩余方案中按 Sharpe 排序")
w(f"    3) 若Sharpe差异<1%，取回撤更小者")
w()
w(f"  >>> 最优 X4 配置: {best_x4_name} (N={best_x4_n})")
w()
w("  X4最优 vs X2基线 vs X1基线 改进幅度:")
w(f"    {'指标':<10}{'X1基线':>12}{'X2基线':>12}{'X4最优':>12}{'X4vsX2':>12}")
w("    " + "-" * 56)
w(f"    {'年化':<10}{m_x1['ann']*100:>11.2f}%{m_x2['ann']*100:>11.2f}%"
  f"{best_x4_m['ann']*100:>11.2f}%{(best_x4_m['ann']-m_x2['ann'])*100:>+11.2f}pp")
w(f"    {'回撤':<10}{m_x1['dd']*100:>11.2f}%{m_x2['dd']*100:>11.2f}%"
  f"{best_x4_m['dd']*100:>11.2f}%{(best_x4_m['dd']-m_x2['dd'])*100:>+11.2f}pp")
w(f"    {'Sharpe':<10}{m_x1['sharpe']:>12.3f}{m_x2['sharpe']:>12.3f}"
  f"{best_x4_m['sharpe']:>12.3f}{best_x4_m['sharpe']-m_x2['sharpe']:>+12.3f}")
w(f"    {'Calmar':<10}{m_x1['calmar']:>12.3f}{m_x2['calmar']:>12.3f}"
  f"{best_x4_m['calmar']:>12.3f}{best_x4_m['calmar']-m_x2['calmar']:>+12.3f}")
w(f"    {'调仓次数':<10}{m_x1['n_trades']:>12d}{m_x2['n_trades']:>12d}"
  f"{best_x4_m['n_trades']:>12d}{best_x4_m['n_trades']-m_x2['n_trades']:>+12d}")
w()
w("-" * 78)
w("五、降仓统计")
w("-" * 78)
w()
w(f"  {'方案':<22}{'MA50触发':>10}{'MA75触发':>10}{'RSRS中度':>10}{'RSRS重度':>10}")
w("  " + "-" * 70)
for name in order:
    if name in weekly_stats:
        s = weekly_stats[name]
        w(f"  {name:<22}{s['n_ma50_cut']:>10}{s['n_ma75_cut']:>10}"
          f"{s['n_rsrs_moderate']:>10}{s['n_rsrs_severe']:>10}")
    else:
        w(f"  {name:<22}{'-':>10}{'-':>10}{'-':>10}{'-':>10}")
w()
w("-" * 78)
w("六、各方案详细 summary")
w("-" * 78)
for name in order:
    r, _ = results[name]
    w()
    w(r.summary(name))
w()
w("=" * 78)
w("  报告生成完毕")
w("=" * 78)

with open(REPORT_PATH, 'w', encoding='utf-8') as f:
    f.write("\n".join(report_lines))

print(f"\n[报告已保存] {REPORT_PATH}")
print("=" * 78)
print("[完成] X4 策略集成回测执行结束")
