"""backtest_x13_engine.py — X13 统一策略最优版
================================================================
归档自 unified_strategy 第26轮(v26)和第27轮(v27)

v26 U164  (无滑点最优): Calmar 2.378  年化49.79% 回撤-20.94%
v27 U175  (滑点最优):   滑点Calmar 2.059

核心创新:
  1. T指标统一: T = RATIO_DEV_Z = (RATIO-RATIO_MA20)/RATIO_DEV_STD20
  2. T+斜率双重确认: |T|<rt AND |斜率|<sl → 空仓
  3. BIAS过滤(Br=0): G/V_BIAS>0.19 → 完全空仓
  4. 持仓时间限制(MHr=0): 持仓>=92天 → 完全空仓
  5. B2改进: V_MOM10<=0 AND V_MOM20<=0 → 改growth
  6. E5止损: 20日跌幅>st → 降仓sw, cd天冷却
  7. 方向确认(dc)+方向冷却(dcd)

依赖:
  - custom/backtests (backtest_engine.py)
  - temp_v72_data/index_480080.csv, index_480081.csv
================================================================
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# ─── 路径 ─────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'backtests'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from backtest_engine import (
    BacktestInput, BacktestConfig, run_backtest_engine_weighted,
)

DATA_DIR = Path(r'c:\temp_v72_data')

# ─── 数据加载 ────────────────────────────────────────
_g_raw = pd.read_csv(str(DATA_DIR / 'index_480080.csv'))
_v_raw = pd.read_csv(str(DATA_DIR / 'index_480081.csv'))
for d in (_g_raw, _v_raw):
    d['date'] = pd.to_datetime(d['date'])
    d['close'] = pd.to_numeric(d['close'], errors='coerce')
G_CLOSE = _g_raw.set_index('date')['close'].astype(float).sort_index().dropna()
V_CLOSE = _v_raw.set_index('date')['close'].astype(float).sort_index().dropna()
_common = G_CLOSE.index.intersection(V_CLOSE.index)
G_CLOSE = G_CLOSE[_common].sort_index()
V_CLOSE = V_CLOSE[_common].sort_index()

# 公共因子
RATIO = G_CLOSE / V_CLOSE
RATIO_MA20 = RATIO.rolling(20).mean()
RATIO_DEV = RATIO / RATIO_MA20 - 1
RATIO_DEV_STD20 = RATIO_DEV.rolling(20).std()
RATIO_DEV_Z = RATIO_DEV / RATIO_DEV_STD20
MA20_SLOPE = (RATIO_MA20 - RATIO_MA20.shift(5)) / RATIO_MA20.shift(5)
V_MOM10 = V_CLOSE.pct_change(10)
V_MOM20 = V_CLOSE.pct_change(20)
G_DD20 = G_CLOSE / G_CLOSE.shift(20) - 1
V_DD20 = V_CLOSE / V_CLOSE.shift(20) - 1

T = RATIO_DEV_Z
SLOPE = MA20_SLOPE


# ============================================================
# 核心策略: build_core
# ============================================================
def build_core(slope_thresh=0.002, sw=0.17, st=0.088, cd=8,
               ms=10, ml=20, rt=1.3, dc=5, dcd=6,
               bias_ma=20, bias_high=0.19, bias_reduce=0.0,
               use_max_hold=True, max_hold_days=92, max_hold_reduce=0.0,
               hold_mode='reset_dir'):
    """X13 统一策略最优核心

    参数:
      slope_thresh: 斜率弱阈值
      sw: E5降仓权重 (0-1)
      st: E5止损阈值 (正值, 如0.088=8.8%跌幅)
      cd: E5冷却天数
      ms, ml: B2短/长动量周期
      rt: T弱阈值 (|T|<rt视为weak)
      dc: 方向确认天数
      dcd: 方向冷却天数
      bias_ma: BIAS计算均线周期
      bias_high: BIAS超阈值
      bias_reduce: BIAS降仓权重 (0=完全空仓)
      max_hold_days: 最大持仓天数
      max_hold_reduce: 持仓超期降仓权重 (0=完全空仓)
      hold_mode: 持仓计数重置模式 (reset_dir/reset_pos)
    """
    G_MA = G_CLOSE.rolling(bias_ma).mean()
    V_MA = V_CLOSE.rolling(bias_ma).mean()
    G_BIAS = (G_CLOSE / G_MA - 1)
    V_BIAS = (V_CLOSE / V_MA - 1)

    V_MOM_S = V_CLOSE.pct_change(ms)
    V_MOM_L = V_CLOSE.pct_change(ml)

    # ── 方向确认 ──
    raw_dir = (T > 0).map({True: 'BULL', False: 'BEAR'})
    mask = np.ones(len(raw_dir), dtype=bool)
    for k in range(1, dc):
        mask = mask & (raw_dir.values == raw_dir.shift(k).values)
    confirmed_dir = raw_dir.where(mask, np.nan).ffill().fillna('BULL')

    # ── 方向冷却 ──
    if dcd > 0:
        new_dir = confirmed_dir.copy()
        last_switch = -dcd - 1
        prev = confirmed_dir.iloc[0]
        for i in range(len(confirmed_dir)):
            if pd.isna(confirmed_dir.iloc[i]):
                new_dir.iloc[i] = prev
                continue
            if confirmed_dir.iloc[i] != prev:
                if i - last_switch >= dcd:
                    last_switch = i
                    prev = confirmed_dir.iloc[i]
                new_dir.iloc[i] = prev
            else:
                new_dir.iloc[i] = prev
        confirmed_dir = new_dir

    dir_raw = confirmed_dir

    # ── T+斜率双重确认 → 空仓 ──
    weak_slope = SLOPE.abs() < slope_thresh
    weak_t = T.abs() < rt
    is_weak = weak_t & weak_slope
    wt = pd.Series(1.0, index=T.index)
    wt[is_weak] = 0.0

    # ── B2: value方向双动量过滤 ──
    wrong_value = (dir_raw == 'BEAR') & (V_MOM_S <= 0) & (V_MOM_L <= 0)
    dir_raw[wrong_value] = 'BULL'
    dir_s = dir_raw.map({'BULL': 'growth', 'BEAR': 'value'})

    # ── BIAS过滤 (Br=0完全空仓) ──
    extreme_g = (dir_s == 'growth') & (G_BIAS > bias_high)
    extreme_v = (dir_s == 'value') & (V_BIAS > bias_high)
    extreme = extreme_g | extreme_v
    wt[extreme] = wt[extreme] * bias_reduce

    # ── E5止损 ──
    gs = (dir_s == 'growth') & (G_DD20 < -st)
    vs = (dir_s == 'value') & (V_DD20 < -st)
    e5_trigger = gs | vs

    in_cooldown = False
    cooldown_count = 0
    for i in range(len(wt)):
        if pd.isna(wt.iloc[i]) or pd.isna(dir_s.iloc[i]):
            continue
        if e5_trigger.iloc[i] and not in_cooldown:
            in_cooldown = True
            cooldown_count = 0
            wt.iloc[i] = wt.iloc[i] * sw
        elif in_cooldown:
            cooldown_count += 1
            if cooldown_count >= cd:
                if e5_trigger.iloc[i]:
                    cooldown_count = 0
                    wt.iloc[i] = wt.iloc[i] * sw
                else:
                    in_cooldown = False
                    if is_weak.iloc[i]:
                        wt.iloc[i] = 0.0
                    else:
                        wt.iloc[i] = 1.0
            else:
                if wt.iloc[i] > 0:
                    wt.iloc[i] = sw

    # ── 持仓时间限制 (MHr=0完全空仓) ──
    if use_max_hold:
        hold_count = 0
        prev_key = None
        for i in range(len(wt)):
            if pd.isna(wt.iloc[i]) or pd.isna(dir_s.iloc[i]):
                continue
            if hold_mode == 'reset_dir':
                key = dir_s.iloc[i]
            elif hold_mode == 'reset_pos':
                key = (dir_s.iloc[i], round(wt.iloc[i], 2))
            else:
                key = None

            if key != prev_key:
                hold_count = 0
                prev_key = key
            else:
                hold_count += 1
                if hold_count >= max_hold_days:
                    if wt.iloc[i] > 0:
                        wt.iloc[i] = wt.iloc[i] * max_hold_reduce

    signal = dir_raw.map({'BULL': 'growth', 'BEAR': 'value'})
    return signal, wt


# ============================================================
# 回测函数
# ============================================================
def run_backtest(signal, weight, impact_slippage=0.0):
    """通用回测函数"""
    common_idx = signal.index.intersection(G_CLOSE.index)
    sig = signal.loc[common_idx]
    wt = weight.loc[common_idx]
    g_a = G_CLOSE.loc[common_idx]
    v_a = V_CLOSE.loc[common_idx]
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
                            impact_slippage=impact_slippage, apply_gap_slippage=False)
    return run_backtest_engine_weighted(bt_input, config, wt.values)


def calc_metrics(result, freq=252, rf_annual=0.025):
    """计算核心指标"""
    df_r = result.to_dataframe()
    r = df_r['daily_ret']
    eq = (1 + r).cumprod()
    n = len(r)
    years = n / freq
    total = eq.iloc[-1] - 1
    ann = (1 + total) ** (1 / years) - 1 if years > 0 else 0
    rf_p = rf_annual / freq
    sharpe = (r.mean() - rf_p) / r.std() * np.sqrt(freq) if r.std() > 0 else 0
    peak = eq.cummax()
    dd = (eq - peak) / peak
    max_dd = dd.min()
    calmar = ann / abs(max_dd) if max_dd < 0 else 0
    return {
        'ann': ann, 'dd': max_dd, 'sharpe': sharpe, 'calmar': calmar,
        'n_trades': result.metrics['num_trades'],
        'total_return': total,
    }


# ============================================================
# 预置参数
# ============================================================
V26_PARAMS = dict(
    slope_thresh=0.002, sw=0.17, st=0.088, cd=8,
    ms=10, ml=20, rt=1.3, dc=5, dcd=6,
    bias_ma=20, bias_high=0.19, bias_reduce=0.0,
    max_hold_days=92, max_hold_reduce=0.0,
)
"""v26 U164 (无滑点最优): Calmar 2.378 年化49.79% 回撤-20.94%"""

V27_PARAMS = dict(
    slope_thresh=0.002, sw=0.16, st=0.088, cd=9,
    ms=10, ml=20, rt=1.3, dc=5, dcd=6,
    bias_ma=20, bias_high=0.19, bias_reduce=0.0,
    max_hold_days=92, max_hold_reduce=0.0,
)
"""v27 U175 (滑点最优): 滑点Calmar 2.059"""


# ============================================================
# 主入口
# ============================================================
if __name__ == '__main__':
    print("=" * 80)
    print("  X13 统一策略最优版 回测验证")
    print("=" * 80)

    for tag, params in [('v26 U164 (无滑点最优)', V26_PARAMS),
                         ('v27 U175 (滑点最优)', V27_PARAMS)]:
        print(f"\n--- {tag} ---")
        sig, wt = build_core(**params)
        result = run_backtest(sig, wt)
        m = calc_metrics(result)
        result_sl = run_backtest(sig, wt, impact_slippage=0.0005)
        m_sl = calc_metrics(result_sl)
        print(f"  年化={m['ann']*100:.2f}%  回撤={m['dd']*100:.2f}%  "
              f"Sharpe={m['sharpe']:.3f}  Calmar={m['calmar']:.3f}  "
              f"交易={m['n_trades']}")
        print(f"  5/万滑点: 年化={m_sl['ann']*100:.2f}%  "
              f"Calmar={m_sl['calmar']:.3f}")

    print(f"\n  X13 归档版本验证通过")
    print("=" * 80)
