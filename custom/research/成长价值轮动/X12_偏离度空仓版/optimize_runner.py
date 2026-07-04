"""optimize_runner.py — X12系列优化通用回测框架
================================================================
用于X13-X22各版本快速回测验证
每轮只需修改build_signal函数，其余回测/指标/记录逻辑复用
================================================================
"""
import sys
sys.path.insert(0, r'c:\XuanJLH\Qbot\custom\backtests')

from pathlib import Path
import numpy as np
import pandas as pd
import json
from backtest_engine import (
    BacktestInput, BacktestConfig,
    run_backtest_engine_weighted,
)

DATA_DIR = Path(r'c:\temp_v72_data')

# ============================================================
# 数据加载（全局缓存）
# ============================================================
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

# 公共因子（全局缓存）
RATIO = G_CLOSE / V_CLOSE
RATIO_MA20 = RATIO.rolling(20).mean()
RATIO_DEV = RATIO / RATIO_MA20 - 1
MA20_SLOPE = (RATIO_MA20 - RATIO_MA20.shift(5)) / RATIO_MA20.shift(5)
SLOPE_OK = MA20_SLOPE.abs() > 0.003
V_MOM20 = V_CLOSE.pct_change(20)
G_DD20 = G_CLOSE / G_CLOSE.shift(20) - 1
V_DD20 = V_CLOSE / V_CLOSE.shift(20) - 1
BASE_DIR = (RATIO > RATIO_MA20).map({True: 'growth', False: 'value'})
N_TOTAL = len(RATIO_DEV.dropna())


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
        'total_return': total
    }


def count_switches(signal, weight):
    """分析交易次数构成"""
    df = pd.DataFrame({'sig': signal, 'wt': weight}).dropna()
    df['pos'] = df['sig'] + '_' + df['wt'].round(2).astype(str)
    df['prev_pos'] = df['pos'].shift(1)
    switches = df[df['pos'] != df['prev_pos']].dropna()

    dir_sw = switches[
        ((switches['sig'] == 'growth') & (switches['prev_pos'].str.startswith('value')) |
         (switches['sig'] == 'value') & (switches['prev_pos'].str.startswith('growth')))
    ].shape[0]

    cash_sw = switches[
        ((switches['wt'].round(2) == 0.0) & (~switches['prev_pos'].str.endswith('0.0'))) |
        ((switches['wt'].round(2) != 0.0) & (switches['prev_pos'].str.endswith('0.0')))
    ].shape[0]

    return {'total': len(switches), 'dir': dir_sw, 'cash': cash_sw}


def test_strategy(name, build_func, params=None, desc=""):
    """测试一个策略变体并返回完整结果

    Args:
        name: 版本名（如X13）
        build_func: 策略构建函数，返回(signal, weight)
        params: 参数字典
        desc: 版本描述
    Returns:
        结果字典
    """
    if params is None:
        params = {}

    signal, weight = build_func(**params)
    result = run_backtest(signal, weight)
    m = calc_metrics(result)
    sw = count_switches(signal, weight)

    # 5/万滑点测试
    result_sl = run_backtest(signal, weight, impact_slippage=0.0005)
    m_sl = calc_metrics(result_sl)

    info = {
        'name': name,
        'desc': desc,
        'params': params,
        'ann': m['ann'],
        'dd': m['dd'],
        'sharpe': m['sharpe'],
        'calmar': m['calmar'],
        'n_trades': m['n_trades'],
        'total_switches': sw['total'],
        'dir_switches': sw['dir'],
        'cash_switches': sw['cash'],
        'ann_slippage5': m_sl['ann'],
        'calmar_slippage5': m_sl['calmar'],
    }

    return info


def print_result(info):
    """打印结果"""
    print(f"\n  {info['name']}: {info['desc']}")
    print(f"    年化={info['ann']*100:.2f}% 回撤={info['dd']*100:.2f}% "
          f"Sharpe={info['sharpe']:.3f} Calmar={info['calmar']:.3f}")
    print(f"    交易={info['n_trades']}次 (方向{info['dir_switches']}+空仓{info['cash_switches']})")
    print(f"    5/万滑点: 年化={info['ann_slippage5']*100:.2f}% Calmar={info['calmar_slippage5']:.3f}")
    return info


# ============================================================
# 基准策略定义
# ============================================================
def build_x11a():
    """X11-A基准: F1+A1(4天)+斜率+B2+E5"""
    dir_s = BASE_DIR.copy()
    mask = np.ones(len(dir_s), dtype=bool)
    for k in range(1, 4):
        mask = mask & (dir_s.values == dir_s.shift(k).values)
    confirmed = dir_s.where(mask, np.nan)
    dir_s = confirmed.where(SLOPE_OK, np.nan).ffill()
    wrong_value = (dir_s == 'value') & (V_MOM20 <= 0)
    dir_s[wrong_value] = 'growth'
    wt = pd.Series(1.0, index=dir_s.index)
    gs = (dir_s == 'growth') & (G_DD20 < -0.10)
    vs = (dir_s == 'value') & (V_DD20 < -0.10)
    wt[gs | vs] = 0.3
    return dir_s, wt


def build_x12(dev_threshold=0.005):
    """X12基准: F1+F0(偏离度空仓)+A1+斜率+B2+E5"""
    dir_s = BASE_DIR.copy()
    wt = pd.Series(1.0, index=dir_s.index)
    low_dev = RATIO_DEV.abs() < dev_threshold
    wt[low_dev] = 0.0
    mask = np.ones(len(dir_s), dtype=bool)
    for k in range(1, 4):
        mask = mask & (dir_s.values == dir_s.shift(k).values)
    confirmed = dir_s.where(mask, np.nan)
    dir_s = confirmed.where(SLOPE_OK, np.nan).ffill()
    wrong_value = (dir_s == 'value') & (V_MOM20 <= 0)
    dir_s[wrong_value] = 'growth'
    gs = (dir_s == 'growth') & (G_DD20 < -0.10)
    vs = (dir_s == 'value') & (V_DD20 < -0.10)
    wt[gs | vs] = wt[gs | vs] * 0.30
    return dir_s, wt


if __name__ == '__main__':
    # 验证框架
    print("=" * 78)
    print("  优化框架验证")
    print("=" * 78)

    print("\n  [基准] X11-A:")
    info = test_strategy("X11-A", build_x11a, desc="F1+A1+斜率+B2+E5")
    print_result(info)

    print("\n  [基准] X12(0.5%):")
    info = test_strategy("X12", build_x12, {'dev_threshold': 0.005}, "F1+F0(0.5%)+A1+斜率+B2+E5")
    print_result(info)

    print("\n  框架验证通过，可开始逐轮优化")
