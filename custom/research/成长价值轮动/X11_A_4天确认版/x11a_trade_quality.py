"""x11a_trade_quality.py — X11-A(4天)调仓质量补算
================================================================
B研究员保留意见：补算调仓胜率、盈亏比、短持仓占比
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

DATA_DIR = Path(r'c:\temp_v72_data')

# 加载数据
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

# 信号
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

# 4天确认
dir_s = base_dir.copy()
mask4 = np.ones(len(dir_s), dtype=bool)
for k in range(1, 4):
    mask4 = mask4 & (dir_s.values == dir_s.shift(k).values)
confirmed = dir_s.where(mask4, np.nan)
x11a_dir = confirmed.where(slope_ok, np.nan).ffill()
wrong_value = (x11a_dir == 'value') & (v_mom20 <= 0)
x11a_dir[wrong_value] = 'growth'
x11a_wt = pd.Series(1.0, index=x11a_dir.index)
gs = (x11a_dir == 'growth') & (g_dd20 < -0.10)
vs = (x11a_dir == 'value') & (v_dd20 < -0.10)
x11a_wt[gs | vs] = 0.3

# 回测
common_idx = x11a_dir.index.intersection(g_close.index)
sig = x11a_dir.loc[common_idx]
wt = x11a_wt.loc[common_idx]
g_a = g_close.loc[common_idx]
v_a = v_close.loc[common_idx]
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

# 构建持仓段
trades_df = result.trades_to_dataframe()
df_daily = result.to_dataframe()
df_daily['date'] = pd.to_datetime(df_daily['date'])
df_daily = df_daily.set_index('date')

trade_records = [{'trade_date': pd.to_datetime(r['trade_date']),
                  'position': r['position']} for _, r in trades_df.iterrows()]
trades = pd.DataFrame(trade_records).sort_values('trade_date').reset_index(drop=True)

holdings = []
for i in range(len(trades) - 1):
    sd = trades.loc[i, 'trade_date']
    ed = trades.loc[i + 1, 'trade_date']
    pos = trades.loc[i, 'position']
    sm = (df_daily.index >= sd) & (df_daily.index < ed)
    sr = df_daily.loc[sm, 'daily_ret']
    if len(sr) > 0:
        holdings.append({
            'start_date': sd, 'end_date': ed, 'position': pos,
            'hold_days': len(sr),
            'segment_return': (1 + sr).prod() - 1,
            'win': (1 + sr).prod() - 1 > 0,
        })

holdings_df = pd.DataFrame(holdings)
total = len(holdings_df)
win_rate = holdings_df['win'].mean() * 100
avg_win = holdings_df.loc[holdings_df['win'], 'segment_return'].mean()
avg_loss = holdings_df.loc[~holdings_df['win'], 'segment_return'].mean()
pl_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0

print("=" * 70)
print("  X11-A(4天确认) 调仓质量补算")
print("=" * 70)
print(f"\n总持仓段数: {total}")
print(f"调仓胜率: {win_rate:.1f}%")
print(f"平均盈利: {avg_win*100:+.2f}%")
print(f"平均亏损: {avg_loss*100:+.2f}%")
print(f"盈亏比: {pl_ratio:.2f}:1")

# 持仓时长
holdings_df['hold_bucket'] = pd.cut(holdings_df['hold_days'],
                                      bins=[0, 7, 14, 30, 60, 999],
                                      labels=['≤7天', '8-14天', '15-30天', '31-60天', '>60天'])
print("\n持仓时长分布:")
for bucket in ['≤7天', '8-14天', '15-30天', '31-60天', '>60天']:
    bdf = holdings_df[holdings_df['hold_bucket'] == bucket]
    if len(bdf) > 0:
        bwr = bdf['win'].mean() * 100
        bar = bdf['segment_return'].mean()
        print(f"  {bucket}: {len(bdf)}段({len(bdf)/total*100:.1f}%), 胜率{bwr:.1f}%, 平均{bar*100:+.2f}%")

# 方向对比
print("\n方向对比:")
for pos in ['growth', 'value']:
    pdf = holdings_df[holdings_df['position'] == pos]
    if len(pdf) > 0:
        pwr = pdf['win'].mean() * 100
        par = pdf['segment_return'].mean()
        print(f"  {pos}: {len(pdf)}段, 胜率{pwr:.1f}%, 平均{par*100:+.2f}%")

# B研究员判定标准
print("\n" + "=" * 70)
print("  B研究员判定标准检查")
print("=" * 70)
print(f"  调仓胜率 {win_rate:.1f}% {'≥50% ✅' if win_rate >= 50 else '<50% ❌'}")
print(f"  盈亏比 {pl_ratio:.2f}:1 {'≥1.3 ✅' if pl_ratio >= 1.3 else '<1.3 ❌'}")
short_pct = (holdings_df['hold_bucket'] == '≤7天').mean() * 100
print(f"  短持仓占比 {short_pct:.1f}% {'≤40% ✅' if short_pct <= 40 else '>40% ❌'}")
