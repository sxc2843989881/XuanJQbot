"""
V72 新旧引擎差异分析
======================
精确对比：原引擎(style_rotation_strategy) vs Qbot backtrader 引擎
找出 31.90% → 28.33% 差异的根源
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
DATA_DIR = Path(r"c:\caches\sxc\style_rotation_strategy\data")

# 加入原项目路径
STYLE_DIR = Path(r"c:\caches\sxc\style_rotation_strategy")
sys.path.insert(0, str(STYLE_DIR))

from backtest_module.backtest_engine import run_backtest_engine_weighted, BacktestInput, BacktestConfig


def load_close(csv_name):
    df = pd.read_csv(DATA_DIR / csv_name)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').sort_index()
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df['open'] = pd.to_numeric(df.get('open', pd.Series(index=df.index)), errors='coerce')
    df['open'] = df['open'].fillna(df['close'].shift(1))
    df.loc[df.index[0], 'open'] = df['close'].iloc[0]
    return df[['open', 'close']]


print("=" * 70)
print("V72 新旧引擎差异精确分析")
print("=" * 70)

# 加载数据
df_g = load_close("index_480080.csv")
df_v = load_close("index_480081.csv")
common = df_g.index.intersection(df_v.index)
df_g = df_g.loc[common]
df_v = df_v.loc[common]
df_g = df_g["2013":"2025"]
df_v = df_v["2013":"2025"]

g = df_g['close']
v = df_v['close']
g_open = df_g['open']
v_open = df_v['open']

print(f"\n数据: {len(g)} 天, {g.index[0].date()} ~ {g.index[-1].date()}")

# ============================================================
# A. 原引擎信号（精确复刻）
# ============================================================
# 原引擎的因子公式（来自 _version_comparison.py）
F1 = 0.5; F2 = 5.0; MA_W = 75; MA_PCT = 0.97; CUT = 0.1; MIN_HOLD = 4

ratio = (g / v).shift(1)
ratio_ma20 = ratio.rolling(20).mean()
ratio_dev = ratio / ratio_ma20 - 1
f1 = np.tanh(ratio_dev * 30) * F1

g_roc21 = g.pct_change(21).shift(1)
v_roc21 = v.pct_change(21).shift(1)
g_accel = g_roc21 - g_roc21.shift(10)
v_accel = v_roc21 - v_roc21.shift(10)
f2 = np.clip(g_accel - v_accel, -0.02, 0.02) * F2

score_orig = f1 + f2

def rolling_slope_r2(close_s, window=63):
    y = np.log(close_s.astype(float))
    n = window
    x = np.arange(n).astype(float)
    x_mean = x.mean()
    x_var = ((x - x_mean) ** 2).sum()
    sum_xy = y.rolling(n).apply(lambda yw: np.sum(yw * x), raw=True)
    sum_y = y.rolling(n).sum()
    sum_y2 = y.rolling(n).apply(lambda yw: np.sum(yw * yw), raw=True)
    cov_num = sum_xy - n * x_mean * (sum_y / n)
    slope = cov_num / x_var
    y_var = sum_y2 - (sum_y ** 2) / n
    r2 = (slope ** 2 * x_var) / y_var.where(y_var != 0, np.nan)
    r2 = r2.clip(0, 1)
    return slope, r2

sg_raw, r2_g = rolling_slope_r2(g, 63)
sv_raw, r2_v = rolling_slope_r2(v, 63)
smom_g = sg_raw * r2_g
smom_v = sv_raw * r2_v

# MA75
g_ma75 = g.shift(1).rolling(MA_W).mean()
v_ma75 = v.shift(1).rolling(MA_W).mean()
both_below = (g.shift(1) < g_ma75 * MA_PCT) & (v.shift(1) < v_ma75 * MA_PCT)

# ============================================================
# B. Backtrader 信号（精确复刻策略的 iloc[-2] 逻辑）
# ============================================================
# backtrader 里每次周五决策，用到该周五收盘价的 "前一天" 数据
# 即：在 weekly 频率下，每个周五的 close 收集后，用 iloc[-2]
# 这相当于用周四的收盘价做决策

ratio_bt = g / v  # no shift
ratio_ma20_bt = ratio_bt.rolling(20).mean()
# 在周五决策，用 iloc[-2] 相当于周四的 close
# 我们这里构造一个 week-by-week 的信号
df_wk_orig = pd.DataFrame({
    'g': g, 'v': v,
    'g_open': g_open, 'v_open': v_open,
    'score_orig': score_orig,
    'smom_g': smom_g, 'smom_v': smom_v,
    'both_below': both_below,
})
df_wk_orig = df_wk_orig.resample('W-FRI').last()

# backtrader 在周五收集 close，然后用 iloc[-2]（周四）
# 所以我们获取每周四的数据
df_thu = pd.DataFrame({
    'g_thu': g.shift(1),  # 周五的"前一天"是周四
    'v_thu': v.shift(1),
    'score_thu': score_orig.shift(1),  # shift(1) 的 score 相当于用周四收盘的数据
    'smom_g_thu': smom_g.shift(1),
    'smom_v_thu': smom_v.shift(1),
    'bb_thu': both_below.shift(1),
})
df_thu_wk = df_thu.resample('W-FRI').last()

print(f"\n原引擎周数: {len(df_wk_orig.dropna(subset=['score_orig']))}")
print(f"BT等效周数: {len(df_thu_wk.dropna(subset=['score_thu']))}")

# ============================================================
# C. 信号生成
# ============================================================
def generate_signals(df_wk_input, score_col, smom_g_col, smom_v_col, bb_col):
    """统一信号生成逻辑"""
    pos = pd.Series(np.nan, index=df_wk_input.index)
    wgt = pd.Series(1.0, index=df_wk_input.index)
    cur = None
    hw = 0
    for i in range(len(df_wk_input)):
        row = df_wk_input.iloc[i]
        sc = row[score_col]
        sg = row[smom_g_col]
        sv = row[smom_v_col]
        bb = bool(row[bb_col])

        if pd.isna(sc) or pd.isna(sg) or pd.isna(sv):
            continue

        if sc > 0:
            decided = 'growth'
        else:
            if cur == 'growth' and sg > 0 and sv <= 0:
                decided = 'growth'
            elif cur is None:
                decided = 'growth' if sg > sv else 'value'
            else:
                decided = 'value'

        if bb:
            tw = CUT
            tp = cur if cur else decided
        else:
            tw = 1.0
            tp = decided

        if cur is None:
            cur = tp; hw = 1; cw = tw
        elif tp != cur and hw >= MIN_HOLD:
            cur = tp; hw = 1; cw = tw
        else:
            cw = tw; hw += 1

        pos.iloc[i] = 1.0 if cur == 'growth' else 0.0
        wgt.iloc[i] = cw
    return pos, wgt


# 1. 原引擎信号
pos_orig, wgt_orig = generate_signals(
    df_wk_orig, 'score_orig', 'smom_g', 'smom_v', 'both_below'
)

# 2. Backtrader 等效信号
pos_bt, wgt_bt = generate_signals(
    df_thu_wk, 'score_thu', 'smom_g_thu', 'smom_v_thu', 'bb_thu'
)

# 对齐
common_idx = pos_orig.dropna().index.intersection(pos_bt.dropna().index)
print(f"\n原引擎有信号的周: {pos_orig.dropna().sum()} / {len(pos_orig.dropna())}")
print(f"BT等效有信号的周: {pos_bt.dropna().sum()} / {len(pos_bt.dropna())}")
print(f"共同有效周: {len(common_idx)}")

# ============================================================
# D. 信号一致性对比
# ============================================================
print("\n" + "=" * 70)
print("D. 信号一致性对比")
print("=" * 70)

# 直接对比
po = pos_orig.reindex(common_idx)
pb = pos_bt.reindex(common_idx)
wo = wgt_orig.reindex(common_idx)
wb = wgt_bt.reindex(common_idx)

match = (po == pb).sum()
total = len(common_idx)
print(f"持仓一致率: {match/total*100:.2f}% ({match}/{total})")

diff_mask = po != pb
if diff_mask.any():
    print(f"\n不一致周 ({diff_mask.sum()}个):")
    for d in common_idx[diff_mask][:10]:
        po_v = po.loc[d]
        pb_v = pb.loc[d]
        row_orig = df_wk_orig.loc[d]
        row_bt = df_thu_wk.loc[d]
        print(f"\n  {d.date()}:")
        print(f"    原引擎: pos={'成长' if po_v==1 else '价值'}, score={row_orig['score_orig']:.4f}, smom_g={row_orig['smom_g']:.4f}, smom_v={row_orig['smom_v']:.4f}, bb={row_orig['both_below']}")
        print(f"    BT等效: pos={'成长' if pb_v==1 else '价值'}, score={row_bt['score_thu']:.4f}, smom_g={row_bt['smom_g_thu']:.4f}, smom_v={row_bt['smom_v_thu']:.4f}, bb={row_bt['bb_thu']}")

# ============================================================
# E. 回测对比
# ============================================================
print("\n" + "=" * 70)
print("E. 回测对比（同一引擎 + 不同信号）")
print("=" * 70)

def backtest_signal(pos_series, wgt_series, label):
    """用原引擎跑回测，隔离信号差异 vs 引擎差异"""
    # 合并到日频
    df_daily = pd.DataFrame(index=g.index)
    df_daily['g_open'] = g_open
    df_daily['v_open'] = v_open
    df_daily['g_close'] = g
    df_daily['v_close'] = v

    sig_df = pd.DataFrame({'sig': pos_series.map({1.0: 'growth', 0.0: 'value'}), 'wgt': wgt_series})
    df_daily = pd.merge_asof(df_daily, sig_df, left_index=True, right_index=True, direction='backward')
    df_daily = df_daily.dropna(subset=['sig'])

    bt_input = BacktestInput(
        dates=df_daily.index.strftime('%Y-%m-%d').values,
        value_open=df_daily['v_open'].values.astype(np.float64),
        value_close=df_daily['v_close'].values.astype(np.float64),
        growth_open=df_daily['g_open'].values.astype(np.float64),
        growth_close=df_daily['g_close'].values.astype(np.float64),
        signal=df_daily['sig'].values.astype(str),
    )
    result = run_backtest_engine_weighted(bt_input, BacktestConfig(commission=0.0001),
                                          df_daily['wgt'].values.astype(np.float64))
    m = result.metrics
    print(f"\n  [{label}]:")
    print(f"    年化: {m['annual_ret']*100:.2f}%")
    print(f"    夏普: {m['sharpe']:.3f}")
    print(f"    回撤: {m['max_dd']*100:.2f}%")
    print(f"    Calmar: {m['calmar']:.3f}")
    print(f"    交易次数: {m['num_trades']}")
    print(f"    最终净值: {m['final_nav']:,.0f}")

# 用共同有效周跑
backtest_signal(po.loc[common_idx], wo.loc[common_idx], "原引擎信号")
backtest_signal(pb.loc[common_idx], wb.loc[common_idx], "BT等效信号")

# ============================================================
# F. 引擎本身差异
# ============================================================
print("\n" + "=" * 70)
print("F. 引擎本身差异")
print("=" * 70)
print("""
原引擎 vs backtrader 的关键区别：
1. 订单执行：原引擎用开盘价成交，backtrader 默认用下一日开盘价
2. 资金管理：原引擎用 exact 比例，backtrader 用 order_target_percent
3. 交易成本处理方式不同
4. 小数股处理（backtrader 支持小数股，原引擎可能取整）
""")

# ============================================================
# G. 启动时机差异
# ============================================================
print("=" * 70)
print("G. 启动时机差异")
print("=" * 70)

# 原引擎：score 和 smom 都有值的第一周
first_valid_orig = df_wk_orig['score_orig'].dropna().index[0]
first_valid_bt = df_thu_wk['score_thu'].dropna().index[0]
print(f"原引擎第一有效周: {first_valid_orig.date()}")
print(f"BT等效第一有效周: {first_valid_bt.date()}")
print(f"差异: {(first_valid_bt - first_valid_orig).days} 天")

# 对比从第一有效周到结束
po_align = pos_orig.loc[first_valid_orig:]
pb_align = pos_bt.loc[first_valid_bt:]
ci = po_align.index.intersection(pb_align.index)
print(f"从各自启动到结束的共同周: {len(ci)}")
