"""analyze_x12_logic.py — X12机制逻辑分析
================================================================
回答用户的质疑：
  1. 交易次数为何从220→637暴增？F0空仓切换频率如何？
  2. F0空仓与斜率确认是否重复/矛盾？
  3. F0空仓与B2过滤是否冗余？
  4. 偏离度0.3%阈值的详细结果
  5. 滞回机制能否解决交易过多问题？
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

# ============================================================
# 1. 加载数据 & 预计算因子
# ============================================================
print("=" * 78)
print("  X12机制逻辑深度分析")
print("=" * 78)

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
# 分析1: F0空仓触发统计 — 为什么交易次数暴增？
# ============================================================
print("\n" + "=" * 78)
print("  分析1: F0空仓触发统计（偏离度0.5%阈值）")
print("=" * 78)

low_dev_05 = ratio_dev.abs() < 0.005
n_low_dev = low_dev_05.sum()
n_total = len(ratio_dev.dropna())
print(f"  偏离度<0.5%的总天数: {n_low_dev} / {n_total} ({n_low_dev/n_total*100:.1f}%)")

# 连续空仓段统计
low_dev_series = low_dev_05.astype(int)
# 找连续段的起始和结束
diff = low_dev_series.diff().fillna(0)
starts = diff[diff == 1].index  # 0→1的转折点
ends = diff[diff == -1].index   # 1→0的转折点
# 处理首尾
if low_dev_series.iloc[0] == 1:
    starts = starts.insert(0, low_dev_series.index[0])
if low_dev_series.iloc[-1] == 1:
    ends = ends.insert(len(ends), low_dev_series.index[-1])

segments = []
for s, e in zip(starts, ends):
    seg_days = (e - s).days + 1
    segments.append(seg_days)

print(f"  空仓触发段数: {len(segments)}")
print(f"  平均持续天数: {np.mean(segments):.1f}天")
print(f"  中位持续天数: {np.median(segments):.0f}天")
print(f"  最长持续: {max(segments)}天, 最短持续: {min(segments)}天")

# 短段统计（1-2天的短切换）
short_segs = sum(1 for s in segments if s <= 2)
print(f"  ≤2天的短切换段: {short_segs} ({short_segs/len(segments)*100:.1f}%)")
print(f"  → 这些短切换是交易次数暴增的主因")

# 对比X11-A和X12的交易次数
print(f"\n  X11-A交易次数: 220次（仅方向切换growth↔value）")
print(f"  X12交易次数: 637次（方向切换 + 满仓↔空仓切换）")
print(f"  净增: 417次，主要来自F0空仓的{len(segments)}次切换")

# ============================================================
# 分析2: F0空仓与斜率确认的重叠度
# ============================================================
print("\n" + "=" * 78)
print("  分析2: F0空仓 vs 斜率确认 — 是否重复/矛盾？")
print("=" * 78)

# F0触发条件: |ratio_dev| < 0.5%
# 斜率确认条件: |ma20_slope| > 0.3%
# 注意：斜率确认不满足时(slope_ok=False)，方向被ffill（保持原方向）

f0_trigger = ratio_dev.abs() < 0.005          # F0空仓触发
slope_fail = ~slope_ok                           # 斜率不满足（保持原方向）

# 重叠分析
both = (f0_trigger & slope_fail).sum()           # F0触发 且 斜率不满足
f0_only = (f0_trigger & ~slope_fail).sum()       # F0触发 但 斜率满足
slope_only = (~f0_trigger & slope_fail).sum()    # F0不触发 但 斜率不满足
neither = (~f0_trigger & ~slope_fail).sum()      # 都不触发

print(f"  {'情况':<35} {'天数':>8} {'占比':>8}")
print(f"  {'-'*55}")
print(f"  {'F0触发 且 斜率不满足(重叠)':<35} {both:>8} {both/n_total*100:>7.1f}%")
print(f"  {'F0触发 但 斜率满足(F0独立)':<35} {f0_only:>8} {f0_only/n_total*100:>7.1f}%")
print(f"  {'F0不触发 但 斜率不满足(斜率独立)':<35} {slope_only:>8} {slope_only/n_total*100:>7.1f}%")
print(f"  {'都不触发':<35} {neither:>8} {neither/n_total*100:>7.1f}%")

overlap_rate = both / f0_trigger.sum() * 100 if f0_trigger.sum() > 0 else 0
print(f"\n  F0触发时斜率也不满足的比例: {overlap_rate:.1f}%")
print(f"  → 重叠率{'高(冗余)' if overlap_rate > 60 else '中等(部分重叠)' if overlap_rate > 30 else '低(独立)'}")

# 逻辑矛盾分析
print(f"\n  逻辑矛盾分析:")
print(f"    F0逻辑: 偏离度小→趋势不明确→空仓(weight=0)")
print(f"    斜率逻辑: 斜率小→趋势不明确→保持原方向(ffill)")
print(f"    矛盾点: 同样是'趋势不明确'，F0选择空仓，斜率选择维持方向")
print(f"    当F0触发但斜率满足时({f0_only}天): F0说'趋势不明确要空仓'，斜率说'趋势明确可以交易'")
print(f"    → 这{f0_only}天是两个机制意见分歧的情况")

# ============================================================
# 分析3: F0空仓与B2过滤的关系
# ============================================================
print("\n" + "=" * 78)
print("  分析3: F0空仓 vs B2过滤 — 是否冗余？")
print("=" * 78)

# B2触发条件: 方向=value 且 v_mom20<=0
# B2不改变仓位(始终满仓)，只改变方向(value→growth)
# F0改变仓位(满仓→空仓)，不改变方向

# 在F0空仓期间，B2是否也触发？
b2_trigger = (base_dir == 'value') & (v_mom20 <= 0)
f0_and_b2 = (f0_trigger & b2_trigger).sum()
print(f"  B2触发总天数: {b2_trigger.sum()}")
print(f"  F0与B2同时触发: {f0_and_b2}天")
print(f"  → F0管仓位(空仓)，B2管方向(value→growth)，作用维度不同")
print(f"  → 两者不冗余，但可能叠加过度过滤")

# ============================================================
# 分析4: 不同偏离度阈值详细对比
# ============================================================
print("\n" + "=" * 78)
print("  分析4: 偏离度阈值详细扫描（0.1%-0.5%）")
print("=" * 78)

def build_signal(dev_threshold=0.005, n_confirm=4, stop_threshold=0.10, stop_weight=0.30):
    dir_s = base_dir.copy()
    wt = pd.Series(1.0, index=dir_s.index)
    low_dev = ratio_dev.abs() < dev_threshold
    wt[low_dev] = 0.0
    mask = np.ones(len(dir_s), dtype=bool)
    for k in range(1, n_confirm):
        mask = mask & (dir_s.values == dir_s.shift(k).values)
    confirmed = dir_s.where(mask, np.nan)
    dir_s = confirmed.where(slope_ok, np.nan).ffill()
    wrong_value = (dir_s == 'value') & (v_mom20 <= 0)
    dir_s[wrong_value] = 'growth'
    gs = (dir_s == 'growth') & (g_dd20 < -stop_threshold)
    vs = (dir_s == 'value') & (v_dd20 < -stop_threshold)
    wt[gs | vs] = wt[gs | vs] * stop_weight
    return dir_s, wt

def run_bt(signal, weight):
    common_idx = signal.index.intersection(g_close.index)
    sig = signal.loc[common_idx]
    wt = weight.loc[common_idx]
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
    return run_backtest_engine_weighted(bt_input, config, wt.values)

def calc_m(result, freq=252, rf_annual=0.025):
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
    return {'ann': ann, 'dd': max_dd, 'sharpe': sharpe, 'calmar': calmar,
            'n_trades': result.metrics['num_trades']}

print(f"  {'阈值':>8} {'年化':>8} {'回撤':>8} {'Sharpe':>8} {'Calmar':>8} {'交易':>6} {'空仓天':>6} {'空仓段':>6}")
print(f"  {'-'*68}")

for thresh in [0.000, 0.001, 0.002, 0.003, 0.004, 0.005]:
    sig, wt = build_signal(dev_threshold=thresh)
    r = run_bt(sig, wt)
    m = calc_m(r)
    n_cash = (wt == 0.0).sum()
    # 计算空仓段数
    low_dev_mask = ratio_dev.abs() < thresh if thresh > 0 else pd.Series(False, index=ratio_dev.index)
    ld_series = low_dev_mask.astype(int)
    ld_diff = ld_series.diff().fillna(0)
    n_segs = (ld_diff == 1).sum() + (1 if ld_series.iloc[0] == 1 else 0)
    label = f"{thresh*100:.1f}%"
    if thresh == 0.000:
        label += "(X11A)"
    elif thresh == 0.003:
        label += "(★用户问)"
    elif thresh == 0.005:
        label += "(当前)"
    print(f"  {label:>8} {m['ann']*100:>7.2f}% {m['dd']*100:>7.2f}% "
          f"{m['sharpe']:>8.3f} {m['calmar']:>8.3f} {m['n_trades']:>6} {n_cash:>6} {n_segs:>6}")

# ============================================================
# 分析5: 0.3%阈值详细结果
# ============================================================
print("\n" + "=" * 78)
print("  分析5: 偏离度0.3%阈值详细结果")
print("=" * 78)

sig_03, wt_03 = build_signal(dev_threshold=0.003)
r_03 = run_bt(sig_03, wt_03)
m_03 = calc_m(r_03)

# X11-A基准
sig_x11a, wt_x11a = build_signal(dev_threshold=0.000)
r_x11a = run_bt(sig_x11a, wt_x11a)
m_x11a = calc_m(r_x11a)

# X12 0.5%
sig_05, wt_05 = build_signal(dev_threshold=0.005)
r_05 = run_bt(sig_05, wt_05)
m_05 = calc_m(r_05)

print(f"\n  {'指标':<15} {'X11-A':>12} {'X12(0.3%)':>12} {'X12(0.5%)':>12}")
print(f"  {'-'*54}")
print(f"  {'年化收益':<15} {m_x11a['ann']*100:>11.2f}% {m_03['ann']*100:>11.2f}% {m_05['ann']*100:>11.2f}%")
print(f"  {'最大回撤':<15} {m_x11a['dd']*100:>11.2f}% {m_03['dd']*100:>11.2f}% {m_05['dd']*100:>11.2f}%")
print(f"  {'Sharpe':<15} {m_x11a['sharpe']:>12.3f} {m_03['sharpe']:>12.3f} {m_05['sharpe']:>12.3f}")
print(f"  {'Calmar':<15} {m_x11a['calmar']:>12.3f} {m_03['calmar']:>12.3f} {m_05['calmar']:>12.3f}")
print(f"  {'交易次数':<15} {m_x11a['n_trades']:>12} {m_03['n_trades']:>12} {m_05['n_trades']:>12}")

n_cash_03 = (wt_03 == 0.0).sum()
n_cash_05 = (wt_05 == 0.0).sum()
print(f"  {'空仓天数':<15} {0:>12} {n_cash_03:>12} {n_cash_05:>12}")
print(f"  {'空仓占比':<15} {0:>11.1f}% {n_cash_03/n_total*100:>11.1f}% {n_cash_05/n_total*100:>11.1f}%")

# ============================================================
# 分析6: 交易次数构成分析
# ============================================================
print("\n" + "=" * 78)
print("  分析6: 交易次数构成 — 交易次数为何暴增？")
print("=" * 78)

for label, sig, wt in [("X11-A", sig_x11a, wt_x11a),
                        ("X12(0.3%)", sig_03, wt_03),
                        ("X12(0.5%)", sig_05, wt_05)]:
    # 分析交易类型
    df = pd.DataFrame({'sig': sig, 'wt': wt})
    df = df.dropna()
    df['pos'] = df['sig'] + '_' + df['wt'].astype(str)  # 如 growth_1.0, value_0.3, growth_0.0
    # 状态切换
    df['prev_pos'] = df['pos'].shift(1)
    switches = df[df['pos'] != df['prev_pos']].dropna()
    
    # 分类切换
    dir_switches = switches[
        ((switches['sig'] == 'growth') & (switches['prev_pos'].str.startswith('value')) |
         (switches['sig'] == 'value') & (switches['prev_pos'].str.startswith('growth')))
    ].shape[0]
    
    cash_switches = switches[
        ((switches['wt'] == 0.0) & (~switches['prev_pos'].str.endswith('0.0'))) |
        ((switches['wt'] != 0.0) & (switches['prev_pos'].str.endswith('0.0')))
    ].shape[0]
    
    stop_switches = switches[
        ((switches['wt'] == 0.3) & (~switches['prev_pos'].str.endswith('0.3')) & 
         (~switches['prev_pos'].str.endswith('0.0'))) |
        ((switches['wt'] == 1.0) & (switches['prev_pos'].str.endswith('0.3')) &
         (~switches['prev_pos'].str.endswith('0.0')))
    ].shape[0]
    
    print(f"\n  {label}: 总切换{len(switches)}次")
    print(f"    方向切换(growth↔value): {dir_switches}次")
    print(f"    空仓切换(满仓↔空仓): {cash_switches}次")
    print(f"    止损切换(满仓↔降仓): {stop_switches}次")

# ============================================================
# 分析7: 滞回机制测试 — 解决交易过多
# ============================================================
print("\n" + "=" * 78)
print("  分析7: 滞回机制测试 — 能否减少交易次数？")
print("=" * 78)
print("  滞回逻辑: 偏离度<低位阈值→空仓, 偏离度>高位阈值→恢复满仓")
print("  目标: 减少阈值附近的频繁切换\n")

def build_signal_hysteresis(low_thresh, high_thresh, n_confirm=4, stop_threshold=0.10, stop_weight=0.30):
    """带滞回的F0空仓过滤"""
    dir_s = base_dir.copy()
    wt = pd.Series(1.0, index=dir_s.index)
    
    # 滞回逻辑
    abs_dev = ratio_dev.abs()
    in_cash = False
    for i in range(len(abs_dev)):
        if pd.isna(abs_dev.iloc[i]):
            continue
        if not in_cash and abs_dev.iloc[i] < low_thresh:
            in_cash = True
        elif in_cash and abs_dev.iloc[i] > high_thresh:
            in_cash = False
        if in_cash:
            wt.iloc[i] = 0.0
    
    mask = np.ones(len(dir_s), dtype=bool)
    for k in range(1, n_confirm):
        mask = mask & (dir_s.values == dir_s.shift(k).values)
    confirmed = dir_s.where(mask, np.nan)
    dir_s = confirmed.where(slope_ok, np.nan).ffill()
    wrong_value = (dir_s == 'value') & (v_mom20 <= 0)
    dir_s[wrong_value] = 'growth'
    gs = (dir_s == 'growth') & (g_dd20 < -stop_threshold)
    vs = (dir_s == 'value') & (v_dd20 < -stop_threshold)
    wt[gs | vs] = wt[gs | vs] * stop_weight
    return dir_s, wt

print(f"  {'方案':>20} {'年化':>8} {'回撤':>8} {'Sharpe':>8} {'Calmar':>8} {'交易':>6} {'空仓天':>6}")
print(f"  {'-'*72}")

# 原始0.3%无滞回
sig, wt = build_signal(dev_threshold=0.003)
r = run_bt(sig, wt)
m = calc_m(r)
n_cash = (wt == 0.0).sum()
print(f"  {'0.3%无滞回(原始)':>20} {m['ann']*100:>7.2f}% {m['dd']*100:>7.2f}% "
      f"{m['sharpe']:>8.3f} {m['calmar']:>8.3f} {m['n_trades']:>6} {n_cash:>6}")

# 原始0.5%无滞回
sig, wt = build_signal(dev_threshold=0.005)
r = run_bt(sig, wt)
m = calc_m(r)
n_cash = (wt == 0.0).sum()
print(f"  {'0.5%无滞回(原始)':>20} {m['ann']*100:>7.2f}% {m['dd']*100:>7.2f}% "
      f"{m['sharpe']:>8.3f} {m['calmar']:>8.3f} {m['n_trades']:>6} {n_cash:>6}")

# 滞回方案
for low, high in [(0.003, 0.005), (0.003, 0.008), (0.002, 0.005), (0.003, 0.010)]:
    sig, wt = build_signal_hysteresis(low, high)
    r = run_bt(sig, wt)
    m = calc_m(r)
    n_cash = (wt == 0.0).sum()
    label = f"滞回{low*100:.1f}%→{high*100:.1f}%"
    print(f"  {label:>20} {m['ann']*100:>7.2f}% {m['dd']*100:>7.2f}% "
          f"{m['sharpe']:>8.3f} {m['calmar']:>8.3f} {m['n_trades']:>6} {n_cash:>6}")

# ============================================================
# 分析8: 逻辑梳理总结
# ============================================================
print("\n" + "=" * 78)
print("  分析8: 逻辑梳理总结")
print("=" * 78)

print("""
  X12的6层结构逻辑梳理:

  层级    机制              作用维度    触发条件              行为
  ─────────────────────────────────────────────────────────────────
  F1     基础信号          方向        比价 vs MA20          growth/value
  F0     偏离度空仓(新增)   仓位        |ratio_dev|<阈值      空仓(weight=0)
  A1     4天确认           方向稳定性  连续4天一致            确认/否决
  斜率   MA20斜率确认      趋势存在性  |slope|>0.3%          确认/ffill
  B2     价值动量过滤      方向选择    value方向+v_mom≤0     改growth
  E5     持仓止损          仓位        20日跌幅>10%          降仓30%

  发现的问题:
  1. [冗余] F0与斜率确认都判断"趋势不明确"，但行为矛盾:
     - F0: 趋势不明确→空仓
     - 斜率: 趋势不明确→保持原方向(ffill)
     重叠率: {overlap_rate:.1f}%的F0触发时斜率也不满足

  2. [交易过多] F0空仓在阈值附近频繁切换:
     - 0.5%阈值: {len(segments)}次切换段，{short_segs}次≤2天短切换
     - 没有确认机制，偏离度刚过阈值就恢复满仓

  3. [疏漏] 空仓→满仓恢复时没有确认机制:
     - A1确认只管方向切换，不管仓位切换
     - 恢复满仓应该也需要确认

  改进方向:
  A. 用滞回区间替代单一阈值（<低位空仓, >高位恢复）
  B. 或对F0空仓加确认机制（连续N天才空仓/恢复）
  C. 或统一F0和斜率确认（去掉一个，避免矛盾）
""")
