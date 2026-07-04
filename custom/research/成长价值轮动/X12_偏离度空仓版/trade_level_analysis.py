"""trade_level_analysis.py — 交易级别逐笔分析
================================================================
用户质疑: 384次交易为什么这么多?多少次正收益?多少无效?
分析每次调仓的类型、盈亏、有效性,找出减少调仓的方法
================================================================
"""
import sys
sys.path.insert(0, r'c:\XuanJLH\Qbot\custom\backtests')
sys.path.insert(0, r'c:\XuanJLH\Qbot\custom\research\成长价值轮动\X12_偏离度空仓版')

from pathlib import Path
import numpy as np
import pandas as pd
from optimize_runner import (
    G_CLOSE, V_CLOSE, RATIO, RATIO_MA20, RATIO_DEV, MA20_SLOPE, SLOPE_OK,
    V_MOM20, G_DD20, V_DD20, BASE_DIR,
    run_backtest, calc_metrics,
)

# 最终最优参数
Z_THRESH = 1.5
SLOPE_THRESH = 0.002
N_CONFIRM = 4
STOP_THRESHOLD = 0.10
STOP_WEIGHT = 0.30

RATIO_DEV_STD20 = RATIO_DEV.rolling(20).std()
RATIO_DEV_Z = RATIO_DEV / RATIO_DEV_STD20


def build_x23_full():
    """X23完整版信号"""
    dir_s = BASE_DIR.copy()
    mask = np.ones(len(dir_s), dtype=bool)
    for k in range(1, N_CONFIRM):
        mask = mask & (dir_s.values == dir_s.shift(k).values)
    confirmed = dir_s.where(mask, np.nan)
    wt = pd.Series(1.0, index=dir_s.index)
    low_dev_z = RATIO_DEV_Z.abs() < Z_THRESH
    low_slope = MA20_SLOPE.abs() < SLOPE_THRESH
    both_weak = low_dev_z & low_slope
    wt[both_weak] = 0.0
    dir_s = confirmed.ffill()
    wrong_value = (dir_s == 'value') & (V_MOM20 <= 0)
    dir_s[wrong_value] = 'growth'
    gs = (dir_s == 'growth') & (G_DD20 < -STOP_THRESHOLD)
    vs = (dir_s == 'value') & (V_DD20 < -STOP_THRESHOLD)
    wt[gs | vs] = wt[gs | vs] * STOP_WEIGHT
    return dir_s, wt


# ============================================================
# 构建信号并回测
# ============================================================
print("=" * 90)
print("  X23交易级别逐笔分析")
print("=" * 90)

signal, weight = build_x23_full()

# 回测获取每日收益
result = run_backtest(signal, weight)
df = result.to_dataframe()
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date')

# 对齐信号和收益
common_idx = signal.index.intersection(df.index)
sig = signal.loc[common_idx]
wt = weight.loc[common_idx]
daily_ret = df.loc[common_idx, 'daily_ret']

# ============================================================
# 识别每次调仓
# ============================================================
# 每日状态: 方向_仓位
state = pd.Series(index=common_idx, dtype=str)
for i, idx in enumerate(common_idx):
    s = sig.loc[idx]
    w = wt.loc[idx]
    if pd.isna(s) or pd.isna(w):
        state.loc[idx] = 'NA'
    else:
        w_round = round(float(w), 2)
        state.loc[idx] = f"{s}_{w_round}"

# 找出调仓点(状态变化)
state_prev = state.shift(1)
change_mask = state != state_prev
# 第一天不算调仓
change_mask.iloc[0] = False
change_dates = state[change_mask].index

print(f"\n  总交易日: {len(common_idx)}")
print(f"  调仓次数: {len(change_dates)}")
print(f"  回测引擎交易数: {result.metrics['num_trades']}")
print(f"  调仓频率: 每{len(common_idx)/len(change_dates):.1f}天调一次")

# ============================================================
# 分析每次调仓
# ============================================================
trades = []
all_dates = list(common_idx)

for i, change_date in enumerate(change_dates):
    pos = all_dates.index(change_date)
    # 找下一次调仓
    if i + 1 < len(change_dates):
        next_change = change_dates[i + 1]
        next_pos = all_dates.index(next_change)
    else:
        next_pos = len(all_dates) - 1
        next_change = all_dates[next_pos]

    # 这次调仓后持有的天数
    hold_days = next_pos - pos

    # 这次调仓到下次调仓的收益
    period_ret = (1 + daily_ret.iloc[pos:next_pos + 1]).prod() - 1

    # 调仓类型
    prev_state = state_prev.loc[change_date]
    curr_state = state.loc[change_date]

    # 解析状态
    def parse_state(s):
        if pd.isna(s) or s == 'NA':
            return None, None
        parts = s.split('_')
        return parts[0], float(parts[1])

    prev_dir, prev_wt = parse_state(prev_state)
    curr_dir, curr_wt = parse_state(curr_state)

    # 分类调仓类型
    if prev_wt == 0 and curr_wt > 0:
        trade_type = '空仓恢复'
    elif prev_wt > 0 and curr_wt == 0:
        trade_type = '进入空仓'
    elif prev_dir != curr_dir and prev_wt > 0 and curr_wt > 0:
        trade_type = '方向切换'
    elif prev_wt != curr_wt and prev_dir == curr_dir:
        if curr_wt < prev_wt:
            trade_type = 'E5降仓'
        else:
            trade_type = 'E5恢复'
    else:
        trade_type = '其他'

    trades.append({
        'date': change_date,
        'type': trade_type,
        'prev_state': prev_state,
        'curr_state': curr_state,
        'hold_days': hold_days,
        'return': period_ret,
        'prev_dir': prev_dir,
        'curr_dir': curr_dir,
        'prev_wt': prev_wt,
        'curr_wt': curr_wt,
    })

trades_df = pd.DataFrame(trades)

# ============================================================
# 调仓类型分布
# ============================================================
print("\n" + "=" * 90)
print("  【调仓类型分布】")
print("=" * 90)

type_counts = trades_df['type'].value_counts()

def count_pos(x):
    return (x > 0).sum()

def count_nonpos(x):
    return (x <= 0).sum()

type_stats = trades_df.groupby('type').agg({
    'return': ['count', 'mean', 'sum', count_pos, count_nonpos],
    'hold_days': 'mean'
}).round(4)
type_stats.columns = ['次数', '平均收益', '总收益', '正收益次数', '非正收益次数', '平均持有天数']

print(f"\n  {'类型':<12} {'次数':>6} {'占比':>8} {'平均收益':>10} {'正收益':>8} {'负收益':>8} {'胜率':>8} {'平均持有':>8}")
print(f"  {'-'*80}")
for t, row in type_stats.iterrows():
    cnt = int(row['次数'])
    pct = cnt / len(trades_df) * 100
    avg_ret = row['平均收益'] * 100
    pos = int(row['正收益次数'])
    neg = int(row['非正收益次数'])
    win_rate = pos / cnt * 100 if cnt > 0 else 0
    hold = row['平均持有天数']
    print(f"  {t:<12} {cnt:>6} {pct:>7.1f}% {avg_ret:>+9.2f}% {pos:>8} {neg:>8} {win_rate:>7.1f}% {hold:>7.1f}天")

# ============================================================
# 正收益 vs 无效调仓
# ============================================================
print("\n" + "=" * 90)
print("  【正收益 vs 无效调仓分析】")
print("=" * 90)

total = len(trades_df)
pos_trades = trades_df[trades_df['return'] > 0]
neg_trades = trades_df[trades_df['return'] <= 0]
zero_trades = trades_df[trades_df['return'] == 0]

print(f"\n  总调仓次数: {total}")
print(f"  正收益调仓: {len(pos_trades)} ({len(pos_trades)/total*100:.1f}%)")
print(f"    平均收益: +{pos_trades['return'].mean()*100:.2f}%")
print(f"    总收益贡献: +{pos_trades['return'].sum()*100:.2f}%")
print(f"  负收益调仓: {len(neg_trades)} ({len(neg_trades)/total*100:.1f}%)")
print(f"    平均收益: {neg_trades['return'].mean()*100:.2f}%")
print(f"    总收益损失: {neg_trades['return'].sum()*100:.2f}%")
print(f"  零收益调仓: {len(zero_trades)} ({len(zero_trades)/total*100:.1f}%)")

# ============================================================
# 短持调仓分析(无效调仓的主要来源)
# ============================================================
print("\n" + "=" * 90)
print("  【短持调仓分析】持有天数少的调仓")
print("=" * 90)

for thresh in [1, 2, 3, 5, 7]:
    short = trades_df[trades_df['hold_days'] <= thresh]
    print(f"\n  持有≤{thresh}天的调仓: {len(short)}次 ({len(short)/total*100:.1f}%)")
    if len(short) > 0:
        print(f"    平均收益: {short['return'].mean()*100:+.2f}%")
        print(f"    正收益: {(short['return']>0).sum()}次")
        print(f"    类型分布:")
        for t, c in short['type'].value_counts().items():
            print(f"      {t}: {c}次")

# ============================================================
# 调仓类型 × 持有天数 交叉分析
# ============================================================
print("\n" + "=" * 90)
print("  【各类型调仓的持有天数分布】")
print("=" * 90)

print(f"\n  {'类型':<12} {'≤2天':>6} {'3-5天':>6} {'6-10天':>6} {'>10天':>6} {'平均':>6}")
print(f"  {'-'*50}")
for t in trades_df['type'].unique():
    subset = trades_df[trades_df['type'] == t]
    le2 = (subset['hold_days'] <= 2).sum()
    d35 = ((subset['hold_days'] >= 3) & (subset['hold_days'] <= 5)).sum()
    d610 = ((subset['hold_days'] >= 6) & (subset['hold_days'] <= 10)).sum()
    gt10 = (subset['hold_days'] > 10).sum()
    avg = subset['hold_days'].mean()
    print(f"  {t:<12} {le2:>6} {d35:>6} {d610:>6} {gt10:>6} {avg:>5.1f}天")

# ============================================================
# 无效调仓根因分析
# ============================================================
print("\n" + "=" * 90)
print("  【无效调仓根因分析】")
print("=" * 90)

# 分析短持调仓(≤2天)的来源
short_trades = trades_df[trades_df['hold_days'] <= 2]
print(f"\n  短持调仓(≤2天)共{len(short_trades)}次,类型分布:")
for t, c in short_trades['type'].value_counts().items():
    pct = c / len(short_trades) * 100
    avg_ret = short_trades[short_trades['type'] == t]['return'].mean()
    print(f"    {t}: {c}次 ({pct:.1f}%) 平均收益{avg_ret*100:+.2f}%")

# 分析空仓恢复→再次进入空仓的模式
print(f"\n  空仓震荡模式分析:")
flat_trades = trades_df[trades_df['type'].isin(['进入空仓', '空仓恢复'])]
print(f"    空仓相关调仓: {len(flat_trades)}次 ({len(flat_trades)/total*100:.1f}%)")
# 找出"恢复→空仓→恢复"的快速震荡
flat_dates = flat_trades['date'].tolist()
rapid_oscillation = 0
for i in range(len(flat_dates) - 1):
    gap = (flat_dates[i + 1] - flat_dates[i]).days
    if gap <= 3:
        rapid_oscillation += 1
print(f"    其中间隔≤3天的快速震荡: {rapid_oscillation}次")

# ============================================================
# 减少调仓的方案
# ============================================================
print("\n" + "=" * 90)
print("  【减少调仓方案分析】")
print("=" * 90)

print(f"""
  当前调仓{total}次的构成:""")

for t in trades_df['type'].unique():
    c = (trades_df['type'] == t).sum()
    print(f"    {t}: {c}次")

print(f"""
  减少调仓的潜在方案:

  方案1: 空仓最短持有期(避免短持空仓)
    - 进入空仓后至少持有N天才恢复
    - 可减少: 短持空仓恢复({(short_trades['type']=='空仓恢复').sum()}次短持)

  方案2: 调仓冷却期(同方向调仓间隔N天)
    - 上次调仓后N天内不调仓
    - 可减少: 频繁的方向切换

  方案3: 信号变化阈值(变化幅度<阈值不调仓)
    - 偏离度z变化<0.3不触发空仓切换
    - 可减少: 边界震荡

  方案4: 合并相邻调仓(3天内的同类型调仓合并)
    - 把间隔≤3天的同类型调仓算作1次
    - 可减少: 快速震荡""")

# 实际测试方案1: 空仓最短持有期
print("\n" + "=" * 90)
print("  【方案1实测: 空仓最短持有期N天】")
print("=" * 90)

def build_x23_with_min_flat(min_flat_days):
    """X23+空仓最短持有期"""
    dir_s = BASE_DIR.copy()
    mask = np.ones(len(dir_s), dtype=bool)
    for k in range(1, N_CONFIRM):
        mask = mask & (dir_s.values == dir_s.shift(k).values)
    confirmed = dir_s.where(mask, np.nan)

    wt = pd.Series(1.0, index=dir_s.index)
    low_dev_z = RATIO_DEV_Z.abs() < Z_THRESH
    low_slope = MA20_SLOPE.abs() < SLOPE_THRESH
    both_weak = low_dev_z & low_slope

    # 状态机: 空仓后至少持有min_flat_days天才恢复
    is_flat = False
    flat_count = 0
    for i in range(len(dir_s)):
        if pd.isna(RATIO_DEV_Z.iloc[i]):
            wt.iloc[i] = 1.0
            continue
        if not is_flat:
            if both_weak.iloc[i]:
                is_flat = True
                flat_count = 0
                wt.iloc[i] = 0.0
            else:
                wt.iloc[i] = 1.0
        else:
            flat_count += 1
            if flat_count >= min_flat_days and not both_weak.iloc[i]:
                is_flat = False
                wt.iloc[i] = 1.0
            else:
                wt.iloc[i] = 0.0

    dir_s = confirmed.ffill()
    wrong_value = (dir_s == 'value') & (V_MOM20 <= 0)
    dir_s[wrong_value] = 'growth'
    gs = (dir_s == 'growth') & (G_DD20 < -STOP_THRESHOLD)
    vs = (dir_s == 'value') & (V_DD20 < -STOP_THRESHOLD)
    wt[gs | vs] = wt[gs | vs] * STOP_WEIGHT
    return dir_s, wt


print(f"\n  {'最短持有':>10} {'年化':>8} {'回撤':>8} {'Sharpe':>8} {'Calmar':>8} {'交易':>6}")
print(f"  {'-'*55}")
# 基准
sig0, wt0 = build_x23_full()
r0 = run_backtest(sig0, wt0)
m0 = calc_metrics(r0)
print(f"  {'无(基准)':>10} {m0['ann']*100:>7.2f}% {m0['dd']*100:>7.2f}% {m0['sharpe']:>8.3f} {m0['calmar']:>8.3f} {m0['n_trades']:>6}")

for n in [2, 3, 5, 7, 10]:
    sig, wt = build_x23_with_min_flat(n)
    r = run_backtest(sig, wt)
    m = calc_metrics(r)
    print(f"  {n:>8}天 {m['ann']*100:>7.2f}% {m['dd']*100:>7.2f}% {m['sharpe']:>8.3f} {m['calmar']:>8.3f} {m['n_trades']:>6}")

# 实际测试方案4: 合并相邻调仓(通过信号平滑)
print("\n" + "=" * 90)
print("  【方案4实测: 信号变化阈值(偏离度z变化<阈值不切换)】")
print("=" * 90)

def build_x23_with_smooth(smart_z_thresh=0.0):
    """X23+空仓切换需偏离度z变化超过阈值"""
    dir_s = BASE_DIR.copy()
    mask = np.ones(len(dir_s), dtype=bool)
    for k in range(1, N_CONFIRM):
        mask = mask & (dir_s.values == dir_s.shift(k).values)
    confirmed = dir_s.where(mask, np.nan)

    wt = pd.Series(1.0, index=dir_s.index)
    # 偏离度z的变化量
    z_change = RATIO_DEV_Z.diff().abs()

    low_dev_z = RATIO_DEV_Z.abs() < Z_THRESH
    low_slope = MA20_SLOPE.abs() < SLOPE_THRESH
    both_weak = low_dev_z & low_slope

    # 只有z变化超过阈值才触发空仓切换
    is_flat = False
    for i in range(len(dir_s)):
        if pd.isna(RATIO_DEV_Z.iloc[i]):
            wt.iloc[i] = 1.0
            continue
        if not is_flat:
            if both_weak.iloc[i] and (smart_z_thresh == 0 or (i > 0 and z_change.iloc[i] > smart_z_thresh)):
                is_flat = True
                wt.iloc[i] = 0.0
            else:
                wt.iloc[i] = 1.0
        else:
            # 恢复也需要z变化超过阈值
            if not both_weak.iloc[i] and (smart_z_thresh == 0 or (i > 0 and z_change.iloc[i] > smart_z_thresh)):
                is_flat = False
                wt.iloc[i] = 1.0
            else:
                wt.iloc[i] = 0.0

    dir_s = confirmed.ffill()
    wrong_value = (dir_s == 'value') & (V_MOM20 <= 0)
    dir_s[wrong_value] = 'growth'
    gs = (dir_s == 'growth') & (G_DD20 < -STOP_THRESHOLD)
    vs = (dir_s == 'value') & (V_DD20 < -STOP_THRESHOLD)
    wt[gs | vs] = wt[gs | vs] * STOP_WEIGHT
    return dir_s, wt


print(f"\n  {'z变化阈值':>10} {'年化':>8} {'回撤':>8} {'Sharpe':>8} {'Calmar':>8} {'交易':>6}")
print(f"  {'-'*55}")
print(f"  {'0(基准)':>10} {m0['ann']*100:>7.2f}% {m0['dd']*100:>7.2f}% {m0['sharpe']:>8.3f} {m0['calmar']:>8.3f} {m0['n_trades']:>6}")
for zc in [0.3, 0.5, 0.8, 1.0]:
    sig, wt = build_x23_with_smooth(zc)
    r = run_backtest(sig, wt)
    m = calc_metrics(r)
    print(f"  {zc:>9.1f} {m['ann']*100:>7.2f}% {m['dd']*100:>7.2f}% {m['sharpe']:>8.3f} {m['calmar']:>8.3f} {m['n_trades']:>6}")

# ============================================================
# 最终建议
# ============================================================
print("\n" + "=" * 90)
print("  【最终结论与建议】")
print("=" * 90)

pos_pct = len(pos_trades) / total * 100
neg_pct = len(neg_trades) / total * 100
short_pct = len(short_trades) / total * 100

print(f"""
  1. 调仓效率:
     - 总调仓{total}次
     - 正收益: {len(pos_trades)}次 ({pos_pct:.1f}%)
     - 负收益: {len(neg_trades)}次 ({neg_pct:.1f}%)
     - 短持≤2天: {len(short_trades)}次 ({short_pct:.1f}%)

  2. 主要问题:
     - 空仓相关调仓占比过高
     - 短持调仓{len(short_trades)}次,其中大部分是空仓震荡

  3. 减少调仓的可行性:
     - 方案1(空仓最短持有期)可有效减少短持空仓
     - 方案4(z变化阈值)可过滤微小震荡
     - 需要平衡: 减少调仓 vs 保持回撤控制""")
