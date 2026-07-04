"""统计BIAS超买信号在历史回测中的触发次数"""
import sys
sys.path.insert(0, r'c:\XuanJLH\Qbot\custom\backtests')
sys.path.insert(0, r'c:\XuanJLH\Qbot\custom\research\成长价值轮动\X12_偏离度空仓版')
sys.path.insert(0, r'c:\XuanJLH\Qbot\custom\research\成长价值轮动\X14_统一策略干净版')

import numpy as np
import pandas as pd
from optimize_runner import (
    G_CLOSE, V_CLOSE, RATIO, RATIO_MA20, RATIO_DEV, MA20_SLOPE,
    V_MOM20, G_DD20, V_DD20, BASE_DIR,
    run_backtest, calc_metrics, count_switches,
)
from run_x33_reduce_trades import RATIO_DEV_Z

T = RATIO_DEV_Z
SLOPE = MA20_SLOPE

def build_signal(bias_high=0.19, bias_ma=20):
    """用X14逻辑生成信号，记录BIAS触发"""
    slope_thresh=0.002; sw=0.17; st=0.09; cd=8
    ms=10; ml=20; rt=1.3; dc=5; dcd=6
    
    G_MA = G_CLOSE.rolling(bias_ma).mean()
    V_MA = V_CLOSE.rolling(bias_ma).mean()
    G_BIAS = G_CLOSE / G_MA - 1
    V_BIAS = V_CLOSE / V_MA - 1
    V_MOM_S = V_CLOSE.pct_change(ms)
    V_MOM_L = V_CLOSE.pct_change(ml)
    
    raw_dir = (T > 0).map({True: 'BULL', False: 'BEAR'})
    mask = np.ones(len(raw_dir), dtype=bool)
    for k in range(1, dc):
        mask = mask & (raw_dir.values == raw_dir.shift(k).values)
    confirmed_dir = raw_dir.where(mask, np.nan).ffill().fillna('BULL')
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
    wt = pd.Series(1.0, index=T.index)
    wrong_value = (dir_raw == 'BEAR') & (V_MOM_S <= 0) & (V_MOM_L <= 0)
    dir_raw[wrong_value] = 'BULL'
    dir_s = dir_raw.map({'BULL': 'growth', 'BEAR': 'value'})
    
    # BIAS触发标记
    extreme_g = (dir_s == 'growth') & (G_BIAS > bias_high)
    extreme_v = (dir_s == 'value') & (V_BIAS > bias_high)
    bias_trigger = extreme_g | extreme_v
    
    return bias_trigger, G_BIAS, V_BIAS, dir_s

# 获取数据
bias_trigger, G_BIAS, V_BIAS, dir_s = build_signal()

# 触发天数统计
total_trigger_days = bias_trigger.sum()
print(f"BIAS > 19% 总触发天数: {total_trigger_days}天")

# 离散事件统计: 连续触发算1次
trigger_dates = bias_trigger[bias_trigger].index
events = []
event_start = None
for i, d in enumerate(trigger_dates):
    if event_start is None:
        event_start = d
    else:
        prev_date = trigger_dates[i-1]
        diff = (d - prev_date).days
        if diff > 1:
            events.append((event_start, trigger_dates[i-1]))
            event_start = d
if event_start is not None:
    events.append((event_start, trigger_dates[-1]))

print(f"离散触发事件(连续触发算1次): {len(events)}次\n")

# 按年份统计
yearly_trigger = bias_trigger.groupby(bias_trigger.index.year).sum()
print(f"{'年份':>6s}  {'触发天数':>8s}  {'方向':>8s}  {'平均G_BIAS':>10s}  {'平均V_BIAS':>10s}  {'最高G_BIAS':>10s}")
print("  " + "-" * 56)
for yr in sorted(yearly_trigger.index):
    if yearly_trigger[yr] == 0:
        continue
    yr_mask = bias_trigger & (bias_trigger.index.year == yr)
    trig_days = int(yearly_trigger[yr])
    g_mean = G_BIAS[yr_mask].mean()
    v_mean = V_BIAS[yr_mask].mean()
    g_max = G_BIAS[yr_mask].max()
    dirs = dir_s[yr_mask].value_counts()
    ndir = int(dirs.get('growth', 0))
    vdir = int(dirs.get('value', 0))
    print(f"  {yr:>4d}  {trig_days:>8d}  {f'G{ndir}V{vdir}':>8s}  {g_mean*100:>9.2f}%  {v_mean*100:>9.2f}%  {g_max*100:>9.2f}%")

# 列出每次事件的详细日期
print(f"\n--- 详细事件列表 ---")
for i, (start, end) in enumerate(events):
    duration = (end - start).days + 1
    g_max = G_BIAS[start:end].max()
    v_max = V_BIAS[start:end].max()
    dir_at = dir_s[start:end].value_counts().idxmax() if len(dir_s[start:end].value_counts()) > 0 else "?"
    print(f"  事件{i+1}: {start.date()} ~ {end.date()}  (持续{duration}天)  "
          f"方向={dir_at}  G_BIAS最高={g_max*100:.2f}%  V_BIAS最高={v_max*100:.2f}%")

# BIAS整体分布
print(f"\n--- BIAS整体分布 ---")
all_g = G_BIAS.dropna()
all_v = V_BIAS.dropna()
print(f"  G_BIAS: 均值={all_g.mean()*100:.2f}%  中位数={all_g.median()*100:.2f}%  "
      f"95分位={all_g.quantile(0.95)*100:.2f}%  99分位={all_g.quantile(0.99)*100:.2f}%  最大值={all_g.max()*100:.2f}%")
print(f"  V_BIAS: 均值={all_v.mean()*100:.2f}%  中位数={all_v.median()*100:.2f}%  "
      f"95分位={all_v.quantile(0.95)*100:.2f}%  99分位={all_v.quantile(0.99)*100:.2f}%  最大值={all_v.max()*100:.2f}%")

# 历年超买天数
print(f"\n--- 历年 BIAS > 19% 天数 ---")
for yr in range(2013, 2027):
    yr_g = G_BIAS[G_BIAS.index.year == yr]
    yr_v = V_BIAS[V_BIAS.index.year == yr]
    g_over = (yr_g > 0.19).sum()
    v_over = (yr_v > 0.19).sum()
    print(f"  {yr}: G超买{g_over}天  V超买{v_over}天  G最高{yr_g.max()*100:.1f}%  V最高{yr_v.max()*100:.1f}%")
