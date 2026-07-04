"""run_x33_reduce_trades.py — 第21轮起: 持续降低交易次数
================================================================
用户指令: "仔细审视你现在的逻辑,看看怎么样减少交易次数。
          尝试减少交易次数,但保证年化大于40%,回撤小于35%。
          请你一直尝试降低。"

约束条件: 年化>40% 且 回撤<-35%(绝对值<35%)
当前基准: X23(z=1.5, slope=0.002) 年化43.50% 回撤-25.85% 交易362次

设计原则:
- 不引入新因子(保持F1+A1+F0双重确认+B2+E5的5层结构)
- 只通过冷却期/最小持有期减少无效调仓
- 持续迭代,记录每版结果

新方案设计:
- X33: 方向切换冷却期(方向切换后N天内不再切换)
- X34: E5降仓冷却期(E5降仓后N天内不恢复)
- X35: 空仓退出确认(恢复满仓需连续N天不满足空仓条件)
- X36: E5触发条件调整(30日跌幅>12%,减少E5降仓次数)
- X37: z阈值提高扫描(z=1.8/2.0/2.5,减少空仓触发)
- X38: 组合最优方案
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
    run_backtest, calc_metrics, count_switches, test_strategy, print_result,
)

# ============================================================
# 最终最优参数(基准)
# ============================================================
Z_THRESH = 1.5
SLOPE_THRESH = 0.002
N_CONFIRM = 4
STOP_THRESHOLD = 0.10
STOP_WEIGHT = 0.30

RATIO_DEV_STD20 = RATIO_DEV.rolling(20).std()
RATIO_DEV_Z = RATIO_DEV / RATIO_DEV_STD20

# V的30日跌幅(用于X36)
V_DD30 = V_CLOSE / V_CLOSE.shift(30) - 1
G_DD30 = G_CLOSE / G_CLOSE.shift(30) - 1


# ============================================================
# 基准策略: X23完整版
# ============================================================
def build_x23_base(z_thresh=Z_THRESH, slope_thresh=SLOPE_THRESH,
                   n_confirm=N_CONFIRM, stop_threshold=STOP_THRESHOLD,
                   stop_weight=STOP_WEIGHT):
    """X23基准: z-score双重确认+B2+E5"""
    dir_s = BASE_DIR.copy()
    mask = np.ones(len(dir_s), dtype=bool)
    for k in range(1, n_confirm):
        mask = mask & (dir_s.values == dir_s.shift(k).values)
    confirmed = dir_s.where(mask, np.nan)
    wt = pd.Series(1.0, index=dir_s.index)
    low_dev_z = RATIO_DEV_Z.abs() < z_thresh
    low_slope = MA20_SLOPE.abs() < slope_thresh
    both_weak = low_dev_z & low_slope
    wt[both_weak] = 0.0
    dir_s = confirmed.ffill()
    wrong_value = (dir_s == 'value') & (V_MOM20 <= 0)
    dir_s[wrong_value] = 'growth'
    gs = (dir_s == 'growth') & (G_DD20 < -stop_threshold)
    vs = (dir_s == 'value') & (V_DD20 < -stop_threshold)
    wt[gs | vs] = wt[gs | vs] * stop_weight
    return dir_s, wt


# ============================================================
# X33: 方向切换冷却期
# ============================================================
def build_x33(dir_cooldown_days=5, z_thresh=Z_THRESH, slope_thresh=SLOPE_THRESH,
              n_confirm=N_CONFIRM, stop_threshold=STOP_THRESHOLD,
              stop_weight=STOP_WEIGHT):
    """X33: 方向切换后N天内不再切换方向

    逻辑:
    - 方向切换后进入冷却期,冷却期内即使F1信号变化也不切换
    - 冷却期内仍允许空仓(F0)和E5降仓(风控优先)
    - 冷却期结束后恢复正常方向切换
    - 目的: 减少方向震荡(45%调仓≤2天中部分是方向快速切换)
    """
    dir_s = BASE_DIR.copy()
    mask = np.ones(len(dir_s), dtype=bool)
    for k in range(1, n_confirm):
        mask = mask & (dir_s.values == dir_s.shift(k).values)
    confirmed = dir_s.where(mask, np.nan)

    # 应用方向冷却期
    raw_dir = confirmed.ffill()
    cooled_dir = raw_dir.copy()
    last_change_idx = -np.inf
    current_dir = raw_dir.iloc[0]
    for i in range(len(raw_dir)):
        if pd.isna(raw_dir.iloc[i]):
            continue
        if raw_dir.iloc[i] != current_dir:
            if i - last_change_idx >= dir_cooldown_days:
                # 冷却期已过,允许切换
                current_dir = raw_dir.iloc[i]
                last_change_idx = i
            # 否则保持原方向(冷却期内不切换)
            cooled_dir.iloc[i] = current_dir
        else:
            cooled_dir.iloc[i] = current_dir

    dir_s = cooled_dir
    wt = pd.Series(1.0, index=dir_s.index)
    low_dev_z = RATIO_DEV_Z.abs() < z_thresh
    low_slope = MA20_SLOPE.abs() < slope_thresh
    both_weak = low_dev_z & low_slope
    wt[both_weak] = 0.0

    wrong_value = (dir_s == 'value') & (V_MOM20 <= 0)
    dir_s[wrong_value] = 'growth'
    gs = (dir_s == 'growth') & (G_DD20 < -stop_threshold)
    vs = (dir_s == 'value') & (V_DD20 < -stop_threshold)
    wt[gs | vs] = wt[gs | vs] * stop_weight
    return dir_s, wt


# ============================================================
# X34: E5降仓冷却期
# ============================================================
def build_x34(e5_cooldown_days=5, z_thresh=Z_THRESH, slope_thresh=SLOPE_THRESH,
              n_confirm=N_CONFIRM, stop_threshold=STOP_THRESHOLD,
              stop_weight=STOP_WEIGHT):
    """X34: E5降仓后N天内不恢复(即使跌幅恢复也保持降仓)

    逻辑:
    - E5触发降仓后进入冷却期,冷却期内保持降仓状态
    - 冷却期结束后才允许恢复满仓
    - 目的: 减少E5快速触发恢复(避免震荡)
    """
    dir_s = BASE_DIR.copy()
    mask = np.ones(len(dir_s), dtype=bool)
    for k in range(1, n_confirm):
        mask = mask & (dir_s.values == dir_s.shift(k).values)
    confirmed = dir_s.where(mask, np.nan)
    wt = pd.Series(1.0, index=dir_s.index)
    low_dev_z = RATIO_DEV_Z.abs() < z_thresh
    low_slope = MA20_SLOPE.abs() < slope_thresh
    both_weak = low_dev_z & low_slope
    wt[both_weak] = 0.0
    dir_s = confirmed.ffill()
    wrong_value = (dir_s == 'value') & (V_MOM20 <= 0)
    dir_s[wrong_value] = 'growth'

    # E5降仓冷却期
    gs = (dir_s == 'growth') & (G_DD20 < -stop_threshold)
    vs = (dir_s == 'value') & (V_DD20 < -stop_threshold)
    e5_trigger = gs | vs

    # 状态机: 触发后冷却期内保持降仓
    in_cooldown = False
    cooldown_count = 0
    for i in range(len(wt)):
        if pd.isna(dir_s.iloc[i]) or pd.isna(wt.iloc[i]):
            continue
        if e5_trigger.iloc[i] and not in_cooldown:
            in_cooldown = True
            cooldown_count = 0
            wt.iloc[i] = wt.iloc[i] * stop_weight
        elif in_cooldown:
            cooldown_count += 1
            if cooldown_count >= e5_cooldown_days:
                # 冷却期结束,检查是否还需要降仓
                if e5_trigger.iloc[i]:
                    cooldown_count = 0
                    wt.iloc[i] = wt.iloc[i] * stop_weight
                else:
                    in_cooldown = False
                    # 恢复满仓(除非空仓)
                    if wt.iloc[i] > 0:
                        wt.iloc[i] = 1.0
            else:
                # 冷却期内保持降仓
                if wt.iloc[i] > 0:
                    wt.iloc[i] = stop_weight
    return dir_s, wt


# ============================================================
# X35: 空仓退出确认(恢复满仓需连续N天不满足空仓条件)
# ============================================================
def build_x35(exit_confirm_days=2, z_thresh=Z_THRESH, slope_thresh=SLOPE_THRESH,
              n_confirm=N_CONFIRM, stop_threshold=STOP_THRESHOLD,
              stop_weight=STOP_WEIGHT):
    """X35: 空仓退出需连续N天不满足空仓条件

    逻辑:
    - 进入空仓: both_weak为True即进入(无延迟,保留F0敏感性)
    - 退出空仓: 需连续N天both_weak=False才恢复满仓
    - 目的: 减少空仓快速震荡(空仓相关占47.7%)
    - 注: 与X16/X28不同——X16/X28是进入空仓需确认,这里是退出需确认
    """
    dir_s = BASE_DIR.copy()
    mask = np.ones(len(dir_s), dtype=bool)
    for k in range(1, n_confirm):
        mask = mask & (dir_s.values == dir_s.shift(k).values)
    confirmed = dir_s.where(mask, np.nan)

    wt = pd.Series(1.0, index=dir_s.index)
    low_dev_z = RATIO_DEV_Z.abs() < z_thresh
    low_slope = MA20_SLOPE.abs() < slope_thresh
    both_weak = low_dev_z & low_slope

    # 状态机: 空仓退出需连续N天不满足条件
    is_flat = False
    not_weak_count = 0
    for i in range(len(dir_s)):
        if pd.isna(RATIO_DEV_Z.iloc[i]):
            wt.iloc[i] = 1.0
            continue
        if not is_flat:
            if both_weak.iloc[i]:
                is_flat = True
                not_weak_count = 0
                wt.iloc[i] = 0.0
            else:
                wt.iloc[i] = 1.0
        else:
            if not both_weak.iloc[i]:
                not_weak_count += 1
                if not_weak_count >= exit_confirm_days:
                    is_flat = False
                    wt.iloc[i] = 1.0
                else:
                    wt.iloc[i] = 0.0  # 等待确认期保持空仓
            else:
                not_weak_count = 0  # 重置计数
                wt.iloc[i] = 0.0

    dir_s = confirmed.ffill()
    wrong_value = (dir_s == 'value') & (V_MOM20 <= 0)
    dir_s[wrong_value] = 'growth'
    gs = (dir_s == 'growth') & (G_DD20 < -stop_threshold)
    vs = (dir_s == 'value') & (V_DD20 < -stop_threshold)
    wt[gs | vs] = wt[gs | vs] * stop_weight
    return dir_s, wt


# ============================================================
# X36: E5触发条件调整(30日跌幅>12%,减少E5降仓次数)
# ============================================================
def build_x36(stop_threshold_30d=0.12, z_thresh=Z_THRESH, slope_thresh=SLOPE_THRESH,
              n_confirm=N_CONFIRM, stop_weight=STOP_WEIGHT):
    """X36: E5改用30日跌幅>12%触发(原20日>10%)

    逻辑:
    - 原: 20日跌幅>10%触发降仓
    - 新: 30日跌幅>12%触发降仓
    - 目的: 30日窗口更稳定,12%阈值更严格,减少E5误触发
    """
    dir_s = BASE_DIR.copy()
    mask = np.ones(len(dir_s), dtype=bool)
    for k in range(1, n_confirm):
        mask = mask & (dir_s.values == dir_s.shift(k).values)
    confirmed = dir_s.where(mask, np.nan)
    wt = pd.Series(1.0, index=dir_s.index)
    low_dev_z = RATIO_DEV_Z.abs() < z_thresh
    low_slope = MA20_SLOPE.abs() < slope_thresh
    both_weak = low_dev_z & low_slope
    wt[both_weak] = 0.0
    dir_s = confirmed.ffill()
    wrong_value = (dir_s == 'value') & (V_MOM20 <= 0)
    dir_s[wrong_value] = 'growth'
    # 使用30日跌幅+12%阈值
    gs = (dir_s == 'growth') & (G_DD30 < -stop_threshold_30d)
    vs = (dir_s == 'value') & (V_DD30 < -stop_threshold_30d)
    wt[gs | vs] = wt[gs | vs] * stop_weight
    return dir_s, wt


# ============================================================
# X37: z阈值提高扫描
# ============================================================
def build_x37(z_thresh=2.0, slope_thresh=SLOPE_THRESH,
              n_confirm=N_CONFIRM, stop_threshold=STOP_THRESHOLD,
              stop_weight=STOP_WEIGHT):
    """X37: 提高z阈值(减少空仓触发频率)

    逻辑: z阈值越高,空仓触发越少(只在偏离度极小时才空仓)
    X31发现z=1.8时Calmar=1.593,可能交易更少
    """
    return build_x23_base(z_thresh=z_thresh, slope_thresh=slope_thresh,
                          n_confirm=n_confirm, stop_threshold=stop_threshold,
                          stop_weight=stop_weight)


# ============================================================
# X38: 组合方案(方向冷却+空仓退出确认+E5冷却)
# ============================================================
def build_x38(dir_cooldown_days=5, exit_confirm_days=2, e5_cooldown_days=5,
              z_thresh=Z_THRESH, slope_thresh=SLOPE_THRESH,
              n_confirm=N_CONFIRM, stop_threshold=STOP_THRESHOLD,
              stop_weight=STOP_WEIGHT):
    """X38: 组合方案(方向冷却+空仓退出确认+E5冷却)

    逻辑: 同时应用三种冷却机制
    - 方向切换冷却期: 减少方向震荡
    - 空仓退出确认: 减少空仓快速震荡
    - E5降仓冷却期: 减少E5快速恢复
    """
    dir_s = BASE_DIR.copy()
    mask = np.ones(len(dir_s), dtype=bool)
    for k in range(1, n_confirm):
        mask = mask & (dir_s.values == dir_s.shift(k).values)
    confirmed = dir_s.where(mask, np.nan)

    # 1. 方向切换冷却期
    raw_dir = confirmed.ffill()
    cooled_dir = raw_dir.copy()
    last_change_idx = -np.inf
    current_dir = raw_dir.iloc[0]
    for i in range(len(raw_dir)):
        if pd.isna(raw_dir.iloc[i]):
            continue
        if raw_dir.iloc[i] != current_dir:
            if i - last_change_idx >= dir_cooldown_days:
                current_dir = raw_dir.iloc[i]
                last_change_idx = i
            cooled_dir.iloc[i] = current_dir
        else:
            cooled_dir.iloc[i] = current_dir
    dir_s = cooled_dir

    # 2. 空仓退出确认
    wt = pd.Series(1.0, index=dir_s.index)
    low_dev_z = RATIO_DEV_Z.abs() < z_thresh
    low_slope = MA20_SLOPE.abs() < slope_thresh
    both_weak = low_dev_z & low_slope

    is_flat = False
    not_weak_count = 0
    for i in range(len(dir_s)):
        if pd.isna(RATIO_DEV_Z.iloc[i]):
            wt.iloc[i] = 1.0
            continue
        if not is_flat:
            if both_weak.iloc[i]:
                is_flat = True
                not_weak_count = 0
                wt.iloc[i] = 0.0
            else:
                wt.iloc[i] = 1.0
        else:
            if not both_weak.iloc[i]:
                not_weak_count += 1
                if not_weak_count >= exit_confirm_days:
                    is_flat = False
                    wt.iloc[i] = 1.0
                else:
                    wt.iloc[i] = 0.0
            else:
                not_weak_count = 0
                wt.iloc[i] = 0.0

    wrong_value = (dir_s == 'value') & (V_MOM20 <= 0)
    dir_s[wrong_value] = 'growth'

    # 3. E5降仓冷却期
    gs = (dir_s == 'growth') & (G_DD20 < -stop_threshold)
    vs = (dir_s == 'value') & (V_DD20 < -stop_threshold)
    e5_trigger = gs | vs

    in_cooldown = False
    cooldown_count = 0
    for i in range(len(wt)):
        if pd.isna(dir_s.iloc[i]) or pd.isna(wt.iloc[i]):
            continue
        if e5_trigger.iloc[i] and not in_cooldown:
            in_cooldown = True
            cooldown_count = 0
            wt.iloc[i] = wt.iloc[i] * stop_weight
        elif in_cooldown:
            cooldown_count += 1
            if cooldown_count >= e5_cooldown_days:
                if e5_trigger.iloc[i]:
                    cooldown_count = 0
                    wt.iloc[i] = wt.iloc[i] * stop_weight
                else:
                    in_cooldown = False
                    if wt.iloc[i] > 0:
                        wt.iloc[i] = 1.0
            else:
                if wt.iloc[i] > 0:
                    wt.iloc[i] = stop_weight
    return dir_s, wt


# ============================================================
# 主测试流程
# ============================================================
if __name__ == '__main__':
    print("=" * 90)
    print("  X33-X38: 第21轮起 持续降低交易次数")
    print("  约束: 年化>40% 且 回撤<-35%")
    print("=" * 90)

    results = []

    # 基准
    print("\n  [基准] X23(z=1.5, slope=0.002):")
    info = test_strategy("X23基准", build_x23_base, desc="z=1.5/slope=0.002/n=4/10%/30%")
    print_result(info)
    results.append(info)

    # X33: 方向切换冷却期
    print("\n" + "=" * 90)
    print("  X33: 方向切换冷却期(方向切换后N天内不再切换)")
    print("=" * 90)
    print(f"  {'冷却天数':>10} {'年化':>8} {'回撤':>8} {'Sharpe':>8} {'Calmar':>8} {'交易':>6} {'满足':>6}")
    print(f"  {'-'*65}")
    for n in [3, 5, 7, 10, 15]:
        info = test_strategy(f"X33(cd={n})", build_x33,
                             {'dir_cooldown_days': n}, f"方向冷却{n}天")
        m = info
        ok = "✅" if m['ann'] > 0.40 and m['dd'] > -0.35 else "❌"
        print(f"  {n:>8}天 {m['ann']*100:>7.2f}% {m['dd']*100:>7.2f}% "
              f"{m['sharpe']:>8.3f} {m['calmar']:>8.3f} {m['n_trades']:>6} {ok:>6}")
        results.append(info)

    # X34: E5降仓冷却期
    print("\n" + "=" * 90)
    print("  X34: E5降仓冷却期(E5降仓后N天内不恢复)")
    print("=" * 90)
    print(f"  {'冷却天数':>10} {'年化':>8} {'回撤':>8} {'Sharpe':>8} {'Calmar':>8} {'交易':>6} {'满足':>6}")
    print(f"  {'-'*65}")
    for n in [3, 5, 7, 10]:
        info = test_strategy(f"X34(e5cd={n})", build_x34,
                             {'e5_cooldown_days': n}, f"E5冷却{n}天")
        m = info
        ok = "✅" if m['ann'] > 0.40 and m['dd'] > -0.35 else "❌"
        print(f"  {n:>8}天 {m['ann']*100:>7.2f}% {m['dd']*100:>7.2f}% "
              f"{m['sharpe']:>8.3f} {m['calmar']:>8.3f} {m['n_trades']:>6} {ok:>6}")
        results.append(info)

    # X35: 空仓退出确认
    print("\n" + "=" * 90)
    print("  X35: 空仓退出确认(恢复满仓需连续N天不满足空仓条件)")
    print("=" * 90)
    print(f"  {'确认天数':>10} {'年化':>8} {'回撤':>8} {'Sharpe':>8} {'Calmar':>8} {'交易':>6} {'满足':>6}")
    print(f"  {'-'*65}")
    for n in [2, 3, 4, 5]:
        info = test_strategy(f"X35(exit={n})", build_x35,
                             {'exit_confirm_days': n}, f"退出确认{n}天")
        m = info
        ok = "✅" if m['ann'] > 0.40 and m['dd'] > -0.35 else "❌"
        print(f"  {n:>8}天 {m['ann']*100:>7.2f}% {m['dd']*100:>7.2f}% "
              f"{m['sharpe']:>8.3f} {m['calmar']:>8.3f} {m['n_trades']:>6} {ok:>6}")
        results.append(info)

    # X36: E5触发条件调整
    print("\n" + "=" * 90)
    print("  X36: E5触发条件调整(30日跌幅>12%替代20日>10%)")
    print("=" * 90)
    print(f"  {'配置':>20} {'年化':>8} {'回撤':>8} {'Sharpe':>8} {'Calmar':>8} {'交易':>6} {'满足':>6}")
    print(f"  {'-'*75}")
    for thr in [0.10, 0.12, 0.15]:
        info = test_strategy(f"X36(30d,{thr})", build_x36,
                             {'stop_threshold_30d': thr}, f"30日跌幅>{thr*100:.0f}%")
        m = info
        ok = "✅" if m['ann'] > 0.40 and m['dd'] > -0.35 else "❌"
        print(f"  {f'30d>{thr*100:.0f}%':>20} {m['ann']*100:>7.2f}% {m['dd']*100:>7.2f}% "
              f"{m['sharpe']:>8.3f} {m['calmar']:>8.3f} {m['n_trades']:>6} {ok:>6}")
        results.append(info)

    # X37: z阈值提高
    print("\n" + "=" * 90)
    print("  X37: z阈值提高扫描(减少空仓触发频率)")
    print("=" * 90)
    print(f"  {'z阈值':>10} {'年化':>8} {'回撤':>8} {'Sharpe':>8} {'Calmar':>8} {'交易':>6} {'满足':>6}")
    print(f"  {'-'*65}")
    for z in [1.5, 1.8, 2.0, 2.5, 3.0]:
        info = test_strategy(f"X37(z={z})", build_x37,
                             {'z_thresh': z}, f"z={z}")
        m = info
        ok = "✅" if m['ann'] > 0.40 and m['dd'] > -0.35 else "❌"
        print(f"  {z:>10} {m['ann']*100:>7.2f}% {m['dd']*100:>7.2f}% "
              f"{m['sharpe']:>8.3f} {m['calmar']:>8.3f} {m['n_trades']:>6} {ok:>6}")
        results.append(info)

    # X38: 组合方案
    print("\n" + "=" * 90)
    print("  X38: 组合方案(方向冷却+空仓退出确认+E5冷却)")
    print("=" * 90)
    print(f"  {'配置':>35} {'年化':>8} {'回撤':>8} {'Sharpe':>8} {'Calmar':>8} {'交易':>6} {'满足':>6}")
    print(f"  {'-'*90}")
    combos = [
        {'dir_cooldown_days': 3, 'exit_confirm_days': 2, 'e5_cooldown_days': 3},
        {'dir_cooldown_days': 5, 'exit_confirm_days': 2, 'e5_cooldown_days': 5},
        {'dir_cooldown_days': 5, 'exit_confirm_days': 3, 'e5_cooldown_days': 5},
        {'dir_cooldown_days': 7, 'exit_confirm_days': 2, 'e5_cooldown_days': 5},
        {'dir_cooldown_days': 5, 'exit_confirm_days': 2, 'e5_cooldown_days': 7},
    ]
    for c in combos:
        label = f"cd{c['dir_cooldown_days']}/ex{c['exit_confirm_days']}/e5{c['e5_cooldown_days']}"
        info = test_strategy(f"X38({label})", build_x38, c, label)
        m = info
        ok = "✅" if m['ann'] > 0.40 and m['dd'] > -0.35 else "❌"
        print(f"  {label:>35} {m['ann']*100:>7.2f}% {m['dd']*100:>7.2f}% "
              f"{m['sharpe']:>8.3f} {m['calmar']:>8.3f} {m['n_trades']:>6} {ok:>6}")
        results.append(info)

    # 汇总: 筛选满足约束的方案并按交易次数排序
    print("\n" + "=" * 90)
    print("  【汇总】满足约束(年化>40%且回撤<-35%)的方案,按交易次数升序")
    print("=" * 90)
    feasible = [r for r in results if r['ann'] > 0.40 and r['dd'] > -0.35]
    feasible.sort(key=lambda x: x['n_trades'])
    print(f"  {'版本':<30} {'年化':>8} {'回撤':>8} {'Sharpe':>8} {'Calmar':>8} {'交易':>6}")
    print(f"  {'-'*75}")
    for r in feasible:
        print(f"  {r['name']:<30} {r['ann']*100:>7.2f}% {r['dd']*100:>7.2f}% "
              f"{r['sharpe']:>8.3f} {r['calmar']:>8.3f} {r['n_trades']:>6}")

    if feasible:
        best = feasible[0]
        print(f"\n  ★最优方案(交易最少): {best['name']}")
        print(f"    年化={best['ann']*100:.2f}% 回撤={best['dd']*100:.2f}% "
              f"Sharpe={best['sharpe']:.3f} Calmar={best['calmar']:.3f} 交易={best['n_trades']}次")
    else:
        print("\n  ⚠️ 没有方案满足约束条件")

    print("\n  完成!")
