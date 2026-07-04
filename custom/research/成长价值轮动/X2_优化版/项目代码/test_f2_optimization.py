# -*- coding: utf-8 -*-
"""
F2 优化方案 A/B/C 测试（X2 策略优化 Task 2）
==================================================
基于 F2 场景分析结果（f2_analysis_report.txt），实现三个 F2 优化方案并与 X1 基线对比：

  - X1 基线：style_score = f1 + f2 决定方向，MA75 降仓（各持5%，总10%）
  - 方案 A：F2 不参与方向判定，改用于仓位调整（成长末期减速 → 70% 仓位）
  - 方案 B：F2 反向过滤（成长初期启动场景忽略 F2，只用 F1）
  - 方案 C：F2 完全移除（只用 F1 决定方向，对照组）

约束：
  - 只优化 F2，降仓逻辑保持 X1 原版（各持5%，总10%）
  - 使用最新数据（2026-07-01）
  - 指数数据 open=close
  - 佣金万1，无冲击滑点，无跳空滑点
  - 周频调仓（W-FRI）

场景定义（与 f2_scenario_analysis.py 一致）：
  - 成长初期启动: ratio_dev < 1%  且  accel_diff > 0.005   (F2 有害噪声)
  - 成长末期减速: ratio_dev > 3%  且  accel_diff < -0.005  (F2 预警但无效)
  - 趋势中段    : ratio_dev > 3%  且  accel_diff > 0       (F2 对但冗余)
  - 其他        : 不属于以上三类                            (F2 略有正贡献)
"""
import sys
sys.path.insert(0, r'c:\XuanJLH\Qbot\custom\backtests')

from pathlib import Path
import numpy as np
import pandas as pd
from backtest_engine import (
    BacktestInput, BacktestConfig,
    run_backtest_engine_weighted
)

# ============================================================
# 1. 参数（与 X1 一致）
# ============================================================
F1 = 0.5          # 比价MA20权重
F2 = 5.0          # 动量加速度权重
MA_W = 75         # 择时均线周期
MA_PCT = 0.97     # MA阈值（跌破97%降仓）
CUT = 0.1         # 降仓后总仓位（各持5%→10%）

# F2 场景阈值（来自 f2_scenario_analysis.py）
RD_EARLY_START = 0.01    # |ratio_dev| < 1%  → 成长初期启动
RD_TREND = 0.03          # ratio_dev > 3%    → 趋势中段 / 成长末期减速
ACCEL_STRONG = 0.005     # |accel_diff| > 0.005 → F2 强信号

# 方案A 减仓权重
WEIGHT_DECEL = 0.7       # 成长末期减速 → 70% 仓位

# ============================================================
# 2. 加载数据
# ============================================================
DATA_DIR = Path(r'c:\temp_v72_data')
print("=" * 70)
print("  F2 优化方案 A/B/C 测试（X2 策略优化 Task 2）")
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
# 3. 信号计算（与 X1 完全一致）
# ============================================================
# 因子1：比价MA20方向
ratio = (g_close / v_close).shift(1)
ratio_ma20 = ratio.rolling(20).mean()
ratio_dev = ratio / ratio_ma20 - 1
f1_signal = np.tanh(ratio_dev * 30) * F1

# 因子2：动量加速度
g_roc21 = g_close.pct_change(21).shift(1)
v_roc21 = v_close.pct_change(21).shift(1)
g_accel = g_roc21 - g_roc21.shift(10)
v_accel = v_roc21 - v_roc21.shift(10)
accel_diff = (g_accel - v_accel).clip(-0.02, 0.02)
f2_signal = accel_diff * F2

# MA75择时
g_ma = g_close.shift(1).rolling(MA_W).mean()
v_ma = v_close.shift(1).rolling(MA_W).mean()
both_below = (g_close.shift(1) < g_ma * MA_PCT) & (v_close.shift(1) < v_ma * MA_PCT)

# 构建日频DataFrame
df = pd.DataFrame(index=g_close.index)
df['g_close'] = g_close
df['v_close'] = v_close
df['ratio_dev'] = ratio_dev
df['accel_diff'] = accel_diff
df['f1_signal'] = f1_signal
df['f2_signal'] = f2_signal
df['both_below'] = both_below

# 周频采样（W-FRI）
weekly = df.resample('W-FRI').last().dropna(
    subset=['f1_signal', 'f2_signal', 'ratio_dev']
).iloc[1:]
weekly['both_below'] = weekly['both_below'].astype(bool)

print(f"周频样本: {len(weekly)} 周, 区间: {weekly.index[0]:%Y-%m-%d} ~ {weekly.index[-1]:%Y-%m-%d}")


# ============================================================
# 4. 场景标记（用于诊断）
# ============================================================
def classify_scenario(row):
    rd = row['ratio_dev']
    ad = row['accel_diff']
    if rd < RD_EARLY_START and ad > ACCEL_STRONG:
        return '成长初期启动'
    if rd > RD_TREND and ad < -ACCEL_STRONG:
        return '成长末期减速'
    if rd > RD_TREND and ad > 0:
        return '趋势中段'
    return '其他'

weekly['scenario'] = weekly.apply(classify_scenario, axis=1)
print("\n[场景分布]")
for sc, n in weekly['scenario'].value_counts().items():
    print(f"  {sc:<10}: {n:>4d} 周 ({n/len(weekly)*100:5.2f}%)")


# ============================================================
# 5. 通用状态机 + 回测函数
# ============================================================
def run_scheme(name, candidate_g_func, weight_func):
    """
    通用回测框架：
      - candidate_g_func(row) -> bool: 该周是否选成长（方向判定）
      - weight_func(row) -> float: 该周仓位权重（1.0/0.7等，不含降仓覆盖）
      - 降仓逻辑统一：both_below 时 weight=0.1, signal='value'（X1 原版行为）
      - 状态机：降仓时保留 current_pos，降仓结束后恢复方向判定
    """
    position = pd.Series(np.nan, index=weekly.index)   # 1.0=growth, 0.0=value, 0.1=CUT
    weight_arr = pd.Series(np.nan, index=weekly.index)  # 1.0/0.7/0.1
    current_pos = None
    direction_switches = 0

    for i in range(len(weekly)):
        row = weekly.iloc[i]

        # MA75降仓优先（与 X1 一致：position=CUT, 保留 current_pos）
        if row['both_below']:
            position.iloc[i] = CUT
            weight_arr.iloc[i] = CUT
            continue

        # 方向判定
        target_g = candidate_g_func(row)
        target = 1.0 if target_g else 0.0

        if current_pos is None:
            current_pos = target
        elif target != current_pos:
            current_pos = target
            direction_switches += 1

        position.iloc[i] = current_pos
        weight_arr.iloc[i] = weight_func(row)

    wk = weekly.copy()
    wk['position'] = position
    wk['weight'] = weight_arr
    wk = wk.dropna(subset=['position', 'weight'])

    # ---- 映射到日频（与 X1 一致）----
    daily = pd.DataFrame(index=g_close.index)
    daily['g_close'] = g_close
    daily['v_close'] = v_close

    wpos = wk['position'].copy()
    wwt = wk['weight'].copy()
    # 对齐到实际交易日
    idx = daily.index.searchsorted(wpos.index, side='right') - 1
    valid = idx >= 0
    wpos = wpos[valid]
    wwt = wwt[valid]
    wpos.index = daily.index[idx[valid]]
    wwt.index = daily.index[idx[valid]]
    wpos = wpos[~wpos.index.duplicated(keep='last')]
    wwt = wwt[~wwt.index.duplicated(keep='last')]

    daily['position'] = np.nan
    daily['weight'] = np.nan
    daily.loc[wpos.index, 'position'] = wpos
    daily.loc[wwt.index, 'weight'] = wwt
    daily['position'] = daily['position'].ffill()
    daily['weight'] = daily['weight'].ffill()
    daily = daily.dropna(subset=['position', 'weight'])

    # 构造引擎输入
    n = len(daily)
    dates = daily.index.strftime('%Y-%m-%d').values
    # 信号：position==1.0 → 'growth'，否则 'value'（含降仓CUT，与X1一致）
    signal = np.array(['growth' if p == 1.0 else 'value' for p in daily['position'].values])
    # 权重：直接用 weight 列
    position_weight = daily['weight'].values.astype(np.float64)

    # 开盘价=收盘价（指数数据）
    v_open = daily['v_close'].values.astype(np.float64)
    v_close_arr = daily['v_close'].values.astype(np.float64)
    g_open = daily['g_close'].values.astype(np.float64)
    g_close_arr = daily['g_close'].values.astype(np.float64)

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

    result = run_backtest_engine_weighted(bt_input, config, position_weight)
    return result, direction_switches, wk


# ============================================================
# 6. 各方案定义
# ============================================================

# --- X1 基线：F1+F2 决定方向 ---
def x1_candidate_g(row):
    """style_score = f1 + f2，与 X1 完全一致"""
    style_score = row['f1_signal'] + row['f2_signal']
    return style_score > 0

def x1_weight(row):
    """正常100%"""
    return 1.0


# --- 方案A：F2 不参与方向判定，改用于仓位调整 ---
def scheme_a_candidate_g(row):
    """只用 F1 决定方向（style_score = f1_signal）"""
    return row['f1_signal'] > 0

def scheme_a_weight(row):
    """
    成长末期减速 → 70%（减仓预警）
    其他场景 → 100%（正常）
    - 成长末期减速: ratio_dev > 3% 且 F2反向强(accel_diff < -0.005)
    """
    rd = row['ratio_dev']
    ad = row['accel_diff']
    if rd > RD_TREND and ad < -ACCEL_STRONG:
        return WEIGHT_DECEL
    return 1.0


# --- 方案B：F2 反向过滤 ---
def scheme_b_candidate_g(row):
    """
    F1+F2 决定方向（与X1相同），但在"成长初期启动"场景忽略F2
    成长初期启动: |ratio_dev| < 1% 且 accel_diff > 0.005 → f2_signal = 0
    """
    rd = row['ratio_dev']
    ad = row['accel_diff']
    f2 = row['f2_signal']
    # 成长初期启动场景：忽略F2（有害噪声）
    if abs(rd) < RD_EARLY_START and ad > ACCEL_STRONG:
        f2 = 0.0
    style_score = row['f1_signal'] + f2
    return style_score > 0

def scheme_b_weight(row):
    """正常100%"""
    return 1.0


# --- 方案C：F2 完全移除（对照组）---
def scheme_c_candidate_g(row):
    """只用 F1 决定方向"""
    return row['f1_signal'] > 0

def scheme_c_weight(row):
    """正常100%"""
    return 1.0


# ============================================================
# 7. 运行所有方案
# ============================================================
schemes = [
    ('X1基线(f1+f2)',    x1_candidate_g,   x1_weight),
    ('方案A(F2仓位调整)', scheme_a_candidate_g, scheme_a_weight),
    ('方案B(F2反向过滤)', scheme_b_candidate_g, scheme_b_weight),
    ('方案C(F2完全移除)', scheme_c_candidate_g, scheme_c_weight),
]

results = []
for name, cg_func, wt_func in schemes:
    print(f"\n[运行] {name} ...")
    result, switches, wk = run_scheme(name, cg_func, wt_func)
    m = result.metrics

    # 统计降仓周数和减仓周数
    cut_weeks = int((wk['weight'] == CUT).sum())
    decel_weeks = int((wk['weight'] == WEIGHT_DECEL).sum())

    results.append({
        'name': name,
        'ann': m['annual_ret'],
        'dd': m['max_dd'],
        'sharpe': m['sharpe'],
        'calmar': m['calmar'],
        'trades': m['num_trades'],
        'switches': switches,
        'cut_weeks': cut_weeks,
        'decel_weeks': decel_weeks,
        'final_nav': m['final_nav'],
        'total_ret': m['total_ret'],
        'win_rate': m['win_rate'],
    })
    print(f"  年化 {m['annual_ret']*100:.2f}% | 回撤 {m['max_dd']*100:.2f}% | "
          f"Sharpe {m['sharpe']:.3f} | Calmar {m['calmar']:.3f} | "
          f"调仓 {m['num_trades']} | 方向切换 {switches} | "
          f"降仓周 {cut_weeks} | 减仓周 {decel_weeks}")


# ============================================================
# 8. 输出对比表
# ============================================================
print("\n" + "=" * 74)
print("  F2 优化方案 A/B/C 测试结果对比")
print("=" * 74)
print(f"  {'方案':<22} {'年化':>8} {'回撤':>9} {'Sharpe':>8} {'Calmar':>8} {'调仓次数':>8}")
print("  " + "-" * 68)
for r in results:
    print(f"  {r['name']:<20} {r['ann']*100:>7.2f}% {r['dd']*100:>8.2f}% "
          f"{r['sharpe']:>8.3f} {r['calmar']:>8.3f} {r['trades']:>8d}")
print("=" * 74)

# 详细指标
print("\n[详细指标]")
print(f"  {'方案':<22} {'总收益':>10} {'最终净值':>12} {'胜率':>8} {'方向切换':>8} {'降仓周':>6} {'减仓周':>6}")
print("  " + "-" * 76)
for r in results:
    print(f"  {r['name']:<20} {r['total_ret']*100:>9.2f}% {r['final_nav']:>12,.0f} "
          f"{r['win_rate']*100:>7.1f}% {r['switches']:>8d} {r['cut_weeks']:>6d} {r['decel_weeks']:>6d}")


# ============================================================
# 9. 分析：哪个方案最优
# ============================================================
print("\n" + "=" * 74)
print("  分析：哪个方案最优")
print("=" * 74)

# 找出最优方案（综合年化、Sharpe、Calmar）
x1 = results[0]
a = results[1]
b = results[2]
c = results[3]

print(f"""
[各方案对比 X1 基线]
  方案A vs X1: 年化 {((a['ann']-x1['ann'])*100):+.2f}pp | 回撤 {((a['dd']-x1['dd'])*100):+.2f}pp | 
                Sharpe {a['sharpe']-x1['sharpe']:+.3f} | Calmar {a['calmar']-x1['calmar']:+.3f} | 
                调仓 {a['trades']-x1['trades']:+d}
  方案B vs X1: 年化 {((b['ann']-x1['ann'])*100):+.2f}pp | 回撤 {((b['dd']-x1['dd'])*100):+.2f}pp | 
                Sharpe {b['sharpe']-x1['sharpe']:+.3f} | Calmar {b['calmar']-x1['calmar']:+.3f} | 
                调仓 {b['trades']-x1['trades']:+d}
  方案C vs X1: 年化 {((c['ann']-x1['ann'])*100):+.2f}pp | 回撤 {((c['dd']-x1['dd'])*100):+.2f}pp | 
                Sharpe {c['sharpe']-x1['sharpe']:+.3f} | Calmar {c['calmar']-x1['calmar']:+.3f} | 
                调仓 {c['trades']-x1['trades']:+d}
""")

# 综合评分（年化40% + Sharpe 30% + Calmar 30%，归一化）
def score(r):
    # 简单加权：年化 + Sharpe*0.15 + Calmar*0.10（粗略排序用）
    return r['ann'] + r['sharpe']*0.15 + r['calmar']*0.10

best = max(results, key=score)
print(f"[综合评分最优] {best['name']}")
print(f"  年化 {best['ann']*100:.2f}% | Sharpe {best['sharpe']:.3f} | Calmar {best['calmar']:.3f}")

# 分析逻辑
print(f"""
[分析结论]

1. F2 的整体价值评估：
   - 方案C（F2完全移除）年化 {c['ann']*100:.2f}% vs X1基线 {x1['ann']*100:.2f}%，差异 {((c['ann']-x1['ann'])*100):+.2f}pp
   - 若方案C优于X1，说明F2整体有害；若劣于X1，说明F2仍有正贡献
   - 当前结果：F2对X1策略{'有正贡献（移除后收益下降）' if c['ann'] < x1['ann'] else '整体有害（移除后收益提升）'}

2. 方案A（F2仓位调整）效果：
   - 年化 {a['ann']*100:.2f}% vs X1 {x1['ann']*100:.2f}%，差异 {((a['ann']-x1['ann'])*100):+.2f}pp
   - 回撤 {a['dd']*100:.2f}% vs X1 {x1['dd']*100:.2f}%，差异 {((a['dd']-x1['dd'])*100):+.2f}pp
   - 减仓周数 {a['decel_weeks']}（成长末期减速场景触发70%仓位）
   - {'有效：F2用于仓位调整改善了风险调整收益' if a['calmar'] > x1['calmar'] else '无效：仓位调整未改善Calmar'}

3. 方案B（F2反向过滤）效果：
   - 年化 {b['ann']*100:.2f}% vs X1 {x1['ann']*100:.2f}%，差异 {((b['ann']-x1['ann'])*100):+.2f}pp
   - 调仓次数 {b['trades']} vs X1 {x1['trades']}（过滤成长初期启动噪声应减少调仓）
   - {'有效：过滤有害噪声提升了收益' if b['ann'] > x1['ann'] else '无效或负面'}

4. 最优方案：{best['name']}
""")

print("=" * 74)
print("  测试完成")
print("=" * 74)
