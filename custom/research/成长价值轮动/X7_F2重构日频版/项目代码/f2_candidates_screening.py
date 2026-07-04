"""f2_candidates_screening.py — F2'候选因子筛选

预先计算多个候选F2'因子与F1的相关性、IC，为X7实现做准备。
目标：找到与F1相关性<0.3且IC>0.03的互补因子。

候选因子：
  1. 相对RSI差（g_rsi_14 - v_rsi_14）
  2. 相对波动率差（g_vol_20 - v_vol_20）
  3. 相对动量一阶差（g_mom_21 - v_mom_21）
  4. 比价动量（ratio.pct_change(20)）
  5. 相对动量差归一化（(g_mom - v_mom).clip(-0.1, 0.1) * 5）
  6. 趋势持续性（连续同向比例）
  7. 相对ROC差（g_roc_10 - v_roc_10）
  8. 比价突破（ratio vs ratio_ma50）
"""
import sys
sys.path.insert(0, r'c:\XuanJLH\Qbot\custom\backtests')

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

# ============================================================
# 1. 加载数据 + F1
# ============================================================
DATA_DIR = Path(r'c:\temp_v72_data')
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

print("=" * 80)
print("  F2' 候选因子筛选（与F1相关性 + IC）")
print("=" * 80)
print(f"数据区间: {g_close.index[0]:%Y-%m-%d} ~ {g_close.index[-1]:%Y-%m-%d}")

# F1（不变）
F1 = 0.5
ratio = (g_close / v_close).shift(1)
ratio_ma20 = ratio.rolling(20).mean()
ratio_dev = ratio / ratio_ma20 - 1
f1_signal = np.tanh(ratio_dev * 30) * F1

# 未来相对收益
fwd_21 = (g_close.pct_change(21) - v_close.pct_change(21)).shift(-21)

# ============================================================
# 2. 计算候选F2'因子
# ============================================================

def calc_rsi(series, period=14):
    """RSI指标"""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi

candidates = {}

# 1. 相对RSI差
g_rsi14 = calc_rsi(g_close, 14).shift(1)
v_rsi14 = calc_rsi(v_close, 14).shift(1)
candidates['RSI差(g_rsi14-v_rsi14)'] = (g_rsi14 - v_rsi14) / 100  # 归一化到[-1,1]

# 2. 相对波动率差（20日）
g_vol20 = g_close.pct_change().rolling(20).std().shift(1)
v_vol20 = v_close.pct_change().rolling(20).std().shift(1)
candidates['波动率差(g_vol20-v_vol20)'] = ((g_vol20 - v_vol20) / (g_vol20 + v_vol20)).fillna(0)

# 3. 相对动量一阶差（21日）
g_mom21 = g_close.pct_change(21).shift(1)
v_mom21 = v_close.pct_change(21).shift(1)
candidates['动量一阶差(g_mom21-v_mom21)'] = (g_mom21 - v_mom21).clip(-0.1, 0.1) * 5

# 4. 比价动量（ratio的20日变化率）
candidates['比价动量(ratio_pct20)'] = ratio.pct_change(20).clip(-0.05, 0.05) * 10

# 5. 相对ROC差（10日）
g_roc10 = g_close.pct_change(10).shift(1)
v_roc10 = v_close.pct_change(10).shift(1)
candidates['ROC10差(g_roc10-v_roc10)'] = (g_roc10 - v_roc10).clip(-0.05, 0.05) * 10

# 6. 趋势持续性（ratio连续>ma20的比例，20日窗口）
ratio_above_ma = (ratio > ratio_ma20).astype(float)
candidates['趋势持续性(ratio>ma20的20日比例)'] = ratio_above_ma.rolling(20).mean().shift(1) - 0.5

# 7. 比价突破（ratio vs ratio_ma50）
ratio_ma50 = ratio.rolling(50).mean()
candidates['比价突破50(ratio/ma50-1)'] = (ratio / ratio_ma50 - 1).clip(-0.05, 0.05) * 10

# 8. 相对强度Z-score（比价的Z-score）
candidates['比价Z-score(60日)'] = ((ratio - ratio.rolling(60).mean()) / ratio.rolling(60).std()).shift(1).clip(-2, 2) * 0.5

# 9. 动量差（63日，季度）
g_mom63 = g_close.pct_change(63).shift(1)
v_mom63 = v_close.pct_change(63).shift(1)
candidates['动量差63日(g_mom63-v_mom63)'] = (g_mom63 - v_mom63).clip(-0.15, 0.15) * 3

# 10. 加速度差（原F2，对照）
g_roc21 = g_close.pct_change(21).shift(1)
v_roc21 = v_close.pct_change(21).shift(1)
g_accel = g_roc21 - g_roc21.shift(10)
v_accel = v_roc21 - v_roc21.shift(10)
accel_diff = (g_accel - v_accel).clip(-0.02, 0.02)
candidates['原F2(动量加速度)*5'] = accel_diff * 5.0

# ============================================================
# 3. 筛选：相关性 + IC
# ============================================================
print(f"\n{'因子':<36} {'与F1相关':>10} {'21日IC':>10} {'p值':>10} {'|IC|>0.03?':>10} {'|相关|<0.3?':>12}")
print("-" * 100)

results = []
for name, factor in candidates.items():
    df = pd.DataFrame({'f1': f1_signal, 'fac': factor, 'fwd': fwd_21}).dropna()
    if len(df) < 100:
        print(f"{name:<36} (样本不足: {len(df)})")
        continue
    corr, _ = stats.pearsonr(df['f1'].values, df['fac'].values)
    ic, p = stats.spearmanr(df['fac'].values, df['fwd'].values)
    ic_ok = "✅" if abs(ic) > 0.03 else "❌"
    corr_ok = "✅" if abs(corr) < 0.3 else "❌"
    print(f"{name:<36} {corr:>+10.4f} {ic:>+10.4f} {p:>10.4f} {ic_ok:>10} {corr_ok:>12}")
    results.append({
        'name': name,
        'corr_f1': corr,
        'ic_21': ic,
        'p_value': p,
        'ic_ok': abs(ic) > 0.03,
        'corr_ok': abs(corr) < 0.3,
        'score': (abs(ic) if abs(ic) > 0.03 else 0) + (1 - abs(corr) if abs(corr) < 0.3 else 0),
    })

# ============================================================
# 4. 排序与推荐
# ============================================================
print("\n" + "=" * 80)
print("  候选因子排名（按 IC有效 且 与F1低相关 综合评分）")
print("=" * 80)

# 筛选：IC>0.03 且 |相关|<0.3
good = [r for r in results if r['ic_ok'] and r['corr_ok']]
good.sort(key=lambda x: -x['score'])

if good:
    print(f"\n✅ 满足条件（IC>0.03 且 |相关|<0.3）的因子共 {len(good)} 个：")
    for i, r in enumerate(good, 1):
        print(f"  {i}. {r['name']}")
        print(f"     与F1相关: {r['corr_f1']:+.4f}  21日IC: {r['ic_21']:+.4f}  综合评分: {r['score']:.4f}")
else:
    print("\n❌ 没有因子同时满足 IC>0.03 且 |相关|<0.3")
    print("\n按IC排名（忽略相关性约束）：")
    by_ic = sorted(results, key=lambda x: -abs(x['ic_21']))
    for i, r in enumerate(by_ic[:5], 1):
        print(f"  {i}. {r['name']}: IC={r['ic_21']:+.4f}, 相关={r['corr_f1']:+.4f}")

print("\n注：综合评分 = |IC|（若IC>0.03）+ (1-|相关|)（若|相关|<0.3）")
print("    高分因子适合作为F2'候选，与F1低相关且自身有效")
