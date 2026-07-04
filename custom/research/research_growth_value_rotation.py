"""
成长100 vs 价值100 风格轮动研究
===================================
目标：深入理解两个指数的行为特征，为策略优化提供依据
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
DATA_DIR = Path(r"c:\caches\sxc\style_rotation_strategy\data")


def load_close(csv_name):
    df = pd.read_csv(DATA_DIR / csv_name)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').sort_index()
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    return df['close'].dropna()


print("=" * 70)
print("成长100 vs 价值100 风格轮动研究")
print("=" * 70)

# 加载数据
g = load_close("index_480080.csv")
v = load_close("index_480081.csv")
common = g.index.intersection(v.index)
g = g.loc[common].sort_index()
v = v.loc[common].sort_index()
g = g["2013":"2025"]
v = v["2013":"2025"]

print(f"\n数据区间: {g.index[0].date()} ~ {g.index[-1].date()}")
print(f"交易日数: {len(g)}")

# ============================================================
# 1. 基础统计
# ============================================================
print("\n" + "=" * 70)
print("1. 基础统计")
print("=" * 70)

g_ret = g / g.shift(1) - 1
v_ret = v / v.shift(1) - 1

for name, s in [("成长100", g), ("价值100", v)]:
    ret = s.iloc[-1] / s.iloc[0] - 1
    days = (s.index[-1] - s.index[0]).days
    ann = (1 + ret) ** (365 / days) - 1
    vol = (s.pct_change().dropna().std() * np.sqrt(252))
    max_dd = ((s / s.cummax() - 1).min())
    print(f"\n  {name}:")
    print(f"    累计收益: {ret*100:>8.2f}%    年化: {ann*100:>6.2f}%")
    print(f"    年化波动: {vol*100:>8.2f}%    最大回撤: {max_dd*100:>6.2f}%")
    print(f"    起点: {s.iloc[0]:>8.2f}     终点: {s.iloc[-1]:>8.2f}")

# ============================================================
# 2. 相关性分析
# ============================================================
print("\n" + "=" * 70)
print("2. 相关性分析（滚动1年窗口）")
print("=" * 70)

# 日收益相关性
corr_daily = g_ret.corr(v_ret)
print(f"\n  日收益相关性: {corr_daily:.4f}")

# 滚动相关性
roll_corr = g_ret.rolling(252).corr(v_ret)
print(f"  滚动1年相关性: 均值={roll_corr.mean():.4f}, "
      f"min={roll_corr.min():.4f}, max={roll_corr.max():.4f}")

# ============================================================
# 3. 比价分析（成长/价值）
# ============================================================
print("\n" + "=" * 70)
print("3. 比价分析 (成长/价值)")
print("=" * 70)

ratio = g / v
ratio_ret = ratio / ratio.shift(1) - 1

print(f"  比价日均变化: {ratio_ret.mean()*100:.4f}%")
print(f"  比价年化波动: {ratio_ret.std()*np.sqrt(252)*100:.2f}%")
print(f"  比价区间: {ratio.min():.3f} ~ {ratio.max():.3f}")

# 比价的分位数
for q in [0.1, 0.25, 0.5, 0.75, 0.9]:
    print(f"  {q*100:.0f}%分位: {ratio.quantile(q):.4f}")

# ============================================================
# 4. 风格轮动周期分析
# ============================================================
print("\n" + "=" * 70)
print("4. 风格轮动周期分析")
print("=" * 70)

# 滚动1年超额收益
g12 = g / g.shift(252) - 1
v12 = v / v.shift(252) - 1
excess_12 = g12 - v12

print(f"  成长-价值 1年滚动超额:")
print(f"    均值: {excess_12.mean()*100:.2f}%")
print(f"    标准差: {excess_12.std()*100:.2f}%")
print(f"    max超额: {excess_12.max()*100:.2f}% (日期: {excess_12.idxmax().date()})")
print(f"    min超额: {excess_12.min()*100:.2f}% (日期: {excess_12.idxmin().date()})")

# 判断风格持续期
g_cum = g.iloc[-1] / g.iloc[0]
v_cum = v.iloc[-1] / v.iloc[0]
print(f"\n  长期来看成长vs价值: {g_cum/v_cum:.2f}x (成长是价值的倍数)")

# 月度超额
monthly_excess = g.resample('M').last().pct_change() - v.resample('M').last().pct_change()
growth_months = (monthly_excess > 0).sum()
value_months = (monthly_excess < 0).sum()
print(f"\n  月度胜出次数:")
print(f"    成长胜出: {growth_months} 个月 ({growth_months/(growth_months+value_months)*100:.1f}%)")
print(f"    价值胜出: {value_months} 个月 ({value_months/(growth_months+value_months)*100:.1f}%)")

# ============================================================
# 5. V72 因子分析
# ============================================================
print("\n" + "=" * 70)
print("5. V72 因子分析")
print("=" * 70)

# 计算 V72 因子
ratio_s = g / v
ma20 = ratio_s.rolling(20).mean()
ratio_dev = ratio_s / ma20 - 1
f1 = np.tanh(ratio_dev * 30) * 0.5

g_roc21 = g.pct_change(21)
v_roc21 = v.pct_change(21)
g_accel = g_roc21 - g_roc21.shift(10)
v_accel = v_roc21 - v_roc21.shift(10)
f2 = np.clip(g_accel - v_accel, -0.02, 0.02) * 5.0

score = f1 + f2

f1_mean = f1.mean()
f1_std = f1.std()
f2_mean = f2.mean()
f2_std = f2.std()
score_mean = score.mean()
score_std = score.std()
print(f"\n  f1(比价偏离): 均值={f1_mean:.4f}, 标准差={f1_std:.4f}")
print(f"  f2(动量加速度): 均值={f2_mean:.4f}, 标准差={f2_std:.4f}")
print(f"  score(综合): 均值={score_mean:.4f}, 标准差={score_std:.4f}")

# 信号方向分布
signal_growth = (score.shift(1) > 0).sum()
signal_value = (score.shift(1) <= 0).sum()
total_sig = signal_growth + signal_value
print(f"\n  信号方向(shift1后):")
print(f"    看多成长: {signal_growth} 天 ({signal_growth/total_sig*100:.1f}%)")
print(f"    看多价值: {signal_value} 天 ({signal_value/total_sig*100:.1f}%)")

# 信号稳定性（看信号转换频率）
sig_dir = score.shift(1).fillna(0)
flips = (sig_dir > 0).astype(int).diff().abs().sum() / 2
print(f"  信号多空切换次数(日频): {flips:.0f} 次")

# ============================================================
# 6. 分析哪些行情下策略表现好/差
# ============================================================
print("\n" + "=" * 70)
print("6. V72 策略分段表现分析")
print("=" * 70)

# 使用周频信号
df_wk = pd.DataFrame({
    "g": g, "v": v,
    "f1": f1, "f2": f2, "score": score
})
df_wk = df_wk.resample("W-FRI").last().dropna().iloc[1:]

# 回看每年收益
for yr in range(2013, 2026):
    gy = g[g.index.year == yr]
    vy = v[g.index.year == yr]
    if len(gy) < 2:
        continue
    g_yr = gy.iloc[-1] / gy.iloc[0] - 1
    v_yr = vy.iloc[-1] / vy.iloc[0] - 1
    excess = g_yr - v_yr
    # 信号方向比例
    sig_g = (score.loc[score.index.year == yr].shift(1) > 0).sum()
    sig_v = (score.loc[score.index.year == yr].shift(1) <= 0).sum()
    total = sig_g + sig_v
    print(f"  {yr}: 成长={g_yr*100:>6.1f}% 价值={v_yr*100:>6.1f}% "
          f"超额={excess*100:>+6.1f}% 信号偏成长={sig_g/total*100:.0f}%")
