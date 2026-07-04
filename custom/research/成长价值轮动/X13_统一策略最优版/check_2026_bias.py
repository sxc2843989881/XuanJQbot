"""检查2026年是否存在超买(G_BIAS或V_BIAS>19%)"""
import sys
sys.path.insert(0, r'c:\XuanJLH\Qbot\custom\backtests')
sys.path.insert(0, r'c:\XuanJLH\Qbot\custom\research\成长价值轮动\X12_偏离度空仓版')

import pandas as pd
from optimize_runner import G_CLOSE, V_CLOSE

# 计算BIAS
G_MA20 = G_CLOSE.rolling(20).mean()
V_MA20 = V_CLOSE.rolling(20).mean()
G_BIAS = G_CLOSE / G_MA20 - 1
V_BIAS = V_CLOSE / V_MA20 - 1

# 2026年数据
g_2026 = G_BIAS['2026']
v_2026 = V_BIAS['2026']

g_max = g_2026.max()
v_max = v_2026.max()

print(f"2026年 G_BIAS 最大值: {g_max*100:.2f}%")
print(f"2026年 V_BIAS 最大值: {v_max*100:.2f}%")
print(f"2026年 G_BIAS > 19% 的天数: {(g_2026 > 0.19).sum()}")
print(f"2026年 V_BIAS > 19% 的天数: {(v_2026 > 0.19).sum()}")

# 列出所有超买日期
g_over = g_2026[g_2026 > 0.19]
v_over = v_2026[v_2026 > 0.19]
if len(g_over) > 0:
    print(f"\n成长超买日期({len(g_over)}天):")
    for date, val in g_over.items():
        print(f"  {date.date()}: G_BIAS={val*100:.2f}%")
if len(v_over) > 0:
    print(f"\n价值超买日期({len(v_over)}天):")
    for date, val in v_over.items():
        print(f"  {date.date()}: V_BIAS={val*100:.2f}%")

if len(g_over) == 0 and len(v_over) == 0:
    print("\n2026年完全没有超买情况")
    # 看看全时段最大值
    print(f"\n全时段 G_BIAS 最大值: {G_BIAS.max()*100:.2f}% 发生在 {G_BIAS.idxmax().date()}")
    print(f"全时段 V_BIAS 最大值: {V_BIAS.max()*100:.2f}% 发生在 {V_BIAS.idxmax().date()}")
    
    # 最近一次超买
    recent_g = G_BIAS[G_BIAS > 0.19].tail(5)
    recent_v = V_BIAS[V_BIAS > 0.19].tail(5)
    if len(recent_g) > 0:
        print(f"\n成长最近超买日期: {recent_g.index[-1].date()} BIAS={recent_g.iloc[-1]*100:.2f}%")
    if len(recent_v) > 0:
        print(f"\n价值最近超买日期: {recent_v.index[-1].date()} BIAS={recent_v.iloc[-1]*100:.2f}%")

# 看看2026年BIAS的最高点到底是多少
print(f"\n2026年 G_BIAS 最高5天:")
g_2026_sorted = g_2026.sort_values(ascending=False)
for date, val in g_2026_sorted.head(5).items():
    print(f"  {date.date()}: {val*100:.2f}%")
print(f"\n2026年 V_BIAS 最高5天:")
v_2026_sorted = v_2026.sort_values(ascending=False)
for date, val in v_2026_sorted.head(5).items():
    print(f"  {date.date()}: {val*100:.2f}%")
