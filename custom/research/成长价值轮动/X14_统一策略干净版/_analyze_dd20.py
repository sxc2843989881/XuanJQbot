"""分析成长和价值的20日涨跌幅分布，找到合理的差异化阈值"""
import sys
sys.path.insert(0, r'c:\XuanJLH\Qbot\custom\research\成长价值轮动\X12_偏离度空仓版')
from optimize_runner import G_CLOSE, V_CLOSE
import numpy as np

g_ret20 = G_CLOSE / G_CLOSE.shift(20) - 1
v_ret20 = V_CLOSE / V_CLOSE.shift(20) - 1

g_ret20 = g_ret20.dropna()
v_ret20 = v_ret20.dropna()

print(f'{"":>20} {"成长":>10} {"价值":>10} {"比值":>10}')
print('-' * 50)
print(f'{"均值":>20} {g_ret20.mean()*100:>9.2f}% {v_ret20.mean()*100:>9.2f}% {g_ret20.mean()/v_ret20.mean():>10.2f}')
print(f'{"标准差":>20} {g_ret20.std()*100:>9.2f}% {v_ret20.std()*100:>9.2f}% {g_ret20.std()/v_ret20.std():>10.2f}')
print(f'{"最小值":>20} {g_ret20.min()*100:>9.2f}% {v_ret20.min()*100:>9.2f}%')
print(f'{"5%分位数":>20} {np.percentile(g_ret20,5)*100:>9.2f}% {np.percentile(v_ret20,5)*100:>9.2f}%')
print(f'{"10%分位数":>20} {np.percentile(g_ret20,10)*100:>9.2f}% {np.percentile(v_ret20,10)*100:>9.2f}%')
print(f'{"20%分位数":>20} {np.percentile(g_ret20,20)*100:>9.2f}% {np.percentile(v_ret20,20)*100:>9.2f}%')

# 统计各阈值下的触发比例
print()
print(f'触发比例（20日跌幅超过阈值的天数占比）:')
print(f'{"阈值":>10} {"成长":>10} {"价值":>10} {"成长/价值":>10}')
print('-' * 45)
for st_pct in [5, 6, 7, 8, 9, 10, 11, 12, 15, 20]:
    g_pct = (g_ret20 < -st_pct/100).mean() * 100
    v_pct = (v_ret20 < -st_pct/100).mean() * 100
    ratio = g_pct / v_pct if v_pct > 0 else float('inf')
    print(f'{-st_pct:>7}% {g_pct:>9.2f}% {v_pct:>9.2f}% {ratio:>9.1f}x')
