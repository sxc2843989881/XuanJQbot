"""compare_data_sources.py — 对比不同数据源"""
from pathlib import Path
import pandas as pd
import numpy as np

print("=" * 70)
print("  多数据源对比分析")
print("=" * 70)

sources = [
    ('temp_v72', Path(r'c:\temp_v72_data\index_480080.csv')),
    ('sxc_data', Path(r'c:\caches\sxc\style_rotation_strategy\data\index_480080.csv')),
    ('sxc_module', Path(r'c:\caches\sxc\style_rotation_strategy\data_module\data\index_480080.csv')),
    ('v72_research', Path(r'c:/XuanJLH/Qbot/custom/research/成长价值轮动/v72/项目代码/data/index_480080.csv')),
    ('X1_research', Path(r'c:/XuanJLH/Qbot/custom/research/成长价值轮动/X1_研究稳定基准版/项目代码/data/index_480080.csv')),
]
v_sources = [
    ('temp_v72', Path(r'c:\temp_v72_data\index_480081.csv')),
    ('sxc_data', Path(r'c:\caches\sxc\style_rotation_strategy\data\index_480081.csv')),
    ('sxc_module', Path(r'c:\caches\sxc\style_rotation_strategy\data_module\data\index_480081.csv')),
    ('X1_research', Path(r'c:/XuanJLH/Qbot/custom/research/成长价值轮动/X1_研究稳定基准版/项目代码/data/index_480081.csv')),
]

def load_idx(path, label):
    try:
        df = pd.read_csv(str(path))
        df['date'] = pd.to_datetime(df['date'])
        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        s = df.set_index('date')['close'].astype(float).sort_index()
        print(f"  [{label}] 路径: {path}")
        print(f"    区间: {s.index[0].date()} ~ {s.index[-1].date()}, {len(s)} 天")
        print(f"    最新价: {s.iloc[-1]:.2f}, 最早价: {s.iloc[0]:.2f}")
        return s
    except Exception as e:
        print(f"  [{label}] 加载失败: {e}")
        return None

print("\n--- 成长指数 480080 ---")
g_data = {}
for name, path in sources:
    s = load_idx(path, name)
    if s is not None:
        g_data[name] = s
    print()

print("--- 对比 ---")
g_names = list(g_data.keys())
for i in range(len(g_names)):
    for j in range(i+1, len(g_names)):
        n1, n2 = g_names[i], g_names[j]
        d1, d2 = g_data[n1], g_data[n2]
        common = d1.index.intersection(d2.index)
        val_diff = (d1.loc[common].values - d2.loc[common].values)
        n_diff = np.sum(np.abs(val_diff) > 0.001)
        max_diff = np.max(np.abs(val_diff))
        print(f"\n  [{n1}] vs [{n2}]:")
        print(f"    共同天数: {len(common)}, 不同天数: {n_diff}, 最大差异: {max_diff:.4f}")
        if n_diff > 0 and n_diff < 10:
            for dt in common[np.abs(val_diff) > 0.001][:5]:
                print(f"      {dt.date()}: [{n1}]={d1.loc[dt]:.2f}, [{n2}]={d2.loc[dt]:.2f}")

print("\n--- 总收益对比 ---")
for name, d in g_data.items():
    tr = (d.iloc[-1] / d.iloc[0] - 1) * 100
    yr = (d.index[-1] - d.index[0]).days / 365.25
    ann = ((1 + tr/100) ** (1/yr) - 1) * 100
    print(f"  [{name}] 总收益={tr:.2f}%, 年化={ann:.2f}%")

# 价值指数
print("\n--- 价值指数 480081 ---")
v_data = {}
for name, path in v_sources:
    s = load_idx(path, name)
    if s is not None:
        v_data[name] = s
    print()

print("--- 总收益对比 ---")
for name, d in v_data.items():
    tr = (d.iloc[-1] / d.iloc[0] - 1) * 100
    yr = (d.index[-1] - d.index[0]).days / 365.25
    ann = ((1 + tr/100) ** (1/yr) - 1) * 100
    print(f"  [{name}] 总收益={tr:.2f}%, 年化={ann:.2f}%")
