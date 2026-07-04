"""X14 v2.0 Walk-Forward 验证
====================================================
滚动窗口: 3年训练 + 1年测试 (步长1年)
评估各窗口测试期的一致性。
====================================================
"""
import sys
sys.path.insert(0, r'c:\XuanJLH\Qbot\custom\backtests')
sys.path.insert(0, r'c:\XuanJLH\Qbot\custom\research\成长价值轮动\X12_偏离度空仓版')
sys.path.insert(0, r'c:\XuanJLH\Qbot\custom\research\成长价值轮动\X14_统一策略干净版')

import numpy as np
import pandas as pd
from optimize_runner import (
    G_CLOSE, V_CLOSE, RATIO, RATIO_MA20, RATIO_DEV, MA20_SLOPE,
    V_MOM20, G_DD20, V_DD20, BASE_DIR, run_backtest, calc_metrics
)
from run_x33_reduce_trades import RATIO_DEV_Z
from backtest_x14_engine import build_core

T = RATIO_DEV_Z

print("=" * 80)
print("  X14 v2.0 Walk-Forward 验证")
print("=" * 80)

# 确定年份范围
years = sorted(set(d.year for d in T.index))
print(f"\n  数据年份范围: {years[0]} ~ {years[-1]}")

# 滚动窗口: 3年训练 + 1年测试, 步长1年
TRAIN_YEARS = 3
TEST_YEARS = 1
STEP = 1

all_windows = []
start_year = years[0]

while start_year + TRAIN_YEARS + TEST_YEARS <= years[-1] + 1:
    train_end = start_year + TRAIN_YEARS
    test_end = train_end + TEST_YEARS
    
    train_mask = (T.index.year >= start_year) & (T.index.year < train_end)
    test_mask = (T.index.year >= train_end) & (T.index.year < test_end)
    
    if train_mask.sum() == 0 or test_mask.sum() == 0:
        start_year += STEP
        continue
    
    all_windows.append({
        'train': f'{start_year}-{train_end}',
        'test': f'{train_end}-{test_end}',
        'train_mask': train_mask,
        'test_mask': test_mask,
    })
    start_year += STEP

print(f"  Walk-Forward窗口数: {len(all_windows)}")
print()

# 运行每个窗口
all_train_results = []
all_test_results = []

for w_idx, w in enumerate(all_windows):
    print(f"\n--- 窗口 {w_idx+1}/{len(all_windows)}: 训练={w['train']}, 测试={w['test']} ---")
    
    # 训练期
    sig_train, wt_train = build_core()
    # 截取训练期
    train_idx = T.index[w['train_mask']]
    sig_train_sub = sig_train.loc[train_idx]
    wt_train_sub = wt_train.loc[train_idx]
    
    r_train = run_backtest(sig_train_sub, wt_train_sub, impact_slippage=0.0005)
    m_train = calc_metrics(r_train)
    all_train_results.append({
        'window': f"{w['train']}→{w['test']}",
        'ann': m_train['ann'],
        'dd': m_train['dd'],
        'sharpe': m_train['sharpe'],
        'calmar': m_train['calmar'],
        'n_trades': m_train['n_trades'],
    })
    
    # 测试期
    test_idx = T.index[w['test_mask']]
    sig_test_sub = sig_train.loc[test_idx]
    wt_test_sub = wt_train.loc[test_idx]
    
    r_test = run_backtest(sig_test_sub, wt_test_sub, impact_slippage=0.0005)
    m_test = calc_metrics(r_test)
    all_test_results.append({
        'window': f"{w['train']}→{w['test']}",
        'ann': m_test['ann'],
        'dd': m_test['dd'],
        'sharpe': m_test['sharpe'],
        'calmar': m_test['calmar'],
        'n_trades': m_test['n_trades'],
    })
    
    print(f"  训练: 年化={m_train['ann']*100:.1f}%  Calmar={m_train['calmar']:.3f}")
    print(f"  测试: 年化={m_test['ann']*100:.1f}%  Calmar={m_test['calmar']:.3f}")

# 汇总
print("\n" + "=" * 80)
print("  Walk-Forward 汇总")
print("=" * 80)

print(f"\n  {'窗口':<20} {'训练年化':>10} {'训练Calmar':>14} {'测试年化':>10} {'测试Calmar':>14}")
print("  " + "-" * 70)
for i in range(len(all_windows)):
    tr = all_train_results[i]
    te = all_test_results[i]
    print(f"  {tr['window']:<20} {tr['ann']*100:>9.1f}% {tr['calmar']:>13.3f} "
          f"{te['ann']*100:>9.1f}% {te['calmar']:>13.3f}")

# 统计
print(f"\n  训练期统计:")
train_anns = [r['ann'] for r in all_train_results]
train_calmars = [r['calmar'] for r in all_train_results]
print(f"    年化: {np.mean(train_anns)*100:.1f}% ± {np.std(train_anns)*100:.1f}%")
print(f"    Calmar: {np.mean(train_calmars):.3f} ± {np.std(train_calmars):.3f}")

print(f"\n  测试期统计:")
test_anns = [r['ann'] for r in all_test_results if r['ann'] != 0]
test_calmars = [r['calmar'] for r in all_test_results if r['calmar'] != 0]
print(f"    年化: {np.mean(test_anns)*100:.1f}% ± {np.std(test_anns)*100:.1f}%")
print(f"    Calmar: {np.mean(test_calmars):.3f} ± {np.std(test_calmars):.3f}")

# 全样本结果
print(f"\n  全样本结果:")
sig_full, wt_full = build_core()
r_full = run_backtest(sig_full, wt_full, impact_slippage=0.0005)
m_full = calc_metrics(r_full)
print(f"    年化: {m_full['ann']*100:.2f}%")
print(f"    回撤: {m_full['dd']*100:.2f}%")
print(f"    Calmar: {m_full['calmar']:.3f}")

# 一致性评估
test_calmars_pos = [c for c in test_calmars if c > 0]
print(f"\n  一致性评估:")
print(f"    测试期Calmar均值: {np.mean(test_calmars):.3f}")
print(f"    全样本Calmar: {m_full['calmar']:.3f}")
print(f"    正Calmar窗口占比: {len(test_calmars_pos)}/{len(test_calmars)} ({len(test_calmars_pos)/len(test_calmars)*100:.0f}%)")
print(f"    Calmar衰减: {(m_full['calmar']/np.mean(test_calmars) - 1)*100:+.1f}%")

print("\n" + "=" * 80)
print("  Walk-Forward 验证完成")
print("=" * 80)
