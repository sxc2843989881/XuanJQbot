"""analyze_extra_trades.py — 分析X23 z-score化后多出来的交易是否贡献收益"""
import sys
sys.path.insert(0, r'c:\XuanJLH\Qbot\custom\backtests')
sys.path.insert(0, r'c:\XuanJLH\Qbot\custom\research\成长价值轮动\X12_偏离度空仓版')

from pathlib import Path
import numpy as np
import pandas as pd

from optimize_runner import (
    G_CLOSE, V_CLOSE, RATIO, RATIO_MA20, RATIO_DEV, MA20_SLOPE,
    V_MOM20, G_DD20, V_DD20, BASE_DIR,
    run_backtest, calc_metrics, count_switches,
)

# z-score因子
RATIO_DEV_STD20 = RATIO_DEV.rolling(20).std()
RATIO_DEV_Z = RATIO_DEV / RATIO_DEV_STD20

Z_THRESH = 1.5
SLOPE_THRESH = 0.002
N_CONFIRM = 4
STOP_THRESHOLD = 0.10
STOP_WEIGHT = 0.30


def build_x18():
    """X18: 固定阈值0.3% + 斜率0.3% 双重确认"""
    dir_s = BASE_DIR.copy()
    mask = np.ones(len(dir_s), dtype=bool)
    for k in range(1, N_CONFIRM):
        mask = mask & (dir_s.values == dir_s.shift(k).values)
    confirmed = dir_s.where(mask, np.nan)

    wt = pd.Series(1.0, index=dir_s.index)
    low_dev = RATIO_DEV.abs() < 0.003
    low_slope = MA20_SLOPE.abs() < 0.003
    both_weak = low_dev & low_slope
    wt[both_weak] = 0.0

    dir_s = confirmed.ffill()
    wrong_value = (dir_s == 'value') & (V_MOM20 <= 0)
    dir_s[wrong_value] = 'growth'
    gs = (dir_s == 'growth') & (G_DD20 < -STOP_THRESHOLD)
    vs = (dir_s == 'value') & (V_DD20 < -STOP_THRESHOLD)
    wt[gs | vs] = wt[gs | vs] * STOP_WEIGHT
    return dir_s, wt


def build_x23_z():
    """X23(z=1.5): z-score化"""
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


def get_daily_returns(signal, weight):
    """计算每日组合收益"""
    common_idx = signal.index.intersection(G_CLOSE.index)
    sig = signal.loc[common_idx]
    wt = weight.loc[common_idx]
    g_c = G_CLOSE.loc[common_idx]
    v_c = V_CLOSE.loc[common_idx]
    mask = ~(sig.isna() | wt.isna())
    sig = sig[mask].astype(str)
    wt = wt[mask].astype(float)
    g_c = g_c[mask]
    v_c = v_c[mask]

    daily_ret = pd.Series(0.0, index=sig.index)
    prev_g = g_c.shift(1)
    prev_v = v_c.shift(1)
    g_ret = g_c / prev_g - 1
    v_ret = v_c / prev_v - 1
    is_growth = sig == 'growth'
    daily_ret.loc[is_growth] = g_ret[is_growth] * wt[is_growth]
    daily_ret.loc[~is_growth] = v_ret[~is_growth] * wt[~is_growth]
    return daily_ret.fillna(0).values


def extract_trades(signal, weight, result, label=""):
    """提取逐笔交易记录"""
    df = pd.DataFrame({'signal': signal.astype(str), 'weight': weight.astype(float)})
    df['prev_sig'] = df['signal'].shift(1)
    df['prev_wt'] = df['weight'].shift(1)
    df['wt_changed'] = (df['weight'] != df['prev_wt']) | (df['signal'] != df['prev_sig'])

    # 从回测结果获取每日收益
    df_r = result.to_dataframe()
    common_idx = df.index.intersection(df_r.index)
    daily_ret_vals = np.zeros(len(df))
    for i, idx in enumerate(df.index):
        if idx in df_r.index:
            daily_ret_vals[i] = df_r.loc[idx, 'daily_ret']
    df['daily_ret'] = daily_ret_vals
    df['equity'] = (1 + df['daily_ret']).cumprod()

    trades = []
    start_i = None
    entry_sig = None
    entry_wt = None

    for i in range(len(df)):
        row = df.iloc[i]
        if start_i is None:
            if row['wt_changed']:
                start_i = i
                entry_sig = row['signal']
                entry_wt = row['weight']
            continue

        if row['wt_changed']:
            exit_sig = row['signal']
            exit_wt = row['weight']

            # 交易期间收益
            chunk = df.iloc[start_i:i]
            if len(chunk) > 0:
                ret = chunk['daily_ret'].sum()
                hold = len(chunk)

                # 交易类型
                if entry_wt == 0 and exit_wt > 0:
                    ttype = '空仓恢复'
                elif entry_wt > 0 and exit_wt == 0:
                    ttype = '进入空仓'
                elif entry_sig != exit_sig:
                    ttype = '方向切换'
                else:
                    ttype = '仓位调整'

                trades.append({
                    'type': ttype,
                    'return': ret,
                    'hold_days': hold,
                    'start_date': df.iloc[start_i].name,
                    'end_date': df.iloc[i].name,
                    'entry_wt': entry_wt,
                    'exit_wt': exit_wt,
                })

            start_i = i
            entry_sig = exit_sig
            entry_wt = exit_wt

    return pd.DataFrame(trades)


# ============================================================
# 主分析
# ============================================================
if __name__ == '__main__':
    print("=" * 80)
    print("  X23 z-score化 vs X18固定阈值 — 交易增量分析")
    print("=" * 80)

    # 1. 构建策略
    sig18, wt18 = build_x18()
    sig23, wt23 = build_x23_z()

    # 2. 基础指标
    r18 = run_backtest(sig18, wt18); m18 = calc_metrics(r18); sw18 = count_switches(sig18, wt18)
    r23 = run_backtest(sig23, wt23); m23 = calc_metrics(r23); sw23 = count_switches(sig23, wt23)

    print(f"\n  X18(固定0.3%+0.3%): 年化{m18['ann']*100:.2f}% 回撤{m18['dd']*100:.2f}% Calmar{m18['calmar']:.3f} 交易{m18['n_trades']}次")
    print(f"  X23(z=1.5+0.2%):    年化{m23['ann']*100:.2f}% 回撤{m23['dd']*100:.2f}% Calmar{m23['calmar']:.3f} 交易{m23['n_trades']}次")
    print(f"  差值: 年化+{m23['ann']*100-m18['ann']*100:.2f}pp 回撤{m23['dd']*100-m18['dd']*100:.2f}pp 交易+{m23['n_trades']-m18['n_trades']}次")

    # 3. 提取交易记录
    t18 = extract_trades(sig18, wt18, r18, "X18")
    t23 = extract_trades(sig23, wt23, r23, "X23")

    print(f"\n  X18交易: {len(t18)}笔 (方向{sw18['dir']}+空仓{sw18['cash']})")
    print(f"  X23交易: {len(t23)}笔 (方向{sw23['dir']}+空仓{sw23['cash']})")

    # 4. 分析每日仓位差异
    # 合并两个策略的每日仓位
    df_compare = pd.DataFrame({
        'wt18': wt18, 'wt23': wt23,
        'sig18': sig18.astype(str), 'sig23': sig23.astype(str),
        'z_score': RATIO_DEV_Z,
        'dev': RATIO_DEV,
    }).dropna()

    # 仓位不同的天数
    diff = df_compare['wt18'] != df_compare['wt23']
    print(f"\n  仓位不同的天数: {diff.sum()} / {len(df_compare)} ({diff.mean()*100:.1f}%)")

    # 分类差异:
    # 类型A: X23空仓但X18不空仓 (z-score更敏感 → 额外空仓)
    # 类型B: X18空仓但X23不空仓 (固定阈值更敏感 → 减少的空仓)
    a_mask = (df_compare['wt23'] == 0) & (df_compare['wt18'] > 0)
    b_mask = (df_compare['wt18'] == 0) & (df_compare['wt23'] > 0)

    print(f"  类型A(X23额外空仓): {a_mask.sum()}天")
    print(f"  类型B(X23减少空仓): {b_mask.sum()}天")

    # z-score在这两类中的表现
    print(f"\n  类型A(X23额外空仓)时z-score均值: {df_compare.loc[a_mask, 'z_score'].abs().mean():.3f}")
    print(f"  类型B(X23减少空仓)时z-score均值: {df_compare.loc[b_mask, 'z_score'].abs().mean():.3f}")

    # 计算每日收益差异
    ret18 = get_daily_returns(sig18, wt18)[:len(df_compare)]
    ret23 = get_daily_returns(sig23, wt23)[:len(df_compare)]
    df_compare['ret18'] = ret18
    df_compare['ret23'] = ret23
    df_compare['ret_diff'] = ret23 - ret18

    # 类型A的收益贡献
    a_ret = df_compare.loc[a_mask, 'ret_diff'].sum()
    a_win = (df_compare.loc[a_mask, 'ret_diff'] > 0).mean()
    print(f"\n  [类型A] X23额外空仓({a_mask.sum()}天):")
    print(f"    累计收益贡献: {a_ret*100:.4f}pp")
    print(f"    胜率(当日): {a_win*100:.1f}%")

    # 类型B的收益贡献
    b_ret = df_compare.loc[b_mask, 'ret_diff'].sum()
    b_win = (df_compare.loc[b_mask, 'ret_diff'] > 0).mean()
    print(f"\n  [类型B] X23减少空仓({b_mask.sum()}天):")
    print(f"    累计收益贡献: {b_ret*100:.4f}pp")
    print(f"    胜率(当日): {b_win*100:.1f}%")

    # 相同仓位时的收益贡献
    same_mask = ~diff
    same_ret = df_compare.loc[same_mask, 'ret_diff'].sum()
    print(f"\n  [相同仓位时] ({same_mask.sum()}天):")
    print(f"    累计收益差异: {same_ret*100:.4f}pp (应接近0)")

    # 总收益差异分解
    total_diff = df_compare['ret_diff'].sum()
    print(f"\n  总收益差异分解:")
    print(f"    类型A贡献: {a_ret*100:.4f}pp ({a_ret/total_diff*100:.1f}%)")
    print(f"    类型B贡献: {b_ret*100:.4f}pp ({b_ret/total_diff*100:.1f}%)")
    print(f"    基准部分:  {same_ret*100:.4f}pp ({same_ret/total_diff*100:.1f}%)")
    print(f"    总差异:    {total_diff*100:.4f}pp")

    # 5. 按时间窗口分析z-score行为
    print("\n" + "-" * 80)
    print("  按波动率环境分析:")
    print("-" * 80)

    df_compare['vol_regime'] = pd.qcut(RATIO_DEV_STD20.loc[df_compare.index], 3,
                                        labels=['低波动', '中波动', '高波动'])
    for regime in ['低波动', '中波动', '高波动']:
        mask_r = df_compare['vol_regime'] == regime
        a_r = (mask_r & a_mask).sum()
        b_r = (mask_r & b_mask).sum()
        n_r = mask_r.sum()
        ret_r = df_compare.loc[mask_r, 'ret_diff'].sum()
        print(f"\n  {regime}({n_r}天):")
        print(f"    X23额外空仓{a_r}天 + X23减少空仓{b_r}天 = 净+{a_r-b_r}天空仓")
        print(f"    净收益贡献: {ret_r*100:.4f}pp")

    # 6. 直接看空仓相关额外交易的质量
    print("\n" + "-" * 80)
    print("  额外空仓交易质量分析:")
    print("-" * 80)

    # 找出X23有但X18没有的空仓事件
    in_extra_flat = False
    extra_flat_entries = []
    for i in range(len(df_compare)):
        if not in_extra_flat and a_mask.iloc[i]:
            in_extra_flat = True
            entry = i
            exit_i = i
        elif in_extra_flat and a_mask.iloc[i]:
            exit_i = i
        elif in_extra_flat and not a_mask.iloc[i]:
            in_extra_flat = False
            extra_flat_entries.append((entry, exit_i))
    if in_extra_flat:
        extra_flat_entries.append((entry, exit_i))

    if extra_flat_entries:
        extra_returns = []
        extra_hold = []
        for entry, exit_i in extra_flat_entries:
            # 空仓收益 = 现金收益(0) vs 如果不空仓
            ret_18 = df_compare['ret18'].iloc[entry:exit_i+1].sum()
            ret_23 = 0  # 空仓时收益为0
            extra_returns.append(ret_18 - ret_23)  # 不空仓 - 空仓 = 少赚的(正数=空仓正确)
            extra_hold.append(exit_i - entry + 1)

        extra_returns = np.array(extra_returns)
        win = (extra_returns > 0).sum()  # 空仓正确(不空仓亏更多)
        print(f"    X23额外空仓事件: {len(extra_flat_entries)}次")
        print(f"    空仓正确(规避下跌)次数: {win}({win/len(extra_flat_entries)*100:.1f}%)")
        print(f"    空仓错误(踏空上涨)次数: {len(extra_flat_entries)-win}")
        print(f"    平均空仓时长: {np.mean(extra_hold):.1f}天")
        print(f"    空仓正确时平均少亏: {extra_returns[extra_returns>0].mean()*100:.2f}%")
        print(f"    空仓错误时平均少赚: {extra_returns[extra_returns<=0].mean()*100:.2f}%")
        print(f"    净收益贡献(规避损失-踏空机会): {extra_returns.sum()*100:.4f}pp")

    # 7. 直接分析为什么z-score化导致更多交易
    print("\n" + "-" * 80)
    print("  为什么z-score化导致交易增加？")
    print("-" * 80)

    # 计算z=1.5对应的实际偏离度阈值范围
    z_equiv_dev = Z_THRESH * RATIO_DEV_STD20  # z=1.5对应多少偏离度
    print(f"\n  z={Z_THRESH}等效的偏离度阈值:")
    print(f"    均值: {z_equiv_dev.mean()*100:.3f}%")
    print(f"    最小值: {z_equiv_dev.min()*100:.3f}%")
    print(f"    25分位: {z_equiv_dev.quantile(0.25)*100:.3f}%")
    print(f"    中位数: {z_equiv_dev.median()*100:.3f}%")
    print(f"    75分位: {z_equiv_dev.quantile(0.75)*100:.3f}%")
    print(f"    最大值: {z_equiv_dev.max()*100:.3f}%")

    # vs 固定0.3%
    fixed_thresh = 0.003
    tighter_pct = (z_equiv_dev < fixed_thresh).mean() * 100  # z-score更严格(更小)的比例
    looser_pct = (z_equiv_dev > fixed_thresh).mean() * 100
    print(f"\n  与固定0.3%对比:")
    print(f"    z-score阈值比0.3%更严格(更容易触发)的比例: {tighter_pct:.1f}%")
    print(f"    z-score阈值比0.3%更宽松(更难触发)的比例: {looser_pct:.1f}%")

    # 净效果
    net = tighter_pct - looser_pct
    print(f"    净更严格(更易触发): {net:.1f}pp → {'更多' if net>0 else '更少'}空仓触发")

    print("\n" + "=" * 80)
    print("  分析完成")
    print("=" * 80)
