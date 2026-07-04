"""unified_strategy_test.py — 统一三态状态机策略测试
================================================================
将"方向判断"+"趋势判断"合成为一个T指标三态状态机
T = RATIO_DEV_Z = (RATIO - RATIO_MA20) / RATIO_DEV_STD20

三态:
- BULL  (满仓growth): T > +entry
- BEAR  (满仓value):  T < -entry
- NEUTRAL (空仓):     |T| < exit (滞回带)

外挂:
- B2: BEAR + V_MOM20<=0 → 强制BULL
- E5: 持仓20日跌>10% → 降仓30% + 冷却N天

目标: 年化>40%, 回撤<-35%, 交易次数 < X61(224次)
================================================================
"""
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
from run_x33_reduce_trades import RATIO_DEV_STD20, RATIO_DEV_Z

OUTPUT_DIR = Path(r'c:\XuanJLH\Qbot\custom\research\成长价值轮动\X12_偏离度空仓版')
T = RATIO_DEV_Z  # 统一指标


# ============================================================
# 统一三态状态机
# ============================================================
def build_unified(entry_thresh=1.5, exit_thresh=0.5,
                  use_b2=True, use_e5=True,
                  stop_threshold=0.10, stop_weight=0.30,
                  e5_cooldown=5):
    """统一三态状态机策略

    Args:
        entry_thresh: 进入BULL/BEAR的T阈值 (|T|>entry)
        exit_thresh: 退出BULL/BEAR的T阈值 (|T|<exit), exit<entry形成滞回
        use_b2: 是否启用B2方向修正
        use_e5: 是否启用E5止损
        stop_threshold: E5触发阈值
        stop_weight: E5降仓后权重
        e5_cooldown: E5冷却天数
    Returns:
        signal (Series: 'growth'/'value'), weight (Series: 0~1)
    """
    states = []
    state = 'NEUTRAL'
    for i in range(len(T)):
        t = T.iloc[i]
        if pd.isna(t):
            states.append(state)
            continue

        if state == 'BULL':
            if t < exit_thresh:
                state = 'BEAR' if t < -entry_thresh else 'NEUTRAL'
        elif state == 'BEAR':
            if t > -exit_thresh:
                state = 'BULL' if t > entry_thresh else 'NEUTRAL'
        else:  # NEUTRAL
            if t > entry_thresh:
                state = 'BULL'
            elif t < -entry_thresh:
                state = 'BEAR'
        states.append(state)

    states = pd.Series(states, index=T.index)

    # 方向: NEUTRAL时ffill上次方向
    dir_raw = states.where(states != 'NEUTRAL', np.nan)
    dir_raw = dir_raw.ffill()
    # 初始NEUTRAL段填充为BULL(默认)
    dir_raw = dir_raw.fillna('BULL')

    # 权重: NEUTRAL=0, 其他=1
    wt = pd.Series(1.0, index=T.index)
    wt[states == 'NEUTRAL'] = 0.0

    # B2: BEAR + 价值下跌 → 强制BULL (只改方向不改权重)
    if use_b2:
        wrong_value = (dir_raw == 'BEAR') & (V_MOM20 <= 0)
        dir_raw[wrong_value] = 'BULL'

    # E5: 持仓20日跌>阈值 → 降仓 + 冷却
    if use_e5:
        dir_s = dir_raw.map({'BULL': 'growth', 'BEAR': 'value'})
        gs = (dir_s == 'growth') & (G_DD20 < -stop_threshold)
        vs = (dir_s == 'value') & (V_DD20 < -stop_threshold)
        e5_trigger = gs | vs

        in_cooldown = False
        cooldown_count = 0
        for i in range(len(wt)):
            if pd.isna(wt.iloc[i]) or pd.isna(dir_s.iloc[i]):
                continue
            if e5_trigger.iloc[i] and not in_cooldown:
                in_cooldown = True
                cooldown_count = 0
                wt.iloc[i] = wt.iloc[i] * stop_weight
            elif in_cooldown:
                cooldown_count += 1
                if cooldown_count >= e5_cooldown:
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

    # 映射方向到信号
    signal = dir_raw.map({'BULL': 'growth', 'BEAR': 'value'})
    return signal, wt


# ============================================================
# 测试单个策略
# ============================================================
def test_unified(name, params, desc=""):
    sig, wt = build_unified(**params)
    result = run_backtest(sig, wt)
    m = calc_metrics(result)
    sw = count_switches(sig, wt)

    # 5/万滑点
    result_sl = run_backtest(sig, wt, impact_slippage=0.0005)
    m_sl = calc_metrics(result_sl)

    info = {
        'name': name, 'desc': desc, 'params': params,
        'ann': m['ann'], 'dd': m['dd'], 'sharpe': m['sharpe'],
        'calmar': m['calmar'], 'n_trades': m['n_trades'],
        'dir_sw': sw['dir'], 'cash_sw': sw['cash'],
        'ann_sl': m_sl['ann'], 'calmar_sl': m_sl['calmar'],
    }
    return info


def print_info(info):
    print(f"  {info['name']}: {info['desc']}")
    print(f"    年化={info['ann']*100:.2f}% 回撤={info['dd']*100:.2f}% "
          f"Sharpe={info['sharpe']:.3f} Calmar={info['calmar']:.3f} "
          f"交易={info['n_trades']}(方向{info['dir_sw']}+空仓{info['cash_sw']})")
    print(f"    5/万滑点: 年化={info['ann_sl']*100:.2f}% Calmar={info['calmar_sl']:.3f}")


# ============================================================
# 多轮测试
# ============================================================
if __name__ == '__main__':
    print("=" * 80)
    print("  统一三态状态机策略测试 (U1-U12)")
    print("=" * 80)

    results = []

    # ---- 第一阶段: 基础测试(逐步加组件) ----
    print("\n--- 第一阶段: 组件逐步加入 ---")

    # U1: 纯状态机(无B2无E5)
    info = test_unified("U1", dict(entry_thresh=1.5, exit_thresh=0.5,
                                    use_b2=False, use_e5=False),
                        "纯状态机 entry=1.5 exit=0.5")
    print_info(info); results.append(info)

    # U2: +B2
    info = test_unified("U2", dict(entry_thresh=1.5, exit_thresh=0.5,
                                    use_b2=True, use_e5=False),
                        "+B2方向修正")
    print_info(info); results.append(info)

    # U3: +B2+E5
    info = test_unified("U3", dict(entry_thresh=1.5, exit_thresh=0.5,
                                    use_b2=True, use_e5=True,
                                    stop_threshold=0.10, stop_weight=0.30,
                                    e5_cooldown=5),
                        "+B2+E5完整版")
    print_info(info); results.append(info)

    # ---- 第二阶段: 滞回带宽度扫描 ----
    print("\n--- 第二阶段: 滞回带宽度扫描 ---")

    for et, xt in [(1.5, 0.3), (1.5, 0.8), (1.5, 1.0), (1.5, 0.0)]:
        name = f"U4_e1.5_x{xt}"
        info = test_unified(name, dict(entry_thresh=1.5, exit_thresh=xt,
                                        use_b2=True, use_e5=True,
                                        e5_cooldown=5),
                            f"entry=1.5 exit={xt}")
        print_info(info); results.append(info)

    # ---- 第三阶段: entry阈值扫描 ----
    print("\n--- 第三阶段: entry阈值扫描 ---")

    for et in [1.0, 1.2, 1.8, 2.0]:
        name = f"U5_e{et}_x0.5"
        info = test_unified(name, dict(entry_thresh=et, exit_thresh=0.5,
                                        use_b2=True, use_e5=True,
                                        e5_cooldown=5),
                            f"entry={et} exit=0.5")
        print_info(info); results.append(info)

    # ---- 第四阶段: E5参数扫描 ----
    print("\n--- 第四阶段: E5参数扫描 ---")

    for st, sw, cd in [(0.08, 0.30, 5), (0.10, 0.20, 5), (0.10, 0.30, 3),
                        (0.10, 0.30, 8), (0.12, 0.30, 5)]:
        name = f"U6_st{st}_sw{sw}_cd{cd}"
        info = test_unified(name, dict(entry_thresh=1.5, exit_thresh=0.5,
                                        use_b2=True, use_e5=True,
                                        stop_threshold=st, stop_weight=sw,
                                        e5_cooldown=cd),
                            f"E5: stop={st} wt={sw} cool={cd}")
        print_info(info); results.append(info)

    # ---- 汇总 ----
    print("\n" + "=" * 80)
    print("  汇总对比 (按Calmar降序)")
    print("=" * 80)
    print(f"  {'名称':<22} {'年化':>7} {'回撤':>7} {'Sharpe':>7} {'Calmar':>7} "
          f"{'交易':>5} {'方向':>5} {'空仓':>5} {'滑点年化':>8}")
    print("  " + "-" * 90)

    # 加入X61基准
    print(f"  {'X61(基准)':<22} {'41.12%':>7} {'-30.21%':>7} {'1.357':>7} "
          f"{'1.361':>7} {'224':>5} {'179':>5} {'20':>5} {'39.04%':>8}")

    results.sort(key=lambda x: -x['calmar'])
    for r in results:
        print(f"  {r['name']:<22} {r['ann']*100:>6.2f}% {r['dd']*100:>6.2f}% "
              f"{r['sharpe']:>7.3f} {r['calmar']:>7.3f} {r['n_trades']:>5} "
              f"{r['dir_sw']:>5} {r['cash_sw']:>5} {r['ann_sl']*100:>7.2f}%")

    # 筛选满足约束的
    print("\n  满足约束(年化>40% 回撤<-35%):")
    ok = [r for r in results if r['ann'] > 0.40 and r['dd'] > -0.35]
    if ok:
        ok.sort(key=lambda x: x['n_trades'])
        for r in ok:
            print(f"    {r['name']}: 年化{r['ann']*100:.2f}% 回撤{r['dd']*100:.2f}% "
                  f"交易{r['n_trades']}次 Calmar{r['calmar']:.3f}")
    else:
        print("    无版本满足约束")

    # 保存结果
    df = pd.DataFrame(results)
    df.to_csv(str(OUTPUT_DIR / 'unified_test_results.csv'), index=False, encoding='utf-8')
    print(f"\n  结果已保存: {OUTPUT_DIR / 'unified_test_results.csv'}")
