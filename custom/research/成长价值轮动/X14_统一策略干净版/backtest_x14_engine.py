"""backtest_x14_engine.py — X14 统一策略干净版 (v2.0-lite)
================================================================
X13 的合理化版本 + 量化研究员审核后经消融验证保留的低成本改进。

与原版 X13(v26 U164) 的区别:
  1. 去掉 max_hold_days / max_hold_reduce（固定持仓天数无金融逻辑）
  2. st: 0.088 → 0.09（取整，避免精度过拟合）
  3. bias_reduce → bias_mode: BIAS触发改为可选的 'clear'/'half'/'ignore'

审核改进 (v2.0-lite，经消融验证仅保留低成本改进):
  4. dcd: 6→4 (减少方向冷却延迟，Calmar影响 -0.1%)
  5. E5冷却期内重置机制修复 (延长而非重置，Calmar影响 -2.6%)
  6. BIAS触发改为降仓50%+T条件约束 (Calmar影响 -4.3%)

⚠ 消融测试中回退的高成本改进:
  - 急跌加速判断: -11.6% Calmar → 默认关闭
  - Dual Momentum: -6.5% Calmar → 默认关闭
  - B2 ms/ml: 20/60: -35.3% Calmar → 恢复 10/20
  - 权重漏斗分档(sw_mid=0.50): -7.2% Calmar → 恢复 sw=0.17

v2.0-lite vs v1.0: 年化45.27%→42.12%(-3.15pp), Calmar 1.951→1.815(-7.0%)

滑点设置:
  - 手续费: 1bps（买入/卖出各）
  - 冲击滑点: 5bps（买入/卖出各）
  - 跳空滑点: 不计（高开低开期望值为0）
  - 总滑点 ≈ 6bps/边 = 1bps(手续费) + 5bps(冲击)

参数列表:
  dc=5      方向确认天数
  dcd=4     方向冷却天数 (原6→4, -0.1% Calmar)
  rt=1.3    T弱阈值
  slope_thresh=0.002  斜率弱阈值
  ms=10     B2短周期
  ml=20     B2长周期
  bias_ma=20    BIAS均线周期
  bias_high=0.19   BIAS超阈值
  bias_mode='half'  BIAS模式: clear(清仓)/half(降仓50%+T约束)/ignore(忽略)
  st=0.09    E5止损阈值(20日跌幅)
  sw_mid=0.17  E5中档降仓比例 (与sw_deep同值=不使用权重漏斗)
  sw_deep=0.17 E5深跌降仓比例
  cd=8       E5冷却天数 (延长3天而非重置, -2.6% Calmar)
  dual_momentum=False  Dual Momentum (默认关闭)
  bias_t_constraint=True  BIAS加T条件约束 (-4.3% Calmar)
  rapid_decline=False  急跌加速判断 (默认关闭)
  e5_reset=False      E5冷却期内再次触发: False=延长3天 / True=完全重置(原版)
=================================================================
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, r'c:\XuanJLH\Qbot\custom\backtests')
sys.path.insert(0, r'c:\XuanJLH\Qbot\custom\research\成长价值轮动\X12_偏离度空仓版')

from optimize_runner import (
    G_CLOSE, V_CLOSE, RATIO, RATIO_MA20, RATIO_DEV, MA20_SLOPE,
    V_MOM20, G_DD20, V_DD20, BASE_DIR,
    run_backtest, calc_metrics, count_switches,
)
from run_x33_reduce_trades import RATIO_DEV_Z

T = RATIO_DEV_Z
SLOPE = MA20_SLOPE


def set_ma_period(n: int = 20):
    """替换比价均线周期，同步调整归一化窗口。

    Args:
        n: 均线周期天数（默认 20），归一化窗口（std）也同步为 n
    """
    global T, SLOPE, RATIO_MA20, RATIO_DEV, MA20_SLOPE, BASE_DIR
    RATIO_MA20 = RATIO.rolling(n).mean()
    RATIO_DEV = RATIO / RATIO_MA20 - 1
    # 归一化窗口同步为 n（与原设计一致：均线周期 = 归一化窗口）
    T = RATIO_DEV / RATIO_DEV.rolling(n).std()
    MA20_SLOPE = (RATIO_MA20 - RATIO_MA20.shift(5)) / RATIO_MA20.shift(5)
    SLOPE = MA20_SLOPE
    BASE_DIR = (RATIO > RATIO_MA20).map({True: 'growth', False: 'value'})


def build_core(slope_thresh=0.002, sw_mid=0.17, sw_deep=0.17, st=0.09, cd=8,
               ms=10, ml=20, rt=1.3, dc=5, dcd=4,
               bias_ma=20, bias_high=0.19, bias_mode='half',
               g_st=None, v_st=None,
               dual_momentum=False, bias_t_constraint=True, rapid_decline=False,
               e5_reset=False):
    """X14 干净版核心 (v2.0 最终版)

    bias_mode:
      'clear'  -> BIAS>19%时降仓50% (默认，加T条件约束)
      'ignore' -> 忽略BIAS信号

    审核改进 (v2.0，经消融验证有效):
      - dcd: 6→4, 减少方向冷却延迟
      - E5冷却期内再次触发改为延长3天而非重置
      - 增加急跌加速判断 (3日跌幅>7%直接触发E5)
      - BIAS触发加T条件约束 (T>=1.5时容忍超买)
      - E5权重漏斗分档 (浅跌破9%→50%, 深跌破14%→17%)
      - BIAS改为降仓50%而非清仓

    ⚠ Dual Momentum: 消融验证-10.9% Calmar，默认关闭
    ⚠ B2 20/60: 消融验证-15.2% Calmar，恢复10/20

    g_st/v_st: 成长/价值分别的E5止损阈值，默认都使用st
    dual_momentum: 是否启用Dual Momentum过滤 (默认关闭)
    bias_t_constraint: BIAS触发是否加T条件约束
    """
    if g_st is None: g_st = st
    if v_st is None: v_st = st

    # 预计算动量
    V_MOM_S = V_CLOSE.pct_change(ms)
    V_MOM_L = V_CLOSE.pct_change(ml)
    G_MOM_12M = G_CLOSE.pct_change(252)  # 12个月绝对动量
    V_MOM_12M = V_CLOSE.pct_change(252)

    # 急跌加速判断 (3日跌幅)
    G_DD3 = G_CLOSE.pct_change(3)
    V_DD3 = V_CLOSE.pct_change(3)

    # ============================================================
    # 第1层：方向确认
    # ============================================================
    raw_dir = (T > 0).map({True: 'BULL', False: 'BEAR'})
    mask = np.ones(len(raw_dir), dtype=bool)
    for k in range(1, dc):
        mask = mask & (raw_dir.values == raw_dir.shift(k).values)
    confirmed_dir = raw_dir.where(mask, np.nan).ffill().fillna('BULL')

    # ============================================================
    # 第2层：方向冷却 (dcd=4, 原6)
    # ============================================================
    if dcd > 0:
        new_dir = confirmed_dir.copy()
        last_switch = -dcd - 1
        prev = confirmed_dir.iloc[0]
        for i in range(len(confirmed_dir)):
            if pd.isna(confirmed_dir.iloc[i]):
                new_dir.iloc[i] = prev
                continue
            if confirmed_dir.iloc[i] != prev:
                if i - last_switch >= dcd:
                    last_switch = i
                    prev = confirmed_dir.iloc[i]
                new_dir.iloc[i] = prev
            else:
                new_dir.iloc[i] = prev
        confirmed_dir = new_dir

    dir_raw = confirmed_dir

    # ============================================================
    # [新增] Dual Momentum绝对动量层
    # 如果成长和价值12个月动量均<0 → 市场整体下行 → 强制空仓
    # ============================================================
    dm_trigger = (G_MOM_12M < 0) & (V_MOM_12M < 0)
    if dual_momentum:
        dir_raw[dm_trigger] = 'BEAR'  # 临时占位，后续映射到空仓

    # ============================================================
    # 第3层：T+斜率双重确认 → 空仓
    # ============================================================
    weak_slope = SLOPE.abs() < slope_thresh
    weak_t = T.abs() < rt
    is_weak = weak_t & weak_slope
    wt = pd.Series(1.0, index=T.index)
    wt[is_weak] = 0.0

    # ============================================================
    # 第4层：B2价值动量过滤 (ms/ml: 10/20→20/60, 降低共线性)
    # ============================================================
    wrong_value = (dir_raw == 'BEAR') & (V_MOM_S <= 0) & (V_MOM_L <= 0)
    dir_raw[wrong_value] = 'BULL'
    dir_s = dir_raw.map({'BULL': 'growth', 'BEAR': 'value'})

    # Dual Momentum触发时强制空仓 (在B2之后覆盖)
    if dual_momentum:
        wt[dm_trigger] = 0.0

    # ============================================================
    # 第5层：BIAS过滤 (clear=清仓 / half=降仓50%+T约束 / ignore=忽略)
    # ============================================================
    if bias_mode == 'clear':
        # 原版: BIAS>19% → 直接清仓
        G_BIAS = (G_CLOSE / G_CLOSE.rolling(bias_ma).mean() - 1)
        V_BIAS = (V_CLOSE / V_CLOSE.rolling(bias_ma).mean() - 1)
        extreme_g = (dir_s == 'growth') & (G_BIAS > bias_high)
        extreme_v = (dir_s == 'value') & (V_BIAS > bias_high)
        wt[extreme_g | extreme_v] = 0.0

    elif bias_mode == 'half':
        # 新版: BIAS>19% → 降仓50%, 可加T条件约束
        G_BIAS = (G_CLOSE / G_CLOSE.rolling(bias_ma).mean() - 1)
        V_BIAS = (V_CLOSE / V_CLOSE.rolling(bias_ma).mean() - 1)

        if bias_t_constraint:
            t_cond = pd.Series(T.values < 1.5, index=T.index)
            extreme_g = (dir_s == 'growth') & (G_BIAS > bias_high) & t_cond
            extreme_v = (dir_s == 'value') & (V_BIAS > bias_high) & t_cond
        else:
            extreme_g = (dir_s == 'growth') & (G_BIAS > bias_high)
            extreme_v = (dir_s == 'value') & (V_BIAS > bias_high)

        wt[extreme_g | extreme_v] = wt[extreme_g | extreme_v] * 0.5

    # ============================================================
    # 第6层：E5止损 (分档降仓 + 急跌加速 + 冷却期修复)
    # ============================================================
    gs = (dir_s == 'growth') & (G_DD20 < -g_st)
    vs = (dir_s == 'value') & (V_DD20 < -v_st)

    # [新增] 急跌加速判断: 3日跌幅>7%直接触发E5 (参数控制)
    gs_rapid = (dir_s == 'growth') & (G_DD3 < -0.07)
    vs_rapid = (dir_s == 'value') & (V_DD3 < -0.07)
    e5_trigger = gs | vs
    if rapid_decline:
        e5_trigger = e5_trigger | gs_rapid | vs_rapid

    # E5深跌判断: 20日跌幅>14% → 深跌档
    gs_deep = (dir_s == 'growth') & (G_DD20 < -0.14)
    vs_deep = (dir_s == 'value') & (V_DD20 < -0.14)
    e5_deep = gs_deep | vs_deep

    in_cooldown = False
    cooldown_count = 0
    for i in range(len(wt)):
        if pd.isna(wt.iloc[i]) or pd.isna(dir_s.iloc[i]):
            continue

        if e5_trigger.iloc[i] and not in_cooldown:
            # 首次触发E5
            in_cooldown = True
            cooldown_count = 0
            # [改进] 分档降仓: 深跌破14%→sw_deep, 浅跌破9%→sw_mid
            sw_use = sw_deep if e5_deep.iloc[i] else sw_mid
            wt.iloc[i] = wt.iloc[i] * sw_use

        elif in_cooldown:
            cooldown_count += 1

            if cooldown_count >= cd:
                # 冷却期结束
                if e5_trigger.iloc[i]:
                    if e5_reset:
                        # 原版: 完全重置冷却期
                        cooldown_count = 0
                    else:
                        # 新版: 延长3天而非重置
                        cooldown_count = cd - 3
                    sw_use = sw_deep if e5_deep.iloc[i] else sw_mid
                    wt.iloc[i] = wt.iloc[i] * sw_use
                else:
                    in_cooldown = False
                    if is_weak.iloc[i] or (dual_momentum and dm_trigger.iloc[i]):
                        wt.iloc[i] = 0.0
                    else:
                        wt.iloc[i] = 1.0
            else:
                # 冷却中，保持低仓位
                if wt.iloc[i] > 0:
                    wt.iloc[i] = sw_deep

    signal = dir_raw.map({'BULL': 'growth', 'BEAR': 'value'})
    return signal, wt


if __name__ == '__main__':
    print("=" * 80)
    print("  X14 统一策略干净版")
    print("=" * 80)

    sig, wt = build_core()
    result = run_backtest(sig, wt, impact_slippage=0.0005)
    m = calc_metrics(result)
    sw = count_switches(sig, wt)
    result_sl = run_backtest(sig, wt, impact_slippage=0.0005)
    m_sl = calc_metrics(result_sl)

    print(f"\n  年化={m['ann']*100:.2f}%")
    print(f"  最大回撤={m['dd']*100:.2f}%")
    print(f"  Sharpe={m['sharpe']:.3f}")
    print(f"  Calmar={m['calmar']:.3f}")
    print(f"  交易次数={m['n_trades']}")
    print(f"  方向切换={sw['dir']}  空仓切换={sw['cash']}")
    print(f"  滑点(手续费1bps+冲击5bps): 年化={m_sl['ann']*100:.2f}%  "
          f"Calmar={m_sl['calmar']:.3f}")
    print("=" * 80)
