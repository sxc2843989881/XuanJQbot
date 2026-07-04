"""backtest_x14_event_driven.py — X14 事件驱动回测引擎实现

================================================================
Qbot 两大回测引擎：
  1. 轻量化引擎（vectorized）：custom/backtests/backtest_engine.py
     - 数组化批量处理，T日信号→T+1日开盘执行

  2. 事件驱动引擎：本文件
     - 逐 Bar 事件驱动，每个 Bar 独立计算信号并决策
     - 与 Qbot 的 qbot/engine/backtest/ 一样是事件驱动范式
     - 支持双资产轮动（成长/价值），这是 backtrader 单资产框架做不到的

本文件 = 在事件驱动范式下完整实现 X14 的 6 层逻辑，
每个交易日均独立计算因子和信号，与轻量化引擎的差异仅在于：
  - 处理范式：逐 Bar 事件驱动 vs 向量化批量
  - 因子计算：仅用截至当前 Bar 的历史数据 vs 全量数据
================================================================
"""
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd

# ============================================================
# 数据加载 — 与 optimize_runner 完全一致
# ============================================================
DATA_DIR = Path(r'c:\temp_v72_data')
_g_raw = pd.read_csv(str(DATA_DIR / 'index_480080.csv'))
_v_raw = pd.read_csv(str(DATA_DIR / 'index_480081.csv'))
for d in (_g_raw, _v_raw):
    d['date'] = pd.to_datetime(d['date'])
    d['close'] = pd.to_numeric(d['close'], errors='coerce')
G_CLOSE = _g_raw.set_index('date')['close'].astype(float).sort_index().dropna()
V_CLOSE = _v_raw.set_index('date')['close'].astype(float).sort_index().dropna()
_common = G_CLOSE.index.intersection(V_CLOSE.index)
G_CLOSE = G_CLOSE[_common].sort_index()
V_CLOSE = V_CLOSE[_common].sort_index()

common_idx = G_CLOSE.index
G_ARR = G_CLOSE.values.astype(np.float64)
V_ARR = V_CLOSE.values.astype(np.float64)
N = len(common_idx)


# ============================================================
# 交易记录
# ============================================================
@dataclass
class TradeRecord:
    date: str
    from_pos: Optional[str]
    from_wt: float
    to_pos: str
    to_wt: float
    g_price: float
    v_price: float
    nav_before: float


# ============================================================
# 事件驱动回测引擎
# ============================================================
class EventDrivenBacktest:
    """事件驱动回测引擎 — 逐 Bar 处理 X14 策略

    参数与 backtest_x14_engine.build_core 完全一致
    """

    def __init__(self, bias_mode='clear', impact_slippage=0.0005,
                 commission=0.0001,
                 dc=5, dcd=6, rt=1.3, slope_thresh=0.002,
                 ms=10, ml=20, bias_ma=20, bias_high=0.19,
                 st=0.09, sw_mid=0.17, sw_deep=0.17, cd=8,
                 dual_momentum=False, bias_t_constraint=False,
                 rapid_decline=False, e5_reset=True):
        # === 策略参数 ===
        self.dc = dc
        self.dcd = dcd
        self.rt = rt
        self.slope_thresh = slope_thresh
        self.ms = ms
        self.ml = ml
        self.bias_ma = bias_ma
        self.bias_high = bias_high
        self.bias_mode = bias_mode
        self.st = st
        self.sw_mid = sw_mid
        self.sw_deep = sw_deep
        self.cd = cd
        self.dual_momentum = dual_momentum
        self.bias_t_constraint = bias_t_constraint
        self.rapid_decline = rapid_decline
        self.e5_reset = e5_reset

        # === 交易成本 ===
        self.impact_slippage = impact_slippage
        self.commission = commission

        # === 回测状态 ===
        self.reset()

    def reset(self):
        """重置回测状态（每次 run 前调用）"""
        self.start_cash = 1_000_000.0
        self.cash = self.start_cash
        self.shares = 0.0          # 当前持有份额
        self.current_pos = None    # 'growth' / 'value' / 'cash'
        self.current_wt = 1.0

        # NAV 序列
        self.dates: List[str] = []
        self.nav: List[float] = []
        self.positions: List[str] = []
        self.weights: List[float] = []

        # 交易记录
        self.trades: List[TradeRecord] = []

        # 冷却状态
        self._last_confirmed_dir = None
        self._last_switch_idx = -self.dcd - 1
        self._e5_in_cooldown = False
        self._e5_cooldown_count = 0
        self._e5_locked_signal = None   # E5 冷却期内锁定的信号

        # 信号缓存（用于第1层方向确认的历史 T 值）
        self._t_history: List[float] = []

        # T+1 待执行调仓（延迟执行，防未来函数）
        self._pending_trade: Optional[Tuple[str, float]] = None

    # ----------------------------------------------------------
    # 因子计算（只用截至当前 Bar 的历史数据）
    # ----------------------------------------------------------
    def _compute_indicators(self, g_prices: np.ndarray, v_prices: np.ndarray):
        """用历史数据计算所有因子，返回当前 Bar 的标量值"""
        n = len(g_prices)
        if n < 260:
            return None

        g = pd.Series(g_prices)
        v = pd.Series(v_prices)

        ratio = g / v
        rma20 = ratio.rolling(20).mean()
        rdev = ratio / rma20 - 1
        rdev_std20 = rdev.rolling(20).std().replace(0, np.nan)
        t = (ratio - rma20) / (rma20 * rdev_std20)
        t = t.fillna(0).replace([np.inf, -np.inf], 0)
        slope = rma20.diff(5) / rma20.shift(5).replace(0, np.nan)
        slope = slope.fillna(0).replace([np.inf, -np.inf], 0)

        return {
            'T': float(t.iloc[-1]),
            'SLOPE': float(slope.iloc[-1]),
            'V_MOM_S': float(v.pct_change(self.ms).fillna(0).replace([np.inf, -np.inf], 0).iloc[-1]),
            'V_MOM_L': float(v.pct_change(self.ml).fillna(0).replace([np.inf, -np.inf], 0).iloc[-1]),
            'G_DD20': float(g.pct_change(20).fillna(0).replace([np.inf, -np.inf], 0).iloc[-1]),
            'V_DD20': float(v.pct_change(20).fillna(0).replace([np.inf, -np.inf], 0).iloc[-1]),
            'G_BIAS': float((g / g.rolling(20).mean() - 1).fillna(0).replace([np.inf, -np.inf], 0).iloc[-1]),
            'V_BIAS': float((v / v.rolling(20).mean() - 1).fillna(0).replace([np.inf, -np.inf], 0).iloc[-1]),
            'G_MOM_12M': float(g.pct_change(252).fillna(0).replace([np.inf, -np.inf], 0).iloc[-1]),
            'V_MOM_12M': float(v.pct_change(252).fillna(0).replace([np.inf, -np.inf], 0).iloc[-1]),
            'G_DD3': float(g.pct_change(3).fillna(0).replace([np.inf, -np.inf], 0).iloc[-1]),
            'V_DD3': float(v.pct_change(3).fillna(0).replace([np.inf, -np.inf], 0).iloc[-1]),
        }

    # ----------------------------------------------------------
    # X14 核心逻辑（与 backtest_x14_engine.build_core 一致）
    # ----------------------------------------------------------
    def _x14_signal(self, ind: dict, bar_idx: int) -> Tuple[str, float]:
        """输出 (signal, weight)"""
        if bar_idx < max(self.dc, self.ms, self.ml, 20, self.bias_ma) + 5:
            return 'growth', 1.0

        # ---- 第1层：方向确认 ----
        t_val = ind['T']
        self._t_history.append(t_val)
        if len(self._t_history) >= self.dc:
            recent_t = self._t_history[-self.dc:]
            if all(t > 0 for t in recent_t):
                cdir = 'BULL'
            elif all(t < 0 for t in recent_t):
                cdir = 'BEAR'
            else:
                cdir = self._last_confirmed_dir or 'BULL'
        else:
            cdir = 'BULL'

        # ---- 第2层：方向冷却 ----
        use_dir = cdir
        if self.dcd > 0 and self._last_confirmed_dir is not None and cdir != self._last_confirmed_dir:
            if bar_idx - self._last_switch_idx >= self.dcd:
                self._last_switch_idx = bar_idx
            else:
                use_dir = self._last_confirmed_dir
        self._last_confirmed_dir = use_dir

        # ---- Dual Momentum ----
        dm_trig = (ind['G_MOM_12M'] < 0) and (ind['V_MOM_12M'] < 0)
        if self.dual_momentum and dm_trig:
            use_dir = 'BEAR'

        # ---- 第3层：T+斜率 → 空仓 ----
        wt = 1.0
        is_weak = (abs(ind['T']) < self.rt) and (abs(ind['SLOPE']) < self.slope_thresh)
        if is_weak or (self.dual_momentum and dm_trig):
            wt = 0.0

        # ---- 第4层：B2 价值动量过滤 ----
        if use_dir == 'BEAR' and ind['V_MOM_S'] <= 0 and ind['V_MOM_L'] <= 0:
            use_dir = 'BULL'
        sig = 'growth' if use_dir == 'BULL' else 'value'

        # ---- 第5层：BIAS ----
        if self.bias_mode == 'clear':
            if (sig == 'growth' and ind['G_BIAS'] > self.bias_high) or \
               (sig == 'value' and ind['V_BIAS'] > self.bias_high):
                wt = 0.0
        elif self.bias_mode == 'half':
            trig = (sig == 'growth' and ind['G_BIAS'] > self.bias_high) or \
                   (sig == 'value' and ind['V_BIAS'] > self.bias_high)
            if trig:
                if not self.bias_t_constraint or abs(ind['T']) < 1.5:
                    wt *= 0.5

        # ---- 第6层：E5 止损 ----
        e5 = e5deep = False
        if sig == 'growth':
            if ind['G_DD20'] < -self.st:
                e5 = True
                if ind['G_DD20'] < -0.14:
                    e5deep = True
        else:
            if ind['V_DD20'] < -self.st:
                e5 = True
                if ind['V_DD20'] < -0.14:
                    e5deep = True
        if self.rapid_decline:
            e5 = e5 or ((sig == 'growth' and ind['G_DD3'] < -0.07) or
                        (sig == 'value' and ind['V_DD3'] < -0.07))

        if e5 and not self._e5_in_cooldown:
            # E5 首次触发：锁定当前信号，进入冷却
            self._e5_in_cooldown = True
            self._e5_cooldown_count = 0
            self._e5_locked_signal = sig
            wt *= self.sw_deep if e5deep else self.sw_mid

        elif self._e5_in_cooldown:
            # E5 冷却中：信号锁定为 E5 触发时的信号
            sig = self._e5_locked_signal
            self._e5_cooldown_count += 1
            if self._e5_cooldown_count >= self.cd:
                if e5:
                    self._e5_cooldown_count = self.cd - 3 if not self.e5_reset else 0
                    wt *= self.sw_deep if e5deep else self.sw_mid
                else:
                    self._e5_in_cooldown = False
                    self._e5_locked_signal = None
                    wt = 0.0 if is_weak else 1.0
            else:
                if wt > 0:
                    wt = self.sw_deep

        return sig, wt

    # ----------------------------------------------------------
    # 调仓执行
    # ----------------------------------------------------------
    def _execute_trade(self, new_pos: str, new_wt: float,
                       g_price: float, v_price: float, date_str: str):
        """在 T+1 日开盘执行调仓（与轻量化引擎规则一致）

        轻量化引擎规则：
          - T 日 signal[i] != signal[i-1] → 触发调仓
          - 调仓在 T+1 日开盘价执行
          - 本实现中：当前 Bar 的 close = T 日收盘
            → 信号变化在 T 日收盘后决定 → T+1 日开盘执行
        """
        old_pos = self.current_pos
        old_wt = self.current_wt

        nav_before = self.cash + self.shares * (
            g_price if old_pos == 'growth' else v_price if old_pos == 'value' else 0
        )

        # === 卖出 ===
        if old_pos and old_pos != 'cash' and self.shares > 0:
            sell_price = (g_price if old_pos == 'growth' else v_price) * (1 - self.impact_slippage)
            sell_value = self.shares * sell_price
            commission_sell = sell_value * self.commission
            self.cash = self.cash + sell_value - commission_sell  # 累加！保留原有现金
            self.shares = 0.0

        # === 买入 ===
        if new_pos != 'cash' and new_wt > 0:
            buy_price = (g_price if new_pos == 'growth' else v_price) * (1 + self.impact_slippage)
            invest_amount = self.cash * new_wt
            commission_buy = invest_amount * self.commission
            self.shares = (invest_amount - commission_buy) / buy_price
            self.cash -= invest_amount
        else:
            # 空仓
            pass

        self.current_pos = new_pos
        self.current_wt = new_wt

    def _get_current_nav(self, g_price, v_price):
        if self.current_pos == 'growth':
            return self.shares * g_price + self.cash
        elif self.current_pos == 'value':
            return self.shares * v_price + self.cash
        else:
            return self.cash

    # ----------------------------------------------------------
    # 主循环 — 事件驱动
    # ----------------------------------------------------------
    def run(self) -> Dict:
        """逐 Bar 运行回测"""
        self.reset()

        print("  运行事件驱动回测...")

        for i in range(N):
            date_str = common_idx[i].strftime('%Y-%m-%d')
            g_price = G_ARR[i]
            v_price = V_ARR[i]

            # === 第1个 Bar：初始建仓 ===
            if i == 0:
                # 默认持有成长
                buy_price = g_price * (1 + self.impact_slippage)
                invest = self.cash
                commission_cost = invest * self.commission
                self.shares = (invest - commission_cost) / buy_price
                self.cash = 0.0
                self.current_pos = 'growth'
                self.current_wt = 1.0

                self.dates.append(date_str)
                nav_val = self.shares * g_price
                self.nav.append(nav_val)
                self.positions.append('growth')
                self.weights.append(1.0)
                continue

            # === 正常 Bar：事件驱动处理 ===
            # 第1步：检查是否有 T-1 日的待执行调仓 → 先执行
            if self._pending_trade is not None:
                pending_sig, pending_wt = self._pending_trade
                self._pending_trade = None
                old_pos, old_wt = self.current_pos, self.current_wt
                nav_before = self._get_current_nav(g_price, v_price)
                self._execute_trade(pending_sig, pending_wt, g_price, v_price, date_str)
                self.trades.append(TradeRecord(
                    date=date_str,
                    from_pos=old_pos,
                    from_wt=old_wt,
                    to_pos=pending_sig,
                    to_wt=pending_wt,
                    g_price=g_price,
                    v_price=v_price,
                    nav_before=nav_before,
                ))

            # 第2步：用截至当前 Bar 的历史数据计算因子
            g_hist = G_ARR[:i + 1]
            v_hist = V_ARR[:i + 1]
            ind = self._compute_indicators(g_hist, v_hist)

            if ind is None:
                # 数据不足，保持现有持仓
                nav_val = self._get_current_nav(g_price, v_price)
                self.dates.append(date_str)
                self.nav.append(nav_val)
                self.positions.append(self.current_pos or 'growth')
                self.weights.append(self.current_wt)
                continue

            # 第3步：用因子计算信号 —— 决定 T 日收盘后的方向
            sig, wt = self._x14_signal(ind, i)

            # 第4步：检查信号/权重变化 → 记录待执行，T+1 日开盘调仓
            sig_changed = (self.current_pos != sig)
            wt_changed = abs((self.current_wt or 0) - wt) > 1e-10

            if sig_changed or wt_changed:
                # 不立即执行，而是记录待执行（T+1 规则，防未来函数）
                self._pending_trade = (sig, wt)

            # 第5步：记录当日 NAV（基于调仓前/保持的持仓 × 当日收盘价）
            nav_val = self._get_current_nav(g_price, v_price)
            if nav_val <= 0 and i > 0:
                print(f'  [DEBUG] NAV=0 at bar {i} ({date_str}): '
                      f'pos={self.current_pos} wt={self.current_wt:.4f} '
                      f'cash={self.cash:.2f} shares={self.shares:.6f} '
                      f'g={g_price:.2f} v={v_price:.2f}')
                # 紧急修复：用现金恢复
                self.cash = 1.0
            self.dates.append(date_str)
            self.nav.append(nav_val)
            self.positions.append(self.current_pos or 'growth')
            self.weights.append(self.current_wt)

        # 计算指标
        metrics = self._calc_metrics()
        return {'nav': np.array(self.nav), 'dates': self.dates,
                'trades': self.trades, 'metrics': metrics,
                'positions': self.positions, 'weights': self.weights}

    def _calc_metrics(self) -> Dict:
        nav_arr = np.array(self.nav)
        n = len(nav_arr)
        if n < 2:
            return {'ann': 0, 'dd': 0, 'sharpe': 0, 'calmar': 0, 'n_trades': 0}

        daily_ret = nav_arr[1:] / nav_arr[:-1] - 1
        equity = nav_arr / nav_arr[0]
        peak = np.maximum.accumulate(equity)
        dd = (equity - peak) / peak

        total_ret = equity[-1] - 1
        years = n / 252
        ann = (1 + total_ret) ** (1 / years) - 1 if years > 0 else 0
        max_dd = dd.min()
        rf = 0.025 / 252
        sharpe = np.sqrt(252) * (daily_ret.mean() - rf) / daily_ret.std() if daily_ret.std() > 0 else 0
        calmar = ann / abs(max_dd) if max_dd < 0 else 0

        return {
            'ann': ann, 'dd': max_dd, 'sharpe': sharpe, 'calmar': calmar,
            'n_trades': len(self.trades), 'total_return': total_ret,
        }


# ============================================================
# 主入口
# ============================================================
def run():
    """运行事件驱动回测并对比轻量化引擎（v1.0 基线: 45.27% / Calmar 1.951）"""
    bias_mode = 'clear'
    impact_slippage = 0.0005

    # ---- 事件驱动引擎 ----
    print("=" * 80)
    print("  X14 策略 — 事件驱动引擎回测")
    print("=" * 80)
    print(f'  BIAS模式: {bias_mode}')
    print(f'  冲击滑点: {impact_slippage*100:.2f}%')
    print(f'  手续费: 0.01%')
    print(f'  数据范围: {common_idx[0].date()} ~ {common_idx[-1].date()} ({N} 天)')

    bt_ed = EventDrivenBacktest(bias_mode=bias_mode, impact_slippage=impact_slippage,
                                 dcd=6, bias_t_constraint=False, e5_reset=True)
    result_ed = bt_ed.run()
    m_ed = result_ed['metrics']

    print(f'\n  事件驱动引擎结果:')
    print(f'    年化={m_ed["ann"]*100:.2f}%')
    print(f'    最大回撤={m_ed["dd"]*100:.2f}%')
    print(f'    Sharpe={m_ed["sharpe"]:.3f}')
    print(f'    Calmar={m_ed["calmar"]:.3f}')
    print(f'    交易次数={m_ed["n_trades"]}')

    # ---- 轻量化引擎对比 ----
    sys.path.insert(0, r'c:\XuanJLH\Qbot\custom\backtests')
    sys.path.insert(0, r'c:\XuanJLH\Qbot\custom\research\成长价值轮动\X12_偏离度空仓版')
    sys.path.insert(0, r'c:\XuanJLH\Qbot\custom\research\成长价值轮动\X14_统一策略干净版')

    from optimize_runner import run_backtest, calc_metrics, count_switches
    from backtest_x14_engine import build_core

    sig_lw, wt_lw = build_core(bias_mode=bias_mode, dcd=6, bias_t_constraint=False, e5_reset=True)
    res_lw = run_backtest(sig_lw, wt_lw, impact_slippage=impact_slippage)
    m_lw = calc_metrics(res_lw)
    sw_lw = count_switches(sig_lw, wt_lw)

    print(f'\n  轻量化引擎结果:')
    print(f'    年化={m_lw["ann"]*100:.2f}%')
    print(f'    最大回撤={m_lw["dd"]*100:.2f}%')
    print(f'    Sharpe={m_lw["sharpe"]:.3f}')
    print(f'    Calmar={m_lw["calmar"]:.3f}')
    print(f'    交易次数={m_lw["n_trades"]}')

    # ---- 差异分析 ----
    print(f'\n  --- 差异分析 ---')
    print(f'  年化差异: {(m_ed["ann"] - m_lw["ann"])*100:+.2f}%')
    print(f'  回撤差异: {(m_ed["dd"] - m_lw["dd"])*100:+.2f}%')
    print(f'  Sharpe差异: {m_ed["sharpe"] - m_lw["sharpe"]:+.3f}')
    print(f'  Calmar差异: {m_ed["calmar"] - m_lw["calmar"]:+.3f}')
    print(f'  交易次数差异: {m_ed["n_trades"] - m_lw["n_trades"]:+d}')

    # ---- 保存结果 ----
    out_dir = Path(r'c:\XuanJLH\Qbot\custom\research\成长价值轮动\X14_统一策略干净版\回测结果')
    out_dir.mkdir(parents=True, exist_ok=True)

    # 净值
    nav_df = pd.DataFrame({
        'date': result_ed['dates'],
        'nav': result_ed['nav'],
        'position': result_ed['positions'],
        'weight': result_ed['weights'],
    })
    nav_df.to_csv(out_dir / 'X14_事件驱动_净值.csv', index=False)

    # 交易记录
    if result_ed['trades']:
        pd.DataFrame([{
            'date': t.date, 'from': t.from_pos, 'from_wt': t.from_wt,
            'to': t.to_pos, 'to_wt': t.to_wt,
            'g_price': t.g_price, 'v_price': t.v_price,
        } for t in result_ed['trades']]).to_csv(
            out_dir / 'X14_事件驱动_交易记录.csv', index=False)

    print(f'\n  {"="*80}')
    print(f'  结果已保存到: {out_dir}')
    print(f'  {"="*80}')

    return result_ed, m_lw


if __name__ == '__main__':
    run()
