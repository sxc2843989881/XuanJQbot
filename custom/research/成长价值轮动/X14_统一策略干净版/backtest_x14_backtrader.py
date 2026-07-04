"""backtest_x14_backtrader.py — X14 策略在 Backtrader 事件驱动引擎下的回测

================================================================
使用 Qbot 集成的 backtrader 引擎（qbot/engine/backtest/）
双 data feed：成长指数 + 价值指数
================================================================
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import backtrader as bt

sys.path.insert(0, r'c:\XuanJLH\Qbot\custom\backtests')
sys.path.insert(0, r'c:\XuanJLH\Qbot\custom\research\成长价值轮动\X12_偏离度空仓版')

# ============================================================
# 数据加载
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
# X14 策略 — Backtrader 事件驱动实现
# ============================================================
class X14BacktraderStrategy(bt.Strategy):
    """X14 v1.0 基线 — Backtrader 事件驱动实现

    双 data feed: datas[0]=成长, datas[1]=价值
    """
    params = (
        ('dc', 5), ('dcd', 6),
        ('rt', 1.3), ('slope_thresh', 0.002),
        ('ms', 10), ('ml', 20),
        ('bias_ma', 20), ('bias_high', 0.19), ('bias_mode', 'clear'),
        ('st', 0.09), ('sw_mid', 0.17), ('sw_deep', 0.17), ('cd', 8),
        ('dual_momentum', False), ('bias_t_constraint', False),
        ('rapid_decline', False), ('e5_reset', True),
    )

    def __init__(self):
        self.g_close = self.datas[0].close
        self.v_close = self.datas[1].close

        self.current_pos = None
        self.current_wt = 1.0
        self.trade_log = []
        self.daily_nav = []
        self._bar_count = 0

        self._last_confirmed_dir = None
        self._last_switch_bar = -self.p.dcd - 1
        self._e5_in_cooldown = False
        self._e5_cooldown_count = 0

    def _get_g_array(self):
        return np.array(self.g_close.array[:self._bar_count + 1], dtype=float)

    def _get_v_array(self):
        return np.array(self.v_close.array[:self._bar_count + 1], dtype=float)

    def _compute_indicators(self):
        if self._bar_count < 259:
            return None
        try:
            g_arr = self._get_g_array()
            v_arr = self._get_v_array()
            if len(g_arr) < 260:
                return None
        except:
            return None

        g = pd.Series(g_arr)
        v = pd.Series(v_arr)

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
            'V_MOM_S': float(v.pct_change(self.p.ms).fillna(0).iloc[-1]),
            'V_MOM_L': float(v.pct_change(self.p.ml).fillna(0).iloc[-1]),
            'G_DD20': float(g.pct_change(20).fillna(0).iloc[-1]),
            'V_DD20': float(v.pct_change(20).fillna(0).iloc[-1]),
            'G_BIAS': float((g / g.rolling(20).mean() - 1).fillna(0).iloc[-1]),
            'V_BIAS': float((v / v.rolling(20).mean() - 1).fillna(0).iloc[-1]),
            'G_MOM_12M': float(g.pct_change(252).fillna(0).iloc[-1]),
            'V_MOM_12M': float(v.pct_change(252).fillna(0).iloc[-1]),
            'G_DD3': float(g.pct_change(3).fillna(0).iloc[-1]),
            'V_DD3': float(v.pct_change(3).fillna(0).iloc[-1]),
            'T_arr': t.values,
        }

    def _x14_signal(self, ind, bar_idx):
        i = bar_idx
        if i < max(self.p.dc, self.p.ms, self.p.ml, 20) + 5:
            return 'growth', 1.0

        # 第1层：方向确认
        if i >= self.p.dc:
            t_slice = ind['T_arr'][i - self.p.dc + 1:i + 1]
            all_bull = np.all(t_slice > 0)
            all_bear = np.all(t_slice < 0)
            cdir = 'BULL' if all_bull else ('BEAR' if all_bear else (self._last_confirmed_dir or 'BULL'))
        else:
            cdir = 'BULL'

        # 第2层：方向冷却
        use_dir = cdir
        if self.p.dcd > 0 and self._last_confirmed_dir is not None and cdir != self._last_confirmed_dir:
            if i - self._last_switch_bar >= self.p.dcd:
                self._last_switch_bar = i
            else:
                use_dir = self._last_confirmed_dir
        self._last_confirmed_dir = use_dir

        dm_trig = (ind['G_MOM_12M'] < 0) and (ind['V_MOM_12M'] < 0)
        if self.p.dual_momentum and dm_trig:
            use_dir = 'BEAR'

        wt = 1.0
        is_weak = (abs(ind['T']) < self.p.rt) and (abs(ind['SLOPE']) < self.p.slope_thresh)
        if is_weak or (self.p.dual_momentum and dm_trig):
            wt = 0.0

        if use_dir == 'BEAR' and ind['V_MOM_S'] <= 0 and ind['V_MOM_L'] <= 0:
            use_dir = 'BULL'
        sig = 'growth' if use_dir == 'BULL' else 'value'

        if self.p.bias_mode == 'clear':
            if (sig == 'growth' and ind['G_BIAS'] > self.p.bias_high) or \
               (sig == 'value' and ind['V_BIAS'] > self.p.bias_high):
                wt = 0.0
        elif self.p.bias_mode == 'half':
            trig = (sig == 'growth' and ind['G_BIAS'] > self.p.bias_high) or \
                   (sig == 'value' and ind['V_BIAS'] > self.p.bias_high)
            if trig:
                if not self.p.bias_t_constraint or abs(ind['T']) < 1.5:
                    wt *= 0.5

        e5 = e5deep = False
        if sig == 'growth':
            if ind['G_DD20'] < -self.p.st:
                e5 = True
                if ind['G_DD20'] < -0.14:
                    e5deep = True
        else:
            if ind['V_DD20'] < -self.p.st:
                e5 = True
                if ind['V_DD20'] < -0.14:
                    e5deep = True
        if self.p.rapid_decline:
            e5 = e5 or ((sig == 'growth' and ind['G_DD3'] < -0.07) or
                        (sig == 'value' and ind['V_DD3'] < -0.07))

        if e5 and not self._e5_in_cooldown:
            self._e5_in_cooldown = True
            self._e5_cooldown_count = 0
            wt *= self.p.sw_deep if e5deep else self.p.sw_mid
        elif self._e5_in_cooldown:
            self._e5_cooldown_count += 1
            if self._e5_cooldown_count >= self.p.cd:
                if e5:
                    self._e5_cooldown_count = self.p.cd - 3 if not self.p.e5_reset else 0
                    wt *= self.p.sw_deep if e5deep else self.p.sw_mid
                else:
                    self._e5_in_cooldown = False
                    wt = 0.0 if is_weak else 1.0
            else:
                if wt > 0:
                    wt = self.p.sw_deep
        return sig, wt

    def next(self):
        self._bar_count += 1
        bar_len = self._bar_count
        current_date = self.datas[0].datetime.date(0).isoformat()

        if self._bar_count < 260:
            if self.current_pos is None:
                self.current_pos = 'growth'
                self.current_wt = 1.0
                self.order_target_percent(data=self.datas[0], target=1.0)
            self.daily_nav.append((current_date, self.broker.getvalue()))
            return

        ind = self._compute_indicators()
        if ind is None:
            return

        sig, wt = self._x14_signal(ind, bar_len - 1)

        g_price = float(self.g_close[0])
        v_price = float(self.v_close[0])

        sig_changed = (self.current_pos != sig)
        wt_changed = abs((self.current_wt or 0) - wt) > 1e-10

        if sig_changed or wt_changed:
            old_pos = self.current_pos
            old_wt = self.current_wt
            pv = self.broker.getvalue()

            self.trade_log.append(dict(
                date=current_date, from_pos=old_pos, from_wt=old_wt,
                to_pos=sig, to_wt=wt, g_price=g_price, v_price=v_price, nav=pv,
            ))

            # 通过 backtrader broker 执行：先清旧仓，再建新仓
            if sig == 'growth':
                self.close(data=self.datas[1])  # 清空价值
                self.order_target_percent(data=self.datas[0], target=wt)
            else:
                self.close(data=self.datas[0])  # 清空成长
                self.order_target_percent(data=self.datas[1], target=wt)

            self.current_pos = sig
            self.current_wt = wt

        self.daily_nav.append((current_date, self.broker.getvalue()))

    def stop(self):
        final_nav = self.broker.getvalue()
        total_ret = final_nav / self.broker.startingcash - 1
        days = len(self.daily_nav)
        years = days / 252
        ann = (1 + total_ret) ** (1 / years) - 1 if years > 0 else 0
        max_dd = self._calc_max_dd()
        sharpe = self._calc_sharpe()
        calmar = ann / abs(max_dd) if max_dd < 0 else 0

        print(f'\n  [Backtrader 事件驱动引擎] X14 回测完成')
        print(f'  初始资金: {self.broker.startingcash:.2f}')
        print(f'  最终净值: {final_nav:.2f}')
        print(f'  总收益: {total_ret*100:.2f}%')
        print(f'  年化: {ann*100:.2f}%')
        print(f'  最大回撤: {max_dd*100:.2f}%')
        print(f'  Sharpe: {sharpe:.3f}')
        print(f'  Calmar: {calmar:.3f}')
        print(f'  交易次数: {len(self.trade_log)}')

        self._compare_lightweight()

    def _calc_max_dd(self):
        if len(self.daily_nav) < 2:
            return 0.0
        nav_arr = np.array([v for _, v in self.daily_nav])
        eq = nav_arr / nav_arr[0]
        peak = np.maximum.accumulate(eq)
        return ((eq - peak) / peak).min()

    def _calc_sharpe(self):
        if len(self.daily_nav) < 10:
            return 0.0
        nav_arr = np.array([v for _, v in self.daily_nav])
        daily_ret = nav_arr[1:] / nav_arr[:-1] - 1
        rf = 0.025 / 252
        return np.sqrt(252) * (daily_ret.mean() - rf) / daily_ret.std() if daily_ret.std() > 0 else 0

    def _compare_lightweight(self):
        from optimize_runner import run_backtest, calc_metrics, count_switches
        from backtest_x14_engine import build_core

        sig_lw, wt_lw = build_core(bias_mode=self.p.bias_mode, dcd=self.p.dcd,
                                    bias_t_constraint=self.p.bias_t_constraint,
                                    e5_reset=self.p.e5_reset)
        res_lw = run_backtest(sig_lw, wt_lw, impact_slippage=0.0005)
        m_lw = calc_metrics(res_lw)

        final_nav = self.broker.getvalue()
        total_ret = final_nav / self.broker.startingcash - 1
        days = len(self.daily_nav)
        years = days / 252
        ann = (1 + total_ret) ** (1 / years) - 1 if years > 0 else 0
        max_dd = self._calc_max_dd()
        sharpe = self._calc_sharpe()
        calmar = ann / abs(max_dd) if max_dd < 0 else 0

        print(f'\n  --- 轻量化引擎对比 ---')
        print(f'  轻量化: 年化={m_lw["ann"]*100:.2f}%  Calmar={m_lw["calmar"]:.3f}  交易={m_lw["n_trades"]}')
        print(f'  事件驱动: 年化={ann*100:.2f}%  Calmar={calmar:.3f}  交易={len(self.trade_log)}')
        print(f'  年化差异: {(ann - m_lw["ann"])*100:+.2f}%')
        print(f'  Calmar差异: {calmar - m_lw["calmar"]:+.3f}')


# ============================================================
# 运行
# ============================================================
def run():
    bias_mode = 'clear'
    impact_slippage = 0.0005
    commission = 0.0001

    # 两个 data feed
    df_g = pd.DataFrame({'close': G_ARR}, index=pd.to_datetime(common_idx))
    df_v = pd.DataFrame({'close': V_ARR}, index=pd.to_datetime(common_idx))

    cerebro = bt.Cerebro()
    cerebro.adddata(bt.feeds.PandasData(dataname=df_g), name='growth')
    cerebro.adddata(bt.feeds.PandasData(dataname=df_v), name='value')
    cerebro.addstrategy(X14BacktraderStrategy,
                        bias_mode=bias_mode)
    cerebro.broker.setcash(1_000_000.0)
    cerebro.broker.setcommission(commission=commission)
    cerebro.broker.set_slippage_perc(perc=impact_slippage)

    print("=" * 80)
    print("  X14 策略 — Backtrader 事件驱动引擎回测")
    print("=" * 80)
    print(f'  data[0]=成长, data[1]=价值')
    print(f'  冲击滑点: {impact_slippage*100:.2f}%, 手续费: {commission*100:.2f}%')
    print(f'  数据范围: {common_idx[0].date()} ~ {common_idx[-1].date()} ({N} 天)')

    result = cerebro.run(runonce=False)

    strat = result[0]
    out_dir = Path(r'c:\XuanJLH\Qbot\custom\research\成长价值轮动\X14_统一策略干净版\回测结果')
    out_dir.mkdir(parents=True, exist_ok=True)
    if strat.daily_nav:
        pd.DataFrame(strat.daily_nav, columns=['date', 'nav']).to_csv(
            out_dir / 'X14_backtrader_净值.csv', index=False)
    if strat.trade_log:
        pd.DataFrame(strat.trade_log).to_csv(
            out_dir / 'X14_backtrader_交易记录.csv', index=False)


if __name__ == '__main__':
    run()
