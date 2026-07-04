"""验证器1：公平基准族

A/B共识核心问题：策略跑输等权基准 → 没有真正alpha
但等权10只是不公平对比（策略持仓2只，承担更高idiosyncratic风险）

本模块建立4种公平基准：
1. 等权全池（EQUAL_WEIGHT）：买入持有10只ETF等权
2. 随机Top2（RANDOM_TOP2）：每次随机选2只持有，蒙特卡洛N次得分布
3. 动量Top2买入持有（MOMENTUM_BUYHOLD）：用动量选2只但不再轮动
   - 分离"选股能力"和"轮动动作"的alpha贡献
4. 蒙特卡洛分位：策略Sharpe在随机Top2分布中的百分位

判定规则（A/B共识）：
- 策略Sharpe必须 > 随机Top2分布的95%分位，才算"非偶然alpha"
- 策略Sharpe必须 > 动量Top2买入持有，才算"轮动动作有效"
- 否则策略只是"选股能力"或"运气"
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple


def calc_metrics(equity: pd.Series, freq=252, risk_free=0.02) -> Dict[str, float]:
    """计算单条equity曲线的核心指标"""
    rets = equity.pct_change().fillna(0)
    n_days = len(rets)
    if n_days < 2:
        return {"annual_return": 0, "sharpe": 0, "max_drawdown": 0, "calmar": 0, "vol": 0}

    ann_ret = (1 + rets.mean()) ** freq - 1
    vol = rets.std() * np.sqrt(freq)
    sharpe = (ann_ret - risk_free) / vol if vol > 0 else 0
    running_max = equity.cummax()
    dd = (equity - running_max) / running_max
    max_dd = float(dd.min())
    calmar = ann_ret / abs(max_dd) if max_dd < 0 else 0
    return {
        "annual_return": float(ann_ret),
        "sharpe": float(sharpe),
        "max_drawdown": max_dd,
        "calmar": float(calmar),
        "vol": float(vol),
    }


def equal_weight_benchmark(close: pd.DataFrame, cost=0.0015) -> Tuple[pd.Series, Dict]:
    """基准1：等权全池买入持有

    一次性买入10只ETF等权，永不调仓
    """
    n_assets = close.shape[1]
    weights = np.ones(n_assets) / n_assets
    # 一次性买入成本
    daily_ret = close.pct_change().fillna(0)
    portfolio_ret = (daily_ret * weights).sum(axis=1)
    # 入场成本摊到第一天
    portfolio_ret.iloc[0] -= cost
    equity = (1 + portfolio_ret).cumprod()
    return equity, calc_metrics(equity)


def random_top2_benchmark(close: pd.DataFrame, rebalance_freq="M",
                          n_runs=1000, cost=0.0015, seed=42) -> Tuple[None, Dict]:
    """基准2：随机Top2 + 蒙特卡洛分布

    每个调仓日随机选2只ETF等权持有，重复n_runs次
    返回Sharpe分布的统计量
    """
    rng = np.random.default_rng(seed)
    daily_ret = close.pct_change().fillna(0)
    rebalance_idx = _get_rebalance_idx(close, rebalance_freq)

    sharpes = []
    ann_rets = []
    max_dds = []
    for run in range(n_runs):
        # 每次run生成不同的随机选择序列
        holdings = []  # (date, picked_list)
        for rebalance_date in rebalance_idx:
            picked = rng.choice(close.columns, size=2, replace=False)
            holdings.append((rebalance_date, picked))

        # 用"逐日前向填充"模式构建权重（与策略 generate_signal_v2 一致）
        # 修复bug：之前用 weights.loc[mask, col] = 0.5 会叠加旧持仓权重
        weights = pd.DataFrame(0.0, index=close.index, columns=close.columns)
        current_holdings = None
        holdings_dict = {date: picked for date, picked in holdings}
        for date in close.index:
            if date in holdings_dict:
                current_holdings = holdings_dict[date]
            if current_holdings is not None:
                for col in current_holdings:
                    weights.loc[date, col] = 0.5

        # shift避免未来函数
        weights = weights.shift(1).fillna(0)
        portfolio_ret = (daily_ret * weights).sum(axis=1)

        # 估算换手成本
        weight_change = weights.diff().abs().sum(axis=1).fillna(0)
        cost_series = weight_change * cost
        portfolio_ret = portfolio_ret - cost_series

        equity = (1 + portfolio_ret).cumprod()
        m = calc_metrics(equity)
        sharpes.append(m["sharpe"])
        ann_rets.append(m["annual_return"])
        max_dds.append(m["max_drawdown"])

    distribution = {
        "sharpe_mean": float(np.mean(sharpes)),
        "sharpe_std": float(np.std(sharpes)),
        "sharpe_p5": float(np.percentile(sharpes, 5)),
        "sharpe_p50": float(np.percentile(sharpes, 50)),
        "sharpe_p95": float(np.percentile(sharpes, 95)),
        "ann_ret_mean": float(np.mean(ann_rets)),
        "ann_ret_p5": float(np.percentile(ann_rets, 5)),
        "ann_ret_p95": float(np.percentile(ann_rets, 95)),
        "max_dd_mean": float(np.mean(max_dds)),
        "max_dd_p5": float(np.percentile(max_dds, 5)),
        "n_runs": n_runs,
        "sharpes_raw": sharpes,  # 保留原始数据用于画图
    }
    return None, distribution


def momentum_buyhold_benchmark(close: pd.DataFrame, momentum_window=25,
                                hold_count=2, cost=0.0015) -> Tuple[pd.Series, Dict]:
    """基准3：动量Top2买入持有（分离选股能力 vs 轮动能力）

    在第1个调仓日用动量选Top2，之后永不调仓
    如果策略跑输此基准，说明"轮动动作"无效，反而毁损了"选股能力"
    """
    daily_ret = close.pct_change().fillna(0)
    rebalance_idx = _get_rebalance_idx(close, "M")
    if len(rebalance_idx) == 0:
        return None, calc_metrics(pd.Series([1.0], index=close.index))

    # 只用第一个调仓日选股
    first_rebalance = rebalance_idx[0]
    momentum = close.pct_change(periods=momentum_window).iloc[
        close.index.get_loc(first_rebalance)
    ]
    top_picks = momentum.nlargest(hold_count).index.tolist()

    weights = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    mask = weights.index >= first_rebalance
    for col in top_picks:
        weights.loc[mask, col] = 1.0 / hold_count

    weights = weights.shift(1).fillna(0)
    portfolio_ret = (daily_ret * weights).sum(axis=1)
    portfolio_ret.iloc[0] -= cost  # 一次性入场成本

    equity = (1 + portfolio_ret).cumprod()
    return equity, calc_metrics(equity)


def monte_carlo_percentile(strategy_sharpe: float, distribution: Dict) -> Dict:
    """基准4：蒙特卡洛分位

    策略Sharpe在随机Top2分布中的百分位
    判定：< 95% → alpha不显著；>= 95% → 非偶然alpha
    """
    sharpes = np.array(distribution["sharpes_raw"])
    percentile = float((sharpes < strategy_sharpe).mean() * 100)
    z_score = float((strategy_sharpe - distribution["sharpe_mean"]) /
                    (distribution["sharpe_std"] + 1e-9))
    return {
        "strategy_sharpe": strategy_sharpe,
        "random_sharpe_mean": distribution["sharpe_mean"],
        "random_sharpe_p95": distribution["sharpe_p95"],
        "percentile": percentile,
        "z_score": z_score,
        "alpha_significant": percentile >= 95,
    }


def _get_rebalance_idx(close: pd.DataFrame, freq="M"):
    """获取调仓日索引"""
    if freq == "M":
        return close.index.to_series().groupby(
            [close.index.year, close.index.month]
        ).last().tolist()
    elif freq == "W":
        return close.index.to_series().groupby(
            [close.index.year, close.index.isocalendar().week]
        ).last().tolist()
    else:
        return close.index.tolist()


def run_all_benchmarks(close: pd.DataFrame, strategy_metrics: Dict,
                       n_mc_runs=500) -> Dict:
    """运行全部4种基准对比

    Args:
        close: 收盘价DataFrame
        strategy_metrics: 策略自身的metrics dict（含sharpe字段）
        n_mc_runs: 蒙特卡洛次数（默认500以加速）

    Returns:
        完整基准对比报告
    """
    print("  [基准1/3] 等权全池...")
    _, eq_metrics = equal_weight_benchmark(close)

    print(f"  [基准2/3] 随机Top2 蒙特卡洛{n_mc_runs}次...")
    _, random_dist = random_top2_benchmark(close, n_runs=n_mc_runs)

    print("  [基准3/3] 动量Top2买入持有...")
    _, mom_bh_metrics = momentum_buyhold_benchmark(close)

    mc_result = monte_carlo_percentile(strategy_metrics["sharpe"], random_dist)

    return {
        "strategy": strategy_metrics,
        "equal_weight": eq_metrics,
        "random_top2_distribution": random_dist,
        "momentum_buyhold": mom_bh_metrics,
        "monte_carlo": mc_result,
        "verdict": _make_verdict(strategy_metrics, eq_metrics, mom_bh_metrics, mc_result),
    }


def _make_verdict(strategy: Dict, eq: Dict, mom_bh: Dict, mc: Dict) -> Dict:
    """生成最终判定"""
    verdicts = []

    # 检查1：打败等权全池？
    if strategy["sharpe"] > eq["sharpe"]:
        verdicts.append(f"✓ 策略Sharpe({strategy['sharpe']:.3f}) > 等权全池({eq['sharpe']:.3f})")
    else:
        verdicts.append(f"✗ 策略Sharpe({strategy['sharpe']:.3f}) ≤ 等权全池({eq['sharpe']:.3f}) — 未通过基础有效性")

    # 检查2：打败动量Top2买入持有？
    if strategy["sharpe"] > mom_bh["sharpe"]:
        verdicts.append(f"✓ 策略Sharpe > 动量Top2买入持有({mom_bh['sharpe']:.3f}) — 轮动动作有效")
    else:
        verdicts.append(f"✗ 策略Sharpe ≤ 动量Top2买入持有({mom_bh['sharpe']:.3f}) — 轮动动作无效，仅靠选股能力")

    # 检查3：蒙特卡洛95%分位？
    if mc["alpha_significant"]:
        verdicts.append(f"✓ 策略Sharpe位于随机分布{mc['percentile']:.1f}分位 (>=95%) — 非偶然alpha")
    else:
        verdicts.append(f"✗ 策略Sharpe位于随机分布{mc['percentile']:.1f}分位 (<95%) — alpha不显著")

    overall_pass = (strategy["sharpe"] > eq["sharpe"] and
                    strategy["sharpe"] > mom_bh["sharpe"] and
                    mc["alpha_significant"])

    return {
        "checks": verdicts,
        "overall_pass": overall_pass,
        "verdict_summary": "策略通过公平基准检验" if overall_pass else "策略未通过公平基准检验",
    }
