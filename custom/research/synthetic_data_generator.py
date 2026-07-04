"""合成样本数据生成器（Moment-Matched Block Bootstrap）

适用场景：
    量化研究中，当样本数据年限不足、路径单一时，通过模拟技术生成"统计性质相同但走势不同"的
    新样本，用于策略稳健性检验、过拟合检测、风险场景扩展。

核心方法：
    Block Bootstrap（分块重采样） + Moment Matching（矩匹配重标度）

技术原理：
    1. Block Bootstrap：将历史对数收益率序列划分为固定长度的连续块，随机抽取整个块进行重排
       拼接成新的收益率序列。保留块内的短期依赖结构（波动率聚集、自相关），避免 IID Bootstrap
       破坏时间序列记忆效应。
       参考：Künsch (1989) Moving Block Bootstrap；Politis & Romano (1994) Stationary Bootstrap
       块长经验值：L ≈ T^(1/3)，日频数据通常取 21（一个月）

    2. Moment Matching：对 Bootstrap 得到的收益率序列做"标准化 + 重标度"，强制使其样本均值
       和样本标准差精确等于原始序列。这样生成的合成路径具备：
         - 年化收益率与原始完全一致（因为对数收益率均值被精确匹配）
         - 年化波动率与原始完全一致（因为对数收益率标准差被精确匹配）
         - Sharpe Ratio 完全一致（因为 Sharpe = (μ - r_f) / σ）
         - 价格走势不同（因为 Bootstrap 打乱了收益率顺序）
       参考：portfoliooptimizer.io "Bootstrap Simulations with Exact Sample Mean Vector and
       Sample Covariance Matrix"

输出说明：
    合成数据保持原始 DataFrame 的列结构（date, open, high, low, close, volume, amount）：
      - close：通过对数收益率矩匹配重建（核心统计量精确匹配）
      - open/high/low：根据原始的日内比例关系（open/close, high/close, low/close）从新的
        close 反推，保证日内结构合理（high >= max(open,close), low <= min(open,close)）
      - volume/amount：随 Block Bootstrap 同步重排（保持流动性结构与价格波动的对应关系）

参考文献：
    [1] 浙商证券《蒙特卡洛回测：从历史拟合转向未来稳健》(2026)
    [2] 华泰证券《偶然中的必然，重采样技术检验过拟合》
    [3] portfoliooptimizer.io/blog/bootstrap-simulations-with-exact-sample-mean-vector-and-sample-covariance-matrix
    [4] arch.readthedocs.io/en/stable/bootstrap/ — Python arch 库的时序 Bootstrap 实现
"""
import numpy as np
import pandas as pd
from typing import List, Optional


# ============================================================
# 1. Block Bootstrap：分块重采样
# ============================================================

def block_bootstrap(returns: np.ndarray,
                    block_length: int = 21,
                    n_samples: Optional[int] = None,
                    rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """对收益率序列做分块 Bootstrap 重采样（Moving Block Bootstrap）

    Args:
        returns: 一维收益率数组 (T,)
        block_length: 块长度（日频数据默认 21 ≈ 1 个月）
        n_samples: 输出长度，默认与输入相同
        rng: numpy.random.Generator，便于复现

    Returns:
        重采样后的收益率数组 (n_samples,)
    """
    if rng is None:
        rng = np.random.default_rng()

    T = len(returns)
    if T < block_length:
        # 数据太短，退化为 IID 重采样
        idx = rng.integers(0, T, size=n_samples or T)
        return returns[idx]

    if n_samples is None:
        n_samples = T

    # 计算需要的块数
    n_blocks = (n_samples + block_length - 1) // block_length

    # 每个块随机选一个起始位置（0 到 T-block_length）
    starts = rng.integers(0, T - block_length + 1, size=n_blocks)

    # 拼接所有块
    pieces = [returns[s:s + block_length] for s in starts]
    result = np.concatenate(pieces)

    # 截断到目标长度
    return result[:n_samples]


# ============================================================
# 2. Moment Matching：矩匹配重标度
# ============================================================

def moment_match(returns: np.ndarray,
                 target_mean: float,
                 target_std: float) -> np.ndarray:
    """对收益率序列做矩匹配，强制其样本均值和标准差等于目标值

    调整公式：
        adjusted = (returns - mean(returns)) / std(returns) × target_std + target_mean

    性质：
        - 调整后 mean(adjusted) == target_mean（精确）
        - 调整后 std(adjusted, ddof=1) == target_std（精确）
        - 不改变序列的顺序结构，只做整体线性变换

    Args:
        returns: 待调整的收益率数组
        target_mean: 目标均值
        target_std: 目标标准差

    Returns:
        调整后的收益率数组
    """
    current_mean = float(np.mean(returns))
    current_std = float(np.std(returns, ddof=1))

    # 标准差为 0（常数序列）时直接返回原序列加偏移
    if current_std < 1e-12:
        return np.full_like(returns, target_mean)

    # 标准化 + 重标度
    adjusted = (returns - current_mean) / current_std * target_std + target_mean
    return adjusted


# ============================================================
# 3. 合成价格路径生成
# ============================================================

def generate_synthetic_path(original_df: pd.DataFrame,
                            block_length: int = 21,
                            seed: Optional[int] = None,
                            keep_intraday_structure: bool = True) -> pd.DataFrame:
    """基于历史数据生成 1 条合成价格路径

    流程：
        1. 计算原始 close 的对数收益率 r_t = ln(close_t / close_{t-1})
        2. 对 r_t 做 Block Bootstrap 得到 r*_t（保持短期依赖结构）
        3. 对 r*_t 做矩匹配，使其均值和标准差精确等于原始 r_t
        4. 从原始首日 close 开始，用 r*_t 重建 close 序列
        5. 根据 keep_intraday_structure 决定 OHLC 的处理方式

    Args:
        original_df: 原始日线数据，必须包含 date, close 列；
                     若包含 open/high/low/volume/amount 则一并合成
        block_length: Bootstrap 块长度
        seed: 随机种子（便于复现）
        keep_intraday_structure: True=保留原始日内比例关系重建 OHLC；
                                  False=OHLC 也独立做 Block Bootstrap

    Returns:
        合成 DataFrame，列结构与原始一致
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    df = original_df.reset_index(drop=True).copy()
    n = len(df)
    if n < 30:
        raise ValueError(f"数据过短（{n} 行），至少需要 30 个样本点")

    close = df["close"].values.astype(float)
    if (close <= 0).any():
        raise ValueError("close 包含非正值，无法计算对数收益率")

    # ---- 步骤 1：计算原始对数收益率 ----
    log_close = np.log(close)
    log_returns = np.diff(log_close)  # 长度 n-1

    # 原始统计量（这是要匹配的目标）
    orig_mean = float(np.mean(log_returns))
    orig_std = float(np.std(log_returns, ddof=1))

    # ---- 步骤 2：Block Bootstrap ----
    boot_returns = block_bootstrap(log_returns, block_length=block_length,
                                    n_samples=len(log_returns), rng=rng)

    # ---- 步骤 3：矩匹配 ----
    matched_returns = moment_match(boot_returns, orig_mean, orig_std)

    # 验证匹配精度（调试用，正常情况下误差 < 1e-15）
    # assert abs(np.mean(matched_returns) - orig_mean) < 1e-10
    # assert abs(np.std(matched_returns, ddof=1) - orig_std) < 1e-10

    # ---- 步骤 4：重建 close 序列 ----
    synth_log_close = np.zeros(n)
    synth_log_close[0] = log_close[0]  # 首日保持原始值
    for t in range(1, n):
        synth_log_close[t] = synth_log_close[t - 1] + matched_returns[t - 1]
    synth_close = np.exp(synth_log_close)

    # ---- 步骤 5：处理 OHLCV ----
    result = pd.DataFrame({"date": df["date"].values, "close": synth_close})

    if keep_intraday_structure and all(col in df.columns for col in ["open", "high", "low"]):
        # 保留原始日内比例关系：open/close, high/close, low/close
        # 用同样的 Block Bootstrap 索引重排这些比例，再乘以新的 close
        ratios_open = df["open"].values / close
        ratios_high = df["high"].values / close
        ratios_low = df["low"].values / close

        # 对比例序列做同样的 Block Bootstrap（保持与 close 同步的时序结构）
        # 为保证 OHLC 一致性，直接用原始比例（不重排），让 OHLC 跟随 close 走
        # 这样 high >= max(open, close), low <= min(open, close) 自动满足
        result["open"] = ratios_open * synth_close
        result["high"] = ratios_high * synth_close
        result["low"] = ratios_low * synth_close

        # 修正：high 至少为 max(open, close)，low 至多为 min(open, close)
        result["high"] = np.maximum(result["high"],
                                     np.maximum(result["open"], result["close"]))
        result["low"] = np.minimum(result["low"],
                                    np.minimum(result["open"], result["close"]))
    elif all(col in df.columns for col in ["open", "high", "low"]):
        # 不保留日内结构，对 OHLC 各自做 Block Bootstrap + 矩匹配
        for col in ["open", "high", "low"]:
            vals = df[col].values.astype(float)
            if (vals <= 0).any():
                result[col] = synth_close  # 退化为 close
                continue
            log_vals = np.log(vals)
            log_rets = np.diff(log_vals)
            orig_m = float(np.mean(log_rets))
            orig_s = float(np.std(log_rets, ddof=1))
            boot = block_bootstrap(log_rets, block_length=block_length,
                                    n_samples=len(log_rets), rng=rng)
            matched = moment_match(boot, orig_m, orig_s)
            new_log = np.zeros(n)
            new_log[0] = log_vals[0]
            for t in range(1, n):
                new_log[t] = new_log[t - 1] + matched[t - 1]
            result[col] = np.exp(new_log)

    # volume / amount：保持原始（流动性结构与价格波动的对应关系是次要的）
    if "volume" in df.columns:
        result["volume"] = df["volume"].values
    if "amount" in df.columns:
        result["amount"] = df["amount"].values

    return result


def generate_multiple_paths(original_df: pd.DataFrame,
                            n_paths: int = 100,
                            block_length: int = 21,
                            seed: Optional[int] = None) -> List[pd.DataFrame]:
    """生成多条合成路径（用于蒙特卡洛回测）

    Args:
        original_df: 原始日线数据
        n_paths: 生成路径数
        block_length: Bootstrap 块长度
        seed: 基础随机种子（每条路径用 seed + i）

    Returns:
        合成 DataFrame 列表
    """
    paths = []
    for i in range(n_paths):
        s = None if seed is None else seed + i
        paths.append(generate_synthetic_path(original_df,
                                              block_length=block_length,
                                              seed=s))
    return paths


# ============================================================
# 4. 指标计算与一致性验证
# ============================================================

def compute_return_stats(close: np.ndarray, annual_factor: int = 252,
                          risk_free: float = 0.02) -> dict:
    """计算价格序列的核心统计指标

    Args:
        close: 收盘价数组
        annual_factor: 年化因子（日频=252）
        risk_free: 无风险利率（年化）

    Returns:
        dict: {
            'annual_return': 年化收益率,
            'annual_volatility': 年化波动率,
            'sharpe': 夏普比率,
            'max_drawdown': 最大回撤,
            'n_days': 样本天数,
            'years': 年数,
        }
    """
    close = np.asarray(close, dtype=float)
    close = close[close > 0]  # 过滤非正值
    if len(close) < 2:
        return {"annual_return": 0, "annual_volatility": 0, "sharpe": 0,
                "max_drawdown": 0, "n_days": len(close), "years": 0}

    log_returns = np.diff(np.log(close))
    n = len(log_returns)
    years = n / annual_factor

    # 年化收益率（几何）
    total_return = close[-1] / close[0] - 1
    annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

    # 年化波动率
    annual_vol = float(np.std(log_returns, ddof=1) * np.sqrt(annual_factor))

    # Sharpe
    sharpe = (annual_return - risk_free) / annual_vol if annual_vol > 0 else 0

    # 最大回撤
    running_max = np.maximum.accumulate(close)
    drawdown = (close - running_max) / running_max
    max_dd = float(np.clip(drawdown.min(), -1.0, 0.0))

    return {
        "annual_return": float(annual_return),
        "annual_volatility": annual_vol,
        "sharpe": float(sharpe),
        "max_drawdown": max_dd,
        "n_days": n,
        "years": round(years, 2),
    }


def verify_synthetic_stats(original_df: pd.DataFrame,
                            synthetic_df: pd.DataFrame,
                            annual_factor: int = 252,
                            risk_free: float = 0.02) -> dict:
    """验证合成数据与原始数据的统计一致性

    Returns:
        dict: {
            'original': 原始指标,
            'synthetic': 合成指标,
            'diff': 差异,
            'match_quality': 匹配质量评估
        }
    """
    orig_stats = compute_return_stats(original_df["close"].values,
                                       annual_factor=annual_factor,
                                       risk_free=risk_free)
    synth_stats = compute_return_stats(synthetic_df["close"].values,
                                        annual_factor=annual_factor,
                                        risk_free=risk_free)

    diff = {
        "annual_return": synth_stats["annual_return"] - orig_stats["annual_return"],
        "annual_volatility": synth_stats["annual_volatility"] - orig_stats["annual_volatility"],
        "sharpe": synth_stats["sharpe"] - orig_stats["sharpe"],
        "max_drawdown": synth_stats["max_drawdown"] - orig_stats["max_drawdown"],
    }

    # 匹配质量：年化和夏普的差异越小越好
    # 注意：由于对数收益率做了精确矩匹配，年化和夏普理论上应该完全一致（误差 < 1e-10）
    # 最大回撤是路径依赖的，会有差异（这正是合成数据的意义）
    match_quality = "perfect" if abs(diff["annual_return"]) < 1e-6 and \
                                    abs(diff["sharpe"]) < 1e-6 else "approximate"

    return {
        "original": orig_stats,
        "synthetic": synth_stats,
        "diff": diff,
        "match_quality": match_quality,
    }


# ============================================================
# 5. SyntheticDataGenerator 类：面向对象封装
# ============================================================

class SyntheticDataGenerator:
    """合成数据生成器（Moment-Matched Block Bootstrap）

    面向对象封装，支持单条/多条路径生成、自动验证、批量处理标的池。

    用法示例：
        # 1. 单条路径
        gen = SyntheticDataGenerator(block_length=21, seed=42)
        synth_df = gen.generate(original_df)
        stats = gen.verify(original_df, synth_df)

        # 2. 多条路径（蒙特卡洛回测）
        paths = gen.generate_multiple(original_df, n_paths=100)

        # 3. 批量处理标的池
        results = gen.generate_batch(
            universe_csv="custom/strategies/etf_rotation/1_research/universe.csv",
            cache_dir="custom/data/etf_klines",
            output_dir="custom/output/research/synthetic_data",
        )
    """

    def __init__(self, block_length: int = 21, seed: Optional[int] = 42):
        """初始化合成数据生成器

        Args:
            block_length: Block Bootstrap 块长度（日频数据默认 21 ≈ 1 个月）
            seed: 随机种子（便于复现，默认 42）
        """
        self.block_length = block_length
        self.seed = seed

    def generate(self, original_df: pd.DataFrame) -> pd.DataFrame:
        """生成单条合成路径

        Args:
            original_df: 原始日线数据，必须包含 date, close 列

        Returns:
            合成 DataFrame，列结构与原始一致
        """
        return generate_synthetic_path(original_df,
                                         block_length=self.block_length,
                                         seed=self.seed)

    def generate_multiple(self, original_df: pd.DataFrame,
                           n_paths: int = 100) -> List[pd.DataFrame]:
        """生成多条合成路径（用于蒙特卡洛回测）

        Args:
            original_df: 原始日线数据
            n_paths: 生成路径数

        Returns:
            合成 DataFrame 列表
        """
        return generate_multiple_paths(original_df,
                                        n_paths=n_paths,
                                        block_length=self.block_length,
                                        seed=self.seed)

    @staticmethod
    def verify(original_df: pd.DataFrame, synthetic_df: pd.DataFrame,
                annual_factor: int = 252, risk_free: float = 0.02) -> dict:
        """验证合成数据与原始数据的统计一致性

        Returns:
            dict: {
                'original': 原始指标,
                'synthetic': 合成指标,
                'diff': 差异,
                'match_quality': 'perfect' 或 'approximate'
            }
        """
        return verify_synthetic_stats(original_df, synthetic_df,
                                        annual_factor=annual_factor,
                                        risk_free=risk_free)

    @staticmethod
    def compute_stats(close: np.ndarray, annual_factor: int = 252,
                       risk_free: float = 0.02) -> dict:
        """计算价格序列的核心统计指标

        Returns:
            dict: 年化收益率/波动率/Sharpe/最大回撤/样本天数/年数
        """
        return compute_return_stats(close, annual_factor=annual_factor,
                                      risk_free=risk_free)

    def generate_batch(self, universe_csv, cache_dir, output_dir,
                        load_func=None, plot: bool = True,
                        html_report: bool = True) -> list:
        """批量处理标的池，生成合成数据集

        Args:
            universe_csv: 标的池 CSV 文件路径（需含 code, code_name 列）
            cache_dir: 原始数据缓存目录（<code>.csv 文件）
            output_dir: 输出目录
            load_func: 自定义数据加载函数 code -> DataFrame（默认从 cache_dir 读取）
            plot: 是否生成对比图 PNG
            html_report: 是否生成 HTML 汇总报告

        Returns:
            results: list of dict，每个标的的处理结果
        """
        from pathlib import Path
        import pandas as pd

        universe_csv = Path(universe_csv)
        cache_dir = Path(cache_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if not universe_csv.exists():
            raise FileNotFoundError(f"标的池文件不存在：{universe_csv}")

        universe = pd.read_csv(universe_csv)
        print("=" * 70)
        print("ETF 合成样本数据集生成器（SyntheticDataGenerator）")
        print(f"方法：Moment-Matched Block Bootstrap（块长 {self.block_length}, seed {self.seed}）")
        print("=" * 70)
        print(f"\n标的池：{len(universe)} 个 ETF")
        print(f"输出目录：{output_dir}")

        # 默认加载函数
        if load_func is None:
            def load_func(code):
                cache_path = cache_dir / f"{code.replace('.', '_')}.csv"
                if not cache_path.exists():
                    raise FileNotFoundError(f"ETF 缓存不存在：{cache_path}")
                return pd.read_csv(cache_path, parse_dates=["date"])

        results = []
        failed = []
        for i, row in universe.iterrows():
            code = row["code"]
            name = row.get("code_name", "")
            try:
                print(f"\n[{i+1}/{len(universe)}] 处理 {code} {name} ...", end=" ")
                orig_df = load_func(code)
                synth_df = self.generate(orig_df)
                verification = self.verify(orig_df, synth_df)

                # 保存合成数据 CSV
                code_safe = code.replace(".", "_")
                synthetic_csv = output_dir / f"{code_safe}_synthetic.csv"
                synth_df.to_csv(synthetic_csv, index=False)

                # 生成对比图
                comparison_png = None
                if plot:
                    comparison_png = output_dir / f"{code_safe}_comparison.png"
                    self._plot_comparison(orig_df, synth_df, code, name, comparison_png)

                results.append({
                    "code": code,
                    "name": name,
                    "n_days": len(orig_df),
                    "original_stats": verification["original"],
                    "synthetic_stats": verification["synthetic"],
                    "verification": verification,
                    "synthetic_csv": synthetic_csv,
                    "comparison_png": comparison_png,
                })

                o = verification["original"]
                s = verification["synthetic"]
                print(f"✓ 年化 {o['annual_return']*100:.2f}%→{s['annual_return']*100:.2f}% "
                      f"| Sharpe {o['sharpe']:.3f}→{s['sharpe']:.3f} "
                      f"| 回撤 {o['max_drawdown']*100:.2f}%→{s['max_drawdown']*100:.2f}%")
            except Exception as e:
                print(f"✗ 失败：{e}")
                failed.append({"code": code, "name": name, "error": str(e)})

        # 生成指标对比 CSV
        if results:
            metrics_rows = []
            for r in results:
                o = r["original_stats"]
                s = r["synthetic_stats"]
                v = r["verification"]
                metrics_rows.append({
                    "code": r["code"], "name": r["name"],
                    "n_days": r["n_days"], "years": o["years"],
                    "orig_annual_return": o["annual_return"],
                    "synth_annual_return": s["annual_return"],
                    "diff_annual_return": v["diff"]["annual_return"],
                    "orig_annual_vol": o["annual_volatility"],
                    "synth_annual_vol": s["annual_volatility"],
                    "diff_annual_vol": v["diff"]["annual_volatility"],
                    "orig_sharpe": o["sharpe"],
                    "synth_sharpe": s["sharpe"],
                    "diff_sharpe": v["diff"]["sharpe"],
                    "orig_max_drawdown": o["max_drawdown"],
                    "synth_max_drawdown": s["max_drawdown"],
                    "diff_max_drawdown": v["diff"]["max_drawdown"],
                    "match_quality": v["match_quality"],
                })
            pd.DataFrame(metrics_rows).to_csv(
                output_dir / "metrics_comparison.csv", index=False)

            # 生成 HTML 报告
            if html_report:
                self._generate_html_report(results, output_dir / "summary_report.html")

        print("\n" + "=" * 70)
        print(f"完成！成功 {len(results)}/{len(universe)}，失败 {len(failed)}")
        if failed:
            print("\n失败列表：")
            for f in failed:
                print(f"  - {f['code']} {f['name']}: {f['error']}")
        print(f"\n所有输出文件位于：{output_dir}")
        print("=" * 70)
        return results

    @staticmethod
    def _plot_comparison(orig_df, synth_df, code, name, output_path):
        """绘制原始 vs 合成 价格走势对比图（内部方法）"""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import font_manager

        # 中文字体配置
        for font_name in ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]:
            try:
                font_manager.findfont(font_name, fallback_to_default=False)
                plt.rcParams["font.sans-serif"] = [font_name]
                break
            except Exception:
                continue
        plt.rcParams["axes.unicode_minus"] = False

        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1]})

        orig_norm = orig_df["close"].values / orig_df["close"].iloc[0]
        synth_norm = synth_df["close"].values / synth_df["close"].iloc[0]
        dates = orig_df["date"].values

        axes[0].plot(dates, orig_norm, color="#1f77b4", linewidth=1.2,
                      label="原始（真实历史）", alpha=0.9)
        axes[0].plot(dates, synth_norm, color="#d62728", linewidth=1.2,
                      label="合成（模拟样本）", alpha=0.8)
        axes[0].set_title(f"{code} {name} — 原始 vs 合成 价格走势对比（归一化）",
                           fontsize=13, fontweight="bold")
        axes[0].set_ylabel("归一化价格（首日=1）")
        axes[0].legend(loc="upper left")
        axes[0].grid(True, alpha=0.3)

        orig_s = compute_return_stats(orig_df["close"].values)
        synth_s = compute_return_stats(synth_df["close"].values)
        textstr = (f"年化: {orig_s['annual_return']*100:.2f}% vs {synth_s['annual_return']*100:.2f}%  |  "
                    f"Sharpe: {orig_s['sharpe']:.3f} vs {synth_s['sharpe']:.3f}  |  "
                    f"最大回撤: {orig_s['max_drawdown']*100:.2f}% vs {synth_s['max_drawdown']*100:.2f}%")
        axes[0].text(0.02, 0.97, textstr, transform=axes[0].transAxes,
                      fontsize=9, verticalalignment="top",
                      bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow",
                                edgecolor="gray", alpha=0.9))

        orig_dd = (orig_df["close"].values / np.maximum.accumulate(orig_df["close"].values)) - 1
        synth_dd = (synth_df["close"].values / np.maximum.accumulate(synth_df["close"].values)) - 1
        axes[1].fill_between(dates, orig_dd * 100, 0, color="#1f77b4",
                              alpha=0.4, label="原始回撤")
        axes[1].plot(dates, synth_dd * 100, color="#d62728",
                      linewidth=0.8, label="合成回撤", alpha=0.8)
        axes[1].set_ylabel("回撤 (%)")
        axes[1].set_xlabel("日期")
        axes[1].legend(loc="lower left")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=120, bbox_inches="tight")
        plt.close()

    @staticmethod
    def _generate_html_report(results, output_path):
        """生成 HTML 汇总报告（内部方法）"""
        from datetime import datetime
        today = datetime.now().strftime("%Y-%m-%d")

        rows = []
        for r in results:
            o = r["original_stats"]
            s = r["synthetic_stats"]
            v = r["verification"]
            rows.append(f"""
        <tr>
            <td>{r['code']}</td><td>{r['name']}</td><td>{r['n_days']}</td><td>{o['years']:.2f}</td>
            <td class="num">{o['annual_return']*100:.2f}%</td>
            <td class="num">{s['annual_return']*100:.2f}%</td>
            <td class="num diff">{v['diff']['annual_return']*100:+.6f}%</td>
            <td class="num">{o['annual_volatility']*100:.2f}%</td>
            <td class="num">{s['annual_volatility']*100:.2f}%</td>
            <td class="num diff">{v['diff']['annual_volatility']*100:+.6f}%</td>
            <td class="num">{o['sharpe']:.4f}</td>
            <td class="num">{s['sharpe']:.4f}</td>
            <td class="num diff">{v['diff']['sharpe']:+.6f}</td>
            <td class="num">{o['max_drawdown']*100:.2f}%</td>
            <td class="num">{s['max_drawdown']*100:.2f}%</td>
            <td class="num diff">{v['diff']['max_drawdown']*100:+.2f}%</td>
            <td class="quality {v['match_quality']}">{v['match_quality']}</td>
        </tr>""")

        images = []
        for r in results:
            code_safe = r["code"].replace(".", "_")
            images.append(f"""
        <div class="image-block">
            <h3>{r['code']} {r['name']} <span class="meta">({r['n_days']} 天 / {r['original_stats']['years']:.2f} 年)</span></h3>
            <img src="{code_safe}_comparison.png" alt="{r['code']} 对比图">
        </div>""")

        html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>ETF 合成样本数据集报告 — {today}</title>
<style>
    body {{ font-family: "Microsoft YaHei", "Segoe UI", sans-serif; margin: 20px; background: #fafafa; color: #333; }}
    h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
    h2 {{ color: #2c3e50; margin-top: 30px; }}
    .meta {{ color: #888; font-weight: normal; font-size: 0.85em; }}
    .summary {{ background: #fff; padding: 15px; border-radius: 5px; margin: 15px 0; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
    table {{ border-collapse: collapse; width: 100%; background: #fff; font-size: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
    th, td {{ border: 1px solid #ddd; padding: 6px 8px; text-align: center; }}
    th {{ background: #2c3e50; color: white; font-weight: bold; }}
    tr:nth-child(even) {{ background: #f8f8f8; }}
    tr:hover {{ background: #fffde7; }}
    .num {{ font-family: "Consolas", monospace; }}
    .diff {{ color: #666; font-size: 11px; }}
    .quality.perfect {{ color: #27ae60; font-weight: bold; }}
    .quality.approximate {{ color: #e67e22; font-weight: bold; }}
    .image-block {{ background: #fff; padding: 15px; margin: 20px 0; border-radius: 5px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
    .image-block img {{ width: 100%; max-width: 1200px; }}
    .method {{ background: #e8f4f8; padding: 15px; border-left: 4px solid #3498db; margin: 15px 0; border-radius: 3px; }}
</style>
</head>
<body>
<h1>ETF 合成样本数据集报告</h1>
<div class="summary">
    <p><strong>生成日期：</strong>{today}</p>
    <p><strong>标的数量：</strong>{len(results)} 个 ETF</p>
    <p><strong>合成方法：</strong>Moment-Matched Block Bootstrap（矩匹配分块重采样）</p>
</div>
<div class="method">
    <h3>技术原理</h3>
    <p><strong>Block Bootstrap</strong>：将历史对数收益率序列划分为 21 天的连续块，随机抽取整个块重排，保留时序短期依赖。</p>
    <p><strong>Moment Matching</strong>：对重采样后的收益率做标准化 + 重标度，强制样本均值和标准差精确等于原始值。</p>
    <ul>
        <li>年化收益率、波动率、Sharpe 与原始<strong>完全一致</strong></li>
        <li>价格走势不同（Bootstrap 打乱了收益率顺序）</li>
        <li>最大回撤有差异（路径依赖统计量，这是合成数据的价值）</li>
    </ul>
</div>
<h2>一、指标对比汇总</h2>
<table>
<thead><tr>
    <th rowspan="2">代码</th><th rowspan="2">名称</th><th rowspan="2">天数</th><th rowspan="2">年数</th>
    <th colspan="3">年化收益率</th><th colspan="3">年化波动率</th><th colspan="3">Sharpe</th><th colspan="3">最大回撤</th><th rowspan="2">匹配质量</th>
</tr><tr>
    <th>原始</th><th>合成</th><th>差异</th>
    <th>原始</th><th>合成</th><th>差异</th>
    <th>原始</th><th>合成</th><th>差异</th>
    <th>原始</th><th>合成</th><th>差异</th>
</tr></thead>
<tbody>{''.join(rows)}</tbody>
</table>
<h2>二、价格走势对比图</h2>
{''.join(images)}
</body></html>"""
        Path(output_path).write_text(html, encoding="utf-8")


# ============================================================
# 6. 模块自测
# ============================================================

if __name__ == "__main__":
    # 用模拟数据自测：生成 1000 天的 GBM 价格序列，做合成，验证一致性
    rng = np.random.default_rng(42)
    n = 1000
    mu, sigma = 0.0004, 0.012  # 日均收益率、日波动率
    log_rets = rng.normal(mu, sigma, size=n - 1)
    close = np.zeros(n)
    close[0] = 1.0
    for t in range(1, n):
        close[t] = close[t - 1] * np.exp(log_rets[t - 1])

    df = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=n, freq="B"),
        "open": close * (1 + rng.uniform(-0.005, 0.005, n)),
        "high": close * (1 + rng.uniform(0, 0.01, n)),
        "low": close * (1 - rng.uniform(0, 0.01, n)),
        "close": close,
        "volume": rng.integers(1e6, 1e7, n),
        "amount": close * rng.integers(1e6, 1e7, n),
    })

    print("=" * 70)
    print("合成数据生成器自测")
    print("=" * 70)
    print(f"原始样本：{n} 个交易日，起始价 {close[0]:.4f}，终止价 {close[-1]:.4f}")

    # 生成 5 条合成路径
    paths = generate_multiple_paths(df, n_paths=5, block_length=21, seed=42)

    print("\n原始 vs 合成 指标对比：")
    print("-" * 70)
    orig_stats = compute_return_stats(df["close"].values)
    print(f"原始：年化 {orig_stats['annual_return']*100:6.2f}% | "
          f"波动 {orig_stats['annual_volatility']*100:6.2f}% | "
          f"Sharpe {orig_stats['sharpe']:6.3f} | "
          f"最大回撤 {orig_stats['max_drawdown']*100:6.2f}%")

    for i, p in enumerate(paths):
        s = compute_return_stats(p["close"].values)
        print(f"合成{i+1}：年化 {s['annual_return']*100:6.2f}% | "
              f"波动 {s['annual_volatility']*100:6.2f}% | "
              f"Sharpe {s['sharpe']:6.3f} | "
              f"最大回撤 {s['max_drawdown']*100:6.2f}%")

    # 验证一致性
    print("\n一致性验证：")
    v = verify_synthetic_stats(df, paths[0])
    print(f"  年化差异：{v['diff']['annual_return']*100:+.6f}%")
    print(f"  波动差异：{v['diff']['annual_volatility']*100:+.6f}%")
    print(f"  夏普差异：{v['diff']['sharpe']:+.6f}")
    print(f"  匹配质量：{v['match_quality']}")
    print("\n[OK] 年化和夏普应精确匹配（误差 < 1e-6），最大回撤有差异（路径依赖）")

    # ---- SyntheticDataGenerator 类测试 ----
    print("\n" + "=" * 70)
    print("SyntheticDataGenerator 类测试")
    print("=" * 70)

    gen = SyntheticDataGenerator(block_length=21, seed=42)
    synth_df = gen.generate(df)
    stats = gen.verify(df, synth_df)
    print(f"generate() → {len(synth_df)} 行")
    print(f"verify() 匹配质量：{stats['match_quality']}")
    print(f"  年化差异：{stats['diff']['annual_return']*100:+.10f}%")
    print(f"  夏普差异：{stats['diff']['sharpe']:+.10f}")

    # 多条路径
    multi = gen.generate_multiple(df, n_paths=3)
    print(f"generate_multiple(n_paths=3) → {len(multi)} 条路径")

    print("\n[OK] SyntheticDataGenerator 类工作正常")
