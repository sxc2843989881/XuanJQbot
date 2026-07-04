"""V75 ETF轮动策略 — 过拟合检测套件

四类检验方法：
  1. Walk-Forward Analysis        滚动训练/测试窗，测样本外收益稳定性
  2. Random Sub-Sample            随机剔除20%交易日，重复200次
  3. Parameter Perturbation       参数±20%单变量扰动，测敏感性
  4. Monte Carlo Permutation      周频信号置换，测信号-收益对应显著性

判定标准：
  - Walk-Forward:         样本外年化 >= 样本内的60%
  - Random Sub-Sample:    5%分位年化 >= 24%
  - Parameter Perturbation: 任一参数±20%扰动收益下降 <= 8pp
  - Monte Carlo Permutation: 实际年化位于随机分布95%分位以上

依赖：
  - QBot 回测引擎 (backtest_module.backtest_engine)
  - V75 策略信号逻辑 (本脚本精确复现)
  - 数据: index_480080.csv (成长100), index_480081.csv (价值100)

注意：本脚本只实现，暂不运行（运行是下一个任务）。
"""
import sys
import base64
import io
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ============================================================
# 路径配置
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent
STYLE_ROTATION_DIR = Path(r"c:\caches\sxc\style_rotation_strategy")
DATA_DIR = STYLE_ROTATION_DIR / "data"
OUTPUT_DIR = (SCRIPT_DIR.parent / "回测结果").resolve()
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
REPORT_PATH = OUTPUT_DIR / "overfitting_report.html"

sys.path.insert(0, str(STYLE_ROTATION_DIR))
from backtest_module.backtest_engine import BacktestInput, BacktestConfig, run_backtest_engine


# ============================================================
# V75 默认参数（与 two_etf_rotation_v75.py 完全一致）
# ============================================================
DEFAULT_PARAMS: Dict = {
    "f1":              0.5,    # 因子1权重（比价MA偏离）
    "f2":              5.0,    # 因子2权重（动量加速度）
    "ratio_ma":         20,    # 比价MA窗口
    "accel_period":    21,    # ROC周期
    "accel_diff":      10,    # 加速度差分周期
    "slope_window":    63,    # 斜率回归窗口
    "min_hold":         4,    # 最小持有期（周）
    "mom3_period":     3,    # 门控动量周期
    "clip_threshold":  0.02,  # 加速度差clip阈值
    "ratio_dev_mult": 30,    # tanh缩放因子
}

# 检验运行次数下限（资源允许可调高到1000）
N_SUBSAMPLE = 200
N_PERMUTATION = 200


# ============================================================
# 数据加载
# ============================================================
def load_data():
    """加载成长100(480080)和价值100(480081)指数数据。

    Returns:
        (g_close, v_close, g_open, v_open) 四个对齐的 pd.Series，按日期索引。
    """
    g_raw = pd.read_csv(DATA_DIR / "index_480080.csv")
    v_raw = pd.read_csv(DATA_DIR / "index_480081.csv")
    for d in (g_raw, v_raw):
        d["date"] = pd.to_datetime(d["date"])
        d["close"] = pd.to_numeric(d["close"], errors="coerce")
        if "open" in d.columns:
            d["open"] = pd.to_numeric(d["open"], errors="coerce")

    g_df = g_raw.set_index("date").sort_index()
    v_df = v_raw.set_index("date").sort_index()
    common = g_df.index.intersection(v_df.index)
    g_df = g_df.loc[common]
    v_df = v_df.loc[common]

    g_close = g_df["close"].astype(float)
    v_close = v_df["close"].astype(float)
    g_open = g_df["open"].astype(float) if "open" in g_df.columns else g_close.shift(1).fillna(g_close)
    v_open = v_df["open"].astype(float) if "open" in v_df.columns else v_close.shift(1).fillna(v_close)
    g_open = g_open.fillna(g_close.shift(1)).fillna(g_close)
    v_open = v_open.fillna(v_close.shift(1)).fillna(v_close)

    valid = ~(g_close.isna() | v_close.isna())
    g_close, v_close = g_close[valid], v_close[valid]
    g_open, v_open = g_open[valid], v_open[valid]

    # 确保索引名为 date（merge_asof 需要）
    for s in (g_close, v_close, g_open, v_open):
        s.index.name = "date"
    return g_close, v_close, g_open, v_open


# ============================================================
# V75 信号生成（精确复现 V75 逻辑）
# ============================================================
def _rolling_slope_r2(close_series: pd.Series, window: int) -> Tuple[pd.Series, pd.Series]:
    """滚动线性回归：返回 (slope, r2)。

    slope: 对数价格的线性回归斜率（per day）
    r2:    R²判定系数 (0~1)
    与 V75 原版完全一致。
    """
    y = np.log(close_series).astype(float)
    n = window
    x = np.arange(n).astype(float)
    x_mean = x.mean()
    x_var = ((x - x_mean) ** 2).sum()

    sum_xy = y.rolling(n).apply(lambda yw: np.sum(yw * x), raw=True)
    sum_y = y.rolling(n).sum()
    sum_y2 = y.rolling(n).apply(lambda yw: np.sum(yw * yw), raw=True)

    cov_num = sum_xy - n * x_mean * (sum_y / n)
    slope = cov_num / x_var

    y_var = sum_y2 - (sum_y ** 2) / n
    r2 = (slope ** 2 * x_var) / y_var.where(y_var != 0, np.nan)
    r2 = r2.clip(0, 1)
    return slope, r2


def _generate_v75_weekly_positions(g_close: pd.Series, v_close: pd.Series,
                                   params: Optional[Dict] = None) -> pd.DataFrame:
    """运行 V75 状态机，返回周频 DataFrame（含 'signal_wk' 列）。

    完整复现 V75 逻辑：
      1. 因子1 (f1): 比价MA偏离 — tanh(ratio_dev*ratio_dev_mult)*f1
      2. 因子2 (f2): 动量加速度 — (g_accel-v_accel).clip(-clip,clip)*f2
      3. style_score = f1 + f2
      4. 63日对数价格回归斜率×R² (smom_g/smom_v，零波动率覆盖用)
      5. 3日短期动量 (g_mom3/v_mom3，门控用)
      6. 周频采样 W-FRI
      7. 状态机：MIN_HOLD=4周 + 零波动率覆盖 + 3日动量门控
         - 主信号 score>0 → growth
         - score<=0 时若 current_pos==growth 且 smom_g>0 且 smom_v<=0 → 覆盖持成长
         - 切换"成长->价值"时门控：v_mom3 > g_mom3 才放行
         - "价值->成长"不经门控（避免延迟入场）
         - NaN动量放行
    所有因子计算使用 shift(1) 防未来函数（与V75原版一致）。

    Args:
        g_close, v_close: 收盘价序列（DatetimeIndex，索引名 'date'）
        params: 参数字典，None则用 DEFAULT_PARAMS

    Returns:
        pd.DataFrame，索引为周频 W-FRI 日期，含 'signal_wk' 列（"growth"/"value"）
    """
    if params is None:
        params = DEFAULT_PARAMS
    p = {**DEFAULT_PARAMS, **params}

    F1            = p["f1"]
    F2            = p["f2"]
    RATIO_MA      = int(p["ratio_ma"])
    ACCEL_PERIOD  = int(p["accel_period"])
    ACCEL_DIFF    = int(p["accel_diff"])
    SLOPE_WINDOW  = int(p["slope_window"])
    MIN_HOLD      = int(p["min_hold"])
    MOM3_PERIOD   = int(p["mom3_period"])
    CLIP_THR      = p["clip_threshold"]
    RATIO_DEV_MULT = p["ratio_dev_mult"]

    # ---- 因子1：比价MA偏离（shift(1) 防未来函数）----
    ratio = (g_close / v_close).shift(1)
    ratio_ma = ratio.rolling(RATIO_MA).mean()
    ratio_dev = ratio / ratio_ma - 1
    f1_signal = np.tanh(ratio_dev * RATIO_DEV_MULT) * F1
    style_score = f1_signal.copy()

    # ---- 因子2：动量加速度（shift(1) 防未来函数）----
    g_roc = g_close.pct_change(ACCEL_PERIOD).shift(1)
    v_roc = v_close.pct_change(ACCEL_PERIOD).shift(1)
    g_accel = g_roc - g_roc.shift(ACCEL_DIFF)
    v_accel = v_roc - v_roc.shift(ACCEL_DIFF)
    accel_diff = (g_accel - v_accel).clip(-CLIP_THR, CLIP_THR)
    f2_signal = accel_diff * F2
    style_score = style_score + f2_signal

    # ---- 63日斜率×R²（零波动率覆盖用）----
    slope_g, r2_g = _rolling_slope_r2(g_close.shift(1), SLOPE_WINDOW)
    slope_v, r2_v = _rolling_slope_r2(v_close.shift(1), SLOPE_WINDOW)
    smom_g = slope_g * r2_g
    smom_v = slope_v * r2_v

    # ---- 3日短期动量（门控用，不计入 style_score）----
    g_mom3 = g_close.pct_change(MOM3_PERIOD).shift(1)
    v_mom3 = v_close.pct_change(MOM3_PERIOD).shift(1)

    # ---- 周频采样 W-FRI（与V75一致，dropna + iloc[1:]）----
    df = pd.DataFrame({
        "style_score": style_score,
        "smom_g": smom_g, "smom_v": smom_v,
        "g_mom3": g_mom3, "v_mom3": v_mom3,
    }, index=g_close.index)

    df_wk = df.resample("W-FRI").last().dropna(
        subset=["style_score", "smom_g", "smom_v"]
    ).iloc[1:]

    # ---- 状态机决策 ----
    position = pd.Series(np.nan, index=df_wk.index)
    current_pos = None
    hold_weeks = 0

    for i in range(len(df_wk)):
        row = df_wk.iloc[i]
        score_i   = row["style_score"]
        smom_g_i  = row["smom_g"]
        smom_v_i  = row["smom_v"]
        g_mom3_i  = row["g_mom3"]
        v_mom3_i  = row["v_mom3"]

        cover_this_iter = False

        # 主信号
        if score_i > 0:
            target = "growth"
        else:
            # score<=0：本应切价值，加入零波动率覆盖判断
            if current_pos == "growth" and smom_g_i > 0 and smom_v_i <= 0:
                # 零波动率场景：成长在涨 + 价值真没涨 → 覆盖持成长
                target = "growth"
                cover_this_iter = True
            elif current_pos is None:
                target = "value" if smom_v_i > smom_g_i else "growth"
            else:
                target = "value"

        # 状态机：最小持有期检查
        if current_pos is None:
            # 初始建仓：不走门控
            current_pos = target
            hold_weeks = 1
            position.iloc[i] = 1.0 if target == "growth" else 0.0
        elif target != current_pos and hold_weeks >= MIN_HOLD:
            # 切换决策 — 应用3日动量门控
            # 门控不干扰覆盖路径（cover_this_iter=True 时跳过门控）
            # 门控仅拦截"成长->价值"退出切换（保护成长持仓）
            # "价值->成长"入场切换不经门控（避免延迟入场）
            gate_pass = True
            if not cover_this_iter and target == "value" and current_pos == "growth":
                if pd.isna(g_mom3_i) or pd.isna(v_mom3_i):
                    gate_pass = True  # NaN放行
                else:
                    gate_pass = v_mom3_i > g_mom3_i  # 价值3日动量须优于成长

            if not gate_pass:
                # 门控拦截：维持当前持仓
                hold_weeks += 1
                position.iloc[i] = 1.0 if current_pos == "growth" else 0.0
            else:
                # 门控通过（或覆盖路径）：执行切换
                current_pos = target
                hold_weeks = 1
                position.iloc[i] = 1.0 if target == "growth" else 0.0
        else:
            hold_weeks += 1
            position.iloc[i] = 1.0 if current_pos == "growth" else 0.0

    df_wk["pos"] = position
    df_wk = df_wk.dropna(subset=["pos"])
    df_wk["signal_wk"] = df_wk["pos"].map({1.0: "growth", 0.0: "value"})
    return df_wk


def _expand_weekly_to_daily(df_wk: pd.DataFrame, daily_index: pd.DatetimeIndex) -> pd.Series:
    """将周频信号扩展回日频（merge_asof backward，与 V75 一致）。

    每个日频日期获取最近一个已发生周频日期（W-FRI）的信号。
    """
    idx_name = daily_index.name or "date"
    df_daily = pd.DataFrame({idx_name: daily_index}).sort_values(idx_name)
    df_wk_reset = df_wk.reset_index().rename(columns={"index": idx_name})
    # 兼容 reset_index 后的列名（若索引名为 'date'，列名就是 'date'）
    if idx_name not in df_wk_reset.columns:
        # 取第一列（原索引列）重命名
        df_wk_reset = df_wk_reset.rename(columns={df_wk_reset.columns[0]: idx_name})
    df_wk_reset = df_wk_reset[[idx_name, "signal_wk"]].sort_values(idx_name)
    df_daily = pd.merge_asof(df_daily, df_wk_reset, on=idx_name, direction="backward")
    df_daily = df_daily.set_index(idx_name).sort_index()
    return df_daily["signal_wk"]


def generate_v75_signals(g_close: pd.Series, v_close: pd.Series,
                         params: Optional[Dict] = None) -> pd.Series:
    """生成 V75 日频信号序列（公开接口）。

    完整复现 V75 逻辑：f1+f2 双因子 + 零波动率覆盖 + 3日动量门控 + 周频状态机。
    所有因子计算使用 shift(1) 防未来函数（与V75原版一致）。

    Args:
        g_close: pd.Series，成长100收盘价（DatetimeIndex，索引名 'date'）
        v_close: pd.Series，价值100收盘价（DatetimeIndex，索引名 'date'）
        params: 参数字典（参见 DEFAULT_PARAMS），None则用默认值

    Returns:
        pd.Series，日频信号（"growth"/"value"），索引与 g_close 对齐。
        信号无效的早期时段（rolling窗口未填满）为 NaN。
    """
    if params is None:
        params = DEFAULT_PARAMS
    df_wk = _generate_v75_weekly_positions(g_close, v_close, params)
    return _expand_weekly_to_daily(df_wk, g_close.index)


# ============================================================
# 回测辅助
# ============================================================
def signals_to_bt_input(signals: pd.Series, g_close: pd.Series, v_close: pd.Series,
                        g_open: pd.Series, v_open: pd.Series) -> Tuple[BacktestInput, pd.DataFrame]:
    """将日频信号序列转换为 QBot BacktestInput。

    Returns:
        (bt_input, df_bt) — bt_input 给回测引擎，df_bt 含对齐后的价格+信号。
    """
    df_bt = pd.DataFrame({
        "g_open": g_open, "g_close": g_close,
        "v_open": v_open, "v_close": v_close,
        "signal": signals,
    }).dropna(subset=["signal"])

    bt_input = BacktestInput(
        dates=df_bt.index.strftime("%Y-%m-%d").values,
        value_open=df_bt["v_open"].values.astype(np.float64),
        value_close=df_bt["v_close"].values.astype(np.float64),
        growth_open=df_bt["g_open"].values.astype(np.float64),
        growth_close=df_bt["g_close"].values.astype(np.float64),
        signal=df_bt["signal"].values.astype(str),
    )
    return bt_input, df_bt


def run_bt(bt_input: BacktestInput, config: Optional[BacktestConfig] = None):
    """运行 QBot 回测引擎（默认与 V75 一致：commission=0.0001, gap_slippage=True）。"""
    if config is None:
        config = BacktestConfig(commission=0.0001, impact_slippage=0.0, apply_gap_slippage=True)
    return run_backtest_engine(bt_input, config)


def compute_metrics_on_slice(nav: np.ndarray, dates: np.ndarray,
                             start_dt=None, end_dt=None) -> Dict:
    """在 nav/dates 的子区间上计算 metrics（用于 walk-forward 的 test 窗口）。

    使用与 QBot 引擎一致的公式：annual_ret = (nav[-1]/nav[0])^(252/(n-1)) - 1。
    对于 walk-forward 的 1年测试窗，n≈252，公式 ≈ calendar-adjusted annual_ret。
    """
    nav_s = pd.Series(nav, index=pd.to_datetime(dates))
    if start_dt is not None:
        nav_s = nav_s[nav_s.index >= pd.to_datetime(start_dt)]
    if end_dt is not None:
        nav_s = nav_s[nav_s.index <= pd.to_datetime(end_dt)]
    if len(nav_s) < 2:
        return {"annual_ret": 0.0, "sharpe": 0.0, "max_dd": 0.0,
                "total_ret": 0.0, "n_days": len(nav_s),
                "start_date": None, "end_date": None}

    arr = nav_s.values
    n = len(arr)
    total_ret = float(arr[-1] / arr[0] - 1.0)
    annual_ret = float((arr[-1] / arr[0]) ** (252.0 / (n - 1)) - 1.0)
    rets = arr[1:] / arr[:-1] - 1.0
    annual_vol = float(np.std(rets, ddof=1) * np.sqrt(252.0)) if len(rets) > 1 else 0.0
    sharpe = float(annual_ret / annual_vol) if annual_vol > 0 else 0.0
    peak = np.maximum.accumulate(arr)
    dd = (arr - peak) / peak
    max_dd = float(np.min(dd))
    return {
        "annual_ret": annual_ret, "sharpe": sharpe, "max_dd": max_dd,
        "total_ret": total_ret, "n_days": n,
        "start_date": nav_s.index[0], "end_date": nav_s.index[-1],
    }


def compute_calendar_annual_ret(nav: np.ndarray, dates: np.ndarray) -> float:
    """用日历时间计算年化收益（用于 random subsample，因剔除了部分日期）。

    subsample 保留80%天数但跨相同日历区间，QBot 引擎的 252/(n-1) 公式会
    高估年化（因 n 缩小但日历时间不变），故改用日历年化。
    """
    if len(nav) < 2:
        return 0.0
    dates_dt = pd.to_datetime(dates)
    calendar_years = (dates_dt[-1] - dates_dt[0]).days / 365.25
    if calendar_years < 0.01:
        return 0.0
    return float((nav[-1] / nav[0]) ** (1.0 / calendar_years) - 1.0)


# ============================================================
# 方法1: Walk-Forward Analysis
# ============================================================
def walk_forward_analysis(g_close: pd.Series, v_close: pd.Series,
                          g_open: pd.Series, v_open: pd.Series,
                          train_years: int = 3, test_years: int = 1,
                          step_months: int = 6) -> Dict:
    """方法1：Walk-Forward Analysis。

    - 训练窗3年（约156周）作为状态机预热
    - 测试窗1年（约52周）评估样本外收益
    - 步长6个月（约26周）
    - V75参数固定（非训练选择参数），重点测样本外收益稳定性
    - 对每个测试窗运行回测（状态机从训练窗起点开始，3年预热）
    - 记录年化、Sharpe、回撤

    判定：样本外平均年化 >= 样本内年化的60%
    """
    print("\n[方法1] Walk-Forward Analysis ...")

    # In-sample baseline：全数据运行 V75
    signals_full = generate_v75_signals(g_close, v_close)
    bt_input_full, _ = signals_to_bt_input(signals_full, g_close, v_close, g_open, v_open)
    result_full = run_bt(bt_input_full)
    in_sample_ar = result_full.metrics["annual_ret"]
    in_sample_sharpe = result_full.metrics["sharpe"]
    in_sample_max_dd = result_full.metrics["max_dd"]
    print(f"  样本内年化: {in_sample_ar*100:.2f}%  Sharpe: {in_sample_sharpe:.3f}  "
          f"最大回撤: {in_sample_max_dd*100:.2f}%")

    # Walk-forward 窗口
    start_date = g_close.index[0]
    end_date = g_close.index[-1]
    train_delta = pd.DateOffset(years=train_years)
    test_delta = pd.DateOffset(years=test_years)
    step_delta = pd.DateOffset(months=step_months)

    test_windows = []
    test_start = start_date + train_delta
    while test_start + test_delta <= end_date + pd.Timedelta(days=1):
        test_end = test_start + test_delta
        train_start = test_start - train_delta

        # 切片数据 [train_start, test_end]（状态机从训练窗起点开始，预热3年）
        mask = (g_close.index >= train_start) & (g_close.index <= test_end)
        g_slice = g_close[mask]
        v_slice = v_close[mask]
        go_slice = g_open[mask]
        vo_slice = v_open[mask]
        if len(g_slice) < 200:
            test_start = test_start + step_delta
            continue

        # 在切片上运行 V75（状态机从 train_start 起点开始）
        signals_slice = generate_v75_signals(g_slice, v_slice)
        bt_input_slice, _ = signals_to_bt_input(signals_slice, g_slice, v_slice,
                                                 go_slice, vo_slice)
        result_slice = run_bt(bt_input_slice)

        # 只在 test 窗口上计算 metrics
        tm = compute_metrics_on_slice(result_slice.nav, result_slice.dates,
                                       start_dt=test_start, end_dt=test_end)
        test_windows.append({
            "train_start": train_start,
            "test_start": test_start,
            "test_end": test_end,
            **tm,
        })
        test_start = test_start + step_delta

    df_windows = pd.DataFrame(test_windows)
    if len(df_windows) == 0:
        return {"method": "walk_forward", "passed": False, "metric": 0.0,
                "threshold": 0.6 * in_sample_ar, "error": "无有效测试窗"}

    oos_ar_mean = float(df_windows["annual_ret"].mean())
    oos_ar_std = float(df_windows["annual_ret"].std()) if len(df_windows) > 1 else 0.0
    oos_sharpe_mean = float(df_windows["sharpe"].mean())
    oos_max_dd_mean = float(df_windows["max_dd"].mean())

    # OOS 累计净值：按测试窗 total_ret 连乘
    oos_cum_nav = 1.0
    for _, w in df_windows.iterrows():
        oos_cum_nav *= (1.0 + w["total_ret"])

    threshold = 0.6 * in_sample_ar
    passed = oos_ar_mean >= threshold
    print(f"  测试窗数: {len(df_windows)}  样本外平均年化: {oos_ar_mean*100:.2f}%  "
          f"阈值: {threshold*100:.2f}%  {'PASS' if passed else 'FAIL'}")

    return {
        "method": "walk_forward",
        "passed": passed,
        "metric": oos_ar_mean,
        "threshold": threshold,
        "in_sample_annual_ret": in_sample_ar,
        "in_sample_sharpe": in_sample_sharpe,
        "in_sample_max_dd": in_sample_max_dd,
        "test_windows": df_windows,
        "oos_annual_mean": oos_ar_mean,
        "oos_annual_std": oos_ar_std,
        "oos_sharpe_mean": oos_sharpe_mean,
        "oos_max_dd_mean": oos_max_dd_mean,
        "oos_cumulative_nav": oos_cum_nav,
        "n_windows": len(df_windows),
    }


# ============================================================
# 方法2: Random Sub-Sample
# ============================================================
def random_subsample(g_close: pd.Series, v_close: pd.Series,
                     g_open: pd.Series, v_open: pd.Series,
                     n_iter: int = N_SUBSAMPLE, drop_pct: float = 0.20,
                     seed: int = 42) -> Dict:
    """方法2：Random Sub-Sample。

    - 每次随机剔除20%交易日（保留80%），保持时序顺序
    - 在保留的子集上重新生成 V75 信号（rolling 窗口在子集上重算）
    - 运行回测，记录年化（日历调整）、Sharpe、最大回撤
    - 重复 n_iter 次（默认200，资源允许可调高到1000）

    判定：5%分位年化 >= 24%
    """
    print(f"\n[方法2] Random Sub-Sample (n={n_iter}, drop={drop_pct*100:.0f}%) ...")
    rng = np.random.default_rng(seed)
    n_total = len(g_close)
    n_keep = int(n_total * (1.0 - drop_pct))

    idx_arr = np.arange(n_total)
    results = []
    for i in range(n_iter):
        kept_idx = np.sort(rng.choice(idx_arr, size=n_keep, replace=False))
        g_sub = g_close.iloc[kept_idx]
        v_sub = v_close.iloc[kept_idx]
        go_sub = g_open.iloc[kept_idx]
        vo_sub = v_open.iloc[kept_idx]

        try:
            signals_sub = generate_v75_signals(g_sub, v_sub)
            bt_input_sub, _ = signals_to_bt_input(signals_sub, g_sub, v_sub, go_sub, vo_sub)
            result_sub = run_bt(bt_input_sub)
            # 日历调整年化（因剔除20%天数但跨相同日历区间）
            ar_cal = compute_calendar_annual_ret(result_sub.nav, result_sub.dates)
            m = result_sub.metrics
            results.append({
                "iter": i,
                "annual_ret": ar_cal,
                "annual_ret_engine": m["annual_ret"],
                "sharpe": m["sharpe"],
                "max_dd": m["max_dd"],
                "n_days": m["num_days"],
                "num_trades": m["num_trades"],
            })
        except Exception as e:
            results.append({"iter": i, "annual_ret": np.nan, "sharpe": np.nan,
                            "max_dd": np.nan, "n_days": 0, "num_trades": 0,
                            "error": str(e)})

        if (i + 1) % 20 == 0:
            print(f"  进度: {i+1}/{n_iter}")

    df_results = pd.DataFrame(results).dropna(subset=["annual_ret"])
    ar = df_results["annual_ret"]
    p5 = float(ar.quantile(0.05))
    p50 = float(ar.quantile(0.50))
    p95 = float(ar.quantile(0.95))
    threshold = 0.24
    passed = p5 >= threshold
    print(f"  年化分布: 5%={p5*100:.2f}%  50%={p50*100:.2f}%  95%={p95*100:.2f}%  "
          f"阈值: {threshold*100:.0f}%  {'PASS' if passed else 'FAIL'}")

    return {
        "method": "random_subsample",
        "passed": passed,
        "metric": p5,
        "threshold": threshold,
        "results": df_results,
        "annual_ret_p5": p5,
        "annual_ret_p50": p50,
        "annual_ret_p95": p95,
        "annual_ret_mean": float(ar.mean()),
        "annual_ret_std": float(ar.std()),
        "sharpe_p5": float(df_results["sharpe"].quantile(0.05)),
        "sharpe_p50": float(df_results["sharpe"].quantile(0.50)),
        "max_dd_p5": float(df_results["max_dd"].quantile(0.05)),
        "max_dd_p50": float(df_results["max_dd"].quantile(0.50)),
        "n_iter": n_iter,
        "drop_pct": drop_pct,
    }


# ============================================================
# 方法3: Parameter Perturbation
# ============================================================
def parameter_perturbation(g_close: pd.Series, v_close: pd.Series,
                           g_open: pd.Series, v_open: pd.Series) -> Dict:
    """方法3：Parameter Perturbation。

    扰动参数（单变量，±20%）：
      - f1:        0.5 ±20% → 0.4 / 0.6
      - f2:        5.0 ±20% → 4.0 / 6.0
      - ratio_ma:  20  ±20% → 16  / 24  (取整)
      - accel_period: 21 ±20% → 17 / 25 (取整，21*0.8=16.8→17, 21*1.2=25.2→25)

    每参数独立扰动，运行回测记录收益变化（pp = percentage points）。

    判定：任一参数±20%扰动收益下降 <= 8pp（即 max_drop_pp >= -8）
    """
    print("\n[方法3] Parameter Perturbation ...")

    # Baseline
    signals_base = generate_v75_signals(g_close, v_close)
    bt_input_base, _ = signals_to_bt_input(signals_base, g_close, v_close, g_open, v_open)
    result_base = run_bt(bt_input_base)
    base_ar = result_base.metrics["annual_ret"]
    base_sharpe = result_base.metrics["sharpe"]
    base_max_dd = result_base.metrics["max_dd"]
    print(f"  基准年化: {base_ar*100:.2f}%  Sharpe: {base_sharpe:.3f}")

    perturbations = [
        ("f1",            0.5, 0.4),
        ("f1",            0.5, 0.6),
        ("f2",            5.0, 4.0),
        ("f2",            5.0, 6.0),
        ("ratio_ma",      20,  16),
        ("ratio_ma",      20,  24),
        ("accel_period",  21,  17),   # 21*0.8=16.8 → 17
        ("accel_period",  21,  25),   # 21*1.2=25.2 → 25
    ]

    rows = []
    for param_name, base_val, pert_val in perturbations:
        params = {**DEFAULT_PARAMS, param_name: pert_val}
        try:
            signals = generate_v75_signals(g_close, v_close, params)
            bt_input, _ = signals_to_bt_input(signals, g_close, v_close, g_open, v_open)
            result = run_bt(bt_input)
            ar = result.metrics["annual_ret"]
            rows.append({
                "param": param_name,
                "base_val": base_val,
                "pert_val": pert_val,
                "direction": "-" if pert_val < base_val else "+",
                "annual_ret": ar,
                "delta_pp": (ar - base_ar) * 100,
                "sharpe": result.metrics["sharpe"],
                "max_dd": result.metrics["max_dd"],
            })
            print(f"  {param_name}={pert_val} (base={base_val}): "
                  f"年化={ar*100:.2f}%  Δ={ (ar - base_ar)*100:+.2f}pp")
        except Exception as e:
            rows.append({
                "param": param_name, "base_val": base_val, "pert_val": pert_val,
                "direction": "-" if pert_val < base_val else "+",
                "annual_ret": np.nan, "delta_pp": np.nan,
                "sharpe": np.nan, "max_dd": np.nan, "error": str(e),
            })

    df = pd.DataFrame(rows)
    max_drop_pp = float(df["delta_pp"].min())
    threshold_pp = -8.0  # 下降 <= 8pp 等价于 max_drop_pp >= -8
    passed = max_drop_pp >= threshold_pp
    print(f"  最大下降: {max_drop_pp:.2f}pp  阈值: {threshold_pp}pp  "
          f"{'PASS' if passed else 'FAIL'}")

    return {
        "method": "parameter_perturbation",
        "passed": passed,
        "metric": max_drop_pp,
        "threshold": threshold_pp,
        "base_annual_ret": base_ar,
        "base_sharpe": base_sharpe,
        "base_max_dd": base_max_dd,
        "results": df,
        "max_drop_pp": max_drop_pp,
        "max_drop_param": df.loc[df["delta_pp"].idxmin(), "param"] if len(df) > 0 else None,
    }


# ============================================================
# 方法4: Monte Carlo Permutation
# ============================================================
def monte_carlo_permutation(g_close: pd.Series, v_close: pd.Series,
                            g_open: pd.Series, v_open: pd.Series,
                            n_iter: int = N_PERMUTATION, seed: int = 42) -> Dict:
    """方法4：Monte Carlo Permutation。

    - 保持信号时间结构（周频W-FRI块结构）
    - 随机打乱周频 signal_wk 序列（破坏信号-收益对应）
    - 重复 n_iter 次（默认200，资源允许可调高到1000）
    - 记录每次年化收益
    - 输出：随机分布 + 策略实际收益的百分位

    判定：实际年化位于随机分布95%分位以上
    """
    print(f"\n[方法4] Monte Carlo Permutation (n={n_iter}) ...")
    rng = np.random.default_rng(seed)

    # 实际策略
    signals_actual = generate_v75_signals(g_close, v_close)
    bt_input_actual, _ = signals_to_bt_input(signals_actual, g_close, v_close, g_open, v_open)
    result_actual = run_bt(bt_input_actual)
    actual_ar = result_actual.metrics["annual_ret"]
    actual_sharpe = result_actual.metrics["sharpe"]
    actual_max_dd = result_actual.metrics["max_dd"]
    print(f"  实际年化: {actual_ar*100:.2f}%  Sharpe: {actual_sharpe:.3f}")

    # 获取周频信号序列（post-state-machine）
    df_wk = _generate_v75_weekly_positions(g_close, v_close)
    weekly_signals = df_wk["signal_wk"].values.copy()

    perm_results = []
    for i in range(n_iter):
        # 随机打乱周频信号（保持块结构，破坏信号-收益对应）
        shuffled = weekly_signals.copy()
        rng.shuffle(shuffled)

        df_wk_perm = df_wk.copy()
        df_wk_perm["signal_wk"] = shuffled

        # 扩展到日频
        daily_signals = _expand_weekly_to_daily(df_wk_perm, g_close.index)
        try:
            bt_input, _ = signals_to_bt_input(daily_signals, g_close, v_close, g_open, v_open)
            result = run_bt(bt_input)
            perm_results.append({
                "iter": i,
                "annual_ret": result.metrics["annual_ret"],
                "sharpe": result.metrics["sharpe"],
                "max_dd": result.metrics["max_dd"],
                "num_trades": result.metrics["num_trades"],
            })
        except Exception as e:
            perm_results.append({"iter": i, "annual_ret": np.nan, "sharpe": np.nan,
                                "max_dd": np.nan, "num_trades": 0, "error": str(e)})

        if (i + 1) % 20 == 0:
            print(f"  进度: {i+1}/{n_iter}")

    df_perm = pd.DataFrame(perm_results).dropna(subset=["annual_ret"])
    ar = df_perm["annual_ret"]
    p5 = float(ar.quantile(0.05))
    p50 = float(ar.quantile(0.50))
    p95 = float(ar.quantile(0.95))
    threshold = p95
    # 实际收益在随机分布中的百分位
    actual_pctile = float((ar < actual_ar).mean() * 100)
    passed = actual_ar >= p95
    print(f"  随机分布: 5%={p5*100:.2f}%  50%={p50*100:.2f}%  95%={p95*100:.2f}%  "
          f"实际百分位: {actual_pctile:.1f}%  {'PASS' if passed else 'FAIL'}")

    return {
        "method": "monte_carlo_permutation",
        "passed": passed,
        "metric": actual_ar,
        "threshold": threshold,
        "actual_annual_ret": actual_ar,
        "actual_sharpe": actual_sharpe,
        "actual_max_dd": actual_max_dd,
        "actual_percentile": actual_pctile,
        "perm_results": df_perm,
        "perm_p5": p5,
        "perm_p50": p50,
        "perm_p95": p95,
        "perm_mean": float(ar.mean()),
        "perm_std": float(ar.std()),
        "n_iter": n_iter,
    }


# ============================================================
# 图表辅助
# ============================================================
def fig_to_base64(fig) -> str:
    """将 matplotlib Figure 转 base64 PNG 字符串。"""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return b64


def plot_walk_forward(result: Dict) -> str:
    """方法1图表：各测试窗年化收益 + 样本内基准线。"""
    df = result["test_windows"]
    fig, ax = plt.subplots(figsize=(14, 6))
    x_labels = [f"{w['test_start'].strftime('%Y-%m')}" for _, w in df.iterrows()]
    bars = ax.bar(range(len(df)), df["annual_ret"] * 100,
                  color=["#27AE60" if v >= result["threshold"] * 100 else "#E74C3C"
                         for v in df["annual_ret"]],
                  alpha=0.8, edgecolor="black", linewidth=0.5)
    ax.axhline(result["in_sample_annual_ret"] * 100, color="#2C3E50", linestyle="-",
               linewidth=1.5, label=f"样本内年化 {result['in_sample_annual_ret']*100:.2f}%")
    ax.axhline(result["threshold"] * 100, color="#E67E22", linestyle="--", linewidth=1.5,
               label=f"阈值(60%) {result['threshold']*100:.2f}%")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("年化收益 (%)", fontsize=11)
    ax.set_title(f"Walk-Forward: 各测试窗样本外年化收益 (n={len(df)}, "
                 f"均值={result['oos_annual_mean']*100:.2f}%)", fontsize=12)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    return fig_to_base64(fig)


def plot_random_subsample(result: Dict) -> str:
    """方法2图表：年化收益分布直方图 + 分位线。"""
    ar = result["results"]["annual_ret"]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(ar * 100, bins=40, color="#3498DB", alpha=0.7, edgecolor="black", linewidth=0.5)
    ax.axvline(result["annual_ret_p5"] * 100, color="#E74C3C", linestyle="--", linewidth=2,
               label=f"5%分位 {result['annual_ret_p5']*100:.2f}%")
    ax.axvline(result["annual_ret_p50"] * 100, color="#F39C12", linestyle="-", linewidth=2,
               label=f"50%分位 {result['annual_ret_p50']*100:.2f}%")
    ax.axvline(result["annual_ret_p95"] * 100, color="#27AE60", linestyle="--", linewidth=2,
               label=f"95%分位 {result['annual_ret_p95']*100:.2f}%")
    ax.axvline(result["threshold"] * 100, color="black", linestyle=":", linewidth=2,
               label=f"阈值 {result['threshold']*100:.0f}%")
    ax.set_xlabel("年化收益 (%)", fontsize=11)
    ax.set_ylabel("频次", fontsize=11)
    ax.set_title(f"Random Sub-Sample: 年化收益分布 (n={result['n_iter']}, "
                 f"日历调整)", fontsize=12)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig_to_base64(fig)


def plot_parameter_perturbation(result: Dict) -> str:
    """方法3图表：参数扰动收益变化（pp）。"""
    df = result["results"]
    fig, ax = plt.subplots(figsize=(13, 6))
    labels = [f"{r['param']}\n{r['base_val']}→{r['pert_val']} ({r['direction']}20%)"
              for _, r in df.iterrows()]
    colors = ["#27AE60" if d >= -8 else "#E74C3C" for d in df["delta_pp"]]
    bars = ax.bar(range(len(df)), df["delta_pp"], color=colors, alpha=0.8,
                  edgecolor="black", linewidth=0.5)
    ax.axhline(0, color="black", linewidth=1)
    ax.axhline(-8, color="#E67E22", linestyle="--", linewidth=1.5, label="阈值 -8pp")
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("收益变化 (pp)", fontsize=11)
    ax.set_title(f"Parameter Perturbation: 参数±20%扰动收益变化 "
                 f"(最大下降 {result['max_drop_pp']:.2f}pp)", fontsize=12)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    # 在柱顶标注数值
    for bar, val in zip(bars, df["delta_pp"]):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + (0.2 if h >= 0 else -0.4),
                f"{val:+.2f}", ha="center", fontsize=8,
                va="bottom" if h >= 0 else "top")
    plt.tight_layout()
    return fig_to_base64(fig)


def plot_monte_carlo(result: Dict) -> str:
    """方法4图表：MC置换年化分布 + 实际收益线。"""
    ar = result["perm_results"]["annual_ret"]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(ar * 100, bins=40, color="#9B59B6", alpha=0.7, edgecolor="black", linewidth=0.5)
    ax.axvline(result["perm_p5"] * 100, color="#E74C3C", linestyle="--", linewidth=1.5,
               label=f"5%分位 {result['perm_p5']*100:.2f}%")
    ax.axvline(result["perm_p50"] * 100, color="#F39C12", linestyle="-", linewidth=1.5,
               label=f"50%分位 {result['perm_p50']*100:.2f}%")
    ax.axvline(result["perm_p95"] * 100, color="#27AE60", linestyle="--", linewidth=2,
               label=f"95%分位 {result['perm_p95']*100:.2f}% (阈值)")
    ax.axvline(result["actual_annual_ret"] * 100, color="black", linestyle=":", linewidth=2.5,
               label=f"实际年化 {result['actual_annual_ret']*100:.2f}% "
                     f"(P{result['actual_percentile']:.1f})")
    ax.set_xlabel("年化收益 (%)", fontsize=11)
    ax.set_ylabel("频次", fontsize=11)
    ax.set_title(f"Monte Carlo Permutation: 信号置换年化分布 (n={result['n_iter']})",
                 fontsize=12)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig_to_base64(fig)


# ============================================================
# HTML 报告生成
# ============================================================
def _fmt_pct(x, digits=2):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "N/A"
    return f"{x*100:.{digits}f}%"


def _fmt_num(x, digits=2):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "N/A"
    return f"{x:.{digits}f}"


def generate_report(results: Dict, output_path) -> str:
    """生成 HTML 报告，含四类方法各一节（matplotlib图表base64嵌入）+ 总结论。

    Args:
        results: dict，key 为方法名，value 为对应函数返回的结果 dict
        output_path: 输出 HTML 路径

    Returns:
        实际写入的路径字符串
    """
    wf = results.get("walk_forward", {})
    rs = results.get("random_subsample", {})
    pp = results.get("parameter_perturbation", {})
    mc = results.get("monte_carlo_permutation", {})

    n_pass = sum(1 for r in [wf, rs, pp, mc] if r.get("passed", False))
    overall_pass = n_pass >= 3  # 至少通过3/4才算总体通过

    html = []
    html.append("<!DOCTYPE html>")
    html.append('<html lang="zh-CN">')
    html.append("<head>")
    html.append('<meta charset="utf-8">')
    html.append('<meta name="viewport" content="width=device-width, initial-scale=1.0">')
    html.append("<title>V75 过拟合检测报告</title>")
    html.append("<style>")
    html.append("""
    body { font-family: "Microsoft YaHei", "SimHei", sans-serif;
           margin: 0; padding: 20px; background: #f5f5f5; color: #2c3e50;
           line-height: 1.6; }
    .container { max-width: 1200px; margin: 0 auto; background: white;
                 padding: 30px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
    h1 { color: #2c3e50; border-bottom: 3px solid #34495e; padding-bottom: 10px; }
    h2 { color: #34495e; border-left: 4px solid #3498db; padding-left: 10px;
         margin-top: 30px; }
    h3 { color: #34495e; margin-top: 20px; }
    .summary { background: #ecf0f1; padding: 15px; border-radius: 6px; margin: 15px 0; }
    .method { background: #fafafa; padding: 15px; border-radius: 6px;
              margin: 15px 0; border-left: 4px solid #bdc3c7; }
    .pass { border-left-color: #27ae60; }
    .fail { border-left-color: #e74c3c; }
    .badge { display: inline-block; padding: 3px 10px; border-radius: 4px;
             color: white; font-weight: bold; font-size: 12px; }
    .badge-pass { background: #27ae60; }
    .badge-fail { background: #e74c3c; }
    .metric { font-size: 18px; font-weight: bold; color: #2c3e50; }
    .threshold { color: #7f8c8d; font-size: 14px; }
    table { border-collapse: collapse; width: 100%; margin: 10px 0; font-size: 13px; }
    th, td { border: 1px solid #ddd; padding: 6px 10px; text-align: left; }
    th { background: #ecf0f1; font-weight: bold; }
    tr:nth-child(even) { background: #f9f9f9; }
    .chart { text-align: center; margin: 15px 0; }
    .chart img { max-width: 100%; border: 1px solid #ddd; border-radius: 4px; }
    .overall { background: #2c3e50; color: white; padding: 20px; border-radius: 6px;
               margin: 20px 0; text-align: center; }
    .overall-pass { background: #27ae60; }
    .overall-fail { background: #c0392b; }
    .small { font-size: 12px; color: #7f8c8d; }
    """)
    html.append("</style>")
    html.append("</head>")
    html.append("<body>")
    html.append('<div class="container">')

    # ---- 标题 ----
    html.append("<h1>V75 ETF轮动策略 — 过拟合检测报告</h1>")
    html.append(f'<p class="small">生成时间: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}</p>')
    html.append(f'<p class="small">策略: V75 (V74双因子 + 零波动率覆盖 + 3日动量门控)</p>')
    html.append(f'<p class="small">数据: 成长100(480080) + 价值100(480081) 周频轮动</p>')

    # ---- 总结论 ----
    overall_class = "overall-pass" if overall_pass else "overall-fail"
    overall_text = "通过" if overall_pass else "未通过"
    html.append(f'<div class="overall {overall_class}">')
    html.append(f'<h2 style="color:white; border:none; margin:0;">总体结论: {overall_text} '
                f'({n_pass}/4 项判定通过)</h2>')
    html.append(f'<p style="margin:8px 0 0 0;">至少通过3/4项判定为总体通过</p>')
    html.append("</div>")

    # ---- 四类判定概览 ----
    html.append('<div class="summary">')
    html.append("<h3>四类判定概览</h3>")
    html.append("<table>")
    html.append("<tr><th>方法</th><th>关键指标</th><th>阈值</th><th>判定</th></tr>")
    for r, name, metric_label in [
        (wf, "Walk-Forward", f"样本外平均年化 = {_fmt_pct(wf.get('oos_annual_mean'))}"),
        (rs, "Random Sub-Sample", f"5%分位年化 = {_fmt_pct(rs.get('annual_ret_p5'))}"),
        (pp, "Parameter Perturbation", f"最大下降 = {_fmt_num(pp.get('max_drop_pp'))}pp"),
        (mc, "Monte Carlo Permutation", f"实际年化 = {_fmt_pct(mc.get('actual_annual_ret'))}"),
    ]:
        badge = "PASS" if r.get("passed") else "FAIL"
        cls = "badge-pass" if r.get("passed") else "badge-fail"
        thr = r.get("threshold")
        if name == "Parameter Perturbation":
            thr_str = f"{thr:.1f}pp"
        else:
            thr_str = _fmt_pct(thr)
        html.append(f"<tr><td>{name}</td><td>{metric_label}</td>"
                    f"<td>{thr_str}</td><td><span class='badge {cls}'>{badge}</span></td></tr>")
    html.append("</table>")
    html.append("</div>")

    # ---- 方法1: Walk-Forward ----
    if wf:
        cls = "pass" if wf.get("passed") else "fail"
        badge = "PASS" if wf.get("passed") else "FAIL"
        bcls = "badge-pass" if wf.get("passed") else "badge-fail"
        html.append(f'<div class="method {cls}">')
        html.append(f'<h2>方法1: Walk-Forward Analysis <span class="badge {bcls}">{badge}</span></h2>')
        html.append('<p><b>方法说明:</b> 训练窗3年(状态机预热) + 测试窗1年 + 步长6个月。'
                    'V75参数固定非训练选择，重点测样本外收益稳定性。</p>')
        html.append(f'<p><b>判定标准:</b> 样本外平均年化 ≥ 样本内年化的60% '
                    f'(样本内 = {_fmt_pct(wf.get("in_sample_annual_ret"))})</p>')
        html.append(f'<p><span class="metric">样本外平均年化: {_fmt_pct(wf.get("oos_annual_mean"))}</span> '
                    f'<span class="threshold">阈值: {_fmt_pct(wf.get("threshold"))}</span></p>')
        html.append(f'<p class="small">样本内 Sharpe: {_fmt_num(wf.get("in_sample_sharpe"), 3)} | '
                    f'样本外 Sharpe均值: {_fmt_num(wf.get("oos_sharpe_mean"), 3)} | '
                    f'样本外 最大回撤均值: {_fmt_pct(wf.get("oos_max_dd_mean"))} | '
                    f'测试窗数: {wf.get("n_windows", 0)} | '
                    f'OOS累计净值: {_fmt_num(wf.get("oos_cumulative_nav"), 4)}</p>')
        if "chart_base64" not in wf:
            try:
                wf["chart_base64"] = plot_walk_forward(wf)
            except Exception as e:
                wf["chart_base64"] = None
                wf["chart_error"] = str(e)
        if wf.get("chart_base64"):
            html.append('<div class="chart">')
            html.append(f'<img src="data:image/png;base64,{wf["chart_base64"]}"/>')
            html.append('</div>')
        # 测试窗明细表
        df_w = wf.get("test_windows")
        if df_w is not None and len(df_w) > 0:
            html.append("<h3>各测试窗明细</h3>")
            html.append("<table>")
            html.append("<tr><th>测试起</th><th>测试止</th><th>年化</th>"
                        "<th>Sharpe</th><th>最大回撤</th><th>天数</th></tr>")
            for _, w in df_w.iterrows():
                html.append(f"<tr><td>{w['test_start'].strftime('%Y-%m-%d')}</td>"
                            f"<td>{w['test_end'].strftime('%Y-%m-%d')}</td>"
                            f"<td>{_fmt_pct(w['annual_ret'])}</td>"
                            f"<td>{_fmt_num(w['sharpe'], 3)}</td>"
                            f"<td>{_fmt_pct(w['max_dd'])}</td>"
                            f"<td>{int(w['n_days'])}</td></tr>")
            html.append("</table>")
        html.append('</div>')

    # ---- 方法2: Random Sub-Sample ----
    if rs:
        cls = "pass" if rs.get("passed") else "fail"
        badge = "PASS" if rs.get("passed") else "FAIL"
        bcls = "badge-pass" if rs.get("passed") else "badge-fail"
        html.append(f'<div class="method {cls}">')
        html.append(f'<h2>方法2: Random Sub-Sample <span class="badge {bcls}">{badge}</span></h2>')
        html.append(f'<p><b>方法说明:</b> 每次随机剔除20%交易日(保留80%)，'
                    f'在子集上重算信号+回测，重复 {rs.get("n_iter")} 次。'
                    f'年化收益用<b>日历调整</b>(因剔除天数但跨相同日历区间)。</p>')
        html.append(f'<p><b>判定标准:</b> 5%分位年化 ≥ 24%</p>')
        html.append(f'<p><span class="metric">5%分位年化: {_fmt_pct(rs.get("annual_ret_p5"))}</span> '
                    f'<span class="threshold">阈值: {_fmt_pct(rs.get("threshold"))}</span></p>')
        html.append(f'<p class="small">分布: 5%={_fmt_pct(rs.get("annual_ret_p5"))} | '
                    f'50%={_fmt_pct(rs.get("annual_ret_p50"))} | '
                    f'95%={_fmt_pct(rs.get("annual_ret_p95"))} | '
                    f'均值={_fmt_pct(rs.get("annual_ret_mean"))} | '
                    f'标准差={_fmt_pct(rs.get("annual_ret_std"))}</p>')
        html.append(f'<p class="small">Sharpe: 5%={_fmt_num(rs.get("sharpe_p5"), 3)} | '
                    f'50%={_fmt_num(rs.get("sharpe_p50"), 3)} | '
                    f'最大回撤 5%={_fmt_pct(rs.get("max_dd_p5"))} (worst)</p>')
        if "chart_base64" not in rs:
            try:
                rs["chart_base64"] = plot_random_subsample(rs)
            except Exception as e:
                rs["chart_base64"] = None
                rs["chart_error"] = str(e)
        if rs.get("chart_base64"):
            html.append('<div class="chart">')
            html.append(f'<img src="data:image/png;base64,{rs["chart_base64"]}"/>')
            html.append('</div>')
        html.append('</div>')

    # ---- 方法3: Parameter Perturbation ----
    if pp:
        cls = "pass" if pp.get("passed") else "fail"
        badge = "PASS" if pp.get("passed") else "FAIL"
        bcls = "badge-pass" if pp.get("passed") else "badge-fail"
        html.append(f'<div class="method {cls}">')
        html.append(f'<h2>方法3: Parameter Perturbation <span class="badge {bcls}">{badge}</span></h2>')
        html.append('<p><b>方法说明:</b> 单变量±20%扰动 f1/f2/ratio_ma/accel_period，'
                    '每参数独立扰动，运行回测记录收益变化。</p>')
        html.append(f'<p><b>判定标准:</b> 任一参数±20%扰动收益下降 ≤ 8pp '
                    f'(基准年化 = {_fmt_pct(pp.get("base_annual_ret"))})</p>')
        html.append(f'<p><span class="metric">最大下降: {_fmt_num(pp.get("max_drop_pp"))}pp</span> '
                    f'<span class="threshold">阈值: ≥ -8pp</span></p>')
        if pp.get("max_drop_param"):
            html.append(f'<p class="small">最敏感参数: {pp["max_drop_param"]}</p>')
        if "chart_base64" not in pp:
            try:
                pp["chart_base64"] = plot_parameter_perturbation(pp)
            except Exception as e:
                pp["chart_base64"] = None
                pp["chart_error"] = str(e)
        if pp.get("chart_base64"):
            html.append('<div class="chart">')
            html.append(f'<img src="data:image/png;base64,{pp["chart_base64"]}"/>')
            html.append('</div>')
        df_pp = pp.get("results")
        if df_pp is not None and len(df_pp) > 0:
            html.append("<h3>参数敏感性表</h3>")
            html.append("<table>")
            html.append("<tr><th>参数</th><th>基准值</th><th>扰动值</th><th>方向</th>"
                        "<th>年化</th><th>Δ (pp)</th><th>Sharpe</th><th>最大回撤</th></tr>")
            for _, r in df_pp.iterrows():
                html.append(f"<tr><td>{r['param']}</td><td>{r['base_val']}</td>"
                            f"<td>{r['pert_val']}</td><td>{r['direction']}20%</td>"
                            f"<td>{_fmt_pct(r['annual_ret'])}</td>"
                            f"<td>{_fmt_num(r['delta_pp'])}</td>"
                            f"<td>{_fmt_num(r['sharpe'], 3)}</td>"
                            f"<td>{_fmt_pct(r['max_dd'])}</td></tr>")
            html.append("</table>")
        html.append('</div>')

    # ---- 方法4: Monte Carlo Permutation ----
    if mc:
        cls = "pass" if mc.get("passed") else "fail"
        badge = "PASS" if mc.get("passed") else "FAIL"
        bcls = "badge-pass" if mc.get("passed") else "badge-fail"
        html.append(f'<div class="method {cls}">')
        html.append(f'<h2>方法4: Monte Carlo Permutation <span class="badge {bcls}">{badge}</span></h2>')
        html.append(f'<p><b>方法说明:</b> 保持周频W-FRI块结构，随机打乱周频signal序列'
                    f'(破坏信号-收益对应)，重复 {mc.get("n_iter")} 次。</p>')
        html.append('<p><b>判定标准:</b> 实际年化 ≥ 随机分布95%分位</p>')
        html.append(f'<p><span class="metric">实际年化: {_fmt_pct(mc.get("actual_annual_ret"))}</span> '
                    f'<span class="threshold">阈值(95%分位): {_fmt_pct(mc.get("perm_p95"))}</span></p>')
        html.append(f'<p class="small">实际百分位: P{mc.get("actual_percentile", 0):.1f} | '
                    f'随机分布: 5%={_fmt_pct(mc.get("perm_p5"))} | '
                    f'50%={_fmt_pct(mc.get("perm_p50"))} | '
                    f'95%={_fmt_pct(mc.get("perm_p95"))} | '
                    f'均值={_fmt_pct(mc.get("perm_mean"))}</p>')
        if "chart_base64" not in mc:
            try:
                mc["chart_base64"] = plot_monte_carlo(mc)
            except Exception as e:
                mc["chart_base64"] = None
                mc["chart_error"] = str(e)
        if mc.get("chart_base64"):
            html.append('<div class="chart">')
            html.append(f'<img src="data:image/png;base64,{mc["chart_base64"]}"/>')
            html.append('</div>')
        html.append('</div>')

    # ---- 脚注 ----
    html.append('<div class="summary">')
    html.append("<h3>判定标准说明</h3>")
    html.append("<ul>")
    html.append("<li><b>Walk-Forward:</b> 样本外平均年化 ≥ 样本内年化的60%</li>")
    html.append("<li><b>Random Sub-Sample:</b> 5%分位年化 ≥ 24% (日历调整)</li>")
    html.append("<li><b>Parameter Perturbation:</b> 任一参数±20%扰动收益下降 ≤ 8pp</li>")
    html.append("<li><b>Monte Carlo Permutation:</b> 实际年化 ≥ 随机分布95%分位</li>")
    html.append("</ul>")
    html.append("<p class='small'>注：Random Sub-Sample 的年化收益采用日历调整 "
                "(nav末值/初值)^(1/日历年数)-1，因剔除20%天数但跨相同日历区间，"
                "QBot引擎的 252/(n-1) 公式会高估年化。</p>")
    html.append("<p class='small'>注：Monte Carlo Permutation 在周频块级别打乱信号序列，"
                "保留W-FRI块结构与growth/value比例，破坏信号-收益时序对应。</p>")
    html.append("</div>")

    html.append("</div>")  # container
    html.append("</body>")
    html.append("</html>")

    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html))
    return str(output_path)


# ============================================================
# 主函数
# ============================================================
def main():
    """主函数：依次运行四类过拟合检验，生成 HTML 报告。"""
    print("=" * 80)
    print("V75 ETF轮动策略 — 过拟合检测套件")
    print("=" * 80)

    # 数据加载
    g_close, v_close, g_open, v_open = load_data()
    print(f"\n[数据] 共 {len(g_close)} 个交易日")
    print(f"  区间: {g_close.index[0]:%Y-%m-%d} ~ {g_close.index[-1]:%Y-%m-%d}")

    # 验证信号生成（与V75原版一致性自检）
    print("\n[自检] 验证 generate_v75_signals 输出 ...")
    test_signals = generate_v75_signals(g_close[:300], v_close[:300])
    print(f"  前300天信号: 有效 {test_signals.notna().sum()} 天, "
          f"NaN {test_signals.isna().sum()} 天")
    sig_counts = test_signals.dropna().value_counts()
    print(f"  分布: {dict(sig_counts)}")

    # 运行四类检验
    results = {}

    print("\n" + "=" * 80)
    print("[1/4] Walk-Forward Analysis")
    print("=" * 80)
    try:
        results["walk_forward"] = walk_forward_analysis(g_close, v_close, g_open, v_open)
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback; traceback.print_exc()
        results["walk_forward"] = {"method": "walk_forward", "passed": False,
                                    "metric": 0.0, "threshold": 0.0, "error": str(e)}

    print("\n" + "=" * 80)
    print("[2/4] Random Sub-Sample")
    print("=" * 80)
    try:
        results["random_subsample"] = random_subsample(g_close, v_close, g_open, v_open,
                                                       n_iter=N_SUBSAMPLE)
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback; traceback.print_exc()
        results["random_subsample"] = {"method": "random_subsample", "passed": False,
                                        "metric": 0.0, "threshold": 0.24, "error": str(e)}

    print("\n" + "=" * 80)
    print("[3/4] Parameter Perturbation")
    print("=" * 80)
    try:
        results["parameter_perturbation"] = parameter_perturbation(g_close, v_close, g_open, v_open)
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback; traceback.print_exc()
        results["parameter_perturbation"] = {"method": "parameter_perturbation", "passed": False,
                                              "metric": 0.0, "threshold": -8.0, "error": str(e)}

    print("\n" + "=" * 80)
    print("[4/4] Monte Carlo Permutation")
    print("=" * 80)
    try:
        results["monte_carlo_permutation"] = monte_carlo_permutation(g_close, v_close, g_open, v_open,
                                                                     n_iter=N_PERMUTATION)
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback; traceback.print_exc()
        results["monte_carlo_permutation"] = {"method": "monte_carlo_permutation", "passed": False,
                                                "metric": 0.0, "threshold": 0.0, "error": str(e)}

    # 生成报告
    print("\n" + "=" * 80)
    print("[报告] 生成 HTML 报告 ...")
    print("=" * 80)
    report_path = generate_report(results, REPORT_PATH)

    # 总结
    n_pass = sum(1 for r in results.values() if r.get("passed", False))
    print(f"\n[总结] 通过 {n_pass}/4 项判定")
    for name, r in results.items():
        status = "PASS" if r.get("passed") else "FAIL"
        print(f"  {name:30s} {status}")

    print(f"\n[输出] 报告已保存: {report_path}")
    print("=" * 80)
    return results


if __name__ == "__main__":
    main()
