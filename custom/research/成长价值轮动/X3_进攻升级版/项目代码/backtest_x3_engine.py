"""backtest_x3_engine.py — X3 策略集成回测（KST+连续加权+ADX过滤+clip测试）
================================================================
X3 策略 = KST-F1 + ADX过滤F2 + style_score连续加权 + 分段降仓(方案A)

  四项优化：
    1. F1升级为KST多周期动量（替代比价MA20方向）
    2. style_score连续加权（替代二元切换）
    3. ADX过滤F2（震荡市ADX<20时F2置0）
    4. F2 clip放宽测试（±0.02 vs ±0.025）

  消融分析：
    X3-full:          KST + 连续加权 + ADX + clip优化
    X3-no-KST:        MA20-F1 + 连续加权 + ADX + clip优化
    X3-no-weighting:  KST + 二元切换 + ADX + clip优化
    X3-no-ADX:        KST + 连续加权 + 无ADX + clip优化
    X3-no-clip:       KST + 连续加权 + ADX + clip=±0.02

  自定义引擎：
    现有 run_backtest_engine_weighted 只支持单方向(signal)+总权重(position_weight)，
    无法表达 "65%成长+35%价值" 的连续加权。因此编写 run_x3_weighted_backtest，
    接收 (w_g, w_v) 双资产权重数组，成本模型与现有引擎完全一致：
      - T日信号变化 → T+1日开盘价调仓
      - "全卖全买"模式（与现有引擎一致，确保X2基线可复现）
      - 佣金万1双边，无冲击滑点，无跳空滑点

数据:
  成长100: c:\\temp_v72_data\\index_480080.csv  （仅close可用）
  价值100: c:\\temp_v72_data\\index_480081.csv
  区间:    2012-12-31 ~ 2026-07-01

约束：
  - 指数 open=close
  - 佣金万1，无冲击滑点，无跳空滑点
  - 周频调仓（W-FRI）
  - 初始资金100万
  - Sharpe 用 rf=2.5%/年（与X1/X2口径一致）
  - 降仓逻辑保持X2方案A（分段降仓：MA50→50%，MA75→25%）
================================================================
"""
import sys
sys.path.insert(0, r'c:\XuanJLH\Qbot\custom\backtests')

from pathlib import Path
import numpy as np
import pandas as pd
from backtest_engine import BacktestConfig

# ============================================================
# 1. 参数
# ============================================================
F1_WEIGHT = 0.5       # F1因子权重
F2_WEIGHT = 5.0       # F2因子权重
MA_PCT = 0.97         # MA跌破阈值
CLIP_X2 = 0.02        # X2默认clip
CLIP_X3 = 0.025       # X3测试clip
ADX_PERIOD = 14       # ADX周期
ADX_THRESHOLD = 20    # ADX震荡市阈值
KST_STD_WINDOW = 252  # KST归一化滚动窗口

DATA_DIR = Path(r'c:\temp_v72_data')
REPORT_PATH = DATA_DIR / 'x3_ablation_report.txt'

# ============================================================
# 2. 加载数据
# ============================================================
print("=" * 70)
print("  X3 策略集成回测 — KST+连续加权+ADX过滤+clip测试")
print("=" * 70)

g_raw = pd.read_csv(str(DATA_DIR / 'index_480080.csv'))
v_raw = pd.read_csv(str(DATA_DIR / 'index_480081.csv'))

for d in (g_raw, v_raw):
    d['date'] = pd.to_datetime(d['date'])
    d['close'] = pd.to_numeric(d['close'], errors='coerce')

g_close = g_raw.set_index('date')['close'].astype(float).sort_index().dropna()
v_close = v_raw.set_index('date')['close'].astype(float).sort_index().dropna()
common = g_close.index.intersection(v_close.index)
g_close = g_close[common].sort_index()
v_close = v_close[common].sort_index()

print(f"数据区间: {g_close.index[0]:%Y-%m-%d} ~ {g_close.index[-1]:%Y-%m-%d}")
print(f"交易日数: {len(g_close)}")

# ============================================================
# 3. 因子计算
# ============================================================
print("\n[因子计算]")

# ---- 比价 ----
ratio = (g_close / v_close).shift(1)

# ---- F1选项A: MA20方向（X2原版） ----
ratio_ma20 = ratio.rolling(20).mean()
ratio_dev = ratio / ratio_ma20 - 1
f1_ma20 = np.tanh(ratio_dev * 30) * F1_WEIGHT

# ---- F1选项B: KST多周期动量 ----
roc10 = ratio.pct_change(10)
roc15 = ratio.pct_change(15)
roc20 = ratio.pct_change(20)
roc30 = ratio.pct_change(30)
kst = roc10 * 1 + roc15 * 2 + roc20 * 3 + roc30 * 4
kst_signal_line = kst.ewm(span=9, adjust=False).mean()  # EMA9信号线
kst_std = kst.rolling(KST_STD_WINDOW).std()
kst_normalized = kst / kst_std
f1_kst = np.tanh(kst_normalized) * F1_WEIGHT

print(f"  KST有效起始日: {kst_normalized.first_valid_index():%Y-%m-%d}")

# ---- F2: 动量加速度 ----
g_roc21 = g_close.pct_change(21).shift(1)
v_roc21 = v_close.pct_change(21).shift(1)
g_accel = g_roc21 - g_roc21.shift(10)
v_accel = v_roc21 - v_roc21.shift(10)
accel_diff_raw = g_accel - v_accel

# ---- ADX过滤（用close近似，无high/low） ----
def wilder_smooth(s, period):
    """正确的 Wilder smoothing 实现。

    Wilder 方法定义：
      - 首个平滑值 = 前 `period` 个有效值之和（放在 index start+period-1）
      - 之后递推：y[t] = y[t-1] * (period-1)/period + x[t]
      - 中间遇到的 NaN 视为 0（保留前值的衰减，避免数据空洞）

    注意：不能用 (s*p).ewm(alpha=1/p, adjust=False).mean() 近似！
      该 EWM 近似的 seed = s[first_non_nan]*p，
      而 Wilder 的 seed = sum(s[first_non_nan : first_non_nan+period])。
      当输入有前导 NaN（如 DX 序列）时，EWM 会把首个非 NaN 值放大 p 倍作为 seed，
      导致结果系统性偏高（实测 ADX 飙到 90~1400，正确范围应为 0~100）。
    """
    s = pd.Series(s, dtype=float)
    result = pd.Series(np.nan, index=s.index, dtype=float)
    first_valid = s.first_valid_index()
    if first_valid is None:
        return result
    start = s.index.get_loc(first_valid)
    if len(s) - start < period:
        return result
    # 首值 = 前 period 个值之和（NaN 视为 0）
    first_window = s.iloc[start:start + period].fillna(0.0)
    result.iloc[start + period - 1] = first_window.sum()
    # 递推
    for i in range(start + period, len(s)):
        prev = result.iloc[i - 1]
        curr = s.iloc[i]
        if np.isnan(curr):
            curr = 0.0
        result.iloc[i] = prev * (period - 1) / period + curr
    return result


def calc_adx_close(close, period=14):
    """用close计算ADX（Wilder方法）。
    无high/low时，用日收盘变化幅度近似：
      TR = |close[i] - close[i-1]|
      +DM = max(close[i] - close[i-1], 0)
      -DM = max(close[i-1] - close[i], 0)
    Wilder smoothing 使用正确实现（首值=前p个值之和），非EWM近似。
    diff[0]=NaN 不填充，让 Wilder 从 index 1 开始（与传统 ATR 起点一致）。
    """
    close = pd.Series(close, dtype=float)
    diff = close.diff()  # diff[0]=NaN，不fillna，让Wilder自然从index 1起
    tr = diff.abs()
    plus_dm = diff.clip(lower=0)
    minus_dm = (-diff).clip(lower=0)

    atr = wilder_smooth(tr, period)
    plus_dm_avg = wilder_smooth(plus_dm, period)
    minus_dm_avg = wilder_smooth(minus_dm, period)

    atr_safe = atr.replace(0, np.nan)
    plus_di = 100 * plus_dm_avg / atr_safe
    minus_di = 100 * minus_dm_avg / atr_safe

    di_sum = (plus_di + minus_di).replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / di_sum
    # ADX = Wilder平滑的DX。wilder_smooth返回sum-scale（首值=sum，稳态≈p*mean），
    # 但ADX惯例为average-scale（0~100），故除以period转换为平均值口径。
    # 这等价于标准Wilder ADX递推：ADX[t] = (ADX[t-1]*(p-1) + DX[t]) / p
    # DI计算不受影响（plus_dm_avg/atr的sum-scale在比值中抵消）。
    adx = wilder_smooth(dx, period) / period
    return adx

adx = calc_adx_close(g_close, ADX_PERIOD).shift(1)  # shift避免前视
print(f"  ADX有效起始日: {adx.first_valid_index():%Y-%m-%d}")
print(f"  ADX<20占比: {(adx < ADX_THRESHOLD).sum() / adx.notna().sum() * 100:.1f}%")

# ---- 连续加权输入 ----
g_roc60 = g_close.pct_change(60).shift(1)
v_roc60 = v_close.pct_change(60).shift(1)
both_up = (g_roc60 > 0) & (v_roc60 > 0)

g_vol = g_close.pct_change().rolling(60).std().shift(1)
v_vol = v_close.pct_change().rolling(60).std().shift(1)
vol_ratio = (g_vol / v_vol).fillna(1.0)  # NaN时无加成

# ---- 择时信号（降仓方案A：MA50→50%, MA75→25%） ----
g_ma50 = g_close.shift(1).rolling(50).mean()
v_ma50 = v_close.shift(1).rolling(50).mean()
both_below_ma50 = (g_close.shift(1) < g_ma50 * MA_PCT) & (v_close.shift(1) < v_ma50 * MA_PCT)

g_ma75 = g_close.shift(1).rolling(75).mean()
v_ma75 = v_close.shift(1).rolling(75).mean()
both_below_ma75 = (g_close.shift(1) < g_ma75 * MA_PCT) & (v_close.shift(1) < v_ma75 * MA_PCT)


# ============================================================
# 4. 连续加权函数
# ============================================================
def compute_continuous_weights(style_score, both_up, vol_ratio):
    """三档连续加权，返回 (w_g, w_v)，w_g+w_v=1.0。
      style_score > 0.3:  成长超配 0.65（都涨时波动率加成，上限0.75）
      style_score < -0.3: 价值超配 0.65（w_g=0.35）
      其他:               均衡 0.5/0.5
    """
    w_g = pd.Series(0.5, index=style_score.index, dtype=float)

    # 成长超配
    mask_g = style_score > 0.3
    w_g[mask_g] = 0.65
    # 都涨时波动率加成
    mask_both = mask_g & both_up
    w_g_boost = (0.65 + (vol_ratio - 1) * 0.1).clip(lower=0.5, upper=0.75)
    w_g[mask_both] = w_g_boost[mask_both]

    # 价值超配
    mask_v = style_score < -0.3
    w_g[mask_v] = 0.35

    w_v = 1.0 - w_g
    return w_g, w_v


def compute_binary_weights(candidate_g):
    """二元切换：100%成长 or 100%价值。返回 (w_g, w_v)。"""
    w_g = (candidate_g > 0).astype(float)
    w_v = 1.0 - w_g
    return w_g, w_v


# ============================================================
# 5. 自定义 X3 回测引擎（支持 w_g/w_v 双资产连续加权）
# ============================================================
def run_x3_weighted_backtest(dates, g_open, g_close_arr, v_open, v_close_arr,
                              w_g, w_v, config):
    """
    X3 自定义回测引擎 — 支持 (w_g, w_v) 连续加权。

    w_g[i], w_v[i]: 第i日收盘后决定的成长/价值权重
    w_g[i] + w_v[i] <= 1.0（剩余为现金）

    调仓规则（与 backtest_engine.py 一致）：
      - 第i日权重变化 → 第i+1日开盘价调仓
      - "全卖全买"模式（与现有引擎一致，确保X2基线可复现）
      - 佣金双边，冲击滑点，跳空滑点（报告用）

    返回: dict(nav, daily_ret, num_trades, metrics_dict)
    """
    n = len(dates)
    nav = np.zeros(n, dtype=np.float64)

    cash = float(config.start_cash)
    g_shares = 0.0
    v_shares = 0.0
    num_trades = 0

    pending = False
    pending_w_g = 0.0
    pending_w_v = 0.0

    for i in range(n):
        if i == 0:
            target_w_g = float(w_g[i])
            target_w_v = float(w_v[i])
            # 买入成长
            invest_g = cash * target_w_g
            if invest_g > 1e-8:
                g_price = g_open[i] * (1.0 + config.impact_slippage)
                comm = invest_g * config.commission
                g_shares = (invest_g - comm) / g_price
                cash -= invest_g
            # 买入价值
            invest_v = cash * target_w_v
            if invest_v > 1e-8:
                v_price = v_open[i] * (1.0 + config.impact_slippage)
                comm = invest_v * config.commission
                v_shares = (invest_v - comm) / v_price
                cash -= invest_v
            num_trades += 1
            nav[i] = g_shares * g_close_arr[i] + v_shares * v_close_arr[i] + cash
            continue

        # 执行待处理调仓（T+1开盘价）
        if pending:
            # 全卖成长
            if g_shares > 1e-12:
                sell_price = g_open[i] * (1.0 - config.impact_slippage)
                sell_amount = g_shares * sell_price
                comm = sell_amount * config.commission
                cash += sell_amount - comm
                g_shares = 0.0
            # 全卖价值
            if v_shares > 1e-12:
                sell_price = v_open[i] * (1.0 - config.impact_slippage)
                sell_amount = v_shares * sell_price
                comm = sell_amount * config.commission
                cash += sell_amount - comm
                v_shares = 0.0
            # 买入新成长
            target_w_g = pending_w_g
            target_w_v = pending_w_v
            invest_g = cash * target_w_g
            if invest_g > 1e-8:
                g_price = g_open[i] * (1.0 + config.impact_slippage)
                comm = invest_g * config.commission
                g_shares = (invest_g - comm) / g_price
                cash -= invest_g
            # 买入新价值
            invest_v = cash * target_w_v
            if invest_v > 1e-8:
                v_price = v_open[i] * (1.0 + config.impact_slippage)
                comm = invest_v * config.commission
                v_shares = (invest_v - comm) / v_price
                cash -= invest_v
            num_trades += 1
            pending = False

        # 当日净值
        nav[i] = g_shares * g_close_arr[i] + v_shares * v_close_arr[i] + cash

        # 检查是否需要调仓
        if abs(w_g[i] - w_g[i - 1]) > 1e-10 or abs(w_v[i] - w_v[i - 1]) > 1e-10:
            pending = True
            pending_w_g = float(w_g[i])
            pending_w_v = float(w_v[i])

    daily_ret = np.zeros(n)
    daily_ret[1:] = nav[1:] / nav[:-1] - 1.0

    return {
        'nav': nav,
        'daily_ret': daily_ret,
        'num_trades': num_trades,
        'dates': dates,
    }


# ============================================================
# 6. 指标计算（与 X2 calc_metrics_v72 一致）
# ============================================================
def calc_metrics_v72(nav, daily_ret, num_trades, freq=252, rf_annual=0.025):
    """V72 风格指标，Sharpe 用 rf=2.5%/年。"""
    r = pd.Series(daily_ret)
    eq = (1 + r).cumprod()
    n = len(r)
    years = n / freq
    total = nav[-1] / nav[0] - 1
    ann = (1 + total) ** (1 / years) - 1 if years > 0 else 0
    vol = r.std() * np.sqrt(freq)
    rf_p = rf_annual / freq
    sharpe = (r.mean() - rf_p) / r.std() * np.sqrt(freq) if r.std() > 0 else 0
    peak = eq.cummax()
    dd = (eq - peak) / peak
    max_dd = dd.min()
    calmar = ann / abs(max_dd) if max_dd < 0 else 0
    wr = (r > 0).sum() / (r != 0).sum() if (r != 0).sum() > 0 else 0
    return {
        'ann': ann,
        'dd': max_dd,
        'sharpe': sharpe,
        'calmar': calmar,
        'wr': wr,
        'total': total,
        'n_trades': num_trades,
        'final_nav': nav[-1],
        'final_multiple': nav[-1] / nav[0],
        'num_days': n,
    }


# ============================================================
# 7. 构建变体（w_g, w_v 数组）
# ============================================================
def build_variant(use_kst=True, use_continuous=True, use_adx=True,
                  clip_val=CLIP_X3, g_close_s=None, v_close_s=None):
    """构建一个变体的 (w_g_daily, w_v_daily) 日频数组。

    参数:
      use_kst:       True=KST-F1, False=MA20-F1
      use_continuous: True=连续加权, False=二元切换
      use_adx:       True=ADX过滤F2, False=无过滤
      clip_val:      F2的clip值（±clip_val）
    """
    # F1
    f1 = f1_kst if use_kst else f1_ma20

    # F2（带clip）
    accel_diff = accel_diff_raw.clip(-clip_val, clip_val)
    f2 = accel_diff * F2_WEIGHT
    # ADX过滤
    if use_adx:
        f2 = f2.where(adx >= ADX_THRESHOLD, 0.0)

    # style_score
    style_score = f1 + f2
    candidate_g = style_score > 0

    # 权重
    if use_continuous:
        w_g_base, w_v_base = compute_continuous_weights(style_score, both_up, vol_ratio)
    else:
        w_g_base, w_v_base = compute_binary_weights(candidate_g)

    # 降仓方案A：MA50→50%, MA75→25%
    cut_mult = pd.Series(1.0, index=style_score.index)
    cut_mult[both_below_ma75] = 0.25
    cut_mult[both_below_ma50 & ~both_below_ma75] = 0.5

    w_g_eff = w_g_base * cut_mult
    w_v_eff = w_v_base * cut_mult

    return w_g_eff, w_v_eff


def expand_to_daily(weekly_series, daily_index):
    """周频信号前向填充到日频（与X2一致）"""
    s = weekly_series.copy()
    idx_pos = daily_index.searchsorted(s.index, side='right') - 1
    valid = idx_pos >= 0
    s = s[valid]
    idx_pos = idx_pos[valid]
    s.index = daily_index[idx_pos]
    s = s[~s.index.duplicated(keep='last')]
    daily = pd.Series(np.nan, index=daily_index, dtype=s.dtype)
    daily.loc[s.index] = s
    return daily.ffill()


def run_variant(w_g_daily, w_v_daily, g_close_s, v_close_s, label=""):
    """运行一个变体的回测，返回 metrics dict。"""
    # 对齐 & 删除起始NaN
    mask = ~(w_g_daily.isna() | w_v_daily.isna())
    w_g_arr = w_g_daily[mask].astype(float)
    w_v_arr = w_v_daily[mask].astype(float)
    g_align = g_close_s.loc[w_g_arr.index]
    v_align = v_close_s.loc[w_g_arr.index]

    n = len(w_g_arr)
    dates = w_g_arr.index.strftime('%Y-%m-%d').values
    # 指数 open=close
    g_open_arr = g_align.values.astype(np.float64)
    g_close_arr = g_align.values.astype(np.float64)
    v_open_arr = v_align.values.astype(np.float64)
    v_close_arr = v_align.values.astype(np.float64)
    w_g_np = w_g_arr.values.astype(np.float64)
    w_v_np = w_v_arr.values.astype(np.float64)

    config = BacktestConfig(
        start_cash=1_000_000.0,
        commission=0.0001,
        impact_slippage=0.0,
        apply_gap_slippage=False,
    )

    result = run_x3_weighted_backtest(
        dates, g_open_arr, g_close_arr, v_open_arr, v_close_arr,
        w_g_np, w_v_np, config
    )
    m = calc_metrics_v72(result['nav'], result['daily_ret'], result['num_trades'])
    print(f"  {label:<24}年化={m['ann']*100:6.2f}%  回撤={m['dd']*100:6.2f}%  "
          f"Sharpe={m['sharpe']:.3f}  Calmar={m['calmar']:.3f}  调仓={m['n_trades']}")
    return m


# ============================================================
# 8. 运行所有变体
# ============================================================
# 确定共同起始日（KST需要252日窗口）
kst_start = kst_normalized.first_valid_index()
print(f"\n[共同起始日] KST有效日: {kst_start:%Y-%m-%d}")
print("  所有变体从该日起回测，确保公平对比\n")

# 构建所有变体（日频）
variants_daily = {}
configs = {
    'X3-full':          dict(use_kst=True,  use_continuous=True,  use_adx=True,  clip_val=CLIP_X3),
    'X3-no-KST':        dict(use_kst=False, use_continuous=True,  use_adx=True,  clip_val=CLIP_X3),
    'X3-no-weighting':  dict(use_kst=True,  use_continuous=False, use_adx=True,  clip_val=CLIP_X3),
    'X3-no-ADX':        dict(use_kst=True,  use_continuous=True,  use_adx=False, clip_val=CLIP_X3),
    'X3-no-clip':       dict(use_kst=True,  use_continuous=True,  use_adx=True,  clip_val=CLIP_X2),
}

# 构建日频权重
print("[构建变体日频权重]")
variant_wg_daily = {}
variant_wv_daily = {}
for name, cfg in configs.items():
    wg, wv = build_variant(g_close_s=g_close, v_close_s=v_close, **cfg)
    variant_wg_daily[name] = wg
    variant_wv_daily[name] = wv
    print(f"  {name:<24} use_kst={cfg['use_kst']}, continuous={cfg['use_continuous']}, "
          f"adx={cfg['use_adx']}, clip=±{cfg['clip_val']}")

# 周频采样 + 展开到日频
print("\n[周频采样 W-FRI 并展开到日频]")
variant_wg_weekly_expanded = {}
variant_wv_weekly_expanded = {}
for name in configs:
    df_tmp = pd.DataFrame({
        'wg': variant_wg_daily[name],
        'wv': variant_wv_daily[name],
    })
    weekly_tmp = df_tmp.resample('W-FRI').last().dropna(subset=['wg']).iloc[1:]
    wg_expanded = expand_to_daily(weekly_tmp['wg'], g_close.index)
    wv_expanded = expand_to_daily(weekly_tmp['wv'], g_close.index)
    variant_wg_weekly_expanded[name] = wg_expanded
    variant_wv_weekly_expanded[name] = wv_expanded

# X2基线（二元切换 + MA20-F1 + 无ADX + clip=0.02 + 分段降仓）
# 与X2正式版方案A完全一致，用于验证引擎一致性
print("\n[构建X2基线（验证引擎一致性）]")
wg_x2_daily, wv_x2_daily = build_variant(
    use_kst=False, use_continuous=False, use_adx=False,
    clip_val=CLIP_X2, g_close_s=g_close, v_close_s=v_close
)
df_x2 = pd.DataFrame({'wg': wg_x2_daily, 'wv': wv_x2_daily})
weekly_x2 = df_x2.resample('W-FRI').last().dropna(subset=['wg']).iloc[1:]
wg_x2_exp = expand_to_daily(weekly_x2['wg'], g_close.index)
wv_x2_exp = expand_to_daily(weekly_x2['wv'], g_close.index)

# 运行 X2基线（全周期）
print("[运行] X2基线（全周期，验证引擎一致性）")
m_x2_full = run_variant(wg_x2_exp, wv_x2_exp, g_close, v_close, "X2基线(全周期)")

# 运行 X2基线（KST起始日，公平对比）
print("\n[运行] X2基线（KST起始日，公平对比）")
mask_kst = wg_x2_exp.index >= kst_start
m_x2_kst = run_variant(
    wg_x2_exp[mask_kst], wv_x2_exp[mask_kst],
    g_close, v_close, "X2基线(KST起始)"
)

# 运行所有X3变体
print("\n[运行] X3 消融分析（所有变体从KST起始日）")
results = {}
for name in configs:
    wg = variant_wg_weekly_expanded[name]
    wv = variant_wv_weekly_expanded[name]
    mask = wg.index >= kst_start
    print(f"[运行] {name}")
    m = run_variant(wg[mask], wv[mask], g_close, v_close, name)
    results[name] = m

# ============================================================
# 9. F2 clip A/B 测试
# ============================================================
print("\n" + "=" * 70)
print("  F2 clip A/B 测试")
print("=" * 70)

clip_results = {}
for clip_val in [CLIP_X2, CLIP_X3]:
    wg, wv = build_variant(
        use_kst=True, use_continuous=True, use_adx=True,
        clip_val=clip_val, g_close_s=g_close, v_close_s=v_close
    )
    df_tmp = pd.DataFrame({'wg': wg, 'wv': wv})
    weekly_tmp = df_tmp.resample('W-FRI').last().dropna(subset=['wg']).iloc[1:]
    wg_exp = expand_to_daily(weekly_tmp['wg'], g_close.index)
    wv_exp = expand_to_daily(weekly_tmp['wv'], g_close.index)
    mask = wg_exp.index >= kst_start
    label = f"clip=±{clip_val}"
    print(f"[运行] {label}")
    m = run_variant(wg_exp[mask], wv_exp[mask], g_close, v_close, label)
    clip_results[clip_val] = m

# 选择更优clip（收益为先：年化高者胜，差<0.5pp看Calmar）
clip_x2_ann = clip_results[CLIP_X2]['ann']
clip_x3_ann = clip_results[CLIP_X3]['ann']
if clip_x3_ann > clip_x2_ann + 0.005:
    best_clip = CLIP_X3
    clip_reason = f"clip=±{CLIP_X3} 年化更高（+{(clip_x3_ann-clip_x2_ann)*100:.2f}pp）"
elif clip_x2_ann > clip_x3_ann + 0.005:
    best_clip = CLIP_X2
    clip_reason = f"clip=±{CLIP_X2} 年化更高（+{(clip_x2_ann-clip_x3_ann)*100:.2f}pp）"
else:
    # 年化接近，看Calmar
    if clip_results[CLIP_X3]['calmar'] >= clip_results[CLIP_X2]['calmar']:
        best_clip = CLIP_X3
        clip_reason = f"年化接近，clip=±{CLIP_X3} Calmar更优"
    else:
        best_clip = CLIP_X2
        clip_reason = f"年化接近，clip=±{CLIP_X2} Calmar更优"
print(f"\n[clip选择] {clip_reason} → best_clip=±{best_clip}")

# ============================================================
# 10. 选择最优X3配置
# ============================================================
# X3-full 已使用 best_clip？如果 best_clip != CLIP_X3，需重算 X3-full
if best_clip != CLIP_X3:
    print(f"\n[更新] X3-full 使用 best_clip=±{best_clip} 重算")
    wg, wv = build_variant(
        use_kst=True, use_continuous=True, use_adx=True,
        clip_val=best_clip, g_close_s=g_close, v_close_s=v_close
    )
    df_tmp = pd.DataFrame({'wg': wg, 'wv': wv})
    weekly_tmp = df_tmp.resample('W-FRI').last().dropna(subset=['wg']).iloc[1:]
    wg_exp = expand_to_daily(weekly_tmp['wg'], g_close.index)
    wv_exp = expand_to_daily(weekly_tmp['wv'], g_close.index)
    mask = wg_exp.index >= kst_start
    m = run_variant(wg_exp[mask], wv_exp[mask], g_close, v_close, "X3-full(更新)")
    results['X3-full'] = m

# 选择最优X3（收益为先：年化最高，若差<0.5pp看Calmar）
print("\n[选择最优X3配置]")
best_name = max(results.keys(), key=lambda k: results[k]['ann'])
best_m = results[best_name]
# 检查是否有年化接近但Calmar更优的
for name, m in results.items():
    if name == best_name:
        continue
    if abs(m['ann'] - best_m['ann']) < 0.005 and m['calmar'] > best_m['calmar']:
        best_name = name
        best_m = m
print(f"  最优X3配置: {best_name}")
print(f"  年化={best_m['ann']*100:.2f}%  回撤={best_m['dd']*100:.2f}%  "
      f"Sharpe={best_m['sharpe']:.3f}  Calmar={best_m['calmar']:.3f}")

# ============================================================
# 11. 输出对比表
# ============================================================
# X1/X2 文档基准（全周期）
x1_doc = {'ann': 0.3629, 'dd': -0.3769, 'sharpe': 1.269, 'calmar': 0.963, 'n_trades': 183}
x2_doc = {'ann': 0.3634, 'dd': -0.3766, 'sharpe': 1.271, 'calmar': 0.965, 'n_trades': 199}

print("\n" + "=" * 76)
print("  X3 策略消融分析（vs X1/X2基准）")
print("=" * 76)
print(f"  {'方案':<24}{'年化':>10}{'回撤':>10}{'Sharpe':>9}{'Calmar':>9}{'调仓次数':>10}")
print("  " + "-" * 72)

# 文档基准（全周期）
print(f"  {'X1基线(文档,全周期)':<24}{x1_doc['ann']*100:>9.2f}%{x1_doc['dd']*100:>9.2f}%"
      f"{x1_doc['sharpe']:>9.3f}{x1_doc['calmar']:>9.3f}{x1_doc['n_trades']:>10}")
print(f"  {'X2基线(文档,全周期)':<24}{x2_doc['ann']*100:>9.2f}%{x2_doc['dd']*100:>9.2f}%"
      f"{x2_doc['sharpe']:>9.3f}{x2_doc['calmar']:>9.3f}{x2_doc['n_trades']:>10}")
print(f"  {'X2基线(引擎复现,全周期)':<24}{m_x2_full['ann']*100:>9.2f}%{m_x2_full['dd']*100:>9.2f}%"
      f"{m_x2_full['sharpe']:>9.3f}{m_x2_full['calmar']:>9.3f}{m_x2_full['n_trades']:>10}")

# 公平对比（KST起始日）
print(f"  {'--- 以下从KST起始日公平对比 ---':<72}")
print(f"  {'X2基线(KST起始)':<24}{m_x2_kst['ann']*100:>9.2f}%{m_x2_kst['dd']*100:>9.2f}%"
      f"{m_x2_kst['sharpe']:>9.3f}{m_x2_kst['calmar']:>9.3f}{m_x2_kst['n_trades']:>10}")
for name in ['X3-full', 'X3-no-KST', 'X3-no-weighting', 'X3-no-ADX', 'X3-no-clip']:
    m = results[name]
    print(f"  {name:<24}{m['ann']*100:>9.2f}%{m['dd']*100:>9.2f}%"
          f"{m['sharpe']:>9.3f}{m['calmar']:>9.3f}{m['n_trades']:>10}")
print("=" * 76)

# clip A/B表
print("\n" + "=" * 60)
print("  F2 clip A/B测试")
print("=" * 60)
print(f"  {'clip值':<12}{'年化':>10}{'回撤':>10}{'Sharpe':>9}{'Calmar':>9}")
print("  " + "-" * 48)
for clip_val in [CLIP_X2, CLIP_X3]:
    m = clip_results[clip_val]
    print(f"  {'±' + str(clip_val):<12}{m['ann']*100:>9.2f}%{m['dd']*100:>9.2f}%"
          f"{m['sharpe']:>9.3f}{m['calmar']:>9.3f}")
print("=" * 60)
print(f"  选择: ±{best_clip}（{clip_reason}）")

# 改进幅度
print("\n" + "=" * 60)
print("  X3-full vs X2基线 改进幅度（KST起始日公平对比）")
print("=" * 60)
x3f = results['X3-full']
x2k = m_x2_kst
d_ann = (x3f['ann'] - x2k['ann']) * 100
d_dd = (x3f['dd'] - x2k['dd']) * 100
d_sharpe = x3f['sharpe'] - x2k['sharpe']
d_calmar = x3f['calmar'] - x2k['calmar']
print(f"  年化:    {x2k['ann']*100:6.2f}% → {x3f['ann']*100:6.2f}%  ({d_ann:+.2f}pp)")
print(f"  回撤:    {x2k['dd']*100:6.2f}% → {x3f['dd']*100:6.2f}%  ({d_dd:+.2f}pp, 正=改善)")
print(f"  Sharpe:  {x2k['sharpe']:6.3f} → {x3f['sharpe']:6.3f}  ({d_sharpe:+.3f})")
print(f"  Calmar:  {x2k['calmar']:6.3f} → {x3f['calmar']:6.3f}  ({d_calmar:+.3f})")
print(f"  调仓:    {x2k['n_trades']:6d} → {x3f['n_trades']:6d}")
print("=" * 60)

# 各组件贡献（相对X3-full的退化）
print("\n" + "=" * 60)
print("  各组件贡献（关闭某组件后相对X3-full的变化）")
print("=" * 60)
print(f"  {'组件':<20}{'年化变化':>10}{'回撤变化':>10}{'Sharpe变化':>12}{'Calmar变化':>12}")
print("  " + "-" * 56)
ablations = [
    ('KST (vs MA20-F1)', 'X3-no-KST'),
    ('连续加权 (vs 二元)', 'X3-no-weighting'),
    ('ADX过滤', 'X3-no-ADX'),
    ('clip优化', 'X3-no-clip'),
]
for label, name in ablations:
    m = results[name]
    d_a = (m['ann'] - x3f['ann']) * 100
    d_d = (m['dd'] - x3f['dd']) * 100
    d_s = m['sharpe'] - x3f['sharpe']
    d_c = m['calmar'] - x3f['calmar']
    print(f"  {label:<20}{d_a:>+9.2f}pp{d_d:>+9.2f}pp{d_s:>+11.3f}{d_c:>+11.3f}")
print("=" * 60)

# ============================================================
# 12. 保存报告
# ============================================================
report_lines = []
def w(line=""):
    report_lines.append(line)

w("=" * 76)
w("  X3 策略消融分析报告")
w("  KST + 连续加权 + ADX过滤 + clip测试")
w("=" * 76)
w()
w(f"数据区间: {g_close.index[0]:%Y-%m-%d} ~ {g_close.index[-1]:%Y-%m-%d}")
w(f"KST有效起始日: {kst_start:%Y-%m-%d}")
w(f"公平对比区间: {kst_start:%Y-%m-%d} ~ {g_close.index[-1]:%Y-%m-%d}")
w(f"  （所有变体从KST起始日回测，确保公平对比）")
w()
w("------------------------------------------------------------------------------")
w("一、策略配置")
w("------------------------------------------------------------------------------")
w()
w("  F1因子（X3）: KST多周期动量")
w("    ratio = (g_close/v_close).shift(1)")
w("    roc10/15/20/30 = ratio.pct_change(10/15/20/30)")
w("    kst = roc10*1 + roc15*2 + roc20*3 + roc30*4")
w("    kst_std = kst.rolling(252).std()")
w("    f1 = tanh(kst / kst_std) * 0.5")
w()
w("  F1因子（X2原版）: 比价MA20方向")
w("    ratio_ma20 = ratio.rolling(20).mean()")
w("    ratio_dev = ratio / ratio_ma20 - 1")
w("    f1 = tanh(ratio_dev * 30) * 0.5")
w()
w("  F2因子: 动量加速度 + ADX过滤")
w("    accel_diff = (g_accel - v_accel).clip(±clip_val)")
w("    f2 = accel_diff * 5.0")
w("    ADX过滤: f2[adx<20] = 0 （震荡市F2置0）")
w(f"    ADX计算: 用close近似（无high/low），Wilder smoothing, period={ADX_PERIOD}")
w()
w("  连续加权:")
w("    style_score > 0.3:  w_g=0.65 (都涨时波动率加成, 上限0.75)")
w("    style_score < -0.3: w_g=0.35")
w("    其他:               w_g=0.5")
w()
w("  降仓: 方案A（MA50→50%, MA75→25%），与X2一致")
w()
w("  回测参数:")
w("    初始资金: 1,000,000")
w("    佣金: 万1（双边）")
w("    冲击滑点: 0（指数无冲击）")
w("    跳空滑点: 关闭（指数 open=close）")
w("    调仓频率: 周频 W-FRI")
w("    Sharpe rf: 2.5%/年")
w()
w("------------------------------------------------------------------------------")
w("二、消融分析对比表")
w("------------------------------------------------------------------------------")
w()
w("=" * 76)
w("  X3 策略消融分析（vs X1/X2基准）")
w("=" * 76)
w(f"  {'方案':<24}{'年化':>10}{'回撤':>10}{'Sharpe':>9}{'Calmar':>9}{'调仓次数':>10}")
w("  " + "-" * 72)
w(f"  {'X1基线(文档,全周期)':<24}{x1_doc['ann']*100:>9.2f}%{x1_doc['dd']*100:>9.2f}%"
  f"{x1_doc['sharpe']:>9.3f}{x1_doc['calmar']:>9.3f}{x1_doc['n_trades']:>10}")
w(f"  {'X2基线(文档,全周期)':<24}{x2_doc['ann']*100:>9.2f}%{x2_doc['dd']*100:>9.2f}%"
  f"{x2_doc['sharpe']:>9.3f}{x2_doc['calmar']:>9.3f}{x2_doc['n_trades']:>10}")
w(f"  {'X2基线(引擎复现,全周期)':<24}{m_x2_full['ann']*100:>9.2f}%{m_x2_full['dd']*100:>9.2f}%"
  f"{m_x2_full['sharpe']:>9.3f}{m_x2_full['calmar']:>9.3f}{m_x2_full['n_trades']:>10}")
w(f"  {'--- 以下从KST起始日公平对比 ---':<72}")
w(f"  {'X2基线(KST起始)':<24}{m_x2_kst['ann']*100:>9.2f}%{m_x2_kst['dd']*100:>9.2f}%"
  f"{m_x2_kst['sharpe']:>9.3f}{m_x2_kst['calmar']:>9.3f}{m_x2_kst['n_trades']:>10}")
for name in ['X3-full', 'X3-no-KST', 'X3-no-weighting', 'X3-no-ADX', 'X3-no-clip']:
    m = results[name]
    w(f"  {name:<24}{m['ann']*100:>9.2f}%{m['dd']*100:>9.2f}%"
      f"{m['sharpe']:>9.3f}{m['calmar']:>9.3f}{m['n_trades']:>10}")
w("=" * 76)
w()
w("------------------------------------------------------------------------------")
w("三、F2 clip A/B测试")
w("------------------------------------------------------------------------------")
w()
w("=" * 60)
w("  F2 clip A/B测试")
w("=" * 60)
w(f"  {'clip值':<12}{'年化':>10}{'回撤':>10}{'Sharpe':>9}{'Calmar':>9}")
w("  " + "-" * 48)
for clip_val in [CLIP_X2, CLIP_X3]:
    m = clip_results[clip_val]
    w(f"  {'±' + str(clip_val):<12}{m['ann']*100:>9.2f}%{m['dd']*100:>9.2f}%"
      f"{m['sharpe']:>9.3f}{m['calmar']:>9.3f}")
w("=" * 60)
w(f"  选择: ±{best_clip}（{clip_reason}）")
w()
w("------------------------------------------------------------------------------")
w("四、各组件贡献（关闭某组件后相对X3-full的变化）")
w("------------------------------------------------------------------------------")
w()
w(f"  {'组件':<20}{'年化变化':>10}{'回撤变化':>10}{'Sharpe变化':>12}{'Calmar变化':>12}")
w("  " + "-" * 56)
for label, name in ablations:
    m = results[name]
    d_a = (m['ann'] - x3f['ann']) * 100
    d_d = (m['dd'] - x3f['dd']) * 100
    d_s = m['sharpe'] - x3f['sharpe']
    d_c = m['calmar'] - x3f['calmar']
    w(f"  {label:<20}{d_a:>+9.2f}pp{d_d:>+9.2f}pp{d_s:>+11.3f}{d_c:>+11.3f}")
w()
w("  正值=关闭该组件后指标变好（即该组件有负贡献）")
w("  负值=关闭该组件后指标变差（即该组件有正贡献）")
w()
w("------------------------------------------------------------------------------")
w("五、最优X3配置选择")
w("------------------------------------------------------------------------------")
w()
w(f"  最优X3配置: {best_name}")
w(f"  clip: ±{best_clip}")
w(f"  年化={best_m['ann']*100:.2f}%  回撤={best_m['dd']*100:.2f}%  "
  f"Sharpe={best_m['sharpe']:.3f}  Calmar={best_m['calmar']:.3f}")
w()
w("------------------------------------------------------------------------------")
w("六、X3-full vs X2基线 改进幅度（KST起始日公平对比）")
w("------------------------------------------------------------------------------")
w()
w(f"  年化:    {x2k['ann']*100:6.2f}% → {x3f['ann']*100:6.2f}%  ({d_ann:+.2f}pp)")
w(f"  回撤:    {x2k['dd']*100:6.2f}% → {x3f['dd']*100:6.2f}%  ({d_dd:+.2f}pp, 正=改善)")
w(f"  Sharpe:  {x2k['sharpe']:6.3f} → {x3f['sharpe']:6.3f}  ({d_sharpe:+.3f})")
w(f"  Calmar:  {x2k['calmar']:6.3f} → {x3f['calmar']:6.3f}  ({d_calmar:+.3f})")
w(f"  调仓:    {x2k['n_trades']:6d} → {x3f['n_trades']:6d}")
w()
w("------------------------------------------------------------------------------")
w("七、X2基线引擎一致性验证")
w("------------------------------------------------------------------------------")
w()
w(f"  X2文档基准(全周期):  年化={x2_doc['ann']*100:.2f}%  回撤={x2_doc['dd']*100:.2f}%  "
  f"Sharpe={x2_doc['sharpe']:.3f}  Calmar={x2_doc['calmar']:.3f}  调仓={x2_doc['n_trades']}")
w(f"  X3引擎复现(全周期):  年化={m_x2_full['ann']*100:.2f}%  回撤={m_x2_full['dd']*100:.2f}%  "
  f"Sharpe={m_x2_full['sharpe']:.3f}  Calmar={m_x2_full['calmar']:.3f}  调仓={m_x2_full['n_trades']}")
w(f"  差异: 年化{(m_x2_full['ann']-x2_doc['ann'])*100:+.2f}pp, "
  f"回撤{(m_x2_full['dd']-x2_doc['dd'])*100:+.2f}pp, "
  f"Sharpe{m_x2_full['sharpe']-x2_doc['sharpe']:+.3f}, "
  f"调仓{m_x2_full['n_trades']-x2_doc['n_trades']:+d}")
w()
w("=" * 76)
w("  报告生成完毕")
w("=" * 76)

with open(REPORT_PATH, 'w', encoding='utf-8') as f:
    f.write("\n".join(report_lines))

print(f"\n[报告已保存] {REPORT_PATH}")
print("=" * 70)
print("[完成] X3 策略集成回测执行结束")
