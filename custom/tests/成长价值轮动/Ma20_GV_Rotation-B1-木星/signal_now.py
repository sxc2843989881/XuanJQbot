"""输出当前最新信号 — 增量更新ETF行情"""
import sys, importlib.util as _util
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

# ====== 引擎 ======
_spec = _util.spec_from_file_location('backtest_engine', r'c:\XuanJLH\Qbot\engines\lightweight\engine.py')
_bte = _util.module_from_spec(_spec)
sys.modules['backtest_engine'] = _bte
_spec.loader.exec_module(_bte)

# ====== 数据缓存路径 ======
DATA_DIR = Path(r'c:\XuanJLH\Qbot\custom\data\etf_cache')
DATA_DIR.mkdir(exist_ok=True)
CACHE_FILES = {
    'growth': DATA_DIR / '159259.csv',
    'value': DATA_DIR / '159263.csv',
}

def fetch_incremental(code, name, cache_file, full_days=400):
    """增量更新：本地有缓存则只拉缺失天数，否则拉全量"""
    import baostock as bs
    
    # 读本地缓存
    if cache_file.exists():
        local = pd.read_csv(cache_file, parse_dates=['date'], index_col='date')
        local = local.sort_index()
        # 兼容旧格式（只有close列），回填open=close
        if 'open' not in local.columns:
            local['open'] = local['close']
            local.to_csv(cache_file)
        latest_local = local.index[-1]
    else:
        local = None
        latest_local = None
    
    # 拉取数据
    bs.login()
    bs_code = f'sz.{code}'
    end = datetime.now().strftime('%Y-%m-%d')
    
    if local is not None:
        # 增量：从最新本地日期的下一天开始
        start = (latest_local + timedelta(days=1)).strftime('%Y-%m-%d')
        if start >= end:
            bs.logout()
            print(f"  ✅ {name}({code}): 已是最新 ({latest_local.date()})")
            return local
        print(f"  🔄 {name}({code}): 本地{latest_local.date()}，增量拉取{start}~{end}")
        rs = bs.query_history_k_data_plus(bs_code,
            'date,open,high,low,close,volume,amount',
            start_date=start, end_date=end,
            frequency='d', adjustflag='2')
    else:
        # 全量
        start = (datetime.now() - timedelta(days=full_days)).strftime('%Y-%m-%d')
        print(f"  📥 {name}({code}): 首次拉取{start}~{end}")
        rs = bs.query_history_k_data_plus(bs_code,
            'date,open,high,low,close,volume,amount',
            start_date=start, end_date=end,
            frequency='d', adjustflag='2')
    
    rows = []
    while rs.next():
        rows.append(rs.get_row_data())
    bs.logout()
    
    if not rows:
        print(f"  ⚠️ {name}({code}): 无新数据")
        return local
    
    new_df = pd.DataFrame(rows, columns=['date','open','high','low','close','volume','amount'])
    new_df = new_df[new_df['close'] != '']
    for col in ['open','close']:
        new_df[col] = new_df[col].astype(float)
    new_df['date'] = pd.to_datetime(new_df['date'])
    new_df = new_df.set_index('date')[['open','close']].sort_index()
    
    # 合并
    if local is not None:
        merged = pd.concat([local, new_df])
        merged = merged[~merged.index.duplicated(keep='last')].sort_index()
    else:
        merged = new_df
    
    # 写回缓存
    merged.to_csv(cache_file)
    print(f"  ✅ {name}({code}): 已更新至{merged.index[-1].date()}，共{len(merged)}条")
    return merged

# ====== 获取数据 ======
IDX_DIR = Path(r'c:\temp_v72_data')

def load_index_data():
    """加载指数数据"""
    g = pd.read_csv(IDX_DIR / 'index_480080.csv').set_index('date')['close'].astype(float).sort_index().dropna()
    v = pd.read_csv(IDX_DIR / 'index_480081.csv').set_index('date')['close'].astype(float).sort_index().dropna()
    g.index = pd.to_datetime(g.index); v.index = pd.to_datetime(v.index)
    idx = g.index.intersection(v.index)
    return g[idx], v[idx]

def merge_idx_etf(idx_close, etf_data, name=""):
    """拼接指数+ETF"""
    if isinstance(etf_data, pd.DataFrame):
        etf_close = etf_data['close']
    else:
        etf_close = etf_data
    
    etf_close = etf_close.sort_index()
    merge_start = etf_close.index[0]
    # 确保索引类型一致
    if not isinstance(merge_start, pd.Timestamp):
        idx_close.index = pd.to_datetime(idx_close.index)
        etf_close.index = pd.to_datetime(etf_close.index)
        merge_start = etf_close.index[0]
    
    if merge_start not in idx_close.index:
        print(f"  ⚠️ {name}: ETF起点{merge_start.date()}不在指数范围内")
        return etf_close
    scale = float(idx_close.loc[merge_start]) / float(etf_close.loc[merge_start])
    before = idx_close[idx_close.index < merge_start]
    # 使用全部ETF数据（不截断到指数范围）
    etf_scaled = etf_close * scale
    merged = pd.concat([before, etf_scaled]).sort_index()
    # 去重（拼接点可能两边都有）
    merged = merged[~merged.index.duplicated(keep='last')]
    return merged

print("检查/更新行情数据...")
g_etf_df = fetch_incremental('159259', '易方达成长ETF', CACHE_FILES['growth'])
v_etf_df = fetch_incremental('159263', '易方达价值ETF', CACHE_FILES['value'])

print("加载指数数据 + 拼接...")
g_idx, v_idx = load_index_data()
g_close = merge_idx_etf(g_idx, g_etf_df, '成长')
v_close = merge_idx_etf(v_idx, v_etf_df, '价值')
# 用并集而非交集：ETF有数据的日子就优先用ETF，没有就用指数
all_dates = g_close.index.union(v_close.index).sort_values()
g_close = g_close.reindex(all_dates).ffill()
v_close = v_close.reindex(all_dates).ffill()
print(f"  拼接后: {g_close.index[0].date()} ~ {g_close.index[-1].date()} ({len(g_close)}天)")

# ====== 注入全局变量 ======
sys.path.extend([
    r'c:\XuanJLH\Qbot\custom\research\成长价值轮动\X12_偏离度空仓版',
    r'c:\XuanJLH\Qbot\custom\research\成长价值轮动\X14_统一策略干净版',
])
import optimize_runner as orun
import backtest_x14_engine as bxe

orun.G_CLOSE = g_close
orun.V_CLOSE = v_close
orun.RATIO = g_close / v_close
bxe.RATIO = g_close / v_close

from backtest_x14_engine import set_ma_period, build_core

set_ma_period(20)
sig, wt = build_core(
    bias_mode='clear', dcd=6, dc=5,
    bias_t_constraint=False, e5_reset=True,
)

# ====== 输出 ======
latest_date = sig.index[-1]
latest_sig = sig.iloc[-1]
latest_wt = wt.iloc[-1]

print("\n" + "=" * 60)
print(f"  X14 策略 — 当前信号")
print(f"  数据截至: {latest_date.date()}")
print("=" * 60)

pos = "成长(159259)" if latest_sig == 'growth' else "价值(159263)"
wt_desc = f"{latest_wt*100:.0f}%" if latest_wt > 0 else "空仓"

print(f"\n  持仓方向: {pos}")
print(f"  仓位: {wt_desc}")

ratio = g_close / v_close
ratio_ma20 = ratio.rolling(20).mean().iloc[-1]
ratio_dev = (ratio.iloc[-1] / ratio_ma20 - 1) * 100
print(f"  成长/价值比价: {ratio.iloc[-1]:.4f}")
print(f"  比价偏离MA20: {ratio_dev:+.2f}%")
print(f"  方向: {'成长跑赢' if ratio.iloc[-1] > ratio_ma20 else '价值跑赢'}")

T = bxe.T
SLOPE = bxe.SLOPE
print(f"\n  各层信号:")
print(f"    T值(z-score): {T.iloc[-1]:+.3f}  → {'BULL(看多成长)' if T.iloc[-1] > 0 else 'BEAR(看多价值)'}")
print(f"    SLOPE(斜率): {SLOPE.iloc[-1]:+.6f}")

print(f"\n  建议:")
if latest_wt == 0:
    print(f"    空仓观望 — 多空信号不明确")
elif latest_sig == 'growth':
    print(f"    买入/持有 成长ETF(159259)")
else:
    print(f"    买入/持有 价值ETF(159263)")

n = min(20, len(sig))
recent = sig.tail(n)
gd = (recent == 'growth').sum()
vd = (recent == 'value').sum()
print(f"\n  最近{n}日: 成长{gd}天 / 价值{vd}天")
print("=" * 60)
