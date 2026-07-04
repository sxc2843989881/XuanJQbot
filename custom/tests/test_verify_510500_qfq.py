"""验证 510500 ETF 前复权（qfq）数据准确性

用户提供基准：
  - 510500 ETF 2013 年成立
  - 前复权首日开盘价 ≈ 2.759
  - 最新收盘价 ≈ 8.861

验证步骤：
  1. 用 akshare fund_etf_hist_em(adjust="qfq") 拉完整历史
  2. 打印首日/末日 OHLC
  3. 对比基准，计算差异
  4. 计算总收益与年化
"""
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import akshare as ak

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

SYMBOL = "510500"
BENCH_FIRST_OPEN = 2.759   # 用户提供：首日开盘
BENCH_LAST_CLOSE = 8.861   # 用户提供：最新收盘

print("=" * 72)
print(f"510500 ETF 前复权（qfq）数据验证  @ {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print("=" * 72)

# 1. 拉取完整历史（qfq）
try:
    df = ak.fund_etf_hist_em(
        symbol=SYMBOL,
        period="daily",
        start_date="20100101",
        end_date=datetime.now().strftime("%Y%m%d"),
        adjust="qfq",
    )
except Exception as e:
    print(f"[ERROR] akshare 调用失败: {e}")
    sys.exit(1)

if df is None or len(df) == 0:
    print("[ERROR] 返回空数据")
    sys.exit(1)

print(f"\n数据条数: {len(df)}")
print(f"列名: {list(df.columns)}")

# 2. 首日/末日
first = df.iloc[0]
last = df.iloc[-1]

print("\n--- 首 3 条 ---")
print(df.head(3).to_string())
print("\n--- 末 3 条 ---")
print(df.tail(3).to_string())

print("\n" + "-" * 72)
print("首日数据:")
print(f"  日期: {first['日期']}")
print(f"  开盘: {first['开盘']}")
print(f"  收盘: {first['收盘']}")
print(f"  最高: {first['最高']}")
print(f"  最低: {first['最低']}")

print("\n末日数据:")
print(f"  日期: {last['日期']}")
print(f"  开盘: {last['开盘']}")
print(f"  收盘: {last['收盘']}")
print(f"  最高: {last['最高']}")
print(f"  最低: {last['最低']}")

# 3. 基准对比
first_open = float(first['开盘'])
last_close = float(last['收盘'])

print("\n" + "=" * 72)
print("基准对比")
print("=" * 72)
print(f"  首日开盘  实测 {first_open:.4f} | 基准 {BENCH_FIRST_OPEN} | 差异 {first_open - BENCH_FIRST_OPEN:+.4f}")
print(f"  末日收盘  实测 {last_close:.4f} | 基准 {BENCH_LAST_CLOSE} | 差异 {last_close - BENCH_LAST_CLOSE:+.4f}")

# 4. 总收益与年化
total_ret = last_close / first_open - 1
days = (pd.to_datetime(last['日期']) - pd.to_datetime(first['日期'])).days
years = days / 365.25
annual = (1 + total_ret) ** (1 / years) - 1 if years > 0 else 0

print("\n" + "=" * 72)
print("收益计算（基于首日开盘 → 末日收盘）")
print("=" * 72)
print(f"  总收益:  {total_ret*100:.2f}%")
print(f"  持有:    {days} 天 = {years:.2f} 年")
print(f"  年化:    {annual*100:.2f}%")

# 5. 判定
print("\n" + "=" * 72)
print("判定")
print("=" * 72)
tol = 0.01  # 1% 容差
ok_open = abs(first_open - BENCH_FIRST_OPEN) / BENCH_FIRST_OPEN < tol
ok_close = abs(last_close - BENCH_LAST_CLOSE) / BENCH_LAST_CLOSE < tol
print(f"  首日开盘 {'✓ 一致' if ok_open else '✗ 不一致'}")
print(f"  末日收盘 {'✓ 一致' if ok_close else '✗ 不一致'}")

if ok_open and ok_close:
    print("\n  [结论] akshare qfq 数据与用户基准一致，数据源可用。")
else:
    print("\n  [结论] 数据与基准不一致，需排查数据源或 API 调用方式。")
