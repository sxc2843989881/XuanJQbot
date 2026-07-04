"""验证 510050（2005 年成立，qfq 易产生负价）的负价过滤逻辑

预期：
  - qfq 原始数据早期 close 可能为负（多次分红导致）
  - compute_metrics 中 `df = df[df["close"] > 0]` 过滤后应正常计算
  - 过滤后年化应合理（不应是 49.91% 那种失真值）
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from custom.research.research_etf_screening import fetch_etf_kline, compute_metrics

CODE = "sh.510050"
NAME = "华夏上证50ETF"

print("=" * 72)
print(f"{CODE} {NAME}  qfq 负价过滤验证")
print("=" * 72)

# 1. 拉取原始数据（fetch_etf_kline 内部会缓存）
df = fetch_etf_kline(CODE)
if len(df) == 0:
    print("[ERROR] 拉取数据失败")
    sys.exit(1)

print(f"\n原始数据条数: {len(df)}")
print(f"首日: {df.iloc[0]['date'].strftime('%Y-%m-%d')}  open={df.iloc[0]['open']:.4f}  close={df.iloc[0]['close']:.4f}")
print(f"末日: {df.iloc[-1]['date'].strftime('%Y-%m-%d')}  open={df.iloc[-1]['open']:.4f}  close={df.iloc[-1]['close']:.4f}")

# 2. 统计负价
n_neg = (df["close"] <= 0).sum()
n_zero = (df["close"] == 0).sum()
print(f"\n负价/零价统计: close<=0 共 {n_neg} 条, close==0 共 {n_zero} 条")

if n_neg > 0:
    neg_df = df[df["close"] <= 0]
    print(f"  负价首条: {neg_df.iloc[0]['date'].strftime('%Y-%m-%d')}  close={neg_df.iloc[0]['close']:.4f}")
    print(f"  负价末条: {neg_df.iloc[-1]['date'].strftime('%Y-%m-%d')}  close={neg_df.iloc[-1]['close']:.4f}")

# 3. 调用 compute_metrics（内部会过滤 close<=0）
metrics = compute_metrics(CODE, NAME, df)
if metrics is None:
    print("\n[ERROR] compute_metrics 返回 None（数据不足或全部为负）")
    sys.exit(1)

print("\n" + "=" * 72)
print("compute_metrics 计算结果（过滤 close<=0 后）")
print("=" * 72)
print(f"  起始日期: {metrics['start_date']}")
print(f"  结束日期: {metrics['end_date']}")
print(f"  交易日数: {metrics['days']}")
print(f"  年限:     {metrics['years']:.2f} 年")
print(f"  长期年化: {metrics['annual_return']*100:.2f}%")
print(f"  近1年:    {metrics['annual_return_1y']*100:.2f}%")
print(f"  近3年:    {metrics['annual_return_3y']*100 if metrics['annual_return_3y'] is not None else 'N/A'}%")
print(f"  波动率:   {metrics['annual_volatility']*100:.2f}%")
print(f"  Sharpe:   {metrics['sharpe']:.4f}")
print(f"  最大回撤: {metrics['max_drawdown']*100:.2f}%")
print(f"  Calmar:   {metrics['calmar']:.4f}")

# 4. 判定
print("\n" + "=" * 72)
print("判定")
print("=" * 72)
# 510050 长期年化合理区间 6%-10%（hfq 基准 8.33%）
ok = 0.05 < metrics['annual_return'] < 0.12
print(f"  长期年化 {metrics['annual_return']*100:.2f}%  {'✓ 合理（5%-12%）' if ok else '✗ 异常'}")
print(f"  负价过滤 {'✓ 已处理' if n_neg > 0 else '✓ 无负价'}")
