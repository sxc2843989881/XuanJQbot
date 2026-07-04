"""研究阶段：为 ETF 标的池生成合成样本数据集

输入：
  - custom/strategies/etf_rotation/1_research/universe.csv（用户挑选的 20 个标的）
  - custom/data/etf_klines/<code>.csv（每个 ETF 的 hfq 日线缓存）

输出（custom/output/research/synthetic_data/）：
  - <code>_synthetic.csv         每个 ETF 的合成数据（保持原始列结构）
  - <code>_comparison.png        原始 vs 合成 价格走势对比图
  - metrics_comparison.csv       全部 ETF 的指标对比汇总
  - summary_report.html          HTML 汇总报告

技术：
  Moment-Matched Block Bootstrap（矩匹配分块重采样）
  - Block Bootstrap 保留时序短期依赖（波动率聚集、自相关）
  - Moment Matching 强制对数收益率均值和标准差精确等于原始值
  - 结果：年化收益率、波动率、Sharpe 与原始完全一致，仅走势不同

用法：
  python -m custom.research.research_synthetic_etf
  python -m custom.research.research_synthetic_etf --block-length 21 --seed 42
"""
import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from custom.research.synthetic_data_generator import SyntheticDataGenerator

# 路径常量
UNIVERSE_CSV = PROJECT_ROOT / "custom" / "strategies" / "etf_rotation" / "1_research" / "universe.csv"
CACHE_DIR = PROJECT_ROOT / "custom" / "data" / "etf_klines"
OUTPUT_DIR = PROJECT_ROOT / "custom" / "output" / "research" / "synthetic_data"


def main():
    parser = argparse.ArgumentParser(description="为 ETF 标的池生成合成样本数据集")
    parser.add_argument("--block-length", type=int, default=21,
                         help="Block Bootstrap 块长度（默认 21 ≈ 1 个月）")
    parser.add_argument("--seed", type=int, default=42,
                         help="随机种子（便于复现，默认 42）")
    parser.add_argument("--universe", type=str, default=str(UNIVERSE_CSV),
                         help="标的池 CSV 路径")
    parser.add_argument("--cache-dir", type=str, default=str(CACHE_DIR),
                         help="原始数据缓存目录")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR),
                         help="输出目录")
    parser.add_argument("--no-plot", action="store_true",
                         help="不生成对比图")
    parser.add_argument("--no-html", action="store_true",
                         help="不生成 HTML 报告")
    args = parser.parse_args()

    # 用 SyntheticDataGenerator 类一键生成
    gen = SyntheticDataGenerator(block_length=args.block_length, seed=args.seed)
    gen.generate_batch(
        universe_csv=args.universe,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        plot=not args.no_plot,
        html_report=not args.no_html,
    )


if __name__ == "__main__":
    main()
