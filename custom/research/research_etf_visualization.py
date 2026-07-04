"""研究阶段：ETF 标的池可视化与统计分析

输入：
  - custom/strategies/etf_rotation/1_research/universe.csv（已选定的 19 个 ETF）
  - custom/data/etf_klines/<code>.csv（K线缓存，由 research_etf_screening.py 生成）

输出（custom/output/research/etf_viz_<YYYYMMDD>/）：
  - etf_stats.csv                 指标统计表
  - etf_groups.csv                分组建议表
  - 01_nav_trend.png              净值走势图（归一化对比）
  - 02_correlation.png            相关性热力图
  - 03_risk_return.png            风险-收益散点图
  - 04_drawdown_compare.png       最大回撤对比柱状图
  - 05_annual_returns.png         长期/近1年/近3年年化收益对比
  - 06_group_performance.png      按分组对比箱线图
  - report.html                   汇总 HTML 报告

分组依据：参考玄机量化知识库 01_ETF轮动策略/01_策略基础原理.md
- 资产轮动：股票/债券/商品/海外
- 行业轮动：周期/科技成长/防御
- 风格轮动：大盘/小盘/价值/成长
"""
import os
import sys
import math
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# 路径常量
UNIVERSE_CSV = PROJECT_ROOT / "custom" / "strategies" / "etf_rotation" / "1_research" / "universe.csv"
KLINE_CACHE_DIR = PROJECT_ROOT / "custom" / "data" / "etf_klines"
OUTPUT_DIR = PROJECT_ROOT / "custom" / "output" / "research"

# 无风险利率（Sharpe 计算用）
RISK_FREE_RATE = 0.02


# ============================================================
# 1. 数据加载
# ============================================================

def code_to_cache_name(code: str) -> str:
    """'sh.518880' → 'sh_518880'"""
    return code.replace(".", "_")


def load_universe() -> pd.DataFrame:
    """读取标的池 CSV"""
    if not UNIVERSE_CSV.exists():
        raise FileNotFoundError(f"未找到 universe.csv: {UNIVERSE_CSV}")
    df = pd.read_csv(UNIVERSE_CSV)
    print(f"[加载] universe.csv: {len(df)} 个 ETF")
    return df


def load_kline(code: str) -> pd.DataFrame:
    """从缓存加载 ETF 日线数据

    Returns:
        DataFrame[date, open, high, low, close, volume, amount]
    """
    cache_path = KLINE_CACHE_DIR / f"{code_to_cache_name(code)}.csv"
    if not cache_path.exists():
        raise FileNotFoundError(f"缺少缓存: {cache_path}")
    df = pd.read_csv(cache_path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


# ============================================================
# 2. 指标计算
# ============================================================

def compute_metrics(code: str, name: str, df: pd.DataFrame) -> dict:
    """计算单只 ETF 的统计指标"""
    # 剔除异常
    df = df[df["close"] > 0].reset_index(drop=True)
    if len(df) < 60:
        return None

    close = df["close"].values
    dates = df["date"].dt.strftime("%Y-%m-%d").values
    amount = df["amount"].values

    # 日收益率
    daily_ret = np.diff(np.log(close))
    n_days = len(daily_ret)
    years = n_days / 252

    # 长期年化（成立至今）
    total_return = close[-1] / close[0] - 1
    annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

    # 近1年年化
    if n_days >= 252:
        ret_1y = close[-1] / close[-253] - 1
        annual_return_1y = ret_1y  # 1年总收益即年化
    else:
        ret_1y = total_return
        annual_return_1y = (1 + ret_1y) ** (1 / years) - 1 if years > 0 else 0

    # 近3年年化
    if n_days >= 756:
        ret_3y = close[-1] / close[-757] - 1
        annual_return_3y = (1 + ret_3y) ** (1 / 3) - 1
    else:
        annual_return_3y = None

    # 年化波动率
    annual_vol = float(np.std(daily_ret, ddof=1) * math.sqrt(252))

    # Sharpe
    sharpe = (annual_return - RISK_FREE_RATE) / annual_vol if annual_vol > 0 else 0

    # 最大回撤（完整周期）
    running_max = np.maximum.accumulate(close)
    drawdown = (close - running_max) / running_max
    max_drawdown = float(np.clip(drawdown.min(), -1.0, 0.0))

    # Calmar
    calmar = annual_return / abs(max_drawdown) if max_drawdown < 0 else 0

    # 日均成交额（近60日）
    avg_turnover = float(np.mean(amount[-60:])) if len(amount) >= 60 else float(np.mean(amount))

    # 最大回撤起止日期
    dd_end_idx = int(np.argmin(drawdown))
    dd_start_idx = int(np.argmax(close[: dd_end_idx + 1])) if dd_end_idx > 0 else 0

    return {
        "code": code,
        "name": name,
        "days": n_days,
        "start_date": dates[0],
        "end_date": dates[-1],
        "years": round(years, 2),
        "annual_return": float(annual_return),
        "annual_return_1y": float(annual_return_1y),
        "annual_return_3y": float(annual_return_3y) if annual_return_3y is not None else None,
        "annual_volatility": annual_vol,
        "sharpe": float(sharpe),
        "max_drawdown": max_drawdown,
        "calmar": float(calmar),
        "avg_turnover": avg_turnover,
        "dd_start": dates[dd_start_idx],
        "dd_end": dates[dd_end_idx],
    }


# ============================================================
# 3. ETF 分组（参考玄机量化知识库）
# ============================================================

def classify_etf_three_dim(name: str) -> dict:
    """根据 ETF 名称进行三维分组

    参考：c:\\XuanJLH\\玄机量化\\01_ETF轮动策略\\01_策略基础原理.md
    - 资产轮动：股票/债券/商品/海外
    - 行业轮动：周期/科技成长/防御/金融
    - 风格轮动：大盘/小盘/价值/成长/均衡

    Returns:
        dict: {asset_class, industry, style}
    """
    # ===== 1. 资产大类 =====
    # 注意特殊情形：
    #   - "标普中国A股" 是 A 股指数（标普公司编制），不是海外资产
    #   - "稀土产业ETF"、"有色金属ETF" 是行业股 ETF，归股票
    #   - "黄金ETF"（不含"产业"）是实物商品 ETF
    #   - "港股通" 标的归海外（港股）
    # 优先判断"标普中国A股"，避免被"标普"关键字误判为海外
    if "标普中国" in name or "标普A股" in name:
        asset_class = "股票"
    elif any(kw in name for kw in ["黄金", "白银", "原油", "石油", "商品", "豆粕"]):
        if "黄金" in name and "产业" not in name:
            asset_class = "大宗商品"
        else:
            asset_class = "大宗商品"
    elif any(kw in name for kw in ["稀土", "有色"]):
        asset_class = "股票"  # 行业股 ETF，归股票
    elif any(kw in name for kw in ["债", "可转债", "国债"]):
        asset_class = "债券"
    elif any(kw in name for kw in ["纳指", "纳斯达克", "中韩", "港股", "港股通", "恒生", "日经", "QDII"]):
        asset_class = "海外"
    elif "标普" in name and "中国" not in name and "A股" not in name:
        # 真正的海外标普（如标普500）
        asset_class = "海外"
    else:
        asset_class = "股票"

    # ===== 2. 行业（仅对股票类细分）=====
    if asset_class == "股票":
        if any(kw in name for kw in ["半导体", "芯片", "通信", "科技", "科创", "互联网", "人工智能", "电子", "软件", "信息技术", "数字"]):
            industry = "科技成长"
        elif any(kw in name for kw in ["医药", "医疗", "创新药", "生物"]):
            industry = "医药防御"
        elif any(kw in name for kw in ["有色", "稀土", "化工", "钢铁", "煤炭", "机械", "地产"]):
            industry = "周期"
        elif any(kw in name for kw in ["消费", "食品", "白酒", "家电"]):
            industry = "消费防御"
        elif any(kw in name for kw in ["金融", "证券", "银行", "保险"]):
            industry = "金融"
        else:
            industry = "宽基"
    elif asset_class == "海外":
        if any(kw in name for kw in ["创新药", "医药", "医疗"]):
            industry = "海外医药"
        elif any(kw in name for kw in ["半导体", "芯片", "科技"]):
            industry = "海外科技"
        elif any(kw in name for kw in ["互联网"]):
            industry = "海外互联网"
        else:
            industry = "海外宽基"
    else:
        industry = "-"  # 商品/债券不细分行业

    # ===== 3. 风格 =====
    if asset_class == "股票":
        if any(kw in name for kw in ["红利", "低波", "价值", "基本面"]):
            style = "价值"
        elif any(kw in name for kw in ["沪深300", "上证50", "上证180", "A50", "大盘"]):
            style = "大盘"
        elif any(kw in name for kw in ["中证500", "中证1000", "中证2000", "国证2000", "创业板", "科创"]):
            style = "小盘成长"
        elif any(kw in name for kw in ["科创创业50", "双创"]):
            style = "成长"
        elif any(kw in name for kw in ["消费", "医药", "半导体", "芯片", "通信", "新能源", "互联网"]):
            style = "成长"
        else:
            style = "均衡"
    elif asset_class == "海外":
        if any(kw in name for kw in ["科技", "半导体", "芯片"]):
            style = "海外成长"
        else:
            style = "海外均衡"
    else:
        style = "-"

    return {"asset_class": asset_class, "industry": industry, "style": style}


# ============================================================
# 4. 可视化
# ============================================================

def setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def plot_nav_trend(plt, kline_dict: dict, universe_df: pd.DataFrame, output_path: Path):
    """01 净值走势图（归一化）"""
    fig, ax = plt.subplots(figsize=(14, 7))

    # 用近3年数据作图，避免早期 ETF 拉长 x 轴
    for code, df in kline_dict.items():
        name = universe_df.loc[universe_df["code"] == code, "code_name"].values[0]
        # 取近3年（756交易日）
        df_recent = df.tail(756).copy()
        if len(df_recent) < 60:
            continue
        nav = df_recent["close"] / df_recent["close"].iloc[0]
        ax.plot(df_recent["date"].values, nav.values, label=name, linewidth=1.2, alpha=0.85)

    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_title("ETF 标的池净值走势（近3年，归一化=1.0）", fontsize=13)
    ax.set_xlabel("日期")
    ax.set_ylabel("归一化净值")
    ax.legend(loc="upper left", fontsize=8, ncol=2, framealpha=0.8)
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_correlation(plt, kline_dict: dict, universe_df: pd.DataFrame, output_path: Path):
    """02 相关性热力图"""
    # 构建收盘价 DataFrame（取所有 ETF 共同时间段）
    close_dict = {}
    for code, df in kline_dict.items():
        name = universe_df.loc[universe_df["code"] == code, "code_name"].values[0]
        s = df.set_index("date")["close"].pct_change().dropna()
        close_dict[name] = s

    ret_df = pd.DataFrame(close_dict)
    # 只保留至少有 100 个共同交易日的列
    valid_cols = [c for c in ret_df.columns if ret_df[c].notna().sum() >= 100]
    ret_df = ret_df[valid_cols]
    corr = ret_df.corr()

    fig, ax = plt.subplots(figsize=(13, 11))
    im = ax.imshow(corr.values, cmap="RdYlGn_r", vmin=-1, vmax=1, aspect="auto")

    # 标签
    labels = [c[:10] for c in corr.columns]  # 截断名称
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)

    # 数值标注
    for i in range(len(corr)):
        for j in range(len(corr)):
            val = corr.values[i, j]
            color = "white" if abs(val) > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6, color=color)

    ax.set_title("ETF 标的池相关性矩阵（日收益率）", fontsize=13)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="相关系数")
    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_risk_return(plt, stats_df: pd.DataFrame, output_path: Path):
    """03 风险-收益散点图（年化波动率 vs 年化收益）"""
    fig, ax = plt.subplots(figsize=(12, 8))

    # 按 asset_class 着色
    colors = {
        "股票": "tab:blue",
        "债券": "tab:green",
        "大宗商品": "tab:orange",
        "海外": "tab:red",
    }
    for asset_class, group in stats_df.groupby("asset_class"):
        ax.scatter(
            group["annual_volatility"] * 100,
            group["annual_return"] * 100,
            s=120,
            alpha=0.75,
            label=asset_class,
            color=colors.get(asset_class, "gray"),
            edgecolors="black",
            linewidths=0.8,
        )

    # 标注 ETF 名称
    for _, row in stats_df.iterrows():
        ax.annotate(
            row["name"][:8],
            xy=(row["annual_volatility"] * 100, row["annual_return"] * 100),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=7,
        )

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("年化波动率 (%)", fontsize=11)
    ax.set_ylabel("年化收益率 (%)", fontsize=11)
    ax.set_title("风险-收益散点图（长期年化）", fontsize=13)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_drawdown_compare(plt, stats_df: pd.DataFrame, output_path: Path):
    """04 最大回撤对比柱状图"""
    df_sorted = stats_df.sort_values("max_drawdown").copy()  # 从最负到最不负

    fig, ax = plt.subplots(figsize=(12, 7))
    colors = ["#d62728" if v < -0.3 else "#ff7f0e" if v < -0.15 else "#2ca02c" for v in df_sorted["max_drawdown"]]
    bars = ax.barh(
        range(len(df_sorted)),
        df_sorted["max_drawdown"].values * 100,
        color=colors,
        edgecolor="black",
        linewidth=0.5,
    )
    ax.set_yticks(range(len(df_sorted)))
    ax.set_yticklabels([n[:12] for n in df_sorted["name"]], fontsize=9)
    ax.set_xlabel("最大回撤 (%)", fontsize=11)
    ax.set_title("ETF 标的池最大回撤对比（完整周期）", fontsize=13)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.grid(True, axis="x", alpha=0.3)

    # 数值标注
    for i, (bar, v) in enumerate(zip(bars, df_sorted["max_drawdown"].values * 100)):
        ax.text(v - 1, i, f"{v:.1f}%", va="center", ha="right", fontsize=8, color="white" if abs(v) > 15 else "black")

    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_annual_returns_compare(plt, stats_df: pd.DataFrame, output_path: Path):
    """05 长期/近1年/近3年年化收益对比"""
    df = stats_df.sort_values("annual_return", ascending=False).copy()

    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(df))
    width = 0.27

    # 长期、近1年、近3年
    ax.bar(x - width, df["annual_return"] * 100, width, label="长期年化", color="steelblue")
    ax.bar(x, df["annual_return_1y"] * 100, width, label="近1年", color="orange")
    # 近3年可能为 None
    three_y = df["annual_return_3y"].fillna(0) * 100
    ax.bar(x + width, three_y, width, label="近3年(缺数据=0)", color="green")

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([n[:8] for n in df["name"]], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("年化收益率 (%)", fontsize=11)
    ax.set_title("ETF 年化收益率对比（长期/近1年/近3年）", fontsize=13)
    ax.legend(loc="best")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_group_performance(plt, stats_df: pd.DataFrame, output_path: Path):
    """06 按分组的年化收益箱线图（asset_class + style）"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # 子图1: 按资产大类
    groups = []
    labels = []
    for ac, g in stats_df.groupby("asset_class"):
        groups.append(g["annual_return"].values * 100)
        labels.append(f"{ac}\n(n={len(g)})")
    axes[0].boxplot(groups, labels=labels, showmeans=True)
    axes[0].set_title("按资产大类 - 年化收益率分布", fontsize=12)
    axes[0].set_ylabel("年化收益率 (%)")
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(0, color="black", linewidth=0.8)

    # 子图2: 按风格
    groups2 = []
    labels2 = []
    for st, g in stats_df.groupby("style"):
        if st == "-":
            continue
        groups2.append(g["annual_return"].values * 100)
        labels2.append(f"{st}\n(n={len(g)})")
    if groups2:
        axes[1].boxplot(groups2, labels=labels2, showmeans=True)
    axes[1].set_title("按风格 - 年化收益率分布", fontsize=12)
    axes[1].set_ylabel("年化收益率 (%)")
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(0, color="black", linewidth=0.8)

    fig.suptitle("ETF 分组绩效对比", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 5. HTML 报告
# ============================================================

def generate_html_report(
    stats_df: pd.DataFrame,
    groups_df: pd.DataFrame,
    today: str,
    output_dir: Path,
):
    """生成 HTML 汇总报告"""
    # 格式化数值列
    stats_fmt = stats_df.copy()
    for col in ["annual_return", "annual_return_1y", "annual_return_3y", "annual_volatility", "max_drawdown"]:
        stats_fmt[col] = (stats_fmt[col] * 100).round(2).astype(str) + "%"
    stats_fmt["sharpe"] = stats_fmt["sharpe"].round(3)
    stats_fmt["calmar"] = stats_fmt["calmar"].round(3)
    stats_fmt["avg_turnover"] = (stats_fmt["avg_turnover"] / 1e4).round(0).astype(int).astype(str) + "万"

    # 列重命名为中文
    stats_fmt = stats_fmt.rename(columns={
        "code": "代码", "name": "名称", "asset_class": "资产大类", "industry": "行业", "style": "风格",
        "years": "年数", "annual_return": "长期年化", "annual_return_1y": "近1年年化",
        "annual_return_3y": "近3年年化", "annual_volatility": "年化波动率",
        "sharpe": "Sharpe", "max_drawdown": "最大回撤", "calmar": "Calmar",
        "avg_turnover": "日均成交额", "start_date": "数据起始", "end_date": "数据截止",
        "dd_start": "回撤起点", "dd_end": "回撤终点",
    })
    cols_order = ["代码", "名称", "资产大类", "行业", "风格", "年数", "长期年化", "近1年年化",
                  "近3年年化", "年化波动率", "Sharpe", "最大回撤", "Calmar", "日均成交额",
                  "数据起始", "数据截止", "回撤起点", "回撤终点"]
    stats_fmt = stats_fmt[cols_order]

    # 分组汇总表
    group_summary = (
        stats_df.groupby(["asset_class", "industry", "style"])
        .agg(
            数量=("code", "count"),
            长期年化均值=("annual_return", "mean"),
            波动率均值=("annual_volatility", "mean"),
            Sharpe均值=("sharpe", "mean"),
            最大回撤均值=("max_drawdown", "mean"),
        )
        .reset_index()
    )
    for col in ["长期年化均值", "波动率均值", "最大回撤均值"]:
        group_summary[col] = (group_summary[col] * 100).round(2).astype(str) + "%"
    group_summary["Sharpe均值"] = group_summary["Sharpe均值"].round(3)

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>ETF 标的池可视化研究报告 - {today}</title>
<style>
body {{ font-family: "Microsoft YaHei", sans-serif; margin: 20px; background: #f5f5f5; }}
h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
h2 {{ color: #34495e; margin-top: 30px; }}
table {{ border-collapse: collapse; width: 100%; background: white; margin: 10px 0; font-size: 12px; }}
th, td {{ border: 1px solid #ddd; padding: 6px 8px; text-align: center; }}
th {{ background: #3498db; color: white; }}
tr:nth-child(even) {{ background: #f9f9f9; }}
tr:hover {{ background: #e3f2fd; }}
img {{ max-width: 100%; border: 1px solid #ddd; margin: 10px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
.summary-box {{ background: white; padding: 15px; border-left: 4px solid #3498db; margin: 15px 0; }}
.note {{ background: #fff9c4; padding: 10px; border-radius: 4px; margin: 10px 0; }}
</style>
</head>
<body>
<h1>ETF 标的池可视化研究报告</h1>
<p>生成日期: {today}</p>

<div class="summary-box">
<b>说明</b>：本报告对 ETF 轮动策略标的池（universe.csv 中的 19 个 ETF）进行统计与可视化分析，
辅助用户观察并对标的进行分组。分组依据参考玄机量化知识库《ETF 轮动策略基础原理》中的三大轮动类型：
<b>资产轮动</b>（股票/债券/商品/海外）、<b>行业轮动</b>（周期/科技成长/防御）、<b>风格轮动</b>（大盘/小盘/价值/成长）。
</div>

<div class="note">
<b>用户提示</b>：以下分组建议仅供参考，您可以基于图表观察自行调整分组。
特别关注相关性热力图（图2）与风险-收益散点图（图3），它们能揭示 ETF 间的真实关联与风险收益特征。
</div>

<h2>一、分组汇总</h2>
{group_summary.to_html(index=False, escape=False)}

<h2>二、ETF 指标统计明细</h2>
{stats_fmt.to_html(index=False, escape=False)}

<h2>三、图表分析</h2>

<h3>图1: 净值走势（近3年归一化）</h3>
<img src="01_nav_trend.png" alt="净值走势">
<p>说明：归一化到 1.0 便于横向比较，仅展示近3年以避免新 ETF 被老 ETF 拉长 x 轴。</p>

<h3>图2: 相关性热力图</h3>
<img src="02_correlation.png" alt="相关性热力图">
<p>说明：基于日收益率计算 Pearson 相关系数。颜色越红表示相关性越高，可用于识别同质化标的。</p>

<h3>图3: 风险-收益散点图</h3>
<img src="03_risk_return.png" alt="风险-收益散点图">
<p>说明：横轴为年化波动率（风险），纵轴为年化收益率。左上角为"高收益低风险"优质标的，右下角为"低收益高风险"劣质标的。</p>

<h3>图4: 最大回撤对比</h3>
<img src="04_drawdown_compare.png" alt="最大回撤对比">
<p>说明：完整周期最大回撤。红色（&lt;-30%）为高风险，橙色（-15%~-30%）为中风险，绿色（&gt;-15%）为低风险。</p>

<h3>图5: 年化收益对比</h3>
<img src="05_annual_returns.png" alt="年化收益对比">
<p>说明：蓝色=长期年化，橙色=近1年，绿色=近3年。注意新 ETF 无近3年数据（显示为0）。</p>

<h3>图6: 分组绩效对比</h3>
<img src="06_group_performance.png" alt="分组绩效对比">
<p>说明：左图按资产大类分组，右图按风格分组，展示各组年化收益率分布（箱线图）。</p>

<h2>四、分组建议</h2>
<p>基于量化知识库的 ETF 轮动策略分类方法，初步分组建议如下：</p>
{groups_df.to_html(index=False, escape=False)}

<div class="note">
<b>下一步</b>：用户观察图表后，可对分组进行调整。确定分组后，进入下一阶段：
每组内选择强弱信号因子（动量/RSRS/RRG），设计轮动规则。
</div>

</body>
</html>
"""
    output_path = output_dir / "report.html"
    output_path.write_text(html, encoding="utf-8")
    return output_path


# ============================================================
# 6. 主流程
# ============================================================

def main():
    today = datetime.now().strftime("%Y%m%d")
    output_dir = OUTPUT_DIR / f"etf_viz_{today}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 加载标的池
    universe_df = load_universe()
    print(f"[1/5] 加载标的池: {len(universe_df)} 个 ETF")

    # 2. 加载K线 + 计算指标
    print(f"[2/5] 加载K线缓存 + 计算指标...")
    kline_dict = {}
    metrics_list = []
    for _, row in universe_df.iterrows():
        code = row["code"]
        name = row["code_name"]
        try:
            df = load_kline(code)
            kline_dict[code] = df
            m = compute_metrics(code, name, df)
            if m is None:
                print(f"  [SKIP] {code} {name}: 数据不足")
                continue
            metrics_list.append(m)
            print(f"  [OK] {code} {name}: {m['years']}年, 年化={m['annual_return']*100:.2f}%, 回撤={m['max_drawdown']*100:.2f}%")
        except Exception as e:
            print(f"  [FAIL] {code} {name}: {e}")

    if not metrics_list:
        print("[FAIL] 无有效数据")
        return

    stats_df = pd.DataFrame(metrics_list)

    # 3. 分组
    print(f"[3/5] 分组分类...")
    classify_results = stats_df["name"].apply(classify_etf_three_dim)
    stats_df["asset_class"] = [r["asset_class"] for r in classify_results]
    stats_df["industry"] = [r["industry"] for r in classify_results]
    stats_df["style"] = [r["style"] for r in classify_results]

    # 分组统计
    print("\n=== 资产大类分布 ===")
    for ac, g in stats_df.groupby("asset_class"):
        print(f"  {ac}: {len(g)} 个 - 年化均值={g['annual_return'].mean()*100:.2f}%")

    print("\n=== 行业分布 ===")
    for ind, g in stats_df.groupby("industry"):
        print(f"  {ind}: {len(g)} 个 - 年化均值={g['annual_return'].mean()*100:.2f}%")

    print("\n=== 风格分布 ===")
    for st, g in stats_df.groupby("style"):
        if st == "-":
            continue
        print(f"  {st}: {len(g)} 个 - 年化均值={g['annual_return'].mean()*100:.2f}%")

    # 4. 生成图表
    print(f"\n[4/5] 生成图表...")
    plt = setup_matplotlib()

    plot_nav_trend(plt, kline_dict, universe_df, output_dir / "01_nav_trend.png")
    print("  [OK] 01_nav_trend.png")

    plot_correlation(plt, kline_dict, universe_df, output_dir / "02_correlation.png")
    print("  [OK] 02_correlation.png")

    plot_risk_return(plt, stats_df, output_dir / "03_risk_return.png")
    print("  [OK] 03_risk_return.png")

    plot_drawdown_compare(plt, stats_df, output_dir / "04_drawdown_compare.png")
    print("  [OK] 04_drawdown_compare.png")

    plot_annual_returns_compare(plt, stats_df, output_dir / "05_annual_returns.png")
    print("  [OK] 05_annual_returns.png")

    plot_group_performance(plt, stats_df, output_dir / "06_group_performance.png")
    print("  [OK] 06_group_performance.png")

    # 5. 输出 CSV + HTML
    print(f"\n[5/5] 输出报告...")
    stats_csv = output_dir / "etf_stats.csv"
    stats_df.to_csv(stats_csv, index=False, encoding="utf-8-sig")
    print(f"  [OK] {stats_csv}")

    # 分组建议表
    groups_df = stats_df[["code", "name", "asset_class", "industry", "style",
                          "annual_return", "annual_volatility", "sharpe", "max_drawdown"]].copy()
    groups_df = groups_df.sort_values(["asset_class", "industry", "style", "annual_return"], ascending=[True, True, True, False])
    groups_csv = output_dir / "etf_groups.csv"
    groups_df.to_csv(groups_csv, index=False, encoding="utf-8-sig")
    print(f"  [OK] {groups_csv}")

    # HTML 报告
    html_path = generate_html_report(stats_df, groups_df, today, output_dir)
    print(f"  [OK] {html_path}")

    print(f"\n=== 完成 ===")
    print(f"输出目录: {output_dir}")
    print(f"共 {len(stats_df)} 个 ETF, {len(groups_df['asset_class'].unique())} 个资产大类, "
          f"{len(groups_df['industry'].unique())} 个行业, {len(groups_df['style'].unique())} 个风格")


if __name__ == "__main__":
    main()
