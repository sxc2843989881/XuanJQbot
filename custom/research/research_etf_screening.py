"""研究阶段：ETF 全市场筛选与分类研究

输入：custom/output/data/etf_list_<YYYYMMDD>.csv（1123 个 ETF，由 test_data_fetch_etf_index.py 生成）
输出：
  - custom/data/etf_klines/<code>.csv（每个 ETF 的日线缓存，便于复跑）
  - custom/output/research/etf_metrics_<YYYYMMDD>.csv（全市场指标表）
  - custom/output/research/etf_screening_<YYYYMMDD>.csv（候选池，每类前 10）
  - custom/output/research/etf_screening_<YYYYMMDD>.png（可视化报告）

工作流：
1. 读取 ETF 列表
2. 用 akshare 拉每个 ETF 近 2 年日线数据（baostock 对 ETF 只返回近 6 个月，故改用 akshare）
3. 计算指标：年化收益率/波动率/Sharpe/最大回撤/Calmar/日均成交额
4. 按名称关键字分类：股票-稳健/股票-增长/债券/大宗商品
5. 综合评分（年化+Sharpe+回撤+流动性加权），每类取前 10
6. 输出 CSV + 可视化报告

分类规则（按 ETF 名称关键字，顺序匹配）：
1. 债券：含"债/国债/信用债/可转债/企业债/利率债/纯债"
2. 大宗商品（窄义，仅商品 ETF）：含"黄金/白银/原油/石油/商品/豆粕/能源化工"（不含行业股 ETF）
3. 股票-增长型（行业 + 主题）：含"消费/医药/科技/金融/半导体/新能源/..."等
4. 股票-稳健型（宽基 + 红利/低波/价值）：含"上证50/180/300/500/创业板指/科创50/红利/..."
5. 其他：QDII/海外/货币/未识别

综合评分（各维度 min-max 归一化到 0-1）：
- 评分 = 0.30 * 年化收益率 + 0.30 * Sharpe + 0.20 * (-最大回撤) + 0.20 * 流动性
- 流动性评分 = 日均成交额对数归一化
"""
import os
import sys
import time
import math
import argparse
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np
import akshare as ak

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# 路径常量
ETF_LIST_DIR = PROJECT_ROOT / "custom" / "output" / "data"
KLINE_CACHE_DIR = PROJECT_ROOT / "custom" / "data" / "etf_klines"
OUTPUT_DIR = PROJECT_ROOT / "custom" / "output" / "research"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
KLINE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# 默认参数
DEFAULT_LOOKBACK_DAYS = 5000  # 拉取完整历史（覆盖 2010 年至今，ETF 最早 2011 年成立）
DEFAULT_TOP_N = 10            # 每类取前 10
DEFAULT_MIN_TURNOVER = 5e6    # 流动性筛选：日均成交额 >= 500 万（避免微型 ETF 噪声）

# 无风险利率（用于 Sharpe 计算，年化）
RISK_FREE_RATE = 0.02


# ============================================================
# 1. 数据获取（akshare + 缓存）
# ============================================================

def code_to_akshare(code: str) -> str:
    """'sh.510300' → '510300'"""
    return code.split(".")[-1]


def fetch_etf_kline(code: str, days: int = DEFAULT_LOOKBACK_DAYS) -> pd.DataFrame:
    """拉取 ETF 日线数据（akshare，含缓存）

    Args:
        code: ETF 代码，如 'sh.510300'
        days: 拉取近 N 个交易日

    Returns:
        DataFrame[date, open, high, low, close, volume, amount]
        失败返回空 DataFrame
    """
    # 缓存文件名（无后缀，复用已有 hfq 缓存）
    cache_path = KLINE_CACHE_DIR / f"{code.replace('.', '_')}.csv"

    # 缓存有效期 7 天（历史数据不需要每天更新）
    if cache_path.exists():
        mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
        if (datetime.now() - mtime) < timedelta(days=7):
            try:
                df = pd.read_csv(cache_path, parse_dates=["date"])
                if len(df) >= 120:  # 至少 120 日数据才有效（约 6 个月）
                    return df
            except Exception:
                pass

    symbol = code_to_akshare(code)
    end_date = datetime.now().strftime("%Y%m%d")
    start_date = (datetime.now() - timedelta(days=int(days * 1.6))).strftime("%Y%m%d")

    try:
        df = ak.fund_etf_hist_em(
            symbol=symbol,
            period="daily",
            start_date=start_date,
            end_date=end_date,
            adjust="hfq",  # 后复权（普通复权，加减法）
                           # 决策依据：参考《复权对 ETF 轮动策略的影响分析》
                           #   - 普通 qfq/hfq 都有涨幅失真缺陷，量化研究标准做法是"等比复权"
                           #   - 但 qfq 对老 ETF（510050 等）会产生负价，导致年化失真 49.91%
                           #   - hfq 不会产生负价，长期年化接近真实（510500 hfq 9.00% vs qfq 9.14%）
                           #   - 当前筛选阶段用 hfq 作为简化方案，未来回测阶段应实现"等比复权"
                           # TODO: 回测阶段实现等比复权算法（不复权数据 + 分红事件计算 factor 连乘）
        )
    except Exception as e:
        return pd.DataFrame()

    if df is None or len(df) == 0:
        return pd.DataFrame()

    # 列名转换（中文 → 英文）
    df = df.rename(columns={
        "日期": "date",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        "成交额": "amount",
    })
    df = df[["date", "open", "high", "low", "close", "volume", "amount"]]
    for col in ["open", "high", "low", "close", "volume", "amount"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["close"]).sort_values("date").reset_index(drop=True)

    # 只保留近 N 个交易日
    if len(df) > days:
        df = df.iloc[-days:].reset_index(drop=True)

    # 缓存
    if len(df) >= 120:
        df.to_csv(cache_path, index=False)

    return df


# ============================================================
# 2. 指标计算
# ============================================================

def compute_metrics(code: str, name: str, df: pd.DataFrame) -> dict:
    """计算 ETF 指标（基于完整历史）

    Returns:
        dict: {
            'code', 'name', 'days', 'start_date', 'end_date', 'years',
            'annual_return',       # 长期年化（成立至今）
            'annual_return_1y',    # 近 1 年年化
            'annual_return_3y',    # 近 3 年年化（数据不足则为 None）
            'annual_volatility', 'sharpe',
            'max_drawdown', 'calmar', 'avg_turnover',
        }
    """
    # 要求至少 120 个交易日（约 6 个月），否则指标不可靠
    if len(df) < 120:
        return None

    # 剔除 close <= 0 的异常数据（前复权可能产生负价）
    df = df[df["close"] > 0].reset_index(drop=True)
    if len(df) < 120:
        return None

    close = df["close"].values
    dates = df["date"].dt.strftime("%Y-%m-%d").values
    amount = df["amount"].values  # 成交额（元）

    # 日收益率（用价格序列直接计算，避免 cumprod 累积误差）
    daily_ret = np.diff(np.log(close))
    n_days = len(daily_ret)
    years = n_days / 252

    # 长期年化收益率（几何，成立至今）
    total_return = close[-1] / close[0] - 1
    annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

    # 近 1 年年化（252 个交易日）
    if n_days >= 252:
        ret_1y = close[-1] / close[-253] - 1  # -253 因为 close[-252] 是 252 天前
        annual_return_1y = (1 + ret_1y) ** (252 / 252) - 1  # 1 年，直接用总收益
    else:
        # 数据不足 1 年，用可用数据年化
        ret_1y = total_return
        annual_return_1y = (1 + ret_1y) ** (1 / years) - 1 if years > 0 else 0

    # 近 3 年年化（756 个交易日）
    if n_days >= 756:
        ret_3y = close[-1] / close[-757] - 1
        annual_return_3y = (1 + ret_3y) ** (1 / 3) - 1
    else:
        annual_return_3y = None  # 数据不足 3 年

    # 年化波动率（长期）
    annual_vol = np.std(daily_ret, ddof=1) * math.sqrt(252)

    # Sharpe（长期）
    sharpe = (annual_return - RISK_FREE_RATE) / annual_vol if annual_vol > 0 else 0

    # 最大回撤（用 close 价格序列直接计算，clip 到 [-1, 0] 防止异常）
    running_max = np.maximum.accumulate(close)
    drawdown = (close - running_max) / running_max
    max_drawdown = float(np.clip(drawdown.min(), -1.0, 0.0))

    # Calmar
    calmar = annual_return / abs(max_drawdown) if max_drawdown < 0 else 0

    # 日均成交额（近 60 日）
    avg_turnover = float(np.mean(amount[-60:])) if len(amount) >= 60 else float(np.mean(amount))

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
        "annual_volatility": float(annual_vol),
        "sharpe": float(sharpe),
        "max_drawdown": max_drawdown,
        "calmar": float(calmar),
        "avg_turnover": avg_turnover,
    }


# ============================================================
# 3. ETF 分类（按名称关键字，顺序匹配）
# ============================================================

# 1. 债券类关键字
BOND_KEYWORDS = ["债", "国债", "信用债", "可转债", "企业债", "利率债", "纯债", "短融", "短融券"]

# 2. 大宗商品类（窄义：仅商品 ETF，不含行业股 ETF 如煤炭/钢铁/有色）
COMMODITY_KEYWORDS = [
    "黄金", "白银", "原油", "石油", "商品", "豆粕", "能源化工", "上海金",
]

# 3. 股票-增长型（行业 + 主题，优先匹配，避免"新能源"被误判为大宗商品）
GROWTH_STOCK_KEYWORDS = [
    # 行业
    "消费", "医药", "生物", "科技", "金融", "地产", "房地产", "新能源", "半导体",
    "芯片", "军工", "国防", "证券", "保险", "银行", "食品饮料", "家电", "白酒",
    "TMT", "5G", "人工智能", "AI", "云计算", "大数据", "新能源车", "光伏", "风电",
    "环保", "传媒", "游戏", "旅游", "教育", "农业", "食品", "电力", "电网",
    "建材", "建筑", "机械", "汽车", "通信", "电子", "计算机", "纺织", "服装",
    "煤炭", "钢铁", "有色", "化工", "石化", "基础", "制造",
    "高端制造", "智能制造", "先进制造", "稀土", "矿业", "有色金属", "金属",
    "通信设备", "集成电路", "全指通信", "全指集成", "全指证券", "全指软件",
    "软件", "软件服务", "酒", "电池", "机器人", "航天航空", "航空", "航天",
    "非银", "非银金融", "证券公司", "全指非银", "信息技术", "数字经济",
    # 主题
    "碳中和", "碳达峰", "ESG", "创新", "创业", "成长", "中小",
    "互联网", "物联网", "车联网", "工业4.0", "国产替代", "自主可控",
    "创业板", "科创板", "机床", "卫星",
    # 增长风格
    "增长", "成长", "高速", "动力", "新能源",
]

# 4. 股票-稳健型（宽基 + 红利/低波/价值/基本面，去掉"中证全指"避免匹配行业 ETF）
STABLE_STOCK_KEYWORDS = [
    # 宽基指数
    "上证50", "上证180", "上证380", "上证综指", "上证指数", "深证100", "深证300",
    "沪深300", "中证500", "中证800", "中证1000", "中证2000",
    "创业板指", "科创50", "科创板50", "A50", "MSCI", "MSCI中国",
    "国证2000", "中证100", "中证700", "中证A500", "A500",
    "中证全指",  # 注意：放最后，且增长型已优先匹配"全指通信"等
    # 稳健风格
    "红利", "低波动", "低波", "价值", "基本面", "公司治理", "央企", "国企",
    "优势", "蓝筹",
]


def classify_etf(name: str) -> str:
    """根据 ETF 名称分类

    Args:
        name: ETF 名称，如 '华夏沪深300ETF'、'易方达黄金ETF'

    Returns:
        分类标签：'债券' / '大宗商品' / '股票-增长' / '股票-稳健' / '其他'

    匹配顺序（关键）：
        1. 债券（优先，避免"可转债ETF"误判）
        2. 大宗商品（窄义，仅商品 ETF）
        3. 股票-增长型（行业 + 主题，优先于稳健型，避免"全指通信设备"误判为宽基）
        4. 股票-稳健型（宽基 + 红利/低波/价值）
        5. 其他
    """
    # 1. 债券
    for kw in BOND_KEYWORDS:
        if kw in name:
            return "债券"

    # 2. 大宗商品（窄义）
    for kw in COMMODITY_KEYWORDS:
        if kw in name:
            return "大宗商品"

    # 3. 股票-增长型（先匹配，避免行业 ETF 被误判为宽基）
    for kw in GROWTH_STOCK_KEYWORDS:
        if kw in name:
            return "股票-增长"

    # 4. 股票-稳健型（宽基 + 红利/低波/价值）
    for kw in STABLE_STOCK_KEYWORDS:
        if kw in name:
            return "股票-稳健"

    # 5. 其他（QDII/海外/货币/未识别）
    return "其他"


# ============================================================
# 3.5 指数标签提取（用于去重：相同底层指数只保留评分最高的 1 个）
# ============================================================

# 顺序匹配规则：(关键字, 标签)
# 注意顺序：特殊组合优先（避免被通用关键字截断），如"黄金产业"在"黄金"前
INDEX_TAG_RULES = [
    # --- 特殊组合优先 ---
    ("黄金产业", "黄金产业股"),
    ("创业板人工智能", "创业板AI"),
    ("中韩半导体", "中韩半导体"),
    ("韩交所", "中韩半导体"),
    ("半导体材料", "半导体材料设备"),
    ("石油天然气", "石油天然气"),
    ("标普石油", "石油天然气"),
    ("投资级可转债", "可转债"),
    ("可转债", "可转债"),
    ("创新药", "创新药"),
    ("标普500", "标普500"),

    # --- 宽基指数 ---
    ("沪深300", "沪深300"),
    ("中证A500", "中证A500"),
    ("A500", "中证A500"),
    ("中证500", "中证500"),
    ("中证1000", "中证1000"),
    ("中证800", "中证800"),
    ("中证2000", "中证2000"),
    ("上证50", "上证50"),
    ("上证180", "上证180"),
    ("深证100", "深证100"),
    ("深证300", "深证300"),
    ("创业板指", "创业板指"),
    ("科创50", "科创50"),
    ("创业板", "创业板"),

    # --- 行业/主题 ---
    ("通信设备", "通信设备"),
    ("5G", "5G通信"),
    ("集成电路", "集成电路"),
    ("全指证券", "证券公司"),
    ("证券公司", "证券公司"),
    ("非银", "非银金融"),
    ("数字经济", "数字经济"),
    ("人工智能", "人工智能"),
    ("半导体", "半导体"),
    ("芯片", "芯片"),
    ("新能源车", "新能源车"),
    ("新能源", "新能源"),
    ("医药", "医药"),
    ("医疗", "医疗"),
    ("煤炭", "煤炭"),
    ("钢铁", "钢铁"),
    ("有色", "有色金属"),
    ("酒", "酒"),
    ("电池", "电池"),
    ("机器人", "机器人"),
    ("航天航空", "航天航空"),
    ("软件", "软件"),
    ("信息技术", "信息技术"),

    # --- 商品（黄金/上海金合并去重，底层都是黄金）---
    ("上海金", "黄金"),
    ("黄金", "黄金"),  # 黄金ETF（实物），黄金产业已优先匹配
    ("白银", "白银"),
    ("原油", "原油"),
    ("石油", "石油"),
    ("豆粕", "豆粕"),

    # --- 债券 ---
    ("30年期国债", "30年国债"),
    ("30年国债", "30年国债"),
    ("10年期国债", "10年国债"),
    ("10年国债", "10年国债"),
    ("公司债", "公司债"),
    ("信用债", "信用债"),
    ("国债", "国债"),

    # --- 海外QDII ---
    ("纳斯达克100", "纳指100"),
    ("纳指100", "纳指100"),
    ("日经225", "日经225"),
    ("恒生医疗", "恒生医疗"),
    ("德国DAX", "德国DAX"),
    ("亚太精选", "亚太精选"),
]


def extract_index_tag(name: str) -> str:
    """从 ETF 名称提取底层指数/标的标签（用于去重）

    相同标签的 ETF 视为同质化（追踪同一指数/标的），只保留评分最高的 1 个。
    未匹配到任何规则的，用完整名称作为标签（即不去重）。
    """
    for keyword, tag in INDEX_TAG_RULES:
        if keyword in name:
            return tag
    return name


# ============================================================
# 4. 综合评分
# ============================================================

def min_max_normalize(series: pd.Series) -> pd.Series:
    """Min-Max 归一化到 [0, 1]"""
    s = series.copy()
    s_min, s_max = s.min(), s.max()
    if s_max - s_min < 1e-9:
        return pd.Series([0.5] * len(s), index=s.index)
    return (s - s_min) / (s_max - s_min)


def compute_score(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """计算综合评分

    评分 = 0.30 * 年化收益率 + 0.30 * Sharpe + 0.20 * (-最大回撤) + 0.20 * 流动性
    各维度 min-max 归一化到 [0, 1]
    """
    df = metrics_df.copy()

    # 流动性 = log(日均成交额)
    df["liquidity_score"] = np.log10(df["avg_turnover"].clip(lower=1))

    # 归一化
    df["norm_return"] = min_max_normalize(df["annual_return"])
    df["norm_sharpe"] = min_max_normalize(df["sharpe"])
    df["norm_dd"] = min_max_normalize(-df["max_drawdown"])  # 回撤越小越好，取负
    df["norm_liq"] = min_max_normalize(df["liquidity_score"])

    # 综合评分
    df["score"] = (
        0.30 * df["norm_return"]
        + 0.30 * df["norm_sharpe"]
        + 0.20 * df["norm_dd"]
        + 0.20 * df["norm_liq"]
    )

    return df


# ============================================================
# 5. 可视化报告
# ============================================================

def plot_screening_report(screening_df: pd.DataFrame, output_path: Path):
    """生成候选池可视化报告

    每个 category 一个子图，展示该类前 10 的 ETF 评分对比
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 中文字体配置（Windows 优先 Microsoft YaHei / SimHei）
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    categories = screening_df["category"].unique()
    n_cat = len(categories)

    fig, axes = plt.subplots(n_cat, 1, figsize=(12, 4 * n_cat), constrained_layout=True)
    if n_cat == 1:
        axes = [axes]

    for ax, cat in zip(axes, categories):
        cat_df = screening_df[screening_df["category"] == cat].sort_values("score", ascending=True)
        names = [f"{n[:12]}" for n in cat_df["name"]]  # 截断名称
        scores = cat_df["score"].values

        # 评分柱状图
        bars = ax.barh(range(len(names)), scores, color="steelblue")
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=9)
        ax.set_xlabel("综合评分")
        ax.set_title(f"{cat} - Top {len(cat_df)}")
        ax.set_xlim(0, 1)
        for i, (bar, score) in enumerate(zip(bars, scores)):
            ax.text(score + 0.01, i, f"{score:.3f}", va="center", fontsize=8)

    fig.suptitle("ETF 全市场筛选 - 各类候选池", fontsize=14)
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 6. 主流程
# ============================================================

def find_latest_etf_list() -> Path:
    """找到最新的 ETF 列表 CSV"""
    files = sorted(ETF_LIST_DIR.glob("etf_list_*.csv"))
    if not files:
        raise FileNotFoundError(f"未找到 ETF 列表，请先运行 custom/tests/test_data_fetch_etf_index.py")
    return files[-1]


def run_screening(
    sample_size: int = None,
    top_n: int = DEFAULT_TOP_N,
    min_turnover: float = DEFAULT_MIN_TURNOVER,
    use_cache_only: bool = False,
):
    """主函数：ETF 全市场筛选

    Args:
        sample_size: 测试模式，只跑前 N 个 ETF（None 表示全量）
        top_n: 每类取前 N
        min_turnover: 流动性筛选阈值（日均成交额，元）
        use_cache_only: 只用缓存数据，不重新拉
    """
    today = datetime.now().strftime("%Y%m%d")

    # 1. 读取 ETF 列表
    etf_list_path = find_latest_etf_list()
    etf_list = pd.read_csv(etf_list_path)
    if sample_size:
        etf_list = etf_list.head(sample_size)
    print(f"[1/5] 读取 ETF 列表: {etf_list_path.name}（共 {len(etf_list)} 个 ETF）")

    # 2. 拉数据 + 计算指标（akshare 不需要登录）
    metrics = []
    failed = []
    cache_hits = 0
    api_calls = 0

    print(f"[2/5] 拉取日线数据 + 计算指标（akshare + 缓存目录: {KLINE_CACHE_DIR}）")
    for i, row in etf_list.iterrows():
        code = row["code"]
        name = row["code_name"]

        try:
            if use_cache_only:
                cache_path = KLINE_CACHE_DIR / f"{code.replace('.', '_')}.csv"
                if not cache_path.exists():
                    continue
                df = pd.read_csv(cache_path, parse_dates=["date"])
                cache_hits += 1
            else:
                # 检查是否命中缓存
                cache_path = KLINE_CACHE_DIR / f"{code.replace('.', '_')}.csv"
                cache_valid = False
                if cache_path.exists():
                    mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
                    if (datetime.now() - mtime) < timedelta(days=7):
                        cache_valid = True

                df = fetch_etf_kline(code)
                if cache_valid:
                    cache_hits += 1
                else:
                    api_calls += 1

                if df is None or len(df) == 0:
                    failed.append((code, name, "拉不到数据"))
                    continue

            m = compute_metrics(code, name, df)
            if m is None:
                failed.append((code, name, "数据不足120日"))
                continue

            # 流动性筛选
            if m["avg_turnover"] < min_turnover:
                failed.append((code, name, f"流动性不足:{m['avg_turnover']/1e4:.0f}万"))
                continue

            metrics.append(m)
        except Exception as e:
            failed.append((code, name, str(e)[:50]))

        # 进度打印
        if (i + 1) % 50 == 0 or (i + 1) == len(etf_list):
            print(f"  进度: {i+1}/{len(etf_list)} | 成功 {len(metrics)} | 失败 {len(failed)} | 缓存命中 {cache_hits} | API {api_calls}")

        # 控制请求频率，避免被 akshare 限流（东方财富网）
        if not use_cache_only and api_calls > 0 and api_calls % 10 == 0:
            time.sleep(0.5)

    if not metrics:
        print("[FAIL] 没有拉到任何有效 ETF 数据")
        return None

    # 3. 输出全市场指标表
    metrics_df = pd.DataFrame(metrics)
    metrics_csv = OUTPUT_DIR / f"etf_metrics_{today}.csv"
    metrics_df.to_csv(metrics_csv, index=False, encoding="utf-8-sig")
    print(f"[3/5] 全市场指标表: {metrics_csv}（{len(metrics_df)} 个 ETF）")

    # 4. 分类 + 评分
    print("[4/5] 分类 + 综合评分 + 去重")
    metrics_df["category"] = metrics_df["name"].apply(classify_etf)
    scored_df = compute_score(metrics_df)

    # 分类统计（去重前）
    cat_counts = scored_df["category"].value_counts()
    for cat, cnt in cat_counts.items():
        print(f"  {cat}: {cnt} 个")

    # 去重：相同底层指数标签只保留评分最高的 1 个（避免多个跟踪同一指数的 ETF 重复入选）
    scored_df["index_tag"] = scored_df["name"].apply(extract_index_tag)
    n_before = len(scored_df)
    scored_df = (
        scored_df
        .sort_values("score", ascending=False)
        .drop_duplicates(subset=["index_tag"], keep="first")
        .reset_index(drop=True)
    )
    n_after = len(scored_df)
    n_dup = n_before - n_after
    print(f"  去重: {n_before} → {n_after}（剔除同指数重复 {n_dup} 个，保留评分最高）")

    # 每类取前 N
    screening_dfs = []
    for cat in ["股票-稳健", "股票-增长", "债券", "大宗商品", "其他"]:
        cat_df = scored_df[scored_df["category"] == cat]
        if len(cat_df) == 0:
            continue
        top_df = cat_df.nlargest(top_n, "score")
        screening_dfs.append(top_df)

    if not screening_dfs:
        print("[FAIL] 分类后无有效 ETF")
        return None

    screening_df = pd.concat(screening_dfs, ignore_index=True)
    screening_csv = OUTPUT_DIR / f"etf_screening_{today}.csv"
    # 输出列：category, rank, code, name, index_tag, score, 长期年化, 近1年年化, 近3年年化, 波动, sharpe, 回撤, calmar, 成交额, 年数
    screening_df_out = screening_df[[
        "category", "code", "name", "index_tag", "score",
        "annual_return", "annual_return_1y", "annual_return_3y",
        "annual_volatility", "sharpe", "max_drawdown", "calmar",
        "avg_turnover", "days", "years", "start_date", "end_date",
    ]].copy()
    screening_df_out.insert(1, "rank", screening_df_out.groupby("category")["score"].rank(ascending=False, method="first").astype(int))
    screening_df_out = screening_df_out.sort_values(["category", "rank"])
    screening_df_out.to_csv(screening_csv, index=False, encoding="utf-8-sig")
    print(f"[OK] 候选池: {screening_csv}（{len(screening_df)} 个 ETF，每类前 {top_n}）")

    # 5. 可视化报告
    print("[5/5] 生成可视化报告")
    chart_path = OUTPUT_DIR / f"etf_screening_{today}.png"
    try:
        plot_screening_report(screening_df, chart_path)
        print(f"[OK] 可视化报告: {chart_path}")
    except Exception as e:
        print(f"[WARN] 可视化报告生成失败: {e}")

    # 失败列表
    if failed:
        failed_csv = OUTPUT_DIR / f"etf_failed_{today}.csv"
        pd.DataFrame(failed, columns=["code", "name", "reason"]).to_csv(
            failed_csv, index=False, encoding="utf-8-sig"
        )
        print(f"[INFO] 失败列表: {failed_csv}（{len(failed)} 个）")

    # 控制台打印候选池摘要
    print("\n" + "=" * 80)
    print("候选池摘要（每类前 10）")
    print("=" * 80)
    for cat in ["股票-稳健", "股票-增长", "债券", "大宗商品", "其他"]:
        cat_df = screening_df_out[screening_df_out["category"] == cat]
        if len(cat_df) == 0:
            continue
        print(f"\n【{cat}】")
        for _, r in cat_df.iterrows():
            r3y = f"{r['annual_return_3y']*100:6.2f}%" if pd.notna(r['annual_return_3y']) else "  N/A "
            print(
                f"  {r['rank']:2d}. {r['code']:10s} {r['name'][:18]:18s} | "
                f"评分 {r['score']:.3f} | "
                f"长期年化 {r['annual_return']*100:6.2f}% | "
                f"近1年 {r['annual_return_1y']*100:6.2f}% | "
                f"近3年 {r3y} | "
                f"{r['years']:5.1f}年 | "
                f"Sharpe {r['sharpe']:5.2f} | "
                f"回撤 {r['max_drawdown']*100:6.2f}%"
            )

    return {
        "metrics_csv": metrics_csv,
        "screening_csv": screening_csv,
        "chart_path": chart_path,
        "n_total": len(etf_list),
        "n_valid": len(metrics_df),
        "n_screening": len(screening_df),
    }


def main():
    parser = argparse.ArgumentParser(description="ETF 全市场筛选与分类研究")
    parser.add_argument("--sample", type=int, default=None, help="测试模式：只跑前 N 个 ETF")
    parser.add_argument("--top", type=int, default=DEFAULT_TOP_N, help="每类取前 N（默认 10）")
    parser.add_argument("--min-turnover", type=float, default=DEFAULT_MIN_TURNOVER, help="流动性筛选阈值（日均成交额，元）")
    parser.add_argument("--cache-only", action="store_true", help="只用缓存数据，不重新拉")
    args = parser.parse_args()

    result = run_screening(
        sample_size=args.sample,
        top_n=args.top,
        min_turnover=args.min_turnover,
        use_cache_only=args.cache_only,
    )

    if result:
        print(f"\n[DONE] 筛选完成：{result['n_valid']}/{result['n_total']} 有效，候选池 {result['n_screening']} 个")


if __name__ == "__main__":
    main()
