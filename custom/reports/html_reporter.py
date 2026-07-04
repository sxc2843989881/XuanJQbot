"""custom.reports.html_reporter - HTML 报告生成

生成自包含 HTML 报告（无外部依赖，所有图片 base64 内联），兼容 Python 3.8。
- 基础版：关键指标表 + 嵌入 4合1 PNG + 交易明细表
- 详细版：基础版 + 扩展图（月度热力图/滚动 Sharpe/回撤期/收益分布） + 扩展指标表
"""
import base64
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from custom.reports.contract import BacktestData, BacktestMetrics, TradeRecord


def _img_to_base64(img_path) -> str:
    """读取图片文件并转 base64 字符串"""
    with open(str(img_path), 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def _format_pct(v, default='N/A') -> str:
    """格式化百分比（输入小数，输出 'xx.xx%'）"""
    if v is None:
        return default
    return f"{v*100:.2f}%"


def _format_float(v, digits=4, default='N/A') -> str:
    if v is None:
        return default
    return f"{v:.{digits}f}"


def _build_metrics_html(metrics: BacktestMetrics) -> str:
    """关键指标摘要表 HTML"""
    color_ret = '#43A047' if metrics.total_return >= 0 else '#E53935'
    sharpe_str = _format_float(metrics.sharpe, 4)
    sortino_str = _format_float(metrics.sortino, 4) if metrics.sortino else 'N/A'
    vol_str = _format_pct(metrics.volatility) if metrics.volatility else 'N/A'
    dd_dur_str = f"{metrics.max_dd_duration} 天" if metrics.max_dd_duration is not None else 'N/A'
    params_str = ', '.join(f"{k}={v}" for k, v in metrics.strategy_params.items()) or '-'

    return f"""
    <table>
        <tr><th>标的</th><td>{metrics.code}</td>
            <th>策略</th><td>{metrics.strategy_name or '-'} ({params_str})</td></tr>
        <tr><th>回测区间</th><td>{metrics.start} ~ {metrics.end}</td>
            <th>天数</th><td>{metrics.days}</td></tr>
        <tr><th>初始资金</th><td>{metrics.cash:.2f}</td>
            <th>最终资金</th><td>{metrics.final_value:.2f}</td></tr>
        <tr><th>总收益率</th><td style="color:{color_ret}">{metrics.total_return*100:.2f}%</td>
            <th>年化收益率</th><td style="color:{color_ret}">{metrics.annual_return*100:.2f}%</td></tr>
        <tr><th>最大回撤</th><td style="color:#E53935">{metrics.max_drawdown*100:.2f}%</td>
            <th>回撤类型</th><td>完整周期</td></tr>
        <tr><th>最大回撤持续</th><td>{dd_dur_str}</td>
            <th>年化波动率</th><td>{vol_str}</td></tr>
        <tr><th>Sharpe Ratio</th><td>{sharpe_str}</td>
            <th>Sortino Ratio</th><td>{sortino_str}</td></tr>
        <tr><th>Calmar Ratio</th><td>{metrics.calmar:.4f}</td>
            <th>总交易次数</th><td>{metrics.total_trades}</td></tr>
        <tr><th>盈利次数</th><td style="color:#43A047">{metrics.won_trades}</td>
            <th>亏损次数</th><td style="color:#E53935">{metrics.lost_trades}</td></tr>
        <tr><th>胜率</th><td>{metrics.win_rate:.2f}%</td><th></th><td></td></tr>
    </table>"""


def _build_trades_html(trades: List[TradeRecord]) -> str:
    """交易明细表 HTML"""
    if not trades:
        return "<p>无交易记录</p>"
    rows = []
    for t in trades:
        tcolor = '#43A047' if t.type.upper() == 'BUY' else '#E53935'
        rows.append(
            f"<tr><td>{t.date}</td><td style='color:{tcolor}'>{t.type}</td>"
            f"<td>{t.price:.2f}</td><td>{t.size}</td></tr>"
        )
    return (
        "<table><tr><th>日期</th><th>类型</th><th>价格</th><th>数量</th></tr>"
        + ''.join(rows)
        + "</table>"
    )


def _build_html(title: str, sections_html: List[str]) -> str:
    """组装完整 HTML 文档"""
    body = '\n'.join(sections_html)
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="utf-8">
    <title>{title}</title>
    <style>
        body {{ font-family: 'Microsoft YaHei', 'SimHei', sans-serif; margin: 30px; background: #f5f5f5; }}
        h1 {{ color: #1E88E5; border-bottom: 2px solid #1E88E5; padding-bottom: 10px; }}
        h2 {{ color: #333; margin-top: 30px; border-left: 4px solid #1E88E5; padding-left: 10px; }}
        h3 {{ color: #555; margin-top: 20px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 10px 0; background: white; }}
        td, th {{ border: 1px solid #ddd; padding: 10px; text-align: left; }}
        th {{ background-color: #f2f2f2; width: 15%; }}
        td {{ width: 35%; }}
        img {{ max-width: 100%; border: 1px solid #ddd; box-shadow: 0 2px 5px rgba(0,0,0,0.1); margin: 10px 0; }}
        .footer {{ margin-top: 30px; color: #888; font-size: 12px; text-align: center; }}
        .note {{ background: #fff9c4; padding: 10px; border-left: 4px solid #fbc02d; margin: 10px 0; }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    {body}
    <div class="footer">
        生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br>
        数据来源：baostock（网络实时获取） | 回测引擎：backtrader | 回撤定义：完整周期<br>
        报告模块：custom.reports
    </div>
</body>
</html>"""


def generate_basic_html_report(data: BacktestData,
                               metrics: BacktestMetrics,
                               chart_path,
                               save_path) -> None:
    """生成基础版 HTML 报告

    内容：关键指标表 + 嵌入 4合1 PNG + 交易明细表
    """
    img_b64 = _img_to_base64(chart_path)
    strategy_label = f"{data.strategy_name} " if data.strategy_name else ""

    sections = [
        "<h2>关键指标摘要</h2>" + _build_metrics_html(metrics),
        "<h2>可视化图表（4合1）</h2>",
        "<p>包含 4 个子图：价格+因子+买卖点 / 资金曲线+回撤区间 / 持仓跳仓变化 / 因子信号</p>",
        f'<img src="data:image/png;base64,{img_b64}" alt="回测可视化图表">',
        "<h2>交易明细</h2>" + _build_trades_html(data.trades),
    ]
    title = f"{strategy_label}策略回测报告 - {data.code}"
    html = _build_html(title, sections)

    with open(str(save_path), 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"  基础版 HTML 报告已保存：{save_path}")


def generate_detailed_html_report(data: BacktestData,
                                  metrics: BacktestMetrics,
                                  basic_chart_path,
                                  detailed_chart_path,
                                  save_path) -> None:
    """生成详细版 HTML 报告

    内容：基础版全部内容 + 扩展指标 + 扩展图（月度热力图/滚动 Sharpe/回撤期/收益分布）
    """
    basic_b64 = _img_to_base64(basic_chart_path)
    detailed_b64 = _img_to_base64(detailed_chart_path)
    strategy_label = f"{data.strategy_name} " if data.strategy_name else ""

    sections = [
        "<h2>关键指标摘要</h2>" + _build_metrics_html(metrics),
        '<div class="note">详细版报告包含：基础 4合1 图 + 扩展分析图（月度收益热力图、滚动 Sharpe、回撤期分析、收益分布）</div>',
        "<h2>可视化图表（基础 4合1）</h2>",
        f'<img src="data:image/png;base64,{basic_b64}" alt="基础 4合1 图表">',
        "<h2>扩展分析图表</h2>",
        "<p>包含 4 个扩展子图：月度收益热力图 / 滚动 Sharpe Ratio / 回撤期分析 / 日收益分布</p>",
        f'<img src="data:image/png;base64,{detailed_b64}" alt="扩展分析图表">',
        "<h2>交易明细</h2>" + _build_trades_html(data.trades),
    ]
    title = f"{strategy_label}策略回测报告（详细版） - {data.code}"
    html = _build_html(title, sections)

    with open(str(save_path), 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"  详细版 HTML 报告已保存：{save_path}")
