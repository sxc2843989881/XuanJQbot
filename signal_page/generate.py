"""B1-木星 每日信号生成器（独立版）
本脚本无本地依赖，可直接在 GitHub Actions 上运行。
依赖: pip install baostock pandas numpy matplotlib
"""
import os, base64, io, warnings
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

# ========== 配置 ==========
CODE_G = '159259'  # 易方达成长ETF
CODE_V = '159263'  # 易方达价值ETF
NAME_G = '易方达成长ETF'
NAME_V = '易方达价值ETF'
FETCH_DAYS = 400  # 拉取天数
MA_PERIOD = 20    # 均线周期

# B1-木星 基线参数
PARAMS = dict(
    dc=5,           # 方向确认天数
    dcd=6,          # 方向冷却天数
    rt=1.3,         # T弱信号阈值
    slope_thresh=0.002,  # 弱斜率阈值
    st=0.09,        # E5止损阈值
    cd=8,           # E5冷却天数
    bias_high=0.19, # BIAS超买阈值
    ms=10, ml=20,   # B2动量窗口
)

# ========== 数据获取 ==========
def fetch_baostock(code, name, days=400):
    """从 baostock 获取 ETF 日线数据"""
    import baostock as bs
    bs.login()
    end = datetime.now().strftime('%Y-%m-%d')
    start = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    rs = bs.query_history_k_data_plus(
        f'sz.{code}', 'date,open,close',
        start_date=start, end_date=end,
        frequency='d', adjustflag='2')
    rows = []
    while rs.next():
        rows.append(rs.get_row_data())
    bs.logout()
    if not rows:
        raise ValueError(f'{name}: 无数据')
    df = pd.DataFrame(rows, columns=['date','open','close'])
    df = df[df['close'] != '']
    for c in ['open','close']:
        df[c] = df[c].astype(float)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').sort_index()
    print(f'  ✅ {name}({code}): {df.index[0].date()} ~ {df.index[-1].date()} ({len(df)}天)')
    return df['close']

# ========== 信号计算 ==========
def compute_signals(g_close, v_close, params):
    """独立实现 B1-木星 完整信号逻辑（与原版 build_core 一致）"""
    ratio = g_close / v_close
    ma20 = ratio.rolling(MA_PERIOD).mean()
    dev = ratio / ma20 - 1
    t = dev / dev.rolling(MA_PERIOD).std()
    slope = ma20.diff(5) / ma20.shift(5) / 5
    g_bias = g_close / g_close.rolling(MA_PERIOD).mean() - 1
    v_bias = v_close / v_close.rolling(MA_PERIOD).mean() - 1
    g_dd20 = g_close.pct_change(20)
    v_dd20 = v_close.pct_change(20)
    v_mom_short = v_close.pct_change(params['ms'])
    v_mom_long = v_close.pct_change(params['ml'])
    
    n = len(t)
    
    # 第1层：方向确认 (DC)
    raw_dir = (t > 0).map({True: 'growth', False: 'value'})
    dir_dc = raw_dir.copy()
    for i in range(params['dc'], n):
        window = raw_dir.iloc[i-params['dc']+1:i+1]
        if (window == 'growth').all() or (window == 'value').all():
            dir_dc.iloc[i] = window.iloc[-1]
        else:
            dir_dc.iloc[i] = dir_dc.iloc[i-1] if i > 0 else raw_dir.iloc[0]
    
    # 第2层：方向冷却 (DCD)
    dcd = params['dcd']
    dir_dcd = dir_dc.copy()
    last_switch = -dcd - 1
    for i in range(n):
        cur = dir_dc.iloc[i]
        prev = dir_dcd.iloc[i-1] if i > 0 else cur
        if cur != prev and i - last_switch >= dcd:
            last_switch = i
            dir_dcd.iloc[i] = cur
        else:
            dir_dcd.iloc[i] = prev
    
    # 第3层：弱信号空仓
    is_weak = (t.abs() < params['rt']) & (slope.abs() < params['slope_thresh'])
    wt = pd.Series(1.0, index=t.index)
    wt[is_weak] = 0.0
    
    # 第4层：B2价值动量
    wrong_value = (dir_dcd == 'value') & (v_mom_short <= 0) & (v_mom_long <= 0)
    dir_b2 = dir_dcd.copy()
    dir_b2[wrong_value] = 'growth'
    dir_s = dir_b2  # 最终方向信号
    
    # 第5层：BIAS超买保护 (clear模式)
    if params.get('bias_mode', 'clear') == 'clear':
        extreme_g = (dir_s == 'growth') & (g_bias > params['bias_high'])
        extreme_v = (dir_s == 'value') & (v_bias > params['bias_high'])
        wt[extreme_g | extreme_v] = 0.0
    
    # 第6层：E5止损
    e5_g = (dir_s == 'growth') & (g_dd20 < -params['st'])
    e5_v = (dir_s == 'value') & (v_dd20 < -params['st'])
    e5_trig = e5_g | e5_v
    sw = 0.17
    
    in_cd = False
    cd_count = 0
    for i in range(n):
        idx = t.index[i]
        if e5_trig.iloc[i] and not in_cd and wt.iloc[i] > 0:
            in_cd = True
            cd_count = 0
            wt.iloc[i] *= sw
        elif in_cd:
            cd_count += 1
            if cd_count >= params['cd']:
                if e5_trig.iloc[i]:
                    cd_count = params['cd'] - 3
                    wt.iloc[i] *= sw
                else:
                    in_cd = False
                    if is_weak.iloc[i]:
                        wt.iloc[i] = 0.0
                    else:
                        wt.iloc[i] = 1.0
            else:
                if wt.iloc[i] > 0:
                    wt.iloc[i] = sw
    
    df = pd.DataFrame(index=t.index)
    df['ratio'] = ratio
    df['ma20'] = ma20
    df['dev'] = dev
    df['T'] = t
    df['slope'] = slope
    df['dir'] = dir_s
    df['wt'] = wt
    df['g_bias'] = g_bias
    df['v_bias'] = v_bias
    df['g_dd20'] = g_dd20
    df['v_dd20'] = v_dd20
    df['dir_dcd'] = dir_dcd
    df['is_weak'] = is_weak
    return df

# ========== 图表生成 ==========
def make_chart(df, rt=1.3, nav=None, start_date=None, init_cap=10000):
    """生成信号图表，返回 base64 PNG
    nav: P&L净值序列（可选），显示在顶部
    """
    n_rows = 4 if nav is not None else 3
    ratios = [1.5, 2, 1, 1] if nav is not None else [2, 1, 1]
    fig, axes = plt.subplots(n_rows, 1, figsize=(12, 8),
        gridspec_kw={'height_ratios': ratios})
    
    # P&L实盘走势（最顶部，从起点日开始，最多30天）
    if nav is not None:
        ax = axes[0]
        nav_valid = nav[nav.index >= start_date] if start_date is not None else nav
        nav_dates = nav_valid.index[-30:]
        nav_recent = nav.reindex(nav_dates).ffill()
        init_val = init_cap
        ax.plot(nav_recent.index, nav_recent.values, color='#E74C3C', lw=2)
        ax.fill_between(nav_recent.index, init_val, nav_recent.values, color='#E74C3C', alpha=0.15)
        ax.axhline(init_val, color='gray', ls=':', lw=0.8)
        ax.annotate(f'{nav_recent.iloc[-1]:.0f}元',
            xy=(nav_recent.index[-1], nav_recent.iloc[-1]),
            fontsize=11, fontweight='bold', color='#E74C3C',
            va='bottom', ha='left')
        pnl_pct = (nav_recent.iloc[-1]/init_val - 1)*100
        ax.set_title(f'实盘走势（起点1万元） 当前 {nav_recent.iloc[-1]:.0f}元  ({pnl_pct:+.2f}%)',
            fontsize=12, fontweight='bold')
        ax.set_ylabel('收益(元)', fontsize=9)
        ax.grid(alpha=0.2)
    
    ax_next = 1 if nav is not None else 0
    dates = df.index[-120:]  # 信号图表取最近120天
    
    # 比价 + MA20
    ax = axes[ax_next]
    ax.plot(dates, df.loc[dates, 'ratio'], color='#333333', lw=1.5, label='成长/价值')
    ax.plot(dates, df.loc[dates, 'ma20'], color='#E74C3C', ls='--', lw=1, label='MA20')
    ax.axhline(1.0, color='gray', ls=':', alpha=0.4)
    ax.set_ylabel('成长/价值', fontsize=9)
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(alpha=0.2)
    
    # 填充方向色
    dir_colors = {'growth': '#F1C40F', 'value': '#2ECC71', 'cash': '#95A5A6'}
    last_dir = None; start = None
    for d in dates:
        nd = df.loc[d, 'dir']
        if nd != last_dir:
            if last_dir is not None and start is not None:
                ax.axvspan(start, d, alpha=0.08, color=dir_colors.get(last_dir, '#CCC'))
            start = d; last_dir = nd
    if last_dir is not None and start is not None:
        ax.axvspan(start, dates[-1], alpha=0.08, color=dir_colors.get(last_dir, '#CCC'))
    
    # T值
    ax = axes[ax_next + 1]
    ax.plot(dates, df.loc[dates, 'T'], color='#3498DB', lw=1.5, label='T(z-score)')
    ax.axhline(0, color='gray', ls='-', lw=0.5)
    ax.axhline(rt, color='#E74C3C', ls='--', lw=0.8, label=f'+rt={rt}')
    ax.axhline(-rt, color='#2ECC71', ls='--', lw=0.8, label=f'-rt={rt}')
    ax.fill_between(dates, rt, -rt, alpha=0.05, color='gray')
    ax.set_ylabel('T (z-score)', fontsize=9)
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(alpha=0.2)
    
    # 仓位
    ax = axes[ax_next + 2]
    ax.fill_between(dates, 0, df.loc[dates, 'wt'], color='#9B59B6', alpha=0.4, step='mid')
    ax.plot(dates, df.loc[dates, 'wt'], color='#9B59B6', lw=1, drawstyle='steps-post')
    ax.set_ylabel('仓位', fontsize=9)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel('日期', fontsize=9)
    ax.grid(alpha=0.2)
    
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

# ========== HTML 生成 ==========
def make_html(signal, df, chart_b64, trades=None):
    """生成信号页面 HTML"""
    latest = signal
    pos_cn = {'growth': '📈 成长', 'value': '📉 价值', 'cash': '⏸ 空仓'}
    pos_color = {'growth': '#F1C40F', 'value': '#2ECC71', 'cash': '#95A5A6'}
    
    # 历史信号摘要
    n = min(30, len(df))
    recent = df.tail(n)
    g_days = (recent['dir'] == 'growth').sum()
    v_days = (recent['dir'] == 'value').sum()
    c_days = (recent['dir'] == 'cash').sum()
    
    # 最近10天表格
    last10 = df.tail(10)[['ratio', 'ma20', 'T', 'slope', 'dir', 'wt']].copy()
    last10.index = [d.strftime('%m-%d') for d in last10.index]
    
    last10['wt_label'] = last10['wt'].apply(lambda w: '空仓' if w==0 else f'{w*100:.0f}%')
    
    return f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>B1-木星 每日信号</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: #f5f6fa; color: #2c3e50; padding: 20px; }}
  .container {{ max-width: 800px; margin: 0 auto; }}
  
  .header {{ background: linear-gradient(135deg, #2c3e50, #3498db);
    color: white; padding: 24px; border-radius: 12px; margin-bottom: 16px;
    display: flex; justify-content: space-between; align-items: center; }}
  .header h1 {{ font-size: 20px; font-weight: 600; }}
  .header .date {{ font-size: 14px; opacity: 0.8; }}
  
  .signal-card {{ background: white; border-radius: 12px; padding: 20px;
    margin-bottom: 16px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); }}
  .signal-main {{ display: flex; align-items: center; gap: 24px; }}
  .signal-badge {{ font-size: 32px; font-weight: 700; padding: 12px 24px;
    border-radius: 10px; color: white;
    background: {pos_color[latest['dir']]}; min-width: 140px; text-align: center; }}
  .signal-meta {{ flex: 1; }}
  .signal-meta .row {{ display: flex; justify-content: space-between;
    padding: 4px 0; font-size: 14px; border-bottom: 1px solid #f0f0f0; }}
  .signal-meta .row:last-child {{ border: none; }}
  .signal-meta .label {{ color: #7f8c8d; }}
  .signal-meta .value {{ font-weight: 600; }}
  
  .charts {{ margin-bottom: 16px; }}
  .charts img {{ width: 100%; border-radius: 8px; }}
  
  .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px;
    margin-bottom: 16px; }}
  .stat-card {{ background: white; border-radius: 10px; padding: 16px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06); text-align: center; }}
  .stat-card .num {{ font-size: 28px; font-weight: 700; color: #2c3e50; }}
  .stat-card .lbl {{ font-size: 12px; color: #95a5a6; margin-top: 2px; }}
  
  .rec-card {{ background: white; border-radius: 12px; padding: 20px;
    margin-bottom: 16px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); }}
  .rec-card .title {{ font-size: 14px; font-weight: 600; margin-bottom: 8px; }}
  .rec-card .text {{ font-size: 16px; line-height: 1.8; }}
  
  table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
  th, td {{ padding: 6px 8px; text-align: center; border-bottom: 1px solid #eee; }}
  th {{ background: #f8f9fa; font-weight: 600; color: #7f8c8d; }}
  
  .footer {{ text-align: center; padding: 20px; color: #95a5a6; font-size: 12px; }}
  
  @media (max-width: 600px) {{
    .signal-main {{ flex-direction: column; }}
    .grid {{ grid-template-columns: 1fr 1fr; }}
    .header h1 {{ font-size: 16px; }}
  }}
</style>
</head>
<body>
<div class="container">
  <div class="header">
    <div>
      <h1>🌲 B1-木星</h1>
      <div>成长价值轮动策略 · 每日信号</div>
    </div>
    <div class="date">{datetime.now().strftime('%Y-%m-%d %H:%M')}</div>
  </div>
  
  <div class="signal-card">
    <div class="signal-main">
      <div class="signal-badge">{pos_cn[latest['dir']]}</div>
      <div class="signal-meta">
        <div class="row"><span class="label">仓位</span>
          <span class="value">{'空仓' if latest['wt']==0 else f'{latest["wt"]*100:.0f}%'}</span></div>
        <div class="row"><span class="label">数据截至</span>
          <span class="value">{latest['date'].strftime('%Y-%m-%d')}</span></div>
        <div class="row"><span class="label">T值 (z-score)</span>
          <span class="value">{latest['T']:+.3f}</span></div>
        <div class="row"><span class="label">SLOPE (斜率)</span>
          <span class="value">{latest['slope']:+.6f}</span></div>
        <div class="row"><span class="label">成长/价值偏离MA20</span>
          <span class="value">{latest['dev_pct']:+.2f}%</span></div>
        <div class="row"><span class="label">持仓标的</span>
          <span class="value">{latest['target']}</span></div>
      </div>
    </div>
  </div>
  
  <div class="charts">
    <img src="data:image/png;base64,{chart_b64}" alt="信号图表">
  </div>
  
  <div class="grid">
    <div class="stat-card">
      <div class="num">{g_days}</div>
      <div class="lbl">最近{n}天 · 成长</div>
    </div>
    <div class="stat-card">
      <div class="num">{v_days}</div>
      <div class="lbl">最近{n}天 · 价值</div>
    </div>
    <div class="stat-card">
      <div class="num">{c_days}</div>
      <div class="lbl">最近{n}天 · 空仓</div>
    </div>
    <div class="stat-card">
      <div class="num">{int(df['T'].tail(60).abs().mean()*100)/100:.2f}</div>
      <div class="lbl">T均值(60天)</div>
    </div>
    <div class="stat-card">
      <div class="num" style="color:{'#E74C3C' if latest.get('profit_pct',0)>=0 else '#2ECC71'}">{latest.get('nav',0):.0f}</div>
      <div class="lbl">账户净值(元)  {latest.get('profit_pct',0):+.2f}%</div>
    </div>
  </div>
  
  <div class="rec-card">
    <div class="title">💰 交易建议</div>
    <div class="text">{latest['advice']}</div>
  </div>
  
  <div class="signal-card">
    <div class="title" style="font-size:14px;font-weight:600;margin-bottom:8px">📋 最近10天信号</div>
    <table>
      <tr><th>日期</th><th>比价</th><th>MA20</th><th>T值</th><th>斜率</th><th>方向</th><th>仓位</th></tr>
      {''.join(f'<tr><td>{d}</td><td>{r["ratio"]:.3f}</td><td>{r["ma20"]:.3f}</td>'
        f'<td>{r["T"]:+.2f}</td><td>{r["slope"]:+.4f}</td>'
        f'<td>{pos_cn[r["dir"]]}</td><td>{r["wt_label"]}</td></tr>'
        for d, r in last10.iterrows())}
    </table>
  </div>
  
  <div class="signal-card">
    <div class="title" style="font-size:14px;font-weight:600;margin-bottom:8px">📋 调仓记录</div>
    <table>
      <tr><th>日期</th><th>操作</th><th>从</th><th>到</th></tr>
      {''.join(f'<tr><td>{t["date"]}</td><td>{t["action"]}</td><td>{t["from"]}</td><td>{t["to"]}</td></tr>' for t in (trades or [])[-20:][::-1])}
    </table>
    <div style="font-size:11px;color:#95a5a6;margin-top:4px">显示最近20笔，最新在前</div>
  </div>
  
  <div class="footer">
    B1-木星 成长价值轮动策略 · 自动生成 · 仅供参考，不构成投资建议<br>
    数据来源: baostock · {datetime.now().strftime('%Y-%m-%d')}
  </div>
</div>
</body>
</html>'''

# ========== 主流程 ==========
def main():
    print('=' * 50)
    print('  B1-木星 每日信号生成器')
    print(f'  {datetime.now().strftime("%Y-%m-%d %H:%M")}')
    print('=' * 50)
    
    # 1. 获取数据
    print('\n获取行情数据...')
    g = fetch_baostock(CODE_G, NAME_G, FETCH_DAYS)
    v = fetch_baostock(CODE_V, NAME_V, FETCH_DAYS)
    common = g.index.intersection(v.index)
    g, v = g[common], v[common]
    print(f'  共同交易日: {len(g)}天')
    
    # 2. 计算信号
    print('\n计算信号...')
    df = compute_signals(g, v, PARAMS)
    
    # 3. 计算P&L（从今日起初始10000元，往后累积）
    print('\n计算P&L...')
    daily_g_ret = g.pct_change()
    daily_v_ret = v.pct_change()
    TRADE_COST = 0.0001  # 佣金 1bps
    IMPACT_COST = 0.0005  # 冲击滑点 5bps
    TOTAL_COST = TRADE_COST + IMPACT_COST  # 0.06%/边
    INIT_CAP = 10000
    START_DATE = df.index[-1]
    nav = pd.Series(INIT_CAP, index=df.index)
    strat_ret = pd.Series(0.0, index=df.index)
    total_cost_sum = 0.0
    for i in range(1, len(df)):
        if df.index[i] <= START_DATE:
            continue
        ps = df.iloc[i-1]; cs = df.iloc[i]
        # 信号变化时扣除交易成本
        cost = 0.0
        if ps['dir'] != cs['dir'] or abs(ps['wt'] - cs['wt']) > 0.01:
            cost = TOTAL_COST * 2  # 卖出+买入 = 双边
        if ps['wt'] > 0 and ps['dir'] == 'growth':
            strat_ret.iloc[i] = daily_g_ret.iloc[i] * ps['wt'] - cost
        elif ps['wt'] > 0 and ps['dir'] == 'value':
            strat_ret.iloc[i] = daily_v_ret.iloc[i] * ps['wt'] - cost
        else:
            strat_ret.iloc[i] = -cost  # 空仓但产生了切换成本
        total_cost_sum += cost
        nav.iloc[i] = nav.iloc[i-1] * (1 + strat_ret.iloc[i])
    current_nav = nav.iloc[-1]
    profit_pct = (current_nav / INIT_CAP - 1) * 100
    print(f'  起点: {START_DATE.date()}  初始资金: {INIT_CAP}  →  当前: {current_nav:.2f}  ({profit_pct:+.2f}%)')
    print(f'  累计交易成本: {total_cost_sum*INIT_CAP:.2f}元')

    # 4. 最新信号
    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) >= 2 else last
    dir_now = last['dir']; dir_prev = prev['dir']
    wt_now = last['wt']; wt_prev = prev['wt']
    
    # 建议文本 - 直接写明操作
    target = f'{NAME_G}({CODE_G})' if dir_now=='growth' else f'{NAME_V}({CODE_V})' if dir_now=='value' else '空仓'
    if wt_now == 0:
        advice = f'空仓 0%'
    elif dir_now != dir_prev:
        old_target = f'{NAME_G}({CODE_G})' if dir_prev=='growth' else f'{NAME_V}({CODE_V})'
        advice = f'切换：卖出 {old_target}，买入 {target}，仓位 {wt_now*100:.0f}%'
    elif wt_now < wt_prev:
        advice = f'降仓：{target} 从 {wt_prev*100:.0f}% 降低至 {wt_now*100:.0f}%'
    else:
        hold_text = '买入' if wt_now >= 1.0 else '持有'
        advice = f'{hold_text} {target}，仓位 {wt_now*100:.0f}%'
    
    latest = {
        'date': df.index[-1],
        'dir': dir_now,
        'wt': wt_now,
        'T': last['T'],
        'slope': last['slope'],
        'dev_pct': last['dev'] * 100,
        'target': target,
        'advice': advice,
        'nav': current_nav,
        'profit_pct': profit_pct,
        'init_cap': INIT_CAP,
    }
    
    print(f'  最新信号: {latest["dir"]}  仓位: {latest["wt"]*100:.0f}%  T={last["T"]:+.3f}')
    
    # 5. 调仓记录
    print('\n生成调仓记录...')
    trades = []
    dir_map = {'growth': NAME_G, 'value': NAME_V, 'cash': '空仓'}
    for i in range(1, len(df)):
        ps = df.iloc[i-1]; cs = df.iloc[i]
        if ps['dir'] != cs['dir'] or abs(ps['wt'] - cs['wt']) > 0.01:
            action = '切换' if ps['dir'] != cs['dir'] else '调仓'
            trades.append({
                'date': df.index[i].strftime('%Y-%m-%d'),
                'from': f"{dir_map.get(ps['dir'],ps['dir'])} {ps['wt']*100:.0f}%",
                'to': f"{dir_map.get(cs['dir'],cs['dir'])} {cs['wt']*100:.0f}%",
                'action': action,
            })
    print(f'  ✅ 共 {len(trades)} 笔调仓记录')

    # 6. 生成图表
    print('\n生成图表...')
    chart_b64 = make_chart(df, PARAMS['rt'], nav, START_DATE, INIT_CAP)
    print('  ✅ 图表生成完成')
    
    # 7. 生成HTML
    print('\n生成HTML...')
    html = make_html(latest, df, chart_b64, trades)
    
    # 6. 输出
    out_dir = os.path.join(os.path.dirname(__file__) or '.', 'docs')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'index.html')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f'  ✅ 已保存: {out_path}')
    print(f'\n完成! 打开 {out_path} 查看信号')
    print('=' * 50)

if __name__ == '__main__':
    main()
