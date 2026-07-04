"""数据获取模块测试 - 拉取所有 ETF 和指数列表

测试目标：
  [1] baostock 获取全市场标的列表（query_all_stock）
  [2] 按 code 前缀分类筛选 ETF 和指数
  [3] 拉取每个标的的基本信息（query_stock_basic）
  [4] 保存为 CSV 到 custom/output/data/

ETF 代码前缀：
  - 沪市 ETF：sh.510xxx ~ sh.519xxx
  - 深市 ETF：sz.159xxx

指数代码前缀：
  - 上证指数：sh.000xxx（部分）、sh.950xxx（中证系列）
  - 深证指数：sz.399xxx

运行：
    cd c:\\XuanJLH\\Qbot
    C:\\Users\\28439\\AppData\\Local\\conda\\conda\\envs\\Qbot\\python.exe -m custom.tests.test_data_fetch_etf_index
"""
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import baostock as bs
import pandas as pd


OUTPUT_DIR = Path(__file__).resolve().parents[1] / 'output' / 'data'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def is_etf(code: str) -> bool:
    """判断是否为 ETF（沪市 51x / 深市 159）"""
    if code.startswith('sh.51'):
        return True
    if code.startswith('sz.159'):
        return True
    return False


def is_index(code: str) -> bool:
    """判断是否为指数（上证 000/950 / 深证 399）

    注意：sh.000xxx 里一部分是指数（如 sh.000001 上证指数），
    另一部分是 B 股或老股票。这里按 baostock 的 type 字段判断更准。
    """
    if code.startswith('sz.399'):
        return True
    if code.startswith('sh.000') or code.startswith('sh.950'):
        return True
    return False


def find_latest_trade_date() -> str:
    """查询最近 14 天的交易日历，返回最近一个交易日"""
    today = datetime.now()
    start = (today - timedelta(days=14)).strftime('%Y-%m-%d')
    end = today.strftime('%Y-%m-%d')
    rs = bs.query_trade_dates(start_date=start, end_date=end)
    if rs.error_code != '0':
        # 兜底：返回昨天
        return (today - timedelta(days=1)).strftime('%Y-%m-%d')
    trade_dates = []
    while rs.next():
        row = rs.get_row_data()
        # fields: calendar_date, is_trading_day
        if row[1] == '1':  # is_trading_day
            trade_dates.append(row[0])
    if not trade_dates:
        return (today - timedelta(days=1)).strftime('%Y-%m-%d')
    return trade_dates[-1]


def fetch_all_stocks() -> pd.DataFrame:
    """获取最新交易日全市场标的列表"""
    trade_day = find_latest_trade_date()
    print(f"[1/4] baostock 拉取 {trade_day}（最近交易日）全市场标的列表...")
    rs = bs.query_all_stock(day=trade_day)
    if rs.error_code != '0':
        raise RuntimeError(f"query_all_stock 失败: {rs.error_msg}")

    rows = []
    while rs.next():
        rows.append(rs.get_row_data())
    # query_all_stock fields: code, tradeStatus, code_name
    fields = rs.fields if rs.fields else ['code', 'tradeStatus', 'code_name']
    df = pd.DataFrame(rows, columns=fields)
    print(f"      共获取到 {len(df)} 个标的，字段: {list(df.columns)}")
    return df


def fetch_stock_basic(code: str) -> dict:
    """获取单只标的基本信息（code/code_name/ipoDate/outDate/type/status）"""
    rs = bs.query_stock_basic(code=code)
    if rs.error_code != '0':
        return {'code': code, 'code_name': '', 'type': '', 'ipoDate': '', 'outDate': '', 'status': ''}
    while rs.next():
        row = rs.get_row_data()
        return dict(zip(rs.fields, row))
    return {'code': code, 'code_name': '', 'type': '', 'ipoDate': '', 'outDate': '', 'status': ''}


def enrich_with_basic(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """对列表补充基本信息（type/status/ipoDate）"""
    print(f"      拉取 {label} 基本信息中（共 {len(df)} 个）...")
    basics = []
    for i, code in enumerate(df['code'].tolist(), 1):
        info = fetch_stock_basic(code)
        basics.append(info)
        if i % 100 == 0:
            print(f"        进度 {i}/{len(df)}")
    basic_df = pd.DataFrame(basics)
    # 合并：以原 df 为准，补充 basic 信息
    result = df.merge(basic_df, on='code', how='left', suffixes=('', '_basic'))
    # 若 basic 里有 code_name，优先用 basic 的
    if 'code_name_basic' in result.columns:
        result['code_name'] = result['code_name_basic'].fillna(result['code_name'])
        result = result.drop(columns=['code_name_basic'])
    return result


def main():
    print("=" * 70)
    print("数据获取模块测试 - ETF 和指数列表")
    print("=" * 70)

    # 登录 baostock
    lg = bs.login()
    if lg.error_code != '0':
        print(f"baostock 登录失败: {lg.error_msg}")
        return 1
    print(f"baostock 登录成功")

    try:
        # 1. 拉全市场标的
        all_stocks = fetch_all_stocks()

        # 2. 分类筛选
        print(f"[2/4] 按代码前缀筛选 ETF 和指数...")
        all_stocks['is_etf'] = all_stocks['code'].apply(is_etf)
        all_stocks['is_index'] = all_stocks['code'].apply(is_index)

        etf_df = all_stocks[all_stocks['is_etf']].copy().reset_index(drop=True)
        index_df = all_stocks[all_stocks['is_index']].copy().reset_index(drop=True)
        print(f"      ETF: {len(etf_df)} 个")
        print(f"      指数: {len(index_df)} 个")

        # 3. 拉基本信息
        print(f"[3/4] 拉取基本信息...")
        etf_enriched = enrich_with_basic(etf_df, 'ETF')
        index_enriched = enrich_with_basic(index_df, '指数')

        # 4. 保存
        print(f"[4/4] 保存到 {OUTPUT_DIR}")
        etf_path = OUTPUT_DIR / f'etf_list_{datetime.now().strftime("%Y%m%d")}.csv'
        index_path = OUTPUT_DIR / f'index_list_{datetime.now().strftime("%Y%m%d")}.csv'

        etf_enriched.to_csv(etf_path, index=False, encoding='utf-8-sig')
        index_enriched.to_csv(index_path, index=False, encoding='utf-8-sig')

        # 汇总
        print()
        print("=" * 70)
        print("测试结果")
        print("=" * 70)
        print(f"ETF 总数:  {len(etf_enriched)}")
        print(f"  - 沪市 ETF (sh.51x): {(etf_enriched['code'].str.startswith('sh.51')).sum()}")
        print(f"  - 深市 ETF (sz.159): {(etf_enriched['code'].str.startswith('sz.159')).sum()}")
        print(f"  保存到: {etf_path}")
        print()
        print(f"指数总数:  {len(index_enriched)}")
        print(f"  - 上证指数 (sh.000/950): {(index_enriched['code'].str.startswith(('sh.000', 'sh.950'))).sum()}")
        print(f"  - 深证指数 (sz.399):     {(index_enriched['code'].str.startswith('sz.399')).sum()}")
        print(f"  保存到: {index_path}")
        print()
        print("-" * 70)
        print("ETF 样本（前 10）:")
        print(etf_enriched[['code', 'code_name', 'type', 'status']].head(10).to_string(index=False))
        print()
        print("指数样本（前 10）:")
        print(index_enriched[['code', 'code_name', 'type', 'status']].head(10).to_string(index=False))
        print("=" * 70)
        return 0
    finally:
        bs.logout()


if __name__ == '__main__':
    sys.exit(main())
