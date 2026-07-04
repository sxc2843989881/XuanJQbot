"""从 cnindex.com.cn 拉取成长100(480080)和价值100(480081)最新数据"""
import urllib.request
import json
import csv
from datetime import datetime

def fetch_index_data(code, start_date='2013-01-01', end_date=None):
    if end_date is None:
        end_date = datetime.today().strftime('%Y-%m-%d')
    
    url = (f'https://hq.cnindex.com.cn/market/market/getIndexDailyDataWithDataFormat'
           f'?indexCode={code}&startDate={start_date}&endDate={end_date}&frequency=DAY')
    
    print(f"  请求: {code}, {start_date} ~ {end_date}")
    
    req = urllib.request.Request(url, headers={
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'application/json'
    })
    
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw = resp.read().decode('utf-8')
    
    data = json.loads(raw)
    
    # cnindex API 返回结构: data['data']['data'] = rows, data['data']['item'] = columns
    inner = data.get('data', {})
    rows = inner.get('data', [])
    items = inner.get('item', [])
    
    print(f"  列: {items}")
    print(f"  获取到 {len(rows)} 条记录")
    
    if not rows:
        print(f"  无数据, 响应: {str(data)[:200]}")
        return None
    
    # 列索引: 0=timestamp, 3=open, 5=close
    return rows, 0, 3, 5

def save_to_csv(rows, date_idx, open_idx, close_idx, filepath):
    count = 0
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['date', 'open', 'close'])
        
        for row in rows:
            ds = str(row[date_idx]) if row[date_idx] else ''
            op = row[open_idx] if len(row) > open_idx and row[open_idx] else ''
            cl = row[close_idx] if len(row) > close_idx and row[close_idx] else ''
            if ds and cl:
                # 确保日期格式为 YYYY-MM-DD
                if ds.isdigit() and len(ds) == 8:
                    ds = f'{ds[:4]}-{ds[4:6]}-{ds[6:8]}'
                writer.writerow([ds, op, cl])
                count += 1
    
    print(f"  已保存 {count} 条记录到 {filepath}")
    return count

# ============================================================
# 主流程
# ============================================================
print("=" * 70)
print("  从 cnindex 获取成长100/价值100 指数数据")
print("=" * 70)

print("\n[成长100 480080]")
result1 = fetch_index_data('480080')
if result1:
    rows1, d_idx, o_idx, c_idx = result1
    save_to_csv(rows1, d_idx, o_idx, c_idx, r'c:\temp_v72_data\index_480080_web.csv')

print("\n[价值100 480081]")
result2 = fetch_index_data('480081')
if result2:
    rows2, d_idx, o_idx, c_idx = result2
    save_to_csv(rows2, d_idx, o_idx, c_idx, r'c:\temp_v72_data\index_480081_web.csv')

print("\n" + "=" * 70)
print("  完成!")
print("=" * 70)
