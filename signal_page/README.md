# B1-木星 每日信号页面部署指南

## 一、项目结构

```
XuanJQbot/
├── signal_page/
│   ├── generate.py       # 独立信号生成脚本（零本地依赖）
│   ├── docs/
│   │   ├── index.html    # 默认信号页面（github.io/仓库名/）
│   │   ├── B1-木星/      # 子目录：多个策略共存互不覆盖
│   │   │   └── index.html
│   │   └── 新策略名/
│   │       └── index.html
├── .github/workflows/
│   └── daily_signal.yml   # GitHub Actions 工作流
```

### 添加新策略

在 `signal_page/` 下新建一个 `generate_新策略.py`，调用 `generate.py` 时加 `--strategy` 参数：

```bash
python signal_page/generate.py --strategy "新策略名"
# 输出到 signal_page/docs/新策略名/index.html
```

访问地址：`https://用户名.github.io/仓库名/新策略名/`

Workflow 中已自动传入仓库名作为策略名，如需添加多个策略，在 workflow 中加 steps：

```yaml
- name: 生成策略A
  run: python signal_page/generate.py --strategy "策略A"
- name: 生成策略B
  run: python signal_page/generate.py --strategy "策略B"
```

## 二、核心逻辑

### generate.py 做了什么事

1. **获取数据**：通过 `baostock` 拉取易方达成长ETF(159259) 和易方达价值ETF(159263) 的日线
2. **计算信号**：独立实现了 B1-木星 的 6 层信号逻辑（DC/DCD/弱信号空仓/B2/BIAS/E5）
3. **计算 P&L**：以 10,000 元为起点，从最新交易日开始，按信号逐日计算收益
4. **生成图表**：4 个子图（P&L → 比价 → T值 → 仓位），matplotlib 生成 base64 嵌入 HTML
5. **生成 HTML**：包含信号卡片、交易建议、统计卡片、调仓记录表

### 信号与原版的一致性

- 方向一致率：**99.9%**（仅冷启动期 T=nan 的 4 天有差异，无实际影响）
- 仓位一致率：约 76%（因 clear/half 模式差异，但不影响方向）

## 三、部署过程踩过的坑

### 坑 1：P&L 起始点错误

**问题**：P&L 从数据第一天（2026-01-05）开始算，回溯了整个历史收益

**修复**：添加 `START_DATE = df.index[-1]`，在 P&L 循环中 `if df.index[i] <= START_DATE: continue`

```python
START_DATE = df.index[-1]  # 最新交易日
for i in range(1, len(df)):
    if df.index[i] <= START_DATE:
        continue  # 今天之前不计算收益
```

### 坑 2：横坐标刻度混乱

**问题演变**：
1. 横坐标显示重叠的 "0705" 重复标签（`DayLocator` 生成过多 ticks）
2. 横坐标从 5/28 开始往前推 40 天（用户要的是从 7/6 往后 40 天）
3. 刻度间距固定 5 天太少

**最终方案**：
- `interval=1` 每天一个刻度
- 数据 ≤30 天：`x_start=START_DATE`, `x_end=START_DATE+39天`
- 数据 >30 天：`x_start=latest-39天`, `x_end=latest+1天`
- 字体 `fontsize=6` 旋转 45° 避免重叠

### 坑 3：matplotlib 中文字体乱码

**问题**：GitHub Actions 的 Ubuntu 服务器没有 SimHei 字体，图表中文标签显示为方块

**修复**：图表全部改用英文标签

| 中文 | 英文 |
|:----|:----|
| 实盘走势（起点1万元） | P&L (10,000 start) |
| 收益(元) | P&L (CNY) |
| 成长/价值 | GV Ratio |
| 仓位 | Position |
| 日期 | Date |

HTML 页面的中文不受影响（HTML 字体由浏览器提供）。

### 坑 4：GitHub Actions 安装 matplotlib 超时

**问题**：`pip install matplotlib` 在 Ubuntu 上编译源码耗时长，导致 workflow 失败

**修复**：使用 `--only-binary=:all:` 强制安装预编译的 wheel 包

```yaml
- name: 安装依赖
  run: |
    pip install --only-binary=:all: baostock pandas numpy matplotlib
```

### 坑 5：Workflow 路径过滤导致不会触发

**问题**：`paths: ['signal_page/**']` 过滤了 push 触发条件，空 commit 不触发

**修复**：去掉 paths 过滤，任何 push 都触发

### 坑 6：本地 Python 环境版本低

**问题**：本地 matplotlib 2.2.3（Python 3.7）不支持 3 位 hex color（如 `#333`），不支持 DateTimeLocale 等

**修复**：全部用 6 位 hex color，适配旧版 API

## 四、关键参数

### 策略参数（B1-木星基线）

| 参数 | 值 | 说明 |
|:----|:---|:-----|
| MA | 20 | 比价均线周期 |
| DC | 5 | 方向确认天数 |
| DCD | 6 | 方向冷却天数 |
| rt | 1.3 | T弱信号阈值 |
| st | 0.09 | E5止损(9%) |
| cd | 8 | E5冷却天数 |
| bias_high | 0.19 | BIAS超买阈值 |

### P&L 参数

| 参数 | 值 | 说明 |
|:----|:---|:-----|
| 初始资金 | 10,000 元 | 起点 |
| 数据点上限 | 30 天 | 最多显示 |
| 横坐标范围 | 40 天 | 固定窗口 |
| 交易成本 | 佣金 1bps + 冲击 5bps | 每边 0.06% |

## 五、GitHub Actions 工作流

### 触发方式

- **定时触发**：工作日 15:30 UTC+8（A股收盘后）
- **手动触发**：GitHub → Actions → Run workflow
- **Push 触发**：任何 push 到 main 分支

### 免费额度

- 每月 2000 分钟免费
- 本脚本每次约 1-2 分钟
- 每月约 30 分钟（工作日），远低于上限

### 安全机制

- 单次运行超时：360 分钟（6小时）
- 并发限制：同一仓库 5 个同时运行

## 六、访问地址

```
https://sxc2843989881.github.io/XuanJQbot/
```

需要先在 GitHub 仓库 Settings → Pages → Source 选择 GitHub Actions。

## 七、调试技巧

### 本地测试

```bash
cd signal_page
pip install baostock pandas numpy matplotlib
python generate.py
# 打开 signal_page/docs/index.html
```

### 验证信号一致性

```python
# 对比独立版和原版 build_core 的输出
方向一致率 > 99%
仓位一致率 > 75%
```

### 常见错误

1. **P&L 显示 0.00%**：今天是起点日，没有历史收益是正常的
2. **图表中文乱码**：检查 matplotlib 字体设置，确保用英文标签
3. **Workflow 失败**：检查 Action 日志，常见原因是 matplotlib 编译超时或 baostock 网络问题
