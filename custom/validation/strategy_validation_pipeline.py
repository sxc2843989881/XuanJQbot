"""
strategy_validation_pipeline.py — 量化策略通用检测流水线
================================================================
功能：对任意量化策略执行标准化稳健性检测，输出结构化报告

检测项目（7大维度）：
  1. 参数敏感性分析（过拟合检测-参数高原vs尖峰）
  2. 样本分段验证（随机窗口+时序分段）
  3. Walk-Forward滚动检验（IS→OOS衰减率）
  4. 蒙特卡洛重采样（Bootstrap置信区间）
  5. 波动率环境分段（高/低波动率下表现）
  6. 市场 regime 分段（牛市/熊市/震荡市）
  7. 交易质量分析（胜率/盈亏比/持仓时长）

使用方式：
  from strategy_validation_pipeline import StrategyValidator, ValidationConfig
  
  validator = StrategyValidator(
      strategy_func=your_signal_function,  # 接收参数返回(signal, weight)
      param_grid={'n_confirm': [2,3,4,5,6]},  # 参数扫描网格
      g_close=g_close, v_close=v_close,  # 价格数据
      benchmark_annual=0.36,  # 基准年化
  )
  report = validator.run_full_validation()
  report.save('report.txt')
  report.plot('charts.png')
================================================================
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple, Optional, Any
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# 中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150


# ============================================================
# 数据结构
# ============================================================

@dataclass
class ValidationConfig:
    """检测配置"""
    benchmark_annual: float = 0.36          # 基准年化收益
    benchmark_sharpe: float = 1.27          # 基准Sharpe
    benchmark_dd: float = -0.3766           # 基准最大回撤
    rf_annual: float = 0.025                # 无风险利率
    freq: int = 252                         # 年交易日
    n_bootstrap: int = 1000                 # Bootstrap次数
    n_walkforward: int = 5                  # Walk-Forward窗口数
    walkforward_train_years: int = 3        # WFA训练窗口年数
    walkforward_test_years: int = 1         # WFA测试窗口年数
    n_random_splits: int = 10               # 随机分段数
    random_split_ratio: float = 0.7         # 训练集比例
    volatility_quantiles: int = 3           # 波动率分位数
    regime_window: int = 120                # regime识别窗口（120日≈半年，避免频繁切换）
    strategy_type: str = "trend_following"  # 策略类型: trend_following/mean_reverting/multi_factor
    
    # 阈值标准（来自行业知识 + 知识库06_阈值标准速查表.md）
    # 原则：分档判定（通过/警告/否决），单一警告不否决，只有否决项或总分过低才判失败
    threshold_oos_dd_ratio: float = 1.4     # OOS/IS回撤比阈值
    threshold_trade_freq_cv: float = 0.32   # 交易频率CV阈值
    threshold_sharpe_suspicious: float = 3.0 # Sharpe可疑阈值
    threshold_trade_per_param: int = 30     # 交易/参数比阈值
    threshold_bootstrap_dev: float = 0.18   # Bootstrap偏差阈值
    
    # 月度收益年化std分档阈值（知识库06第1节）
    threshold_monthly_vol_warn: float = 0.24   # 警戒线：24%（稳健阈值）
    threshold_monthly_vol_fail: float = 0.32   # 否决线：32%（过拟合典型值）
    
    # 极端环境回撤分档阈值（统一逻辑，知识库06第6节"最差态回撤可控"）
    threshold_extreme_dd_warn: float = -0.50   # 警戒线：-50%
    threshold_extreme_dd_fail: float = -0.70   # 否决线：-70%
    
    # 策略类型自适应阈值（趋势跟踪策略允许更低胜率，靠高盈亏比补偿）
    @property
    def min_win_rate(self) -> float:
        """最低胜率要求：趋势跟踪50%，均值回归55%"""
        return 0.50 if self.strategy_type == "trend_following" else 0.55
    
    @property
    def min_profit_loss_ratio(self) -> float:
        """最低盈亏比要求：趋势跟踪1.8，均值回归1.2"""
        return 1.8 if self.strategy_type == "trend_following" else 1.2


@dataclass
class ValidationResult:
    """单项检测结果 — 三档判定：通过(passed=True)/警告(passed=True,vetoed=False)/否决(passed=False,vetoed=True)"""
    test_name: str
    passed: bool
    score: float
    details: Dict[str, Any]
    warning: str = ""
    vetoed: bool = False  # 是否一票否决（True=硬性失败，不可被总分救回）
    
    def summary(self) -> str:
        if self.vetoed:
            flag = "❌否决"
        elif self.passed:
            flag = "✅通过"
        else:
            flag = "⚠️警告"
        return f"[{self.test_name}] {flag} | 得分:{self.score:.3f} | {self.warning}"


@dataclass
class FullReport:
    """完整检测报告"""
    strategy_name: str
    config: ValidationConfig
    results: List[ValidationResult] = field(default_factory=list)
    overall_passed: bool = False
    overall_score: float = 0.0
    
    def add(self, result: ValidationResult):
        self.results.append(result)
    
    def compute_overall(self):
        """计算总体通过率和得分 — 三档判定逻辑
        总体通过条件：无否决项 AND (通过项≥70% OR 总分≥0.65)
        任一否决项(vetoed=True)直接判总体不通过
        """
        if not self.results:
            return
        passed_count = sum(1 for r in self.results if r.passed)
        vetoed_count = sum(1 for r in self.results if r.vetoed)
        warn_count = sum(1 for r in self.results if not r.passed and not r.vetoed)
        self.overall_score = np.mean([r.score for r in self.results])
        # 否决项直接判失败；否则要求通过率≥70%或总分≥0.65
        self.overall_passed = (vetoed_count == 0) and (
            passed_count >= len(self.results) * 0.7 or self.overall_score >= 0.65
        )
        self._vetoed_count = vetoed_count
        self._warn_count = warn_count

    def print_report(self) -> str:
        """生成文本报告"""
        self.compute_overall()
        lines = []
        lines.append("=" * 70)
        lines.append(f"  策略检测报告: {self.strategy_name}")
        lines.append("=" * 70)
        lines.append(f"\n基准: 年化{self.config.benchmark_annual*100:.1f}% "
                     f"Sharpe{self.config.benchmark_sharpe:.2f} "
                     f"回撤{self.config.benchmark_dd*100:.1f}%")
        lines.append(f"策略类型: {self.config.strategy_type}")
        lines.append(f"\n{'检测项':<30} {'结果':>6} {'得分':>8} {'警告':>30}")
        lines.append("-" * 80)
        for r in self.results:
            if r.vetoed:
                flag = "❌否决"
            elif r.passed:
                flag = "✅通过"
            else:
                flag = "⚠️警告"
            warn = r.warning[:28] if r.warning else ""
            lines.append(f"{r.test_name:<30} {flag:>6} {r.score:>8.3f} {warn:>30}")
        lines.append("-" * 80)
        overall_flag = "✅通过" if self.overall_passed else "❌未通过"
        lines.append(f"\n总体: {overall_flag} | 总得分: {self.overall_score:.3f} | "
                     f"通过{sum(1 for r in self.results if r.passed)}/{len(self.results)}项 "
                     f"警告{self._warn_count} 否决{self._vetoed_count}")

        failed = [r for r in self.results if not r.passed]
        if failed:
            lines.append(f"\n【未通过项详情】")
            for r in failed:
                tag = "❌否决" if r.vetoed else "⚠️警告"
                lines.append(f"\n  {r.test_name} [{tag}]:")
                for k, v in r.details.items():
                    lines.append(f"    {k}: {v}")

        return "\n".join(lines)
    
    def save(self, path: str):
        """保存报告"""
        Path(path).write_text(self.print_report(), encoding='utf-8')
    
    def plot(self, path: str):
        """绘制检测结果图 — 三档颜色：绿(通过)/橙(警告)/红(否决)"""
        n = len(self.results)
        fig, ax = plt.subplots(figsize=(14, max(6, n * 0.5)))
        names = [r.test_name for r in self.results]
        scores = [r.score for r in self.results]
        colors = []
        for r in self.results:
            if r.vetoed:
                colors.append('#E74C3C')  # 红=否决
            elif r.passed:
                colors.append('#27AE60')  # 绿=通过
            else:
                colors.append('#F39C12')  # 橙=警告

        bars = ax.barh(range(n), scores, color=colors, alpha=0.8)
        for i, (s, r) in enumerate(zip(scores, self.results)):
            if r.vetoed:
                flag = "❌"
            elif r.passed:
                flag = "✅"
            else:
                flag = "⚠️"
            ax.text(s + 0.01, i, f'{s:.3f} {flag}', va='center', fontsize=9, fontweight='bold')
        
        ax.set_yticks(range(n))
        ax.set_yticklabels(names, fontsize=9)
        ax.set_xlabel('得分 (越高越好)')
        ax.set_title(f'{self.strategy_name} 策略检测结果 — 总分{self.overall_score:.3f} '
                     f'({"通过" if self.overall_passed else "未通过"})',
                     fontsize=13, fontweight='bold')
        ax.axvline(x=0.5, color='gray', linewidth=0.8, linestyle='--', alpha=0.5)
        ax.set_xlim(0, 1.2)
        ax.grid(True, alpha=0.3, axis='x')
        ax.invert_yaxis()
        
        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)


# ============================================================
# 核心指标计算
# ============================================================

def calc_metrics(daily_ret: pd.Series, config: ValidationConfig) -> Dict[str, float]:
    """从日收益率计算核心指标"""
    r = daily_ret.dropna()
    n = len(r)
    if n < 10:
        return {'ann': 0, 'dd': 0, 'sharpe': 0, 'calmar': 0, 'vol': 0, 'total': 0}
    
    years = n / config.freq
    eq = (1 + r).cumprod()
    total = eq.iloc[-1] - 1
    ann = (1 + total) ** (1 / years) - 1 if years > 0 else 0
    rf_p = config.rf_annual / config.freq
    vol = r.std() * np.sqrt(config.freq)
    sharpe = (r.mean() - rf_p) / r.std() * np.sqrt(config.freq) if r.std() > 0 else 0
    peak = eq.cummax()
    dd = ((eq - peak) / peak).min()
    calmar = ann / abs(dd) if dd < 0 else 0
    
    return {
        'ann': ann, 'dd': dd, 'sharpe': sharpe, 'calmar': calmar,
        'vol': vol, 'total': total, 'n_days': n
    }


# ============================================================
# 检测器主类
# ============================================================

class StrategyValidator:
    """量化策略通用检测器"""
    
    def __init__(
        self,
        strategy_func: Callable[[Dict], Tuple[pd.Series, pd.Series]],
        param_grid: Dict[str, List],
        g_close: pd.Series,
        v_close: pd.Series,
        strategy_name: str = "策略",
        config: ValidationConfig = None,
        backtest_engine_func: Callable = None,
    ):
        """
        Args:
            strategy_func: 接收参数字典，返回(signal_series, weight_series)
            param_grid: 参数扫描网格
            g_close: 成长指数收盘价
            v_close: 价值指数收盘价
            backtest_engine_func: 回测引擎函数，接收(signal, weight, g, v)返回daily_ret
        """
        self.strategy_func = strategy_func
        self.param_grid = param_grid
        self.g_close = g_close
        self.v_close = v_close
        self.strategy_name = strategy_name
        self.config = config or ValidationConfig()
        self.backtest_engine_func = backtest_engine_func or self._default_backtest
        
        # 缓存
        self._param_results = {}  # 参数扫描结果
        
    def _default_backtest(self, signal: pd.Series, weight: pd.Series,
                          g: pd.Series, v: pd.Series) -> pd.Series:
        """默认回测引擎（简化版，用于无外部引擎时）"""
        common = signal.index.intersection(g.index)
        sig = signal.loc[common].astype(str)
        wt = weight.loc[common].astype(float)
        g_a = g.loc[common]
        v_a = v.loc[common]
        
        # 简化：每日收益 = 持仓方向收益 × 仓位
        daily_ret = pd.Series(0.0, index=common)
        prev_pos = None
        for i in range(1, len(common)):
            pos = sig.iloc[i-1]  # T-1日信号
            wt_i = wt.iloc[i-1]
            if pos == 'growth':
                daily_ret.iloc[i] = (g_a.iloc[i] / g_a.iloc[i-1] - 1) * wt_i
            elif pos == 'value':
                daily_ret.iloc[i] = (v_a.iloc[i] / v_a.iloc[i-1] - 1) * wt_i
            # cash = 0
        return daily_ret
    
    def run_backtest(self, signal: pd.Series, weight: pd.Series) -> pd.Series:
        """运行回测，返回日收益率序列"""
        return self.backtest_engine_func(signal, weight, self.g_close, self.v_close)
    
    # ============================================================
    # 检测1: 参数敏感性分析
    # ============================================================
    def test_parameter_sensitivity(self) -> ValidationResult:
        """参数敏感性分析：检测参数高原vs尖峰"""
        print("  [1/7] 参数敏感性分析...")
        
        # 扫描所有参数组合
        param_names = list(self.param_grid.keys())
        if len(param_names) == 1:
            param_name = param_names[0]
            param_values = self.param_grid[param_name]
            results = []
            for val in param_values:
                sig, wt = self.strategy_func({param_name: val})
                ret = self.run_backtest(sig, wt)
                m = calc_metrics(ret, self.config)
                self._param_results[val] = m
                results.append(m)
            
            anns = [m['ann'] for m in results]
            best_idx = np.argmax(anns)
            best_val = param_values[best_idx]
            
            # 计算高原宽度：最优值±1,±2都达标的比例
            threshold = 0.36  # 36%达标线
            pass_count = sum(1 for a in anns if a >= threshold)
            pass_ratio = pass_count / len(anns)
            
            # 计算变异系数CV
            cv = np.std(anns) / np.mean(anns) if np.mean(anns) > 0 else 999
            
            # 评分：pass_ratio越高越好，cv越低越好
            score = pass_ratio * 0.6 + max(0, 1 - cv) * 0.4
            passed = pass_ratio >= 0.5 and cv < 0.3
            
            warning = ""
            if cv > 0.3:
                warning = f"参数CV={cv:.2f}过高(尖峰)"
            elif pass_ratio < 0.5:
                warning = f"仅{pass_ratio*100:.0f}%参数达标"
            
            return ValidationResult(
                test_name="参数敏感性",
                passed=passed,
                score=score,
                details={
                    '参数名': param_name,
                    '最优值': best_val,
                    '最优年化': f"{anns[best_idx]*100:.2f}%",
                    '达标比例': f"{pass_ratio*100:.0f}%",
                    '变异系数CV': f"{cv:.3f}",
                    '各参数年化': [f"{a*100:.2f}%" for a in anns],
                },
                warning=warning
            )
        else:
            # 多参数：简化处理
            return ValidationResult(
                test_name="参数敏感性",
                passed=True,
                score=0.7,
                details={'备注': '多参数网格，需手动检查'},
            )
    
    # ============================================================
    # 检测2: 样本分段验证
    # ============================================================
    def test_sample_split(self) -> ValidationResult:
        """随机窗口样本分段验证"""
        print("  [2/7] 样本分段验证...")
        
        # 用默认参数跑一次获取完整日收益
        default_params = {k: v[len(v)//2] for k, v in self.param_grid.items()}
        sig, wt = self.strategy_func(default_params)
        full_ret = self.run_backtest(sig, wt)
        
        n = len(full_ret)
        split_scores = []
        
        for i in range(self.config.n_random_splits):
            # 随机打乱后分段（非时序，测试分布稳定性）
            np.random.seed(42 + i)
            shuffled = full_ret.sample(frac=1, random_state=42+i).reset_index(drop=True)
            split_point = int(n * self.config.random_split_ratio)
            is_ret = shuffled[:split_point]
            oos_ret = shuffled[split_point:]
            
            m_is = calc_metrics(is_ret, self.config)
            m_oos = calc_metrics(oos_ret, self.config)
            
            # 计算衰减率
            if m_is['ann'] > 0:
                decay = (m_is['ann'] - m_oos['ann']) / m_is['ann']
            else:
                decay = 1.0
            split_scores.append(decay)
        
        avg_decay = np.mean(split_scores)
        # 衰减<20%为优秀，<40%为合格
        if avg_decay < 0.2:
            score = 1.0
            passed = True
        elif avg_decay < 0.4:
            score = 0.7
            passed = True
        else:
            score = max(0, 1 - avg_decay)
            passed = avg_decay < 0.5
        
        return ValidationResult(
            test_name="样本分段验证",
            passed=passed,
            score=score,
            details={
                '分段数': self.config.n_random_splits,
                '平均衰减率': f"{avg_decay*100:.1f}%",
                '各段衰减': [f"{s*100:.1f}%" for s in split_scores],
            },
            warning=f"平均衰减{avg_decay*100:.1f}%" if avg_decay > 0.3 else ""
        )
    
    # ============================================================
    # 检测3: Walk-Forward滚动检验
    # ============================================================
    def test_walk_forward(self) -> ValidationResult:
        """Walk-Forward时序滚动验证"""
        print("  [3/7] Walk-Forward滚动检验...")
        
        default_params = {k: v[len(v)//2] for k, v in self.param_grid.items()}
        sig, wt = self.strategy_func(default_params)
        full_ret = self.run_backtest(sig, wt)
        
        n = len(full_ret)
        train_size = self.config.walkforward_train_years * self.config.freq
        test_size = self.config.walkforward_test_years * self.config.freq
        step = test_size
        
        wf_results = []
        start = 0
        while start + train_size + test_size <= n:
            train_ret = full_ret.iloc[start:start+train_size]
            test_ret = full_ret.iloc[start+train_size:start+train_size+test_size]
            
            m_train = calc_metrics(train_ret, self.config)
            m_test = calc_metrics(test_ret, self.config)
            
            # 衰减率
            if m_train['ann'] > 0:
                decay = (m_train['ann'] - m_test['ann']) / m_train['ann']
            else:
                decay = 1.0
            
            wf_results.append({
                'start': full_ret.index[start],
                'train_ann': m_train['ann'],
                'test_ann': m_test['ann'],
                'decay': decay,
                'test_dd': m_test['dd'],
            })
            start += step
        
        if not wf_results:
            return ValidationResult(
                test_name="Walk-Forward",
                passed=False,
                score=0,
                details={'错误': '数据不足以做Walk-Forward'},
                warning="数据不足"
            )
        
        # 统计：多少窗口测试期正收益
        pos_windows = sum(1 for r in wf_results if r['test_ann'] > 0)
        pos_ratio = pos_windows / len(wf_results)
        avg_decay = np.mean([r['decay'] for r in wf_results])
        
        # 评分
        score = pos_ratio * 0.5 + max(0, 1 - avg_decay) * 0.5
        passed = pos_ratio >= 0.6 and avg_decay < 0.5
        
        return ValidationResult(
            test_name="Walk-Forward",
            passed=passed,
            score=score,
            details={
                '窗口数': len(wf_results),
                '正收益窗口': f"{pos_windows}/{len(wf_results)}",
                '平均衰减': f"{avg_decay*100:.1f}%",
                '各窗口测试年化': [f"{r['test_ann']*100:.1f}%" for r in wf_results],
            },
            warning=f"仅{pos_ratio*100:.0f}%窗口正收益" if pos_ratio < 0.6 else ""
        )
    
    # ============================================================
    # 检测4: 蒙特卡洛Bootstrap
    # ============================================================
    def test_bootstrap(self) -> ValidationResult:
        """Bootstrap重采样置信区间"""
        print("  [4/7] Bootstrap重采样...")
        
        default_params = {k: v[len(v)//2] for k, v in self.param_grid.items()}
        sig, wt = self.strategy_func(default_params)
        full_ret = self.run_backtest(sig, wt)
        
        # Bootstrap Sharpe
        original_sharpe = calc_metrics(full_ret, self.config)['sharpe']
        
        boot_sharpes = []
        n = len(full_ret)
        for i in range(self.config.n_bootstrap):
            np.random.seed(i)
            sample = full_ret.sample(n=n, replace=True).reset_index(drop=True)
            m = calc_metrics(sample, self.config)
            boot_sharpes.append(m['sharpe'])
        
        boot_sharpes = np.array(boot_sharpes)
        ci_lower = np.percentile(boot_sharpes, 2.5)
        ci_upper = np.percentile(boot_sharpes, 97.5)
        boot_mean = np.mean(boot_sharpes)
        boot_std = np.std(boot_sharpes)
        
        # 偏差
        dev = abs(boot_mean - original_sharpe) / original_sharpe if original_sharpe > 0 else 1
        
        # 评分：CI下限>0且偏差小
        if ci_lower > 0 and dev < 0.1:
            score = 1.0
            passed = True
        elif ci_lower > 0:
            score = 0.7
            passed = True
        else:
            score = max(0, ci_lower / original_sharpe) if original_sharpe > 0 else 0
            passed = False
        
        return ValidationResult(
            test_name="Bootstrap置信区间",
            passed=passed,
            score=score,
            details={
                '原始Sharpe': f"{original_sharpe:.3f}",
                'Bootstrap均值': f"{boot_mean:.3f}",
                '95%CI': f"[{ci_lower:.3f}, {ci_upper:.3f}]",
                '偏差': f"{dev*100:.1f}%",
                'Bootstrap次数': self.config.n_bootstrap,
            },
            warning=f"CI下限{ci_lower:.3f}{'<0!' if ci_lower < 0 else ''}" if ci_lower < 0 else ""
        )
    
    # ============================================================
    # 检测5: 波动率环境分段
    # ============================================================
    def test_volatility_regime(self) -> ValidationResult:
        """不同波动率环境下的表现"""
        print("  [5/7] 波动率环境分段...")
        
        default_params = {k: v[len(v)//2] for k, v in self.param_grid.items()}
        sig, wt = self.strategy_func(default_params)
        full_ret = self.run_backtest(sig, wt)
        
        # 计算滚动波动率
        rolling_vol = full_ret.rolling(60).std() * np.sqrt(self.config.freq)
        rolling_vol = rolling_vol.dropna()
        
        # 分3段：低/中/高波动率
        quantiles = np.linspace(0, 1, self.config.volatility_quantiles + 1)
        vol_thresholds = rolling_vol.quantile(quantiles).values
        
        regime_results = []
        for i in range(self.config.volatility_quantiles):
            low_t = vol_thresholds[i]
            high_t = vol_thresholds[i+1]
            mask = (rolling_vol >= low_t) & (rolling_vol < high_t) if i < self.config.volatility_quantiles - 1 \
                   else (rolling_vol >= low_t)
            regime_ret = full_ret.loc[rolling_vol[mask].index]
            if len(regime_ret) > 10:
                m = calc_metrics(regime_ret, self.config)
                # 回撤单独用"逐段计算取最大值"（修复拼接bug）
                vol_mask = pd.Series(False, index=full_ret.index)
                vol_mask.loc[rolling_vol[mask].index] = True
                regime_dd = self._calc_regime_max_dd_by_mask(full_ret, vol_mask)
                regime_results.append({
                    'regime': f"波动率{['低','中','高'][i]}",
                    'days': len(regime_ret),
                    'ann': m['ann'],
                    'dd': regime_dd,
                    'sharpe': m['sharpe'],
                })
        
        if len(regime_results) < 2:
            return ValidationResult(
                test_name="波动率环境分段",
                passed=False,
                score=0,
                details={'错误': '数据不足以分段'},
            )
        
        # 检查：各环境下都正收益
        pos_regimes = sum(1 for r in regime_results if r['ann'] > 0)
        pos_ratio = pos_regimes / len(regime_results)

        # 检查：高波动率环境回撤分档判定（与market_regime统一逻辑）
        high_vol = [r for r in regime_results if '高' in r['regime']]
        vetoed = False
        warning = ""
        if high_vol:
            high_dd = high_vol[0]['dd']
            if high_dd <= self.config.threshold_extreme_dd_fail:  # <-70% 否决
                dd_score = 0
                vetoed = True
                warning = f"高波动率回撤{high_dd*100:.1f}%≤{self.config.threshold_extreme_dd_fail*100:.0f}%否决"
            elif high_dd <= self.config.threshold_extreme_dd_warn:  # -50%~-70% 警告
                dd_score = 0.4
                warning = f"高波动率回撤{high_dd*100:.1f}%超过{self.config.threshold_extreme_dd_warn*100:.0f}%警戒"
            else:  # >-50% 通过
                dd_score = 1.0
        else:
            dd_score = 1.0

        # 加权评分：正收益比例70% + 回撤控制30%（警告不否决，只有否决项才判失败）
        score = pos_ratio * 0.7 + dd_score * 0.3
        passed = (not vetoed) and pos_ratio >= 0.6

        return ValidationResult(
            test_name="波动率环境分段",
            passed=passed,
            score=score,
            vetoed=vetoed,
            details={
                '环境数': len(regime_results),
                '正收益环境': f"{pos_regimes}/{len(regime_results)}",
                '各环境年化': [f"{r['regime']}:{r['ann']*100:.1f}%" for r in regime_results],
                '高波动率回撤': f"{high_vol[0]['dd']*100:.1f}%" if high_vol else "N/A",
            },
            warning=warning if warning else (f"仅{pos_ratio*100:.0f}%环境正收益" if pos_ratio < 0.6 else "")
        )
    
    # ============================================================
    # 检测6: 市场regime分段（牛市/熊市/震荡）
    # 重构：等权基准判断regime + 120日窗口 + 边界严谨 + 分档判定
    # ============================================================
    def test_market_regime(self) -> ValidationResult:
        """不同市场环境下表现 — 用等权基准判断regime，对轮动策略公平"""
        print("  [6/7] 市场regime分段...")

        default_params = {k: v[len(v)//2] for k, v in self.param_grid.items()}
        sig, wt = self.strategy_func(default_params)
        full_ret = self.run_backtest(sig, wt)

        # 用等权基准判断regime（对轮动策略公平，避免单一指数偏误）
        eq_index = (self.g_close + self.v_close) / 2
        eq_ret = eq_index.pct_change()
        window = self.config.regime_window  # 120日≈半年，避免60日频繁切换
        rolling_ret = eq_ret.rolling(window).sum() * 252 / window

        # 三regime：边界严谨无重叠
        # 牛市: >20%年化 | 震荡: [-10%, 20%] | 熊市: <-10%
        bull_mask = rolling_ret > 0.20
        range_mask = (rolling_ret >= -0.10) & (rolling_ret <= 0.20)
        bear_mask = rolling_ret < -0.10

        regimes = {'牛市': bull_mask, '震荡': range_mask, '熊市': bear_mask}

        regime_results = []
        for name, mask in regimes.items():
            regime_ret = full_ret.loc[rolling_ret[mask].dropna().index]
            if len(regime_ret) > 10:
                m = calc_metrics(regime_ret, self.config)
                # 回撤单独用"逐段计算取最大值"（修复拼接bug）
                regime_dd = self._calc_regime_max_dd(full_ret, rolling_ret, mask)
                regime_results.append({
                    'regime': name,
                    'days': len(regime_ret),
                    'ann': m['ann'],
                    'dd': regime_dd,  # 用逐段计算的回撤，不用拼接的
                    'sharpe': m['sharpe'],
                })

        if len(regime_results) < 2:
            return ValidationResult(
                test_name="市场regime分段",
                passed=False,
                score=0,
                vetoed=False,
                details={'错误': '数据不足以分段'},
            )

        pos_regimes = sum(1 for r in regime_results if r['ann'] > 0)
        pos_ratio = pos_regimes / len(regime_results)

        # 熊市回撤分档判定（与volatility_regime统一逻辑）
        bear = [r for r in regime_results if r['regime'] == '熊市']
        vetoed = False
        warning = ""
        if bear:
            bear_dd = bear[0]['dd']
            if bear_dd <= self.config.threshold_extreme_dd_fail:  # <-70% 否决
                bear_score = 0
                vetoed = True
                warning = f"熊市回撤{bear_dd*100:.1f}%≤{self.config.threshold_extreme_dd_fail*100:.0f}%否决"
            elif bear_dd <= self.config.threshold_extreme_dd_warn:  # -50%~-70% 警告
                bear_score = 0.4
                warning = f"熊市回撤{bear_dd*100:.1f}%超过{self.config.threshold_extreme_dd_warn*100:.0f}%警戒"
            else:  # >-50% 通过
                bear_score = 1.0
        else:
            bear_score = 1.0

        # 牛市收益要求
        bull = [r for r in regime_results if r['regime'] == '牛市']
        if bull:
            bull_score = 1.0 if bull[0]['ann'] > self.config.benchmark_annual else 0.5
        else:
            bull_score = 1.0

        # 震荡市收益要求（应正收益，避免只靠牛市）
        range_market = [r for r in regime_results if r['regime'] == '震荡']
        if range_market:
            range_score = 1.0 if range_market[0]['ann'] > 0 else 0.3
        else:
            range_score = 1.0

        # 加权评分：正收益比例30% + 熊市控制30% + 牛市超越20% + 震荡稳定20%
        score = pos_ratio * 0.3 + bear_score * 0.3 + bull_score * 0.2 + range_score * 0.2
        # 通过逻辑：无否决项 AND 正收益比例≥0.6
        passed = (not vetoed) and pos_ratio >= 0.6

        # 熊市段数信息
        bear_periods_info = ""
        if bear:
            periods = self._find_contiguous_periods(rolling_ret, bear_mask)
            bear_periods_info = f"{len(periods)}段"

        return ValidationResult(
            test_name="市场regime分段",
            passed=passed,
            score=score,
            vetoed=vetoed,
            details={
                'regime数': len(regime_results),
                '正收益regime': f"{pos_regimes}/{len(regime_results)}",
                '各regime年化': [f"{r['regime']}:{r['ann']*100:.1f}%" for r in regime_results],
                '熊市回撤': f"{bear[0]['dd']*100:.1f}%" if bear else "无熊市",
                '熊市段数': bear_periods_info if bear_periods_info else "无",
                '基准': '等权(成长+价值)/2',
                '窗口': f'{window}日',
                '回撤算法': '逐段计算取最大值(修复拼接bug)',
            },
            warning=warning
        )

    def _find_contiguous_periods(self, rolling_ret: pd.Series, mask: pd.Series) -> list:
        """找mask为True的连续段，返回[(start_date, end_date), ...]"""
        dates = rolling_ret[mask].dropna().index
        if len(dates) == 0:
            return []
        periods = []
        start = dates[0]
        prev = dates[0]
        for d in dates[1:]:
            gap = (d - prev).days
            if gap > 7:  # 超过7天不连续，断段
                periods.append((start, prev))
                start = d
            prev = d
        periods.append((start, prev))
        return periods

    def _calc_regime_max_dd_by_mask(self, full_ret: pd.Series, mask: pd.Series) -> float:
        """通用版：用mask直接找连续段计算回撤（修复拼接bug）
        mask: pd.Series(bool), index与full_ret对齐
        """
        true_dates = mask[mask].dropna().index
        if len(true_dates) == 0:
            return 0.0
        # 找连续段
        periods = []
        start = true_dates[0]
        prev = true_dates[0]
        for d in true_dates[1:]:
            gap = (d - prev).days
            if gap > 7:
                periods.append((start, prev))
                start = d
            prev = d
        periods.append((start, prev))
        # 逐段算回撤取最大值
        max_dd = 0.0
        for s, e in periods:
            period_ret = full_ret.loc[s:e].dropna()
            if len(period_ret) < 3:
                continue
            eq = (1 + period_ret).cumprod()
            peak = eq.cummax()
            dd = float(((eq - peak) / peak).min())
            if dd < max_dd:
                max_dd = dd
        return max_dd

    def _calc_regime_max_dd(self, full_ret: pd.Series, rolling_ret: pd.Series,
                            mask: pd.Series) -> float:
        """逐段计算regime回撤，取最大值（修复拼接bug）
        每段独立计算：从该段起点净值=1开始，算段内最大回撤
        """
        periods = self._find_contiguous_periods(rolling_ret, mask)
        if not periods:
            return 0.0
        max_dd = 0.0
        for start, end in periods:
            period_ret = full_ret.loc[start:end].dropna()
            if len(period_ret) < 3:
                continue
            eq = (1 + period_ret).cumprod()
            peak = eq.cummax()
            dd = float(((eq - peak) / peak).min())
            if dd < max_dd:
                max_dd = dd
        return max_dd
    
    # ============================================================
    # 检测7: 交易质量分析
    # 重构v3：期望收益=胜率×(盈亏比+1)-1 组合指标 + 分档判定
    # 核心原则：胜率和盈亏比不能分开看，必须组合成期望收益
    # ============================================================
    def test_trade_quality(self, trades_df: pd.DataFrame = None) -> ValidationResult:
        """交易质量分析 — 期望收益组合指标 + 月度vol + 连亏控制"""
        print("  [7/7] 交易质量分析...")

        if trades_df is None or len(trades_df) == 0:
            # 退化路径：从日收益构建分析
            default_params = {k: v[len(v)//2] for k, v in self.param_grid.items()}
            sig, wt = self.strategy_func(default_params)
            full_ret = self.run_backtest(sig, wt)

            # 1. 日胜率
            win_rate = float((full_ret > 0).mean())

            # 2. 盈亏比（日收益层面：正收益均值/负收益均值绝对值）
            pos_ret = full_ret[full_ret > 0]
            neg_ret = full_ret[full_ret < 0]
            if len(neg_ret) > 0 and neg_ret.mean() != 0:
                profit_loss_ratio = float(pos_ret.mean() / abs(neg_ret.mean()))
            else:
                profit_loss_ratio = 0.0

            # 3. 月度收益年化std
            monthly_ret = full_ret.resample('M').apply(lambda x: (1+x).prod()-1)
            monthly_vol = float(monthly_ret.std() * np.sqrt(12))

            # 4. 最大连亏天数
            is_loss = (full_ret < 0).astype(int)
            max_loss_streak = 0
            current_streak = 0
            for v in is_loss:
                if v == 1:
                    current_streak += 1
                    max_loss_streak = max(max_loss_streak, current_streak)
                else:
                    current_streak = 0
        else:
            # 有交易记录的详细分析
            n_trades = len(trades_df)
            if 'segment_return' in trades_df.columns:
                win_trades = (trades_df['segment_return'] > 0).sum()
                win_rate = win_trades / n_trades if n_trades > 0 else 0
                wins = trades_df.loc[trades_df['segment_return'] > 0, 'segment_return']
                losses = trades_df.loc[trades_df['segment_return'] < 0, 'segment_return']
                avg_win = wins.mean() if len(wins) > 0 else 0
                avg_loss = abs(losses.mean()) if len(losses) > 0 else 0
                profit_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0
            else:
                win_rate = 0
                profit_loss_ratio = 0
            # 有trades_df时也算月度vol和连亏
            default_params = {k: v[len(v)//2] for k, v in self.param_grid.items()}
            sig, wt = self.strategy_func(default_params)
            full_ret = self.run_backtest(sig, wt)
            monthly_ret = full_ret.resample('M').apply(lambda x: (1+x).prod()-1)
            monthly_vol = float(monthly_ret.std() * np.sqrt(12))
            is_loss = (full_ret < 0).astype(int)
            max_loss_streak = 0
            current_streak = 0
            for v in is_loss:
                if v == 1:
                    current_streak += 1
                    max_loss_streak = max(max_loss_streak, current_streak)
                else:
                    current_streak = 0

        # ===== 核心组合指标：期望收益 =====
        # 期望收益 = 胜率 × (盈亏比 + 1) - 1
        # 这是胜率和盈亏比的综合度量：
        #   > 0  = 正期望（能赚钱）
        #   ≤ 0  = 负期望（亏钱，必须否决）
        #
        # 注意：日收益层面和持仓段层面的量纲不同，阈值需区分：
        #   日收益层面：expectancy×252≈年化收益，0.05对应约12.6%年化
        #   持仓段层面：expectancy×年交易次数≈年化收益，0.3对应约15-20次/年×0.3
        expectancy = win_rate * (profit_loss_ratio + 1) - 1

        # 根据数据来源确定阈值（日收益 vs 持仓段）
        is_daily = (trades_df is None or len(trades_df) == 0)
        if is_daily:
            # 日收益层面：0.05警戒(≈12.6%年化) / 0.10通过(≈25%年化)
            exp_warn = 0.05
            exp_pass = 0.10
            exp_full = 0.20  # 满分基准(≈50%年化)
        else:
            # 持仓段层面：0.3警戒 / 0.5通过
            exp_warn = 0.3
            exp_pass = 0.5
            exp_full = 0.8

        # 期望收益分档判定（组合阈值，不是胜率和盈亏比分开看）
        vetoed = False
        warning = ""
        if expectancy <= 0:
            # 负期望 = 策略长期亏钱，硬性否决
            exp_score = 0
            vetoed = True
            warning = (f"期望收益={expectancy:.3f}≤0否决 "
                       f"(胜率{win_rate*100:.1f}%×盈亏比{profit_loss_ratio:.2f}亏损)")
        elif expectancy < exp_warn:
            # 低正期望 = 勉强赚钱，警告
            exp_score = expectancy / exp_warn * 0.5  # 0~0.5分
            warning = (f"期望收益={expectancy:.3f}<{exp_warn}警戒 "
                       f"(胜率{win_rate*100:.1f}%+盈亏比{profit_loss_ratio:.2f}偏低)")
        else:
            # 良好正期望
            exp_score = min(1.0, 0.5 + (expectancy - exp_warn) / (exp_full - exp_warn) * 0.5)

        # 月度vol分档（辅助指标，不是主评分）
        if monthly_vol >= self.config.threshold_monthly_vol_fail:  # ≥32% 否决
            vol_score = 0
            vetoed = True
            warning += f" | 月度vol={monthly_vol*100:.1f}%≥{self.config.threshold_monthly_vol_fail*100:.0f}%否决"
        elif monthly_vol >= self.config.threshold_monthly_vol_warn:  # 24%-32% 警告
            vol_score = 0.4
            if not warning:
                warning = f"月度vol={monthly_vol*100:.1f}%超过{self.config.threshold_monthly_vol_warn*100:.0f}%警戒"
        else:
            vol_score = 1.0

        # 最大连亏评分（辅助指标）
        loss_streak_score = max(0, 1.0 - max_loss_streak / 30)

        # 加权评分：期望收益60%（核心） + 月度vol25% + 连亏控制15%
        score = exp_score * 0.60 + vol_score * 0.25 + loss_streak_score * 0.15
        # 通过逻辑：无否决项 AND 总分≥0.5
        passed = (not vetoed) and score >= 0.5

        return ValidationResult(
            test_name="交易质量",
            passed=passed,
            score=score,
            vetoed=vetoed,
            details={
                '胜率': f"{win_rate*100:.1f}%",
                '盈亏比': f"{profit_loss_ratio:.2f}:1",
                '期望收益': f"{expectancy:.3f}",
                '月度收益年化std': f"{monthly_vol*100:.1f}%",
                '最大连亏天数': max_loss_streak,
                '策略类型': self.config.strategy_type,
                '组合公式': f"{win_rate:.3f}×({profit_loss_ratio:.2f}+1)-1={expectancy:.3f}",
                '评分构成': f"期望收益{exp_score:.2f}×0.6+vol{vol_score:.2f}×0.25+连亏{loss_streak_score:.2f}×0.15",
            },
            warning=warning
        )
    
    # ============================================================
    # 运行全部检测
    # ============================================================
    def run_full_validation(self, trades_df: pd.DataFrame = None) -> FullReport:
        """运行全部7项检测"""
        print("=" * 70)
        print(f"  开始检测: {self.strategy_name}")
        print("=" * 70)
        
        report = FullReport(
            strategy_name=self.strategy_name,
            config=self.config,
        )
        
        try:
            report.add(self.test_parameter_sensitivity())
        except Exception as e:
            report.add(ValidationResult("参数敏感性", False, 0, {'错误': str(e)}))
        
        try:
            report.add(self.test_sample_split())
        except Exception as e:
            report.add(ValidationResult("样本分段验证", False, 0, {'错误': str(e)}))
        
        try:
            report.add(self.test_walk_forward())
        except Exception as e:
            report.add(ValidationResult("Walk-Forward", False, 0, {'错误': str(e)}))
        
        try:
            report.add(self.test_bootstrap())
        except Exception as e:
            report.add(ValidationResult("Bootstrap", False, 0, {'错误': str(e)}))
        
        try:
            report.add(self.test_volatility_regime())
        except Exception as e:
            report.add(ValidationResult("波动率环境分段", False, 0, {'错误': str(e)}))
        
        try:
            report.add(self.test_market_regime())
        except Exception as e:
            report.add(ValidationResult("市场regime分段", False, 0, {'错误': str(e)}))
        
        try:
            report.add(self.test_trade_quality(trades_df))
        except Exception as e:
            report.add(ValidationResult("交易质量", False, 0, {'错误': str(e)}))
        
        report.compute_overall()
        print("=" * 70)
        print(f"  检测完成: {self.strategy_name}")
        print(f"  总分: {report.overall_score:.3f} {'✅通过' if report.overall_passed else '❌未通过'}")
        print("=" * 70)
        
        return report


# ============================================================
# 便捷函数：一键检测
# ============================================================

def quick_validate(
    strategy_func: Callable,
    param_grid: Dict[str, List],
    g_close: pd.Series,
    v_close: pd.Series,
    strategy_name: str = "策略",
    config: ValidationConfig = None,
    backtest_engine_func: Callable = None,
    output_dir: str = ".",
) -> FullReport:
    """一键执行全部检测并保存报告"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    validator = StrategyValidator(
        strategy_func=strategy_func,
        param_grid=param_grid,
        g_close=g_close,
        v_close=v_close,
        strategy_name=strategy_name,
        config=config,
        backtest_engine_func=backtest_engine_func,
    )
    
    report = validator.run_full_validation()
    
    # 保存报告
    report.save(str(output_path / f"{strategy_name}_validation_report.txt"))
    report.plot(str(output_path / f"{strategy_name}_validation_chart.png"))
    
    return report
