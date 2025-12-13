import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging
import json
import csv
import os
from typing import Dict, List, Tuple, Optional, Any, Union
from enum import Enum
import hashlib
import pickle
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod

# ============================================================================
# CONFIGURATION AND SETUP
# ============================================================================

@dataclass
class StrategyConfig:
    """Configuration for trading strategy"""
    # Position sizing
    position_size_percent: float = 0.1
    max_position_size_percent: float = 0.25
    min_position_size: float = 100
    
    # Risk management
    max_drawdown_percent: float = 20.0
    daily_loss_limit: float = 500.0
    max_consecutive_losses: int = 5
    
    # Trading parameters
    commission_rate: float = 0.001
    slippage: float = 0.0005
    risk_reward_ratio: float = 2.0
    
    # Time restrictions
    trading_hours_start: str = "09:30"
    trading_hours_end: str = "16:00"
    blackout_periods: List[str] = None
    
    # Validation thresholds
    minimum_volume: int = 1000
    maximum_spread: float = 0.002
    data_quality_threshold: float = 0.95
    
    # Resilience
    max_retries: int = 3
    retry_backoff_base: float = 2.0
    circuit_breaker_threshold: int = 10
    
    # Reporting
    report_formats: List[str] = None
    auto_save_reports: bool = True
    report_directory: str = "reports"
    
    def __post_init__(self):
        if self.blackout_periods is None:
            self.blackout_periods = ["09:30-10:00", "15:30-16:00"]
        if self.report_formats is None:
            self.report_formats = ["markdown", "html", "csv"]

# Configure logging with DinoSay wrapper
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_strategy.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# ENUMS AND DATA CLASSES
# ============================================================================

class WaveType(Enum):
    MA_200 = "MA_200"
    MA_350 = "MA_350"
    MA_500 = "MA_500"

class StrategyType(Enum):
    A = "A"
    B = "B"

class ExitPath(Enum):
    STEP_3 = "STEP_3"
    CAPTURE = "CAPTURE"
    MA_200_WAVE = "MA_200_WAVE"
    ENHANCED = "ENHANCED"
    STOP_LOSS = "STOP_LOSS"
    MANUAL = "MANUAL"

class TradeStatus(Enum):
    PENDING = "PENDING"
    EXECUTED = "EXECUTED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"

@dataclass
class DinoSayExecution:
    """DinoSay execution record"""
    timestamp: datetime
    message: str
    level: str
    strategy: Optional[str]
    ma_wave: Optional[str]
    data_hash: Optional[str] = None
    execution_id: Optional[str] = None
    
    def __post_init__(self):
        if not self.execution_id:
            self.execution_id = hashlib.md5(
                f"{self.timestamp}{self.message}{self.level}".encode()
            ).hexdigest()[:8]

@dataclass
class TradeRecord:
    """Complete trade record"""
    trade_id: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    position_size: float
    pnl: float
    pnl_percent: float
    commission: float
    slippage: float
    net_pnl: float
    reason: str
    strategy: str
    ma_wave: str
    exit_path: str
    status: str
    risk_reward: float
    drawdown: float
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []

# ============================================================================
# RESILIENCE AND ERROR HANDLING
# ============================================================================

class CircuitBreaker:
    """Circuit breaker pattern for error handling"""
    def __init__(self, threshold: int = 10, reset_timeout: int = 300):
        self.threshold = threshold
        self.reset_timeout = reset_timeout
        self.error_count = 0
        self.last_error_time = None
        self.tripped = False
        
    def record_error(self):
        """Record an error and check if circuit should trip"""
        self.error_count += 1
        self.last_error_time = datetime.now()
        
        if self.error_count >= self.threshold:
            self.tripped = True
            logger.warning(f"Circuit breaker TRIPPED after {self.error_count} errors")
            
    def can_execute(self) -> bool:
        """Check if execution is allowed"""
        if not self.tripped:
            return True
            
        # Check if reset timeout has passed
        if self.last_error_time:
            time_since_error = (datetime.now() - self.last_error_time).total_seconds()
            if time_since_error > self.reset_timeout:
                self.reset()
                return True
                
        return False
        
    def reset(self):
        """Reset the circuit breaker"""
        self.error_count = 0
        self.tripped = False
        logger.info("Circuit breaker RESET")

class RetryManager:
    """Manage retry logic with exponential backoff"""
    def __init__(self, max_retries: int = 3, backoff_base: float = 2.0):
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        
    def execute_with_retry(self, func, *args, **kwargs):
        """Execute function with retry logic"""
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                logger.warning(f"Attempt {attempt + 1}/{self.max_retries} failed: {str(e)}")
                
                if attempt < self.max_retries - 1:
                    wait_time = self.backoff_base ** attempt
                    logger.info(f"Retrying in {wait_time:.2f} seconds...")
                    time.sleep(wait_time)
                    
        raise Exception(f"Max retries exceeded. Last error: {str(last_exception)}")

# ============================================================================
# DATA VALIDATION AND QUALITY
# ============================================================================

class DataValidator:
    """Validate market data quality"""
    @staticmethod
    def validate_dataframe(data: pd.DataFrame, min_rows: int = 500) -> Tuple[bool, List[str]]:
        """Validate dataframe for trading"""
        errors = []
        
        # Basic structure checks
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            errors.append(f"Missing columns: {missing_columns}")
            
        # Length check
        if len(data) < min_rows:
            errors.append(f"Insufficient data: {len(data)} rows, need at least {min_rows}")
            
        # Null checks
        null_counts = data[required_columns].isnull().sum()
        if null_counts.any():
            errors.append(f"Null values found: {null_counts.to_dict()}")
            
        # Price consistency checks
        if not (data['high'] >= data['low']).all():
            errors.append("High prices less than low prices")
            
        if not (data['high'] >= data['close']).all():
            errors.append("High prices less than close prices")
            
        if not (data['low'] <= data['close']).all():
            errors.append("Low prices greater than close prices")
            
        if not (data['close'] >= 0).all():
            errors.append("Negative close prices")
            
        # Volume checks
        if not (data['volume'] >= 0).all():
            errors.append("Negative volumes")
            
        # Spread checks
        spreads = (data['high'] - data['low']) / data['low']
        if spreads.max() > 0.1:  # 10% max spread
            errors.append(f"Excessive spread detected: {spreads.max():.2%}")
            
        return len(errors) == 0, errors
    
    @staticmethod
    def calculate_data_quality_score(data: pd.DataFrame) -> float:
        """Calculate data quality score (0-1)"""
        checks = []
        
        # Completeness
        null_ratio = data.isnull().sum().sum() / (data.shape[0] * data.shape[1])
        checks.append(1 - null_ratio)
        
        # Consistency
        price_checks = [
            (data['high'] >= data['low']).mean(),
            (data['high'] >= data['close']).mean(),
            (data['low'] <= data['close']).mean()
        ]
        checks.append(np.mean(price_checks))
        
        # Volume validity
        valid_volume = (data['volume'] > 0).mean()
        checks.append(valid_volume)
        
        return np.mean(checks)

# ============================================================================
# DOCUMENT AND REPORT GENERATION
# ============================================================================

class ReportGenerator:
    """Generate various report formats"""
    def __init__(self, strategy):
        self.strategy = strategy
        
    def generate_markdown_report(self) -> str:
        """Generate markdown report"""
        report = []
        report.append("# Trading Strategy Performance Report")
        report.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"**Strategy**: Exact Flowchart Matching v1.0")
        report.append("")
        
        # Summary
        perf = self.strategy.get_performance_report()
        report.append("## Summary")
        report.append(f"- Total Trades: {perf['total_trades']}")
        report.append(f"- Win Rate: {perf['win_rate']:.2f}%")
        report.append(f"- Total P&L: ${perf['total_pnl']:.2f}")
        report.append(f"- Final Balance: ${perf['final_balance']:.2f}")
        report.append(f"- Return: {perf['return_percentage']:.2f}%")
        report.append("")
        
        # Detailed Breakdown
        report.append("## Detailed Breakdown")
        
        # By Strategy
        report.append("### By Strategy")
        for strategy, count in perf['trades_by_strategy'].items():
            report.append(f"- Strategy {strategy}: {count} trades")
            
        # By Wave Type
        report.append("### By Wave Type")
        for wave, count in perf['trades_by_wave'].items():
            report.append(f"- {wave}: {count} trades")
            
        # By Exit Path
        report.append("### By Exit Path")
        for path, stats in perf['trades_by_exit_path'].items():
            report.append(f"- {path}: {stats['count']} trades, P&L: ${stats['total_pnl']:.2f}")
            
        # Recent Trades
        if self.strategy.trades:
            report.append("## Recent Trades")
            report.append("| Date | Strategy | Wave | Exit | P&L | Reason |")
            report.append("|------|----------|------|------|-----|--------|")
            for trade in self.strategy.trades[-10:]:
                date_str = trade['exit_time'].strftime('%Y-%m-%d')
                report.append(f"| {date_str} | {trade['strategy']} | {trade['ma_wave']} | {trade['exit_path']} | ${trade['pnl']:.2f} | {trade['reason'][:30]}... |")
                
        return "\n".join(report)
    
    def generate_html_report(self) -> str:
        """Generate HTML report"""
        md_report = self.generate_markdown_report()
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Trading Strategy Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #333; }}
                .summary {{ background: #f5f5f5; padding: 20px; border-radius: 5px; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #4CAF50; color: white; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
                .positive {{ color: green; }}
                .negative {{ color: red; }}
            </style>
        </head>
        <body>
            <div id="content">{self._markdown_to_html(md_report)}</div>
            <script>
                // Add color coding to P&L values
                document.querySelectorAll('td').forEach(td => {{
                    if (td.textContent.includes('$') && td.textContent.includes('-')) {{
                        td.classList.add('negative');
                    }} else if (td.textContent.includes('$') && !td.textContent.includes('-')) {{
                        td.classList.add('positive');
                    }}
                }});
            </script>
        </body>
        </html>
        """
        return html
    
    def _markdown_to_html(self, markdown: str) -> str:
        """Simple markdown to HTML conversion"""
        html = markdown
        html = html.replace('# ', '<h1>').replace('\n#', '</h1>\n<h1>')
        html = html.replace('## ', '<h2>').replace('\n##', '</h2>\n<h2>')
        html = html.replace('### ', '<h3>').replace('\n###', '</h3>\n<h3>')
        html = html.replace('**', '<strong>').replace('**', '</strong>')
        html = html.replace('\n- ', '\n<li>').replace('\n-', '</li>\n<li>')
        html = html.replace('|', '</td><td>').replace('\n|', '</td></tr>\n<tr><td>')
        html = html.replace('<tr><td></td><td>', '<table><tr><th>').replace('</td></tr>', '</th></tr>', 1)
        return html
    
    def generate_csv_report(self, filename: str = None):
        """Generate CSV report"""
        if not self.strategy.trades:
            return "No trades to report"
            
        if filename is None:
            filename = f"trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            
        # Prepare data
        rows = []
        for trade in self.strategy.trades:
            row = {
                'trade_id': trade['trade_id'],
                'entry_time': trade['entry_time'].strftime('%Y-%m-%d %H:%M:%S'),
                'exit_time': trade['exit_time'].strftime('%Y-%m-%d %H:%M:%S'),
                'entry_price': trade['entry_price'],
                'exit_price': trade['exit_price'],
                'position_size': trade['position_size'],
                'pnl': trade['pnl'],
                'pnl_percent': trade['pnl_percent'],
                'commission': trade['commission'],
                'slippage': trade['slippage'],
                'net_pnl': trade['net_pnl'],
                'strategy': trade['strategy'],
                'ma_wave': trade['ma_wave'],
                'exit_path': trade['exit_path'],
                'reason': trade['reason'],
                'status': trade['status'],
                'risk_reward': trade['risk_reward'],
                'drawdown': trade['drawdown'],
                'tags': ','.join(trade['tags'])
            }
            rows.append(row)
            
        # Write to CSV
        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
            
        return f"CSV report saved to {filename}"
    
    def save_report(self, filename: str = None, format: str = "markdown"):
        """Save report to file"""
        os.makedirs(self.strategy.config.report_directory, exist_ok=True)
        
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{self.strategy.config.report_directory}/report_{timestamp}"
            
        if format == "markdown":
            content = self.generate_markdown_report()
            full_filename = f"{filename}.md"
        elif format == "html":
            content = self.generate_html_report()
            full_filename = f"{filename}.html"
        elif format == "csv":
            return self.generate_csv_report(f"{filename}.csv")
        else:
            raise ValueError(f"Unsupported format: {format}")
            
        with open(full_filename, 'w') as f:
            f.write(content)
            
        return f"Report saved to {full_filename}"

# ============================================================================
# TRADE FILTERING AND RISK MANAGEMENT
# ============================================================================

class TradeFilter:
    """Apply trade filters based on configuration"""
    def __init__(self, config: StrategyConfig):
        self.config = config
        
    def filter_trade(self, trade_signal: Dict[str, Any], current_time: datetime = None) -> Tuple[bool, str]:
        """Apply all trade filters"""
        if current_time is None:
            current_time = datetime.now()
            
        # Time of day filter
        if not self._check_trading_hours(current_time):
            return False, "Outside trading hours"
            
        # Blackout periods
        if self._in_blackout_period(current_time):
            return False, "In blackout period"
            
        # Volume filter
        if trade_signal.get('volume', 0) < self.config.minimum_volume:
            return False, f"Insufficient volume: {trade_signal.get('volume')} < {self.config.minimum_volume}"
            
        # Spread filter
        spread = trade_signal.get('spread', 0)
        if spread > self.config.maximum_spread:
            return False, f"Spread too high: {spread:.4f} > {self.config.maximum_spread:.4f}"
            
        # Data quality filter
        data_quality = trade_signal.get('data_quality', 1.0)
        if data_quality < self.config.data_quality_threshold:
            return False, f"Low data quality: {data_quality:.2f} < {self.config.data_quality_threshold:.2f}"
            
        return True, "All filters passed"
    
    def _check_trading_hours(self, current_time: datetime) -> bool:
        """Check if current time is within trading hours"""
        time_str = current_time.strftime('%H:%M')
        return self.config.trading_hours_start <= time_str <= self.config.trading_hours_end
    
    def _in_blackout_period(self, current_time: datetime) -> bool:
        """Check if current time is in blackout period"""
        time_str = current_time.strftime('%H:%M')
        for period in self.config.blackout_periods:
            start_str, end_str = period.split('-')
            if start_str <= time_str <= end_str:
                return True
        return False

class RiskManager:
    """Manage trading risk"""
    def __init__(self, config: StrategyConfig):
        self.config = config
        self.daily_pnl = 0
        self.consecutive_losses = 0
        self.max_drawdown = 0
        self.equity_high = config.min_position_size
        
    def check_trade_risk(self, position_size: float, entry_price: float, 
                        stop_loss: float, take_profit: float) -> Tuple[bool, str]:
        """Check if trade meets risk criteria"""
        # Position size limits
        if position_size < self.config.min_position_size:
            return False, f"Position size too small: ${position_size:.2f}"
            
        # Risk-reward ratio
        risk = abs(entry_price - stop_loss) * position_size
        reward = abs(take_profit - entry_price) * position_size
        if risk > 0:
            rr_ratio = reward / risk
            if rr_ratio < self.config.risk_reward_ratio:
                return False, f"Risk-reward ratio too low: {rr_ratio:.2f}"
                
        # Daily loss limit
        if self.daily_pnl < -self.config.daily_loss_limit:
            return False, f"Daily loss limit reached: ${self.daily_pnl:.2f}"
            
        # Consecutive losses
        if self.consecutive_losses >= self.config.max_consecutive_losses:
            return False, f"Max consecutive losses reached: {self.consecutive_losses}"
            
        return True, "Risk check passed"
    
    def update_risk_metrics(self, pnl: float):
        """Update risk metrics after trade"""
        self.daily_pnl += pnl
        
        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
    
    def calculate_drawdown(self, current_equity: float):
        """Calculate current drawdown"""
        if current_equity > self.equity_high:
            self.equity_high = current_equity
            
        drawdown = (self.equity_high - current_equity) / self.equity_high * 100
        self.max_drawdown = max(self.max_drawdown, drawdown)
        
        if drawdown > self.config.max_drawdown_percent:
            logger.warning(f"Drawdown exceeded: {drawdown:.2f}% > {self.config.max_drawdown_percent:.2f}%")
            
        return drawdown
    
    def reset_daily_metrics(self):
        """Reset daily metrics"""
        self.daily_pnl = 0
        logger.info("Daily risk metrics reset")

# ============================================================================
# PERFORMANCE ANALYTICS
# ============================================================================

class PerformanceAnalytics:
    """Advanced performance analytics"""
    @staticmethod
    def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) < 2:
            return 0.0
            
        excess_returns = returns - risk_free_rate/252
        if np.std(excess_returns) == 0:
            return 0.0
            
        return np.sqrt(252) * np.mean(excess_returns) / np.std(excess_returns)
    
    @staticmethod
    def calculate_sortino_ratio(returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio"""
        if len(returns) < 2:
            return 0.0
            
        excess_returns = returns - risk_free_rate/252
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0 or np.std(downside_returns) == 0:
            return 0.0
            
        return np.sqrt(252) * np.mean(excess_returns) / np.std(downside_returns)
    
    @staticmethod
    def calculate_max_drawdown(equity_curve: List[float]) -> float:
        """Calculate maximum drawdown"""
        if not equity_curve:
            return 0.0
            
        equity = np.array(equity_curve)
        running_max = np.maximum.accumulate(equity)
        drawdowns = (running_max - equity) / running_max
        return np.max(drawdowns) * 100 if len(drawdowns) > 0 else 0.0
    
    @staticmethod
    def calculate_calmar_ratio(returns: np.ndarray, max_drawdown: float) -> float:
        """Calculate Calmar ratio"""
        if max_drawdown == 0 or len(returns) == 0:
            return 0.0
            
        annual_return = np.mean(returns) * 252
        return annual_return / max_drawdown
    
    @staticmethod
    def calculate_winning_streaks(trades: List[Dict]) -> Dict[str, Any]:
        """Calculate winning and losing streaks"""
        if not trades:
            return {"max_winning_streak": 0, "max_losing_streak": 0}
            
        current_streak = 0
        max_winning_streak = 0
        max_losing_streak = 0
        current_type = None
        
        for trade in trades:
            is_win = trade['pnl'] > 0
            
            if current_type is None:
                current_type = is_win
                current_streak = 1
            elif current_type == is_win:
                current_streak += 1
            else:
                if current_type:  # Winning streak ended
                    max_winning_streak = max(max_winning_streak, current_streak)
                else:  # Losing streak ended
                    max_losing_streak = max(max_losing_streak, current_streak)
                    
                current_type = is_win
                current_streak = 1
        
        # Check final streak
        if current_type:
            max_winning_streak = max(max_winning_streak, current_streak)
        else:
            max_losing_streak = max(max_losing_streak, current_streak)
            
        return {
            "max_winning_streak": max_winning_streak,
            "max_losing_streak": max_losing_streak
        }

# ============================================================================
# CORE TRADING STRATEGY (WITH ALL FEATURES)
# ============================================================================

class FibonacciLevels:
    """Fibonacci retracement levels"""
    LEVELS = {
        0: 0.0,      # Fibo_0
        236: 0.236,  # Fibo_23.6%
        382: 0.382,
        500: 0.5,
        618: 0.618,
        764: 0.764,  # Fibo_76.4%
        1000: 1.0    # Fibo_100% (Fibo_1)
    }
    
    @staticmethod
    def calculate_levels(high: float, low: float) -> Dict[str, float]:
        """Calculate Fibonacci levels based on swing high/low"""
        diff = high - low
        return {
            'Fibo_0': low,
            'Fibo_23.6%': high - diff * 0.236,
            'Fibo_38.2%': high - diff * 0.382,
            'Fibo_50%': high - diff * 0.5,
            'Fibo_61.8%': high - diff * 0.618,
            'Fibo_76.4%': high - diff * 0.764,
            'Fibo_100%': high  # Fibo_1
        }

class TradingStrategy:
    def __init__(self, config: StrategyConfig = None, initial_balance: float = 10000.0):
        # Configuration
        self.config = config or StrategyConfig()
        self.initial_balance = initial_balance
        self.balance = initial_balance
        
        # Core trading state
        self.position = None
        self.position_size = 0
        self.entry_price = 0
        self.entry_time = None
        self.stop_loss = 0
        self.take_profit = 0
        
        # Fibonacci levels storage
        self.captured_fibo_0 = None
        self.captured_fibo_1 = None
        self.captured_fibo_100 = None
        self.captured_fibo_1_dip = None
        
        # State tracking (EXACT flowchart matching)
        self.strategy_active = None
        self.ma_wave_type = None
        self.phase = 1
        self.exit_path = None
        
        # MA_200 Wave specific tracking
        self.ma_200_wave_exit = {
            'step_1_complete': False,
            'step_2_complete': False,
        }
        
        # Strategy B specific tracking
        self.strategy_b_steps = {
            'step_1_complete': False,
            'step_2_complete': False,
            'step_3_complete': False,
            'step_4_active': False,
        }
        
        # Strategy A specific tracking
        self.strategy_a_state = {
            'phase_1_monitoring': False,
            'lesser_mas_checked': False,
            'lesser_mas_result': None,
            'step_3_path_active': False,
            'capture_path_active': False,
            'dual_monitoring_active': False,
            'branch_type': None,
            'new_fibo_1_captured': False,
        }
        
        # RSI sequence tracking
        self.rsi_state = 'waiting_for_55'
        self.rsi_stop_loss_state = 'waiting_for_53'
        
        # DinoSay executions
        self.dinosay_executions = []
        
        # Trade history (enhanced)
        self.trades = []
        self.equity_curve = [initial_balance]
        
        # Resilience features
        self.circuit_breaker = CircuitBreaker(
            threshold=self.config.circuit_breaker_threshold
        )
        self.retry_manager = RetryManager(
            max_retries=self.config.max_retries,
            backoff_base=self.config.retry_backoff_base
        )
        
        # Data validation
        self.data_validator = DataValidator()
        
        # Trade filtering and risk management
        self.trade_filter = TradeFilter(self.config)
        self.risk_manager = RiskManager(self.config)
        
        # Reporting
        self.report_generator = ReportGenerator(self)
        
        # Performance analytics
        self.performance_analytics = PerformanceAnalytics()
        
        logger.info("Trading Strategy initialized with ALL features and exact flowchart matching")
        self.dinosay("Strategy initialized with all features", "INFO")

    # ============================================================================
    # DINO SAY EXECUTIONS
    # ============================================================================
    
    def dinosay(self, message: str, level: str = "INFO"):
        """Record DinoSay execution"""
        execution = DinoSayExecution(
            timestamp=datetime.now(),
            message=message,
            level=level,
            strategy=self.strategy_active.value if self.strategy_active else None,
            ma_wave=self.ma_wave_type.value if self.ma_wave_type else None
        )
        self.dinosay_executions.append(execution)
        
        # Also log to standard logger
        getattr(logger, level.lower())(message)
        
        return execution.execution_id
    
    def get_dinosay_summary(self, last_n: int = 50) -> str:
        """Get summary of recent DinoSay executions"""
        recent = self.dinosay_executions[-last_n:] if self.dinosay_executions else []
        
        summary = []
        summary.append("DinoSay Execution Summary:")
        summary.append(f"Total executions: {len(self.dinosay_executions)}")
        summary.append(f"Recent executions (last {len(recent)}):")
        
        for exec in recent:
            time_str = exec.timestamp.strftime('%H:%M:%S')
            summary.append(f"[{time_str}] {exec.level}: {exec.message}")
            
        return "\n".join(summary)
    
    # ============================================================================
    # RESILIENT EXECUTION
    # ============================================================================
    
    def execute_resiliently(self, func, *args, **kwargs):
        """Execute function with resilience features"""
        # Check circuit breaker
        if not self.circuit_breaker.can_execute():
            raise Exception("Circuit breaker is tripped")
            
        try:
            # Execute with retry logic
            return self.retry_manager.execute_with_retry(func, *args, **kwargs)
        except Exception as e:
            # Record error in circuit breaker
            self.circuit_breaker.record_error()
            self.dinosay(f"Error in resilient execution: {str(e)}", "ERROR")
            raise
    
    # ============================================================================
    # DATA VALIDATION AND PROCESSING
    # ============================================================================
    
    def validate_and_prepare_data(self, data: pd.DataFrame) -> Tuple[bool, pd.DataFrame, str]:
        """Validate and prepare data for strategy execution"""
        try:
            # Validate data quality
            is_valid, errors = self.data_validator.validate_dataframe(data)
            if not is_valid:
                error_msg = f"Data validation failed: {', '.join(errors)}"
                self.dinosay(error_msg, "ERROR")
                return False, data, error_msg
                
            # Calculate data quality score
            quality_score = self.data_validator.calculate_data_quality_score(data)
            if quality_score < self.config.data_quality_threshold:
                warning = f"Low data quality score: {quality_score:.2f}"
                self.dinosay(warning, "WARNING")
                
            # Calculate moving averages
            data = self.calculate_moving_averages(data)
            
            # Calculate spread
            data['spread'] = (data['high'] - data['low']) / data['low']
            
            self.dinosay(f"Data prepared successfully. Quality score: {quality_score:.2f}", "INFO")
            return True, data, f"Data quality: {quality_score:.2f}"
            
        except Exception as e:
            error_msg = f"Data preparation failed: {str(e)}"
            self.dinosay(error_msg, "ERROR")
            return False, data, error_msg
    
    # ============================================================================
    # CORE STRATEGY METHODS (EXACT FLOWCHART MATCHING)
    # ============================================================================
    
    def calculate_moving_averages(self, data: pd.DataFrame, price_column: str = 'close') -> pd.DataFrame:
        """Calculate all required moving averages"""
        ma_periods = [2, 7, 14, 21, 50, 100, 200, 350, 500]
        
        for period in ma_periods:
            col_name = f'MA_{period}'
            if col_name not in data.columns or data[col_name].isnull().any():
                data[col_name] = data[price_column].rolling(window=period).mean()
            
        # Calculate RSI MA50
        if 'RSI_MA50' not in data.columns or data['RSI_MA50'].isnull().any():
            data['RSI'] = self.calculate_rsi(data[price_column])
            data['RSI_MA50'] = data['RSI'].rolling(window=50).mean()
        
        return data

    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_daily_difference(self, data: pd.DataFrame) -> float:
        """Calculate daily price difference percentage"""
        if len(data) < 2:
            return 0
        latest_close = data['close'].iloc[-1]
        prev_close = data['close'].iloc[-2]
        daily_diff = ((latest_close - prev_close) / prev_close) * 100
        return daily_diff

    def check_entry_setup(self, data: pd.DataFrame, fibo_levels: Dict[str, float]) -> Dict[str, Any]:
        """Check entry setups for MA_200, MA_350, MA_500 waves EXACTLY as per flowchart"""
        latest = data.iloc[-1]
        entry_setups = {
            'MA_200': False,
            'MA_350': False,
            'MA_500': False,
            'details': {}
        }
        
        # MA_200 Wave Entry Condition (EXACTLY as per flowchart)
        ma_200_condition = (
            latest['MA_100'] >= fibo_levels['Fibo_23.6%'] and
            latest['MA_350'] >= fibo_levels['Fibo_23.6%'] and
            latest['MA_500'] >= fibo_levels['Fibo_23.6%'] and
            latest['MA_200'] <= fibo_levels['Fibo_23.6%']
        )
        
        # MA_350 Wave Entry Condition (EXACTLY as per flowchart)
        ma_350_condition = (
            latest['MA_200'] >= fibo_levels['Fibo_23.6%'] and
            latest['MA_100'] >= fibo_levels['Fibo_23.6%'] and
            latest['MA_500'] >= fibo_levels['Fibo_23.6%'] and
            latest['MA_350'] <= fibo_levels['Fibo_23.6%']
        )
        
        # MA_500 Wave Entry Condition (EXACTLY as per flowchart)
        ma_500_condition = (
            latest['MA_200'] >= fibo_levels['Fibo_23.6%'] and
            latest['MA_100'] >= fibo_levels['Fibo_23.6%'] and
            latest['MA_350'] >= fibo_levels['Fibo_23.6%'] and
            latest['MA_500'] <= fibo_levels['Fibo_23.6%']
        )
        
        entry_setups['MA_200'] = ma_200_condition
        entry_setups['MA_350'] = ma_350_condition
        entry_setups['MA_500'] = ma_500_condition
        
        # Store details for validation flow (as per flowchart)
        entry_setups['details'] = {
            'MA_200': {
                'initial_signal': ma_200_condition,
                'wait_for_condition': latest['MA_200'] >= fibo_levels['Fibo_23.6%'],
                'final_validation': all([
                    latest['MA_100'] >= fibo_levels['Fibo_23.6%'],
                    latest['MA_200'] >= fibo_levels['Fibo_23.6%'],
                    latest['MA_350'] >= fibo_levels['Fibo_23.6%'],
                    latest['MA_500'] >= fibo_levels['Fibo_23.6%']
                ])
            },
            'MA_350': {
                'initial_signal': ma_350_condition,
                'wait_for_condition': latest['MA_350'] >= fibo_levels['Fibo_23.6%'],
                'final_validation': all([
                    latest['MA_100'] >= fibo_levels['Fibo_23.6%'],
                    latest['MA_200'] >= fibo_levels['Fibo_23.6%'],
                    latest['MA_350'] >= fibo_levels['Fibo_23.6%'],
                    latest['MA_500'] >= fibo_levels['Fibo_23.6%']
                ])
            },
            'MA_500': {
                'initial_signal': ma_500_condition,
                'wait_for_condition': latest['MA_500'] >= fibo_levels['Fibo_23.6%'],
                'final_validation': all([
                    latest['MA_100'] >= fibo_levels['Fibo_23.6%'],
                    latest['MA_200'] >= fibo_levels['Fibo_23.6%'],
                    latest['MA_350'] >= fibo_levels['Fibo_23.6%'],
                    latest['MA_500'] >= fibo_levels['Fibo_23.6%']
                ])
            }
        }
        
        return entry_setups

    def strategy_a_activation(self, daily_diff: float) -> bool:
        """Check Strategy A activation condition EXACTLY as per flowchart"""
        return -1.0 <= daily_diff <= 4.0

    def strategy_b_activation(self, daily_diff: float) -> bool:
        """Check Strategy B activation condition EXACTLY as per flowchart"""
        return daily_diff <= -1.0

    def enter_position(self, data: pd.DataFrame, ma_wave: str, entry_price: float, fibo_levels: Dict[str, float]):
        """Enter position based on MA wave type EXACTLY as per flowchart"""
        # Check trade filters
        trade_signal = {
            'volume': data['volume'].iloc[-1],
            'spread': data['spread'].iloc[-1] if 'spread' in data.columns else 0,
            'data_quality': self.data_validator.calculate_data_quality_score(data)
        }
        
        filter_passed, filter_message = self.trade_filter.filter_trade(trade_signal)
        if not filter_passed:
            self.dinosay(f"Trade filtered: {filter_message}", "WARNING")
            return
        
        # Calculate position size with risk management
        position_value = self.balance * self.config.position_size_percent
        self.position_size = position_value / entry_price
        
        # Check risk
        risk_check_passed, risk_message = self.risk_manager.check_trade_risk(
            self.position_size, entry_price, 
            fibo_levels['Fibo_23.6%'], entry_price * 1.02  # 2% take profit
        )
        
        if not risk_check_passed:
            self.dinosay(f"Risk check failed: {risk_message}", "WARNING")
            return
        
        # Execute trade
        self.position = 'long'
        self.entry_price = entry_price
        self.entry_time = datetime.now()
        self.ma_wave_type = WaveType(ma_wave)
        
        # Set initial stop loss at Fibo_23.6% as per flowchart
        self.stop_loss = fibo_levels['Fibo_23.6%']
        
        # Calculate commission and slippage
        commission = position_value * self.config.commission_rate
        slippage_cost = position_value * self.config.slippage
        
        # DinoSay execution record
        self.dinosay(f"ENTER POSITION - {ma_wave} Wave", "INFO")
        self.dinosay(f"Entry Price: ${entry_price:.2f}, Size: {self.position_size:.4f}", "INFO")
        self.dinosay(f"Stop Loss: ${self.stop_loss:.2f}, Commission: ${commission:.2f}", "INFO")
        
        # Initialize monitoring based on strategy EXACTLY as per flowchart
        if self.strategy_active == StrategyType.A:
            self.phase = 1
            self.strategy_a_state['phase_1_monitoring'] = True
            self.dinosay("Strategy A: Phase 1 Dual Monitoring Activated", "INFO")
            
            if self.ma_wave_type == WaveType.MA_200:
                self.ma_200_wave_exit['step_1_complete'] = False
                self.ma_200_wave_exit['step_2_complete'] = False
                self.dinosay("Strategy A MA_200 Wave: Exit logic initialized", "INFO")
                
        elif self.strategy_active == StrategyType.B:
            self.strategy_b_steps['step_1_complete'] = False
            self.strategy_b_steps['step_2_complete'] = False
            self.strategy_b_steps['step_3_complete'] = False
            self.strategy_b_steps['step_4_active'] = False
            self.dinosay("Strategy B: Enhanced Exit Logic Activated", "INFO")
            
            if self.ma_wave_type == WaveType.MA_200:
                self.dinosay("Strategy B MA_200 Wave: Specific exit logic activated", "INFO")

    def stop_loss_condition_met(self, data: pd.DataFrame, fibo_levels: Dict[str, float]) -> bool:
        """Check if stop loss condition is triggered EXACTLY as per flowchart (MA_200 ≤ Fibo_23.6%)"""
        latest = data.iloc[-1]
        
        if latest['MA_200'] <= fibo_levels['Fibo_23.6%']:
            self.dinosay("STOP LOSS TRIGGERED: MA_200 ≤ Fibo_23.6%", "WARNING")
            
            # CAPTURE Fibo_0 level EXACTLY as per flowchart
            self.captured_fibo_0 = fibo_levels['Fibo_0']
            self.dinosay(f"Captured Fibo_0 level: ${self.captured_fibo_0:.2f}", "INFO")
            
            return True
        return False

    def execute_stop_loss_flow(self, data: pd.DataFrame, current_price: float) -> bool:
        """Execute the complete stop loss flow EXACTLY as per flowchart"""
        if not self.captured_fibo_0:
            self.dinosay("No captured Fibo_0 level for stop loss flow", "ERROR")
            return False
            
        latest = data.iloc[-1]
        
        # CHECK TWO CONDITIONS EXACTLY as per flowchart
        condition_1 = latest['MA_14'] <= self.captured_fibo_0
        condition_2 = latest['MA_100'] >= latest['MA_500']
        
        self.dinosay(f"Stop Loss Condition 1 (MA_14 ≤ Fibo_0): {condition_1}", "INFO")
        self.dinosay(f"Stop Loss Condition 2 (MA_100 ≥ MA_500): {condition_2}", "INFO")
        
        if condition_1:
            # PATH 1: Wait for MA_100 ≥ MA_200 EXACTLY as per flowchart
            self.dinosay("Stop Loss PATH 1: Waiting for MA_100 ≥ MA_200", "INFO")
            if latest['MA_100'] >= latest['MA_200']:
                self.exit_position(current_price, "Stop Loss - Condition 1 Met (MA_100 ≥ MA_200)", ExitPath.STOP_LOSS)
                return True
                
        elif condition_2:
            # PATH 2: Wait for MA_100 ≤ MA_500 AND MA_200 ≤ MA_500 EXACTLY as per flowchart
            self.dinosay("Stop Loss PATH 2: Waiting for MA_100 ≤ MA_500 AND MA_200 ≤ MA_500", "INFO")
            if latest['MA_100'] <= latest['MA_500'] and latest['MA_200'] <= latest['MA_500']:
                # Check RSI: RSI_MA50 ≥ 53 EXACTLY as per flowchart
                self.dinosay("Stop Loss PATH 2: Checking RSI_MA50 ≥ 53", "INFO")
                if self.check_rsi_stop_loss_sequence(data):
                    self.exit_position(current_price, "Stop Loss - Condition 2 Met with RSI sequence", ExitPath.STOP_LOSS)
                    return True
        
        return False

    def check_rsi_stop_loss_sequence(self, data: pd.DataFrame) -> bool:
        """Check RSI sequence for stop loss: 1. First ≥ 53, 2. THEN ≤ 51 EXACTLY as per flowchart"""
        latest = data.iloc[-1]
        
        if self.rsi_stop_loss_state == 'waiting_for_53' and latest['RSI_MA50'] >= 53:
            self.dinosay("Stop Loss RSI Sequence: First condition met (RSI_MA50 ≥ 53)", "INFO")
            self.rsi_stop_loss_state = 'waiting_for_51'
            return False
            
        elif self.rsi_stop_loss_state == 'waiting_for_51' and latest['RSI_MA50'] <= 51:
            self.dinosay("Stop Loss RSI Sequence: Second condition met (RSI_MA50 ≤ 51)", "INFO")
            self.rsi_stop_loss_state = 'waiting_for_53'
            return True
            
        return False

    def ma_200_wave_exit_logic(self, data: pd.DataFrame, fibo_levels: Dict[str, float], current_price: float) -> bool:
        """Execute MA_200 Wave exit logic EXACTLY as per flowchart (for both Strategy A and B)"""
        latest = data.iloc[-1]
        
        # STEP 1: Wait for BOTH conditions EXACTLY as per flowchart
        if not self.ma_200_wave_exit['step_1_complete']:
            step_1_condition = (
                latest['MA_100'] >= fibo_levels['Fibo_76.4%'] and
                latest['MA_500'] >= fibo_levels['Fibo_76.4%']
            )
            
            if step_1_condition:
                self.dinosay("MA_200 Wave Exit: Step 1 COMPLETE - MA_100 ≥ Fibo_0.764 AND MA_500 ≥ Fibo_0.764", "INFO")
                self.ma_200_wave_exit['step_1_complete'] = True
            else:
                return False
        
        # STEP 2: THEN Wait for MA_350 ≤ MA_500 EXACTLY as per flowchart
        if self.ma_200_wave_exit['step_1_complete'] and not self.ma_200_wave_exit['step_2_complete']:
            step_2_condition = latest['MA_350'] <= latest['MA_500']
            
            if step_2_condition:
                self.dinosay("MA_200 Wave Exit: Step 2 COMPLETE - MA_350 ≤ MA_500", "INFO")
                self.ma_200_wave_exit['step_2_complete'] = True
            else:
                return False
        
        # EXECUTE MARKET SELL ORDER EXACTLY as per flowchart
        if self.ma_200_wave_exit['step_2_complete']:
            self.dinosay("MA_200 Wave Exit: EXECUTING MARKET SELL ORDER", "INFO")
            self.exit_position(current_price, "MA_200 Wave Exit - Steps 1&2 Complete", ExitPath.MA_200_WAVE)
            return True
        
        return False

    def strategy_a_exit_logic(self, data: pd.DataFrame, fibo_levels: Dict[str, float], current_price: float) -> bool:
        """Execute Strategy A exit logic EXACTLY as per flowchart"""
        latest = data.iloc[-1]
        
        # Handle MA_200 Wave separately (as per flowchart)
        if self.ma_wave_type == WaveType.MA_200:
            return self.ma_200_wave_exit_logic(data, fibo_levels, current_price)
        
        # For MA_350/500 Wave in Strategy A
        if self.phase == 1:
            # PHASE 1: DUAL MONITORING EXACTLY as per flowchart
            if not self.strategy_a_state['phase_1_monitoring']:
                self.strategy_a_state['phase_1_monitoring'] = True
                self.dinosay("Strategy A Phase 1: Dual Monitoring CONSTANTLY CHECK BOTH conditions", "INFO")
            
            condition_a = latest['MA_200'] >= fibo_levels['Fibo_76.4%']
            condition_b = latest['MA_200'] <= fibo_levels['Fibo_23.6%']
            
            self.dinosay(f"Phase 1 Monitoring - Condition A (MA_200 ≥ Fibo_0.764): {condition_a}", "INFO")
            self.dinosay(f"Phase 1 Monitoring - Condition B (MA_200 ≤ Fibo_0.236): {condition_b}", "INFO")
            
            if condition_a:
                # Event A: MA_200 ≥ Fibo_0.764 EXACTLY as per flowchart
                self.dinosay("Strategy A: MA_200 ≥ Fibo_0.764 (Event A) - Proceed to Phase 2", "INFO")
                self.phase = 2
                return False
                
            elif condition_b:
                # Condition B met: MA_200 ≤ Fibo_0.236 EXACTLY as per flowchart
                self.dinosay("Strategy A: MA_200 ≤ Fibo_0.236 - Triggering Stop Loss", "WARNING")
                if self.stop_loss_condition_met(data, fibo_levels):
                    self.execute_stop_loss_flow(data, current_price)
                return True
        
        elif self.phase == 2:
            # PHASE 2: Check ALL lesser MAs (002, 007, 21, 50) ≥ MA_200 EXACTLY as per flowchart
            if not self.strategy_a_state['lesser_mas_checked']:
                self.dinosay("Strategy A Phase 2: Checking ALL lesser MAs ≥ MA_200", "INFO")
                
                lesser_mas_check = all([
                    latest['MA_2'] >= latest['MA_200'],
                    latest['MA_7'] >= latest['MA_200'],
                    latest['MA_21'] >= latest['MA_200'],
                    latest['MA_50'] >= latest['MA_200']
                ])
                
                self.strategy_a_state['lesser_mas_checked'] = True
                self.strategy_a_state['lesser_mas_result'] = lesser_mas_check
                
                if lesser_mas_check:
                    self.dinosay("Strategy A: YES - All lesser MAs ≥ MA_200 - Entering STEP 3 PATH", "INFO")
                    self.strategy_a_state['step_3_path_active'] = True
                else:
                    self.dinosay("Strategy A: NO - Some lesser MAs < MA_200 - Entering CAPTURE PATH", "INFO")
                    self.strategy_a_state['capture_path_active'] = True
            
            # STEP 3 PATH EXACTLY as per flowchart
            if self.strategy_a_state['step_3_path_active']:
                # Wait for MA_100 ≤ MA_500 EXACTLY as per flowchart
                if latest['MA_100'] <= latest['MA_500']:
                    self.dinosay("Step 3 Path: MA_100 ≤ MA_500 - CAPTURE NEW Fibo_1 at MA_100 dip position", "INFO")
                    
                    # CAPTURE NEW Fibo_1 at MA_100 dip position EXACTLY as per flowchart
                    self.captured_fibo_1_dip = latest['MA_100']
                    self.strategy_a_state['new_fibo_1_captured'] = True
                    self.dinosay(f"Captured new Fibo_1 at MA_100 dip: ${self.captured_fibo_1_dip:.2f}", "INFO")
                    
                    # Determine BRANCH based on MA_500 vs Fibo 76.4% EXACTLY as per flowchart
                    if latest['MA_500'] >= fibo_levels['Fibo_76.4%']:
                        self.strategy_a_state['branch_type'] = 'A'
                        self.dinosay("Step 3 Path: BRANCH A - MA_500 ≥ Fibo 76.4%", "INFO")
                    else:
                        self.strategy_a_state['branch_type'] = 'B'
                        self.dinosay("Step 3 Path: BRANCH B - MA_500 ≤ Fibo 76.4%", "INFO")
                
                # Execute branch logic if new Fibo_1 captured
                if self.strategy_a_state['new_fibo_1_captured']:
                    if self.strategy_a_state['branch_type'] == 'A':
                        # BRANCH A: Wait for MA_100 ≥ New Fibo_1 EXACTLY as per flowchart
                        if latest['MA_100'] >= self.captured_fibo_1_dip:
                            self.dinosay("Branch A: MA_100 ≥ New Fibo_1", "INFO")
                            # Wait for MA_50 ≤ Fibo 76.4% EXACTLY as per flowchart
                            if latest['MA_50'] <= fibo_levels['Fibo_76.4%']:
                                self.dinosay("Branch A: MA_50 ≤ Fibo 76.4%", "INFO")
                                # Wait for RSI sequence EXACTLY as per flowchart
                                if self.check_rsi_sequence(latest['RSI_MA50']):
                                    self.exit_position(current_price, "Strategy A - Step 3 Path Branch A", ExitPath.STEP_3)
                                    return True
                    else:
                        # BRANCH B: Wait for MA_350 ≥ New Fibo_1 EXACTLY as per flowchart
                        if latest['MA_350'] >= self.captured_fibo_1_dip:
                            self.dinosay("Branch B: MA_350 ≥ New Fibo_1", "INFO")
                            # Wait for MA_50 ≤ Fibo 76.4% EXACTLY as per flowchart
                            if latest['MA_50'] <= fibo_levels['Fibo_76.4%']:
                                self.dinosay("Branch B: MA_50 ≤ Fibo 76.4%", "INFO")
                                # Wait for RSI sequence EXACTLY as per flowchart
                                if self.check_rsi_sequence(latest['RSI_MA50']):
                                    self.exit_position(current_price, "Strategy A - Step 3 Path Branch B", ExitPath.STEP_3)
                                    return True
            
            # CAPTURE PATH EXACTLY as per flowchart
            elif self.strategy_a_state['capture_path_active']:
                # CAPTURE Fibo_1 level (level_100) EXACTLY as per flowchart
                if not self.captured_fibo_1:
                    self.captured_fibo_1 = fibo_levels['Fibo_100%']
                    self.captured_fibo_100 = latest['MA_100']
                    self.dinosay(f"Capture Path: CAPTURED Fibo_1 level at MA_100: ${self.captured_fibo_100:.2f}", "INFO")
                    
                    # INITIATE DUAL MONITORING EXACTLY as per flowchart
                    self.strategy_a_state['dual_monitoring_active'] = True
                    self.dinosay("Capture Path: INITIATE DUAL MONITORING - Listen for TWO SPECIFIC CONDITIONS", "INFO")
                
                # DUAL MONITORING for two conditions EXACTLY as per flowchart
                if self.strategy_a_state['dual_monitoring_active']:
                    condition_1 = latest['MA_100'] >= self.captured_fibo_100
                    condition_2 = latest['MA_200'] <= fibo_levels['Fibo_23.6%']
                    
                    self.dinosay(f"Dual Monitoring - Condition 1 (MA_100 ≥ Captured Fibo_1): {condition_1}", "INFO")
                    self.dinosay(f"Dual Monitoring - Condition 2 (MA_200 ≤ Fibo_0.236): {condition_2}", "INFO")
                    
                    if condition_1:
                        # THIS CONDITION MET FIRST EXACTLY as per flowchart
                        self.dinosay("Capture Path: Condition 1 (MA_100 ≥ Fibo_1) MET FIRST", "INFO")
                        self.exit_position(current_price, "Strategy A - Capture Path Exit (MA_100 ≥ Fibo_1)", ExitPath.CAPTURE)
                        return True
                        
                    elif condition_2:
                        # THIS CONDITION MET FIRST EXACTLY as per flowchart
                        self.dinosay("Capture Path: Condition 2 (MA_200 ≤ Fibo_0.236) MET FIRST", "WARNING")
                        self.dinosay("Capture Path: TRIGGER STOP LOSS MECHANISM", "WARNING")
                        if self.stop_loss_condition_met(data, fibo_levels):
                            self.execute_stop_loss_flow(data, current_price)
                        return True
        
        return False

    def strategy_b_exit_logic(self, data: pd.DataFrame, fibo_levels: Dict[str, float], current_price: float) -> bool:
        """Execute Strategy B exit logic EXACTLY as per flowchart"""
        latest = data.iloc[-1]
        
        # Handle MA_200 Wave separately (as per flowchart)
        if self.ma_wave_type == WaveType.MA_200:
            # Strategy B specific MA_200 Wave exit (enhanced)
            self.dinosay("Strategy B MA_200 Wave: Enhanced Exit Logic Activated", "INFO")
            return self.ma_200_wave_exit_logic(data, fibo_levels, current_price)
        
        # For MA_350/500 Wave in Strategy B
        # STEP 1: PROFIT TARGET EXACTLY as per flowchart
        if not self.strategy_b_steps['step_1_complete']:
            step_1_condition = (
                latest['MA_100'] >= fibo_levels['Fibo_76.4%'] and
                latest['MA_500'] >= fibo_levels['Fibo_76.4%']
            )
            
            if step_1_condition:
                self.dinosay("Strategy B Step 1 COMPLETE: MA_100 ≥ Fibo_0.764 AND MA_500 ≥ Fibo_0.764", "INFO")
                self.strategy_b_steps['step_1_complete'] = True
            else:
                return False
        
        # STEP 2: REVERSAL SIGNAL EXACTLY as per flowchart
        if self.strategy_b_steps['step_1_complete'] and not self.strategy_b_steps['step_2_complete']:
            step_2_condition = latest['MA_350'] <= latest['MA_500']
            
            if step_2_condition:
                self.dinosay("Strategy B Step 2 COMPLETE: MA_350 ≤ MA_500 (reversal signal)", "INFO")
                self.strategy_b_steps['step_2_complete'] = True
            else:
                return False
        
        # STEP 3: CAPTURE KEY LEVEL EXACTLY as per flowchart
        if self.strategy_b_steps['step_2_complete'] and not self.strategy_b_steps['step_3_complete']:
            self.captured_fibo_1 = fibo_levels['Fibo_100%']
            self.captured_fibo_100 = latest['MA_100']
            self.dinosay(f"Strategy B Step 3 COMPLETE: CAPTURED Fibo_1 level at MA_100: ${self.captured_fibo_100:.2f}", "INFO")
            self.strategy_b_steps['step_3_complete'] = True
        
        # STEP 4: DUAL MONITORING EXACTLY as per flowchart
        if self.strategy_b_steps['step_3_complete']:
            if not self.strategy_b_steps['step_4_active']:
                self.strategy_b_steps['step_4_active'] = True
                self.dinosay("Strategy B Step 4: DUAL MONITORING - Listen for TWO SPECIFIC CONDITIONS", "INFO")
            
            condition_1 = latest['MA_200'] <= fibo_levels['Fibo_23.6%']
            condition_2 = latest['MA_350'] >= self.captured_fibo_100
            
            self.dinosay(f"Dual Monitoring - Condition 1 (MA_200 ≤ Fibo_0.236): {condition_1}", "INFO")
            self.dinosay(f"Dual Monitoring - Condition 2 (MA_350 ≥ Captured Fibo_1): {condition_2}", "INFO")
            
            if condition_1:
                # CONDITION 1 MET FIRST EXACTLY as per flowchart
                self.dinosay("Strategy B: Condition 1 (MA_200 ≤ Fibo_0.236) MET FIRST", "WARNING")
                self.dinosay("Strategy B: TRIGGER STOP LOSS MECHANISM (Shared System)", "WARNING")
                if self.stop_loss_condition_met(data, fibo_levels):
                    self.execute_stop_loss_flow(data, current_price)
                return True
                
            elif condition_2:
                # CONDITION 2 MET FIRST EXACTLY as per flowchart
                self.dinosay("Strategy B: Condition 2 (MA_350 ≥ Captured Fibo_1) MET FIRST", "INFO")
                self.exit_position(current_price, "Strategy B - Enhanced Exit (MA_350 ≥ Fibo_1)", ExitPath.ENHANCED)
                return True
        
        return False

    def check_rsi_sequence(self, rsi_ma50: float) -> bool:
        """Check RSI sequence EXACTLY as per flowchart: 1. First ≥ 55, 2. THEN ≤ 52"""
        if self.rsi_state == 'waiting_for_55' and rsi_ma50 >= 55:
            self.dinosay("RSI Sequence: First condition met (RSI_MA50 ≥ 55)", "INFO")
            self.rsi_state = 'waiting_for_52'
            return False
            
        elif self.rsi_state == 'waiting_for_52' and rsi_ma50 <= 52:
            self.dinosay("RSI Sequence: Second condition met (RSI_MA50 ≤ 52)", "INFO")
            self.rsi_state = 'waiting_for_55'
            return True
            
        return False

    def exit_position(self, exit_price: float, reason: str, exit_path: ExitPath):
        """Exit current position with enhanced tracking"""
        if not self.position:
            return
            
        # Calculate P&L
        profit_loss = (exit_price - self.entry_price) * self.position_size
        
        # Calculate costs
        trade_value = self.entry_price * self.position_size
        commission = trade_value * self.config.commission_rate * 2  # Entry and exit
        slippage_cost = trade_value * self.config.slippage * 2
        
        net_pnl = profit_loss - commission - slippage_cost
        self.balance += net_pnl
        
        # Calculate risk metrics
        risk = abs(self.entry_price - self.stop_loss) * self.position_size
        reward = abs(exit_price - self.entry_price) * self.position_size
        risk_reward = reward / risk if risk > 0 else 0
        
        # Calculate drawdown for this trade
        trade_drawdown = self.risk_manager.calculate_drawdown(self.balance)
        
        # Generate trade ID
        trade_id = hashlib.md5(
            f"{self.entry_time}{self.entry_price}{exit_price}".encode()
        ).hexdigest()[:12]
        
        # Create enhanced trade record
        trade_record = TradeRecord(
            trade_id=trade_id,
            entry_time=self.entry_time,
            exit_time=datetime.now(),
            entry_price=self.entry_price,
            exit_price=exit_price,
            position_size=self.position_size,
            pnl=profit_loss,
            pnl_percent=(profit_loss / (self.entry_price * self.position_size)) * 100,
            commission=commission,
            slippage=slippage_cost,
            net_pnl=net_pnl,
            reason=reason,
            strategy=self.strategy_active.value if self.strategy_active else None,
            ma_wave=self.ma_wave_type.value if self.ma_wave_type else None,
            exit_path=exit_path.value,
            status=TradeStatus.EXECUTED.value,
            risk_reward=risk_reward,
            drawdown=trade_drawdown,
            tags=[self.strategy_active.value if self.strategy_active else "Unknown",
                  exit_path.value,
                  "Profitable" if net_pnl > 0 else "Loss"]
        )
        
        self.trades.append(asdict(trade_record))
        self.equity_curve.append(self.balance)
        
        # Update risk manager
        self.risk_manager.update_risk_metrics(net_pnl)
        
        # DinoSay execution
        self.dinosay(f"EXIT POSITION - {reason}", "INFO")
        self.dinosay(f"Exit Price: ${exit_price:.2f}, Gross P&L: ${profit_loss:.2f}", "INFO")
        self.dinosay(f"Net P&L: ${net_pnl:.2f} (Commission: ${commission:.2f}, Slippage: ${slippage_cost:.2f})", "INFO")
        self.dinosay(f"New Balance: ${self.balance:.2f}, Exit Path: {exit_path.value}", "INFO")
        self.dinosay(f"Risk-Reward: {risk_reward:.2f}, Drawdown: {trade_drawdown:.2f}%", "INFO")
        
        # Reset position
        self.reset_trading_state()

    def reset_trading_state(self):
        """Reset trading state after exit"""
        self.position = None
        self.position_size = 0
        self.entry_price = 0
        self.entry_time = None
        self.stop_loss = 0
        self.take_profit = 0
        self.strategy_active = None
        self.ma_wave_type = None
        self.phase = 1
        self.exit_path = None
        
        # Reset tracking states but keep Fibonacci captures for reference
        self.reset_tracking_states()

    def reset_tracking_states(self):
        """Reset all tracking states"""
        self.ma_200_wave_exit = {
            'step_1_complete': False,
            'step_2_complete': False,
        }
        
        self.strategy_b_steps = {
            'step_1_complete': False,
            'step_2_complete': False,
            'step_3_complete': False,
            'step_4_active': False,
        }
        
        self.strategy_a_state = {
            'phase_1_monitoring': False,
            'lesser_mas_checked': False,
            'lesser_mas_result': None,
            'step_3_path_active': False,
            'capture_path_active': False,
            'dual_monitoring_active': False,
            'branch_type': None,
            'new_fibo_1_captured': False,
        }
        
        self.rsi_state = 'waiting_for_55'
        self.rsi_stop_loss_state = 'waiting_for_53'
        
        # Note: We keep captured Fibonacci levels for reference
        # self.captured_fibo_1_dip = None  # Reset only this one

    # ============================================================================
    # MAIN STRATEGY EXECUTION
    # ============================================================================
    
    def run_strategy(self, data: pd.DataFrame):
        """Main strategy execution loop with ALL features"""
        try:
            # Execute resiliently
            return self.execute_resiliently(self._run_strategy_internal, data)
        except Exception as e:
            self.dinosay(f"Strategy execution failed: {str(e)}", "ERROR")
            raise
    
    def _run_strategy_internal(self, data: pd.DataFrame):
        """Internal strategy execution with exact flowchart matching"""
        self.dinosay("Starting strategy execution with ALL features...", "INFO")
        
        # Validate and prepare data
        is_valid, processed_data, message = self.validate_and_prepare_data(data)
        if not is_valid:
            self.dinosay(f"Cannot run strategy: {message}", "ERROR")
            return
            
        data = processed_data
        
        # Calculate Fibonacci levels
        high = data['high'].max()
        low = data['low'].min()
        fibo_levels = FibonacciLevels.calculate_levels(high, low)
        
        # Calculate daily difference
        daily_diff = self.calculate_daily_difference(data)
        current_price = data['close'].iloc[-1]
        
        self.dinosay(f"Current Price: ${current_price:.2f}, Daily Diff: {daily_diff:.2f}%", "INFO")
        
        # STRATEGY SELECTION EXACTLY as per flowchart
        if self.strategy_a_activation(daily_diff):
            self.strategy_active = StrategyType.A
            self.dinosay("STRATEGY A ACTIVATED: Daily Diff -1% to +4%", "INFO")
            
        elif self.strategy_b_activation(daily_diff):
            self.strategy_active = StrategyType.B
            self.dinosay("STRATEGY B ACTIVATED: Daily Diff ≤ -1%", "INFO")
        
        # ENTRY LOGIC EXACTLY as per flowchart
        if not self.position and self.strategy_active:
            self.dinosay("CHECK ENTRY SETUPS (MA_200/MA_350/MA_500 Wave)", "INFO")
            entry_setups = self.check_entry_setup(data, fibo_levels)
            
            for ma_wave in ['MA_200', 'MA_350', 'MA_500']:
                if entry_setups[ma_wave]:
                    self.dinosay(f"{ma_wave} WAVE ENTRY CONDITION MET", "INFO")
                    
                    details = entry_setups['details'][ma_wave]
                    
                    # Check WAIT FOR condition EXACTLY as per flowchart
                    if details['wait_for_condition']:
                        self.dinosay(f"{ma_wave}: WAIT FOR {ma_wave} ≥ Fibo_23.6%... Condition Met", "INFO")
                        
                        # FINAL VALIDATION EXACTLY as per flowchart
                        if details['final_validation']:
                            self.dinosay(f"{ma_wave}: FINAL VALIDATION PASSED - All MAs ≥ Fibo_23.6%", "INFO")
                            self.enter_position(data, ma_wave, current_price, fibo_levels)
                            break
                        else:
                            self.dinosay(f"{ma_wave}: FINAL VALIDATION FAILED", "INFO")
                    else:
                        self.dinosay(f"{ma_wave}: WAIT FOR {ma_wave} ≥ Fibo_23.6%... Still waiting", "INFO")
        
        # EXIT LOGIC EXACTLY as per flowchart
        if self.position:
            # First check stop loss condition
            if self.stop_loss_condition_met(data, fibo_levels):
                self.execute_stop_loss_flow(data, current_price)
                return
                
            # Strategy-specific exit logic
            if self.strategy_active == StrategyType.A:
                if self.strategy_a_exit_logic(data, fibo_levels, current_price):
                    return
                    
            elif self.strategy_active == StrategyType.B:
                if self.strategy_b_exit_logic(data, fibo_levels, current_price):
                    return

    # ============================================================================
    # REPORTING AND ANALYTICS
    # ============================================================================
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        if not self.trades:
            return {"message": "No trades executed"}
        
        # Basic metrics
        total_trades = len(self.trades)
        winning_trades = len([t for t in self.trades if t['net_pnl'] > 0])
        losing_trades = len([t for t in self.trades if t['net_pnl'] <= 0])
        
        total_pnl = sum(t['net_pnl'] for t in self.trades)
        gross_pnl = sum(t['pnl'] for t in self.trades)
        total_commission = sum(t['commission'] for t in self.trades)
        total_slippage = sum(t['slippage'] for t in self.trades)
        
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
        # Calculate returns
        returns = []
        for i in range(1, len(self.equity_curve)):
            ret = (self.equity_curve[i] - self.equity_curve[i-1]) / self.equity_curve[i-1]
            returns.append(ret)
        
        returns_array = np.array(returns)
        
        # Advanced analytics
        sharpe_ratio = self.performance_analytics.calculate_sharpe_ratio(returns_array)
        sortino_ratio = self.performance_analytics.calculate_sortino_ratio(returns_array)
        max_drawdown = self.performance_analytics.calculate_max_drawdown(self.equity_curve)
        calmar_ratio = self.performance_analytics.calculate_calmar_ratio(returns_array, max_drawdown)
        
        # Streak analysis
        streaks = self.performance_analytics.calculate_winning_streaks(self.trades)
        
        # Group by exit path
        exit_paths = {}
        for path in ExitPath:
            path_trades = [t for t in self.trades if t['exit_path'] == path.value]
            if path_trades:
                exit_paths[path.value] = {
                    'count': len(path_trades),
                    'total_pnl': sum(t['net_pnl'] for t in path_trades),
                    'avg_pnl': np.mean([t['net_pnl'] for t in path_trades]) if path_trades else 0,
                    'win_rate': (len([t for t in path_trades if t['net_pnl'] > 0]) / len(path_trades) * 100) if path_trades else 0
                }
        
        # DinoSay summary
        dinosay_summary = {
            'total_executions': len(self.dinosay_executions),
            'error_count': len([e for e in self.dinosay_executions if e.level == "ERROR"]),
            'warning_count': len([e for e in self.dinosay_executions if e.level == "WARNING"]),
            'last_execution': self.dinosay_executions[-1].timestamp if self.dinosay_executions else None
        }
        
        return {
            "summary": {
                "total_trades": total_trades,
                "winning_trades": winning_trades,
                "losing_trades": losing_trades,
                "win_rate": win_rate,
                "total_pnl": total_pnl,
                "gross_pnl": gross_pnl,
                "total_commission": total_commission,
                "total_slippage": total_slippage,
                "final_balance": self.balance,
                "initial_balance": self.initial_balance,
                "net_return": ((self.balance - self.initial_balance) / self.initial_balance) * 100,
                "max_drawdown": max_drawdown,
                "sharpe_ratio": sharpe_ratio,
                "sortino_ratio": sortino_ratio,
                "calmar_ratio": calmar_ratio
            },
            "streaks": streaks,
            "trades_by_strategy": {
                'A': len([t for t in self.trades if t['strategy'] == 'A']),
                'B': len([t for t in self.trades if t['strategy'] == 'B'])
            },
            "trades_by_wave": {
                'MA_200': len([t for t in self.trades if t['ma_wave'] == 'MA_200']),
                'MA_350': len([t for t in self.trades if t['ma_wave'] == 'MA_350']),
                'MA_500': len([t for t in self.trades if t['ma_wave'] == 'MA_500'])
            },
            "trades_by_exit_path": exit_paths,
            "dinosay_summary": dinosay_summary,
            "risk_metrics": {
                "current_drawdown": self.risk_manager.calculate_drawdown(self.balance),
                "max_drawdown": self.risk_manager.max_drawdown,
                "daily_pnl": self.risk_manager.daily_pnl,
                "consecutive_losses": self.risk_manager.consecutive_losses
            },
            "configuration": asdict(self.config) if self.config else {}
        }
    
    def print_comprehensive_status(self):
        """Print comprehensive status report"""
        status = []
        status.append("=" * 80)
        status.append("TRADING STRATEGY - COMPREHENSIVE STATUS REPORT")
        status.append("=" * 80)
        status.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        status.append("")
        
        # Core status
        status.append("[CORE STATUS]")
        status.append(f"Balance: ${self.balance:,.2f} (Initial: ${self.initial_balance:,.2f})")
        status.append(f"Position: {self.position or 'None'}")
        status.append(f"Strategy: {self.strategy_active.value if self.strategy_active else 'None'}")
        status.append(f"MA Wave: {self.ma_wave_type.value if self.ma_wave_type else 'None'}")
        status.append(f"Phase: {self.phase}")
        status.append("")
        
        # Risk metrics
        status.append("[RISK METRICS]")
        drawdown = self.risk_manager.calculate_drawdown(self.balance)
        status.append(f"Current Drawdown: {drawdown:.2f}% (Max: {self.risk_manager.max_drawdown:.2f}%)")
        status.append(f"Daily P&L: ${self.risk_manager.daily_pnl:,.2f}")
        status.append(f"Consecutive Losses: {self.risk_manager.consecutive_losses}")
        status.append("")
        
        # DinoSay summary
        status.append("[DINOSAY EXECUTIONS]")
        status.append(f"Total: {len(self.dinosay_executions)}")
        errors = len([e for e in self.dinosay_executions if e.level == "ERROR"])
        warnings = len([e for e in self.dinosay_executions if e.level == "WARNING"])
        status.append(f"Errors: {errors}, Warnings: {warnings}")
        if self.dinosay_executions:
            last = self.dinosay_executions[-1]
            status.append(f"Last: [{last.level}] {last.message[:50]}...")
        status.append("")
        
        # Recent trades
        if self.trades:
            status.append("[RECENT TRADES]")
            for trade in self.trades[-3:]:
                date_str = trade['exit_time'].strftime('%m-%d %H:%M')
                pnl_color = "✓" if trade['net_pnl'] > 0 else "✗"
                status.append(f"{date_str} {pnl_color} {trade['strategy']}-{trade['ma_wave']}: "
                            f"${trade['net_pnl']:+,.2f} ({trade['exit_path']})")
        
        print("\n".join(status))
    
    def save_state(self, filename: str = None):
        """Save strategy state to file"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"strategy_state_{timestamp}.pkl"
            
        state = {
            'balance': self.balance,
            'trades': self.trades,
            'equity_curve': self.equity_curve,
            'dinosay_executions': [asdict(e) for e in self.dinosay_executions],
            'config': asdict(self.config),
            'saved_at': datetime.now()
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(state, f)
            
        self.dinosay(f"Strategy state saved to {filename}", "INFO")
        return filename
    
    def load_state(self, filename: str):
        """Load strategy state from file"""
        with open(filename, 'rb') as f:
            state = pickle.load(f)
            
        self.balance = state['balance']
        self.trades = state['trades']
        self.equity_curve = state['equity_curve']
        
        # Restore DinoSay executions
        self.dinosay_executions = []
        for e_dict in state['dinosay_executions']:
            exec = DinoSayExecution(
                timestamp=e_dict['timestamp'],
                message=e_dict['message'],
                level=e_dict['level'],
                strategy=e_dict['strategy'],
                ma_wave=e_dict['ma_wave'],
                data_hash=e_dict.get('data_hash'),
                execution_id=e_dict.get('execution_id')
            )
            self.dinosay_executions.append(exec)
        
        self.dinosay(f"Strategy state loaded from {filename}", "INFO")
        return state

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def main():
    """Example usage of the complete trading strategy"""
    
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=2000, freq='H')
    base_price = 100
    prices = base_price + np.cumsum(np.random.randn(2000) * 0.5)
    
    data = pd.DataFrame({
        'timestamp': dates,
        'open': prices - np.random.rand(2000) * 0.1,
        'high': prices + np.random.rand(2000) * 0.2,
        'low': prices - np.random.rand(2000) * 0.2,
        'close': prices,
        'volume': np.random.randint(1000, 50000, 2000)
    })
    
    # Create custom configuration
    config = StrategyConfig(
        position_size_percent=0.1,
        max_drawdown_percent=15.0,
        daily_loss_limit=300.0,
        commission_rate=0.0005,
        slippage=0.0003,
        report_directory="my_reports"
    )
    
    # Initialize strategy with all features
    strategy = TradingStrategy(config=config, initial_balance=50000.0)
    
    # Run strategy in chunks (simulating real-time)
    chunk_size = 600  # Need enough for 500-period MA
    for i in range(chunk_size, len(data), 100):
        chunk = data.iloc[i-chunk_size:i]
        
        try:
            strategy.run_strategy(chunk)
        except Exception as e:
            print(f"Error at iteration {i}: {e}")
            break
        
        # Print status every 5 iterations
        if i % 500 == 0:
            strategy.print_comprehensive_status()
            
        # Auto-save reports if configured
        if strategy.config.auto_save_reports and i % 1000 == 0:
            for fmt in strategy.config.report_formats:
                strategy.report_generator.save_report(format=fmt)
    
    # Final reports
    print("\n" + "=" * 80)
    print("FINAL PERFORMANCE ANALYSIS")
    print("=" * 80)
    
    report = strategy.get_performance_report()
    
    # Print summary
    summary = report['summary']
    print(f"\nSUMMARY:")
    print(f"  Total Trades: {summary['total_trades']}")
    print(f"  Win Rate: {summary['win_rate']:.2f}%")
    print(f"  Total P&L: ${summary['total_pnl']:,.2f}")
    print(f"  Net Return: {summary['net_return']:.2f}%")
    print(f"  Sharpe Ratio: {summary['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown: {summary['max_drawdown']:.2f}%")
    
    # Print DinoSay summary
    dinosay = report['dinosay_summary']
    print(f"\nDINOSAY EXECUTIONS:")
    print(f"  Total: {dinosay['total_executions']}")
    print(f"  Errors: {dinosay['error_count']}")
    print(f"  Warnings: {dinosay['warning_count']}")
    
    # Generate and save reports
    print(f"\nGENERATING REPORTS...")
    for fmt in strategy.config.report_formats:
        result = strategy.report_generator.save_report(format=fmt)
        print(f"  {result}")
    
    # Save strategy state
    state_file = strategy.save_state()
    print(f"\nStrategy state saved to: {state_file}")
    
    # Print final DinoSay summary
    print(f"\nDINOSAY EXECUTION SUMMARY:")
    print(strategy.get_dinosay_summary(last_n=20))

if __name__ == "__main__":
    main()