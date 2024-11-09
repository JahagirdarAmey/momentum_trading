from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from datetime import datetime


@dataclass
class DatabaseConfig:
    host: str = "localhost"
    port: int = 5432
    database: str = "momentum_trading"
    user: str = "trader"
    password: str = "your_password"


@dataclass
class StrategyConfig:
    # Core strategy parameters
    lookback_periods: int = 252  # 52 weeks of trading days
    min_history_periods: int = 20  # Minimum required history

    # Entry conditions
    entry_breakout_threshold: float = 1.0  # Multiple of 52-week high

    # Exit conditions
    initial_stoploss_pct: float = 0.02  # 2% initial stoploss
    profit_target_pct: float = 0.04  # 4% target
    trailing_stop_pct: float = 0.02  # 2% trailing stop
    partial_exit_size: float = 0.5  # 50% exit at target

@dataclass
class TradingConfig:
    symbols: List[str]
    position_size: float
    cycle_interval: int = 60

@dataclass
class BacktestConfig:
    start_date: datetime
    end_date: datetime
    initial_capital: float = 100000.0
    commission_pct: float = 0.001  # 0.1% commission per trade


@dataclass
class Config:
    db: DatabaseConfig = DatabaseConfig()
    trading: TradingConfig = TradingConfig()
    backtest: Optional[BacktestConfig] = None
    log_path: Path = Path("logs/trading.log")
    data_interval: str = "15m"