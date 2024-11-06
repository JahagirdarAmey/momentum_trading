from dataclasses import dataclass
from typing import List, Dict, Optional
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
class TradingConfig:
    symbols: List[str] = ("AAPL", "MSFT", "GOOGL", "AMZN", "NVDA")
    initial_capital: float = 100000.0
    position_size: float = 0.5  # 50% exit at target
    profit_target_pct: float = 0.04  # 4% target
    trailing_stop_pct: float = 0.02  # 2% trailing stop

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