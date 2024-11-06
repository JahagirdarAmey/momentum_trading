from dataclasses import dataclass
from typing import List, Dict
from pathlib import Path

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
    entry_breakout_period: str = "52w"
    profit_target_pct: float = 0.04  # 4% target
    trailing_stop_pct: float = 0.02  # 2% trailing stop

@dataclass
class Config:
    db: DatabaseConfig = DatabaseConfig()
    trading: TradingConfig = TradingConfig()
    log_path: Path = Path("logs/trading.log")
    data_interval: str = "15m"
