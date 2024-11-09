from abc import ABC, abstractmethod
from typing import Tuple, Dict, Optional
import logging
from datetime import datetime

from config.config import Config
from data.database import DatabaseConnection

logger = logging.getLogger(__name__)

class BaseStrategy(ABC):
    def __init__(self, db: DatabaseConnection, config: Config):
        self.db = db
        self.config = config
        self.strategy_config = config.trading.strategy

    @abstractmethod
    def check_entry_signal(self, symbol: str, current_price: float, timestamp: Optional[datetime] = None) -> bool:
        pass

    @abstractmethod
    def check_exit_signal(self, symbol: str, entry_price: float, current_price: float,
                         position_size: float, timestamp: Optional[datetime] = None) -> Tuple[bool, float]:
        pass

    def get_strategy_params(self) -> Dict:
        """Return strategy parameters for logging/tracking"""
        return {
            "lookback_periods": self.strategy_config.lookback_periods,
            "initial_stoploss": self.strategy_config.initial_stoploss_pct,
            "profit_target": self.strategy_config.profit_target_pct,
            "trailing_stop": self.strategy_config.trailing_stop_pct
        }