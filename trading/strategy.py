import pandas as pd
import numpy as np
from typing import Tuple, Dict
import logging

from sqlalchemy import text

from config.config import Config
from data.database import DatabaseConnection

logger = logging.getLogger(__name__)


class MomentumStrategy:
    def __init__(self, db: DatabaseConnection, config: Config):
        self.db = db
        self.config = config

    def calculate_52_week_high(self, symbol: str) -> float:
        """Calculate 52-week high for a symbol"""
        query = text("""
            SELECT MAX(high) 
            FROM trading.price_data 
            WHERE symbol = :symbol 
            AND timestamp >= NOW() - INTERVAL '52 weeks'
        """)
        with self.db.engine.connect() as conn:
            return conn.execute(query, {"symbol": symbol}).scalar()

    def check_entry_signal(self, symbol: str, current_price: float) -> bool:
        """Check if current price breaks 52-week high"""
        high_52w = self.calculate_52_week_high(symbol)
        return current_price > high_52w if high_52w else False

    def check_exit_signal(self, symbol: str, entry_price: float,
                          current_price: float, position_size: float) -> Tuple[bool, float]:
        """Check exit conditions"""
        profit_target = entry_price * (1 + self.config.trading.profit_target_pct)

        if current_price >= profit_target and position_size == 1.0:
            return True, 0.5  # Exit 50% of position

        # Trailing stop logic
        trailing_stop = entry_price * (1 - self.config.trading.trailing_stop_pct)
        if current_price < trailing_stop:
            return True, position_size  # Exit remaining position

        return False, 0.0