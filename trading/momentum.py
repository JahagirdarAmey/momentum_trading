from typing import Tuple, Optional
from datetime import datetime
import logging

from sqlalchemy import text

from trading.base import BaseStrategy

logger = logging.getLogger(__name__)


class MomentumStrategy(BaseStrategy):
    def calculate_52_week_high(self, symbol: str, current_time: Optional[datetime] = None) -> float:
        """Calculate 52-week high for a symbol"""
        time_condition = "timestamp >= NOW() - INTERVAL '52 weeks'"
        if current_time:  # For backtesting
            time_condition = "timestamp >= :current_time - INTERVAL '52 weeks' AND timestamp <= :current_time"

        query = text(f"""
            SELECT MAX(high) 
            FROM trading.price_data 
            WHERE symbol = :symbol 
            AND {time_condition}
        """)

        params = {"symbol": symbol}
        if current_time:
            params["current_time"] = current_time

        with self.db.engine.connect() as conn:
            result = conn.execute(query, params).scalar()
            return result if result is not None else float('inf')

    def check_entry_signal(self, symbol: str, current_price: float,
                           timestamp: Optional[datetime] = None) -> bool:
        """Check if current price breaks 52-week high"""
        high_52w = self.calculate_52_week_high(symbol, timestamp)
        threshold = high_52w * self.strategy_config.entry_breakout_threshold
        return current_price > threshold

    def check_exit_signal(self, symbol: str, entry_price: float, current_price: float,
                          position_size: float, timestamp: Optional[datetime] = None) -> Tuple[bool, float]:
        """Check exit conditions"""
        # Initial stoploss check
        if position_size == self.config.trading.position_size:  # Full position
            initial_stop = entry_price * (1 - self.strategy_config.initial_stoploss_pct)
            if current_price <= initial_stop:
                return True, position_size

        # Profit target check for partial exit
        profit_target = entry_price * (1 + self.strategy_config.profit_target_pct)
        if current_price >= profit_target and position_size == self.config.trading.position_size:
            return True, self.strategy_config.partial_exit_size

        # Trailing stop check for remaining position
        if position_size == self.strategy_config.partial_exit_size:  # After partial exit
            trailing_stop = entry_price * (1 - self.strategy_config.trailing_stop_pct)
            if current_price < trailing_stop:
                return True, position_size

        return False, 0.0
