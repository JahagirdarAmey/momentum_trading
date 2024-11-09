from datetime import datetime
from typing import Dict, Optional, Tuple
import logging
from sqlalchemy import text

from config.config import Config
from data.database import DatabaseConnection
from exceptions.trading_exceptions import ExecutionError

logger = logging.getLogger(__name__)


class TradeExecutor:
    """Handles trade execution and position management"""

    def __init__(self, db: DatabaseConnection, config: Config):
        self.db = db
        self.config = config
        self.positions: Dict[str, float] = {}
        self.entry_prices: Dict[str, float] = {}
        self._load_positions()

    def _load_positions(self):
        """Load existing positions from database"""
        try:
            with self.db.engine.connect() as conn:
                query = text("""
                    SELECT symbol, quantity, entry_price 
                    FROM trading.positions 
                    WHERE status = 'OPEN'
                """)
                result = conn.execute(query)

                for row in result:
                    self.positions[row.symbol] = row.quantity
                    self.entry_prices[row.symbol] = row.entry_price

        except Exception as e:
            logger.error(f"Error loading positions: {str(e)}")
            raise ExecutionError("Failed to load positions") from e

    def get_position(self, symbol: str) -> float:
        """Get current position size for a symbol"""
        return self.positions.get(symbol, 0.0)

    def get_entry_price(self, symbol: str) -> Optional[float]:
        """Get entry price for current position"""
        return self.entry_prices.get(symbol)

    def enter_position(self, symbol: str, quantity: float, price: Optional[float] = None):
        """
        Enter a new position

        Args:
            symbol: Trading symbol
            quantity: Position size to enter
            price: Entry price (optional, for backtesting)
        """
        try:
            if self.get_position(symbol) > 0:
                raise ExecutionError(f"Position already exists for {symbol}")

            # Execute trade
            entry_price = price or self._get_market_price(symbol)
            self._execute_trade(symbol, 'BUY', quantity, entry_price)

            # Update position tracking
            self.positions[symbol] = quantity
            self.entry_prices[symbol] = entry_price

            logger.info(f"Entered {quantity} {symbol} at {entry_price:.2f}")

        except Exception as e:
            raise ExecutionError(f"Failed to enter position: {str(e)}") from e

    def exit_position(self, symbol: str, quantity: Optional[float] = None,
                      price: Optional[float] = None):
        """
        Exit an existing position

        Args:
            symbol: Trading symbol
            quantity: Amount to exit (None for full position)
            price: Exit price (optional, for backtesting)
        """
        try:
            current_position = self.get_position(symbol)
            if current_position <= 0:
                raise ExecutionError(f"No position exists for {symbol}")

            exit_quantity = quantity or current_position
            if exit_quantity > current_position:
                raise ExecutionError(f"Exit size {exit_quantity} exceeds position {current_position}")

            # Execute trade
            exit_price = price or self._get_market_price(symbol)
            self._execute_trade(symbol, 'SELL', exit_quantity, exit_price)

            # Update position tracking
            new_position = current_position - exit_quantity
            if new_position <= 0:
                self.positions.pop(symbol, None)
                self.entry_prices.pop(symbol, None)
            else:
                self.positions[symbol] = new_position

            logger.info(f"Exited {exit_quantity} {symbol} at {exit_price:.2f}")

        except Exception as e:
            raise ExecutionError(f"Failed to exit position: {str(e)}") from e

    def _execute_trade(self, symbol: str, side: str, quantity: float, price: float):
        """Record trade in database"""
        try:
            with self.db.engine.connect() as conn:
                # Insert trade record
                query = text("""
                    INSERT INTO trading.trades 
                    (symbol, side, quantity, price, timestamp)
                    VALUES (:symbol, :side, :quantity, :price, :timestamp)
                """)

                conn.execute(query, {
                    "symbol": symbol,
                    "side": side,
                    "quantity": quantity,
                    "price": price,
                    "timestamp": datetime.now()
                })

                # Update position
                if side == 'BUY':
                    position_query = text("""
                        INSERT INTO trading.positions 
                        (symbol, quantity, entry_price, status)
                        VALUES (:symbol, :quantity, :price, 'OPEN')
                        ON CONFLICT (symbol) DO UPDATE 
                        SET quantity = positions.quantity + :quantity,
                            entry_price = :price
                        WHERE positions.status = 'OPEN'
                    """)
                else:
                    position_query = text("""
                        UPDATE trading.positions 
                        SET quantity = quantity - :quantity,
                            status = CASE 
                                WHEN quantity - :quantity <= 0 THEN 'CLOSED'
                                ELSE status 
                            END
                        WHERE symbol = :symbol AND status = 'OPEN'
                    """)

                conn.execute(position_query, {
                    "symbol": symbol,
                    "quantity": quantity,
                    "price": price
                })

        except Exception as e:
            raise ExecutionError(f"Database error executing trade: {str(e)}") from e

    def _get_market_price(self, symbol: str) -> float:
        """Get current market price for a symbol"""
        # In real implementation, this would fetch live price
        # For now, using placeholder
        return 0.0  # Replace with actual price fetching