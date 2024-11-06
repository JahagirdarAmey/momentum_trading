import logging

from config.config import Config
from data.database import Database
from trading.strategy import MomentumStrategy

logger = logging.getLogger(__name__)

class TradeExecutor:
    def __init__(self, db: Database, config: Config):
        self.db = db
        self.config = config
        self.strategy = MomentumStrategy(db, config)

    def execute_trades(self):
        """Main trading loop"""
        try:
            for symbol in self.config.trading.symbols:
                current_price = self.get_current_price(symbol)
                position = self.get_current_position(symbol)

                if position == 0 and self.strategy.check_entry_signal(symbol, current_price):
                    self.execute_entry(symbol, current_price)
                elif position > 0:
                    exit_signal, exit_size = self.strategy.check_exit_signal(
                        symbol, self.get_entry_price(symbol), current_price, position
                    )
                    if exit_signal:
                        self.execute_exit(symbol, current_price, exit_size)

        except Exception as e:
            logger.error(f"Error in trade execution: {str(e)}")
            raise