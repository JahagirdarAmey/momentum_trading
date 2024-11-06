import logging
import time
from pathlib import Path

from config.config import Config
from data.data_fetcher import DataFetcher
from data.database import Database
from trading.execution import TradeExecutor


def setup_logging(config: Config):
    """Setup logging configuration"""
    log_path = config.log_path
    log_path.parent.mkdir(exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )


def main():
    config = Config()
    setup_logging(config)

    try:
        db = Database(config.db)
        db.initialize_database()

        data_fetcher = DataFetcher(db, config)
        executor = TradeExecutor(db, config)

        # Main loop
        while True:
            for symbol in config.trading.symbols:
                data_fetcher.fetch_latest_data(symbol)
            executor.execute_trades()
            time.sleep(900)  # Wait 15 minutes

    except Exception as e:
        logging.error(f"Critical error in main loop: {str(e)}")
        raise


if __name__ == "__main__":
    main()