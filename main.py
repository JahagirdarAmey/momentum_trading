# main.py
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
import argparse
import sys

from config.config import BacktestConfig, Config
from data.data_fetcher import DataFetcher
from data.database import DatabaseConnection
from trading.backtester import Backtester
from trading.execution import TradeExecutor
from trading.strategy import MomentumStrategy


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


def run_backtest(config: Config):
    """Run backtest with given configuration"""
    logger = logging.getLogger(__name__)
    logger.info("Starting backtest...")

    try:
        db = DatabaseConnection(config.db)
        backtester = Backtester(db, config)

        # Process each symbol
        for symbol in config.trading.symbols:
            logger.info(f"Running backtest for {symbol}")
            data = backtester.load_backtest_data(symbol)

            if data.empty:
                logger.warning(f"No data found for {symbol} in specified date range")
                continue

            backtester.simulate_trades(symbol, data)

        # Calculate and save results
        metrics = backtester.calculate_metrics()
        backtester.save_results()

        # Log results summary
        logger.info("Backtest completed successfully")
        logger.info("Performance Metrics:")
        for metric, value in metrics.items():
            logger.info(f"{metric}: {value}")

        return metrics

    except Exception as e:
        logger.error(f"Backtest failed: {str(e)}")
        raise


def run_live_trading(config: Config):
    """Run live trading with given configuration"""
    logger = logging.getLogger(__name__)
    logger.info("Starting live trading...")

    try:
        db = DatabaseConnection(config.db)
        data_fetcher = DataFetcher(db, config)
        strategy = MomentumStrategy(db, config)
        executor = TradeExecutor(db, config)

        while True:
            try:
                current_time = datetime.now()

                # Only trade during market hours (9:30 AM - 4:00 PM ET)
                if not is_market_hours(current_time):
                    logger.info("Outside market hours, waiting...")
                    time.sleep(300)  # Sleep for 5 minutes
                    continue

                # Fetch latest data for all symbols
                for symbol in config.trading.symbols:
                    logger.info(f"Fetching data for {symbol}")
                    latest_data = data_fetcher.fetch_latest_data(symbol)
                    data_fetcher.save_to_database(symbol, latest_data)

                # Execute trading logic
                executor.execute_trades()

                # Wait for next 15-minute interval
                next_interval = calculate_next_interval(current_time)
                sleep_seconds = (next_interval - datetime.now()).total_seconds()
                if sleep_seconds > 0:
                    logger.info(f"Waiting {sleep_seconds:.0f} seconds until next interval")
                    time.sleep(sleep_seconds)

            except Exception as e:
                logger.error(f"Error in trading loop: {str(e)}")
                time.sleep(60)  # Wait a minute before retrying

    except KeyboardInterrupt:
        logger.info("Shutting down trading system...")
    except Exception as e:
        logger.error(f"Critical error in live trading: {str(e)}")
        raise


def is_market_hours(current_time: datetime) -> bool:
    """Check if current time is during market hours (9:30 AM - 4:00 PM ET)"""
    # Convert current time to ET
    et_time = current_time - timedelta(hours=5)  # Simplified EST conversion

    # Check if weekend
    if et_time.weekday() > 4:  # Saturday = 5, Sunday = 6
        return False

    market_open = et_time.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = et_time.replace(hour=16, minute=0, second=0, microsecond=0)

    return market_open <= et_time <= market_close


def calculate_next_interval(current_time: datetime) -> datetime:
    """Calculate the next 15-minute interval"""
    minute = current_time.minute
    next_interval = current_time.replace(
        minute=(minute // 15 + 1) * 15,
        second=0,
        microsecond=0
    )

    if next_interval <= current_time:
        next_interval = next_interval + timedelta(minutes=15)

    return next_interval


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Momentum Trading System")
    parser.add_argument(
        '--mode',
        choices=['live', 'backtest'],
        required=True,
        help='Trading mode: live or backtest'
    )
    parser.add_argument(
        '--start-date',
        help='Start date for backtest (YYYY-MM-DD)',
        type=lambda s: datetime.strptime(s, '%Y-%m-%d')
    )
    parser.add_argument(
        '--end-date',
        help='End date for backtest (YYYY-MM-DD)',
        type=lambda s: datetime.strptime(s, '%Y-%m-%d')
    )
    parser.add_argument(
        '--config',
        help='Path to configuration file',
        type=str,
        default='config.yaml'
    )

    args = parser.parse_args()

    if args.mode == 'backtest' and (not args.start_date or not args.end_date):
        parser.error("Backtest mode requires both --start-date and --end-date")

    return args


def load_config(config_path: str, args: argparse.Namespace) -> Config:
    """Load and validate configuration"""
    # In a real implementation, you might want to load from a YAML file
    config = Config()

    if args.mode == 'backtest':
        config.backtest = BacktestConfig(
            start_date=args.start_date,
            end_date=args.end_date
        )

    return config


def main():
    """Main entry point"""
    args = parse_args()
    config = load_config(args.config, args)
    setup_logging(config)

    logger = logging.getLogger(__name__)
    logger.info(f"Starting trading system in {args.mode} mode")

    try:
        if args.mode == 'live':
            run_live_trading(config)
        else:
            run_backtest(config)

    except Exception as e:
        logger.error(f"Trading system failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()