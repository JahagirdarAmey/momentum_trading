# main.py
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict
import argparse
import sys
import yaml
from dataclasses import asdict

from pip._internal.exceptions import ConfigurationError

from config.config import BacktestConfig, Config, TradingConfig, StrategyConfig
from data.data_fetcher import DataFetcher
from data.database import DatabaseConnection
from trading.backtester import Backtester
from trading.executor import TradeExecutor
from trading.momentum import MomentumStrategy


class TradingSystem:
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.db = DatabaseConnection(config.db)
        self.running = False

    def setup_logging(self):
        """Setup logging configuration"""
        log_path = self.config.log_path
        log_path.parent.mkdir(exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path),
                logging.StreamHandler()
            ]
        )

    def run_backtest(self) -> Dict:
        """Run backtest with given configuration"""
        if not self.config.backtest:
            raise ConfigurationError("Backtest configuration not provided")

        self.logger.info("Starting backtest...")
        try:
            backtester = Backtester(self.db, self.config)
            metrics = {}

            # Process each symbol
            for symbol in self.config.trading.symbols:
                self.logger.info(f"Running backtest for {symbol}")
                data = backtester.load_backtest_data(symbol)

                if data.empty:
                    self.logger.warning(f"No data found for {symbol} in specified date range")
                    continue

                backtester.simulate_trades(symbol, data)

            # Calculate and save results
            metrics = backtester.calculate_metrics()
            backtester.save_results()

            # Log results summary
            self.log_backtest_results(metrics)
            return metrics

        except Exception as e:
            raise TradingSystemError(f"Backtest failed: {str(e)}") from e

    def run_live_trading(self):
        """Run live trading with given configuration"""
        self.logger.info("Starting live trading...")

        try:
            self.running = True
            data_fetcher = DataFetcher(self.db, self.config)
            executor = TradeExecutor(self.db, self.config)

            while self.running:
                try:
                    self._execute_trading_cycle(data_fetcher, executor)
                except Exception as e:
                    self.logger.error(f"Error in trading cycle: {str(e)}")
                    time.sleep(60)  # Wait before retrying

        except KeyboardInterrupt:
            self.shutdown()
        except Exception as e:
            raise TradingSystemError(f"Critical error in live trading: {str(e)}") from e

    def shutdown(self):
        """Gracefully shutdown the trading system"""
        self.logger.info("Initiating trading system shutdown...")
        self.running = False