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
from exceptions.trading_exceptions import TradingSystemError, ExecutionError
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

    def log_backtest_results(self, metrics: Dict):
        """Log summary of backtest results"""
        self.logger.info("Backtest Results Summary:")
        self.logger.info("-" * 40)
        self.logger.info(f"Total Return: {metrics['total_return']:.2%}")
        self.logger.info(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        self.logger.info(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
        self.logger.info(f"Total Trades: {metrics['total_trades']}")
        self.logger.info(f"Win Rate: {metrics['win_rate']:.2%}")
        self.logger.info(f"Profit Factor: {metrics['profit_factor']:.2f}")
        self.logger.info(f"Average Holding Period: {metrics['average_holding_period']:.1f} hours")
        self.logger.info("-" * 40)

    def _execute_trading_cycle(self, data_fetcher: DataFetcher, executor: TradeExecutor):
        """Execute a single trading cycle"""
        try:
            # Update market data
            for symbol in self.config.trading.symbols:
                data_fetcher.update_market_data(symbol)

            # Check for signals and execute trades
            current_time = datetime.now()
            strategy = MomentumStrategy(self.db, self.config)

            for symbol in self.config.trading.symbols:
                # Get current market data
                current_data = data_fetcher.get_latest_data(symbol)
                if current_data is None:
                    continue

                current_price = current_data['close']
                position = executor.get_position(symbol)

                # Check for entry signals
                if position == 0:
                    if strategy.check_entry_signal(symbol, current_price, current_time):
                        executor.enter_position(symbol, self.config.trading.position_size)

                # Check for exit signals
                elif position > 0:
                    exit_signal, exit_size = strategy.check_exit_signal(
                        symbol, executor.get_entry_price(symbol),
                        current_price, position, current_time
                    )
                    if exit_signal:
                        executor.exit_position(symbol, exit_size)

            # Sleep until next cycle
            time.sleep(self.config.trading.cycle_interval)

        except Exception as e:
            raise ExecutionError(f"Error in trading cycle: {str(e)}") from e