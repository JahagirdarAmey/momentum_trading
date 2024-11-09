import datetime
import logging
from typing import Dict

import numpy as np
import pandas as pd

from config.config import Config
from data.database import DatabaseConnection
from trading.momentum import MomentumStrategy

logger = logging.getLogger(__name__)

class Backtester:

    def __init__(self, db: DatabaseConnection, config: Config):
        self.db = db
        self.config = config
        self.strategy = MomentumStrategy(db, config)
        self.results = self._initialize_results()

    def _initialize_results(self) -> Dict:
        return {
            'trades': [],
            'equity_curve': [],
            'positions': {},
            'strategy_params': self.strategy.get_strategy_params()
        }

    def simulate_trades(self, symbol: str, data: pd.DataFrame):
        """Simulate trading for a single symbol"""
        position = 0
        entry_price = 0

        for i in range(len(data)):
            current_price = data['close'].iloc[i]
            current_time = data.index[i]

            if position == 0:
                if self.strategy.check_entry_signal(symbol, current_price, current_time):
                    position = self.config.trading.position_size
                    entry_price = current_price
                    self._record_trade(symbol, current_time, 'ENTRY', current_price, position)

            elif position > 0:
                exit_signal, exit_size = self.strategy.check_exit_signal(
                    symbol, entry_price, current_price, position, current_time
                )
                if exit_signal:
                    self._record_trade(symbol, current_time, 'EXIT', current_price, exit_size)
                    position -= exit_size
                    if position <= 0:
                        entry_price = 0

    def _record_trade(self, symbol: str, timestamp: datetime,
                      trade_type: str, price: float, quantity: float, note: str):
        """Record trade in backtest results with additional metadata"""
        commission = price * quantity * self.config.backtest.commission_pct

        trade = {
            'symbol': symbol,
            'timestamp': timestamp,
            'type': trade_type,
            'price': price,
            'quantity': quantity,
            'commission': commission,
            'note': note
        }
        self.results['trades'].append(trade)

        # Log trade details
        logger.info(f"Trade: {symbol} {trade_type} - Price: {price:.2f}, "
                    f"Qty: {quantity}, Note: {note}")

    def calculate_metrics(self) -> Dict:
        """Calculate backtest performance metrics"""
        trades_df = pd.DataFrame(self.results['trades'])
        if trades_df.empty:
            logger.warning("No trades executed in backtest")
            return self._get_empty_metrics()

        # Calculate trade PnL
        trades_df['pnl'] = 0.0
        current_position = {}  # Track open positions for each symbol

        for symbol in trades_df['symbol'].unique():
            symbol_trades = trades_df[trades_df['symbol'] == symbol].copy()
            position = 0
            cost_basis = 0

            for idx, trade in symbol_trades.iterrows():
                if trade['type'] == 'ENTRY':
                    position += trade['quantity']
                    cost_basis = trade['price']
                else:
                    # Calculate PnL for exits
                    trade_pnl = (trade['price'] - cost_basis) * trade['quantity']
                    trades_df.loc[idx, 'pnl'] = trade_pnl - trade['commission']
                    position -= trade['quantity']

                    # Update metrics for partial exits
                    if position > 0:
                        cost_basis = cost_basis  # Keep same cost basis for remaining position
                    else:
                        cost_basis = 0

        # Calculate detailed metrics
        total_pnl = trades_df['pnl'].sum()
        winning_trades = trades_df[trades_df['pnl'] > 0]
        losing_trades = trades_df[trades_df['pnl'] < 0]

        # Group trades by exit reason
        exit_analysis = trades_df[trades_df['type'].isin(['STOP_EXIT', 'PARTIAL_EXIT'])].groupby('note').agg({
            'pnl': ['count', 'mean', 'sum']
        }).round(2)

        metrics = {
            'total_return': total_pnl / self.config.backtest.initial_capital,
            'sharpe_ratio': self._calculate_sharpe_ratio(trades_df),
            'max_drawdown': self._calculate_max_drawdown(trades_df),
            'total_trades': len(trades_df[trades_df['type'] == 'ENTRY']),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / len(trades_df[trades_df['type'] != 'ENTRY']),
            'avg_win': winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0,
            'avg_loss': losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0,
            'profit_factor': abs(winning_trades['pnl'].sum()) / abs(losing_trades['pnl'].sum())
            if len(losing_trades) > 0 and losing_trades['pnl'].sum() != 0 else 0,
            'exit_analysis': exit_analysis.to_dict()
        }

        # Calculate additional risk metrics
        trades_df['return_pct'] = trades_df['pnl'] / (trades_df['price'] * trades_df['quantity'])
        metrics.update({
            'avg_return_per_trade': trades_df['return_pct'].mean(),
            'return_std': trades_df['return_pct'].std(),
            'max_consecutive_losses': self._calculate_max_consecutive_losses(trades_df),
            'average_holding_period': self._calculate_avg_holding_period(trades_df)
        })

        return metrics

    @staticmethod
    def _calculate_max_consecutive_losses(self, trades_df: pd.DataFrame) -> int:
        """Calculate maximum consecutive losing trades"""
        if trades_df.empty:
            return 0

        consecutive_losses = 0
        max_consecutive_losses = 0

        for pnl in trades_df['pnl']:
            if pnl < 0:
                consecutive_losses += 1
                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
            else:
                consecutive_losses = 0

        return max_consecutive_losses

    @staticmethod
    def _calculate_avg_holding_period(self, trades_df: pd.DataFrame) -> float:
        """Calculate average holding period in hours"""
        if trades_df.empty:
            return 0

        entry_exits = trades_df.groupby(['symbol', 'timestamp'])['type'].first()
        holding_periods = []

        for symbol in trades_df['symbol'].unique():
            symbol_trades = trades_df[trades_df['symbol'] == symbol]
            entries = symbol_trades[symbol_trades['type'] == 'ENTRY']
            exits = symbol_trades[symbol_trades['type'].isin(['STOP_EXIT', 'PARTIAL_EXIT'])]

            for entry in entries.itertuples():
                next_exit = exits[exits['timestamp'] > entry.timestamp].iloc[0] if len(exits) > 0 else None
                if next_exit is not None:
                    holding_period = (next_exit['timestamp'] - entry.timestamp).total_seconds() / 3600
                    holding_periods.append(holding_period)

        return np.mean(holding_periods) if holding_periods else 0

    @staticmethod
    def _calculate_sharpe_ratio(trades_df: pd.DataFrame) -> float:
        """
        Calculate the Sharpe ratio based on trade returns

        Args:
            trades_df: DataFrame containing trade information with 'pnl' column

        Returns:
            float: Annualized Sharpe ratio, or 0 if insufficient data
        """
        if trades_df.empty or 'pnl' not in trades_df.columns:
            return 0.0

        # Convert PnL to returns and drop non-exit trades
        trade_returns = trades_df[trades_df['type'].isin(['STOP_EXIT', 'PARTIAL_EXIT'])]['pnl'].values

        if len(trade_returns) < 2:  # Need at least 2 trades for meaningful calculation
            return 0.0

        # Calculate daily returns statistics
        returns_mean = np.mean(trade_returns)
        returns_std = np.std(trade_returns, ddof=1)  # Using sample standard deviation

        if returns_std == 0:  # Avoid division by zero
            return 0.0

        # Assuming 252 trading days per year
        trading_days = 252

        # Annualized the Sharpe ratio
        sharpe_ratio = (returns_mean / returns_std) * np.sqrt(trading_days)

        return float(sharpe_ratio)

    @staticmethod
    def _calculate_max_drawdown(trades_df: pd.DataFrame) -> float:
        """
        Calculate the maximum drawdown from peak equity

        Args:
            trades_df: DataFrame containing trade information with 'pnl' column

        Returns:
            float: Maximum drawdown as a percentage (0 to 1)
        """
        if trades_df.empty or 'pnl' not in trades_df.columns:
            return 0.0

        # Calculate cumulative PnL
        cumulative_pnl = trades_df['pnl'].cumsum()

        if len(cumulative_pnl) < 2:  # Need at least 2 trades for drawdown
            return 0.0

        # Calculate running maximum
        running_max = pd.Series(index=cumulative_pnl.index)
        running_max.iloc[0] = cumulative_pnl.iloc[0]

        for i in range(1, len(cumulative_pnl)):
            running_max.iloc[i] = max(running_max.iloc[i - 1], cumulative_pnl.iloc[i])

        # Calculate drawdowns
        drawdowns = (cumulative_pnl - running_max) / running_max

        # Get the maximum drawdown
        max_drawdown = abs(drawdowns.min()) if len(drawdowns) > 0 else 0.0

        return float(max_drawdown)

    @staticmethod
    def calculate_52_week_high(data: pd.DataFrame, current_index: int) -> float:
        """
        Calculate the 52-week high up to the current index

        Args:
            data: DataFrame containing price data with 'high' column
            current_index: Current position in the DataFrame

        Returns:
            float: 52-week high price
        """
        # Calculate number of periods for 52 weeks
        # Assuming daily data, 252 trading days per year
        lookback_periods = 252

        # Get start index for lookback, ensuring we don't go below 0
        start_index = max(0, current_index - lookback_periods + 1)

        # Extract the relevant price history
        price_history = data['high'].iloc[start_index:current_index + 1]

        # Handle case where we don't have enough history
        if len(price_history) < 20:  # Minimum required history
            logger.warning(f"Insufficient price history for 52-week high calculation. "
                           f"Only {len(price_history)} periods available.")
            return float('inf')  # Return infinity to prevent entry signals

        # Calculate 52-week high
        high_52w = price_history.max()

        # Log for debugging if needed
        logger.debug(f"52-week high calculated: {high_52w:.2f} "
                     f"using {len(price_history)} periods of history")

        return float(high_52w)

    @staticmethod
    def _get_empty_metrics(self) -> Dict:
        """Return empty metrics when no trades are executed"""
        return {
            'total_return': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'profit_factor': 0,
            'exit_analysis': {},
            'avg_return_per_trade': 0,
            'return_std': 0,
            'max_consecutive_losses': 0,
            'average_holding_period': 0
        }