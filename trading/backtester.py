import datetime
import logging
from typing import Dict

import numpy as np
import pandas as pd

from config.config import Config
from data.database import DatabaseConnection


logger = logging.getLogger(__name__)

class Backtester:
    def __init__(self, db: DatabaseConnection, config: Config):
        self.db = db
        self.config = config
        self.results = {
            'trades': [],
            'equity_curve': [],
            'positions': {}
        }
        self.initial_stoploss_pct = 0.02  # 2% fixed initial stoploss

    def simulate_trades(self, symbol: str, data: pd.DataFrame):
        """Simulate trading for a single symbol"""
        position = 0
        entry_price = 0
        highest_price = 0
        initial_stoploss = 0
        partial_exit_hit = False

        for i in range(len(data)):
            current_price = data['close'].iloc[i]
            current_time = data.index[i]

            if position == 0:
                # Check entry condition
                high_52w = self.calculate_52_week_high(data, i)
                if current_price > high_52w:
                    position = 1
                    entry_price = current_price
                    highest_price = current_price
                    initial_stoploss = entry_price * (1 - self.initial_stoploss_pct)
                    partial_exit_hit = False

                    self._record_trade(
                        symbol=symbol,
                        timestamp=current_time,
                        trade_type='ENTRY',
                        price=current_price,
                        quantity=self.config.trading.position_size,
                        note="Entry on 52-week high breakout"
                    )

            elif position > 0:
                # Check for initial stoploss before partial exit
                if not partial_exit_hit and current_price <= initial_stoploss:
                    # Exit entire position on initial stoploss
                    self._record_trade(
                        symbol=symbol,
                        timestamp=current_time,
                        trade_type='STOP_EXIT',
                        price=current_price,
                        quantity=position,
                        note="Initial 2% stoploss hit"
                    )
                    position = 0
                    continue

                # Check profit target for partial exit
                if not partial_exit_hit and current_price >= entry_price * (1 + self.config.trading.profit_target_pct):
                    # Exit half position
                    partial_exit_hit = True
                    position = 0.5
                    self._record_trade(
                        symbol=symbol,
                        timestamp=current_time,
                        trade_type='PARTIAL_EXIT',
                        price=current_price,
                        quantity=0.5,
                        note="4% profit target hit - partial exit"
                    )
                    # Reset highest price for trailing stop
                    highest_price = current_price
                    continue

                # After partial exit, use trailing stop
                if partial_exit_hit:
                    highest_price = max(highest_price, current_price)
                    trail_price = highest_price * (1 - self.config.trading.trailing_stop_pct)

                    if current_price < trail_price:
                        # Exit remaining position
                        self._record_trade(
                            symbol=symbol,
                            timestamp=current_time,
                            trade_type='STOP_EXIT',
                            price=current_price,
                            quantity=position,
                            note="Trailing stop hit after partial exit"
                        )
                        position = 0

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