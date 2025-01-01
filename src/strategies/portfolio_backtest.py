from datetime import datetime
from typing import List, Dict

import numpy as np
import pandas as pd

from src.strategies.breakout_strategy import BreakoutStrategy
from src.strategies.portfolio_manager import PortfolioManager
from src.strategies.position import Position
from src.strategies.trade import Trade


class PortfolioBacktest:
    def __init__(self, symbols: List[str], start_date: datetime,
                 end_date: datetime, initial_capital: float = 1_000_000):
        self.portfolio = PortfolioManager(initial_capital)
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.current_date = None

        # Initialize strategies for each symbol
        self.strategies = {
            symbol: BreakoutStrategy(initial_capital=initial_capital)
            for symbol in symbols
        }

        # Data management
        self.data_feeds = {}  # symbol -> DataFrame
        self.synced_dates = None

        # Performance tracking
        self.daily_returns = []
        self.portfolio_values = []
        self.position_history = []

    def _load_all_data(self):
        """Load historical data for all symbols"""
        for symbol in self.symbols:
            try:
                # Assuming data loading function exists
                data = self._load_symbol_data(symbol)
                data = data[(data.index >= self.start_date) &
                            (data.index <= self.end_date)]
                self.data_feeds[symbol] = data
            except Exception as e:
                print(f"Error loading data for {symbol}: {str(e)}")

    def _sync_timestamps(self):
        """Synchronize timestamps across all symbols"""
        all_dates = set()
        for df in self.data_feeds.values():
            all_dates.update(df.index.tolist())
        self.synced_dates = sorted(list(all_dates))

    def _process_timeframe(self):
        """Process single timeframe across all symbols"""
        current_prices = {}

        # Update all positions
        for symbol, df in self.data_feeds.items():
            if self.current_date in df.index:
                current_data = df.loc[self.current_date]
                current_prices[symbol] = current_data['close']

                # Update existing position if any
                if symbol in self.portfolio.positions:
                    self.portfolio.positions[symbol].update(current_data)

        # Mark to market
        self.portfolio.current_capital = self.portfolio.update_portfolio_value(current_prices)

        # Check for new entries
        self._check_entry_signals()

    def _check_entry_signals(self):
        """Check entry signals across all symbols"""
        for symbol in self.symbols:
            if not self._can_take_new_position(symbol):
                continue

            strategy = self.strategies[symbol]
            current_data = self.data_feeds[symbol].loc[self.current_date]

            if self._entry_signal_triggered(symbol, current_data):
                size = self._calculate_position_size(symbol, current_data['close'])
                if size > 0:
                    self._create_position(symbol, size, current_data)

    def _can_take_new_position(self, symbol: str) -> bool:
        """Check if new position is allowed"""
        # Check maximum positions limit
        if len(self.portfolio.positions) >= self.portfolio.max_positions:
            return False

        # Check sector exposure
        sector = self._get_sector(symbol)
        if self._get_sector_exposure(sector) >= self.portfolio.max_sector_exposure:
            return False

        # Check correlation limits
        if not self._check_correlation_limits(symbol):
            return False

        return True

    def _entry_signal_triggered(self, symbol: str, current_data: pd.Series) -> bool:
        """Check if entry signal is triggered for symbol"""
        strategy = self.strategies[symbol]
        return strategy._check_entry_conditions(current_data)

    def _calculate_position_size(self, symbol: str, price: float) -> int:
        """Calculate position size based on risk parameters"""
        available_capital = self.portfolio.current_capital * self.portfolio.max_position_size
        return int(available_capital / price)

    def _generate_reports(self) -> Dict:
        """Generate comprehensive portfolio reports"""
        return {
            "portfolio_summary": self._get_portfolio_summary(),
            "position_summary": self._get_position_summary(),
            "risk_metrics": self._calculate_risk_metrics(),
            "daily_metrics": self._get_daily_metrics(),
            "trade_history": self._get_trade_history()
        }

    def _get_portfolio_summary(self) -> Dict:
        """Generate portfolio level summary"""
        if not self.portfolio_values:
            return {"error": "No portfolio data available"}

        returns = pd.Series(self.daily_returns)

        return {
            "initial_capital": self.portfolio.initial_capital,
            "final_capital": self.portfolio.current_capital,
            "total_return_pct": ((self.portfolio.current_capital /
                                  self.portfolio.initial_capital) - 1) * 100,
            "max_drawdown": self._calculate_max_drawdown(),
            "sharpe_ratio": self._calculate_sharpe_ratio(),
            "total_trades": len(self.portfolio.historical_trades),
            "win_rate": self._calculate_win_rate(),
            "avg_winner": self._calculate_avg_winner(),
            "avg_loser": self._calculate_avg_loser(),
            "max_concurrent_positions": self._get_max_concurrent_positions()
        }

    @staticmethod
    def _load_symbol_data(symbol: str) -> pd.DataFrame:
        """Load historical data for a symbol"""
        try:
            # Your data loading implementation here
            # This should return a DataFrame with columns: open, high, low, close, volume
            # and DateTimeIndex for timestamps
            pass
        except Exception as e:
            raise Exception(f"Error loading data for {symbol}: {str(e)}")

    def _create_position(self, symbol: str, size: int, current_data: pd.Series):
        """Create new position for symbol"""
        if symbol not in self.portfolio.positions:
            strategy = self.strategies[symbol]
            self.portfolio.positions[symbol] = Position(symbol, strategy)

        position = self.portfolio.positions[symbol]
        position.active_trade = Trade(
            symbol=symbol,
            entry_date=current_data.name,
            entry_price=current_data['close'],
            quantity=size
        )

    def _get_sector(self, symbol: str) -> str:
        """Get sector for symbol - implement your sector mapping"""
        # Implement your sector mapping logic here
        return "UNKNOWN"

    def _get_sector_exposure(self, sector: str) -> float:
        """Calculate total exposure for a sector"""
        sector_value = 0
        total_value = self.portfolio.current_capital

        for symbol, position in self.portfolio.positions.items():
            if self._get_sector(symbol) == sector:
                sector_value += position.get_position_value()

        return sector_value / total_value if total_value > 0 else 0

    def _check_correlation_limits(self, symbol: str) -> bool:
        """Check correlation with existing positions"""
        if not self.portfolio.positions:
            return True

        # Calculate returns for potential new position
        symbol_returns = self.data_feeds[symbol]['close'].pct_change().dropna()

        # Check correlation with existing positions
        for existing_symbol in self.portfolio.positions:
            existing_returns = self.data_feeds[existing_symbol]['close'].pct_change().dropna()

            # Align return series
            common_index = symbol_returns.index.intersection(existing_returns.index)
            if len(common_index) > 0:
                correlation = symbol_returns[common_index].corr(existing_returns[common_index])
                if abs(correlation) > self.portfolio.max_correlation:
                    return False

        return True

    def _get_position_summary(self) -> Dict:
        """Get summary of all positions"""
        return {
            symbol: {
                "current_value": position.get_position_value(),
                "entry_price": position.active_trade.entry_price if position.active_trade else None,
                "quantity": position.active_trade.quantity if position.active_trade else 0,
                "pnl": position.active_trade.pnl() if position.active_trade else 0,
                "sector": self._get_sector(symbol)
            } for symbol, position in self.portfolio.positions.items()
        }

    def _calculate_risk_metrics(self) -> Dict:
        """Calculate portfolio risk metrics"""
        returns = pd.Series(self.daily_returns)
        return {
            "volatility": returns.std() * np.sqrt(252),
            "var_95": returns.quantile(0.05),
            "var_99": returns.quantile(0.01),
            "max_drawdown": self._calculate_max_drawdown(),
            "beta": self._calculate_portfolio_beta()
        }

    def _get_daily_metrics(self) -> List[Dict]:
        """Get daily portfolio metrics"""
        return [
            {
                "date": date,
                "portfolio_value": value,
                "daily_return": return_val,
                "drawdown": dd
            }
            for date, value, return_val, dd in zip(
                self.synced_dates,
                self.portfolio_values,
                self.daily_returns,
                self._calculate_drawdown_series()
            )
        ]

    def _get_trade_history(self) -> List[Dict]:
        """Get history of all trades"""
        all_trades = []
        for symbol, position in self.portfolio.positions.items():
            all_trades.extend(position.historical_trades)

        return [
            {
                "symbol": t.symbol,
                "entry_date": t.entry_date,
                "entry_price": t.entry_price,
                "exit_date": t.exit_date,
                "exit_price": t.exit_price,
                "quantity": t.quantity,
                "pnl": t.pnl(),
                "return_pct": (t.pnl() / (t.entry_price * t.quantity)) * 100
            }
            for t in sorted(all_trades, key=lambda x: x.entry_date)
        ]

    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown percentage"""
        values = pd.Series(self.portfolio_values)
        peak = values.expanding(min_periods=1).max()
        drawdown = (values - peak) / peak
        return drawdown.min() * 100

    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio"""
        returns = pd.Series(self.daily_returns)
        if len(returns) < 2:
            return 0

        annual_return = returns.mean() * 252
        annual_vol = returns.std() * np.sqrt(252)
        risk_free_rate = 0.02  # Assuming 2% risk-free rate

        return (annual_return - risk_free_rate) / annual_vol if annual_vol != 0 else 0

    def _calculate_win_rate(self) -> float:
        """Calculate win rate percentage"""
        all_trades = []
        for position in self.portfolio.positions.values():
            all_trades.extend(position.historical_trades)

        if not all_trades:
            return 0

        winning_trades = sum(1 for trade in all_trades if trade.pnl() > 0)
        return (winning_trades / len(all_trades)) * 100

    def _calculate_avg_winner(self) -> float:
        """Calculate average winning trade return"""
        winners = [trade.pnl() for position in self.portfolio.positions.values()
                   for trade in position.historical_trades if trade.pnl() > 0]
        return np.mean(winners) if winners else 0

    def _calculate_avg_loser(self) -> float:
        """Calculate average losing trade return"""
        losers = [trade.pnl() for position in self.portfolio.positions.values()
                  for trade in position.historical_trades if trade.pnl() <= 0]
        return np.mean(losers) if losers else 0

    def _get_max_concurrent_positions(self) -> int:
        """Get maximum number of concurrent positions held"""
        position_counts = []
        for date in self.synced_dates:
            count = sum(1 for position in self.portfolio.positions.values()
                        if position.active_trade and
                        position.active_trade.entry_date <= date and
                        (not position.active_trade.exit_date or
                         position.active_trade.exit_date > date))
            position_counts.append(count)

        return max(position_counts) if position_counts else 0

    def _calculate_portfolio_beta(self) -> float:
        """Calculate portfolio beta against market index"""
        # Implement market beta calculation if market data is available
        return 1.0

    def _calculate_drawdown_series(self) -> List[float]:
        """Calculate drawdown series for all dates"""
        values = pd.Series(self.portfolio_values)
        peak = values.expanding(min_periods=1).max()
        drawdown = (values - peak) / peak
        return drawdown.tolist()