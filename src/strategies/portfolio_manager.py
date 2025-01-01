from typing import Dict

from src.strategies.breakout_strategy import BreakoutStrategy
from src.strategies.position import Position


class PortfolioManager:
    def __init__(self, initial_capital: float = 1_000_000):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.historical_trades = []

        # Risk limits
        self.max_positions = 10
        self.max_position_size = 0.15  # 15% per position
        self.max_sector_exposure = 0.30  # 30% per sector
        self.max_correlation = 0.7

        # Performance tracking
        self.daily_returns = []
        self.portfolio_values = []
        self.position_history = []

    def add_position(self, symbol: str, strategy: BreakoutStrategy):
        """Add new position to portfolio"""
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol, strategy)

    def update_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """Update portfolio value based on current prices"""
        total_value = self.cash
        for symbol, position in self.positions.items():
            if symbol in current_prices:
                position_value = position.get_position_value(current_prices[symbol])
                total_value += position_value
        return total_value

    def get_position_exposure(self, symbol: str) -> float:
        """Calculate exposure for a specific position"""
        if symbol not in self.positions:
            return 0
        return self.positions[symbol].get_position_value() / self.current_capital