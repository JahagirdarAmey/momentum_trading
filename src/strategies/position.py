import pandas as pd

from src.strategies.breakout_strategy import BreakoutStrategy


class Position:
    def __init__(self, symbol: str, strategy: BreakoutStrategy):
        self.symbol = symbol
        self.strategy = strategy
        self.active_trade = None
        self.historical_trades = []

    def update(self, current_data: pd.Series):
        """Update position status with latest data"""
        if self.active_trade:
            self.strategy.update_position(current_data)
        return self.get_position_value(current_data['close'])

    def get_position_value(self, current_price: float) -> float:
        """Calculate current position value"""
        if not self.active_trade:
            return 0
        return current_price * (self.active_trade.quantity -
                              sum(exit['quantity'] for exit in self.active_trade.partial_exits))

    def update_position(self, current_data: pd.Series):
        """Update position status with latest data and check exits"""
        if not self.active_trade:
            return 0

        current_price = current_data['close']
        current_date = current_data.name  # Assuming DateTimeIndex

        # Check first partial target (1.5%)
        if (current_price >= self.active_trade.entry_price * 1.015 and
                not any(exit.get('target') == 'first' for exit in self.active_trade.partial_exits)):
            exit_quantity = int(self.active_trade.quantity * 0.4)  # 40% of position
            self.active_trade.partial_exits.append({
                'date': current_date,
                'price': current_price,
                'quantity': exit_quantity,
                'target': 'first'
            })

        # Check second target (3%)
        elif (current_price >= self.active_trade.entry_price * 1.03 and
              not any(exit.get('target') == 'second' for exit in self.active_trade.partial_exits)):
            remaining = self.active_trade.quantity - sum(exit['quantity'] for exit in self.active_trade.partial_exits)
            exit_quantity = int(remaining * 0.5)  # 50% of remaining
            self.active_trade.partial_exits.append({
                'date': current_date,
                'price': current_price,
                'quantity': exit_quantity,
                'target': 'second'
            })

        # Check final target (4.5%), stop loss (1.8%), or time exit (7 days)
        exit_triggered = False
        exit_reason = None

        if current_price >= self.active_trade.entry_price * 1.045:
            exit_triggered = True
            exit_reason = 'Final Target'
        elif current_price <= self.active_trade.entry_price * 0.982:
            exit_triggered = True
            exit_reason = 'Stop Loss'
        elif (current_date - self.active_trade.entry_date).days >= 7:
            exit_triggered = True
            exit_reason = 'Time Exit'

        if exit_triggered:
            self.active_trade.exit_date = current_date
            self.active_trade.exit_price = current_price
            self.active_trade.exit_reason = exit_reason
            self.historical_trades.append(self.active_trade)
            self.active_trade = None

        return self.get_position_value(current_price)
