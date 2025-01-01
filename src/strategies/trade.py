
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd

@dataclass
class Trade:
    symbol: str
    entry_date: datetime
    entry_price: float
    quantity: int
    exit_date: datetime = None
    exit_price: float = None
    exit_reason: str = None
    partial_exits: List[Dict] = None

    def __post_init__(self):
        self.partial_exits = []

    def holding_period_str(self) -> str:
        """Return holding period in a readable format"""
        if not self.exit_date:
            return "Position Open"

        delta = self.exit_date - self.entry_date
        days = delta.days
        hours = delta.seconds // 3600
        minutes = (delta.seconds % 3600) // 60

        if days > 0:
            return f"{days} days {hours} hrs {minutes} mins"
        elif hours > 0:
            return f"{hours} hrs {minutes} mins"
        else:
            return f"{minutes} mins"

    def holding_period_minutes(self) -> int:
        """Return total holding period in minutes"""
        if not self.exit_date:
            return 0
        delta = self.exit_date - self.entry_date
        return delta.days * 24 * 60 + delta.seconds // 60

    def pnl(self) -> float:
        """Calculate total PnL including partial exits"""
        total_pnl = 0

        # Add PnL from partial exits
        for exit in self.partial_exits:
            exit_pnl = (exit['price'] - self.entry_price) * exit['quantity']
            total_pnl += exit_pnl

        # Add PnL from final exit if exists
        if self.exit_price:
            remaining_quantity = self.quantity - sum(exit['quantity'] for exit in self.partial_exits)
            final_pnl = (self.exit_price - self.entry_price) * remaining_quantity
            total_pnl += final_pnl

        return total_pnl