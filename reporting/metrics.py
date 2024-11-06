from typing import Dict

import numpy as np
import pandas as pd
from sqlalchemy import text

from data.database import Database


class PerformanceMetrics:
    def __init__(self, db: Database):
        self.db = db

    def calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio"""
        return np.sqrt(252) * (returns.mean() / returns.std())

    def calculate_max_drawdown(self, equity_curve: pd.Series) -> float:
        """Calculate maximum drawdown"""
        rolling_max = equity_curve.cummax()
        drawdown = (equity_curve - rolling_max) / rolling_max
        return drawdown.min()

    def save_backtest_results(self, results: Dict):
        """Save backtest results to database"""
        query = text("""
            INSERT INTO trading.backtest_results 
            (start_date, end_date, initial_capital, final_capital, 
             sharpe_ratio, max_drawdown, total_trades, 
             winning_trades, losing_trades)
            VALUES 
            (:start_date, :end_date, :initial_capital, :final_capital,
             :sharpe_ratio, :max_drawdown, :total_trades,
             :winning_trades, :losing_trades)
        """)

        with self.db.engine.connect() as conn:
            conn.execute(query, results)
            conn.commit()