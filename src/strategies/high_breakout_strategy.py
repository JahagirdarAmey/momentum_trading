# src/strategies/high_breakout_strategy.py
import pandas as pd
import numpy as np  # Correct import
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


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
        if not self.exit_price:
            return 0
        return (self.exit_price - self.entry_price) * self.quantity


class HighBreakoutStrategy:
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.trades: List[Trade] = []
        self.current_trade = None
        self.trailing_sl_pct = 0.02  # 2% trailing stop loss
        self.target_pct = 0.05  # 5% target for partial exit

    def calculate_52week_high(self, df: pd.DataFrame) -> pd.Series:
        """Calculate 52-week high for each point"""
        # Assuming 15-minute data with market hours 9:15 AM to 3:30 PM (6.25 hours)
        # 6.25 hours * 4 (15-min periods per hour) = 25 periods per day
        periods_per_day = 25
        window = 252 * periods_per_day  # 252 trading days
        return df['high'].rolling(window=window).max()

    def backtest(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Run backtest on the data"""
        try:
            # Reset index if it's datetime index
            if isinstance(df.index, pd.DatetimeIndex):
                df = df.reset_index()

            # Calculate 52-week high
            df['52_week_high'] = self.calculate_52week_high(df)

            # Initialize columns for signals
            df['buy_signal'] = False
            df['sell_signal'] = False
            df['trailing_sl'] = None
            df['partial_exit'] = False

            position_size = 0
            stop_loss = 0

            for i in range(1, len(df)):
                current_price = df.iloc[i]['close']

                if self.current_trade is None:
                    # Check for buy signal
                    if current_price > df.iloc[i - 1]['52_week_high']:
                        # Calculate position size
                        quantity = int(self.capital / current_price)
                        if quantity > 0:
                            self.current_trade = Trade(
                                symbol=symbol,
                                entry_date=df.iloc[i]['date'],
                                entry_price=current_price,
                                quantity=quantity
                            )
                            df.loc[i, 'buy_signal'] = True
                            stop_loss = current_price * (1 - self.trailing_sl_pct)
                            position_size = quantity

                else:
                    # Check for partial exit at 5% profit
                    target_price = self.current_trade.entry_price * (1 + self.target_pct)
                    if not self.current_trade.partial_exits and current_price >= target_price:
                        # Exit half position
                        exit_quantity = position_size // 2
                        self.current_trade.partial_exits.append({
                            'date': df.iloc[i]['date'],
                            'price': current_price,
                            'quantity': exit_quantity
                        })
                        position_size -= exit_quantity
                        df.loc[i, 'partial_exit'] = True

                        # Update trailing stop loss
                        stop_loss = max(stop_loss, current_price * (1 - self.trailing_sl_pct))

                    # Check trailing stop loss
                    if current_price <= stop_loss:
                        # Exit remaining position
                        self.current_trade.exit_date = df.iloc[i]['date']
                        self.current_trade.exit_price = current_price
                        self.current_trade.exit_reason = 'Stop Loss'
                        df.loc[i, 'sell_signal'] = True

                        self.trades.append(self.current_trade)
                        self.current_trade = None
                        position_size = 0
                    else:
                        # Update trailing stop loss
                        new_stop_loss = current_price * (1 - self.trailing_sl_pct)
                        stop_loss = max(stop_loss, new_stop_loss)

                    df.loc[i, 'trailing_sl'] = stop_loss

            return df

        except Exception as e:
            logger.error(f"Error in backtest: {str(e)}")
            raise

    def plot_backtest(self, df: pd.DataFrame, symbol: str):
        """Plot backtest results"""
        try:
            plt.figure(figsize=(15, 10))

            # Plot price and 52-week high
            plt.plot(df['date'], df['close'], label='Price', alpha=0.7)
            plt.plot(df['date'], df['52_week_high'], label='52-week High', alpha=0.5)

            # Plot buy signals
            buy_signals = df[df['buy_signal']]
            if not buy_signals.empty:
                plt.scatter(buy_signals['date'], buy_signals['close'],
                            marker='^', color='g', label='Buy Signal', s=100)

            # Plot sell signals
            sell_signals = df[df['sell_signal']]
            if not sell_signals.empty:
                plt.scatter(sell_signals['date'], sell_signals['close'],
                            marker='v', color='r', label='Sell Signal', s=100)

            # Plot partial exits
            partial_exits = df[df['partial_exit']]
            if not partial_exits.empty:
                plt.scatter(partial_exits['date'], partial_exits['close'],
                            marker='o', color='orange', label='Partial Exit', s=100)

            plt.title(f'Backtest Results for {symbol}')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.legend()
            plt.grid(True)
            plt.xticks(rotation=45)
            plt.tight_layout()

            return plt

        except Exception as e:
            logger.error(f"Error in plotting: {str(e)}")
            raise

    def generate_report(self) -> Dict:
        """Generate backtest report"""
        try:
            if not self.trades:
                return {"error": "No trades found"}

            total_trades = len(self.trades)
            profitable_trades = len([t for t in self.trades if t.pnl() > 0])
            total_pnl = sum(t.pnl() for t in self.trades)

            # Calculate holding periods
            holding_periods = [t.holding_period_minutes() for t in self.trades if t.exit_date]
            total_holding_minutes = sum(holding_periods)
            avg_holding_minutes = total_holding_minutes / len(holding_periods) if holding_periods else 0

            # Convert to days, hours, minutes
            total_days = total_holding_minutes // (24 * 60)
            remaining_minutes = total_holding_minutes % (24 * 60)
            total_hours = remaining_minutes // 60
            total_minutes = remaining_minutes % 60

            avg_days = avg_holding_minutes // (24 * 60)
            avg_remaining_minutes = avg_holding_minutes % (24 * 60)
            avg_hours = avg_remaining_minutes // 60
            avg_minutes = avg_remaining_minutes % 60

            return {
                "summary": {
                    "initial_capital": self.initial_capital,
                    "final_capital": self.initial_capital + total_pnl,
                    "total_return": (total_pnl / self.initial_capital) * 100,
                    "total_trades": total_trades,
                    "profitable_trades": profitable_trades,
                    "win_rate": (profitable_trades / total_trades) * 100 if total_trades > 0 else 0,
                    "total_holding_time": {
                        "days": int(total_days),
                        "hours": int(total_hours),
                        "minutes": int(total_minutes),
                        "total_minutes": total_holding_minutes
                    },
                    "average_holding_time": {
                        "days": int(avg_days),
                        "hours": int(avg_hours),
                        "minutes": int(avg_minutes),
                        "total_minutes": int(avg_holding_minutes)
                    }
                },
                "trades": [
                    {
                        "symbol": t.symbol,
                        "entry_date": t.entry_date.strftime('%Y-%m-%d %H:%M:%S'),
                        "entry_price": t.entry_price,
                        "quantity": t.quantity,
                        "partial_exits": [
                            {
                                "date": e['date'].strftime('%Y-%m-%d %H:%M:%S'),
                                "price": e['price'],
                                "quantity": e['quantity']
                            } for e in t.partial_exits
                        ],
                        "exit_date": t.exit_date.strftime('%Y-%m-%d %H:%M:%S') if t.exit_date else None,
                        "exit_price": t.exit_price,
                        "exit_reason": t.exit_reason,
                        "holding_period": t.holding_period_str(),
                        "holding_minutes": t.holding_period_minutes(),
                        "pnl": t.pnl(),
                        "return_pct": (t.pnl() / (t.entry_price * t.quantity)) * 100 if t.exit_price else 0
                    }
                    for t in self.trades
                ],
                "trade_stats": {
                    "max_profit_trade": max(self.trades, key=lambda t: t.pnl() if t.exit_price else 0).pnl(),
                    "max_loss_trade": min(self.trades, key=lambda t: t.pnl() if t.exit_price else 0).pnl(),
                    "avg_profit_per_trade": total_pnl / total_trades if total_trades > 0 else 0,
                    "longest_trade": max(self.trades, key=lambda t: t.holding_period_minutes()).holding_period_str(),
                    "shortest_trade": min(self.trades, key=lambda t: t.holding_period_minutes()).holding_period_str(),
                }
            }

        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
            raise