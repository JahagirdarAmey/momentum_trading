import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd

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


class BreakoutStrategy:
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.trades: List[Trade] = []
        self.current_trade = None
        self.lookback_period = 20  # For volatility calculation
        self.risk_percent = 0.01  # 1% risk per trade
        self.max_holding_days = 7  # Extended from 5 to 7 days
        self.rsi_period = 14  # RSI period for trend confirmation

        # Modified profit targets
        self.first_target_pct = 0.015  # 1.5% for first exit
        self.second_target_pct = 0.03  # 3% for second exit
        self.final_target_pct = 0.045  # 4.5% for final exit
        self.partial_exit_pct = 0.40  # 40% size for first exit
        self.stop_loss_pct = 0.018  # 1.8% stop loss

        # Trend confirmation parameters
        self.ema_short = 10  # Faster EMA for quick moves
        self.ema_long = 21  # Slower EMA for trend
        self.rsi_oversold = 40  # RSI levels
        self.rsi_overbought = 60

    @staticmethod
    def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def calculate_volatility(df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate price volatility using standard deviation of returns"""
        returns = df['close'].pct_change()
        return returns.rolling(window=period).std()

    @staticmethod
    def is_same_day(date1: datetime, date2: datetime) -> bool:
        """Check if two dates are on the same trading day"""
        return date1.date() == date2.date()

    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Average True Range"""
        high = df['high']
        low = df['low']
        close = df['close'].shift(1)

        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()

        return atr

    @staticmethod
    def is_below_ema(current_price: float, ema_value: float) -> bool:
        """Check if price is below EMA"""
        return current_price < ema_value

    def plot_balance_chart(self, interval='daily') -> plt.Figure:
        """
        Generate a line chart showing balance progression over time

        Args:
            interval (str): 'daily', 'weekly', or 'monthly'
        """
        try:
            # Sort trades by entry date
            sorted_trades = sorted(self.trades, key=lambda x: x.entry_date)
            if not sorted_trades:
                return None

            # Initialize balance tracking
            balances = {}
            running_balance = self.initial_capital

            # Record initial balance at start date
            start_date = sorted_trades[0].entry_date
            balances[start_date] = running_balance

            # Track all events (trades and partial exits) chronologically
            all_events = []

            for trade in sorted_trades:
                # Add partial exits
                for partial_exit in trade.partial_exits:
                    partial_pnl = (partial_exit['price'] - trade.entry_price) * partial_exit['quantity']
                    all_events.append({
                        'date': partial_exit['date'],
                        'pnl': partial_pnl
                    })

                # Add final trade exit
                if trade.exit_date and trade.exit_price:
                    remaining_quantity = trade.quantity - sum(exit['quantity'] for exit in trade.partial_exits)
                    final_pnl = (trade.exit_price - trade.entry_price) * remaining_quantity
                    all_events.append({
                        'date': trade.exit_date,
                        'pnl': final_pnl
                    })

            # Sort events chronologically
            all_events.sort(key=lambda x: x['date'])

            # Calculate balances for each day
            for event in all_events:
                running_balance += event['pnl']
                balances[event['date']] = running_balance

            # Convert to DataFrame
            df = pd.DataFrame(list({'date': date, 'balance': balance}
                                   for date, balance in balances.items()))
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            df.sort_index(inplace=True)

            # Resample based on interval
            if interval == 'weekly':
                df = df.resample('W').last().fillna(method='ffill')
            elif interval == 'monthly':
                df = df.resample('M').last().fillna(method='ffill')

            # Create the plot
            plt.figure(figsize=(15, 8))

            # Plot line chart
            plt.plot(df.index, df['balance'],
                     color='blue',
                     linewidth=2,
                     marker='o',
                     markersize=4)

            # Customize the plot
            plt.title(f'Account Balance Progression ({interval.capitalize()})',
                      fontsize=14,
                      pad=20)
            plt.ylabel('Balance ($)', fontsize=12)
            plt.xlabel('Date', fontsize=12)

            # Format y-axis labels as currency
            plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

            # Rotate x-axis labels
            plt.xticks(rotation=45)

            # Add grid for better readability
            plt.grid(True, linestyle='--', alpha=0.7)

            # Calculate and show key metrics
            initial_balance = df['balance'].iloc[0]
            final_balance = df['balance'].iloc[-1]
            total_return = ((final_balance - initial_balance) / initial_balance) * 100

            # Add annotations for min and max values
            min_balance = df['balance'].min()
            max_balance = df['balance'].max()
            min_date = df[df['balance'] == min_balance].index[0]
            max_date = df[df['balance'] == max_balance].index[0]

            plt.annotate(f'Min: ${min_balance:,.0f}',
                         xy=(min_date, min_balance),
                         xytext=(10, -20),
                         textcoords='offset points',
                         ha='left',
                         va='top',
                         bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

            plt.annotate(f'Max: ${max_balance:,.0f}',
                         xy=(max_date, max_balance),
                         xytext=(10, 20),
                         textcoords='offset points',
                         ha='left',
                         va='bottom',
                         bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

            plt.figtext(0.99, 0.01,
                        f'Total Return: {total_return:,.2f}%\n' +
                        f'Initial: ${initial_balance:,.0f}\n' +
                        f'Final: ${final_balance:,.0f}',
                        ha='right',
                        va='bottom',
                        fontsize=10,
                        bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

            plt.tight_layout()
            return plt.gcf()

        except Exception as e:
            logger.error(f"Error in plotting balance progression: {str(e)}")
            raise

    @staticmethod
    def is_uptrend(df: pd.DataFrame, current_idx: int) -> bool:
        """
        Determine if stock is in uptrend using 20-period EMA
        """
        if current_idx < 20:  # Not enough data
            return False

        current_price = df.iloc[current_idx]['close']
        ema_20 = df['close'].ewm(span=20, adjust=False).mean().iloc[current_idx]
        return current_price > ema_20

    def backtest(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Run backtest on the data"""
        try:
            if isinstance(df.index, pd.DatetimeIndex):
                df = df.reset_index()

            # Calculate indicators
            window = 252 * 25  # 52-week high
            df['52_week_high'] = df['high'].rolling(window=window).max()

            # EMAs and RSI
            df['ema_short'] = df['close'].ewm(span=self.ema_short, adjust=False).mean()
            df['ema_long'] = df['close'].ewm(span=self.ema_long, adjust=False).mean()
            df['rsi'] = self.calculate_rsi(df, self.rsi_period)

            # Calculate volume metrics
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']

            # Initialize columns
            df['buy_signal'] = False
            df['sell_signal'] = False
            df['profit_target_exit'] = False
            df['stop_loss_exit'] = False
            df['time_exit'] = False
            df['stop_loss_price'] = None
            df['first_target_price'] = None
            df['second_target_price'] = None
            df['final_target_price'] = None

            position_size = 0

            for i in range(1, len(df)):
                current_row = df.iloc[i]
                current_price = current_row['close']
                current_date = current_row['date']

                if self.current_trade is None:
                    # Entry conditions
                    if (current_price > df.iloc[i - 1]['52_week_high'] and
                            current_row['ema_short'] > current_row['ema_long'] and
                            self.rsi_oversold < current_row['rsi'] < self.rsi_overbought and
                            current_row['volume_ratio'] > 1.2):

                        quantity = int(self.capital * 0.15 / current_price)  # 15% position size
                        if quantity > 0:
                            self.current_trade = Trade(
                                symbol=symbol,
                                entry_date=current_date,
                                entry_price=current_price,
                                quantity=quantity
                            )

                            # Set price levels
                            stop_loss_price = current_price * (1 - self.stop_loss_pct)
                            first_target = current_price * (1 + self.first_target_pct)
                            second_target = current_price * (1 + self.second_target_pct)
                            final_target = current_price * (1 + self.final_target_pct)

                            df.loc[i:, 'stop_loss_price'] = stop_loss_price
                            df.loc[i:, 'first_target_price'] = first_target
                            df.loc[i:, 'second_target_price'] = second_target
                            df.loc[i:, 'final_target_price'] = final_target
                            df.loc[i, 'buy_signal'] = True
                            position_size = quantity

                else:
                    # Skip same day entries
                    if current_date.date() == self.current_trade.entry_date.date():
                        continue

                    # Calculate holding time
                    time_held = current_date - self.current_trade.entry_date
                    days_held = time_held.total_seconds() / (24 * 60 * 60)

                    # Check first partial target (1.5%)
                    if (current_price >= df.iloc[i]['first_target_price'] and
                            not any(exit.get('target') == 'first' for exit in self.current_trade.partial_exits)):
                        exit_quantity = int(self.current_trade.quantity * self.partial_exit_pct)
                        self.current_trade.partial_exits.append({
                            'date': current_date,
                            'price': current_price,
                            'quantity': exit_quantity,
                            'target': 'first'
                        })
                        position_size -= exit_quantity
                        df.loc[i, 'profit_target_exit'] = True

                    # Check second partial target (3%)
                    elif (current_price >= df.iloc[i]['second_target_price'] and
                          not any(exit.get('target') == 'second' for exit in self.current_trade.partial_exits)):
                        exit_quantity = int((self.current_trade.quantity - sum(
                            exit['quantity'] for exit in self.current_trade.partial_exits)) * 0.5)
                        self.current_trade.partial_exits.append({
                            'date': current_date,
                            'price': current_price,
                            'quantity': exit_quantity,
                            'target': 'second'
                        })
                        position_size -= exit_quantity
                        df.loc[i, 'profit_target_exit'] = True

                    # Check final target (4.5%)
                    if current_price >= df.iloc[i]['final_target_price']:
                        self.current_trade.exit_date = current_date
                        self.current_trade.exit_price = current_price
                        self.current_trade.exit_reason = 'Final Target'
                        df.loc[i, 'profit_target_exit'] = True
                        df.loc[i, 'sell_signal'] = True
                        self.trades.append(self.current_trade)
                        self.current_trade = None
                        position_size = 0
                        continue

                    # Check stop loss (1.8%)
                    if current_price <= df.iloc[i]['stop_loss_price']:
                        self.current_trade.exit_date = current_date
                        self.current_trade.exit_price = current_price
                        self.current_trade.exit_reason = 'Stop Loss'
                        df.loc[i, 'stop_loss_exit'] = True
                        df.loc[i, 'sell_signal'] = True
                        self.trades.append(self.current_trade)
                        self.current_trade = None
                        position_size = 0
                        continue

                    # Time-based exit (7 days)
                    if days_held >= self.max_holding_days:
                        self.current_trade.exit_date = current_date
                        self.current_trade.exit_price = current_price
                        self.current_trade.exit_reason = 'Time Exit'
                        df.loc[i, 'time_exit'] = True
                        df.loc[i, 'sell_signal'] = True
                        self.trades.append(self.current_trade)
                        self.current_trade = None
                        position_size = 0

            return df

        except Exception as e:
            logger.error(f"Error in backtest: {str(e)}")
            raise

    @staticmethod
    def plot_backtest(df: pd.DataFrame, symbol: str):
        """Plot backtest results"""
        try:
            plt.figure(figsize=(15, 10))

            # Plot price and EMAs
            plt.plot(df['date'], df['close'], label='Price', alpha=0.7)
            plt.plot(df['date'], df['52_week_high'], label='52-week High', alpha=0.5)
            plt.plot(df['date'], df['ema_short'], label='EMA10', alpha=0.5, linestyle='--')
            plt.plot(df['date'], df['ema_long'], label='EMA21', alpha=0.5, linestyle='--')

            # Plot buy signals
            buy_signals = df[df['buy_signal']]
            if not buy_signals.empty:
                plt.scatter(buy_signals['date'], buy_signals['close'],
                            marker='^', color='g', label='Buy Signal', s=100)

            # Plot profit target exits
            profit_exits = df[df['profit_target_exit']]
            if not profit_exits.empty:
                plt.scatter(profit_exits['date'], profit_exits['close'],
                            marker='s', color='blue', label='Profit Exit', s=100)

            # Plot stop loss exits
            stop_loss_exits = df[df['stop_loss_exit']]
            if not stop_loss_exits.empty:
                plt.scatter(stop_loss_exits['date'], stop_loss_exits['close'],
                            marker='v', color='r', label='Stop Loss', s=100)

            # Plot time exits
            time_exits = df[df['time_exit']]
            if not time_exits.empty:
                plt.scatter(time_exits['date'], time_exits['close'],
                            marker='d', color='purple', label='Time Exit', s=100)

            plt.title(f'Backtest Results for {symbol}\nStrategy: Multi-Target EMA')
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
        """Generate backtest report with annualized returns"""
        try:
            if not self.trades:
                return {"error": "No trades found"}

            total_trades = len(self.trades)
            profitable_trades = len([t for t in self.trades if t.pnl() > 0])
            total_pnl = sum(t.pnl() for t in self.trades)

            # Calculate total returns percentage
            total_return_pct = (total_pnl / self.initial_capital) * 100

            # Calculate time period
            start_date = self.trades[0].entry_date
            end_date = self.trades[-1].exit_date
            trading_days = (end_date - start_date).total_seconds() / (24 * 60 * 60)
            years = trading_days / 365.25

            # Calculate annualized return
            if years > 0:
                annualized_return = ((1 + (total_return_pct / 100)) ** (1 / years) - 1) * 100
            else:
                annualized_return = 0

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
                    "total_return_pct": total_return_pct,
                    "trading_period_days": trading_days,
                    "annualized_return_pct": annualized_return,
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
                    "max_profit_trade": max(t.pnl() for t in self.trades) if self.trades else 0,
                    "max_loss_trade": min(t.pnl() for t in self.trades) if self.trades else 0,
                    "avg_profit_per_trade": total_pnl / total_trades if total_trades > 0 else 0,
                    "avg_return_per_trade_pct": total_return_pct / total_trades if total_trades > 0 else 0,
                    "longest_trade": max([t.holding_period_str() for t in self.trades]) if self.trades else "0",
                    "shortest_trade": min([t.holding_period_str() for t in self.trades]) if self.trades else "0"
                }
            }

        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
            raise
