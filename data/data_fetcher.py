import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Union

import pandas as pd
import yfinance as yf
from sqlalchemy import text, exc

from config.config import Config
from data.database import DatabaseConnection
from exceptions.trading_exceptions import DataError

logger = logging.getLogger(__name__)


class DataFetcher:
    """
    Handles market data fetching and storage operations
    """

    def __init__(self, db: DatabaseConnection, config: Config):
        self.db = db
        self.config = config
        self._data_cache: Dict[str, pd.DataFrame] = {}
        self._last_update: Dict[str, datetime] = {}

    def fetch_latest_data(self, symbol: str) -> pd.DataFrame:
        """
        Fetch latest market data for a symbol

        Args:
            symbol: Trading symbol to fetch data for

        Returns:
            DataFrame with OHLCV data

        Raises:
            DataError: If data fetching fails
        """
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(
                interval=self.config.data_interval,
                period="1d"
            )

            if data.empty:
                raise DataError(f"No data received for {symbol}")

            # Validate data quality
            self._validate_market_data(data, symbol)

            # Update cache
            self._data_cache[symbol] = data
            self._last_update[symbol] = datetime.now()

            logger.info(f"Fetched {len(data)} records for {symbol}")
            return data

        except Exception as e:
            raise DataError(f"Error fetching data for {symbol}: {str(e)}") from e

    def get_latest_data(self, symbol: str) -> Optional[Dict]:
        """
        Get most recent price data for a symbol

        Args:
            symbol: Trading symbol

        Returns:
            Dict with latest OHLCV data or None if no data available
        """
        try:
            # Check if cache needs refresh
            if self._needs_refresh(symbol):
                self.fetch_latest_data(symbol)

            if symbol not in self._data_cache:
                return None

            data = self._data_cache[symbol]
            if data.empty:
                return None

            latest = data.iloc[-1]
            return {
                'timestamp': latest.name,
                'open': float(latest['Open']),
                'high': float(latest['High']),
                'low': float(latest['Low']),
                'close': float(latest['Close']),
                'volume': float(latest['Volume'])
            }

        except Exception as e:
            logger.error(f"Error getting latest data for {symbol}: {str(e)}")
            return None

    def save_to_database(self, symbol: str, data: pd.DataFrame):
        """
        Save new market data to database

        Args:
            symbol: Trading symbol
            data: DataFrame with OHLCV data to save

        Raises:
            DataError: If database operation fails
        """
        try:
            with self.db.engine.connect() as conn:
                # Get latest timestamp in database
                query = text("""
                    SELECT MAX(timestamp) 
                    FROM trading.price_data 
                    WHERE symbol = :symbol
                """)
                last_timestamp = conn.execute(query, {"symbol": symbol}).scalar()

                # Filter only new data
                if last_timestamp:
                    new_data = data[data.index > last_timestamp]
                else:
                    new_data = data

                if not new_data.empty:
                    # Prepare data for insertion
                    df_to_save = new_data.copy()
                    df_to_save['symbol'] = symbol

                    # Rename columns to match database schema
                    df_to_save.rename(columns={
                        'Open': 'open',
                        'High': 'high',
                        'Low': 'low',
                        'Close': 'close',
                        'Volume': 'volume'
                    }, inplace=True)

                    # Save to database
                    df_to_save.to_sql(
                        'price_data',
                        con=self.db.engine,
                        schema='trading',
                        if_exists='append',
                        index=True,
                        index_label='timestamp'
                    )

                    logger.info(f"Saved {len(new_data)} new records for {symbol}")
                else:
                    logger.info(f"No new data to save for {symbol}")

        except exc.SQLAlchemyError as e:
            raise DataError(f"Database error saving data for {symbol}: {str(e)}") from e
        except Exception as e:
            raise DataError(f"Error saving data for {symbol}: {str(e)}") from e

    def update_market_data(self, symbols: Union[str, List[str]], parallel: bool = True):
        """
        Update market data for multiple symbols

        Args:
            symbols: Single symbol or list of symbols to update
            parallel: Whether to fetch data in parallel
        """
        if isinstance(symbols, str):
            symbols = [symbols]

        if parallel and len(symbols) > 1:
            self._parallel_update(symbols)
        else:
            self._sequential_update(symbols)

    def _parallel_update(self, symbols: List[str]):
        """Update market data for multiple symbols in parallel"""
        with ThreadPoolExecutor() as executor:
            future_to_symbol = {
                executor.submit(self._update_single_symbol, symbol): symbol
                for symbol in symbols
            }

            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Failed to update {symbol}: {str(e)}")

    def _sequential_update(self, symbols: List[str]):
        """Update market data for symbols sequentially"""
        for symbol in symbols:
            try:
                self._update_single_symbol(symbol)
            except Exception as e:
                logger.error(f"Failed to update {symbol}: {str(e)}")

    def _update_single_symbol(self, symbol: str):
        """Update market data for a single symbol"""
        data = self.fetch_latest_data(symbol)
        self.save_to_database(symbol, data)

    @staticmethod
    def _validate_market_data(data: pd.DataFrame, symbol: str):
        """
        Validate market data quality

        Args:
            data: DataFrame to validate
            symbol: Symbol for logging purposes

        Raises:
            DataError: If data fails validation
        """
        if data.empty:
            raise DataError(f"Empty dataset received for {symbol}")

        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise DataError(f"Missing required columns for {symbol}: {missing_columns}")

        # Check for missing values
        if data[required_columns].isna().any().any():
            logger.warning(f"Missing values detected in {symbol} data")
            # Fill missing values using forward fill
            data.fillna(method='ffill', inplace=True)

        # Validate price relationships
        invalid_prices = (
                (data['High'] < data['Low']) |
                (data['Close'] > data['High']) |
                (data['Close'] < data['Low'])
        )
        if invalid_prices.any():
            bad_rows = data[invalid_prices].index
            logger.warning(f"Invalid price relationships found in {symbol} at: {bad_rows}")

    def _needs_refresh(self, symbol: str) -> bool:
        """Check if cached data needs to be refreshed"""
        if symbol not in self._last_update:
            return True

        elapsed = datetime.now() - self._last_update[symbol]
        refresh_interval = timedelta(minutes=int(self.config.data_interval[:-1]))
        return elapsed > refresh_interval

    def clear_cache(self, symbol: Optional[str] = None):
        """
        Clear data cache for one or all symbols

        Args:
            symbol: Specific symbol to clear, or None to clear all
        """
        if symbol:
            self._data_cache.pop(symbol, None)
            self._last_update.pop(symbol, None)
        else:
            self._data_cache.clear()
            self._last_update.clear()