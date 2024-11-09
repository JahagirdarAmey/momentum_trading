import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import List

from sqlalchemy import text

from config.config import Config
from data.database import DatabaseConnection

logger = logging.getLogger(__name__)

class DataFetcher:
    def __init__(self, db: DatabaseConnection, config: Config):
        self.db = db
        self.config = config

    def fetch_latest_data(self, symbol: str) -> pd.DataFrame:
        """Fetch latest 15-minute data for a symbol"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(interval=self.config.data_interval, period="1d")
            logger.info(f"Fetched {len(data)} records for {symbol}")
            return data
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            raise

    def save_to_database(self, symbol: str, data: pd.DataFrame):
        """Save new data to PostgreSQL"""
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
                    new_data.to_sql(
                        'price_data',
                        con=self.db.engine,
                        schema='trading',
                        if_exists='append',
                        index=True
                    )
                    logger.info(f"Saved {len(new_data)} new records for {symbol}")
        except Exception as e:
            logger.error(f"Error saving data for {symbol}: {str(e)}")
            raise