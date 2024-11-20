# src/data/stock_data/data_processor.py
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict
from datetime import datetime
import logging
from .cache_manager import StockDataCache
import pytz

logger = logging.getLogger(__name__)


class StockDataProcessor:
    def __init__(self):
        self.cache = StockDataCache()
        self.timezone = pytz.timezone('Asia/Kolkata')  # Indian timezone

    def load_pickle_to_cache(self, stock_name: str) -> bool:
        """Load pickle file data into memory cache"""
        try:
            pickle_path = self._get_pickle_path(stock_name)
            logger.info(f"Loading data for {stock_name} from {pickle_path}")

            df = pd.read_pickle(pickle_path)
            if not isinstance(df, pd.DataFrame):
                df = pd.DataFrame(df)

            # Ensure date column is datetime with consistent timezone
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date']).dt.tz_convert('Asia/Kolkata')
                df.set_index('date', inplace=True)
            else:
                # If date is already the index
                df.index = pd.to_datetime(df.index).tz_convert('Asia/Kolkata')

            self.cache.set(f'stock:{stock_name}', df)
            return True

        except Exception as e:
            logger.error(f"Error loading data for {stock_name}: {str(e)}")
            return False

    def get_stock_data(self, symbol: str,
                       start_date: Optional[datetime] = None,
                       end_date: Optional[datetime] = None) -> Optional[pd.DataFrame]:
        """Retrieve stock data from cache"""
        try:
            df = self.cache.get(f'stock:{symbol}')

            if df is None:
                logger.warning(f"No data found in cache for {symbol}")
                return None

            # Make a copy to avoid modifying cached data
            df = df.copy()

            # Filter by date range if provided
            if start_date:
                # Localize the start_date to match DataFrame timezone
                start_date = pd.to_datetime(start_date).tz_localize('UTC').tz_convert('Asia/Kolkata')
                df = df[df.index >= start_date]
            if end_date:
                # Localize the end_date to match DataFrame timezone
                end_date = pd.to_datetime(end_date).tz_localize('UTC').tz_convert('Asia/Kolkata')
                df = df[df.index <= end_date]

            return df

        except Exception as e:
            logger.error(f"Error retrieving data for {symbol}: {str(e)}")
            raise

    @staticmethod
    def _get_pickle_path(stock_name: str) -> Path:
        current_dir = Path(__file__).resolve().parent
        project_root = current_dir.parent.parent
        return project_root / 'data' / f'{stock_name}-15minute-Hist'

    def get_cache_status(self) -> Dict:
        """Get current cache status"""
        try:
            cache_info = {}
            for key in self.cache.cache:
                df = self.cache.get(key)
                cache_info[key] = {
                    'rows': len(df),
                    'start_date': df.index.min().isoformat(),
                    'end_date': df.index.max().isoformat(),
                    'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 * 1024)
                }

            return {
                'status': 'ok',
                'cache_info': cache_info,
                'total_stocks': len(self.cache.list_stocks())
            }
        except Exception as e:
            logger.error(f"Error getting cache status: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }