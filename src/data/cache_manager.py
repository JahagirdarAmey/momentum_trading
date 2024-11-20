# src/data/stock_data/cache_manager.py
import pandas as pd
from typing import Dict, Optional
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StockDataCache:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(StockDataCache, cls).__new__(cls)
            cls._instance.cache: Dict[str, pd.DataFrame] = {}
            cls._instance.last_updated: Dict[str, datetime] = {}
        return cls._instance

    def set(self, key: str, value: pd.DataFrame):
        if not isinstance(value, pd.DataFrame):
            value = pd.DataFrame(value)
        self.cache[key] = value
        self.last_updated[key] = datetime.now()
        logger.info(f"Cached data for {key}")

    def get(self, key: str) -> Optional[pd.DataFrame]:
        return self.cache.get(key)

    def list_stocks(self) -> list:
        return [k.split(':')[1] for k in self.cache.keys()]

    def get_cache_info(self):
        info = {}
        for key in self.cache:
            try:
                df = self.cache[key]
                info[key] = {
                    'rows': len(df),
                    'columns': list(df.columns),
                    'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 * 1024),
                    'last_updated': self.last_updated[key].isoformat(),
                    'start_date': df.index.min().isoformat() if isinstance(df.index, pd.DatetimeIndex) else None,
                    'end_date': df.index.max().isoformat() if isinstance(df.index, pd.DatetimeIndex) else None
                }
            except Exception as e:
                logger.error(f"Error getting info for {key}: {str(e)}")
                info[key] = {
                    'error': str(e),
                    'last_updated': self.last_updated[key].isoformat()
                }
        return info