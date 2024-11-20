# src/data/stock_data/__init__.py
from .data_processor import StockDataProcessor

# This makes the class available when someone imports from the package
__all__ = ['StockDataProcessor']