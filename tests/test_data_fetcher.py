import pytest
from unittest.mock import Mock, patch
import pandas as pd

from config.config import Config
from data.data_fetcher import DataFetcher


def test_fetch_latest_data():
    mock_db = Mock()
    config = Config()
    fetcher = DataFetcher(mock_db, config)

    with patch('yfinance.Ticker') as mock_ticker:
        mock_data = pd.DataFrame({
            'Open': [100],
            'High': [101],
            'Low': [99],
            'Close': [100.5],
            'Volume': [1000]
        })
        mock_ticker.return_value.history.return_value = mock_data

        result = fetcher.fetch_latest_data('AAPL')
        assert len(result) == 1
        assert 'Open' in result.columns


def test_save_to_database():
    mock_db = Mock()
    config = Config()
    fetcher = DataFetcher(mock_db, config)

    test_data = pd.DataFrame({
        'Open': [100],
        'High': [101],
        'Low': [99],
        'Close': [100.5],
        'Volume': [1000]
    })

    with patch('sqlalchemy.create_engine'):
        fetcher.save_to_database('AAPL', test_data)
        mock_db.engine.connect.assert_called_once()