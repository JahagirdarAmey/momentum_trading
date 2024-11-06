from unittest.mock import Mock, patch

from config.config import Config
from trading.strategy import MomentumStrategy


def test_check_entry_signal():
    mock_db = Mock()
    config = Config()
    strategy = MomentumStrategy(mock_db, config)

    with patch.object(strategy, 'calculate_52_week_high', return_value=100.0):
        assert strategy.check_entry_signal('AAPL', 101.0) == True
        assert strategy.check_entry_signal('AAPL', 99.0) == False


def test_check_exit_signal():
    mock_db = Mock()
    config = Config()
    strategy = MomentumStrategy(mock_db, config)

    # Test profit target exit
    exit_signal, exit_size = strategy.check_exit_signal('AAPL', 100.0, 104.0, 1.0)
    assert exit_signal == True
    assert exit_size == 0.5

    # Test trailing stop exit
    exit_signal, exit_size = strategy.check_exit_signal('AAPL', 100.0, 97.0, 0.5)
    assert exit_signal == True
    assert exit_size == 0.5
