class TradingSystemError(Exception):
    """Base exception class for trading system errors"""
    pass

class DataError(TradingSystemError):
    """Exception raised for errors during data fetching or processing"""
    pass

class BacktestError(TradingSystemError):
    """Exception raised for errors during backtesting"""
    pass

class ExecutionError(TradingSystemError):
    """Exception raised for errors during trade execution"""
    pass

class ConfigError(TradingSystemError):
    """Exception raised for configuration-related errors"""
    pass