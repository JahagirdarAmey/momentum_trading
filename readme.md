# Momentum Trading

### Project Structure
```
momentum_trading/
│
├── config/
│   └── config.py
│
├── data/
│   ├── __init__.py
│   ├── data_fetcher.py
│   └── database.py
│
├── trading/
│   ├── __init__.py
│   ├── strategy.py
│   └── execution.py
│
├── reporting/
│   ├── __init__.py
│   └── metrics.py
│
├── tests/
│   ├── __init__.py
│   ├── test_data_fetcher.py
│   ├── test_strategy.py
│   └── test_metrics.py
│
├── logs/
├── requirements.txt
└── main.py
```


### Key Features:

- Data fetching from Yahoo Finance API with 15-minute intervals
- PostgreSQL integration with partitioned tables
- Momentum strategy based on 52-week highs
- Position management with partial exits
- Trailing stop-loss implementation
- Performance metrics calculation and storage
- Comprehensive error handling and logging

### Database Schema:

- Partitioned price_data table for efficient historical data storage
- Signals table for tracking entry/exit points
- Backtest_results table for storing performance metrics

###  Trading Logic:

- Entry on 52-week high breakout
- 50% position exit at 4% profit target
- Trailing stop-loss for remaining position
- Continuous monitoring and execution


### Performance Tracking:

- Sharpe ratio calculation
- Maximum drawdown tracking
- Trade statistics storage

### Complete backtest functionality:

- Processes each symbol independently
- Calculates comprehensive performance metrics
- Saves results to database
- Provides detailed logging


### Live trading functionality:

- Market hours checking
- 15-minute interval scheduling
- Error handling and recovery
- Proper shutdown handling

### The system can be extended by:

- Adding more sophisticated market hours checking (holidays, half-days)
- Implementing configuration file loading (YAML/JSON)
- Adding position sizing and risk management
- Implementing real-time performance monitoring
- Adding email/SMS alerts for important events