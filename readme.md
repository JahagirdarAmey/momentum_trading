# Stock Market Analysis Service

A FastAPI-based service for analyzing historical stock data with a focus on high breakout trading strategy implementation.

## Overview

This service provides access to 15-minute historical stock data and implements a high breakout trading strategy with backtest capabilities. The system features real-time data processing, caching mechanisms, and comprehensive trade analysis tools.

## Technical Stack

- **Backend Framework**: FastAPI
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib
- **Concurrent Processing**: Python's concurrent.futures
- **Logging**: Python's built-in logging module

## Features

### Data Management
- 15-minute historical stock data processing
- Efficient data caching system
- Parallel data loading using ThreadPoolExecutor
- Date range filtering support

### Trading Strategy
- 52-week high breakout strategy implementation
- Real-time trade execution simulation
- Position sizing based on capital
- Risk management features:
  - 2% trailing stop loss
  - 5% target for partial position exit
  - Half position exit at target price

### Analysis Tools
- Comprehensive backtest capabilities
- Performance metrics calculation
- Trade statistics reporting
- Visual analysis through matplotlib plots

## API Endpoints

### Stock Data Endpoints
- `GET /stocks`: List all available stocks
- `GET /stocks/{symbol}`: Get historical data for a specific stock
- `GET /cache/status`: Check cache status
- `GET /health`: Service health check

### Backtest Endpoint
- `GET /backtest/{symbol}`: Run backtest analysis for a specific stock

## Key Components

### HighBreakoutStrategy Class
The core trading strategy implementation includes:
- Trade entry at 52-week high breakouts
- Automated position sizing
- Trailing stop loss management
- Partial exit strategy
- Comprehensive trade tracking

### Trade Class
Handles individual trade management:
- Entry/exit price tracking
- Position sizing
- Holding period calculations
- Partial exit tracking
- P&L calculations

### StockDataProcessor
Manages data operations:
- Data loading and caching
- Date range filtering
- Data validation
- Cache status monitoring

## Installation & Setup

1. Clone the repository
2. Install dependencies:
```bash
pip install fastapi uvicorn pandas numpy matplotlib
```
3. Start the server:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

## Configuration

### Trading Parameters
- Initial Capital: 100,000 (default, configurable)
- Trailing Stop Loss: 2%
- Partial Exit Target: 5%
- Time Frame: 15-minute candles

### System Configuration
- Logging Level: INFO
- Thread Pool: 4 workers for data loading
- CORS: Enabled for all origins

## Usage Examples

### Running a Backtest
```
# API Request
GET /backtest/AAPL?start_date=2023-01-01&end_date=2023-12-31&initial_capital=100000
```

### Fetching Historical Data
```
# API Request
GET /stocks/AAPL?start_date=2023-01-01&end_date=2023-12-31
```

## Performance Reporting

The system generates detailed performance reports including:
- Total return and P&L
- Win rate and trade statistics
- Average holding periods
- Maximum profit/loss trades
- Position sizing details
- Trade duration analytics