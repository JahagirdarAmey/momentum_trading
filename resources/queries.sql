-- Create database
CREATE DATABASE momentum_trading;

-- Connect to database
\c momentum_trading;

-- Create user
CREATE USER trader WITH PASSWORD 'your_password';

-- Create schemas
CREATE SCHEMA trading;
CREATE SCHEMA backtest;

-- Grant privileges
GRANT ALL PRIVILEGES ON DATABASE momentum_trading TO trader;
GRANT ALL PRIVILEGES ON SCHEMA trading TO trader;
GRANT ALL PRIVILEGES ON SCHEMA backtest TO trader;

-- Create price data table with partitioning
CREATE TABLE trading.price_data (
    symbol VARCHAR(10),
    timestamp TIMESTAMP,
    open DECIMAL(10,2),
    high DECIMAL(10,2),
    low DECIMAL(10,2),
    close DECIMAL(10,2),
    volume BIGINT,
    PRIMARY KEY (symbol, timestamp)
) PARTITION BY RANGE (timestamp);

-- Create yearly partitions for price data
CREATE TABLE trading.price_data_2024 PARTITION OF trading.price_data
    FOR VALUES FROM ('2024-01-01') TO ('2025-01-01');
CREATE TABLE trading.price_data_2023 PARTITION OF trading.price_data
    FOR VALUES FROM ('2023-01-01') TO ('2024-01-01');
CREATE TABLE trading.price_data_2023 PARTITION OF trading.price_data
    FOR VALUES FROM ('2022-01-01') TO ('2023-01-01');
CREATE TABLE trading.price_data_2023 PARTITION OF trading.price_data
    FOR VALUES FROM ('2021-01-01') TO ('2022-01-01');
CREATE TABLE trading.price_data_2023 PARTITION OF trading.price_data
    FOR VALUES FROM ('2020-01-01') TO ('2021-01-01');

-- Create signals table
CREATE TABLE trading.signals (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10),
    timestamp TIMESTAMP,
    signal_type VARCHAR(10),
    price DECIMAL(10,2),
    quantity INTEGER,
    reason VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create positions table
CREATE TABLE trading.positions (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10),
    entry_price DECIMAL(10,2),
    entry_date TIMESTAMP,
    quantity INTEGER,
    current_stop_loss DECIMAL(10,2),
    status VARCHAR(10),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create backtest results table
CREATE TABLE backtest.results (
    id SERIAL PRIMARY KEY,
    strategy_name VARCHAR(50),
    start_date TIMESTAMP,
    end_date TIMESTAMP,
    initial_capital DECIMAL(10,2),
    final_capital DECIMAL(10,2),
    total_return DECIMAL(10,4),
    sharpe_ratio DECIMAL(10,4),
    max_drawdown DECIMAL(10,4),
    total_trades INTEGER,
    winning_trades INTEGER,
    losing_trades INTEGER,
    win_rate DECIMAL(10,4),
    avg_win DECIMAL(10,2),
    avg_loss DECIMAL(10,2),
    profit_factor DECIMAL(10,4),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create backtest trades table
CREATE TABLE backtest.trades (
    id SERIAL PRIMARY KEY,
    backtest_id INTEGER REFERENCES backtest.results(id),
    symbol VARCHAR(10),
    entry_date TIMESTAMP,
    entry_price DECIMAL(10,2),
    exit_date TIMESTAMP,
    exit_price DECIMAL(10,2),
    quantity INTEGER,
    pnl DECIMAL(10,2),
    return_pct DECIMAL(10,4),
    exit_reason VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_backtest
        FOREIGN KEY(backtest_id)
        REFERENCES backtest.results(id)
        ON DELETE CASCADE
);

-- Create indexes
CREATE INDEX idx_price_data_symbol_timestamp ON trading.price_data(symbol, timestamp);
CREATE INDEX idx_signals_symbol ON trading.signals(symbol);
CREATE INDEX idx_positions_symbol ON trading.positions(symbol);
CREATE INDEX idx_backtest_trades_backtest_id ON backtest.trades(backtest_id);

-- Grant table privileges
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA trading TO trader;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA backtest TO trader;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA trading TO trader;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA backtest TO trader;