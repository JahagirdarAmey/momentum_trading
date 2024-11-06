import logging
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

from config.config import DatabaseConfig

logger = logging.getLogger(__name__)


class Database:
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.engine = create_engine(
            f'postgresql://{config.user}:{config.password}@{config.host}:{config.port}/{config.database}'
        )

    def initialize_database(self):
        """Create database schema and tables"""
        queries = [
            """
            CREATE SCHEMA IF NOT EXISTS trading;
            """,
            """
            CREATE TABLE IF NOT EXISTS trading.price_data (
                symbol VARCHAR(10),
                timestamp TIMESTAMP,
                open DECIMAL,
                high DECIMAL,
                low DECIMAL,
                close DECIMAL,
                volume BIGINT,
                PRIMARY KEY (symbol, timestamp)
            ) PARTITION BY RANGE (timestamp);
            """,
            """
            CREATE TABLE IF NOT EXISTS trading.signals (
                id SERIAL PRIMARY KEY,
                symbol VARCHAR(10),
                timestamp TIMESTAMP,
                signal_type VARCHAR(10),
                price DECIMAL,
                quantity INTEGER,
                reason VARCHAR(100)
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS trading.backtest_results (
                id SERIAL PRIMARY KEY,
                start_date TIMESTAMP,
                end_date TIMESTAMP,
                initial_capital DECIMAL,
                final_capital DECIMAL,
                sharpe_ratio DECIMAL,
                max_drawdown DECIMAL,
                total_trades INTEGER,
                winning_trades INTEGER,
                losing_trades INTEGER
            );
            """
        ]

        try:
            for query in queries:
                with self.engine.connect() as conn:
                    conn.execute(text(query))
                    conn.commit()
            logger.info("Database initialized successfully")
        except SQLAlchemyError as e:
            logger.error(f"Database initialization failed: {str(e)}")
            raise