import logging
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
import pandas as pd

from config.config import DatabaseConfig

logger = logging.getLogger(__name__)

class DatabaseConnection:
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.engine = create_engine(
            f'postgresql://{config.user}:{config.password}@{config.host}:{config.port}/{config.database}'
        )

    def execute_query(self, query: str, params: dict = None) -> pd.DataFrame:
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(query), params or {})
                return pd.DataFrame(result.fetchall(), columns=result.keys())
        except SQLAlchemyError as e:
            logger.error(f"Database query failed: {str(e)}")
            raise

    def insert_dataframe(self, df: pd.DataFrame, table: str, schema: str):
        try:
            df.to_sql(table, self.engine, schema=schema, if_exists='append', index=False)
        except SQLAlchemyError as e:
            logger.error(f"Failed to insert data into {schema}.{table}: {str(e)}")
            raise