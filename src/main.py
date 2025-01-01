# src/main.py
import concurrent.futures
import logging
from contextlib import asynccontextmanager
from typing import Optional

import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from matplotlib import pyplot as plt
from starlette.middleware.cors import CORSMiddleware

from data.data_processor import StockDataProcessor
from data.stock_list import STOCKS
from src.strategies.breakout_strategy import BreakoutStrategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize data processor
processor = StockDataProcessor()


def load_stock_data(stock: str) -> tuple[str, bool]:
    """Load single stock data and return result"""
    try:
        success = processor.load_pickle_to_cache(stock)
        return stock, success
    except Exception as e:
        logger.error(f"Error loading {stock}: {str(e)}")
        return stock, False


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load all stock data into cache
    logger.info("Starting data loading process...")

    # Use ThreadPoolExecutor for parallel loading
    success_count = 0
    failed_stocks = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        # Submit all stock loading tasks
        future_to_stock = {executor.submit(load_stock_data, stock): stock for stock in STOCKS}

        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_stock):
            stock = future_to_stock[future]
            try:
                _, success = future.result()
                if success:
                    success_count += 1
                else:
                    failed_stocks.append(stock)
            except Exception as e:
                logger.error(f"Error loading {stock}: {str(e)}")
                failed_stocks.append(stock)

    logger.info(f"Successfully loaded {success_count} stocks")
    if failed_stocks:
        logger.warning(f"Failed to load {len(failed_stocks)} stocks: {failed_stocks}")

    yield

    # Cleanup (if needed)
    logger.info("Shutting down...")


app = FastAPI(
    title="Stock Data Service",
    description="Service providing 15-minute historical stock data",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/stocks")
async def list_available_stocks():
    """Get list of all available stocks"""
    try:
        cached_stocks = processor.cache.list_stocks()
        total_stocks = len(cached_stocks)
        return {
            "status": "success",
            "total_stocks": total_stocks,
            "stocks": cached_stocks
        }
    except Exception as e:
        logger.error(f"Error listing stocks: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stocks/{symbol}")
async def get_stock_data(
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
):
    """Get stock data for a specific symbol"""
    try:
        # Convert string dates to pandas Timestamp
        start = pd.to_datetime(start_date) if start_date else None
        end = pd.to_datetime(end_date) if end_date else None

        data = processor.get_stock_data(symbol, start, end)

        if data is None:
            raise HTTPException(status_code=404, detail=f"No data found for {symbol}")

        # Convert DataFrame to records and handle datetime
        records = data.reset_index().to_dict(orient='records')

        # Add metadata to response
        return {
            "status": "success",
            "symbol": symbol,
            "start_date": start_date,
            "end_date": end_date,
            "total_records": len(records),
            "data_range": {
                "first_date": data.index.min().isoformat(),
                "last_date": data.index.max().isoformat()
            },
            "data": records
        }

    except Exception as e:
        logger.error(f"Error retrieving data for {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/cache/status")
async def get_cache_status():
    """Get current cache status"""
    try:
        status = processor.get_cache_status()
        return {
            "status": "success",
            **status
        }
    except Exception as e:
        logger.error(f"Error getting cache status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/backtest/{symbol}")
async def run_backtest(
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        initial_capital: float = 100000
):
    """Run backtest for a symbol"""
    try:
        # Get data
        data = processor.get_stock_data(symbol, start_date, end_date)

        if data is None:
            raise HTTPException(status_code=404, detail=f"No data found for {symbol}")

        # Initialize strategy
        strategy = BreakoutStrategy(initial_capital=initial_capital)

        # Run backtest
        backtest_results = strategy.backtest(data.copy(), symbol)

        # Generate strategy plot
        strategy_plot = strategy.plot_backtest(backtest_results, symbol)
        strategy_plot_path = f"backtest_results_{symbol}.png"
        strategy_plot.savefig(strategy_plot_path)
        plt.close()

        # Generate balance chart
        balance_plot = strategy.plot_balance_chart()
        balance_plot_path = f"balance_chart_{symbol}.png"
        balance_plot.savefig(balance_plot_path)
        plt.close()

        # Generate report
        report = strategy.generate_report()

        return {
            "status": "success",
            "symbol": symbol,
            "report": report,
            "strategy_plot": strategy_plot_path,
            "balance_plot": balance_plot_path
        }

    except Exception as e:
        logger.error(f"Error running backtest for {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/portfolio/backtest")
async def run_portfolio_backtest(
        request: PortfolioBacktestRequest
):
    """Run portfolio-wide backtest"""
    try:
        backtest = PortfolioBacktest(
            symbols=request.symbols,
            start_date=request.start_date,
            end_date=request.end_date,
            initial_capital=request.initial_capital
        )

        results = backtest.run_simulation()

        # Generate plots
        portfolio_plots = generate_portfolio_plots(results)

        return {
            "status": "success",
            "results": results,
            "plots": portfolio_plots
        }

    except Exception as e:
        logger.error(f"Portfolio backtest error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "cached_stocks": len(processor.cache.list_stocks())
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)