"""
Data loading and preprocessing for the trading system.
"""

# Import standard libraries
import logging

# Import typing
from typing import Tuple

# Import third-party libraries
import numpy as np
import pandas as pd
import yfinance as yf

# Import technical indicators
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands


# Initialize logger
logger = logging.getLogger(__name__)


# Download OHLCV data from Yahoo Finance
def fetch_data(ticker, start_date, end_date, interval="1d"):
    # Log request
    logger.info("Fetching %s [%s -> %s] interval=%s", ticker, start_date, end_date, interval)

    # Download data
    dataframe = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        interval=interval,
        auto_adjust=True,
        progress=False,
    )

    # Flatten MultiIndex columns if needed
    if isinstance(dataframe.columns, pd.MultiIndex):
        dataframe.columns = dataframe.columns.get_level_values(0)

    # Validate data
    if dataframe.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'.")

    # Clean data
    dataframe.dropna(inplace=True)

    # Log result
    logger.info("Fetched %d rows for %s.", len(dataframe), ticker)

    return dataframe


# Add technical indicators to dataframe
def add_technical_indicators(dataframe):
    # Copy dataframe
    dataframe = dataframe.copy()

    # Compute price-based features
    dataframe["close_pct"] = dataframe["Close"].pct_change()
    dataframe["high_low_pct"] = (dataframe["High"] - dataframe["Low"]) / dataframe["Close"]

    # Compute RSI
    rsi_indicator = RSIIndicator(close=dataframe["Close"], window=14)
    dataframe["rsi"] = rsi_indicator.rsi()

    # Compute MACD
    macd_indicator = MACD(close=dataframe["Close"])
    dataframe["macd"] = macd_indicator.macd()
    dataframe["macd_signal"] = macd_indicator.macd_signal()
    dataframe["macd_diff"] = macd_indicator.macd_diff()

    # Compute Bollinger Bands
    bollinger = BollingerBands(close=dataframe["Close"], window=20, window_dev=2)
    dataframe["bb_pct"] = bollinger.bollinger_pband()
    dataframe["bb_width"] = bollinger.bollinger_wband()

    # Compute normalized volume
    rolling_volume_mean = dataframe["Volume"].rolling(20).mean()
    dataframe["volume_norm"] = dataframe["Volume"] / rolling_volume_mean.replace(0, np.nan)

    # Remove invalid rows
    dataframe.dropna(inplace=True)

    return dataframe


# Split dataframe into train and test sets
def split_data(dataframe, train_split=0.8):
    # Compute split index
    split_index = int(len(dataframe) * train_split)

    # Create train and test sets
    train_dataframe = dataframe.iloc[:split_index].copy().reset_index(drop=False)
    test_dataframe = dataframe.iloc[split_index:].copy().reset_index(drop=False)

    # Log result
    logger.info("Split -> train=%d rows, test=%d rows.", len(train_dataframe), len(test_dataframe))

    return train_dataframe, test_dataframe


# Load and prepare dataset
def load_data(config):
    # Extract configuration
    ticker = config["ticker"]
    start_date = config["start_date"]
    end_date = config["end_date"]
    interval = config.get("interval", "1d")
    train_split = config.get("train_split", 0.8)

    # Fetch data
    dataframe = fetch_data(
        ticker,
        start_date,
        end_date,
        interval,
    )

    # Add indicators
    dataframe = add_technical_indicators(dataframe)

    # Split dataset
    train_dataframe, test_dataframe = split_data(
        dataframe,
        train_split,
    )

    return train_dataframe, test_dataframe