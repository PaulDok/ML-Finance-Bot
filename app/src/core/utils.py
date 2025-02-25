import gc
import logging
import sqlite3
from datetime import datetime, timedelta
from typing import Union

import pandas as pd
import yfinance as yf
from sqlalchemy import Engine, create_engine
from src.core import config

logger = logging.getLogger()

# # # # # # # # # # # # # # # #
# ~ @ # $ = WRAPPERS  = $ # @ ~ #
# # # # # # # # # # # # # # # #


def gc_collect(func):
    """
    Wrapper to call gc.collect after critical functions
    """

    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        _ = gc.collect()
        return result

    return wrapper


# # # # # # # # # # # # # # # #
# ~ @ # $ = LOGGING = $ # @ ~ #
# # # # # # # # # # # # # # # #
IS_LOGGER_INITIALIZED = False


def tzoffsetconverter(s):
    """
    Обработчик timestamp'ов для логгирования.
    Выдаёт московское время вместо UTC для упрощения отладки
    """
    return datetime.timetuple(datetime.fromtimestamp(s) + timedelta(hours=3))


def init_logger(conf: config.Config) -> None:
    """
    Для инициализации логгера
    """
    # root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Форматирование сообщений для логгера
    log_formatter = logging.Formatter(
        "[%(levelname)-7.7s] %(asctime)s: %(message)s", "%Y-%m-%d@%H:%M:%S"
    )
    log_formatter.converter = tzoffsetconverter

    # Обработчик записи в файл
    fileHandler = logging.FileHandler(conf.LOG_FILE)
    fileHandler.setFormatter(log_formatter)
    logger.addHandler(fileHandler)

    # Обработчик записи в STDout
    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(log_formatter)
    logger.addHandler(consoleHandler)

    global IS_LOGGER_INITIALIZED
    IS_LOGGER_INITIALIZED = True


# # # # # # # # # # # # # # # # # # # #
# ~ @ # $ = DATABASE UTILS  = $ # @ ~ #
# # # # # # # # # # # # # # # # # # # #

PATH_TO_DB = "data/cache.db"


def get_sqlite_connection() -> sqlite3.Connection:
    """
    Service function to make sure everything is connected to the same DB object
    """
    return sqlite3.connect(f"./{PATH_TO_DB}")


def get_sqlite_engine() -> Engine:
    return create_engine(f"sqlite:///{PATH_TO_DB}", echo=False)


def create_table_for_interval(interval: str) -> None:
    """
    Create a table to store tickers history, name dependent on INTERVAL
    """
    # NOTE: table name cannot start with digit -> reverse in such cases
    table_name = interval if interval[0].isalpha() else interval[::-1]
    table_create_query = f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
    Date TEXT,
    Ticker TEXT NOT NULL,
    Open REAL,
    Low REAL,
    High REAL,
    Close REAL,
    Volume REAL
    )
    """
    # Connect and run query
    connection = get_sqlite_connection()
    cursor = connection.cursor()
    cursor.execute(table_create_query)
    # commit and close
    connection.commit()
    connection.close()


def delete_ticker_data_from_sqlite(
    tickers: Union[str, list[str]], interval: str
) -> None:
    """
    DELETE Ticker(s) data from SQLite
    """
    # Form an SQL statement
    table_name = interval if interval[0].isalpha() else interval[::-1]
    ticker_filter = (
        "('" + "', '".join(tickers) + "')"
        if (type(tickers) is list)
        else f"('{tickers}')"
    )
    delete_query = f"""
    DELETE FROM {table_name}
    WHERE Ticker IN {ticker_filter}
    """

    # Run on DB
    with get_sqlite_connection() as conn:
        conn.execute(delete_query)
        conn.commit()


def upload_data_to_sqlite(data: pd.DataFrame, interval: str) -> None:
    """
    Append data to SQLite database
    """
    engine = get_sqlite_engine()
    with engine.connect() as conn:
        table_name = interval if interval[0].isalpha() else interval[::-1]
        data.to_sql(table_name, conn, if_exists="append", index=False)
        conn.commit()
    engine.dispose()


# # # # # # # # # # # # # # # # # # # #
# ~ @ # $ = HISTORY GETTING  = $ # @ ~ #
# # # # # # # # # # # # # # # # # # # #


def get_tickers_history(
    tickers: Union[str, list[str]] = "BTC-USD",
    start: str = "2024-03-30",
    end: str = "2024-06-02",
    interval: str = "1h",
) -> pd.DataFrame:
    """
    Get tickers history in requested timeframe with requested interval using yfinance and process it to Pandas DataFrame
    """
    logger.info("Downloading data using yfinance")
    data = yf.download(tickers, start=start, end=end, interval=interval)
    logger.info(f"Downloaded data shape: {data.shape}")

    # Reshape: put tickers to column and select only necessary columns
    data = data.stack("Ticker").reset_index()[
        ["Date", "Ticker", "Open", "Low", "High", "Close", "Volume"]
    ]
    logger.info(f"reshaped: {data.shape}")

    # Make sure 'Date' is in DateTime format
    data["Date"] = pd.to_datetime(data["Date"])

    return data
