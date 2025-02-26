import gc
import logging
import sqlite3
from datetime import datetime, timedelta
from typing import Optional, Union

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
    tickers: Union[str, list[str]],
    interval: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
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
    if start_date is not None:
        delete_query += f" AND Date >= '{start_date}'"
    if end_date is not None:
        delete_query += f" AND Date <= '{end_date}'"

    # Run on DB
    with get_sqlite_connection() as conn:
        # logger.info(delete_query)
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


def get_unavailable_parts(
    tickers: Union[str, list[str]],
    start_dt: str,
    end_dt: str,
    interval: str,
) -> tuple[list, list, str, str, list, str, str]:
    """
    Check which part of requested data already exists in DB and return what really should be updated
    """
    if type(tickers) is str:
        tickers = [tickers]
    logger.info("Checking already available data...")
    # Form an SQL statement
    table_name = interval if interval[0].isalpha() else interval[::-1]
    query = f"""
    SELECT
        Ticker,
        MIN(Date) AS MINDate,
        MAX(Date) AS MAXDate
    FROM {table_name}
    GROUP BY Ticker
    """

    # Run query on DB
    conn = get_sqlite_connection()
    tickers_date_borders = pd.read_sql(query, con=conn)
    conn.close()

    # Filter tickers
    tickers_date_borders = tickers_date_borders[
        tickers_date_borders["Ticker"].isin(tickers)
    ].reset_index(drop=True)
    # Format datetime columns
    tickers_date_borders["MINDate"] = pd.to_datetime(tickers_date_borders["MINDate"])
    tickers_date_borders["MAXDate"] = pd.to_datetime(tickers_date_borders["MAXDate"])

    # Compare which tickers don't have history before, after and at all
    # a) If the ticker is not present at all - will try to download all requested time frame
    unavailable_tickers = [
        ticker
        for ticker in tickers
        if ticker not in tickers_date_borders["Ticker"].unique()
    ]
    logger.info(f"{len(unavailable_tickers):,d} tickers have no data at all")

    # b) If MINDate is later than start_dt - have to get data before
    lack_earlier_history = tickers_date_borders[
        tickers_date_borders["MINDate"] > start_dt
    ].reset_index(drop=True)
    lack_early_tickers = lack_earlier_history["Ticker"].unique()
    lack_early_start = start_dt
    lack_early_end = lack_earlier_history["MINDate"].max()
    logger.info(f"{len(lack_early_tickers):,d} tickers lack history in start part")

    # c) If MAXDate is earlier than end_dt - have to get data later
    # TODO: it actually depends on interval, so everything is loaded now, need to rework
    lack_later_history = tickers_date_borders[
        tickers_date_borders["MAXDate"] < end_dt
    ].reset_index(drop=True)
    lack_later_tickers = lack_later_history["Ticker"].unique()
    lack_later_start = lack_later_history["MAXDate"].min()
    lack_later_end = end_dt
    logger.info(f"{len(lack_later_tickers):,d} tickers lack history in end part")

    return (
        list(unavailable_tickers),
        list(lack_early_tickers),
        lack_early_start,
        lack_early_end,
        list(lack_later_tickers),
        lack_later_start,
        lack_later_end,
    )


def update_tickers_data(
    tickers: Union[str, list[str]], start_dt: str, end_dt: str, interval: str
) -> None:
    """
    Update historical data in cached database, optimizing network load
    """
    # 1. Create database table, if it doesn't exist yet
    create_table_for_interval(interval)
    logger.info(f"Made sure table in database for {interval=} exists")

    # 2. Get current data and correct which data parts should in fact be updated
    (
        unavailable_tickers,
        lack_early_tickers,
        lack_early_start,
        lack_early_end,
        lack_later_tickers,
        lack_later_start,
        lack_later_end,
    ) = get_unavailable_parts(tickers, start_dt, end_dt, interval)

    # 3. Update data on unavailable tickers
    if len(unavailable_tickers) > 0:
        logger.info("Updating unavailable tickers data...")
        # 3.1. Get data from web
        unavailable_tickers_history = get_tickers_history(
            tickers=unavailable_tickers, start=start_dt, end=end_dt, interval=interval
        )
        # 3.2. Delete old data from SQLite to avoid duplicates
        delete_ticker_data_from_sqlite(unavailable_tickers, interval)
        # 3.3. Upload data to SQLite
        upload_data_to_sqlite(unavailable_tickers_history, interval)

    # 4. Update data on tickers lacking earlier history
    if len(lack_early_tickers) > 0:
        logger.info("Updating tickers lacking early history data...")
        # 4.1. Get data from web
        lack_early_tickers_history = get_tickers_history(
            tickers=lack_early_tickers,
            start=lack_early_start,
            end=lack_early_end,
            interval=interval,
        )
        # 4.2. Delete old data from SQLite to avoid duplicates
        delete_ticker_data_from_sqlite(
            lack_early_tickers,
            interval,
            start_date=lack_early_start,
            end_date=lack_early_end,
        )
        # 4.3. Upload data to SQLite
        upload_data_to_sqlite(lack_early_tickers_history, interval)

    # 5. Update data on tickers lacking later history
    if len(lack_later_tickers) > 0:
        logger.info("Updating tickers lacking late history data...")
        # 5.1. Get data from web
        lack_later_tickers_history = get_tickers_history(
            tickers=lack_later_tickers,
            start=lack_later_start,
            end=lack_later_end,
            interval=interval,
        )
        # 5.2. Delete old data from SQLite to avoid duplicates
        delete_ticker_data_from_sqlite(
            lack_later_tickers,
            interval,
            start_date=lack_later_start,
            end_date=lack_later_end,
        )
        # 5.3. Upload data to SQLite
        upload_data_to_sqlite(lack_later_tickers_history, interval)

    logger.info("Data in caching DB updated")
