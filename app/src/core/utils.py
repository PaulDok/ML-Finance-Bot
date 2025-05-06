import gc
import itertools
import json
import logging
import sqlite3
from datetime import datetime, timedelta
from typing import Optional, Union

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import talib
import yfinance as yf
from backtesting import Backtest
from plotly.subplots import make_subplots
from scipy.stats import zscore
from sqlalchemy import Engine, create_engine
from src.core import config, features

logger = logging.getLogger()

CACHED_CONFIG: config.Config = None

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


def create_table_for_interval_preprocessed(interval: str) -> None:
    """
    Create a table to store tickers history after cleaning and feature engineering, name dependent on INTERVAL
    """
    # NOTE: table name cannot start with digit -> reverse in such cases
    table_name = interval if interval[0].isalpha() else interval[::-1]
    table_name = f"{table_name}_preprocessed"
    table_create_query = f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
    Date TEXT,
    Ticker TEXT NOT NULL,
    Open REAL,
    Low REAL,
    High REAL,
    Close REAL,
    Volume REAL,
    features TEXT
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


def upload_data_to_sqlite(data: pd.DataFrame, interval: str, table_suffix=None) -> None:
    """
    Append data to SQLite database
    """
    engine = get_sqlite_engine()
    with engine.connect() as conn:
        table_name = interval if interval[0].isalpha() else interval[::-1]
        if table_suffix is not None:
            table_name = f"{table_name}{table_suffix}"
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

    logger.info("= = = Preprocessing pipeline started... = = =")
    preprocess_data(tickers, start_dt, end_dt, interval)
    logger.info("= = = Preprocessing complete! = = =")


def get_history(
    tickers: Union[str, list[str]] = "BTC-USD",
    start: str = "2024-03-30",
    end: str = "2024-06-02",
    interval: str = "1d",
    update_cache: bool = False,
) -> pd.DataFrame:
    """
    Get history from local cache
    """
    # Step 0: update cache if needed
    if update_cache:
        update_tickers_data(tickers, start, end, interval)

    logger.info("Getting history from local cache DB...")

    # Form an SQL statement
    table_name = interval if interval[0].isalpha() else interval[::-1]
    ticker_filter = (
        "('" + "', '".join(tickers) + "')"
        if (type(tickers) is list)
        else f"('{tickers}')"
    )
    select_query = f"""
    SELECT
        Date,
        Ticker,
        Open,
        Low,
        High,
        Close,
        Volume
    FROM {table_name}
    WHERE 
        1 = 1
        AND Ticker IN {ticker_filter}
    """
    if start is not None:
        select_query += f" AND Date >= '{start}'"
    if end is not None:
        select_query += f" AND Date <= '{end}'"

    # Run on DB
    with get_sqlite_connection() as conn:
        # logger.info(select_query)
        data = pd.read_sql(select_query, con=conn)

    # Sort by Date
    data.sort_values(by="Date", ascending=True, inplace=True)
    logger.info(f"Got history of shape {data.shape}, {data.isna().sum().sum():,d} NaNs")

    return data


def get_preprocessed_history(
    tickers: Union[str, list[str]] = "BTC-USD",
    start: str = "2024-03-30",
    end: str = "2024-06-02",
    interval: str = "1d",
) -> tuple[pd.DataFrame, list[str]]:
    """
    Get preprocessed history from local cache DB, parse features from JSON field and return DataFrame and features as list
    """
    logger.info("Getting preprocessed history from local cache DB...")

    # Form an SQL statement
    table_name = interval if interval[0].isalpha() else interval[::-1]
    table_name = f"{table_name}_preprocessed"
    ticker_filter = (
        "('" + "', '".join(tickers) + "')"
        if (type(tickers) is list)
        else f"('{tickers}')"
    )

    select_query = f"""
    SELECT
        Date,
        Ticker,
        Open,
        Low,
        High,
        Close,
        Volume,
        features
    FROM {table_name}
    WHERE 
        1 = 1
        AND Ticker IN {ticker_filter}
    """
    if start is not None:
        select_query += f" AND Date >= '{start}'"
    if end is not None:
        select_query += f" AND Date <= '{end}'"

    # Run on DB
    with get_sqlite_connection() as conn:
        # logger.info(select_query)
        data = pd.read_sql(select_query, con=conn)

    # Sort by Date
    data.sort_values(by="Date", ascending=True, inplace=True)
    logger.info(f"Got history of shape {data.shape}, {data.isna().sum().sum():,d} NaNs")

    # Parse features from 'features' JSON field
    features_df = data["features"].apply(lambda x: pd.Series(json.loads(x)))
    features = features_df.columns.to_list()
    data = (
        pd.concat([data, features_df], axis=1)
        .drop(columns=["features"])
        .reset_index(drop=True)
    )
    del features_df
    logger.info(
        f"Parsed features from JSON to separate columns: {data.shape}, {data.isna().sum().sum():,d} NaNs"
    )

    return data, features


def draw_figure(
    data: pd.DataFrame,
    draw_close: bool = True,
    draw_volume: bool = True,
    scale_price: bool = False,
    draw_ma: bool = True,
    ma_smooth_periods: int = 3,
):
    """
    Draw a Plotly Figure with tickers data
    Close price - line
    Volume - bar
    parameters:
    :data: - pd.DataFrame, containing history
    :draw_close: - bool, default True - whether to draw Close price line plot
    :draw_volume: - bool, default True - whether to draw Volume bar plot
    :scale_price: - bool, default False - whether to Standardize Close price
    """
    color_swatch = px.colors.qualitative.Dark24

    # Create Figure with secondary y-axis
    fig = make_subplots(rows=2, shared_xaxes=True, row_heights=[0.7, 0.3])

    # Add traces for each ticker
    color_idx = 0
    for ticker in data["Ticker"].unique():
        ticker_data = data[data["Ticker"] == ticker].reset_index(drop=True)

        if scale_price:
            # ~StandardScaler
            close_mean = ticker_data["Close"].mean()
            close_std = ticker_data["Close"].std()
            ticker_data["Close"] = (ticker_data["Close"] - close_mean) / close_std

        if draw_close:
            # Close price scatter plot
            fig.add_trace(
                go.Scatter(
                    x=ticker_data["Date"],
                    y=ticker_data["Close"],
                    name=f"{ticker} Close",
                    marker=dict(color=color_swatch[color_idx]),  # "red"),
                ),
                row=1,
                col=1,
            )
            color_idx += 1
            if color_idx == len(color_swatch):
                color_idx = 0

        if draw_ma:
            # SMA and EMA lines
            sma = go.Scatter(
                x=ticker_data["Date"],
                y=talib.SMA(ticker_data["Close"], ma_smooth_periods),
                name=f"{ticker} SMA_{ma_smooth_periods}",
                marker=dict(color=color_swatch[color_idx]),  # "red"),
            )
            color_idx += 1
            if color_idx == len(color_swatch):
                color_idx = 0
            ema = go.Scatter(
                x=ticker_data["Date"],
                y=talib.EMA(ticker_data["Close"], ma_smooth_periods),
                name=f"{ticker} EMA_{ma_smooth_periods}",
                marker=dict(color=color_swatch[color_idx]),  # "red"),
            )
            color_idx += 1
            if color_idx == len(color_swatch):
                color_idx = 0
            fig.add_trace(sma, row=1, col=1)
            fig.add_trace(ema, row=1, col=1)

        if draw_volume:
            # Volume bar plot
            fig.add_trace(
                go.Bar(
                    x=ticker_data["Date"],
                    y=ticker_data["Volume"],
                    name=f"{ticker} Volume",
                    marker=dict(color=color_swatch[color_idx]),  # "teal"),
                ),
                row=2,
                col=1,
            )
            color_idx += 1
            if color_idx == len(color_swatch):
                color_idx = 0

    # Add figure title
    fig.update_layout(
        title_text=f"{', '.join(data['Ticker'].unique())}{' - Standard Scaled' if scale_price else ''}"
    )

    # Set y-axes titles
    fig.update_yaxes(title_text=f"<b>price</b>", row=1, col=1)
    fig.update_yaxes(title_text=f"<b>volume</b>", row=2, col=1)

    # Draw range slider
    fig.update_xaxes(rangeslider={"visible": True}, row=1, col=1)
    fig.update_xaxes(rangeslider={"visible": True}, row=2, col=1)
    fig.update_xaxes(rangeslider_thickness=0.1)

    # Total height
    fig.update_layout(height=800)

    return fig


def draw_waterfall_chart(
    data: pd.DataFrame,
    scale_price: bool = False,
):
    """
    Draw a Plotly Waterfall Figure with tickers data
    """
    # Create Figure
    fig = go.Figure()

    # Add traces for each ticker
    for ticker in data["Ticker"].unique():
        ticker_data = data[data["Ticker"] == ticker].reset_index(drop=True)

        if scale_price:
            # ~StandardScaler
            close_mean = ticker_data["Close"].mean()
            close_std = ticker_data["Close"].std()
            ticker_data["Close"] = (ticker_data["Close"] - close_mean) / close_std

        fig.add_trace(
            go.Waterfall(
                name=f"{ticker}",
                orientation="v",
                x=ticker_data["Date"],
                textposition="auto",
                y=ticker_data["Close"].diff(),
                text=round(ticker_data["Close"].diff(), 2),
                connector={"line": {"color": "#b20710"}},
                increasing={"marker": {"color": "green"}},
                decreasing={"marker": {"color": "red"}},
            )
        )

    # Add figure title
    fig.update_layout(
        title_text=f"{', '.join(data['Ticker'].unique())}{' - Standard Scaled' if scale_price else ''} Waterfall"
    )

    # Set y-axes titles
    fig.update_yaxes(title_text=f"<b>close price change</b>")

    # Draw range slider
    fig.update_xaxes(rangeslider={"visible": True})
    fig.update_xaxes(rangeslider_thickness=0.1)

    # Total height
    fig.update_layout(height=800)

    return fig


def draw_stochastic_oscillator_chart(
    data: pd.DataFrame,
    scale_price: bool = False,
    fastk_period: int = 14,
    slowd_period: int = 3,
):
    """
    Draw a Plotly Figure with tickers data and Stochastic oscillators
    """
    # Create Figure
    fig = make_subplots(rows=2, shared_xaxes=True, row_heights=[0.5, 0.5])
    # fig = go.Figure()

    # Add traces for each ticker
    for ticker in data["Ticker"].unique():
        ticker_data = data[data["Ticker"] == ticker].reset_index(drop=True)

        if scale_price:
            # ~StandardScaler
            close_mean = ticker_data["Close"].mean()
            close_std = ticker_data["Close"].std()
            ticker_data["Close"] = (ticker_data["Close"] - close_mean) / close_std
            ticker_data["High"] = (ticker_data["High"] - close_mean) / close_std
            ticker_data["Low"] = (ticker_data["Low"] - close_mean) / close_std

        # Calculate stochastic oscillators
        stoch_k, stoch_d = talib.STOCH(
            ticker_data["High"],
            ticker_data["Low"],
            ticker_data["Close"],
            fastk_period=fastk_period,
            slowd_period=slowd_period,
        )
        stoch_k.name = "STOCH_K"
        stoch_d.name = "STOCH_D"
        ticker_data = pd.concat([ticker_data, stoch_k, stoch_d], axis=1)

        # Close price
        fig.add_trace(
            go.Scatter(
                x=ticker_data["Date"],
                y=ticker_data["Close"],
                name=f"{ticker} Close",
                line=dict(color="blue"),
            ),
            row=1,
            col=1,
        )

        # Stochastic oscillator lines
        fig.add_trace(
            go.Scatter(
                x=ticker_data["Date"],
                y=ticker_data["STOCH_K"],
                name=f"Slow %K {fastk_period}",
                line=dict(color="red"),
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=ticker_data["Date"],
                y=ticker_data["STOCH_D"],
                name=f"Slow %D {slowd_period}",
                line=dict(color="green"),
            ),
            row=2,
            col=1,
        )

        # Overbought / oversold lines
        fig.add_hline(
            y=80,
            line_dash="dot",
            line_color="gray",
            annotation_text="Overbought (80)",
            row=2,
            col=1,
        )
        fig.add_hline(
            y=20,
            line_dash="dot",
            line_color="gray",
            annotation_text="Oversold (20)",
            row=2,
            col=1,
        )

    # Add figure title
    fig.update_layout(
        title_text=f"{', '.join(data['Ticker'].unique())}{' - Standard Scaled' if scale_price else ''} with Stochastic Ostillator"
    )

    # Set y-axes titles
    fig.update_yaxes(title_text=f"<b>close price</b>")

    # Draw range slider
    fig.update_xaxes(rangeslider={"visible": True})
    fig.update_xaxes(rangeslider_thickness=0.1)

    # Total height
    fig.update_layout(height=800)

    return fig


def get_data_and_draw_figure(
    tickers: list[str],
    start: str = "2024-03-30",
    end: str = "2024-06-02",
    interval: str = "1d",
    update_cache: bool = False,
    draw_close: bool = True,
    draw_volume: bool = True,
    scale_price: bool = False,
    draw_ma: bool = True,
    ma_smooth_periods: int = 3,
    draw_waterfall: bool = True,
    draw_stochastic: bool = True,
    fastk_period: int = 14,
    slowd_period: int = 3,
):
    """
    Get data from local cache (optionally - update), draw Plotly chart and return it
    """
    # Get data from DB
    data = get_history(
        tickers=tickers,
        start=start,
        end=end,
        interval=interval,
        update_cache=update_cache,
    )

    # Generate price & volume chart
    fig = draw_figure(
        data=data,
        draw_close=draw_close,
        draw_volume=draw_volume,
        scale_price=scale_price,
        draw_ma=draw_ma,
        ma_smooth_periods=ma_smooth_periods,
    )

    # Generage waterfall chart
    if draw_waterfall:
        fig_waterfall = draw_waterfall_chart(data=data, scale_price=scale_price)
    else:
        fig_waterfall = None

    # Generate Stochastic Oscillator chart
    if draw_stochastic:
        fig_stochastic = draw_stochastic_oscillator_chart(
            data=data,
            scale_price=scale_price,
            fastk_period=fastk_period,
            slowd_period=slowd_period,
        )
    else:
        fig_stochastic = None

    return {"main": fig, "waterfall": fig_waterfall, "stochastic": fig_stochastic}


def train_test_valid_split(
    ticker_data: pd.DataFrame,
    train_start: Optional[str],
    train_end: str,
    test_end: str,
    valid_end: str,
    drop_leaky: bool = True,
) -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    """
    Split ticker data to training, testing and validation datasets
    """
    logger.info("Splitting ticker data to train/test/validation parts")
    # 0. Make sure that Date is ascending. Also reset index
    ticker_data["Date"] = pd.to_datetime(ticker_data["Date"])
    ticker_data = ticker_data.sort_values(by=["Date"], ascending=True).reset_index(
        drop=True
    )

    # 1. Drop leaky columns
    try:
        ticker_data.drop(columns=["Ticker"], inplace=True)
    except:
        pass
    if drop_leaky:
        for col in ["Open", "Low", "High", "Volume"]:
            try:
                ticker_data.drop(columns=col, inplace=True)
            except:
                pass

    # 2. Perform train/test/valid split based on 'Date'
    # we don't need anything after validation end
    ticker_data = ticker_data[ticker_data["Date"] < valid_end].reset_index(drop=True)
    # in case train_start is defined - cut it
    if train_start is not None:
        ticker_data = ticker_data[ticker_data["Date"] >= train_start].reset_index(
            drop=True
        )
    # Train parts
    X_train = (
        ticker_data[ticker_data["Date"] < train_end]
        .drop(columns=["Close"])
        .reset_index(drop=True)
    )
    y_train = ticker_data[ticker_data["Date"] < train_end]["Close"].reset_index(
        drop=True
    )
    # Test parts
    X_test = (
        ticker_data[
            (ticker_data["Date"] >= train_end) & (ticker_data["Date"] < test_end)
        ]
        .drop(columns=["Close"])
        .reset_index(drop=True)
    )
    y_test = ticker_data[
        (ticker_data["Date"] >= train_end) & (ticker_data["Date"] < test_end)
    ]["Close"].reset_index(drop=True)
    # Validation parts
    X_val = (
        ticker_data[ticker_data["Date"] >= test_end]
        .drop(columns=["Close"])
        .reset_index(drop=True)
    )
    y_val = ticker_data[ticker_data["Date"] >= test_end]["Close"].reset_index(drop=True)

    return X_train, y_train, X_test, y_test, X_val, y_val


def transform_for_backtesting(
    y_test: pd.DataFrame, X_test: pd.DataFrame
) -> pd.DataFrame:
    """
    Transform data after train/test split to the format fit for Backtesing library
    """
    bt_df = pd.concat([X_test, y_test], axis=1)
    # bt_df["Open"] = 0
    # bt_df["High"] = 0
    # bt_df["Low"] = 0
    bt_df["Date"] = pd.to_datetime(bt_df["Date"])
    bt_df.set_index("Date", inplace=True)
    return bt_df


def backtest_strategy(
    strategy_class, y_test: pd.Series, X_test: pd.DataFrame, strategy_params: dict
):
    """
    Perform a backtest of strategy_class with params on test dataset
    """
    # Initialize Backtest object
    bt = Backtest(
        transform_for_backtesting(y_test=y_test, X_test=X_test),
        strategy_class,
        cash=100000,
        commission=0.002,
        exclusive_orders=True,
    )
    # Run, using selected Strategy parameters
    stats = bt.run(**strategy_params)

    return stats, bt


def get_best_strategy_params(
    strategy_class,
    y_test: pd.Series,
    X_test: pd.DataFrame,
    strategy_params_options: dict,
    kpi: str = "Return [%]",
):
    """
    Iterate over combinations of strategy_params_options for strategy_class
    Perform backtesting experiment for each of them, and return best (on "Return [%]") params and performance
    """
    # Variables to hold best params
    best_params = None
    best_performance = -float("inf")
    best_fig = None

    for param_option in itertools.product(*strategy_params_options.values()):
        # Generate strategy params from options
        strategy_params = dict(zip(strategy_params_options.keys(), param_option))
        # logger.info(f"Testing with params: {strategy_params}")

        # Perform backtest
        stats, bt = backtest_strategy(
            strategy_class,
            y_test=y_test,
            X_test=X_test,
            strategy_params=strategy_params,
        )

        # Get key performance indicator among all metrics
        performance = stats[kpi]

        # Check if it's better than others
        if performance > best_performance:
            best_performance = performance
            best_params = strategy_params
            # Create a Bokeh figure
            best_fig = bt.plot(superimpose=False, open_browser=False)

    logger.info(f"Best Performance: {best_performance}")
    logger.info(f"Best Parameters: {best_params}")

    return best_params, best_performance, best_fig


def get_best_strategy(
    full_strategy_test_list: dict,
    y_test: pd.Series,
    X_test: pd.DataFrame,
    kpi: str = "Return [%]",
):
    # A dictionary to return all bests for strategies
    full_test_summary = {}

    # Variables to hold best params
    best_strategy_class = None
    best_params = None
    best_performance = -float("inf")

    for e in full_strategy_test_list:
        logger.info("= = = = = = = = = = = = = = = = = = = = = = = =")
        logger.info(f"Searching best params for {e['strategy_type']}...")
        class_params, class_performance, class_test_fig = get_best_strategy_params(
            e["strategy_class"],
            y_test=y_test,
            X_test=X_test,
            strategy_params_options=e["strategy_params_options"],
            kpi=kpi,
        )

        full_test_summary[e["strategy_type"]] = {
            "strategy_class": e["strategy_class"],
            "params": class_params,
            "performance": class_performance,
            "test_fig": class_test_fig,
        }

        # Check if it's better than others
        if class_performance > best_performance:
            best_performance = class_performance
            best_params = class_params
            best_strategy_class = e["strategy_class"]

    logger.info("= = = = = = = = = = = = = = = = = = = = = = = =")
    logger.info(f"Best Strategy Class: {str(best_strategy_class)}")
    logger.info(f"Best Performance: {best_performance}")
    logger.info(f"Best Parameters: {best_params}")
    logger.info("= = = = = = = = = = = = = = = = = = = = = = = =")

    return best_strategy_class, best_params, best_performance, full_test_summary


def validate_model_performances(
    y_val: pd.Series,
    X_val: pd.DataFrame,
    full_test_summary: dict,
    kpi: str = "Return[%]",
):
    """
    Validate best model hyperparameters (base on Test dataset) on Validation dataset
    """
    result = {}
    for strategy_type, strategy_test_result in full_test_summary.items():
        logger.info(f"Validating {strategy_type}...")
        # Perform a Backtest on Validation dataset
        bt = Backtest(
            transform_for_backtesting(y_val, X_val),
            strategy_test_result["strategy_class"],
            cash=100000,
            commission=0.002,
            exclusive_orders=True,
        )
        stats = bt.run(**strategy_test_result["params"])
        val_kpi = stats[kpi]
        logger.info(
            f"{kpi}: TEST - {strategy_test_result['performance']} | VAL - {val_kpi}"
        )

        # Create a Bokeh figure
        fig = bt.plot(superimpose=False, open_browser=False)

        # Save to output
        result[strategy_type] = {"val_performance": val_kpi, "val_figure": fig}

    return result


def detect_anomalies_z_diff(df: pd.DataFrame, column: str):
    df_copy = df.copy()
    df_copy["Z-score"] = zscore(df_copy[column].diff().fillna(0))
    anomalies_idx = df_copy[abs(df_copy["Z-score"]) > 1.5].index
    return anomalies_idx


def clean_anomalies_and_fill_gaps(
    data: pd.DataFrame, end_dt: str, interval: str
) -> pd.DataFrame:
    """
    Data preparation function:
    1) Detect anomalies (separately per Ticker and all its value columns), remove them;
    2) Fill missing values (both from missing in raw data and from anomaly detection step)
    """
    # 1) Detect anomalies separately for each ticker and for each column, replace them with NANs
    logger.info("Detecting and removing anomalies...")
    outlier_free_data = []
    for ticker in data["Ticker"].unique():
        data_ticker = data[data["Ticker"] == ticker]

        for column in ["Open", "Low", "High", "Close", "Volume"]:
            column_anomalies = detect_anomalies_z_diff(data_ticker, column)
            data_ticker.loc[column_anomalies, column] = None

        outlier_free_data.append(data_ticker)
    data = pd.concat(outlier_free_data, axis=0).reset_index(drop=True)

    # 2) Fill missing values
    logger.info("Filling missing values...")
    gap_filled_data = []
    for ticker in data["Ticker"].unique():
        data_ticker = data[data["Ticker"] == ticker]

        # Make sure we have end of requested time period
        data_ticker["Date"] = data_ticker["Date"].astype("datetime64[ns]")
        if pd.to_datetime(end_dt) not in data_ticker["Date"].values:
            # Append a row with end date
            data_ticker = pd.concat(
                [
                    data_ticker,
                    pd.DataFrame({"Date": pd.to_datetime(end_dt)}, index=[0]),
                ],
                axis=0,
            ).reset_index(drop=True)

        # Time resampling based on 'Date' field
        data_ticker["Date"] = data_ticker["Date"].astype("datetime64[ns]")
        data_ticker.set_index("Date", inplace=True)
        data_ticker = data_ticker.resample(interval).asfreq()

        # Ticker is always same, ffill/bfill it
        data_ticker["Ticker"] = data_ticker["Ticker"].ffill().bfill()
        # Linear interpolation for value columns
        for column in ["Open", "Low", "High", "Close", "Volume"]:
            data_ticker[column] = data_ticker[column].interpolate(
                "linear"
            )  # it does ffill as well, but run next steps just in case
            data_ticker[column] = data_ticker[column].ffill().bfill()

        # Reset index to make output shape same as input
        data_ticker = data_ticker.reset_index(drop=False)

        gap_filled_data.append(data_ticker)
    data = pd.concat(gap_filled_data, axis=0).reset_index(drop=True)

    logger.info("Data cleaned from anomalies. Filled missing values in it.")

    return data


def filter_new_data_only(
    data: pd.DataFrame, interval: str, tickers: list[str]
) -> pd.DataFrame:
    """
    Check which relevant data (after preprocessing) already exists in DB, and filter only the new part
    """
    # Form an SQL statement
    table_name = interval if interval[0].isalpha() else interval[::-1]
    table_name = f"{table_name}_preprocessed"
    query = f"""
    SELECT
        Date,
        Ticker,
        1 AS IS_IN_DB
    FROM {table_name}
    """

    # Run query on DB
    conn = get_sqlite_connection()
    preprocessed_already_existing = pd.read_sql(query, con=conn)
    conn.close()

    # Filter tickers
    preprocessed_already_existing = preprocessed_already_existing[
        preprocessed_already_existing["Ticker"].isin(tickers)
    ].reset_index(drop=True)

    # Format datetime columns
    data["Date"] = pd.to_datetime(data["Date"])
    preprocessed_already_existing["Date"] = pd.to_datetime(
        preprocessed_already_existing["Date"]
    )

    # Merge with input data
    data = pd.merge(
        data, preprocessed_already_existing, on=["Date", "Ticker"], how="outer"
    )

    # New data has NANs in 'IS_IN_DB' column, filter it
    data = (
        data[data["IS_IN_DB"].isna()].reset_index(drop=True).drop(columns=["IS_IN_DB"])
    )

    return data


def preprocess_data(
    tickers: list[str], start_dt: Optional[str], end_dt: Optional[str], interval: str
) -> None:
    """
    Perform preprocessing of data, from DB table with cached raw to the one with preprocessed state
    """
    # 1) Get data from DB
    logger.info(f"Getting raw data from DB...")
    data = get_history(
        tickers=tickers,
        start=start_dt,  # TODO: при первом запуске правильнее всё же обработать весь объем истории
        end=end_dt,  # TODO: при первом запуске правильнее всё же обработать весь объем истории
        interval=interval,
        update_cache=False,  # мы только что обновили данные в utils.update_tickers_data()
    )
    logger.info(f"Input Data in DB: {data.shape=}, {data.isna().sum().sum():,d} NaNs")

    # 2) Clean it
    logger.info("Cleaning data from anomalies and filling missing values...")
    data = clean_anomalies_and_fill_gaps(data, end_dt=end_dt, interval=interval)
    logger.info(
        f"Cleaned from anomalies and filled gaps: {data.shape=}, {data.isna().sum().sum():,d} NaNs"
    )

    # 3) Feature engineering
    logger.info("Performing feature engineering...")
    data, all_feature_columns = features.add_features(data)
    logger.info(
        f"Feature engineering complete: {data.shape=}, {data.isna().sum().sum():,d} NaNs"
    )
    logger.info(f"{len(all_feature_columns):,d} features in total")

    # 4) Create a table, if it doesn't exist yet
    logger.info(
        f"Making sure table with processed data in database for {interval=} exists"
    )
    create_table_for_interval_preprocessed(interval)

    # 5) Filter only new dates / tickers
    data = filter_new_data_only(data, interval, tickers)
    logger.info(
        f"Filtered only new data: {data.shape=}, {data.isna().sum().sum():,d} NaNs"
    )

    # 6) Convert features into a single JSON column
    data["features"] = data[all_feature_columns].apply(lambda x: x.to_json(), axis=1)
    data.drop(columns=all_feature_columns, inplace=True)
    logger.info(
        f"Features converted in a single JSON column: {data.shape=}, {data.isna().sum().sum():,d} NaNs"
    )

    # 7) Upload to DB
    logger.info("Uploading to DB...")
    upload_data_to_sqlite(data, interval, table_suffix="_preprocessed")

    logger.info("Preprocessing complete, new data uploaded!")
