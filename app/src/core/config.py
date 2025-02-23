import logging
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from json import JSONDecodeError
from typing import Any, Optional

import pandas as pd
from pydantic_settings import BaseSettings

logger = logging.getLogger()


class Config(BaseSettings):
    # ~ ~ ~ LOGGING AND DATA ~ ~ ~
    # Log filename should be unique (initialized in get_settings if it is used)
    LOG_TO_FILE: bool = (
        False  # when it's finally deployed - better to log using ELK stack and not bloat local file
    )
    LOG_FILE: Optional[str] = None

    # ~ ~ ~ Project Params ~ ~ ~
    TICKERS: Optional[list[str]] = (
        None  # List of tickers to work with. If None - will be initialized to all S&P500 + S&P500 itself + set of major Cryptocurrencies
    )
    INTERVAL: str = "1d"  # TODO: change this to Literal when more clarity appears
    START_DT: Optional[str] = None
    END_DT: Optional[str] = None

    class Config:
        env_file = "dev_env"

        @classmethod
        def parse_env_var(cls, field: str, raw_val: str) -> Any:
            """
            Для преобразования из json кавычки имеют значение
            мы не можем гарантировать что элементы списка придут с обрамлением в двойные кавычки
            если получаем ошибку декодирования json, меняем одинарные кавычки на двойные и пробуем снова.
            """
            try:
                return cls.json_loads(raw_val)  # type: ignore
            except JSONDecodeError:
                raw_val = raw_val.replace("'", '"')
            return cls.json_loads(raw_val)  # type: ignore


@lru_cache
def get_config() -> Config:
    # Parse params from env variables, add default ones
    config = Config()

    # Initialize log file
    if config.LOG_TO_FILE:
        init_dt = datetime.now().astimezone(timezone.utc) + timedelta(hours=3)
        config.LOG_FILE = init_dt.strftime("./logs/%Y_%m_%d@%H_%M.log")

    if config.TICKERS is None:
        config.TICKERS = get_default_tickers()

    if config.END_DT is None:
        # Up to now
        config.END_DT = datetime.now().strftime("%Y-%m-%d")

    if config.START_DT is None:
        # 1 year from END_DT
        config.START_DT = (
            datetime.strptime(config.END_DT, "%Y-%m-%d") - timedelta(days=365)
        ).strftime("%Y-%m-%d")

    return config


def get_default_tickers() -> list:
    """
    Get a list of all tickers defined in project objective
    """
    crypto_tickers = ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD"]
    snp500_ticker = ["^GSPC"]
    # List of all S&P500 companies
    snp500_tickers = pd.read_html(
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    )[0]["Symbol"].to_list()
    # Just in case there are duplicates
    snp500_tickers = list(set(snp500_tickers))
    logger.info(
        f"Got list of S&P500 companies from Wikipedia, {len(snp500_tickers):,d} tickers"
    )

    all_tickers = crypto_tickers + snp500_ticker + snp500_tickers
    logger.info(f"{len(all_tickers):,d} tickers will be processed")

    return all_tickers
