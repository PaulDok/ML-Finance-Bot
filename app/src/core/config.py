from datetime import datetime, timedelta, timezone
from functools import lru_cache
from json import JSONDecodeError
from typing import Any, Optional

from pydantic_settings import BaseSettings


class Config(BaseSettings):
    # ~ ~ ~ LOGGING AND DATA ~ ~ ~
    # Log filename should be unique (initialized in get_settings if it is used)
    LOG_TO_FILE: bool = (
        False  # when it's finally deployed - better to log using ELK stack and not bloat local file
    )
    LOG_FILE: Optional[str] = None

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

    return config
