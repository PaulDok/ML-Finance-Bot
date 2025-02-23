import gc
import logging
from datetime import datetime, timedelta

from src.core import config

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
