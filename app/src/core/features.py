import logging

import numpy as np
import pandas as pd

logger = logging.getLogger()


def add_simple_datetime_features(data: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Add simple Datetime features, inferred directly from 'Date' column
    """
    # NOTE: looks like some will be relevant for a given interval, some will not
    data["year"] = data["Date"].dt.strftime("%Y")
    data["month"] = data["Date"].dt.strftime("%m")
    data["day"] = data["Date"].dt.strftime("%d")
    data["year_month"] = data["Date"].dt.strftime("%Y_%m")
    data["hour"] = data["Date"].dt.strftime("%H")
    data["minute"] = data["Date"].dt.strftime("%M")

    feature_columns = ["year", "month", "day", "year_month", "hour", "minute"]

    return data, feature_columns


def add_sin_cos_datetime_features(data: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Add sin/cos transformations for month / day / hour / minute
    """
    sin_cos_map = {"month": 12, "day": 30, "hour": 24, "minute": 60}
    feature_columns = []
    for feature_in, max_val in sin_cos_map.items():
        data[f"{feature_in}_sin"] = np.sin(
            data[feature_in].astype(int) * (2.0 * np.pi / max_val)
        )
        data[f"{feature_in}_cos"] = np.cos(
            data[feature_in].astype(int) * (2.0 * np.pi / max_val)
        )
        feature_columns += [f"{feature_in}_sin", f"{feature_in}_cos"]

    return data, feature_columns


def add_lag_features(
    data: pd.DataFrame, features: list[str], lag_periods: int
) -> tuple[pd.DataFrame, list[str]]:
    """
    Добавляет лаги для указанных признаков на указанное количество периодов назад.

    data: DataFrame с исходными данными
    features: список признаков, для которых необходимо добавить лаги
    lag_periods: сколько лагов назад необходимо создать
    Возвращает:
    - обновленный DataFrame с лагами
    - список новых колонок, которые можно использовать как признаки
    """
    data = data.copy()  # Работаем с копией DataFrame
    feature_columns = []  # Список для хранения новых колонок

    # Для каждого признака создаем лаги
    for feature in features:
        for lag in range(1, lag_periods + 1):
            new_col_name = f"{feature}_lag_{lag}"
            data[new_col_name] = data[feature].shift(lag)
            feature_columns.append(new_col_name)

    # Удаляем строки с NaN значениями, которые появились из-за сдвигов
    data = data.dropna()

    return data, feature_columns


def add_rolling_features(
    data: pd.DataFrame, features: list[str], window_sizes: list[int]
) -> tuple[pd.DataFrame, list[str]]:
    """
    Добавляет скользящие характеристики для указанных признаков и окон.

    data: DataFrame с исходными данными
    features: список признаков, для которых необходимо добавить скользящие характеристики
    window_sizes: список размеров окон для расчета характеристик (например, [5, 14, 30])

    Возвращает:
    - обновленный DataFrame с новыми фичами
    - список новых колонок, которые можно использовать как признаки
    """
    data = data.copy()  # Работаем с копией DataFrame
    feature_columns = []  # Список для хранения новых колонок

    # Для каждого признака и для каждого окна
    for feature in features:
        for window_size in window_sizes:
            # Скользящее среднее
            data[f"{feature}_mean_{window_size}"] = (
                data[feature].rolling(window=window_size).mean()
            )
            feature_columns.append(f"{feature}_mean_{window_size}")

            # Скользящая медиана
            data[f"{feature}_median_{window_size}"] = (
                data[feature].rolling(window=window_size).median()
            )
            feature_columns.append(f"{feature}_median_{window_size}")

            # Скользящий минимум
            data[f"{feature}_min_{window_size}"] = (
                data[feature].rolling(window=window_size).min()
            )
            feature_columns.append(f"{feature}_min_{window_size}")

            # Скользящий максимум
            data[f"{feature}_max_{window_size}"] = (
                data[feature].rolling(window=window_size).max()
            )
            feature_columns.append(f"{feature}_max_{window_size}")

            # Скользящее стандартное отклонение
            data[f"{feature}_std_{window_size}"] = (
                data[feature].rolling(window=window_size).std()
            )
            feature_columns.append(f"{feature}_std_{window_size}")

            # Скользящий размах (макс - мин)
            data[f"{feature}_range_{window_size}"] = (
                data[f"{feature}_max_{window_size}"]
                - data[f"{feature}_min_{window_size}"]
            )
            feature_columns.append(f"{feature}_range_{window_size}")

            # Скользящее абсолютное отклонение от медианы (mad)
            data[f"{feature}_mad_{window_size}"] = (
                data[feature]
                .rolling(window=window_size)
                .apply(lambda x: np.median(np.abs(x - np.median(x))), raw=True)
            )
            feature_columns.append(f"{feature}_mad_{window_size}")

    # Удаление строк с NaN значениями, которые появляются из-за сдвигов
    data = data.dropna()

    return data, feature_columns


def add_trend_features(
    data: pd.DataFrame, features: list[str], lag_periods: int
) -> tuple[pd.DataFrame, list[str]]:
    """
    Добавляет классические финансовые признаки: отношение к предыдущим периодам, логарифмические изменения и индикаторы трендов.

    data: DataFrame с исходными данными
    features: список признаков, для которых необходимо добавить индикаторы
    lag_periods: сколько периодов назад учитывать для расчетов

    Возвращает:
    - обновленный DataFrame с новыми фичами
    - список новых колонок, которые можно использовать как признаки
    """
    data = data.copy()  # Работаем с копией DataFrame
    feature_columns = []  # Список для хранения новых колонок

    for feature in features:
        # Отношение текущего значения к предыдущему (лаг = 1)
        data[f"{feature}_ratio_1"] = data[feature] / data[feature].shift(1)
        feature_columns.append(f"{feature}_ratio_1")

        # Логарифмическое изменение (логарифм отношения текущего значения к предыдущему)
        data[f"{feature}_log_diff_1"] = np.log(data[feature] / data[feature].shift(1))
        feature_columns.append(f"{feature}_log_diff_1")

        # Momentum (разница между текущим значением и значением N периодов назад)
        data[f"{feature}_momentum_{lag_periods}"] = data[feature] - data[feature].shift(
            lag_periods
        )
        feature_columns.append(f"{feature}_momentum_{lag_periods}")

        # Rate of Change (ROC): процентное изменение за N периодов
        data[f"{feature}_roc_{lag_periods}"] = (
            (data[feature] - data[feature].shift(lag_periods))
            / data[feature].shift(lag_periods)
            * 100
        )
        feature_columns.append(f"{feature}_roc_{lag_periods}")

        # Exponential Moving Average (EMA) с периодом N
        data[f"{feature}_ema_{lag_periods}"] = (
            data[feature].ewm(span=lag_periods, adjust=False).mean()
        )
        feature_columns.append(f"{feature}_ema_{lag_periods}")

    # Удаление строк с NaN значениями, которые появились из-за сдвигов
    data = data.dropna()

    return data, feature_columns


def add_macd(
    data: pd.DataFrame, feature: str, short_window: int = 12, long_window: int = 26
) -> tuple[pd.DataFrame, list[str]]:
    """
    Добавляет индикатор MACD (разница между краткосрочным и долгосрочным EMA).

    data: DataFrame с исходными данными
    feature: признак, для которого необходимо рассчитать MACD
    short_window: окно для краткосрочного EMA (по умолчанию 12)
    long_window: окно для долгосрочного EMA (по умолчанию 26)

    Возвращает:
    - обновленный DataFrame с MACD
    - название новой колонки с MACD
    """
    data = data.copy()

    # Рассчитываем краткосрочное и долгосрочное EMA
    ema_short = data[feature].ewm(span=short_window, adjust=False).mean()
    ema_long = data[feature].ewm(span=long_window, adjust=False).mean()

    # Разница между краткосрочным и долгосрочным EMA (MACD)
    data[f"{feature}_macd"] = ema_short - ema_long

    return data, [f"{feature}_macd"]


def add_features(data: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Add features to the dataframe, calling individual specific feature functions
    data DataFrame may contain multiple Tickers
    """
    logger.info("Adding features...")
    all_feature_columns = []
    # Part 1. Features which can be mapped on multiple tickers at once

    # Simple Datetime features
    logger.info("Adding simple datetime features...")
    data, datetime_features = add_simple_datetime_features(data)
    all_feature_columns += datetime_features

    # Sin-Cos transformations
    logger.info("Adding sin-cos transformations for datetime features...")
    data, sincos_features = add_sin_cos_datetime_features(data)
    all_feature_columns += sincos_features

    # Part 2. Features which require only a single ticker
    full_data = []
    for ticker in data["Ticker"].unique():
        logger.info(f"Adding ticker-specific features for {ticker}")
        ticker_data = data[data["Ticker"] == ticker]

        # Lag features
        logger.info("Adding lag features...")
        lag_periods = 30  # пальцем в небо, TODO протестировать / подобрать лучшее
        features_to_lag = ["Open", "High", "Low", "Close", "Volume"]
        ticker_data, lag_features = add_lag_features(
            ticker_data, features_to_lag, lag_periods
        )
        all_feature_columns += lag_features

        # Rolling features
        logger.info("Adding rolling features...")
        window_sizes = [5, 14, 30]
        features_to_rolling = ["Open", "High", "Low", "Close", "Volume"]
        ticker_data, rolling_features = add_rolling_features(
            ticker_data, features_to_rolling, window_sizes
        )
        all_feature_columns += rolling_features

        # Trend features
        logger.info("Adding trend features...")
        lag_periods = 3
        features_to_trend = ["Open", "High", "Low", "Close", "Volume"]
        ticker_data, trend_features = add_trend_features(
            ticker_data, features_to_trend, lag_periods
        )
        all_feature_columns += trend_features

        # MACD
        logger.info("Adding MACD indicator...")
        macd_short_window = 12
        macd_long_window = 26
        ticker_data, macd_columns = add_macd(
            ticker_data, "Close", macd_short_window, macd_long_window
        )
        all_feature_columns += macd_columns

        full_data.append(ticker_data)

    data = pd.concat(full_data, axis=0).reset_index(drop=True)
    del full_data

    # Drop duplicates from all_feature_columns
    all_feature_columns = list(set(all_feature_columns))

    logger.info(f"Features added, {len(all_feature_columns):,d} new columns added")
    logger.info(
        f"Dataframe after features mapping: {data.shape=}, {data.isna().sum().sum():,d} NaNs"
    )

    return data, all_feature_columns
