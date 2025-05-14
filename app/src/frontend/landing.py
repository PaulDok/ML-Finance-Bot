import logging
import os
import sys

import pandas as pd
import streamlit as st
from catboost import CatBoostClassifier
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)
from sklearn.linear_model import LogisticRegression
from src.core import utils
from src.models.cnn_model import CNNModel
from src.models.gru_model import GRUModel
from src.models.lstm_model import LSTMModel
from src.models.training_loop import ml_model_strategy_training_loop
from src.strategy.bb_strategy import BollingerBandsStrategy
from src.strategy.macd_strategy import MACDStrategy
from src.strategy.sma_cross_strategy import SmaCross
from src.strategy.stoch_strategy import STOCHStrategy
from src.strategy.tema_strategy import TEMAStrategy
from streamlit_bokeh import streamlit_bokeh

logger = logging.getLogger()


# ~ ~ ~Callback functions ~ ~ ~
def data_update_callback() -> None:
    utils.update_tickers_data(
        tickers=st.session_state["update_tickers_ms"],
        start_dt=st.session_state["update_start_dt"].strftime("%Y-%m-%d"),
        end_dt=st.session_state["update_end_dt"].strftime("%Y-%m-%d"),
        interval=st.session_state["update_interval_sb"],
    )


def backtest_and_tune_callback() -> None:
    # Get history from cache and split it
    ticker_data = utils.get_history(
        tickers=st.session_state["backtest_ticker_sb"],
        start=st.session_state["backtest_train_start_dt"].strftime("%Y-%m-%d"),
        end=st.session_state["backtest_validation_end_dt"].strftime("%Y-%m-%d"),
        interval=st.session_state["backtest_interval_sb"],
        update_cache=False,
    )
    X_train, y_train, X_test, y_test, X_val, y_val = utils.train_test_valid_split(
        ticker_data,
        train_start=st.session_state["backtest_train_start_dt"].strftime("%Y-%m-%d"),
        train_end=st.session_state["backtest_train_end_dt"].strftime("%Y-%m-%d"),
        test_end=st.session_state["backtest_test_end_dt"].strftime("%Y-%m-%d"),
        valid_end=st.session_state["backtest_validation_end_dt"].strftime("%Y-%m-%d"),
        drop_leaky=False,
    )

    # Perform selection of best hyperparameters
    full_strategy_test_list = [
        {
            "strategy_type": "SmaCross",
            "strategy_class": SmaCross,
            "strategy_params_options": {
                "ma_fast_periods": [2, 3, 5, 7, 10],
                "ma_slow_periods": [5, 7, 10, 14, 20, 30],
            },
        },
        {
            "strategy_type": "MACDStrategy",
            "strategy_class": MACDStrategy,
            "strategy_params_options": {
                "fastperiod": [3, 5, 7, 14, 20],
                "slowperiod": [14, 20, 26, 30, 40],
                "signalperiod": [7, 9, 11, 14],
            },
        },
        {
            "strategy_type": "STOCHStrategy",
            "strategy_class": STOCHStrategy,
            "strategy_params_options": {
                "fastk_period": [7, 10, 14, 20],
                "slowk_period": [5, 7, 10],
                "slowk_matype": [0],
                "slowd_period": [3, 7, 10],
                "slowd_matype": [0],
            },
        },
        {
            "strategy_type": "TEMAStrategy",
            "strategy_class": TEMAStrategy,
            "strategy_params_options": {
                "period": [14, 21, 28, 40, 55, 70, 90],
            },
        },
        {
            "strategy_type": "BollingerBandsStrategy",
            "strategy_class": BollingerBandsStrategy,
            "strategy_params_options": {
                "period": [14, 21, 28, 40, 55, 70, 90],
            },
        },
    ]
    best_strategy_class, best_params, best_performance, test_summary = (
        utils.get_best_strategy(
            full_strategy_test_list,
            y_test=y_test,
            X_test=X_test,
            kpi=st.session_state["backtest_kpi_sb"],
        )
    )

    # Validate results on Validation dataset
    validation_summary = utils.validate_model_performances(
        y_val=y_val,
        X_val=X_val,
        full_test_summary=test_summary,
        kpi=st.session_state["backtest_kpi_sb"],
    )

    # Save result to session_state
    overall_result = {}
    for strategy_type in test_summary.keys():
        overall_result[strategy_type] = {
            "params": test_summary[strategy_type]["params"],
            "performance_test": test_summary[strategy_type]["performance"],
            "performance_val": validation_summary[strategy_type]["val_performance"],
            "fig_test": test_summary[strategy_type]["test_fig"],
            "fig_val": validation_summary[strategy_type]["val_figure"],
        }
    st.session_state["strategy_backtesting_result"] = overall_result


def ml_model_strategy_training_loop_callback() -> None:
    """
    Run tests using ML strategies
    """
    # Filter model options
    model_options = {
        CatBoostClassifier: {
            "int": {
                "iterations": {"low": 100, "high": 1000},
                "depth": {"low": 3, "high": 10},
            },
            "loguniform": {
                "learning_rate": {"low": 0.01, "high": 0.3},
                "l2_leaf_reg": {"low": 1e-3, "high": 10},
            },
            "float": {
                "bagging_temperature": {"low": 0, "high": 1},
                "rsm": {"low": 0.5, "high": 1.0},
                "subsample": {"low": 0.5, "high": 1.0},
            },
            "const": {"random_seed": 42, "verbose": 0, "early_stopping_rounds": 50},
        },
        LogisticRegression: {
            "loguniform": {
                "l1_ratio": {"low": 0.01, "high": 0.7},
            },
            "const": {"solver": "saga", "penalty": "elasticnet"},
            "use_eval_set": False,
        },
        CNNModel: {
            "int": {
                "window_size": {"low": 5, "high": 60},
                "batch_size": {"low": 4, "high": 128},
                "num_epochs": {"low": 5, "high": 50},
            },
            "float": {"lr": {"low": 0.0005, "high": 0.01}},
        },
        LSTMModel: {
            "int": {
                "window_size": {"low": 5, "high": 60},
                "hidden_size": {"low": 8, "high": 128},
                "num_layers": {"low": 1, "high": 3},
                "batch_size": {"low": 4, "high": 64},
                "num_epochs": {"low": 5, "high": 50},
            },
            "float": {"lr": {"low": 0.0005, "high": 0.01}},
        },
        GRUModel: {
            "int": {
                "window_size": {"low": 5, "high": 60},
                "hidden_size": {"low": 8, "high": 128},
                "num_layers": {"low": 1, "high": 3},
                "batch_size": {"low": 4, "high": 64},
                "num_epochs": {"low": 5, "high": 50},
            },
            "float": {"lr": {"low": 0.0005, "high": 0.01}},
        },
    }
    model_options = {
        k: v
        for k, v in model_options.items()
        if k.__qualname__ in st.session_state["classic_ml_models_ms"]
    }

    # Call training loop function. We don't need a result - it should be updated in DB
    _ = ml_model_strategy_training_loop(
        tickers=st.session_state["classic_ml_tickers_ms"],
        interval=st.session_state["classic_ml_interval_sb"],
        train_start=st.session_state["classic_ml_train_start_dt"].strftime("%Y-%m-%d"),
        train_end=st.session_state["classic_ml_train_end_dt"].strftime("%Y-%m-%d"),
        test_end=st.session_state["classic_ml_test_end_dt"].strftime("%Y-%m-%d"),
        valid_end=st.session_state["classic_ml_validation_end_dt"].strftime("%Y-%m-%d"),
        model_options=model_options,
    )

    st.toast("ML Model training and validation loop complete!", icon="🎉")


# ~ ~ ~ UI helper functions ~ ~ ~
def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a UI on top of a dataframe to let viewers filter columns

    Args:
        df (pd.DataFrame): Original dataframe

    Returns:
        pd.DataFrame: Filtered dataframe
    """
    modify = st.checkbox("Add filters")

    if not modify:
        return df

    df = df.copy()

    # Try to convert datetimes into a standard format (datetime, no timezone)
    for col in df.columns:
        if is_object_dtype(df[col]):
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass

        if is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.tz_localize(None)

    modification_container = st.container()

    with modification_container:
        to_filter_columns = st.multiselect("Filter dataframe on", df.columns)
        for column in to_filter_columns:
            left, right = st.columns((1, 20))
            # Treat columns with < 10 unique values as categorical
            if is_categorical_dtype(df[column]) or df[column].nunique() < 10:
                user_cat_input = right.multiselect(
                    f"Values for {column}",
                    df[column].unique(),
                    default=list(df[column].unique()),
                )
                df = df[df[column].isin(user_cat_input)]
            elif is_numeric_dtype(df[column]):
                _min = float(df[column].min())
                _max = float(df[column].max())
                step = (_max - _min) / 100
                user_num_input = right.slider(
                    f"Values for {column}",
                    min_value=_min,
                    max_value=_max,
                    value=(_min, _max),
                    step=step,
                )
                df = df[df[column].between(*user_num_input)]
            elif is_datetime64_any_dtype(df[column]):
                user_date_input = right.date_input(
                    f"Values for {column}",
                    value=(
                        df[column].min(),
                        df[column].max(),
                    ),
                )
                if len(user_date_input) == 2:
                    user_date_input = tuple(map(pd.to_datetime, user_date_input))
                    start_date, end_date = user_date_input
                    df = df.loc[df[column].between(start_date, end_date)]
            else:
                user_text_input = right.text_input(
                    f"Substring or regex in {column}",
                )
                if user_text_input:
                    df = df[df[column].astype(str).str.contains(user_text_input)]

    return df


# ~ ~ ~ UI ~ ~ ~
# Make sure to use full screen width
st.set_page_config(layout="wide")

st.title("ML Finance homework demo page")

# Organize logical blocks as tabs
tab_download, tab_history_chart, tab_backtesting, tab_classic_ml = st.tabs(
    [
        "Download to cache",
        "📈 History visualization",
        "Strategy backtesting",
        "ML models (Classic + NN)",
    ]
)

with tab_download:
    # = = = = = = = = = = = = = = = = = = = = =
    st.header("Download to local cache section")
    # Selectors to update local data
    with st.expander(
        label="Update local data cache controls",
        expanded=False,
        icon=":material/change_circle:",
    ):
        # st.header("Update local data cache controls")

        # Filters
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.selectbox(label="Interval", options=["1d"], key="update_interval_sb")
        with col2:
            st.multiselect(
                label="Tickers",
                options=utils.CACHED_CONFIG.TICKERS,  # ["BTC-USD", "A"],  # TODO: change to depend on config
                default=utils.CACHED_CONFIG.TICKERS,  # ["BTC-USD", "A"],  # TODO: change to depend on config
                key="update_tickers_ms",
            )
        with col3:
            st.date_input(
                label="From", key="update_start_dt", value=utils.CACHED_CONFIG.START_DT
            )
        with col4:
            st.date_input(
                label="To", key="update_end_dt", value=utils.CACHED_CONFIG.END_DT
            )

        # Button to run refresh
        st.button(
            label="Update data in local cache",
            use_container_width=True,
            on_click=data_update_callback,
        )

with tab_history_chart:
    # = = = = = = = = = = = = = = = = = = = = =
    st.header("History visualization")
    # Visualization controls
    with st.container(border=True):
        # Filters
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.selectbox(label="Interval", options=["1d"], key="show_interval_sb")
        with col2:
            st.multiselect(
                label="Tickers",
                options=utils.CACHED_CONFIG.TICKERS,
                default=["BTC-USD"],
                key="show_tickers_ms",
            )
        with col3:
            st.date_input(
                label="From", key="show_start_dt", value=utils.CACHED_CONFIG.START_DT
            )
        with col4:
            st.date_input(
                label="To", key="show_end_dt", value=utils.CACHED_CONFIG.END_DT
            )

        # Bool options
        with st.expander(
            label="Chart options",
            expanded=False,
            icon=":material/settings:",
        ):
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            with col1:
                st.checkbox(label="Update cache", value=False, key="options_update_cb")
            with col2:
                st.checkbox(label="Draw Close", value=True, key="options_draw_close_cb")
            with col3:
                st.checkbox(
                    label="Draw Volume", value=True, key="options_draw_volume_cb"
                )
            with col4:
                st.checkbox(
                    label="Scale Price", value=False, key="options_scale_price_cb"
                )
            with col5:
                st.checkbox(
                    label="Draw SMA and EMA", value=True, key="options_draw_ma_cb"
                )
            with col6:
                st.slider(
                    "MA smoothing period", 0, 30, 3, 1, key="options_draw_ma_slider"
                )

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.checkbox(
                    label="Draw Waterfall chart",
                    value=True,
                    key="options_draw_waterfall",
                )
            with col2:
                st.checkbox(
                    label="Draw Stochastic Oscillator chart",
                    value=True,
                    key="options_draw_stochastic",
                )
            with col3:
                st.slider(
                    "Fast_K Stochastic period",
                    0,
                    30,
                    14,
                    1,
                    key="options_fast_k_slider",
                )
            with col4:
                st.slider(
                    "Slow_D Stochastic period", 0, 30, 3, 1, key="options_slow_d_slider"
                )

        # Display chart
        if len(st.session_state["show_tickers_ms"]) > 0:
            charts = utils.get_data_and_draw_figure(
                tickers=st.session_state["show_tickers_ms"],
                start=st.session_state["show_start_dt"].strftime("%Y-%m-%d"),
                end=st.session_state["show_end_dt"].strftime("%Y-%m-%d"),
                interval=st.session_state["show_interval_sb"],
                update_cache=st.session_state["options_update_cb"],
                draw_close=st.session_state["options_draw_close_cb"],
                draw_volume=st.session_state["options_draw_volume_cb"],
                scale_price=st.session_state["options_scale_price_cb"],
                draw_ma=st.session_state["options_draw_ma_cb"],
                ma_smooth_periods=st.session_state["options_draw_ma_slider"],
                draw_waterfall=st.session_state["options_draw_waterfall"],
                draw_stochastic=st.session_state["options_draw_stochastic"],
                fastk_period=st.session_state["options_fast_k_slider"],
                slowd_period=st.session_state["options_slow_d_slider"],
            )

            # Draw main chart
            with st.expander(
                label="Price and Volume chart",
                expanded=True,
            ):
                st.plotly_chart(charts["main"])

            # Draw Waterfall
            if st.session_state["options_draw_waterfall"]:
                with st.expander(
                    label="Waterfall chart",
                    expanded=True,
                ):
                    st.plotly_chart(charts["waterfall"])

            # Draw Stochastic
            if st.session_state["options_draw_stochastic"]:
                with st.expander(
                    label="Stochastic oscillator chart",
                    expanded=True,
                ):
                    st.plotly_chart(charts["stochastic"])

with tab_backtesting:
    # = = = = = = = = = = = = = = = = = = = = =
    st.header("Strategy Backtesting")
    with st.container(border=True):
        st.write("**Please update local cache before running experiment**")

        # Container with controls
        with st.container(border=True):
            # Filters
            col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
            with col1:
                st.selectbox(
                    label="Interval", options=["1d"], key="backtest_interval_sb"
                )
            with col2:
                st.selectbox(
                    label="Ticker",
                    options=utils.CACHED_CONFIG.TICKERS,
                    index=0,  # 0: "BTC-USD", 1: "ETH-USD"
                    key="backtest_ticker_sb",
                )
            with col3:
                st.date_input(
                    label="Train start",
                    key="backtest_train_start_dt",
                    value="2010-01-01",
                )
            with col4:
                st.date_input(
                    label="Train end", key="backtest_train_end_dt", value="2023-01-01"
                )
            with col5:
                st.date_input(
                    label="Test end", key="backtest_test_end_dt", value="2024-01-01"
                )
            with col6:
                st.date_input(
                    label="Validation end",
                    key="backtest_validation_end_dt",
                    value="2025-01-01",
                )
            with col7:
                st.selectbox(label="KPI", options=["Return [%]"], key="backtest_kpi_sb")

            # Button to run model parameter tuning on Test dataset, and validate results on Validation dataset
            st.button(
                label="Run parameter tuning for different Strategies",
                use_container_width=True,
                on_click=backtest_and_tune_callback,
            )

        # Container with charts
        with st.container(border=True):
            if "strategy_backtesting_result" not in st.session_state:
                st.write("**Run a test to see result visualizations**")
            else:
                for strategy_type, result in st.session_state[
                    "strategy_backtesting_result"
                ].items():
                    with st.container(border=True):
                        st.subheader(strategy_type)
                        st.write(f"**Tuned parameters: {result['params']}**")
                        col_test, col_val = st.columns(2)
                        with col_test:
                            st.write(
                                f"**TEST {st.session_state['backtest_kpi_sb']}: {result['performance_test']}**"
                            )
                            streamlit_bokeh(
                                result["fig_test"],
                                use_container_width=True,
                                key=f"{strategy_type}_test_fig",
                            )
                        with col_val:
                            st.write(
                                f"**VALIDATION {st.session_state['backtest_kpi_sb']}: {result['performance_val']}**"
                            )
                            streamlit_bokeh(
                                result["fig_val"],
                                use_container_width=True,
                                key=f"{strategy_type}_val_fig",
                            )

with tab_classic_ml:
    st.header(
        "ML-based Strategy Backtesting, model tuning and experiment result review"
    )
    with st.container(border=True):
        st.write("**Please update local cache before running experiment**")

        # Container with controls
        with st.container(border=True):
            # Filters
            col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
            with col1:
                st.selectbox(
                    label="Interval",
                    options=["1d"],
                    key="classic_ml_interval_sb",
                )
            with col2:
                st.multiselect(
                    label="Tickers",
                    options=utils.CACHED_CONFIG.TICKERS,
                    default=["BTC-USD", "AAPL", "^GSPC"],
                    key="classic_ml_tickers_ms",
                )
            with col3:
                st.date_input(
                    label="Train start",
                    key="classic_ml_train_start_dt",
                    value="2020-01-01",
                )
            with col4:
                st.date_input(
                    label="Train end", key="classic_ml_train_end_dt", value="2024-10-01"
                )
            with col5:
                st.date_input(
                    label="Test end", key="classic_ml_test_end_dt", value="2025-01-01"
                )
            with col6:
                st.date_input(
                    label="Validation end",
                    key="classic_ml_validation_end_dt",
                    value="2025-04-01",
                )
            with col7:
                st.multiselect(
                    label="ML model types",
                    options=[
                        "LogisticRegression",
                        "CatBoostClassifier",
                        "CNNModel",
                        "LSTMModel",
                        "GRUModel",
                    ],
                    default=[
                        "LogisticRegression",
                        "CatBoostClassifier",
                        "CNNModel",
                        "LSTMModel",
                        "GRUModel",
                    ],
                    key="classic_ml_models_ms",
                )

            # Button to launch experiments
            st.button(
                label="Run ML Model selection, tuning and validation loop",
                use_container_width=True,
                on_click=ml_model_strategy_training_loop_callback,
            )

    # Container with experiment results
    with st.container(border=True):
        st.write("**Experiment results (from local DB)**")

        # Load data from DB
        df = utils.get_experiment_results()

        if df is None:
            st.write("**No experiments with ML models available yet**")
        else:
            st.dataframe(filter_dataframe(df))
