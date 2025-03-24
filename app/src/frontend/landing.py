import logging
import os
import sys

import streamlit as st
from src.core import utils
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


# ~ ~ ~ UI ~ ~ ~
# Make sure to use full screen width
st.set_page_config(layout="wide")

st.title("ML Finance homework demo page")

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
        st.date_input(label="To", key="update_end_dt", value=utils.CACHED_CONFIG.END_DT)

    # Button to run refresh
    st.button(
        label="Update data in local cache",
        use_container_width=True,
        on_click=data_update_callback,
    )

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
        st.date_input(label="To", key="show_end_dt", value=utils.CACHED_CONFIG.END_DT)

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
            st.checkbox(label="Draw Volume", value=True, key="options_draw_volume_cb")
        with col4:
            st.checkbox(label="Scale Price", value=False, key="options_scale_price_cb")
        with col5:
            st.checkbox(label="Draw SMA and EMA", value=True, key="options_draw_ma_cb")
        with col6:
            st.slider("MA smoothing period", 0, 30, 3, 1, key="options_draw_ma_slider")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.checkbox(
                label="Draw Waterfall chart", value=True, key="options_draw_waterfall"
            )
        with col2:
            st.checkbox(
                label="Draw Stochastic Oscillator chart",
                value=True,
                key="options_draw_stochastic",
            )
        with col3:
            st.slider(
                "Fast_K Stochastic period", 0, 30, 14, 1, key="options_fast_k_slider"
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


# = = = = = = = = = = = = = = = = = = = = =
st.header("Strategy Backtesting")
with st.container(border=True):
    st.write("**Please update local cache before running experiment**")

    # Container with controls
    with st.container(border=True):
        # Filters
        col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
        with col1:
            st.selectbox(label="Interval", options=["1d"], key="backtest_interval_sb")
        with col2:
            st.selectbox(
                label="Ticker",
                options=utils.CACHED_CONFIG.TICKERS,
                index=0,  # 0: "BTC-USD", 1: "ETH-USD"
                key="backtest_ticker_sb",
            )
        with col3:
            st.date_input(
                label="Train start", key="backtest_train_start_dt", value="2010-01-01"
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
