import logging
import os
import sys

import streamlit as st
from src.core import utils

logger = logging.getLogger()


# ~ ~ ~Callback functions ~ ~ ~
def data_update_callback() -> None:
    utils.update_tickers_data(
        tickers=st.session_state["update_tickers_ms"],
        start_dt=st.session_state["update_start_dt"].strftime("%Y-%m-%d"),
        end_dt=st.session_state["update_end_dt"].strftime("%Y-%m-%d"),
        interval=st.session_state["update_interval_sb"],
    )


# ~ ~ ~ UI ~ ~ ~

st.title("ML Finance homework demo page")

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
