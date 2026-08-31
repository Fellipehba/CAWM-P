"""Minimal Streamlit harness for a one-click combined-acquisition AppTest."""
import pandas as pd
import streamlit as st

from combined_acquisition import acquire_rainfall_and_streamflow
from station_acquisition import AcquisitionPolicy

st.session_state.setdefault("calls", [])
calls = []


def fetch(code, timeout_seconds):
    calls.append(code)
    return pd.Series([1.0], index=[pd.Timestamp("2020-01-01")])


if st.button("Acquire rainfall and outlet streamflow (ANA)", type="primary"):
    result = acquire_rainfall_and_streamflow(
        ["P1", "P2"], "F1", fetch, fetch,
        policy=AcquisitionPolicy(max_attempts=1, backoff_seconds=0,
                                 jitter_seconds=0, max_workers=1),
    )
    st.session_state.calls = calls
    st.session_state.report_rows = len(result.rainfall_report) + len(result.outlet_report)
