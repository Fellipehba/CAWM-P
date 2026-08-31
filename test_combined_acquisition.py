import pandas as pd

from combined_acquisition import acquire_rainfall_and_streamflow
from station_acquisition import AcquisitionPolicy
from station_acquisition import NoDataError, conservation_identity


def _series(code):
    return pd.Series([1.0, 2.0], index=pd.date_range("2020-01-01", periods=2), name=code)


def test_primary_action_calls_every_rain_station_and_outlet_once():
    calls = []
    def fetch(code, timeout_seconds):
        calls.append(code)
        return _series(code)
    result = acquire_rainfall_and_streamflow(
        ["P1", "P2"], "F1", fetch, fetch,
        policy=AcquisitionPolicy(max_attempts=1, backoff_seconds=0,
                                 jitter_seconds=0, max_workers=1),
    )
    assert calls == ["P1", "P2", "F1"]
    assert set(result.rainfall) == {"P1", "P2"}
    assert result.streamflow is not None
    assert len(result.rainfall_report) == 2 and len(result.outlet_report) == 1


def test_selective_retry_does_not_reacquire_valid_items():
    old_rain = pd.DataFrame([
        {"request_index": 1, "station_id": "OK", "status": "success"},
        {"request_index": 2, "station_id": "FAIL", "status": "failed_after_retries"},
    ])
    old_flow = pd.DataFrame([{"request_index": 1, "station_id": "F1", "status": "success"}])
    calls = []
    def fetch(code, timeout_seconds):
        calls.append(code)
        return _series(code)
    result = acquire_rainfall_and_streamflow(
        ["OK", "FAIL"], "F1", fetch, fetch, retry_failed_only=True,
        previous_rainfall={"OK": _series("OK")}, previous_streamflow=_series("F1"),
        previous_rainfall_report=old_rain, previous_outlet_report=old_flow,
        policy=AcquisitionPolicy(max_attempts=1, backoff_seconds=0,
                                 jitter_seconds=0, max_workers=1),
    )
    assert calls == ["FAIL"]
    assert set(result.rainfall) == {"OK", "FAIL"}


def test_partial_batch_reports_every_station_without_disappearing():
    def fetch(code, timeout_seconds):
        if code == "EMPTY":
            raise NoDataError("valid empty response")
        if code == "FAIL":
            raise TimeoutError("network timeout")
        return _series(code)
    result = acquire_rainfall_and_streamflow(
        ["OK", "EMPTY", "FAIL"], "F1", fetch, fetch,
        policy=AcquisitionPolicy(max_attempts=2, backoff_seconds=0,
                                 jitter_seconds=0, max_workers=2),
    )
    assert result.rainfall_report.status.tolist() == ["success", "no_data", "failed_after_retries"]
    assert result.rainfall_report.station_id.tolist() == ["OK", "EMPTY", "FAIL"]
    assert conservation_identity(result.rainfall_report)
