"""One-action rainfall + outlet-flow acquisition with selective retries."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import pandas as pd

from station_acquisition import AcquisitionPolicy, BatchResult, acquire_stations


@dataclass(frozen=True)
class CombinedAcquisitionResult:
    rainfall: Mapping[str, pd.Series]
    streamflow: pd.Series | None
    rainfall_report: pd.DataFrame
    outlet_report: pd.DataFrame


def _failed_ids(report: pd.DataFrame | None) -> list[str]:
    if report is None or report.empty:
        return []
    return report.loc[report["status"].eq("failed_after_retries"), "station_id"].astype(str).tolist()


def _merge_report(old: pd.DataFrame | None, new: pd.DataFrame) -> pd.DataFrame:
    if old is None or old.empty:
        return new.copy()
    replaced = set(new["station_id"].astype(str))
    kept = old[~old["station_id"].astype(str).isin(replaced)].copy()
    return pd.concat([kept, new], ignore_index=True).sort_values("request_index").reset_index(drop=True)


def acquire_rainfall_and_streamflow(
    rain_codes,
    outlet_code: str,
    fetch_rain: Callable,
    fetch_flow: Callable,
    *,
    policy: AcquisitionPolicy = AcquisitionPolicy(),
    previous_rainfall: Mapping[str, pd.Series] | None = None,
    previous_streamflow: pd.Series | None = None,
    previous_rainfall_report: pd.DataFrame | None = None,
    previous_outlet_report: pd.DataFrame | None = None,
    retry_failed_only: bool = False,
    retry_scope: str = "all",
    progress=None,
) -> CombinedAcquisitionResult:
    """Acquire both required inputs once, or retry only prior technical failures."""
    rain_codes = [str(c) for c in rain_codes]
    outlet_code = str(outlet_code)
    if retry_scope not in {"all", "rainfall", "outlet"}:
        raise ValueError("retry_scope must be all, rainfall, or outlet")
    if retry_failed_only:
        requested_rain = (_failed_ids(previous_rainfall_report)
                          if retry_scope in {"all", "rainfall"} else [])
        requested_flow = (_failed_ids(previous_outlet_report)[:1]
                          if retry_scope in {"all", "outlet"} else [])
    else:
        requested_rain = rain_codes
        requested_flow = [outlet_code]

    rain_batch = acquire_stations(requested_rain, fetch_rain, policy=policy, progress=progress)
    flow_policy = AcquisitionPolicy(
        max_attempts=policy.max_attempts, timeout_seconds=policy.timeout_seconds,
        backoff_seconds=policy.backoff_seconds, jitter_seconds=policy.jitter_seconds,
        max_workers=1,
    )
    flow_batch = acquire_stations(requested_flow, fetch_flow, policy=flow_policy)

    rainfall = dict(previous_rainfall or {})
    rainfall.update(rain_batch.series)
    streamflow = previous_streamflow
    if flow_batch.series:
        streamflow = next(iter(flow_batch.series.values()))
    rain_report = _merge_report(previous_rainfall_report if retry_failed_only else None, rain_batch.report)
    outlet_report = _merge_report(previous_outlet_report if retry_failed_only else None, flow_batch.report)
    return CombinedAcquisitionResult(rainfall, streamflow, rain_report, outlet_report)
