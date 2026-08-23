"""Auditable, bounded station acquisition for CAWM-P.

The module is independent of Streamlit and of the IDW implementation. It
records one row per requested item, including duplicates and stations that
were not attempted, so partial batches cannot masquerade as complete runs.
"""
from __future__ import annotations

import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Iterable, Mapping, Optional

import pandas as pd


class NoDataError(RuntimeError):
    """The provider answered successfully but supplied no usable observations."""


class AcquisitionStatus(str, Enum):
    SUCCESS = "success"
    NO_DATA = "no_data"
    FAILED = "failed_after_retries"
    USER_UPLOADED = "user_uploaded"
    NOT_ATTEMPTED = "not_attempted"


@dataclass(frozen=True)
class AcquisitionPolicy:
    max_attempts: int = 3
    timeout_seconds: float = 30.0
    backoff_seconds: float = 0.25
    jitter_seconds: float = 0.1
    max_workers: int = 8

    def __post_init__(self):
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if self.timeout_seconds <= 0 or self.backoff_seconds < 0 or self.jitter_seconds < 0:
            raise ValueError("timeout_seconds must be > 0; backoff and jitter must be >= 0")
        if self.max_workers < 1:
            raise ValueError("max_workers must be >= 1")


@dataclass(frozen=True)
class BatchResult:
    series: Mapping[str, pd.Series]
    report: pd.DataFrame

    @property
    def completed_with_warnings(self) -> bool:
        return bool((self.report["status"] != AcquisitionStatus.SUCCESS.value).any())


REPORT_COLUMNS = [
    "request_index", "station_id", "selected_spatially", "attempted", "status",
    "provider_method", "start_date", "end_date", "n_records", "n_valid",
    "error_type", "error_message", "retries", "elapsed_seconds",
    "timeout_seconds", "duplicate_of_request_index",
]


def normalize_station_code(value) -> str:
    code = str(value).strip()
    return code[:-2] if code.endswith(".0") else code


def acquire_stations(
    station_codes: Iterable[str],
    fetch_station: Callable[..., Optional[pd.Series]],
    *,
    policy: AcquisitionPolicy = AcquisitionPolicy(),
    progress: Optional[Callable[[int, int, str], None]] = None,
    sleep: Callable[[float], None] = time.sleep,
) -> BatchResult:
    """Acquire in deterministic input order with bounded retries.

    ``fetch_station`` must accept ``(station_code, timeout_seconds=...)``.
    Provider ``NoDataError`` or an empty/None series becomes ``no_data``;
    other exceptions become ``failed`` after the retry budget.
    """
    codes = [normalize_station_code(c) for c in station_codes]
    output: dict[str, pd.Series] = {}
    rows_by_index: dict[int, dict] = {}
    first_index: dict[str, int] = {}
    total = len(codes)

    for request_index, code in enumerate(codes, 1):
        if code in first_index:
            rows_by_index[request_index] = {
                "request_index": request_index, "station_id": code,
                "selected_spatially": True, "attempted": False,
                "status": AcquisitionStatus.NOT_ATTEMPTED.value,
                "provider_method": "ANA_HidroSerieHistorica",
                "start_date": "", "end_date": "", "n_records": 0, "n_valid": 0,
                "error_type": "DuplicateStationCode",
                "error_message": "duplicate request; first occurrence retained",
                "retries": 0, "elapsed_seconds": 0.0,
                "timeout_seconds": policy.timeout_seconds,
                "duplicate_of_request_index": first_index[code],
            }
            continue
        first_index[code] = request_index

    def _one(request_index: int, code: str):
        started = time.perf_counter()
        status = AcquisitionStatus.FAILED
        attempts = 0
        error_type = ""
        message = ""
        valid_days = 0
        for attempt in range(1, policy.max_attempts + 1):
            attempts = attempt
            try:
                series = fetch_station(code, timeout_seconds=policy.timeout_seconds)
                if series is None or not isinstance(series, pd.Series) or not series.notna().any():
                    raise NoDataError("provider returned no usable daily observations")
                series = series.copy()
                series.index = pd.to_datetime(series.index)
                series = series.sort_index()
                valid_days = int(series.notna().sum())
                status = AcquisitionStatus.SUCCESS
                return code, series, {
                    "request_index": request_index, "station_id": code,
                    "selected_spatially": True, "attempted": True,
                    "status": status.value, "provider_method": "ANA_HidroSerieHistorica",
                    "start_date": str(series.index.min().date()),
                    "end_date": str(series.index.max().date()),
                    "n_records": int(len(series)), "n_valid": valid_days,
                    "error_type": "", "error_message": "", "retries": attempts - 1,
                    "elapsed_seconds": round(time.perf_counter() - started, 6),
                    "timeout_seconds": policy.timeout_seconds,
                    "duplicate_of_request_index": "",
                }
            except NoDataError as exc:
                status = AcquisitionStatus.NO_DATA
                error_type = type(exc).__name__
                message = str(exc)
                break
            except Exception as exc:  # provider/network errors are retryable here
                error_type = type(exc).__name__
                message = str(exc)
                if attempt < policy.max_attempts and policy.backoff_seconds:
                    sleep(policy.backoff_seconds * (2 ** (attempt - 1))
                          + random.uniform(0.0, policy.jitter_seconds))

        return code, None, {
            "request_index": request_index, "station_id": code,
            "selected_spatially": True, "attempted": True,
            "status": status.value, "provider_method": "ANA_HidroSerieHistorica",
            "start_date": "", "end_date": "", "n_records": 0, "n_valid": valid_days,
            "error_type": error_type, "error_message": message,
            "retries": max(0, attempts - 1),
            "elapsed_seconds": round(time.perf_counter() - started, 6),
            "timeout_seconds": policy.timeout_seconds,
            "duplicate_of_request_index": "",
        }

    unique_jobs = [(idx, code) for code, idx in first_index.items()]
    completed = len(rows_by_index)
    with ThreadPoolExecutor(max_workers=min(policy.max_workers, max(1, len(unique_jobs))),
                            thread_name_prefix="cawm-ana") as executor:
        futures = {executor.submit(_one, idx, code): (idx, code)
                   for idx, code in unique_jobs}
        for future in as_completed(futures):
            idx, code = futures[future]
            returned_code, series, row = future.result()
            rows_by_index[idx] = row
            if series is not None:
                output[returned_code] = series
            completed += 1
            if progress:
                progress(completed, total, code)

    report = pd.DataFrame([rows_by_index[i] for i in sorted(rows_by_index)],
                          columns=REPORT_COLUMNS)
    ordered_output = {code: output[code] for code in codes if code in output}
    return BatchResult(ordered_output, report)


def summarize_report(report: pd.DataFrame) -> dict[str, int]:
    counts = report["status"].value_counts().to_dict() if len(report) else {}
    return {status.value: int(counts.get(status.value, 0)) for status in AcquisitionStatus}


def conservation_identity(report: pd.DataFrame) -> bool:
    """Every request must belong to exactly one terminal status."""
    counts = summarize_report(report)
    return sum(counts.values()) == len(report) and report["request_index"].is_unique
