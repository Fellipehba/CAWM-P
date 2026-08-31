"""Two-stage, auditable rainfall quality-control decisions for CAWM-P."""
from __future__ import annotations

import numpy as np
import pandas as pd

from consistencia_chuva import auditar_serie

EXCLUDE_STATION_MONTH = "exclude_station_month"
EXCLUDE_FLAGGED_DAY = "exclude_flagged_day"
KEEP_AS_RECORDED = "keep_as_recorded"
ACTIONS = (EXCLUDE_STATION_MONTH, EXCLUDE_FLAGGED_DAY, KEEP_AS_RECORDED)
PRESETS = ("recommended", "exclude_all", "keep_all")


def detect_flags(series: dict[str, pd.Series], progress=None, **kwargs) -> pd.DataFrame:
    """Detection only: never mutate or clean the supplied raw series."""
    rows = []
    for i, (station_id, values) in enumerate(series.items(), start=1):
        if progress:
            progress(i, len(series), str(station_id))
        flags = auditar_serie(values, **kwargs)
        if len(flags):
            flags = flags.rename(columns={"motivo": "reason", "data": "date", "valor": "value", "teste": "test"})
            flags.insert(0, "station_id", str(station_id))
            rows.append(flags)
    columns = ["station_id", "date", "value", "test", "reason"]
    return pd.concat(rows, ignore_index=True)[columns] if rows else pd.DataFrame(columns=columns)


def recommended_action(test: str) -> str:
    return EXCLUDE_STATION_MONTH if str(test) == "total_mensal" else KEEP_AS_RECORDED


def build_decisions(flags: pd.DataFrame, preset: str = "recommended") -> pd.DataFrame:
    if preset not in PRESETS:
        raise ValueError(f"unknown QC preset: {preset}")
    out = flags.copy()
    out["recommended_action"] = out["test"].map(recommended_action)
    out["scope"] = np.where(out["test"].eq("total_mensal"), "station_month", "observation")
    if preset == "recommended":
        out["selected_action"] = out["recommended_action"]
    elif preset == "exclude_all":
        out["selected_action"] = np.where(out["scope"].eq("station_month"), EXCLUDE_STATION_MONTH, EXCLUDE_FLAGGED_DAY)
    else:
        out["selected_action"] = KEEP_AS_RECORDED
    return out[["station_id", "date", "value", "test", "reason", "recommended_action", "selected_action", "scope"]]


def apply_decisions(raw_series: dict[str, pd.Series], decisions: pd.DataFrame) -> dict[str, pd.Series]:
    """Apply reviewed decisions to copies; the raw dictionary remains untouched."""
    if len(decisions) and not set(decisions["selected_action"]).issubset(ACTIONS):
        raise ValueError("invalid selected_action in QC decisions")
    invalid_scope = decisions["selected_action"].eq(EXCLUDE_STATION_MONTH) & ~decisions["test"].eq("total_mensal")
    if bool(invalid_scope.any()):
        raise ValueError("exclude_station_month is only valid for total_mensal flags")
    cleaned = {str(code): values.copy() for code, values in raw_series.items()}
    for row in decisions.itertuples(index=False):
        code = str(row.station_id)
        if code not in cleaned or row.selected_action == KEEP_AS_RECORDED:
            continue
        s = cleaned[code]
        idx = pd.DatetimeIndex(pd.to_datetime(s.index, errors="coerce"))
        date = pd.Timestamp(row.date)
        if row.selected_action == EXCLUDE_STATION_MONTH:
            mask = idx.to_period("M") == date.to_period("M")
        else:
            mask = idx == date
        s.iloc[np.flatnonzero(mask)] = np.nan
    return cleaned


def summarize_qc(flags: pd.DataFrame, decisions: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {"metric": "flagged_occurrences", "value": int(len(flags))},
        {"metric": "flagged_stations", "value": int(flags["station_id"].nunique()) if len(flags) else 0},
    ]
    for action in ACTIONS:
        n = int(decisions["selected_action"].eq(action).sum()) if len(decisions) else 0
        rows.append({"metric": f"decision_{action}", "value": n})
    return pd.DataFrame(rows)
