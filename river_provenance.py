"""Authoritative BHAE river-name enrichment for prepared CAWM-P basins.

River names are joined by the hydrologic outlet identifier only:
``bacias_flu_ana.exutorio`` -> ``geoft_bhae_trecho_drenagem.cotrecho``.
Station names are never parsed or used as a fallback.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

RIVER_TABLE = "geoft_bhae_trecho_drenagem"
RIVER_FIELD = "noriocomp"
SOURCE_VERSION_FIELD = "dsversao"


def normalize_station_code(value) -> str:
    code = str(value).strip()
    return code[:-2] if code.endswith(".0") else code


def read_official_river_map(gpkg: str | Path, outlets) -> pd.DataFrame:
    """Read exact outlet matches from the official BHAE GeoPackage, read-only."""
    ids = sorted({int(float(x)) for x in outlets if pd.notna(x)})
    columns = ["exutorio", "rio_bhae", "bhae_versao"]
    if not ids:
        return pd.DataFrame(columns=columns)
    placeholders = ",".join("?" for _ in ids)
    query = f"""SELECT cotrecho, {RIVER_FIELD}, {SOURCE_VERSION_FIELD}
                FROM {RIVER_TABLE} WHERE cotrecho IN ({placeholders})"""
    uri = Path(gpkg).resolve().as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as con:
        rows = con.execute(query, ids).fetchall()
    out = pd.DataFrame(rows, columns=columns)
    if out["exutorio"].duplicated().any():
        conflicts = out.groupby("exutorio")["rio_bhae"].nunique(dropna=False)
        if bool((conflicts > 1).any()):
            raise ValueError("BHAE source has conflicting river names for an outlet")
        out = out.drop_duplicates("exutorio", keep="first")
    out["rio_bhae"] = out["rio_bhae"].astype("string").str.strip()
    out.loc[out["rio_bhae"].isin(["", "None", "nan"]), "rio_bhae"] = pd.NA
    return out


def enrich_by_outlet(basins: pd.DataFrame, outlet_map: pd.DataFrame) -> pd.DataFrame:
    """Attach the official river field while preserving basin row order."""
    if "exutorio" not in basins:
        raise ValueError("basin table must contain exutorio")
    if outlet_map["exutorio"].duplicated().any():
        raise ValueError("outlet map must be unique by exutorio")
    out = basins.copy()
    out["_row_order"] = range(len(out))
    out["exutorio"] = pd.to_numeric(out["exutorio"], errors="coerce").astype("Int64")
    out = out.merge(outlet_map, on="exutorio", how="left", validate="many_to_one")
    return out.sort_values("_row_order").drop(columns="_row_order").reset_index(drop=True)


def station_river_table(raw_basins: pd.DataFrame, outlet_map: pd.DataFrame) -> pd.DataFrame:
    """Return a unique station-to-river mapping with explicit join provenance."""
    cols = ["cod_posto", "exutorio"]
    merged = enrich_by_outlet(raw_basins[cols], outlet_map)
    merged["cod_posto"] = merged["cod_posto"].map(normalize_station_code)
    if merged["cod_posto"].duplicated().any():
        conflicts = merged.groupby("cod_posto")["rio_bhae"].nunique(dropna=False)
        if bool((conflicts > 1).any()):
            raise ValueError("station has conflicting BHAE river names")
        merged = merged.drop_duplicates("cod_posto", keep="first")
    return merged
