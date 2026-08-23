# -*- coding: utf-8 -*-
"""
chuva_media_idw — Chuva média da bacia por IDW (Inverse Distance Weighting)
============================================================================

Porte fiel do método de duas etapas da planilha PLU_MEDIA_IDW_SIRINHAEM
(validado contra a saída do Excel em diferença zero):

  Etapa 1 — soma ponderada com pesos fixos, posto faltante = contribuição 0:
            media1(t) = Σ_i  w_i · P_i(t)         (P_i < 0  →  tratado como 0)

  Etapa 2 — correção pela cobertura disponível (renormalização):
            cobertura(t) = Σ_i w_i · [P_i(t) válido]
            media_corr(t) = media1(t) / cobertura(t)

A renormalização é hidrologicamente necessária: se apenas parte da bacia
(em peso) reportou num dia, a soma crua subestima a chuva. O peso w_i é
1/d_i^power normalizado (IDW clássico, power = 1 na planilha original),
calculado a partir de coordenadas + centroide, OU fornecido pronto.

Código de falha: valores < 0 (ex.: -1, convenção da ANA/planilha) e NaN
são tratados como ausência. A saída é uma série diária pronta para o
CAWM Simplex; dias com cobertura abaixo do limiar viram NaN (falha), que
o modelo já tolera.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------------
# 1. Cálculo dos pesos IDW a partir de coordenadas + centroide
# ----------------------------------------------------------------------------

def idw_weights(lons: np.ndarray, lats: np.ndarray,
                centroid_lon: float, centroid_lat: float,
                power: float = 1.0,
                geographic: bool = True) -> np.ndarray:
    """Pesos IDW normalizados (somam 1) para postos em (lons, lats) em
    relação ao centroide da bacia.

    power : expoente do IDW (1 = planilha original; 2 também é comum).
    geographic : se True, converte graus em km (cos-lat) antes da distância;
                 se False, usa distância euclidiana direta em graus (idêntico
                 à planilha, que opera em graus decimais).

    Posto coincidente com o centroide (d = 0) recebe todo o peso.
    """
    lons = np.asarray(lons, float); lats = np.asarray(lats, float)
    if geographic:
        km_lat = 111.32
        dx = (lons - centroid_lon) * km_lat * np.cos(np.radians(centroid_lat))
        dy = (lats - centroid_lat) * km_lat
    else:
        dx = lons - centroid_lon
        dy = lats - centroid_lat
    dist = np.sqrt(dx ** 2 + dy ** 2)
    if np.any(dist == 0):                       # posto sobre o centroide
        w = (dist == 0).astype(float)
        return w / w.sum()
    inv = 1.0 / dist ** power
    return inv / inv.sum()


# ----------------------------------------------------------------------------
# 2. Chuva média IDW com correção de cobertura
# ----------------------------------------------------------------------------

@dataclass
class IDWResult:
    rainfall: pd.Series          # chuva média corrigida (mm/dia); NaN p/ falha
    coverage: pd.Series          # fração da bacia (em peso) coberta por dia
    media1: pd.Series            # soma ponderada bruta (etapa 1)
    weights: np.ndarray          # pesos usados (normalizados)
    n_below_threshold: int       # dias marcados como falha por baixa cobertura
    n_all_missing: int           # dias com todos os postos ausentes


def basin_mean_rainfall(
        station_data: pd.DataFrame,
        weights: Optional[np.ndarray] = None,
        coords: Optional[dict] = None,
        centroid: Optional[tuple] = None,
        power: float = 1.0,
        geographic: bool = True,
        min_coverage: float = 0.5,
        missing_flag: float = 0.0,
        renormalize: bool = True) -> IDWResult:
    """Calcula a chuva média da bacia por IDW.

    station_data : DataFrame (índice = datas; colunas = postos) com a chuva
                   diária. Valores < `missing_flag_threshold` (ver abaixo) e
                   NaN são tratados como ausência.
    weights      : pesos prontos (mesma ordem das colunas). Se None, são
                   calculados de `coords` + `centroid`.
    coords       : {coluna: (lon, lat)} para calcular os pesos. Necessário
                   se weights=None.
    centroid     : (lon, lat) da bacia. Necessário se weights=None.
    power, geographic : parâmetros do IDW (ver idw_weights).
    min_coverage : fração mínima da bacia (em peso) que precisa ter dado no
                   dia; abaixo disso o dia vira NaN (falha). 0.5 = exige que
                   metade da bacia, em peso, esteja coberta.
    missing_flag : valores estritamente menores que este são ausência
                   (default 0.0 → trata negativos como o -1 da planilha).
                   Zeros de chuva (0.0) NÃO são ausência.
    renormalize  : True reproduz a "média corrigida" da planilha; False
                   devolve a soma ponderada crua ("MÉDIA1").

    Retorna IDWResult.
    """
    cols = list(station_data.columns)
    if weights is None:
        if coords is None or centroid is None:
            raise ValueError("Forneça `weights`, ou `coords` + `centroid`.")
        lons = np.array([coords[c][0] for c in cols])
        lats = np.array([coords[c][1] for c in cols])
        weights = idw_weights(lons, lats, centroid[0], centroid[1],
                              power=power, geographic=geographic)
    weights = np.asarray(weights, float)
    if len(weights) != len(cols):
        raise ValueError(f"{len(weights)} pesos para {len(cols)} postos.")
    weights = weights / weights.sum()           # garante soma 1

    vals = station_data.to_numpy(dtype=float)
    valid = ~np.isnan(vals) & (vals >= missing_flag)   # negativos = ausência
    contrib = np.where(valid, vals, 0.0)

    media1 = (weights[None, :] * contrib).sum(axis=1)          # etapa 1
    coverage = (weights[None, :] * valid).sum(axis=1)          # Σ pesos válidos

    if renormalize:
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(coverage > 0, media1 / coverage, np.nan)
    else:
        mean = media1.copy()

    below = coverage < min_coverage                            # falha
    mean = np.where(below, np.nan, mean)

    idx = station_data.index
    return IDWResult(
        rainfall=pd.Series(mean, index=idx, name="p"),
        coverage=pd.Series(coverage, index=idx, name="coverage"),
        media1=pd.Series(media1, index=idx, name="media1"),
        weights=weights,
        n_below_threshold=int(below.sum()),
        n_all_missing=int((coverage == 0).sum()))


# ----------------------------------------------------------------------------
# 3. Leitor do formato da planilha (postos em colunas, -1 = falha)
# ----------------------------------------------------------------------------

def read_station_csv(path_or_buf, sep: str = ";", encoding: str = "latin-1",
                     date_col: int = 0, first_station_col: int = 1,
                     n_stations: Optional[int] = None,
                     skip_header_rows: int = 7,
                     dayfirst: bool = True) -> pd.DataFrame:
    """Lê um CSV no formato da planilha IDW: cabeçalhos nas primeiras linhas,
    datas em `date_col`, postos a partir de `first_station_col`. Decimal com
    vírgula é aceito. Retorna DataFrame (datas × postos)."""
    raw = pd.read_csv(path_or_buf, sep=sep, encoding=encoding, header=None,
                      dtype=str)
    last = (first_station_col + n_stations) if n_stations else raw.shape[1]
    dates = pd.to_datetime(raw.iloc[skip_header_rows:, date_col],
                           dayfirst=dayfirst, errors="coerce")
    block = raw.iloc[skip_header_rows:, first_station_col:last]
    block = block.apply(lambda s: pd.to_numeric(
        s.str.replace(",", ".", regex=False), errors="coerce"))
    block.index = dates
    block = block[~block.index.isna()]
    block.columns = [f"posto_{i+1}" for i in range(block.shape[1])]
    return block.dropna(axis=1, how="all")
