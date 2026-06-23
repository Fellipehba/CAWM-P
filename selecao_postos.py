# -*- coding: utf-8 -*-
"""
selecao_postos — Seleção de postos por buffer ao redor da bacia delineada
==========================================================================

Dado o polígono da bacia (saída de bhae_delineamento) e um inventário de
postos com coordenadas, seleciona os postos relevantes para a chuva média:
os que caem dentro da bacia mais os que estão num anel (buffer) ao redor,
para garantir cobertura espacial nas bordas (prática do IDW).

Validado no Sirinhaém contra a planilha IDW: dos 8 postos usados pela
planilha, 5 caem dentro da bacia delineada (incluindo o posto dominante,
peso 0,556) e os 8 são capturados com buffer de 10 km — confirmando que a
planilha incluiu vizinhos externos para reforçar a cobertura, exatamente o
efeito do buffer.

Saída integra-se diretamente ao módulo chuva_media_idw: os pesos podem ser
calculados aqui (centroide da bacia) e passados adiante.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

ALBERS_BHO = ("+proj=aea +lat_1=-2 +lat_2=-22 +lat_0=-12 +lon_0=-54 "
              "+x_0=0 +y_0=0 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 "
              "+units=m +no_defs")


@dataclass
class SelecaoPostos:
    postos: gpd.GeoDataFrame      # selecionados, com 'dentro', 'dist_borda_km', peso
    centroide: tuple             # (lon, lat) da bacia
    n_dentro: int
    n_buffer: int
    buffer_km: float


def inventario_to_gdf(df: pd.DataFrame, col_cod: str = "cod",
                      col_lon: str = "lon", col_lat: str = "lat"
                      ) -> gpd.GeoDataFrame:
    """Converte um DataFrame de inventário (código, lon, lat) em GeoDataFrame."""
    geom = [Point(x, y) for x, y in zip(df[col_lon], df[col_lat])]
    g = gpd.GeoDataFrame(df.copy(), geometry=geom, crs=4674)
    return g.rename(columns={col_cod: "cod", col_lon: "lon", col_lat: "lat"})


def _idw_weights(lons, lats, clon, clat, power=1.0, geographic=True):
    lons = np.asarray(lons, float); lats = np.asarray(lats, float)
    if geographic:
        kmlat = 111.32
        dx = (lons - clon) * kmlat * np.cos(np.radians(clat))
        dy = (lats - clat) * kmlat
    else:
        dx, dy = lons - clon, lats - clat
    d = np.sqrt(dx**2 + dy**2)
    if np.any(d == 0):
        w = (d == 0).astype(float); return w / w.sum()
    inv = 1.0 / d**power
    return inv / inv.sum()


def selecionar(bacia_poly: gpd.GeoDataFrame, postos: gpd.GeoDataFrame,
               buffer_km: float = 10.0, com_pesos: bool = True,
               idw_power: float = 1.0, idw_geographic: bool = True,
               tipos: tuple = ("pluviometrica",)) -> SelecaoPostos:
    """Seleciona postos dentro da bacia + anel de `buffer_km` ao redor.

    bacia_poly : GeoDataFrame com o polígono da bacia (EPSG:4674).
    postos     : GeoDataFrame de postos (EPSG:4674), coluna 'cod'.
    com_pesos  : se True, calcula o peso IDW de cada posto em relação ao
                 centroide da bacia (compatível com chuva_media_idw).
    tipos      : tipos de estação elegíveis. PADRÃO = só pluviométricas, pois
                 a chuva média (IDW) NÃO pode ser ponderada por estações
                 fluviométricas (que não medem precipitação). Passe
                 ('pluviometrica','fluviometrica') só se realmente quiser ambas.
    """
    if tipos is not None and "tipo" in postos.columns:
        postos = postos[postos["tipo"].astype(str).str.lower()
                        .str.startswith(tuple(t[:6] for t in tipos))].copy()
    bacia_m = bacia_poly.to_crs(ALBERS_BHO)
    postos_m = postos.to_crs(ALBERS_BHO)
    poly = bacia_m.union_all()
    poly_buf = poly.buffer(buffer_km * 1000.0)

    sel_mask = postos_m.within(poly_buf).values
    sel = postos[sel_mask].copy()
    sel_m = postos_m[sel_mask]

    dentro = sel_m.within(poly).values
    dist = sel_m.geometry.distance(poly.boundary).values / 1000.0
    dist = np.where(dentro, -dist, dist)            # negativo = dentro
    sel["dentro"] = dentro
    sel["dist_borda_km"] = dist

    cen_proj = bacia_m.geometry.centroid.iloc[0]
    cen = gpd.GeoSeries([cen_proj], crs=ALBERS_BHO).to_crs(4674).iloc[0]
    centroide = (float(cen.x), float(cen.y))

    if com_pesos and len(sel):
        sel["peso_idw"] = _idw_weights(sel["lon"].values, sel["lat"].values,
                                       centroide[0], centroide[1],
                                       power=idw_power,
                                       geographic=idw_geographic)

    sel = sel.sort_values("dist_borda_km").reset_index(drop=True)
    return SelecaoPostos(postos=sel, centroide=centroide,
                         n_dentro=int(dentro.sum()),
                         n_buffer=int((~dentro).sum()),
                         buffer_km=buffer_km)
