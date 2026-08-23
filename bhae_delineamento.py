# -*- coding: utf-8 -*-
"""
bhae_delineamento — Delimitação de bacia pela Base Hidrográfica Atlas-Estudos
=============================================================================

Delimita a bacia a montante de um ponto (estação fluviométrica) usando a
BHAE da ANA (camada de TRECHOS de drenagem + camada de ÁREAS de drenagem),
sem cálculo sobre raster e sem reimplementar a aritmética de Pfafstetter.

Método (validado no Sirinhaém: 1.288,5 km², −2,1% vs. referência, com os
três estimadores coincidindo em 0,00%):

  1. SNAP do ponto ao trecho: entre os trechos a < snap_m metros, escolhe o
     de maior `nuareamont` (a estação fica no rio principal, não num afluente
     lateral próximo).
  2. ÁREA: lida direto do atributo oficial `nuareamont` do trecho do exutório
     (área a montante já calculada pela ANA na projeção Albers equivalente).
  3. TOPOLOGIA: navega os trechos a montante seguindo `nutrjus` (trecho
     imediatamente a jusante) — grafo de conectividade explícito, imune à
     paridade dos códigos Otto.
  4. POLÍGONO: dissolve as ottobacias (áreas de drenagem) dos trechos da rede.

Projeção oficial de área da BHO (do dicionário de variáveis):
  Albers equivalente +lat_1=-2 +lat_2=-22 +lat_0=-12 +lon_0=-54 (GRS80).
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Optional

import numpy as np
import geopandas as gpd
from shapely.geometry import Point

ALBERS_BHO = ("+proj=aea +lat_1=-2 +lat_2=-22 +lat_0=-12 +lon_0=-54 "
              "+x_0=0 +y_0=0 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 "
              "+units=m +no_defs")


@dataclass
class Bacia:
    area_km2: float                  # área oficial (nuareamont do exutório)
    area_topo_km2: float             # área pela soma dos trechos (controle)
    area_poly_km2: float             # área do polígono dissolvido (controle)
    cotrecho_exutorio: int
    n_trechos: int
    rio: str
    snap_dist_m: float
    polygon: gpd.GeoDataFrame        # geometria da bacia (EPSG:4674)
    trechos_montante: set            # ids dos trechos a montante


def _build_upstream_index(trechos: gpd.GeoDataFrame) -> dict:
    up = defaultdict(list)
    for t, j in zip(trechos["cotrecho"], trechos["nutrjus"]):
        if j is not None and not (isinstance(j, float) and np.isnan(j)):
            up[int(j)].append(int(t))
    return up


def _upstream_set(root: int, up_index: dict) -> set:
    seen = {root}; q = deque([root])
    while q:
        cur = q.popleft()
        for u in up_index.get(cur, []):
            if u not in seen:
                seen.add(u); q.append(u)
    return seen


def delineate(trechos: gpd.GeoDataFrame, areas: gpd.GeoDataFrame,
              lon: float, lat: float, snap_m: float = 500.0,
              dissolve: bool = True) -> Bacia:
    """Delimita a bacia a montante de (lon, lat).

    trechos : GeoDataFrame da camada GEOFT_BHAE_TRECHO_DRENAGEM
              (precisa de cotrecho, nutrjus, nuareamont, nuareacont).
    areas   : GeoDataFrame da camada GEOFT_BHAE_AREA_DRENAGEM
              (precisa de cotrecho; usada para montar o polígono).
    snap_m  : raio de busca do trecho do exutório, em metros.
    """
    tr = trechos.copy()
    tr["cotrecho"] = tr["cotrecho"].astype("Int64")
    tr["nutrjus"] = tr["nutrjus"].astype("Int64")

    tr_m = tr.to_crs(ALBERS_BHO)
    pt_m = gpd.GeoSeries([Point(lon, lat)], crs=4674).to_crs(ALBERS_BHO).iloc[0]
    tr["_dist"] = tr_m.geometry.distance(pt_m)

    prox = tr[tr["_dist"] <= snap_m]
    if prox.empty:                                  # ponto longe da rede
        prox = tr.nsmallest(5, "_dist")
    outlet = prox.nlargest(1, "nuareamont").iloc[0]  # rio principal
    oc = int(outlet["cotrecho"])

    up_index = _build_upstream_index(tr)
    montante = _upstream_set(oc, up_index)
    sub = tr[tr["cotrecho"].isin(montante)]
    area_topo = float(sub["nuareacont"].sum())

    if dissolve:
        ar = areas.copy()
        ar["cotrecho"] = ar["cotrecho"].astype("Int64")
        poly = ar[ar["cotrecho"].isin(montante)].dissolve()
        poly = poly[["geometry"]].reset_index(drop=True)
        area_poly = float(poly.to_crs(ALBERS_BHO).area.iloc[0] / 1e6)
    else:
        poly = gpd.GeoDataFrame(geometry=[], crs=4674)
        area_poly = np.nan

    rio = str(outlet.get("noriocomp", "") or "")
    return Bacia(area_km2=float(outlet["nuareamont"]),
                 area_topo_km2=area_topo, area_poly_km2=area_poly,
                 cotrecho_exutorio=oc, n_trechos=len(montante),
                 rio=rio, snap_dist_m=float(outlet["_dist"]),
                 polygon=poly, trechos_montante=montante)


def load_bhae(trechos_path: str, areas_path: str):
    """Carrega as duas camadas da BHAE (Shapefile ou GeoPackage)."""
    return gpd.read_file(trechos_path), gpd.read_file(areas_path)
