# -*- coding: utf-8 -*-
"""
etp_base — base de ETP LEVE: climatologia mensal nacional pré-processada
========================================================================

Ideia central: o CAWM Simplex precisa de apenas 12 valores mensais de ETP
(mm/dia) por bacia. Uma climatologia mensal nacional (12 camadas) é ~3 ordens
de grandeza menor que a série diária — cabe em poucos MB e dispensa token e
download em runtime.

Fluxo em dois tempos:
  (1) BAKE (uma vez, LOCAL): a partir de uma base contínua (BR-DWGD/Xavier
      ETo, TerraClimate PET ou ERA5-Land), calcule a média de longo termo por
      mês de calendário e salve uma grade regular leve (.npz). Função-guia:
      `salvar_grade(...)`.
  (2) RUNTIME (no app, leve): `amostrar_bacia(grade, geom)` devolve os 12
      valores mensais (mm/dia) já no formato que o `read_etp_flexible` aceita —
      média da bacia sobre as células cobertas, sem GIS pesado.

Dependências: apenas numpy (shapely é opcional, melhora o recorte por polígono).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class GradeETP:
    """Grade regular de climatologia mensal de ETP.

    lats : (ny,) latitudes dos centros das células (graus, crescente ou não).
    lons : (nx,) longitudes dos centros.
    etp  : (12, ny, nx) ETP média mensal [mm/dia]; mês 0 = janeiro.
    fonte: string descrevendo a origem (vai para o artigo).
    """
    lats: np.ndarray
    lons: np.ndarray
    etp: np.ndarray
    fonte: str = ""

    def salvar(self, caminho: str) -> None:
        np.savez_compressed(caminho, lats=self.lats, lons=self.lons,
                            etp=self.etp.astype(np.float32), fonte=self.fonte)

    @staticmethod
    def carregar(caminho: str) -> "GradeETP":
        z = np.load(caminho, allow_pickle=True)
        return GradeETP(z["lats"], z["lons"], z["etp"].astype(float),
                        str(z["fonte"]))


def _mascara_poligono(lons2d, lats2d, geom) -> np.ndarray:
    """Máscara booleana das células cujo centro cai dentro do polígono.
    Usa shapely se disponível; senão, cai para a bounding box do geom."""
    try:
        from shapely.geometry import Point
        from shapely.prepared import prep
        pg = prep(geom)
        flat = np.column_stack([lons2d.ravel(), lats2d.ravel()])
        m = np.fromiter((pg.contains(Point(x, y)) for x, y in flat),
                        bool, len(flat)).reshape(lons2d.shape)
        if m.any():
            return m
    except Exception:
        pass
    minx, miny, maxx, maxy = geom.bounds  # shapely ou objeto com .bounds
    return ((lons2d >= minx) & (lons2d <= maxx) &
            (lats2d >= miny) & (lats2d <= maxy))


def _celula_nao_nan_mais_proxima(grade: "GradeETP", lon: float, lat: float):
    """Índice (i, j) da célula NÃO-NaN mais próxima de (lon, lat).

    Protege bacias litorâneas: a célula exatamente sobre a costa pode ser
    oceano (NaN na grade BR-DWGD); busca a terra mais próxima.
    """
    lons2d, lats2d = np.meshgrid(grade.lons, grade.lats)
    valida = ~np.isnan(grade.etp).all(axis=0)        # célula com algum mês válido
    if not valida.any():
        raise ValueError("Grade de ETP sem células válidas.")
    d2 = (lons2d - lon) ** 2 + (lats2d - lat) ** 2
    d2 = np.where(valida, d2, np.inf)
    i, j = np.unravel_index(int(np.argmin(d2)), d2.shape)
    return int(i), int(j)


def amostrar_bacia(grade: GradeETP, geom=None, ponto: Optional[tuple] = None
                   ) -> np.ndarray:
    """Retorna 12 valores mensais de ETP (mm/dia) para a bacia.

    geom  : polígono shapely (ou objeto com .bounds) — MÉDIA das células
            cobertas, ignorando NaN (oceano). Se nenhuma célula válida cair
            dentro, cai para a célula de terra mais próxima do centroide.
    ponto : (lon, lat) — célula NÃO-NaN mais próxima.
    Saída no formato que o read_etp_flexible aceita (climatologia mensal).
    """
    if geom is not None:
        lons2d, lats2d = np.meshgrid(grade.lons, grade.lats)
        m = _mascara_poligono(lons2d, lats2d, geom)
        if m.any():
            col = grade.etp[:, m]                      # (12, n_celulas)
            with np.errstate(invalid="ignore"):
                v = np.nanmean(col, axis=1)
            if not np.isnan(v).any():
                return v
        # bacia menor que 1 célula, ou só pegou oceano → terra mais próxima
        minx, miny, maxx, maxy = geom.bounds
        cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
        i, j = _celula_nao_nan_mais_proxima(grade, cx, cy)
        return grade.etp[:, i, j]
    if ponto is not None:
        i, j = _celula_nao_nan_mais_proxima(grade, ponto[0], ponto[1])
        return grade.etp[:, i, j]
    raise ValueError("Forneça `geom` (polígono) ou `ponto` (lon, lat).")


# ----------------------------------------------------------------------------
# GUIA DE BAKE (rodar LOCAL, uma vez) — pseudocódigo operacional
# ----------------------------------------------------------------------------
def salvar_grade_exemplo(caminho: str, res_graus: float = 0.25) -> GradeETP:
    """Gera uma grade SINTÉTICA leve (placeholder) só para testar o pipeline.

    SUBSTITUA pela base real: leia o NetCDF/GeoTIFF mensal (BR-DWGD ETo,
    TerraClimate PET ou ERA5-Land), agregue por mês de calendário
    (`groupby(time.month).mean()`), reamostre para `res_graus` e monte a
    GradeETP. O .npz resultante é o que o app embarca.
    """
    lats = np.arange(5.0, -34.0 - 1e-9, -res_graus)      # Brasil N→S
    lons = np.arange(-74.0, -34.0 + 1e-9, res_graus)     # O→L
    ny, nx = len(lats), len(lons)
    meses = np.arange(12)[:, None, None]
    base = 3.0 + 2.0 * np.cos((lats[None, :, None] + 10) / 25.0)   # ~clima lat
    sazonal = 0.8 * np.cos((meses - 0) / 12 * 2 * np.pi)
    etp = np.clip(base + sazonal + np.zeros((12, ny, nx)), 0.5, None)
    g = GradeETP(lats, lons, etp, fonte="SINTÉTICA (placeholder de teste)")
    g.salvar(caminho)
    return g
