# -*- coding: utf-8 -*-
"""
etp_era5 — ETP automática a partir do ERA5-Land (Copernicus CDS)
=================================================================

Fornece a 4ª entrada do CAWM Simplex (evapotranspiração potencial) de forma
automática, em dois formatos que o read_etp_flexible já aceita:
  * CLIMATOLOGIA MENSAL (padrão): 12 valores (mm/dia), média de longo termo.
  * SÉRIE DIÁRIA (opção): mm/dia ao longo do período.

Fonte: ERA5-Land, variável `potential_evaporation` (~9 km, diária, desde
1950), recortada na bacia. Cobertura global contínua — escolhida em vez do
INMET por garantir dado em qualquer bacia (megabacias do Norte têm rede
INMET esparsa). A mesma fonte serve mensal e diário (mensal = média da
diária), simplificando o pipeline.

ARMADILHAS DO ERA5 TRATADAS AQUI (a parte que exige cuidado):
  1. SINAL: `potential_evaporation` do ERA5 é NEGATIVO (fluxo para cima,
     convenção de evaporação). ETP usável = -PEV.
  2. UNIDADE: vem em METROS de água equivalente → ×1000 = mm.
  3. ACUMULAÇÃO: a variável é acumulada; em produtos horários precisa de
     desacumulação. No ERA5-Land DIÁRIO agregado, cada registro já é o total
     do dia — tratar conforme a origem (ver `desacumular`).
  4. VIÉS: a PEV do ERA5 tende a SUPERESTIMAR a ETP de referência; expor um
     fator de correção opcional `fator_vies` (default 1.0) e registrar a
     ressalva no artigo.

ACESSO: API do CDS (cdsapi), cadastro + token gratuitos. A rede do ambiente
de desenvolvimento não alcança o CDS — o provedor de download é validado
localmente; o CONVERSOR é testável com dados sintéticos.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import pandas as pd


# ============================================================================
# 1. CONVERSOR (testável aqui) — PEV bruta do ERA5 → ETP em mm/dia
# ============================================================================

def pev_para_etp_mm(pev, unidade: str = "m", desacumular: bool = False,
                    fator_vies: float = 1.0) -> np.ndarray:
    """Converte evaporação potencial bruta do ERA5 em ETP (mm/dia, positiva).

    pev        : array de `potential_evaporation` do ERA5 (negativa, em metros
                 por padrão; pode vir já em mm se unidade='mm').
    unidade    : 'm' (ERA5 nativo) ou 'mm'.
    desacumular: se True, trata a série como acumulado e aplica diferença
                 (para produtos horários acumulados). Em ERA5-Land diário já
                 totalizado, manter False.
    fator_vies : correção multiplicativa do viés do ERA5 (default 1.0).

    Retorna ETP positiva em mm/dia. Valores negativos espúrios → 0.
    """
    x = np.asarray(pev, dtype=float)
    if desacumular:
        x = np.diff(x, prepend=x[0])
    etp = -x                                   # inverte o sinal (ETP > 0)
    if unidade == "m":
        etp = etp * 1000.0                     # m → mm
    etp = etp * fator_vies
    return np.clip(etp, 0.0, None)             # remove negativos residuais


def etp_serie_diaria(datas, pev, unidade: str = "m", desacumular: bool = False,
                     fator_vies: float = 1.0) -> pd.Series:
    """Série diária de ETP (mm/dia) a partir da PEV do ERA5."""
    etp = pev_para_etp_mm(pev, unidade, desacumular, fator_vies)
    s = pd.Series(etp, index=pd.DatetimeIndex(datas), name="etp")
    return s.sort_index().asfreq("D")


def etp_climatologia_mensal(serie_diaria: pd.Series) -> np.ndarray:
    """12 valores mensais (mm/dia) = média de longo termo por mês de calendário.

    Formato esperado pelo read_etp_flexible (climatologia mensal).
    """
    s = serie_diaria.dropna()
    clim = s.groupby(s.index.month).mean()
    return clim.reindex(range(1, 13)).to_numpy(dtype=float)


# ============================================================================
# 2. PROVEDOR DE DOWNLOAD (CDS) — estrutura pronta; validar localmente
# ============================================================================

@dataclass
class ResultadoETP:
    climatologia_mensal: Optional[np.ndarray]   # 12 valores mm/dia
    serie_diaria: Optional[pd.Series]           # mm/dia
    fonte: str
    ok: bool
    mensagem: str = ""


@dataclass
class ERA5LandETPProvider:
    """Baixa `potential_evaporation` do ERA5-Land via CDS, recortada na bacia.

    Requer `cdsapi` instalado e ~/.cdsapirc com a credencial (token CDS).
    O download do CDS é ASSÍNCRONO e lento (fila) — usar no App 1 Preparador
    (offline), não no Simulador interativo.
    """
    dataset: str = "reanalysis-era5-land"
    fator_vies: float = 1.0

    def baixar(self, bbox: tuple, ano_ini: int, ano_fim: int,
               saida_nc: str = "/tmp/era5_pev.nc") -> ResultadoETP:
        """bbox = (lat_norte, lon_oeste, lat_sul, lon_leste) — ordem do CDS.

        Baixa PEV diária no período e na janela; agrega na bacia (média
        espacial) e retorna climatologia mensal + série diária.
        """
        try:
            import cdsapi
            import xarray as xr
        except ImportError as e:
            return ResultadoETP(None, None, "era5_land", False,
                                 f"Instale cdsapi e xarray: {e}")
        try:
            c = cdsapi.Client()
            anos = [str(a) for a in range(ano_ini, ano_fim + 1)]
            c.retrieve(self.dataset, {
                "variable": "potential_evaporation",
                "year": anos,
                "month": [f"{m:02d}" for m in range(1, 13)],
                "day": [f"{d:02d}" for d in range(1, 32)],
                "time": "00:00",
                "area": list(bbox),          # N, W, S, E
                "format": "netcdf",
            }, saida_nc)
            ds = xr.open_dataset(saida_nc)
            var = ds["pev"] if "pev" in ds else ds["potential_evaporation"]
            # média espacial na janela da bacia
            serie = var.mean(dim=[d for d in var.dims if d != "time"]).to_series()
            diaria = etp_serie_diaria(serie.index, serie.values,
                                      unidade="m", fator_vies=self.fator_vies)
            return ResultadoETP(etp_climatologia_mensal(diaria), diaria,
                                "era5_land", ok=not diaria.dropna().empty)
        except Exception as e:
            return ResultadoETP(None, None, "era5_land", False, str(e))


def bbox_da_bacia(poligono_gdf, margem_graus: float = 0.2) -> tuple:
    """Caixa (N, W, S, E) ao redor da bacia, no formato que o CDS espera."""
    g = poligono_gdf.to_crs(4674)
    minx, miny, maxx, maxy = g.total_bounds
    return (round(maxy + margem_graus, 3), round(minx - margem_graus, 3),
            round(miny - margem_graus, 3), round(maxx + margem_graus, 3))
