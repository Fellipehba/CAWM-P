# -*- coding: utf-8 -*-
"""
bhae_online — bacia pronta por cod_posto + seleção de postos PLU (Estágio 1)
============================================================================

Substitui o delineamento BHAE de 5 GB em tempo real. Lê o GeoParquet
SIMPLIFICADO (~21 MB, gerado por simplificar_bhae.py) de um caminho local OU
de uma URL (R2/host) e, dado um `cod_posto` FLU, devolve a bacia pronta e os
postos PLU dentro dela (para a IDW). HydroBR/ANA seguem baixando as séries.

Fluxo do novo Passo 1:
    idx = carregar_indice(FONTE_IDX)         # busca por código/nome (sem geom)
    bacia = bacia_por_cod(FONTE_GEO, cod)    # 1 polígono pronto
    plu  = selecionar_plu(bacia, inv_plu)    # postos PLU dentro (+buffer)
"""
from __future__ import annotations

from functools import lru_cache
from typing import Optional

import pandas as pd
import geopandas as gpd


def _garantir_confiabilidade(g):
    """Garante as colunas area_geom_km2 / geometria_confiavel. Se o parquet já
    veio com elas (gerado pelo simplificar_bhae.py atual), usa direto — é
    quase instantâneo. Só recalcula (mais lento) se faltarem, para não quebrar
    com parquets antigos."""
    if "geometria_confiavel" in g.columns and "area_geom_km2" in g.columns:
        return g
    if g.geometry.name not in g.columns:
        return g   # índice sem geometria: nada a calcular aqui
    area_geom = g.to_crs(6933).area / 1e6
    area_of = g["area_usada_km2"].astype(float)
    ratio = (area_geom / area_of).where(area_of > 0)
    g["area_geom_km2"] = area_geom.round(1)
    g["geometria_confiavel"] = (ratio >= 0.5).fillna(False)
    return g


@lru_cache(maxsize=4)
def carregar_bacias(fonte_geo: str) -> gpd.GeoDataFrame:
    """Lê o GeoParquet simplificado (local ou URL). Cacheado em memória."""
    g = gpd.read_parquet(fonte_geo)
    g["cod_posto"] = g["cod_posto"].astype(str).str.strip()
    return _garantir_confiabilidade(g)


@lru_cache(maxsize=4)
def carregar_indice(fonte_idx: str) -> pd.DataFrame:
    """Índice sem geometria, para o selectbox de código/nome (instantâneo)."""
    idx = pd.read_parquet(fonte_idx)
    idx["cod_posto"] = idx["cod_posto"].astype(str).str.strip()
    return idx


def buscar(idx: pd.DataFrame, termo: str) -> pd.DataFrame:
    """Busca por código (prefixo) ou nome (contém, sem acento/caixa em AMBOS
    os lados — antes só o nome era normalizado, então buscar 'Ibiaí' com
    acento não achava 'IBIAÍ')."""
    def _sem_acento(s):
        return (s.str.normalize("NFKD")
                .str.encode("ascii", "ignore").str.decode("ascii").str.lower())
    t = _sem_acento(pd.Series([str(termo).strip()])).iloc[0]
    if not t:
        return idx.head(0)
    por_cod = idx["cod_posto"].str.startswith(t)
    nome = _sem_acento(idx["nome_posto"].astype(str))
    por_nome = nome.str.contains(t, na=False, regex=False)
    return idx[por_cod | por_nome]


def bacia_por_cod(fonte_geo: str, cod_posto: str) -> Optional[gpd.GeoDataFrame]:
    """Devolve a feição (1 linha) da bacia do posto FLU, ou None."""
    g = carregar_bacias(fonte_geo)
    cod = str(cod_posto).strip()
    sel = g[g["cod_posto"] == cod]
    return sel if len(sel) else None


class BaciaPronta:
    """Compatível com o que o Preparador espera de `ss.bacia`. Carrega os campos
    que a tela lê (`area_km2`, `rio`, `n_trechos`, `snap_dist_m`, `area_poly_km2`,
    `polygon`, `bounds`) e `trechos_montante` vazio (a bacia pronta não traz a
    rede de drenagem — o download secundário de drenagem fica indisponível)."""
    def __init__(self, polygon: gpd.GeoDataFrame, area_km2: float,
                 rio: str = "", cod_posto: str = "",
                 area_geom_km2: Optional[float] = None,
                 geometria_confiavel: Optional[bool] = None):
        self.polygon = polygon
        self.area_km2 = float(area_km2)
        self.area_poly_km2 = float(area_km2)      # mesma (geometria simplificada)
        self.area_topo_km2 = float(area_km2)      # área topológica = oficial aqui
        self.cotrecho_exutorio = 0                # metadado de proveniência (n/d na pronta)
        self.rio = rio or "—"
        self.n_trechos = 0
        self.snap_dist_m = 0.0
        self.cod_posto = str(cod_posto)
        self.trechos_montante = frozenset()
        self.trechos = None
        # Confiabilidade da geometria: PREFERE o valor pré-calculado (embutido
        # no parquet por simplificar_bhae.py — instantâneo e consistente com o
        # índice). Só recalcula se não vier pronto (parquet antigo).
        if area_geom_km2 is not None and geometria_confiavel is not None:
            self.area_geom_km2 = float(area_geom_km2)
            self.geometria_confiavel = bool(geometria_confiavel)
        else:
            try:
                self.area_geom_km2 = float(polygon.to_crs(6933).area.sum() / 1e6)
            except Exception:
                self.area_geom_km2 = float("nan")
            self.geometria_confiavel = bool(
                self.area_geom_km2 == self.area_geom_km2
                and self.area_km2 > 0
                and self.area_geom_km2 >= 0.5 * self.area_km2)

    @property
    def bounds(self):
        return tuple(self.polygon.total_bounds)

    def __getattr__(self, name):
        # Atributos de proveniência que o delineamento real expõe e a bacia
        # pronta não tem → devolve 0 em vez de quebrar (são só metadados).
        # O guard '_' evita interferir em dunders (pickle/cache do Streamlit).
        if name.startswith("_"):
            raise AttributeError(name)
        return 0.0


def bacia_pronta(fonte_geo: str, cod_posto: str) -> Optional["BaciaPronta"]:
    """Carrega a bacia pronta do posto FLU como objeto compatível com o app."""
    sel = bacia_por_cod(fonte_geo, cod_posto)
    if sel is None or sel.empty:
        return None
    r = sel.iloc[0]
    area = float(r.get("area_usada_km2", float("nan")))
    rio = str(r.get("rio_bhae", "") or "")
    ageo = r.get("area_geom_km2")
    conf = r.get("geometria_confiavel")
    return BaciaPronta(sel[[sel.geometry.name]].to_crs(4674), area,
                       rio=rio, cod_posto=cod_posto,
                       area_geom_km2=(None if pd.isna(ageo) else float(ageo)),
                       geometria_confiavel=(None if pd.isna(conf) else bool(conf)))


def selecionar_plu(bacia: gpd.GeoDataFrame, inventario_plu: pd.DataFrame,
                   buffer_km: float = 0.0,
                   col_lon: str = "lon", col_lat: str = "lat",
                   col_cod: str = "cod") -> gpd.GeoDataFrame:
    """Postos PLU dentro da bacia (opcionalmente com buffer em km).

    inventario_plu: DataFrame com colunas de código, lon, lat (pluviométricas).
    Retorna GeoDataFrame dos postos contidos, com a coluna de código preservada.
    """
    poly = bacia.to_crs(4674)
    if buffer_km and buffer_km > 0:                 # buffer métrico, depois volta
        poly = poly.to_crs(6933)
        poly["geometry"] = poly.geometry.buffer(buffer_km * 1000.0)
        poly = poly.to_crs(4674)
    pts = gpd.GeoDataFrame(
        inventario_plu.copy(),
        geometry=gpd.points_from_xy(inventario_plu[col_lon],
                                    inventario_plu[col_lat]),
        crs=4674)
    dentro = gpd.sjoin(pts, poly[["geometry"]], predicate="within", how="inner")
    return dentro.drop(columns=[c for c in dentro.columns
                                if c.startswith("index_")])


def exutorio_do_posto(idx_ou_bacia, cod_posto: str) -> dict:
    """Coordenadas e nome do exutório (o próprio posto FLU), para o pacote."""
    df = idx_ou_bacia
    cod = str(cod_posto).strip()
    row = df[df["cod_posto"].astype(str).str.strip() == cod]
    if not len(row):
        return {}
    r = row.iloc[0]
    return {"cod": cod, "nome": str(r.get("nome_posto", "")),
            "lon": float(r["lon_posto"]), "lat": float(r["lat_posto"]),
            "area_km2": float(r.get("area_usada_km2", float("nan")))}
