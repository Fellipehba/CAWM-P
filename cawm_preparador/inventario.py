# -*- coding: utf-8 -*-
"""
inventario — utilitários leves para o inventário nacional de estações da ANA
============================================================================

Sustenta o novo fluxo do Preparador (inventário como default, busca por
código/nome, clique no mapa) sem custo computacional alto: o CSV nacional
(~18k linhas, ~1 MB) é carregado uma vez e todas as operações são lookups
vetorizados em memória. NUNCA renderize as ~18k estações no mapa de uma vez —
use `subset_bbox` para mostrar só a vizinhança da área de interesse.

Distingue explicitamente estações PLUVIOMÉTRICAS (chuva → IDW) de
FLUVIOMÉTRICAS (vazão → exutório). São papéis diferentes e não devem ser
misturados na ponderação da chuva.
"""
from __future__ import annotations

import unicodedata
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Caminho default do inventário embarcado no pacote.
INVENTARIO_DEFAULT = Path(__file__).with_name("dados") / "inventario_ana_estacoes.csv"

_COLS = ["cod", "nome", "lon", "lat", "tipo", "area_km2"]


def _sem_acento(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", str(s))
                   if not unicodedata.combining(c)).upper().strip()


def carregar_inventario(path: Optional[str] = None) -> pd.DataFrame:
    """Carrega o inventário nacional. Detecta separador e encoding.

    Retorna DataFrame com colunas cod(str), nome(str), lon(float), lat(float),
    tipo(str: 'pluviometrica'/'fluviometrica'), area_km2(float|NaN).
    """
    p = Path(path) if path else INVENTARIO_DEFAULT
    raw = p.read_bytes()
    for enc in ("utf-8-sig", "latin-1"):
        try:
            text = raw.decode(enc); break
        except (UnicodeDecodeError, LookupError):
            continue
    else:
        text = raw.decode("latin-1", errors="replace")
    head = next((ln for ln in text.splitlines() if ln.strip()), "")
    sep = max((";", ",", "\t", "|"), key=head.count)
    if head.count(sep) == 0:
        sep = ","
    from io import StringIO
    df = pd.read_csv(StringIO(text), sep=sep, dtype=str)
    df.columns = [c.lower().strip() for c in df.columns]
    for c in _COLS:
        if c not in df.columns:
            df[c] = np.nan
    df["cod"] = df["cod"].astype(str).str.strip()
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["area_km2"] = pd.to_numeric(df["area_km2"], errors="coerce")
    df["tipo"] = df["tipo"].astype(str).str.lower().str.strip()
    df["_nome_norm"] = df["nome"].map(_sem_acento)
    return df.dropna(subset=["lon", "lat"]).reset_index(drop=True)


def buscar_estacao(inv: pd.DataFrame, termo: str) -> pd.DataFrame:
    """Busca por código (igualdade) OU nome (contém, sem acento/caixa).

    Devolve as linhas correspondentes (use a 1ª para preencher coordenadas).
    """
    termo = str(termo).strip()
    if not termo:
        return inv.iloc[0:0]
    por_cod = inv[inv["cod"] == termo]
    if len(por_cod):
        return por_cod
    alvo = _sem_acento(termo)
    return inv[inv["_nome_norm"].str.contains(alvo, na=False)]


def subset_bbox(inv: pd.DataFrame, lat: float, lon: float,
                raio_graus: float = 0.75, tipos: Optional[tuple] = None,
                max_pontos: int = 500) -> pd.DataFrame:
    """Estações dentro de uma caixa lat/lon ao redor de (lat, lon).

    Para renderizar no mapa SEM travar: limita a `max_pontos` (as mais próximas
    do centro). `tipos` filtra por tipo (ex.: ('pluviometrica',)).
    """
    m = ((inv["lat"].sub(lat).abs() <= raio_graus) &
         (inv["lon"].sub(lon).abs() <= raio_graus))
    sub = inv[m]
    if tipos:
        sub = sub[sub["tipo"].isin(tipos)]
    if len(sub) > max_pontos:
        d2 = (sub["lat"] - lat) ** 2 + (sub["lon"] - lon) ** 2
        sub = sub.loc[d2.nsmallest(max_pontos).index]
    return sub.reset_index(drop=True)


def separar_tipos(inv: pd.DataFrame):
    """Retorna (pluviometricas, fluviometricas)."""
    plu = inv[inv["tipo"].str.startswith("pluvio", na=False)].reset_index(drop=True)
    flu = inv[inv["tipo"].str.startswith("fluvio", na=False)].reset_index(drop=True)
    return plu, flu


def estacoes_pluviometricas(inv: pd.DataFrame) -> pd.DataFrame:
    """Apenas estações de chuva — as únicas válidas para o IDW de precipitação."""
    return separar_tipos(inv)[0]
