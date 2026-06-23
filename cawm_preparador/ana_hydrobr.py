# -*- coding: utf-8 -*-
"""
ana_hydrobr — provedor de aquisição ANA via HydroBR refatorado (vendorizado)
============================================================================

Decisão fechada (reteste GO): o HydroBR REFATORADO recupera a série COMPLETA
(vazão 39480000: 13.454 dias, 37 anos; chuva 835140: 13.452 dias). O parser
deduplica datas (consistido preferido) e baixa o período inteiro. O PyPI 0.1.1
era o que truncava/quebrava.

Este módulo é uma camada FINA sobre o parser vendorizado (`hydrobr_get_data.py`):
  * usa os nomes NOVOS: prec(), flow(), list_prec(), list_flow();
  * `only_consisted=False` = bruto+consistido deduplicado (consistido preferido)
    — máxima cobertura, alinhado à preferência por nível 2;
  * estação sem dados (XML vazio) é PULADA e registrada, nunca derruba o lote.

Caminho durável: quando o token do HidroWebService chegar, migrar para o
cliente oficial `data_access.ANA` (REST/JSON). Este provedor é o que destrava
o fluxo AGORA, sem token.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Iterable

import pandas as pd

# importa o parser vendorizado ao lado deste arquivo
_GD = Path(__file__).with_name("hydrobr_get_data.py")
_spec = importlib.util.spec_from_file_location("hydrobr_get_data", str(_GD))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
ANA = _mod.ANA


def listar_estacoes(estado: str = "", tipo: str = "prec") -> pd.DataFrame:
    """Lista estações (colunas Name, Code, Latitude, Longitude, StartDate...).
    Obs.: a listagem vem de um inventário estático (ANAF/GitHub), não da ANA ao
    vivo — para seleção, prefira o nosso `inventario.py`."""
    fn = ANA.list_prec if tipo == "prec" else ANA.list_flow
    return fn(state=estado, source="ANAF")


def _baixar(codigos, tipo, only_consisted):
    fn = ANA.prec if tipo == "prec" else ANA.flow
    cods = [str(c) for c in dict.fromkeys(codigos)]      # dedup, preserva ordem
    try:
        df = fn(cods, only_consisted=only_consisted)
    except ValueError as e:
        if "No objects to concatenate" in str(e):        # todas vazias
            return pd.DataFrame(), {c: "vazia" for c in cods}
        raise
    df.index = pd.to_datetime(df.index)
    df.columns = [str(c).lstrip("0") or "0" for c in df.columns]  # normaliza códigos
    pedidos = {c.lstrip("0") or "0" for c in cods}
    retornou = set(df.columns)
    log = {c: ("ok" if (c.lstrip("0") or "0") in retornou else "vazia") for c in cods}
    return df, log


def baixar_chuva(codigos: Iterable[str], only_consisted: bool = False):
    """DataFrame diário de chuva [mm/dia], 1 coluna por estação que retornou.
    Retorna (df, log: cod→'ok'/'vazia')."""
    return _baixar(codigos, "prec", only_consisted)


def _variantes_codigo(cod: str):
    """Tenta o código como veio e também preenchido a 8 dígitos (zero à esq.),
    caso o inventário tenha perdido zeros à esquerda na extração do .mdb."""
    c = str(cod).strip()
    if c.endswith(".0"):                 # código que veio como float (ex.: 835140.0)
        c = c[:-2]
    vs = [c]
    if c.isdigit() and len(c) < 8:
        vs.append(c.zfill(8))
    return list(dict.fromkeys(vs))


def baixar_vazao(codigo: str, only_consisted: bool = False) -> pd.Series:
    """Série diária de vazão [m³/s] da estação-exutório."""
    for v in _variantes_codigo(codigo):
        df, _ = _baixar([v], "flow", only_consisted)
        if not df.empty:
            return df.iloc[:, 0]
    return pd.Series(dtype=float, name=str(codigo))


def baixar_serie_chuva(codigo: str, only_consisted: bool = False):
    """Série diária de chuva [mm/dia] de UMA estação (ou None se vazia)."""
    for v in _variantes_codigo(codigo):
        df, _ = _baixar([v], "prec", only_consisted)
        if not df.empty:
            return df.iloc[:, 0]
    return None


if __name__ == "__main__":   # autoteste offline (XML sintético, sem rede)
    from unittest import mock

    def _mes(consist, base, mes="01"):
        dias = "".join(f"<Chuva{i:02d}>{base+i}</Chuva{i:02d}>" for i in range(1, 32))
        return (f"<SerieHistorica><EstacaoCodigo>835140</EstacaoCodigo>"
                f"<NivelConsistencia>{consist}</NivelConsistencia>"
                f"<DataHora>2001-{mes}-01T00:00:00</DataHora>{dias}</SerieHistorica>")
    xml_140 = ("<root>" + _mes(1, 100) + _mes(2, 200) + "</root>").encode()
    xml_vazio = b"<root></root>"

    def fake_get(url, params=None, timeout=None, **k):
        cod = (params or {}).get("codEstacao", "")
        class R:
            content = xml_140 if str(cod) == "835140" else xml_vazio
        return R()

    with mock.patch.object(_mod.requests, "get", side_effect=fake_get):
        df, log = baixar_chuva(["835140", "835197", "835179"], only_consisted=False)
    print("colunas retornadas:", list(df.columns), "| log:", log)
    print("2001-01-05 =", float(df.loc["2001-01-05"].iloc[0]), "(205 = consistido)")
    print("sem crash, estações vazias puladas:", "OK")
