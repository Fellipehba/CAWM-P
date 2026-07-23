# -*- coding: utf-8 -*-
"""
consistencia_chuva — auditoria de consistência de séries pluviométricas diárias
================================================================================

Motivação (caso real, Óbidos/17050001): postos antigos do HidroWeb lançavam o
TOTAL MENSAL como se fosse um dado diário — tipicamente no último dia do mês.
Na chuva média de Óbidos, 31 dos 32 dias com p>60 mm caíam exatamente no
último dia do mês (1972–1981), com magnitudes típicas de totais mensais
amazônicos (100–144 mm). Com poucos postos operando na época, o IDW era
dominado por esses valores.

Testes implementados (cada um marca, nunca apaga silenciosamente):

  T1 total_mensal   — o valor do ÚLTIMO dia do mês concentra ≥ `frac_mes` do
                      total do mês (default 80%) e supera `min_valor_mensal`
                      (default 40 mm). Chuva real extrema no fim do mês NÃO
                      dispara: os demais dias chuvosos do mês diluem a fração.
                      Também testa o DIA 1 (outra convenção de lançamento).
  T2 limite_fisico  — valor acima de `limite_fisico` mm/dia (default 250;
                      recorde diário brasileiro ~430 mm, mas >250 num posto
                      comum merece revisão).
  T3 valor_repetido — sequência de `n_repeticao`+ dias consecutivos com o
                      MESMO valor não nulo (pluviômetro travado/preenchimento).

Uso:
    from consistencia_chuva import auditar_serie, limpar_serie, auditar_conjunto

    flags = auditar_serie(serie)             # DataFrame de suspeitas
    s_ok  = limpar_serie(serie, flags)       # série com suspeitas → NaN
    rel, series_ok = auditar_conjunto({cod: serie, ...})  # em lote

Filosofia: INVALIDAR (NaN), nunca redistribuir — distribuir o total mensal
pelos dias seria inventar dado. Com o valor invalidado, o IDW usa os demais
postos do dia; se nenhum sobrar, o dia fica ausente e a cobertura mínima
(min_coverage) do IDW decide.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
def _flag_total_mensal(s: pd.Series, frac_mes: float = 0.8,
                       min_valor_mensal: float = 40.0) -> pd.DataFrame:
    """T1: total do mês lançado como diário no último (ou primeiro) dia."""
    flags = []
    df = s.dropna().to_frame("v")
    if df.empty:
        return pd.DataFrame(columns=["data", "valor", "teste", "motivo"])
    df["ym"] = df.index.to_period("M")
    for ym, gm in df.groupby("ym"):
        soma = float(gm["v"].sum())
        if soma < min_valor_mensal:
            continue
        vmax = float(gm["v"].max())
        if vmax < min_valor_mensal or (vmax / soma) < frac_mes:
            continue
        dia_max = gm["v"].idxmax()
        ultimo_dia_cal = ym.to_timestamp("M")          # último dia do calendário
        primeiro_dia_cal = ym.to_timestamp("D")        # dia 1
        if dia_max.normalize() == ultimo_dia_cal.normalize():
            pos = "último dia do mês"
        elif dia_max.normalize() == primeiro_dia_cal.normalize():
            pos = "primeiro dia do mês"
        else:
            continue  # concentração alta mas em dia qualquer: pode ser evento real isolado
        flags.append({"data": dia_max, "valor": vmax, "teste": "total_mensal",
                      "motivo": f"{vmax:.1f} mm no {pos} = "
                                f"{100*vmax/soma:.0f}% do total do mês ({soma:.1f} mm)"})
    return pd.DataFrame(flags)


def _flag_limite_fisico(s: pd.Series, limite_fisico: float = 250.0) -> pd.DataFrame:
    """T2: acima do plausível físico diário."""
    sus = s[s > limite_fisico].dropna()
    return pd.DataFrame([{"data": d, "valor": float(v), "teste": "limite_fisico",
                          "motivo": f"{v:.1f} mm > limite {limite_fisico:.0f} mm/dia"}
                         for d, v in sus.items()])


def _flag_valor_repetido(s: pd.Series, n_repeticao: int = 5) -> pd.DataFrame:
    """T3: n+ dias consecutivos com o mesmo valor não nulo."""
    flags = []
    v = s.dropna()
    if v.empty:
        return pd.DataFrame(columns=["data", "valor", "teste", "motivo"])
    grupo = (v != v.shift()).cumsum()
    for _, bloco in v.groupby(grupo):
        if len(bloco) >= n_repeticao and float(bloco.iloc[0]) > 0:
            for d, val in bloco.items():
                flags.append({"data": d, "valor": float(val), "teste": "valor_repetido",
                              "motivo": f"{val:.1f} mm repetido por {len(bloco)} dias "
                                        f"desde {bloco.index[0].date()}"})
    return pd.DataFrame(flags)


# ---------------------------------------------------------------------------
def auditar_serie(s: pd.Series, frac_mes: float = 0.8,
                  min_valor_mensal: float = 40.0,
                  limite_fisico: float = 250.0,
                  n_repeticao: int = 5) -> pd.DataFrame:
    """Roda os três testes numa série diária (index datetime, valores mm/dia).
    Retorna DataFrame [data, valor, teste, motivo] (vazio se nada suspeito)."""
    partes = [
        _flag_total_mensal(s, frac_mes, min_valor_mensal),
        _flag_limite_fisico(s, limite_fisico),
        _flag_valor_repetido(s, n_repeticao),
    ]
    flags = pd.concat([p for p in partes if len(p)], ignore_index=True) \
        if any(len(p) for p in partes) else \
        pd.DataFrame(columns=["data", "valor", "teste", "motivo"])
    return flags.drop_duplicates(subset=["data", "teste"]).sort_values("data") \
        .reset_index(drop=True)


def limpar_serie(s: pd.Series, flags: pd.DataFrame) -> pd.Series:
    """Devolve cópia da série com os dias marcados → NaN (invalidados)."""
    s2 = s.copy()
    if len(flags):
        s2.loc[s2.index.isin(pd.to_datetime(flags["data"]))] = np.nan
    return s2


def auditar_conjunto(series: dict, **kwargs):
    """Audita um dicionário {cod_posto: serie}. Retorna (relatorio, series_limpas).
    relatorio: DataFrame [cod, data, valor, teste, motivo]."""
    rel, limpas = [], {}
    for cod, s in series.items():
        flags = auditar_serie(s, **kwargs)
        if len(flags):
            f2 = flags.copy()
            f2.insert(0, "cod", str(cod))
            rel.append(f2)
            limpas[cod] = limpar_serie(s, flags)
        else:
            limpas[cod] = s
    relatorio = pd.concat(rel, ignore_index=True) if rel else \
        pd.DataFrame(columns=["cod", "data", "valor", "teste", "motivo"])
    return relatorio, limpas
