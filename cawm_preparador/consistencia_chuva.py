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
    """T1: total do mês lançado como diário no último (ou primeiro) dia.
    VETORIZADO (transforms por grupo mensal, sem loop Python por mês) —
    essencial para bacias com centenas de postos (ex. Óbidos: 915)."""
    v = s.dropna()
    if v.empty:
        return pd.DataFrame(columns=["data", "valor", "teste", "motivo"])
    ym = v.index.to_period("M")
    g = v.groupby(ym)
    soma = g.transform("sum")
    vmax = g.transform("max")
    e_max = v.eq(vmax)
    # último dia do mês = o dia seguinte é dia 1 (aritmética pura, sem períodos)
    ultimo = (v.index + pd.Timedelta(days=1)).day == 1
    primeiro = v.index.day == 1
    # concentração = vmax/soma, calculada só onde soma>0 (evita 0/0 em meses
    # inteiramente secos; where() sozinho não basta, pois a divisão ocorre antes)
    conc = pd.Series(np.divide(vmax.values, soma.values,
                               out=np.zeros(len(vmax), dtype="float64"),
                               where=soma.values > 0),
                     index=v.index)
    cand = (e_max & (vmax >= min_valor_mensal) & (soma >= min_valor_mensal)
            & (conc >= frac_mes) & (ultimo | primeiro))
    if not bool(cand.any()):
        return pd.DataFrame(columns=["data", "valor", "teste", "motivo"])
    sel = v[cand.values]
    pos = np.where(sel.index.day == 1, "primeiro dia do mês", "último dia do mês")
    fr = (sel.values / soma[cand.values].values) * 100
    tot = soma[cand.values].values
    return pd.DataFrame({
        "data": sel.index, "valor": sel.values.astype(float),
        "teste": "total_mensal",
        "motivo": [f"{v_:.1f} mm no {p_} = {f_:.0f}% do total do mês ({t_:.1f} mm)"
                   for v_, p_, f_, t_ in zip(sel.values, pos, fr, tot)],
    })


def _flag_limite_fisico(s: pd.Series, limite_fisico: float = 250.0) -> pd.DataFrame:
    """T2: acima do plausível físico diário."""
    sus = s[s > limite_fisico].dropna()
    return pd.DataFrame([{"data": d, "valor": float(v), "teste": "limite_fisico",
                          "motivo": f"{v:.1f} mm > limite {limite_fisico:.0f} mm/dia"}
                         for d, v in sus.items()])


def _flag_valor_repetido(s: pd.Series, n_repeticao: int = 5) -> pd.DataFrame:
    """T3: n+ dias consecutivos com o mesmo valor não nulo. VETORIZADO —
    o início do bloco só é computado para os (raros) blocos suspeitos."""
    v = s.dropna()
    if v.empty:
        return pd.DataFrame(columns=["data", "valor", "teste", "motivo"])
    grupo = (v != v.shift()).cumsum()
    tam = grupo.groupby(grupo).transform("size")
    mask = (tam >= n_repeticao) & (v > 0)
    if not bool(mask.any()):
        return pd.DataFrame(columns=["data", "valor", "teste", "motivo"])
    sel = v[mask]
    g_sel = grupo[mask]
    ini_por_grupo = sel.index.to_series().groupby(g_sel.values).min()
    ini = g_sel.map(ini_por_grupo)
    return pd.DataFrame({
        "data": sel.index, "valor": sel.values.astype(float),
        "teste": "valor_repetido",
        "motivo": [f"{val:.1f} mm repetido por {t} dias desde {d0.date()}"
                   for val, t, d0 in zip(sel.values, tam[mask].values,
                                         pd.to_datetime(ini.values))],
    })


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


def auditar_conjunto(series: dict, progresso=None, **kwargs):
    """Audita um dicionário {cod_posto: serie}. Retorna (relatorio, series_limpas).
    relatorio: DataFrame [cod, data, valor, teste, motivo].

    `progresso`: callback opcional f(i, n, cod) chamado a cada posto — permite
    barra de progresso na interface sem acoplar este módulo ao Streamlit."""
    rel, limpas = [], {}
    n = len(series)
    for i, (cod, s) in enumerate(series.items(), start=1):
        if progresso is not None:
            progresso(i, n, str(cod))
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
