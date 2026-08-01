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
def _serie_float(s: pd.Series) -> pd.Series:
    """Normaliza a série para float64 puro, índice datetime, sem NaN.

    As séries da ANA chegam com dtypes variados (object com vírgula decimal,
    Int64/Float64 nullable do pandas, ou já float). Normalizar UMA VEZ aqui
    evita erro de casting em cada operação vetorizada dos testes."""
    v = s.copy()
    if not isinstance(v.index, pd.DatetimeIndex):
        v.index = pd.to_datetime(v.index, errors="coerce")
    if v.dtype == object:
        # texto brasileiro: "1.234,5" → "1234.5"; "130,0" → "130.0".
        # Sem isso, pd.to_numeric transformaria em NaN silenciosamente.
        t = v.astype(str).str.strip()
        tem_virgula = t.str.contains(",", na=False)
        t = t.mask(tem_virgula,
                   t.str.replace(".", "", regex=False).str.replace(",", ".",
                                                                  regex=False))
        v = t.replace({"None": None, "nan": None, "": None})
    v = pd.to_numeric(v, errors="coerce").astype("float64")
    return v[v.notna() & v.index.notna()]


def _flag_total_mensal(s: pd.Series, frac_mes: float = 0.8,
                       min_valor_mensal: float = 40.0) -> pd.DataFrame:
    """T1: total do mês lançado como diário no último (ou primeiro) dia.
    VETORIZADO (transforms por grupo mensal, sem loop Python por mês) —
    essencial para bacias com centenas de postos (ex. Óbidos: 915)."""
    v = _serie_float(s)
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
    # concentração = vmax/soma. Divide APENAS onde soma>0, por máscara booleana
    # (sem np.divide/out=, que falha com dtypes nullable Int64/Float64 da ANA).
    soma_v = soma.to_numpy(dtype="float64", na_value=np.nan)
    vmax_v = vmax.to_numpy(dtype="float64", na_value=np.nan)
    conc_v = np.zeros(len(v), dtype="float64")
    pos_ok = soma_v > 0
    conc_v[pos_ok] = vmax_v[pos_ok] / soma_v[pos_ok]
    conc = pd.Series(conc_v, index=v.index)
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
    v = _serie_float(s)
    sus = v[v > limite_fisico]
    return pd.DataFrame([{"data": d, "valor": float(v), "teste": "limite_fisico",
                          "motivo": f"{v:.1f} mm > limite {limite_fisico:.0f} mm/dia"}
                         for d, v in sus.items()])


def _flag_valor_repetido(s: pd.Series, n_repeticao: int = 5) -> pd.DataFrame:
    """T3: n+ dias consecutivos com o mesmo valor não nulo. VETORIZADO —
    o início do bloco só é computado para os (raros) blocos suspeitos."""
    v = _serie_float(s)
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


def limpar_serie(s: pd.Series, flags: pd.DataFrame,
                 invalidar=("total_mensal",),
                 modo_total_mensal: str = "mes") -> pd.Series:
    """Devolve cópia da série com os dias INVALIDADOS (→ NaN).

    Separa deliberadamente *sinalizar* de *invalidar*:

    `invalidar`: quais testes de fato removem dado. Por padrão **apenas
        `total_mensal`** — os testes `limite_fisico` e `valor_repetido` ficam
        como ALERTA (aparecem no relatório, não alteram a série), pois um
        valor extremo pode ser real e valores repetidos podem ser chuva
        legítima. Para invalidá-los, passe p.ex. ("total_mensal","limite_fisico").

    `modo_total_mensal`:
      * "mes" (padrão) — invalida o MÊS INTEIRO daquele posto. Motivo: quando o
        total do mês é lançado num dia só, os demais dias ficam com ZERO que
        NÃO é medição. Deixá-los introduz viés sistemático negativo (medido:
        −0,44 mm/dia num experimento controlado). Com o mês em NaN, o IDW usa
        os outros postos e a média fica sem viés (+0,02 mm/dia).
      * "dia" — invalida somente o dia do lançamento (alteração mínima do
        registro). Preserva mais dado bruto, mas mantém o viés acima.
    """
    limpa = s.copy()
    if flags is None or len(flags) == 0:
        return limpa
    usar = flags[flags["teste"].isin(invalidar)]
    if not len(usar):
        return limpa
    datas = pd.to_datetime(usar["data"])
    idx = limpa.index
    if not isinstance(idx, pd.DatetimeIndex):
        idx = pd.to_datetime(idx, errors="coerce")
    tm = usar["teste"] == "total_mensal"
    # dias marcados por testes que não o total_mensal → sempre pontuais
    pontuais = pd.to_datetime(usar.loc[~tm, "data"])
    if len(pontuais):
        limpa[idx.isin(pontuais)] = np.nan
    if bool(tm.any()):
        d_tm = pd.to_datetime(usar.loc[tm, "data"])
        if modo_total_mensal == "mes":
            meses = set(d_tm.dt.to_period("M"))
            limpa[pd.Index(idx.to_period("M")).isin(meses)] = np.nan
        else:
            limpa[idx.isin(d_tm)] = np.nan
    return limpa


def auditar_conjunto(series: dict, progresso=None,
                     invalidar=("total_mensal",),
                     modo_total_mensal: str = "mes", **kwargs):
    """Audita um dicionário {cod_posto: serie}. Retorna (relatorio, series_limpas).
    relatorio: DataFrame [cod, data, valor, teste, motivo].

    `progresso`: callback opcional f(i, n, cod) chamado a cada posto — permite
    barra de progresso na interface sem acoplar este módulo ao Streamlit.
    `invalidar` / `modo_total_mensal`: ver `limpar_serie` (por padrão só o
    total_mensal invalida, e invalida o mês inteiro do posto)."""
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
            limpas[cod] = limpar_serie(s, flags, invalidar=invalidar,
                                       modo_total_mensal=modo_total_mensal)
        else:
            limpas[cod] = s
    relatorio = pd.concat(rel, ignore_index=True) if rel else \
        pd.DataFrame(columns=["cod", "data", "valor", "teste", "motivo"])
    if len(relatorio):
        relatorio["acao"] = np.where(
            relatorio["teste"].isin(invalidar),
            ("mês invalidado" if modo_total_mensal == "mes" else "dia invalidado"),
            "apenas alerta")
    return relatorio, limpas
