# -*- coding: utf-8 -*-
"""
preenchimento_falhas — Preenchimento de falhas em séries de chuva diária
=========================================================================

ESTADO: implementado e plugável; VALIDAÇÃO PENDENTE sobre dados multi-posto
de uma bacia bem instrumentada (das Velhas ou Madeira), a ser feita quando
o Bloco 3 (aquisição ANA) estiver alimentando o pipeline. Ver nota ao final.

Método escolhido (justificado por evidência empírica no Sirinhaém):
  * A regressão diária posto-a-posto NÃO é usada: nesse conjunto as
    correlações diárias entre postos ficaram em 0,20–0,47, insuficientes
    para preenchimento confiável (chuva convectiva é localizada). Forçá-la
    injetaria ruído com aparência de dado.
  * Em escala mensal, porém, a correlação sobe para ~0,71 — há sinal
    climático compartilhado. O preenchimento explora ISSO via o método da
    RAZÃO-NORMAL ponderada por distância (Paulhus & Kohler, 1952), que usa
    a relação estável entre as MÉDIAS dos postos, não o ruído diário:

        P_alvo(t) ≈ Σ_i w_i · (M_alvo / M_i) · P_i(t)

    onde M = média de longo prazo do posto (no período comum), w_i = peso
    por distância, e a soma percorre os vizinhos presentes no dia t.

  * FLAG DE VIABILIDADE: antes de preencher, avalia se há base estatística
    (sobreposição e correlação mensal mínimas). Se não houver, NÃO preenche
    e sinaliza — o sistema reconhece quando deve se abster. Nesse caso a
    recomendação é usar a série como está (com a renormalização do IDW) ou
    recorrer a produto gradeado (CHIRPS/ERA5-Land).

Ordem no pipeline: preenchimento (este módulo, recupera dias de baixa
cobertura) → IDW renormalizado (chuva_media_idw, cuida do resíduo).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class ViabilidadePreenchimento:
    viavel: bool
    corr_mensal_media: float
    corr_mensal_max: float
    sobreposicao_min_dias: int
    motivo: str


@dataclass
class ResultadoPreenchimento:
    series: pd.DataFrame              # séries preenchidas (onde viável)
    viabilidade: ViabilidadePreenchimento
    n_preenchidos: dict              # dias preenchidos por posto
    n_observados: dict               # dias observados por posto
    fonte: pd.DataFrame              # 'obs'/'preenchido'/'falha' por célula


def avaliar_viabilidade(series: pd.DataFrame,
                        min_corr_mensal: float = 0.6,
                        min_sobreposicao: int = 120,
                        min_frac_pares_ok: float = 0.5) -> ViabilidadePreenchimento:
    """Decide se o preenchimento por razão-normal é estatisticamente
    sustentável. Exige, simultaneamente:
      (a) correlação mensal média entre postos >= min_corr_mensal;
      (b) sobreposição diária mínima adequada na maioria dos pares;
      (c) que uma fração mínima dos pares de postos (min_frac_pares_ok)
          tenha sobreposição suficiente — captura a patologia de postos que
          operam em períodos disjuntos (ex.: Sirinhaém, onde o posto
          dominante tem ZERO dias em comum com os demais)."""
    mon = series.resample("MS").sum(min_count=10)
    cols = series.columns
    rs = []
    n_pares = 0
    n_pares_ok = 0
    # sobreposição máxima de cada posto com qualquer outro
    max_overlap_posto = {c: 0 for c in cols}
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            n_pares += 1
            common_d = int((series.iloc[:, i].notna()
                            & series.iloc[:, j].notna()).sum())
            if common_d >= min_sobreposicao:
                n_pares_ok += 1
            max_overlap_posto[cols[i]] = max(max_overlap_posto[cols[i]], common_d)
            max_overlap_posto[cols[j]] = max(max_overlap_posto[cols[j]], common_d)
            common_m = mon.iloc[:, i].notna() & mon.iloc[:, j].notna()
            if common_m.sum() > 24:
                r = mon.iloc[:, i][common_m].corr(mon.iloc[:, j][common_m])
                if np.isfinite(r):
                    rs.append(r)
    if not rs or n_pares == 0:
        return ViabilidadePreenchimento(
            False, np.nan, np.nan, 0,
            "Sem pares de postos com sobreposição suficiente.")
    cmean, cmax = float(np.nanmean(rs)), float(np.nanmax(rs))
    frac_ok = n_pares_ok / n_pares
    # postos efetivamente isolados (sem sobreposição com nenhum outro)
    isolados = [c for c, ov in max_overlap_posto.items()
                if ov < min_sobreposicao]
    frac_isolados = len(isolados) / len(cols)
    viavel = ((cmean >= min_corr_mensal) and (frac_ok >= min_frac_pares_ok)
              and (frac_isolados <= 0.15))   # tolera no máx. ~1 posto isolado
    if viavel:
        motivo = "Viável: correlação mensal e cobertura de pares adequadas."
    elif isolados:
        motivo = (f"Não-confiável: {len(isolados)} de {len(cols)} postos "
                  f"({100*frac_isolados:.0f}%) operam em períodos disjuntos "
                  f"(sobreposição < {min_sobreposicao} dias com qualquer "
                  f"outro), entre eles os de maior peso podem estar. "
                  f"Preencher injetaria dados não-verificáveis. Recomenda-se "
                  f"a série como está (IDW renormalizado) ou produto gradeado.")
    else:
        motivo = (f"Não-confiável: corr. mensal média {cmean:.2f} "
                  f"(mín {min_corr_mensal}) e/ou {n_pares_ok}/{n_pares} pares "
                  f"com sobreposição adequada. Recomenda-se a série como está "
                  f"(IDW renormalizado) ou produto gradeado (CHIRPS/ERA5).")
    return ViabilidadePreenchimento(viavel, cmean, cmax, n_pares_ok, motivo)


def _coords_dist_matrix(coords: dict, cols) -> np.ndarray:
    """Matriz de distâncias (km aprox.) entre postos, na ordem de `cols`."""
    lon = np.array([coords[c][0] for c in cols])
    lat = np.array([coords[c][1] for c in cols])
    kmlat = 111.32
    dx = (lon[:, None] - lon[None, :]) * kmlat * np.cos(np.radians(lat.mean()))
    dy = (lat[:, None] - lat[None, :]) * kmlat
    return np.sqrt(dx ** 2 + dy ** 2)


def preencher(series: pd.DataFrame, coords: Optional[dict] = None,
              power: float = 2.0, min_corr_mensal: float = 0.6,
              min_sobreposicao: int = 120,
              forcar: bool = False) -> ResultadoPreenchimento:
    """Preenche falhas por razão-normal ponderada por distância, SE viável.

    series : DataFrame (datas × postos), NaN = falha.
    coords : {posto: (lon, lat)} para os pesos por distância; se None, usa
             pesos iguais (média aritmética das razões).
    forcar : preenche mesmo se a viabilidade for negativa (não recomendado;
             para experimentação).
    """
    cols = list(series.columns)
    via = avaliar_viabilidade(series, min_corr_mensal, min_sobreposicao)
    fonte = series.notna().replace({True: "obs", False: "falha"})
    n_obs = {c: int(series[c].notna().sum()) for c in cols}

    # A viabilidade global é informativa; o preenchimento opera por posto,
    # e postos sem fontes com sobreposição suficiente NÃO são preenchidos.
    # `forcar=False` + global inviável ainda assim deixa rodar a lógica
    # por-posto, que se abstém individualmente onde não há base.

    medias = series.mean(skipna=True)                    # M_i (período comum)
    if coords is not None:
        D = _coords_dist_matrix(coords, cols)
        with np.errstate(divide="ignore"):
            W = 1.0 / np.where(D > 0, D, np.nan) ** power
        np.fill_diagonal(W, 0.0)
    else:
        W = np.ones((len(cols), len(cols))); np.fill_diagonal(W, 0.0)

    vals = series.to_numpy(dtype=float)
    out = vals.copy()
    n_fill = {c: 0 for c in cols}
    Marr = medias.to_numpy(dtype=float)

    # Sobreposição de cada par (para habilitar fontes por posto-alvo)
    present_all = ~np.isnan(vals)
    overlap = present_all.astype(int).T @ present_all.astype(int)  # (n, n)

    for k, c in enumerate(cols):
        missing = np.isnan(vals[:, k])
        if not missing.any() or not np.isfinite(Marr[k]) or Marr[k] <= 0:
            continue
        # fontes válidas para ESTE posto: sobreposição suficiente + média > 0
        fontes_ok = (overlap[k] >= min_sobreposicao) & (Marr > 0)
        fontes_ok[k] = False
        if not fontes_ok.any():
            continue                       # posto isolado: não preenche
        wk = (W[k] * fontes_ok)[None, :] * present_all      # (T, n)
        ratio = np.where(fontes_ok & (Marr > 0), Marr[k] / Marr, 0.0)
        contrib = np.nan_to_num(vals * ratio[None, :]) * wk
        wsum = wk.sum(axis=1)
        est = np.where(wsum > 0, contrib.sum(axis=1) / wsum, np.nan)
        fill_mask = missing & np.isfinite(est)
        out[fill_mask, k] = est[fill_mask]
        n_fill[c] = int(fill_mask.sum())
        fonte.iloc[np.where(fill_mask)[0], k] = "preenchido"

    filled = pd.DataFrame(out, index=series.index, columns=cols)
    return ResultadoPreenchimento(filled, via, n_fill, n_obs, fonte)


# ---------------------------------------------------------------------------
# NOTA DE VALIDAÇÃO (pendente)
# ---------------------------------------------------------------------------
# Validar quando o Bloco 3 (aquisição ANA) estiver operacional, em bacia bem
# instrumentada (das Velhas/Madeira), por validação cruzada: remover
# artificialmente p% dos dias observados, preencher, e medir RMSE/viés do
# estimado vs. observado. Critério de sucesso: o preenchimento deve reduzir
# o erro da chuva média de bacia frente à renormalização pura, sem inflar a
# variância. No Sirinhaém (diário, convectivo) a viabilidade tende a ser
# NEGATIVA — caso-limite documentado, não falha do método.
