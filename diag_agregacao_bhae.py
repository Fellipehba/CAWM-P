# -*- coding: utf-8 -*-
"""
diag_agregacao_bhae — classificar as falhas de agregação no parquet-fonte
=========================================================================

CONTEXTO: 143 dos 4.722 postos têm polígono ~ottobacia local (mediana 37 km²)
em vez da bacia agregada — incluindo bacias 100% brasileiras (Marabá 703 mil
km² → 9,7 km²; Paulo Afonso 599 mil → 14 km²). Logo é FALHA DE AGREGAÇÃO na
geração, não fronteira. Padrão aparente: estações de UHE e de calha principal.

Este script roda no PARQUET-FONTE de 2 GB (que tem as colunas de proveniência
n_areas_agregadas / n_cotrechos_montante) e classifica cada falha:

  * NAVEGACAO  — n_areas_agregadas <= 3: a topologia de montante não foi
                 percorrida (snap caiu em cotrecho sem montante ligado, ex.
                 barramento/fronteira). Corrigir o ponto de snap / navegação.
  * DISSOLVE   — n_areas_agregadas alto mas polígono minúsculo: as áreas foram
                 encontradas mas a união/dissolve descartou quase tudo.
  * INDEFINIDO — colunas ausentes ou inconsistentes.

Exporta `bhae_falhas_agregacao.csv` (para a frente de geometria regenerar
esses postos) e imprime o resumo.

Rodar:
    "C:\\ProgramData\\miniconda3\\python.exe" diag_agregacao_bhae.py "CAMINHO\\bacias_postos_flu_bhae.parquet"
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import geopandas as gpd


def main():
    if len(sys.argv) < 2:
        sys.exit("uso: diag_agregacao_bhae.py CAMINHO\\bacias_postos_flu_bhae.parquet")
    arq = Path(sys.argv[1])
    if not arq.exists():
        sys.exit(f"não encontrei {arq}")

    print(f"lendo {arq} …")
    g = gpd.read_parquet(str(arq))
    g["cod_posto"] = g["cod_posto"].astype(str).str.strip()

    # área real do polígono vs oficial
    area_poly = g.to_crs(6933).area / 1e6
    area_attr = g["area_usada_km2"].astype(float)
    ratio = (area_poly / area_attr).where(area_attr > 0)

    falhas = g[ratio < 0.1].copy()
    falhas["area_poly_km2"] = area_poly[ratio < 0.1].round(1)
    falhas["ratio"] = ratio[ratio < 0.1].round(4)
    print(f"falhas de agregação (polígono <10% da área oficial): {len(falhas)}")

    # classificação pela proveniência
    def classificar(r):
        n_ag = r.get("n_areas_agregadas")
        if pd.isna(n_ag):
            return "INDEFINIDO"
        if float(n_ag) <= 3:
            return "NAVEGACAO"       # topologia de montante não percorrida
        return "DISSOLVE"            # achou áreas mas a união descartou

    if "n_areas_agregadas" in falhas.columns:
        falhas["classe"] = falhas.apply(classificar, axis=1)
    else:
        falhas["classe"] = "INDEFINIDO (coluna ausente)"

    print("\nresumo por classe:")
    print(falhas["classe"].value_counts().to_string())

    # é estação de UHE/barramento? (heurística pelo nome)
    nome = falhas["nome_posto"].astype(str).str.upper()
    falhas["uhe_ou_barramento"] = nome.str.contains("UHE|BARRAMENTO|JUSANTE|MONTANTE")
    print(f"\ncom 'UHE/barramento/jusante/montante' no nome: "
          f"{int(falhas['uhe_ou_barramento'].sum())} de {len(falhas)}")

    cols = [c for c in ["cod_posto", "nome_posto", "rio_bhae", "area_usada_km2",
                        "area_poly_km2", "ratio", "n_areas_agregadas",
                        "n_cotrechos_montante", "cotrecho_exutorio",
                        "dist_snap_m", "classe", "uhe_ou_barramento"]
            if c in falhas.columns]
    out = Path("bhae_falhas_agregacao.csv")
    falhas[cols].sort_values("area_usada_km2", ascending=False) \
        .to_csv(out, index=False, encoding="utf-8")
    print(f"\n>>> {out} salvo ({len(falhas)} postos) — leve à frente de geometria "
          "para regenerar esses postos (e me envie para eu acompanhar).")

    print("\nmaiores 10 falhas:")
    for _, r in falhas.sort_values("area_usada_km2", ascending=False).head(10).iterrows():
        n_ag = r.get("n_areas_agregadas", "?")
        print(f"  {r['cod_posto']} · {str(r['nome_posto'])[:30]:30s} "
              f"oficial {r['area_usada_km2']:>12,.0f} | poly {r['area_poly_km2']:>9,.1f} "
              f"| n_agregadas {n_ag} | {r['classe']}")


if __name__ == "__main__":
    main()
