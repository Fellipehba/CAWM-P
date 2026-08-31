# -*- coding: utf-8 -*-
"""
simplificar_bhae — parquet nacional de bacias FLU → GeoParquet leve p/ Preparador
==================================================================================

**Adaptado ao NOVO schema** `bacias_flu_ana.parquet` (gerador CAWM-D, 2ª geração):
colunas `cod_posto, nome, status, exutorio, n_trechos, area_incremental_km2,
nuareamont_km2, area_geometrica_km2, area_oficial_km2, desvio_oficial,
desvio_fechamento, geometry` — CRS EPSG:4674.

Diferenças em relação ao schema antigo, tratadas aqui:
  * `nome_posto`→`nome`; `area_usada_km2`→`area_oficial_km2`.
  * `rio_bhae` é obtido exclusivamente por `exutorio`→`cotrecho` no
    GeoPackage oficial BHAE, campo `noriocomp`; nunca pelo nome do posto.
  * **Sem lon/lat no parquet** → obtidos por junção com o inventário
    (`inventario_ana_estacoes.csv`) via `cod_posto == cod`. OBRIGATÓRIO: o
    Preparador precisa da coordenada do posto para o exutório e o mapa.
  * **Qualidade auditada na origem**: usa a coluna `status` (`ok`,
    `snap_fraco`, `sem_trecho_proximo`, `fechamento_fora`, `transfronteirica`,
    …) em vez da heurística de área da versão anterior. O aviso na interface
    passa a ser preciso.

Saída (nomes de coluna que o `bhae_online.py` espera):
  bhae_bacias.parquet  — cod_posto, nome_posto, lon_posto, lat_posto,
                         area_usada_km2, area_geom_km2, status,
                         geometria_confiavel, geometry (simplificada)
  bhae_indice.parquet  — idem, sem geometria (busca rápida)

Rodar:
    python simplificar_bhae.py "D:\\...\\bacias_flu_ana.parquet" \
        --inventario "caminho\\inventario_ana_estacoes.csv" \
        --tol 0.003 --compression zstd
"""
from __future__ import annotations

import sys
import argparse
from pathlib import Path


# status considerados de geometria CONFIÁVEL (utilizável no Preparador)
STATUS_CONFIAVEL = {"ok"}
# status que existem mas NÃO são geometria confiável para seleção de postos
STATUS_RUIM = {"snap_fraco", "sem_trecho_proximo", "fechamento_fora",
               "transfronteirica", "exutorio_sem_area"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("entrada", help="caminho do bacias_flu_ana.parquet")
    ap.add_argument("--inventario", default=None,
                    help="inventario_ana_estacoes.csv (para lon/lat). Se omitido, "
                         "procura dados/inventario_ana_estacoes.csv ao lado da saída.")
    ap.add_argument("--tol", type=float, default=0.003,
                    help="tolerância de simplificação em graus "
                         "(0.003~300m ≈ 19 MB, cabe no upload web do GitHub)")
    ap.add_argument("--compression", default="zstd",
                    help="codec do parquet: zstd (menor) ou snappy (rápido).")
    ap.add_argument("--saida", default="bhae_bacias.parquet")
    ap.add_argument("--rios-gpkg", default=None,
                    help="GeoPackage oficial geoft_bhae_trecho_drenagem; "
                         "adiciona rio_bhae por exutorio->cotrecho/noriocomp.")
    args = ap.parse_args()

    import geopandas as gpd
    import pandas as pd

    ent = Path(args.entrada)
    if not ent.exists():
        sys.exit(f"não encontrei {ent}")

    print(f"lendo {ent} ({ent.stat().st_size/1024/1024:.0f} MB)…")
    g = gpd.read_parquet(str(ent))
    print(f"  {len(g):,} bacias | CRS EPSG:{g.crs.to_epsg()}".replace(",", "."))

    # --- normaliza chave e valida schema esperado -------------------------
    if "cod_posto" not in g.columns:
        sys.exit(f"schema inesperado: não há 'cod_posto'. Colunas: {list(g.columns)}")
    g["cod_posto"] = g["cod_posto"].astype(str).str.strip()

    col_nome = "nome" if "nome" in g.columns else ("nome_posto" if "nome_posto" in g.columns else None)
    col_area = ("area_oficial_km2" if "area_oficial_km2" in g.columns
                else ("area_usada_km2" if "area_usada_km2" in g.columns else None))
    if col_area is None:
        sys.exit(f"não achei coluna de área oficial. Colunas: {list(g.columns)}")

    # --- junta lon/lat do inventário --------------------------------------
    inv_path = Path(args.inventario) if args.inventario else \
        Path(args.saida).with_name("dados") / "inventario_ana_estacoes.csv"
    if not inv_path.exists():
        sys.exit("Inventário não encontrado (necessário para lon/lat). "
                 f"Tentei: {inv_path}\nPasse --inventario CAMINHO.")
    print(f"juntando lon/lat do inventário: {inv_path}")
    # leitura tolerante de encoding/delimitador
    for enc in ("utf-8-sig", "latin-1"):
        try:
            inv = pd.read_csv(inv_path, sep=None, engine="python", dtype=str, encoding=enc)
            break
        except Exception:
            continue
    inv.columns = [c.lower().strip() for c in inv.columns]
    inv["cod"] = inv["cod"].astype(str).str.strip()
    inv["lon"] = pd.to_numeric(inv["lon"], errors="coerce")
    inv["lat"] = pd.to_numeric(inv["lat"], errors="coerce")
    inv = inv[["cod", "lon", "lat"]].drop_duplicates("cod")

    g = g.merge(inv, left_on="cod_posto", right_on="cod", how="left")
    n_sem_coord = int(g["lon"].isna().sum())
    if n_sem_coord:
        print(f"  aviso: {n_sem_coord} postos sem lon/lat no inventário "
              "(o mapa/exutório desses ficará indisponível).")

    # --- monta o quadro de saída com os NOMES que o bhae_online espera ----
    out = gpd.GeoDataFrame({
        "cod_posto": g["cod_posto"],
        "nome_posto": g[col_nome].astype(str) if col_nome else g["cod_posto"],
        "lon_posto": g["lon"],
        "lat_posto": g["lat"],
        "area_usada_km2": pd.to_numeric(g[col_area], errors="coerce"),
        "status": g["status"].astype(str) if "status" in g.columns else "ok",
    }, geometry=g.geometry, crs=g.crs)

    if args.rios_gpkg:
        from river_provenance import read_official_river_map
        rios = read_official_river_map(args.rios_gpkg, g["exutorio"])
        chave = pd.to_numeric(g["exutorio"], errors="coerce").astype("Int64")
        por_exutorio = rios.set_index("exutorio")["rio_bhae"]
        out["rio_bhae"] = chave.map(por_exutorio).astype("string")
        print(f"  rios oficiais BHAE: {out['rio_bhae'].notna().sum()} de {len(out)}")
    else:
        out["rio_bhae"] = pd.Series(pd.NA, index=out.index, dtype="string")

    # --- simplifica preservando topologia ---------------------------------
    print(f"simplificando a {args.tol}°…")
    out["geometry"] = out.geometry.simplify(args.tol, preserve_topology=True)
    out = out[~out.geometry.is_empty & out.geometry.notna()].copy()

    # --- qualidade: PREFERE o status auditado na origem -------------------
    # área do polígono (informativa) + flag de confiabilidade a partir do status.
    print("derivando qualidade a partir do status auditado…")
    out["area_geom_km2"] = (out.to_crs(6933).area / 1e6).round(1)
    st = out["status"].str.lower().str.strip()
    out["geometria_confiavel"] = st.isin(STATUS_CONFIAVEL)
    n_ruins = int((~out["geometria_confiavel"]).sum())
    print("  distribuição de status:")
    for s, c in st.value_counts().items():
        print(f"    {s:22s} {c}")
    print(f"  {n_ruins} de {len(out)} não confiáveis (status ≠ ok) — "
          "marcadas, não removidas.")

    # --- grava geometria leve + índice ------------------------------------
    saida = Path(args.saida)
    out.to_parquet(saida, row_group_size=256, compression=args.compression)
    print(f"  → {saida}  ({saida.stat().st_size/1024/1024:.1f} MB) · "
          f"{100*(1 - saida.stat().st_size/ent.stat().st_size):.1f}% menor")

    idx = out.drop(columns=[out.geometry.name])
    idx_path = saida.with_name("bhae_indice.parquet")
    idx.to_parquet(idx_path)
    print(f"  → {idx_path}  ({idx_path.stat().st_size/1024:.0f} KB)")

    # lista de revisão (não confiáveis) para acompanhamento
    if n_ruins:
        rev = saida.with_name("bhae_para_revisao.csv")
        (idx.loc[~idx["geometria_confiavel"],
                 ["cod_posto", "nome_posto", "status",
                  "area_usada_km2", "area_geom_km2"]]
         .sort_values("area_usada_km2", ascending=False)
         .to_csv(rev, index=False, encoding="utf-8"))
        print(f"  → {rev}  (postos com status ≠ ok, para revisão)")

    print("\nPronto. Coloque bhae_bacias.parquet e bhae_indice.parquet em "
          "cawm_preparador/dados/ e reinicie o app.")


if __name__ == "__main__":
    main()
