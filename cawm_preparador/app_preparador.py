# -*- coding: utf-8 -*-
"""
app_preparador — App 1 "Preparador de dados" do CAWM Simplex
=============================================================

Fluxo visual em passos: exutório → delineamento (BHAE) → seleção de postos
pluviométricos → aquisição/upload de séries → chuva média (IDW) → ETP mensal
→ pacote da bacia para o App 2 (Simulador).

Esta versão incorpora:
  • ETP mensal embarcada em dados/etp_brasil.npz;
  • inventário ANA default em dados/inventario_ana_estacoes.csv;
  • separação rígida entre estações FLU (exutório/vazão) e PLU (chuva/IDW);
  • busca por código/nome;
  • clique em estação no mapa com filtro espacial;
  • download do limite consolidado da bacia;
  • download opcional da drenagem a montante.

Execução:
    streamlit run app_preparador.py
"""
from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st
import geopandas as gpd

import bhae_delineamento as bd
import bhae_online as bo

# Bacia pronta (BHAE simplificada). Caminho ancorado no arquivo do app (não no
# diretório de execução), igual ao inventário — funciona local e no Streamlit
# Cloud. Pode ser trocado por uma URL (R2) no futuro.
BHAE_BACIAS = str(Path(__file__).with_name("dados") / "bhae_bacias.parquet")
BHAE_INDICE = str(Path(__file__).with_name("dados") / "bhae_indice.parquet")


@st.cache_data(show_spinner=False)
def _bhae_indice():
    return bo.carregar_indice(BHAE_INDICE)


@st.cache_resource(show_spinner=False)
def _bhae_bacias_disponivel():
    from pathlib import Path
    return Path(BHAE_BACIAS).exists() or str(BHAE_BACIAS).startswith("http")
import selecao_postos as sp
import chuva_media_idw as idw
import aquisicao_ana as ana
import inventario as invmod
import etp_base

st.set_page_config(page_title="CAWM Simplex — Preparador", layout="wide",
                   page_icon="🗺️")

DADOS_DIR = Path(__file__).with_name("dados")
ETP_DEFAULT = DADOS_DIR / "etp_brasil.npz"

MESES = ["jan", "fev", "mar", "abr", "mai", "jun",
         "jul", "ago", "set", "out", "nov", "dez"]
DIAS_MES = np.array([31, 28.2425, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], float)


# ---- caches leves -----------------------------------------------------------
@st.cache_data(show_spinner=False)
def _carregar_inventario_default() -> pd.DataFrame:
    return invmod.carregar_inventario()


@st.cache_resource(show_spinner=False)
def _carregar_grade_etp() -> etp_base.GradeETP:
    return etp_base.GradeETP.carregar(str(ETP_DEFAULT))


# ---- estado da sessão -------------------------------------------------------
ss = st.session_state
ss.setdefault("bacia", None)
ss.setdefault("postos_sel", None)
ss.setdefault("trechos", None)
ss.setdefault("areas", None)
ss.setdefault("series", {})
ss.setdefault("chuva_media", None)
ss.setdefault("etp_mensal", None)
ss.setdefault("aq", ana.AquisicaoANA(api=None))
ss.setdefault("lon", -35.3056)
ss.setdefault("lat", -8.6375)
ss.setdefault("exutorio_estacao", None)
ss.setdefault("map_click_pending", None)

st.title("🗺️ CAWM Simplex — Preparador de dados da bacia")
st.caption("Delineamento BHAE · inventário ANA default · PLU/FLU separados · "
           "chuva média IDW · ETP mensal embarcada.")

try:
    INV = _carregar_inventario_default()
    INV_PLU, INV_FLU = invmod.separar_tipos(INV)
except Exception as e:
    INV = pd.DataFrame()
    INV_PLU = pd.DataFrame()
    INV_FLU = pd.DataFrame()
    st.error(f"Falha ao carregar o inventário ANA default: {e}")


# ---- utilitários ------------------------------------------------------------
def _load_geo(upload):
    """Lê camada enviada (gpkg/shp/zip) num GeoDataFrame."""
    if upload is None:
        return None
    suffix = Path(upload.name).suffix.lower()
    tmp = Path("/tmp") / upload.name
    tmp.write_bytes(upload.getbuffer())
    if suffix == ".zip":
        return gpd.read_file(f"zip://{tmp}")
    return gpd.read_file(tmp)


def _geojson_bytes(gdf: gpd.GeoDataFrame) -> bytes:
    """GeoJSON UTF-8 em EPSG:4674, pronto para download."""
    if gdf is None or len(gdf) == 0:
        return b""
    out = gdf.to_crs(4674).copy()
    return out.to_json(drop_id=True).encode("utf-8")


def _metadata_dict() -> dict:
    b = ss.bacia
    meta = {
        "area_km2": float(b.area_km2),
        "area_topo_km2": float(b.area_topo_km2),
        "area_poly_km2": float(b.area_poly_km2),
        "cotrecho_exutorio": int(b.cotrecho_exutorio),
        "n_trechos": int(b.n_trechos),
        "rio": str(b.rio),
        "snap_dist_m": float(b.snap_dist_m),
        "exutorio_lon": float(ss.lon),
        "exutorio_lat": float(ss.lat),
        "exutorio_estacao": ss.exutorio_estacao,
        "fonte_inventario": str(invmod.INVENTARIO_DEFAULT.name),
        "fonte_etp": None,
    }
    if ss.etp_mensal is not None:
        meta["fonte_etp"] = str(ss.etp_mensal.attrs.get("fonte", ""))
    return meta


def _drenagem_montante() -> Optional[gpd.GeoDataFrame]:
    """Trechos a montante filtrados para download opcional."""
    if ss.bacia is None or ss.trechos is None:
        return None
    tr = ss.trechos.copy()
    if "cotrecho" not in tr.columns:
        return None
    cot = pd.to_numeric(tr["cotrecho"], errors="coerce").astype("Int64")
    out = tr[cot.isin(list(ss.bacia.trechos_montante))].copy()
    return out.to_crs(4674)


def _calcular_etp_bacia():
    """Amostra a grade ETP no polígono da bacia e salva em session_state."""
    if ss.bacia is None:
        return None
    if not ETP_DEFAULT.exists():
        st.warning("Arquivo dados/etp_brasil.npz não encontrado. A ETP não será calculada.")
        return None
    grade = _carregar_grade_etp()
    gs = ss.bacia.polygon.to_crs(4674).geometry
    geom = gs.union_all() if hasattr(gs, "union_all") else gs.unary_union
    etp_mm_dia = etp_base.amostrar_bacia(grade, geom=geom)
    df = pd.DataFrame({
        "mes": MESES,
        "etp_mm_dia": etp_mm_dia,
        "dias_mes_medio": DIAS_MES,
        "etp_mm_mes": etp_mm_dia * DIAS_MES,
    })
    df.attrs["fonte"] = grade.fonte
    ss.etp_mensal = df
    return df


def _aplicar_estacao_exutorio(row: pd.Series):
    """Preenche lon/lat a partir de uma estação fluviométrica."""
    ss.lon = float(row["lon"])
    ss.lat = float(row["lat"])
    ss.exutorio_estacao = {
        "cod": str(row.get("cod", "")),
        "nome": str(row.get("nome", "")),
        "tipo": str(row.get("tipo", "")),
        "lon": float(row["lon"]),
        "lat": float(row["lat"]),
    }
    ss.bacia = None
    ss.postos_sel = None
    ss.etp_mensal = None


def _map_marker_tooltip(row: pd.Series) -> str:
    # Formato lido por _processar_clique_mapa.
    nome = str(row.get("nome", "")).replace("|", " ")
    return f"ESTACAO|{row.get('tipo','')}|{row.get('cod','')}|{nome}|{row.get('lat')}|{row.get('lon')}"


def _processar_clique_mapa(map_data):
    """Se o usuário clicar em marcador FLU no mapa, atualiza coordenadas."""
    if not map_data:
        return
    tooltip = map_data.get("last_object_clicked_tooltip")
    if not tooltip or not str(tooltip).startswith("ESTACAO|"):
        return
    parts = str(tooltip).split("|", 5)
    if len(parts) != 6:
        return
    _, tipo, cod, nome, lat_s, lon_s = parts
    # Só FLU preenche exutório. PLU é mostrado para seleção/diagnóstico, mas não
    # deve substituir o ponto de exutório da bacia.
    if not str(tipo).lower().startswith("fluvio"):
        st.toast("Clique registrado em estação PLU. Para o exutório, use estação FLU.")
        return
    try:
        row = pd.Series({"cod": cod, "nome": nome, "tipo": tipo,
                         "lat": float(lat_s), "lon": float(lon_s)})
        if ss.map_click_pending != cod:
            ss.map_click_pending = cod
            _aplicar_estacao_exutorio(row)
            st.rerun()
    except Exception:
        return


def _render_map(bacia, postos_sel, lon, lat, estacoes_mapa=None):
    """Mapa: bacia, exutório, estações filtradas e postos PLU selecionados."""
    try:
        import folium
        from streamlit_folium import st_folium
    except ImportError:
        st.info("Instale folium e streamlit-folium para o mapa interativo.")
        if bacia is not None:
            st.write(f"Bacia em torno de ({lat:.4f}, {lon:.4f}), "
                     f"{bacia.area_km2:,.0f} km².")
        return

    if bacia is not None:
        poly = bacia.polygon.to_crs(4674)
        cen = poly.geometry.centroid.iloc[0]
        zoom = 10
    else:
        poly = None
        cen = type("P", (), {"x": lon, "y": lat})()
        zoom = 9

    m = folium.Map(location=[cen.y, cen.x], zoom_start=zoom,
                   tiles="CartoDB positron")

    if poly is not None:
        folium.GeoJson(poly.to_json(), name="Bacia",
                       style_function=lambda _: {"fillColor": "#4a90d9",
                                                 "color": "#1a4f72",
                                                 "weight": 2,
                                                 "fillOpacity": 0.25}
                       ).add_to(m)

    folium.Marker([lat, lon], tooltip="Exutório atual",
                  icon=folium.Icon(color="red", icon="tint")).add_to(m)

    # Estações candidatas filtradas espacialmente. FLU pode preencher o exutório.
    if estacoes_mapa is not None and len(estacoes_mapa):
        for _, p in estacoes_mapa.iterrows():
            tipo = str(p.get("tipo", "")).lower()
            is_flu = tipo.startswith("fluvio")
            cor = "green" if is_flu else "blue"
            raio = 5 if is_flu else 4
            label = f"{p.get('cod')} · {p.get('nome')} · {p.get('tipo')}"
            folium.CircleMarker(
                [float(p["lat"]), float(p["lon"])], radius=raio,
                color=cor, fill=True, fill_opacity=0.75,
                tooltip=_map_marker_tooltip(p),
                popup=label,
            ).add_to(m)

    # Postos pluviométricos efetivamente selecionados para IDW.
    if postos_sel is not None and len(postos_sel.postos):
        for _, p in postos_sel.postos.iterrows():
            cor = "darkblue" if bool(p["dentro"]) else "gray"
            folium.CircleMarker(
                [float(p["lat"]), float(p["lon"])], radius=6,
                color=cor, fill=True, fill_opacity=0.85,
                tooltip=f"PLU selecionada: {p['cod']} (peso {p.get('peso_idw', 0):.3f})",
            ).add_to(m)

    map_data = st_folium(m, height=460, use_container_width=True)
    _processar_clique_mapa(map_data)


# ==========================================================================
# Barra lateral: inventário, exutório, BHAE e parâmetros
# ==========================================================================
with st.sidebar:
    st.header("Inventário ANA default")
    if len(INV):
        st.caption(f"{len(INV):,} estações carregadas · "
                   f"PLU: {len(INV_PLU):,} · FLU: {len(INV_FLU):,}")
    else:
        st.caption("Inventário default indisponível.")

    st.divider()
    st.header("Exutório selecionado")
    if ss.exutorio_estacao:
        st.caption(f"{ss.exutorio_estacao['cod']} · {ss.exutorio_estacao['nome']}")
    else:
        st.caption("Selecione a bacia no Passo 1 (busca por posto FLU).")
    buffer_km = st.slider("Buffer de postos PLU (km)", 0, 50, 10, 5)

    st.divider()
    st.header("Mapa de estações")
    tipo_mapa = st.selectbox("Estações filtradas no mapa", ["FLU", "PLU", "FLU + PLU"], index=0,
                             help="O mapa nunca renderiza o inventário nacional inteiro; mostra só a vizinhança filtrada.")
    raio_graus = st.slider("Raio aproximado do filtro espacial (graus)", 0.25, 2.50, 0.75, 0.25)
    max_pontos_mapa = st.slider("Máximo de estações no mapa", 50, 800, 300, 50)

    st.divider()
    st.header("Chuva média (IDW)")
    idw_power = st.selectbox("Potência IDW", [1.0, 2.0], index=0)
    min_cov = st.slider("Cobertura mínima/dia", 0.0, 1.0, 0.1, 0.05,
                        help="Dias abaixo deste limiar viram falha (NaN).")

    with st.expander("Opções avançadas: inventário externo", expanded=False):
        inv_file = st.file_uploader(
            "Substituir inventário default por CSV externo (cod, lon, lat, tipo)",
            type=["csv"], key="inv_externo")
        st.caption("Uso subsidiário. O fluxo principal usa o inventário ANA embarcado.")


# Estações candidatas para o mapa, sempre com filtro espacial.
if tipo_mapa == "FLU":
    tipos_mapa = ("fluviometrica",)
elif tipo_mapa == "PLU":
    tipos_mapa = ("pluviometrica",)
else:
    tipos_mapa = ("fluviometrica", "pluviometrica")

estacoes_mapa = invmod.subset_bbox(INV, lat=float(ss.lat), lon=float(ss.lon),
                                   raio_graus=float(raio_graus),
                                   tipos=tipos_mapa,
                                   max_pontos=int(max_pontos_mapa)) if len(INV) else pd.DataFrame()


# ==========================================================================
# Passo 1 — Bacia da estação (BHAE pronta)  [método principal]
# ==========================================================================
st.subheader("Passo 1 · Bacia da estação (BHAE pronta)")
try:
    idx_bhae = _bhae_indice()
except Exception as e:
    idx_bhae = None
    st.warning(f"Índice BHAE não disponível ({e}). Use o delineamento por upload "
               "mais abaixo.")

if idx_bhae is not None:
    st.caption(f"{len(idx_bhae):,} postos FLU com bacia pronta. "
               "Busque por código ou nome — sem delinear em tempo real.")
    termo_b = st.text_input("Buscar posto FLU (código ou nome)", key="busca_bhae")
    res_b = bo.buscar(idx_bhae, termo_b).head(30) if termo_b else idx_bhae.head(0)
    if len(res_b):
        labels_b = [f"{'⚠️ ' if not r.geometria_confiavel else ''}{r.cod_posto} · "
                    f"{r.nome_posto} · {r.area_usada_km2:,.0f} km² · {r.rio_bhae}"
                    .replace(",", ".")
                    for r in res_b.itertuples()]
        escolha_b = st.selectbox("Resultado", labels_b, key="sel_bhae",
                                 help="⚠️ = geometria BHAE incompleta para este posto "
                                      "(travessia de montante não terminou). Pode "
                                      "carregar, mas o contorno pode não representar "
                                      "a bacia real.")
        r0 = res_b.iloc[labels_b.index(escolha_b)]
        cod_sel = str(r0["cod_posto"])
        if st.button("Carregar bacia pronta", type="primary",
                     use_container_width=True):
            with st.spinner("Carregando bacia…"):
                try:
                    bacia = bo.bacia_pronta(BHAE_BACIAS, cod_sel)
                except Exception as e:
                    bacia = None
                    st.error(f"Falha ao ler a geometria: {e}")
                if bacia is None:
                    st.error(f"Bacia não encontrada para {cod_sel}.")
                else:
                    ss.bacia = bacia
                    ss.exutorio_estacao = {"cod": cod_sel,
                                           "nome": str(r0["nome_posto"]),
                                           "lon": float(r0["lon_posto"]),
                                           "lat": float(r0["lat_posto"])}
                    ss.lon = float(r0["lon_posto"])
                    ss.lat = float(r0["lat_posto"])
                    ss.postos_sel = None
                    ss.chuva_media = None
                    _calcular_etp_bacia()
                    st.success(f"Bacia {cod_sel} carregada: "
                               f"{bacia.area_km2:,.0f} km².".replace(",", "."))
    elif termo_b:
        st.info("Nenhum posto FLU com bacia pronta para esse termo.")

st.divider()

# ==========================================================================
# Passo 1 — Resumo da bacia + mapa
# ==========================================================================
col1, col2 = st.columns([1, 2])
with col1:
    if ss.bacia is not None:
        b = ss.bacia
        st.metric("Área a montante (oficial ANA)",
                  f"{b.area_km2:,.0f} km²".replace(",", "."))
        st.caption(f"Rio: {b.rio}")
        _msg = b.mensagem_status() if hasattr(b, "mensagem_status") else (
            None if getattr(b, "geometria_confiavel", True) else "Geometria incompleta.")
        if _msg:
            st.warning("⚠️ " + _msg + " Você pode prosseguir, mas o contorno e "
                       "a seleção de postos PLU podem não representar a bacia real.")
        st.download_button("Baixar limite da bacia (GeoJSON)",
                           _geojson_bytes(b.polygon),
                           file_name="bacia_consolidada.geojson",
                           mime="application/geo+json",
                           use_container_width=True)
        st.download_button("Baixar metadados da bacia (JSON)",
                           json.dumps(_metadata_dict(), ensure_ascii=False,
                                      indent=2).encode("utf-8"),
                           file_name="metadados_bacia.json",
                           mime="application/json",
                           use_container_width=True)
    else:
        st.info("Carregue uma bacia acima (busca por posto FLU).")
with col2:
    _render_map(ss.bacia, ss.postos_sel, float(ss.lon), float(ss.lat), estacoes_mapa)
    st.caption(f"Mapa: {len(estacoes_mapa):,} estação(ões) na vizinhança."
               .replace(",", "."))


# ==========================================================================
# Passo 2 — Seleção de postos pluviométricos
# ==========================================================================
st.subheader("Passo 2 · Seleção de postos pluviométricos")
st.caption("A seleção para chuva média usa apenas estações PLU. Estações FLU não entram no IDW de precipitação.")

if st.button("Selecionar postos PLU", use_container_width=False):
    if ss.bacia is None:
        st.error("Delineie a bacia primeiro (Passo 1).")
    else:
        if inv_file is not None:
            inv_df = ana.parse_inventario_csv(inv_file)
            inv_gdf_all = sp.inventario_to_gdf(inv_df)
            fonte_inv = "inventário externo enviado pelo usuário"
        else:
            inv_gdf_all = sp.inventario_to_gdf(INV)
            fonte_inv = "inventário ANA default"
        # Regra rígida: IDW de chuva só usa PLU.
        inv_gdf_plu = inv_gdf_all[inv_gdf_all["tipo"].astype(str).str.lower().str.startswith("pluvio")].copy()
        if inv_gdf_plu.empty:
            st.error("Nenhuma estação pluviométrica válida encontrada no inventário.")
        else:
            ss.postos_sel = sp.selecionar(ss.bacia.polygon, inv_gdf_plu,
                                          buffer_km=buffer_km,
                                          idw_power=idw_power,
                                          tipos=("pluviometrica",))
            ss.postos_sel.fonte_inventario = fonte_inv

if ss.postos_sel is not None:
    s = ss.postos_sel
    c1, c2, c3 = st.columns(3)
    c1.metric("Postos PLU selecionados", len(s.postos))
    c2.metric("Dentro da bacia", s.n_dentro)
    c3.metric("No buffer", s.n_buffer)
    cols = [c for c in ["cod", "nome", "tipo", "dentro", "dist_borda_km", "peso_idw"] if c in s.postos.columns]
    st.dataframe(s.postos[cols].round(4), use_container_width=True, height=240)
    st.download_button("Baixar postos e pesos IDW (CSV)",
                       s.postos.drop(columns="geometry", errors="ignore").to_csv(index=False).encode("utf-8"),
                       file_name="postos_plu_idw.csv", mime="text/csv")


# ==========================================================================
# Passo 3 — Séries de chuva dos postos
# ==========================================================================
st.subheader("Passo 3 · Séries da ANA (chuva dos postos + vazão do exutório)")

# --- Opção A: download automático da ANA (HydroBR refatorado, sem token) ----
if ss.postos_sel is not None:
    cods_plu = [str(c) for c in ss.postos_sel.postos["cod"]]
    if st.button(f"⬇ Baixar chuva dos {len(cods_plu)} postos PLU (ANA)",
                 type="primary"):
        try:
            import ana_hydrobr as ah
        except Exception as e:
            st.error(f"Provedor indisponível ({e}). Use o upload manual abaixo.")
        else:
            prog = st.progress(0.0)
            area = st.empty()
            n_ok, n_vazio = 0, 0
            primeiro_erro, status = None, []
            total = len(cods_plu)
            try:
                for i, cod in enumerate(cods_plu, 1):
                    cod = str(cod).strip()
                    if cod.endswith(".0"):          # código que virou float
                        cod = cod[:-2]
                    try:
                        s = ah.baixar_serie_chuva(cod, only_consisted=False)
                        if s is not None and s.notna().any():
                            ss.series[cod] = s
                            n_ok += 1
                            status.append({"posto": cod, "dias": int(s.notna().sum()),
                                           "motivo": "ok"})
                        else:
                            n_vazio += 1
                            status.append({"posto": cod, "dias": 0,
                                           "motivo": "vazio na ANA"})
                    except Exception as e:
                        n_vazio += 1
                        status.append({"posto": cod, "dias": "-",
                                       "motivo": f"{type(e).__name__}: {e}"})
                        if primeiro_erro is None:
                            primeiro_erro = f"{type(e).__name__}: {e}"
                    # atualização da barra NUNCA pode quebrar o loop
                    try:
                        prog.progress(i / total)
                        area.write(f"posto {cod} ({i}/{total})")
                    except Exception:
                        pass
            except Exception as e:
                st.exception(e)        # traceback completo se algo escapar do loop
            prog.empty(); area.empty()
            ss.download_status = status
            st.success(f"Processados {len(status)}/{total} · {n_ok} com dados, "
                       f"{n_vazio} sem dado.")
            if primeiro_erro:
                st.error(f"1º erro ocorrido: {primeiro_erro}")
            if n_ok == 0 and not primeiro_erro:
                st.warning("Nenhum posto retornou dados. Veja a coluna 'motivo' "
                           "na tabela abaixo.")
    if ss.get("download_status"):
        st.dataframe(pd.DataFrame(ss.download_status), use_container_width=True,
                     height=200)
else:
    st.info("Selecione os postos PLU no Passo 2 para habilitar o download automático.")

# --- Vazão do exutório (mesma estação FLU do Passo 1, sem redigitar) -------
st.markdown("**Vazão observada do exutório**")
cod_flu = (ss.exutorio_estacao or {}).get("cod", "").strip() if ss.get("exutorio_estacao") else ""
if cod_flu:
    st.caption(f"Estação do exutório (Passo 1): {cod_flu} · "
               f"{(ss.exutorio_estacao or {}).get('nome', '')}")
else:
    st.info("Escolha o exutório por estação FLU no Passo 1 para baixar a vazão automaticamente.")
if st.button("⬇ Baixar vazão do exutório (ANA)", disabled=not cod_flu):
    if not cod_flu:
        st.error("Defina o exutório por estação no Passo 1.")
    else:
        try:
            import ana_hydrobr as ah
            with st.spinner(f"Baixando vazão da estação {cod_flu}…"):
                sv = ah.baixar_vazao(cod_flu.strip(), only_consisted=False)
            if sv.notna().any():
                ss.vazao = sv
                st.success(f"Vazão baixada: {int(sv.notna().sum())} dias válidos, "
                           f"{sv.index.min().date()}–{sv.index.max().date()}, "
                           f"média {float(sv.mean()):.1f} m³/s.")
            else:
                st.warning("A estação não retornou vazão na ANA. Tente outro código ou upload manual.")
        except Exception as e:
            st.error(f"Falha ao baixar vazão: {e}")
if ss.get("vazao") is not None:
    st.caption(f"Vazão do exutório carregada ({int(ss.vazao.notna().sum())} dias).")


# ==========================================================================
# Passo 4 — Chuva média
# ==========================================================================
st.subheader("Passo 4 · Chuva média da bacia")
if st.button("Calcular chuva média (IDW)", type="primary"):
    if ss.postos_sel is None or not ss.series:
        st.error("Selecione postos (Passo 2) e carregue séries (Passo 3).")
    else:
        def _norm(c):
            c = str(c).strip()
            return c[:-2] if c.endswith(".0") else c
        postos = ss.postos_sel.postos.copy()
        postos["_cod"] = postos["cod"].map(_norm)
        peso_por_cod = postos.set_index("_cod")["peso_idw"].to_dict()
        # casa as séries baixadas (chaves já normalizadas) com os postos PLU
        codes = [k for k in ss.series.keys() if _norm(k) in peso_por_cod]
        if not codes:                       # fallback: usa o que baixou
            codes = list(ss.series.keys())
        if not codes:
            st.error("Nenhuma série carregada. Baixe no Passo 3 ou suba manual.")
        else:
            mat = pd.DataFrame({c: ss.series[c] for c in codes})
            weights = [peso_por_cod.get(_norm(c), 1.0) for c in codes]
            res = idw.basin_mean_rainfall(mat, weights=weights,
                                          min_coverage=min_cov)
            ss.chuva_media = res
            st.success(f"{res.rainfall.notna().sum()} dias válidos · "
                       f"{len(codes)} postos no IDW · "
                       f"cobertura mediana {res.coverage.median():.2f}")

if ss.chuva_media is not None:
    res = ss.chuva_media
    st.line_chart(res.rainfall.dropna().rename("Chuva média (mm/d)"))
    out = res.rainfall.dropna().rename("p").to_frame()
    out.index.name = "data"
    st.download_button("Baixar chuva média (CSV)",
                       out.to_csv().encode("utf-8"),
                       file_name="chuva_media_bacia.csv", mime="text/csv")


# ==========================================================================
# Passo 5 — ETP mensal da bacia
# ==========================================================================
st.subheader("Passo 5 · ETP mensal da bacia")
if ss.bacia is None:
    st.info("Delineie a bacia para amostrar a ETP embarcada.")
elif ss.etp_mensal is None:
    if st.button("Calcular ETP mensal pela grade embarcada"):
        df_etp = _calcular_etp_bacia()
        if df_etp is not None:
            st.rerun()
else:
    df_etp = ss.etp_mensal
    fonte = str(df_etp.attrs.get("fonte", ""))
    st.caption(f"Fonte: {fonte}")
    st.dataframe(df_etp.round({"etp_mm_dia": 3, "etp_mm_mes": 1}),
                 use_container_width=True, hide_index=True)
    etp_anual = float(df_etp["etp_mm_mes"].sum())
    st.metric("ETP média anual da bacia", f"{etp_anual:,.0f} mm/ano")
    st.download_button("Baixar ETP mensal da bacia (CSV)",
                       df_etp.to_csv(index=False).encode("utf-8"),
                       file_name="etp_mensal_bacia.csv", mime="text/csv")


# ==========================================================================
# Passo 6 — Pacote consolidado mínimo
# ==========================================================================
st.subheader("Passo 6 · Pacote consolidado")
st.caption("Downloads individuais ficam disponíveis nos passos acima. O pacote mínimo combina limite, metadados, ETP e, quando calculada, chuva média.")

if ss.bacia is not None:
    import zipfile
    pacote = io.BytesIO()
    with zipfile.ZipFile(pacote, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("bacia_consolidada.geojson", _geojson_bytes(ss.bacia.polygon))
        z.writestr("metadados_bacia.json", json.dumps(_metadata_dict(), ensure_ascii=False, indent=2))
        if ss.etp_mensal is not None:
            z.writestr("etp_mensal_bacia.csv", ss.etp_mensal.to_csv(index=False))
        if ss.chuva_media is not None:
            out = ss.chuva_media.rainfall.dropna().rename("p").to_frame()
            out.index.name = "data"
            z.writestr("chuva_media_bacia.csv", out.to_csv())
        if ss.get("vazao") is not None:
            qv = ss.vazao.dropna().rename("q_m3s").to_frame()
            qv.index.name = "data"
            z.writestr("vazao_exutorio.csv", qv.to_csv())
        if ss.postos_sel is not None:
            z.writestr("postos_plu_idw.csv", ss.postos_sel.postos.drop(columns="geometry", errors="ignore").to_csv(index=False))
    st.download_button("Baixar pacote mínimo da bacia (.zip)",
                       pacote.getvalue(),
                       file_name="cawm_pacote_bacia.zip",
                       mime="application/zip",
                       use_container_width=True)
else:
    st.info("Delineie a bacia para gerar o pacote consolidado.")
