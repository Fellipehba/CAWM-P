# -*- coding: utf-8 -*-
"""
app_preparador — CAWM-P basin data preparation app in CAWM Web
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
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st
import geopandas as gpd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from cawm_i18n import select_language

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
from station_acquisition import (AcquisitionPolicy, summarize_report,
                                 user_uploaded_report)
from combined_acquisition import acquire_rainfall_and_streamflow
import rainfall_qc as rqc

st.set_page_config(page_title="CAWM-P", layout="wide",
                   page_icon="🗺️")

tr = select_language(st, "cawm_p_language")

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
ss.setdefault("vazao", None)
ss.setdefault("chuva_media", None)
ss.setdefault("etp_mensal", None)
ss.setdefault("aq", ana.AquisicaoANA(api=None))
ss.setdefault("lon", -35.3056)
ss.setdefault("lat", -8.6375)
ss.setdefault("exutorio_estacao", None)
ss.setdefault("map_click_pending", None)
ss.setdefault("station_acquisition_report", None)
ss.setdefault("outlet_acquisition_report", None)
ss.setdefault("qc_flags", None)
ss.setdefault("qc_decisions", None)
ss.setdefault("qc_summary", None)
ss.setdefault("series_qc", None)

st.title(tr("prep.title"))
st.caption(tr("prep.caption"))

try:
    INV = _carregar_inventario_default()
    INV_PLU, INV_FLU = invmod.separar_tipos(INV)
except Exception as e:
    INV = pd.DataFrame()
    INV_PLU = pd.DataFrame()
    INV_FLU = pd.DataFrame()
    st.error(tr.choose(f"Could not load the default ANA inventory: {e}",
                       f"Falha ao carregar o inventário ANA default: {e}"))


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


def _cod_saida() -> str:
    """Código do posto exutório para prefixar arquivos de saída (só dígitos/
    letras seguras). Vazio → 'bacia' como fallback."""
    cod = ""
    if ss.get("exutorio_estacao"):
        cod = str(ss.exutorio_estacao.get("cod", "")).strip()
    cod = "".join(ch for ch in cod if ch.isalnum())
    return cod or "bacia"


def _nome_saida(base: str, ext: str) -> str:
    """Ex.: _nome_saida('chuva_media', 'csv') → '39480000_chuva_media.csv'."""
    return f"{_cod_saida()}_{base}.{ext}"


def _metadata_dict() -> dict:
    b = ss.bacia
    meta = {
        "area_km2": float(b.area_km2),
        "area_topo_km2": float(b.area_topo_km2),
        "area_poly_km2": float(b.area_poly_km2),
        "cotrecho_exutorio": int(b.cotrecho_exutorio),
        "n_trechos": int(b.n_trechos),
        "rio": str(b.rio),
        "rio_fonte": ("BHAE geoft_bhae_trecho_drenagem.noriocomp; "
                       "join exutorio->cotrecho" if b.rio != "—" else None),
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
        st.warning(tr.choose("File dados/etp_brasil.npz was not found; PET will not be calculated.",
                             "Arquivo dados/etp_brasil.npz não encontrado. A ETP não será calculada."))
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
        st.toast(tr.choose("PLU station selected. Use an FLU station for the outlet.",
                           "Clique em posto PLU. Para o exutório, use posto FLU."))
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
        st.info(tr.choose("Install folium and streamlit-folium for the interactive map.",
                          "Instale folium e streamlit-folium para o mapa interativo."))
        if bacia is not None:
            st.write(tr.choose(
                f"Basin around ({lat:.4f}, {lon:.4f}), {bacia.area_km2:,.0f} km².",
                f"Bacia em torno de ({lat:.4f}, {lon:.4f}), {bacia.area_km2:,.0f} km²."))
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
        folium.GeoJson(poly.to_json(), name=tr.choose("Basin", "Bacia"),
                       style_function=lambda _: {"fillColor": "#4a90d9",
                                                 "color": "#1a4f72",
                                                 "weight": 2,
                                                 "fillOpacity": 0.25}
                       ).add_to(m)

    folium.Marker([lat, lon], tooltip=tr.choose("Current outlet", "Exutório atual"),
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
                tooltip=tr.choose(
                    f"Selected PLU: {p['cod']} (weight {p.get('peso_idw', 0):.3f})",
                    f"PLU selecionada: {p['cod']} (peso {p.get('peso_idw', 0):.3f})"),
            ).add_to(m)

    map_data = st_folium(m, height=460, use_container_width=True)
    _processar_clique_mapa(map_data)


# ==========================================================================
# Barra lateral: inventário, exutório, BHAE e parâmetros
# ==========================================================================
with st.sidebar:
    st.header(tr("prep.inventory"))
    if len(INV):
        st.caption(tr.choose(
            f"{len(INV):,} stations loaded · PLU: {len(INV_PLU):,} · FLU: {len(INV_FLU):,}",
            f"{len(INV):,} estações carregadas · PLU: {len(INV_PLU):,} · FLU: {len(INV_FLU):,}"))
    else:
        st.caption(tr.choose("Default inventory unavailable.", "Inventário default indisponível."))

    st.divider()
    st.header(tr("prep.outlet"))
    if ss.exutorio_estacao:
        st.caption(f"{ss.exutorio_estacao['cod']} · {ss.exutorio_estacao['nome']}")
    else:
        st.caption(tr.choose("Select a basin in Step 1 (search by FLU station).",
                             "Selecione a bacia no Passo 1 (busca por posto FLU)."))
    buffer_km = st.slider(tr.choose("PLU station buffer (km)", "Buffer de postos PLU (km)"), 0, 50, 10, 5)

    st.divider()
    st.header(tr("prep.map"))
    tipo_mapa = st.selectbox(tr.choose("Stations filtered on the map", "Estações filtradas no mapa"),
                             ["FLU", "PLU", "FLU + PLU"], index=0,
                             help=tr.choose("The full national inventory is never rendered; only a filtered neighborhood is shown.",
                                            "O inventário nacional inteiro nunca é renderizado; apenas a vizinhança filtrada é mostrada."))
    raio_graus = st.slider(tr.choose("Approximate spatial-filter radius (degrees)", "Raio aproximado do filtro espacial (graus)"), 0.25, 2.50, 0.75, 0.25)
    max_pontos_mapa = st.slider(tr.choose("Maximum stations on map", "Máximo de estações no mapa"), 50, 800, 300, 50)

    st.divider()
    st.header(tr("prep.idw"))
    idw_power = st.selectbox(tr.choose("IDW power", "Potência IDW"), [1.0, 2.0], index=0)
    min_cov = st.slider(tr.choose("Minimum daily coverage", "Cobertura mínima/dia"), 0.0, 1.0, 0.1, 0.05,
                        help=tr.choose("Days below this threshold become missing (NaN).", "Dias abaixo deste limiar viram falha (NaN)."))

    with st.expander(tr.choose("Advanced: external inventory", "Opções avançadas: inventário externo"), expanded=False):
        inv_file = st.file_uploader(
            tr.choose("Replace default inventory with external CSV (cod, lon, lat, tipo)",
                      "Substituir inventário default por CSV externo (cod, lon, lat, tipo)"),
            type=["csv"], key="inv_externo")
        st.caption(tr.choose("Optional fallback; the primary flow uses the embedded ANA inventory.",
                             "Uso subsidiário; o fluxo principal usa o inventário ANA embarcado."))


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
st.subheader(tr("prep.step1"))
try:
    idx_bhae = _bhae_indice()
except Exception as e:
    idx_bhae = None
    st.warning(tr.choose(f"BHAE index unavailable ({e}). Check the prepared data files.",
                         f"Índice BHAE não disponível ({e}). Verifique os arquivos preparados."))

if idx_bhae is not None:
    st.caption(tr.choose(
        f"{len(idx_bhae):,} FLU stations have prepared basins. Search by code or name.",
        f"{len(idx_bhae):,} postos FLU com bacia pronta. Busque por código ou nome."))
    termo_b = st.text_input(tr("prep.search"), key="busca_bhae")
    res_b = bo.buscar(idx_bhae, termo_b).head(30) if termo_b else idx_bhae.head(0)
    if len(res_b):
        labels_b = []
        for r in res_b.itertuples():
            river = getattr(r, "rio_bhae", None)
            river_text = (str(river) if pd.notna(river) and str(river).strip()
                          else tr.choose("River unavailable", "Rio não disponível"))
            label = (f"{'⚠️ ' if not r.geometria_confiavel else ''}{r.cod_posto} · "
                     f"{r.nome_posto} · {river_text} · {r.area_usada_km2:,.0f} km²"
                     f"{'' if r.geometria_confiavel else ' · ' + str(r.status)}")
            labels_b.append(label.replace(",", "."))
        escolha_b = st.selectbox(tr("prep.result"), labels_b, key="sel_bhae",
                                 help=tr.choose(
                                     "⚠️ = incomplete BHAE geometry. It can be loaded, but may not represent the real basin.",
                                     "⚠️ = geometria BHAE incompleta. Pode ser carregada, mas talvez não represente a bacia real."))
        r0 = res_b.iloc[labels_b.index(escolha_b)]
        cod_sel = str(r0["cod_posto"])
        if st.button(tr("prep.load"), type="primary",
                     use_container_width=True):
            with st.spinner(tr.choose("Loading basin…", "Carregando bacia…")):
                try:
                    bacia = bo.bacia_pronta(BHAE_BACIAS, cod_sel)
                except Exception as e:
                    bacia = None
                    st.error(tr.choose(f"Could not read geometry: {e}", f"Falha ao ler a geometria: {e}"))
                if bacia is None:
                    st.error(tr.choose(f"Basin not found for {cod_sel}.", f"Bacia não encontrada para {cod_sel}."))
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
                    st.success(tr.choose(
                        f"Basin {cod_sel} loaded: {bacia.area_km2:,.0f} km².",
                        f"Bacia {cod_sel} carregada: {bacia.area_km2:,.0f} km²."))
    elif termo_b:
        st.info(tr.choose("No prepared FLU basin matches this search.",
                          "Nenhum posto FLU com bacia pronta para esse termo."))

st.divider()

# ==========================================================================
# Passo 1 — Resumo da bacia + mapa
# ==========================================================================
col1, col2 = st.columns([1, 2])
with col1:
    if ss.bacia is not None:
        b = ss.bacia
        st.metric(tr.choose("Upstream area (official ANA)", "Área a montante (oficial ANA)"),
                  f"{b.area_km2:,.0f} km²".replace(",", "."))
        if b.rio == "—":
            st.caption(tr.choose("River: unavailable (no official BHAE match)",
                                 "Rio: indisponível (sem correspondência oficial BHAE)"))
        else:
            st.caption(tr.choose(f"River: {b.rio} · source: BHAE noriocomp",
                                 f"Rio: {b.rio} · fonte: BHAE noriocomp"))
        _msg = b.mensagem_status() if hasattr(b, "mensagem_status") else (
            None if getattr(b, "geometria_confiavel", True) else "Geometria incompleta.")
        if _msg:
            if tr.language == "en":
                _msg = (f"Geometry quality warning ({getattr(b, 'status', None) or 'incomplete'}).")
            st.warning("⚠️ " + _msg + tr.choose(
                " You may continue, but the boundary and PLU selection may not represent the real basin.",
                " Você pode prosseguir, mas o contorno e a seleção PLU podem não representar a bacia real."))
        st.download_button(tr.choose("Download basin boundary (GeoJSON)", "Baixar limite da bacia (GeoJSON)"),
                           _geojson_bytes(b.polygon),
                           file_name=_nome_saida("bacia_consolidada", "geojson"),
                           mime="application/geo+json",
                           use_container_width=True)
        st.download_button(tr.choose("Download basin metadata (JSON)", "Baixar metadados da bacia (JSON)"),
                           json.dumps(_metadata_dict(), ensure_ascii=False,
                                      indent=2).encode("utf-8"),
                           file_name=_nome_saida("metadados_bacia", "json"),
                           mime="application/json",
                           use_container_width=True)
    else:
        st.info(tr.choose("Load a basin above (search by FLU station).",
                          "Carregue uma bacia acima (busca por posto FLU)."))
with col2:
    _render_map(ss.bacia, ss.postos_sel, float(ss.lon), float(ss.lat), estacoes_mapa)
    st.caption(tr.choose(f"Map: {len(estacoes_mapa):,} nearby station(s).",
                         f"Mapa: {len(estacoes_mapa):,} estação(ões) na vizinhança."))


# ==========================================================================
# Passo 2 — Seleção de postos pluviométricos
# ==========================================================================
st.subheader(tr("prep.step2"))
st.caption(tr.choose("Mean-rainfall selection uses PLU stations only; FLU stations never enter precipitation IDW.",
                     "A seleção da chuva média usa somente postos PLU; postos FLU não entram no IDW."))

if st.button(tr("prep.select"), use_container_width=False):
    if ss.bacia is None:
        st.error(tr.choose("Load the basin first (Step 1).", "Carregue a bacia primeiro (Passo 1)."))
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
            st.error(tr.choose("No valid rain gauge was found in the inventory.",
                               "Nenhuma estação pluviométrica válida encontrada no inventário."))
        else:
            ss.postos_sel = sp.selecionar(ss.bacia.polygon, inv_gdf_plu,
                                          buffer_km=buffer_km,
                                          idw_power=idw_power,
                                          tipos=("pluviometrica",))
            ss.postos_sel.fonte_inventario = fonte_inv

if ss.postos_sel is not None:
    s = ss.postos_sel
    c1, c2, c3 = st.columns(3)
    c1.metric(tr.choose("Selected PLU stations", "Postos PLU selecionados"), len(s.postos))
    c2.metric(tr.choose("Inside basin", "Dentro da bacia"), s.n_dentro)
    c3.metric(tr.choose("In buffer", "No buffer"), s.n_buffer)
    cols = [c for c in ["cod", "nome", "tipo", "dentro", "dist_borda_km", "peso_idw"] if c in s.postos.columns]
    st.dataframe(s.postos[cols].round(4), use_container_width=True, height=240)
    st.download_button(tr.choose("Download stations and IDW weights (CSV)", "Baixar postos e pesos IDW (CSV)"),
                       s.postos.drop(columns="geometry", errors="ignore").to_csv(index=False).encode("utf-8"),
                       file_name=_nome_saida("postos_plu_idw", "csv"), mime="text/csv")


# ==========================================================================
# Passo 3 — Aquisição conjunta, auditável e com retries seletivos
# ==========================================================================
st.subheader(tr("prep.step3"))
cod_flu = (ss.exutorio_estacao or {}).get("cod", "").strip() if ss.get("exutorio_estacao") else ""
cods_plu = ([str(c) for c in ss.postos_sel.postos["cod"]]
            if ss.postos_sel is not None else [])
ready = bool(cods_plu and cod_flu)
st.caption(tr.choose(
    "One primary action acquires rainfall for every selected PLU station and streamflow for the Step 1 FLU outlet. Each request is reported.",
    "Uma ação principal baixa chuva de todos os postos PLU selecionados e vazão do exutório FLU do Passo 1. Cada requisição é registrada."))

def _run_combined(retry_failed_only=False, retry_scope="all"):
    import ana_hydrobr as ah
    prog = st.progress(0.0)
    area = st.empty()
    def _progress(i, n, code):
        prog.progress(i / max(1, n))
        area.write(tr("prep.acquiring", code=code, index=i, total=n))
    result = acquire_rainfall_and_streamflow(
        cods_plu, cod_flu,
        lambda code, timeout_seconds: ah.fetch_station_precipitation(
            code, timeout_seconds=timeout_seconds, only_consisted=False),
        lambda code, timeout_seconds: ah.fetch_station_streamflow(
            code, timeout_seconds=timeout_seconds, only_consisted=False),
        policy=AcquisitionPolicy(max_attempts=3, timeout_seconds=30.0,
                                 backoff_seconds=1.0, jitter_seconds=0.25,
                                 max_workers=4),
        previous_rainfall=ss.series,
        previous_streamflow=ss.get("vazao"),
        previous_rainfall_report=ss.get("station_acquisition_report"),
        previous_outlet_report=ss.get("outlet_acquisition_report"),
        retry_failed_only=retry_failed_only, retry_scope=retry_scope,
        progress=_progress,
    )
    prog.empty(); area.empty()
    ss.series = dict(result.rainfall)
    ss.vazao = result.streamflow
    ss.station_acquisition_report = result.rainfall_report
    ss.outlet_acquisition_report = result.outlet_report
    ss.qc_flags = ss.qc_decisions = ss.qc_summary = ss.series_qc = None
    ss.chuva_media = None

st.caption(tr.choose(f"Current selection: {len(cods_plu)} PLU station(s) · outlet {cod_flu or 'not selected'}.",
                     f"Seleção atual: {len(cods_plu)} posto(s) PLU · exutório {cod_flu or 'não selecionado'}."))
if st.button(tr.choose(
        "⬇ Acquire rainfall and outlet streamflow (ANA)",
        "⬇ Baixar chuva e vazão do exutório (ANA)"),
        type="primary", disabled=not ready, use_container_width=True):
    try:
        _run_combined(False)
        st.success(tr.choose("Combined acquisition completed; review both reports below.",
                             "Aquisição conjunta concluída; revise os dois relatórios abaixo."))
    except Exception as e:
        st.exception(e)
if not ready:
    st.info(tr.choose("Load a FLU basin in Step 1 and select PLU stations in Step 2.",
                      "Carregue uma bacia FLU no Passo 1 e selecione postos PLU no Passo 2."))

failed_rain = (int(ss.station_acquisition_report["status"].eq("failed_after_retries").sum())
               if isinstance(ss.get("station_acquisition_report"), pd.DataFrame) else 0)
failed_flow = (int(ss.outlet_acquisition_report["status"].eq("failed_after_retries").sum())
               if isinstance(ss.get("outlet_acquisition_report"), pd.DataFrame) else 0)
retry_col1, retry_col2 = st.columns(2)
with retry_col1:
    if st.button(tr.choose(f"Retry failed rainfall stations ({failed_rain})",
                           f"Repetir postos de chuva com falha ({failed_rain})"),
                 disabled=(failed_rain == 0)):
        try:
            _run_combined(True, "rainfall")
            st.success(tr.choose("Rainfall retry completed; valid series were not downloaded again.",
                                 "Retry de chuva concluído; séries válidas não foram baixadas novamente."))
        except Exception as e:
            st.exception(e)
with retry_col2:
    if st.button(tr.choose(f"Retry outlet streamflow ({failed_flow})",
                           f"Repetir vazão do exutório ({failed_flow})"),
                 disabled=(failed_flow == 0)):
        try:
            _run_combined(True, "outlet")
            st.success(tr.choose("Outlet retry completed; valid rainfall was not downloaded again.",
                                 "Retry do exutório concluído; chuva válida não foi baixada novamente."))
        except Exception as e:
            st.exception(e)

if isinstance(ss.get("station_acquisition_report"), pd.DataFrame):
    rain_counts = summarize_report(ss.station_acquisition_report)
    outlet_report_for_summary = (ss.outlet_acquisition_report
                                 if isinstance(ss.get("outlet_acquisition_report"), pd.DataFrame)
                                 else pd.DataFrame(columns=["status"]))
    outlet_counts = summarize_report(outlet_report_for_summary)
    rain_success = rain_counts["success"] + rain_counts["user_uploaded"]
    outlet_success = outlet_counts["success"] + outlet_counts["user_uploaded"]
    overall_warning = rain_counts["no_data"] + rain_counts["failed_after_retries"] + outlet_counts["no_data"] + outlet_counts["failed_after_retries"]
    st.markdown(tr.choose(
        f"**Rainfall stations:** Selected {len(ss.station_acquisition_report)} · Success {rain_success} · No data {rain_counts['no_data']} · Failed {rain_counts['failed_after_retries']}  \n"
        f"**Outlet streamflow:** Success {outlet_success} · No data {outlet_counts['no_data']} · Failed {outlet_counts['failed_after_retries']}  \n"
        f"**Overall:** {'Completed with warnings' if overall_warning else 'Completed'}",
        f"**Postos de chuva:** Selecionados {len(ss.station_acquisition_report)} · Sucesso {rain_success} · Sem dados {rain_counts['no_data']} · Falha {rain_counts['failed_after_retries']}  \n"
        f"**Vazão do exutório:** Sucesso {outlet_success} · Sem dados {outlet_counts['no_data']} · Falha {outlet_counts['failed_after_retries']}  \n"
        f"**Estado geral:** {'Concluído com avisos' if overall_warning else 'Concluído'}"))

for report_key, title, filename in [
    ("station_acquisition_report", tr.choose("Rain-gauge acquisition report", "Relatório de aquisição PLU"), "station_acquisition_report.csv"),
    ("outlet_acquisition_report", tr.choose("Outlet acquisition report", "Relatório de aquisição do exutório"), "outlet_streamflow_acquisition_report.csv"),
]:
    report = ss.get(report_key)
    if isinstance(report, pd.DataFrame):
        st.markdown(f"**{title}**")
        st.caption(tr("prep.report_caption"))
        st.dataframe(report, use_container_width=True, height=220)
        st.download_button(tr.choose(f"Download {filename}", f"Baixar {filename}"),
                           report.to_csv(index=False, lineterminator="\n").encode("utf-8"),
                           file_name=filename, mime="text/csv", key=f"download_{report_key}")

with st.expander(tr.choose("Manual upload fallback", "Fallback por upload manual"), expanded=False):
    st.caption(tr.choose(
        "Use classic HidroWeb monthly CSV files. Rainfall filenames must contain the station code; raw uploaded series are preserved.",
        "Use CSVs mensais clássicos do HidroWeb. O nome de cada arquivo de chuva deve conter o código do posto; as séries brutas enviadas são preservadas."))
    rain_uploads = st.file_uploader(tr.choose("Rainfall station files", "Arquivos dos postos de chuva"),
                                    type=["csv", "txt"], accept_multiple_files=True,
                                    key="manual_rain_files")
    if st.button(tr.choose("Load manual rainfall files", "Carregar arquivos manuais de chuva"),
                 disabled=not bool(rain_uploads)):
        rows = []
        for upload in rain_uploads:
            matches = re.findall(r"(?<!\d)(\d{6,8})(?!\d)", upload.name)
            if not matches:
                st.error(tr.choose(f"Station code missing from filename: {upload.name}",
                                   f"Código do posto ausente no nome: {upload.name}"))
                continue
            code = matches[0].lstrip("0") or "0"
            series = ana.parse_serie_hidroweb(upload, ana.TIPO_CHUVA)
            ss.series[code] = series
            rows.append(user_uploaded_report(code, series, len(rows) + 1))
        if rows:
            manual = pd.concat(rows, ignore_index=True)
            prior = ss.get("station_acquisition_report")
            if isinstance(prior, pd.DataFrame):
                prior = prior[~prior.station_id.astype(str).isin(manual.station_id.astype(str))]
                manual = pd.concat([prior, manual], ignore_index=True)
            ss.station_acquisition_report = manual
            ss.qc_flags = ss.qc_decisions = ss.qc_summary = ss.series_qc = None
            ss.chuva_media = None
    flow_upload = st.file_uploader(tr.choose("Outlet streamflow file", "Arquivo de vazão do exutório"),
                                   type=["csv", "txt"], key="manual_flow_file")
    if st.button(tr.choose("Load manual outlet streamflow", "Carregar vazão manual do exutório"),
                 disabled=(flow_upload is None or not cod_flu)):
        series = ana.parse_serie_hidroweb(flow_upload, ana.TIPO_VAZAO)
        ss.vazao = series
        ss.outlet_acquisition_report = user_uploaded_report(cod_flu, series)


# ==========================================================================
# Passo 4 — QC em duas etapas e chuva média
# ==========================================================================
st.subheader(tr("prep.step4"))
st.markdown(tr.choose("**4A · Detect and review flags**", "**4A · Detectar e revisar flags**"))
preset = st.selectbox(tr.choose("Decision preset", "Preset de decisões"), list(rqc.PRESETS),
                      format_func=lambda x: {"recommended": tr.choose("Recommended", "Recomendado"),
                                              "exclude_all": tr.choose("Exclude all", "Excluir tudo"),
                                              "keep_all": tr.choose("Keep all", "Manter tudo")}[x])
if st.button(tr.choose("Detect rainfall QC flags", "Detectar flags de QC da chuva"), disabled=not bool(ss.series)):
    flags = rqc.detect_flags(ss.series)
    ss.qc_flags = flags
    ss.qc_decisions = rqc.build_decisions(flags, preset)
    ss.qc_summary = rqc.summarize_qc(flags, ss.qc_decisions)
if isinstance(ss.get("qc_decisions"), pd.DataFrame):
    if st.button(tr.choose("Apply preset to decision table", "Aplicar preset à tabela de decisões")):
        ss.qc_decisions = rqc.build_decisions(ss.qc_flags, preset)
    st.caption(tr.choose(
        "Recommended policy: monthly-total flags exclude the entire station-month; physical-limit and repeated-value flags remain recorded. Edit any selected_action before application.",
        "Política recomendada: flags de total mensal excluem todo o posto-mês; limite físico e valor repetido permanecem registrados. Edite selected_action antes da aplicação."))
    st.info(tr.choose(
        "Quality-control flags identify suspect observations; not every flag is an error.",
        "As marcações de controle de qualidade identificam observações suspeitas; nem todo alerta é um erro."))
    ss.qc_decisions = st.data_editor(
        ss.qc_decisions, disabled=["station_id", "date", "value", "test", "reason", "recommended_action", "scope"],
        column_config={"selected_action": st.column_config.SelectboxColumn(options=list(rqc.ACTIONS), required=True)},
        use_container_width=True, hide_index=True, key="qc_decision_editor")
    ss.qc_summary = rqc.summarize_qc(ss.qc_flags, ss.qc_decisions)
    c1, c2, c3 = st.columns(3)
    c1.download_button(tr.choose("Download QC flags", "Baixar flags de QC"), ss.qc_flags.to_csv(index=False).encode(), "rainfall_qc_flags.csv", "text/csv")
    c2.download_button(tr.choose("Download QC decisions", "Baixar decisões de QC"), ss.qc_decisions.to_csv(index=False).encode(), "rainfall_qc_decisions.csv", "text/csv")
    c3.download_button(tr.choose("Download QC summary", "Baixar resumo de QC"),
                       ss.qc_summary.to_json(orient="records", indent=2).encode(),
                       "rainfall_qc_summary.json", "application/json")

st.markdown(tr.choose("**4B · Apply reviewed decisions and calculate IDW**", "**4B · Aplicar decisões revisadas e calcular IDW**"))
if st.button(tr.choose("Apply QC decisions and calculate mean rainfall (IDW)",
                       "Aplicar decisões de QC e calcular chuva média (IDW)"),
             type="primary", disabled=not isinstance(ss.get("qc_decisions"), pd.DataFrame)):
    def _norm(c):
        c = str(c).strip()
        return c[:-2] if c.endswith(".0") else c
    postos = ss.postos_sel.postos.copy()
    postos["_cod"] = postos["cod"].map(_norm)
    peso_por_cod = postos.set_index("_cod")["peso_idw"].to_dict()
    codes = [k for k in ss.series if _norm(k) in peso_por_cod]
    ss.series_qc = rqc.apply_decisions({c: ss.series[c] for c in codes}, ss.qc_decisions)
    mat = pd.DataFrame(ss.series_qc)
    weights = [peso_por_cod.get(_norm(c), 1.0) for c in codes]
    res = idw.basin_mean_rainfall(mat, weights=weights, min_coverage=min_cov)
    ss.chuva_media = res
    st.success(tr.choose(
        f"{res.rainfall.notna().sum()} valid days · {len(codes)} IDW stations · raw series preserved.",
        f"{res.rainfall.notna().sum()} dias válidos · {len(codes)} postos IDW · séries brutas preservadas."))

if ss.chuva_media is not None:
    res = ss.chuva_media
    st.line_chart(res.rainfall.dropna().rename(tr.choose("Mean rainfall (mm/d)", "Chuva média (mm/d)")))
    out = res.rainfall.dropna().rename("p").to_frame()
    out.index.name = "data"
    st.download_button(tr.choose("Download mean rainfall (CSV)", "Baixar chuva média (CSV)"),
                       out.to_csv().encode("utf-8"),
                       file_name=_nome_saida("chuva_media_bacia", "csv"), mime="text/csv")


# ==========================================================================
# Passo 5 — ETP mensal da bacia
# ==========================================================================
st.subheader(tr("prep.step5"))
if ss.bacia is None:
    st.info(tr.choose("Load a basin to sample embedded PET.", "Carregue a bacia para amostrar a ETP embarcada."))
elif ss.etp_mensal is None:
    if st.button(tr.choose("Calculate monthly PET from embedded grid", "Calcular ETP mensal pela grade embarcada")):
        df_etp = _calcular_etp_bacia()
        if df_etp is not None:
            st.rerun()
else:
    df_etp = ss.etp_mensal
    fonte = str(df_etp.attrs.get("fonte", ""))
    st.caption(tr.choose(f"Source: {fonte}", f"Fonte: {fonte}"))
    st.dataframe(df_etp.round({"etp_mm_dia": 3, "etp_mm_mes": 1}),
                 use_container_width=True, hide_index=True)
    etp_anual = float(df_etp["etp_mm_mes"].sum())
    st.metric(tr.choose("Mean annual basin PET", "ETP média anual da bacia"),
              tr.choose(f"{etp_anual:,.0f} mm/year", f"{etp_anual:,.0f} mm/ano"))
    st.download_button(tr.choose("Download monthly basin PET (CSV)", "Baixar ETP mensal da bacia (CSV)"),
                       df_etp.to_csv(index=False).encode("utf-8"),
                       file_name=_nome_saida("etp_mensal_bacia", "csv"), mime="text/csv")


# ==========================================================================
# Passo 6 — Pacote consolidado mínimo
# ==========================================================================
st.subheader(tr("prep.step6"))
st.caption(tr.choose("Individual downloads remain available above. The package combines boundary, metadata, PET and mean rainfall when calculated.",
                     "Downloads individuais ficam disponíveis acima. O pacote combina limite, metadados, ETP e chuva média quando calculada."))

if ss.bacia is not None:
    import zipfile
    pacote = io.BytesIO()
    with zipfile.ZipFile(pacote, "w", compression=zipfile.ZIP_DEFLATED) as z:
        _dir = f"{_cod_saida()}_pacote_bacia/"
        z.writestr(_dir + _nome_saida("bacia_consolidada", "geojson"), _geojson_bytes(ss.bacia.polygon))
        z.writestr(_dir + _nome_saida("metadados_bacia", "json"), json.dumps(_metadata_dict(), ensure_ascii=False, indent=2))
        if ss.etp_mensal is not None:
            z.writestr(_dir + _nome_saida("etp_mensal_bacia", "csv"), ss.etp_mensal.to_csv(index=False))
        if ss.chuva_media is not None:
            out = ss.chuva_media.rainfall.dropna().rename("p").to_frame()
            out.index.name = "data"
            z.writestr(_dir + _nome_saida("chuva_media_bacia", "csv"), out.to_csv())
        if ss.get("vazao") is not None:
            qv = ss.vazao.dropna().rename("q_m3s").to_frame()
            qv.index.name = "data"
            z.writestr(_dir + _nome_saida("vazao_exutorio", "csv"), qv.to_csv())
        if ss.postos_sel is not None:
            z.writestr(_dir + _nome_saida("postos_plu_idw", "csv"), ss.postos_sel.postos.drop(columns="geometry", errors="ignore").to_csv(index=False))
        if ss.get("station_acquisition_report") is not None:
            z.writestr(_dir + "station_acquisition_report.csv",
                       ss.station_acquisition_report.to_csv(index=False, lineterminator="\n"))
        if ss.get("outlet_acquisition_report") is not None:
            z.writestr(_dir + "outlet_streamflow_acquisition_report.csv",
                       ss.outlet_acquisition_report.to_csv(index=False, lineterminator="\n"))
        if isinstance(ss.get("qc_flags"), pd.DataFrame):
            z.writestr(_dir + "rainfall_qc_flags.csv", ss.qc_flags.to_csv(index=False, lineterminator="\n"))
            z.writestr(_dir + "rainfall_qc_decisions.csv", ss.qc_decisions.to_csv(index=False, lineterminator="\n"))
            z.writestr(_dir + "rainfall_qc_summary.json", ss.qc_summary.to_json(orient="records", indent=2))
    st.download_button(tr("prep.package"),
                       pacote.getvalue(),
                       file_name=_nome_saida("cawm_pacote_bacia", "zip"),
                       mime="application/zip",
                       use_container_width=True)
else:
    st.info(tr("prep.no_basin"))
