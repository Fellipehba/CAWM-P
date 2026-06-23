# -*- coding: utf-8 -*-
"""
aquisicao_ana — Bloco 3: aquisição de dados da ANA (HidroWeb)
==============================================================

ESTADO: parser implementado e testável; provedores de download
implementados mas NÃO testáveis neste ambiente (a rede aqui só acessa
PyPI/GitHub, não o servidor da ANA). Validação dos provedores é local.

Arquitetura de PROVEDORES plugáveis com degradação elegante:
  1. ApiOficialProvider   — nova API HidroWebService (requer token, validade
                            60 min; cadastro por e-mail a hidro@ana.gov.br).
                            Aposta de futuro (serviço antigo expira 30/06/2026).
  2. RestHidrowebProvider — rota REST do HidroWeb sem token (mais frágil).
  3. UploadManualProvider — arquivos baixados manualmente pelo usuário no
                            HidroWeb. NUNCA quebra; é o fallback final.

O orquestrador tenta 1 → 2 → 3, conforme disponibilidade, e sempre deixa o
upload manual acessível. Cada provedor retorna o MESMO formato canônico
(pd.Series diária, índice de datas, valores em mm/dia ou m³/s), via os
parsers abaixo — assim o resto do pipeline independe da origem.

Tipos de dado (convenção HidroWeb): 2 = chuva, 3 = vazão.
"""
from __future__ import annotations

import io
import time
from dataclasses import dataclass, field
from typing import Optional, Protocol

import numpy as np
import pandas as pd

API_BASE = "https://www.ana.gov.br/hidrowebservice"
REST_BASE = "https://www.snirh.gov.br/hidroweb/rest/api"
# Endpoints REST candidatos (validar localmente; a ANA alterna formatos):
#   séries históricas: {REST_BASE}/documento/gerar?codigosEstacoes=&tipoArquivo=
#   telemétricas:      {REST_BASE}/documento/gerarTelemetricas?...
# tipoArquivo: 1=cotas, 2=chuva, 3=vazão
TIPO_CHUVA, TIPO_VAZAO = 2, 3


# ============================================================================
# 1. PARSERS (testáveis aqui) — formato HidroWeb
# ============================================================================

def parse_inventario_csv(path_or_buf, sep: str | None = None,
                         encoding: str | None = None) -> pd.DataFrame:
    """Lê o inventário de estações da ANA (CSV do HidroWeb OU inventário nacional).

    Normaliza para colunas: cod (código), nome, lon, lat, tipo
    ('pluviometrica'/'fluviometrica'), area_km2 (se houver).
    Aceita variações de nome de coluna comuns no HidroWeb.

    Por padrão, DETECTA AUTOMATICAMENTE o separador (',' ';' tab ou '|') e o
    encoding (utf-8 ou latin-1). Exportações do HidroWeb costumam usar ';' com
    decimal vírgula; o inventário nacional usa ','. Passe `sep`/`encoding`
    explicitamente para forçar.
    """
    from io import StringIO

    # 1) Lê o texto bruto resolvendo o encoding (funciona p/ caminho ou buffer).
    if hasattr(path_or_buf, "read"):
        raw = path_or_buf.read()
        if hasattr(path_or_buf, "seek"):
            path_or_buf.seek(0)            # rebobina p/ não consumir o upload
    else:
        with open(path_or_buf, "rb") as _f:
            raw = _f.read()
    if isinstance(raw, str):
        text = raw
    else:
        for _enc in ([encoding] if encoding else ["utf-8-sig", "latin-1"]):
            try:
                text = raw.decode(_enc); break
            except (UnicodeDecodeError, LookupError):
                continue
        else:
            text = raw.decode("latin-1", errors="replace")

    # 2) Detecta o separador pela 1ª linha (o que mais ocorre vence).
    if sep is None:
        head = next((ln for ln in text.splitlines() if ln.strip()), "")
        contagem = {d: head.count(d) for d in (";", ",", "\t", "|")}
        sep = max(contagem, key=contagem.get)
        if contagem[sep] == 0:
            sep = ","

    df = pd.read_csv(StringIO(text), sep=sep, dtype=str)
    cols = {c.lower().strip(): c for c in df.columns}

    def pick(*names):
        for n in names:
            if n in cols:
                return cols[n]
        return None

    c_cod = pick("codigo", "código", "cod", "estacao_codigo", "codigoestacao")
    c_nome = pick("nome", "estacao_nome", "nomeestacao")
    c_lon = pick("longitude", "lon", "long")
    c_lat = pick("latitude", "lat")
    c_tipo = pick("tipo", "tipoestacao", "tipo_estacao")
    c_area = pick("areadrenagem", "area_km2", "areadrenagemdesignada",
                  "area")
    if c_cod is None or c_lon is None or c_lat is None:
        raise ValueError("Inventário sem colunas mínimas (código, lon, lat). "
                         f"Colunas encontradas: {list(df.columns)}")

    def num(s):
        return pd.to_numeric(s.astype(str).str.replace(",", ".", regex=False),
                             errors="coerce")

    out = pd.DataFrame({
        "cod": df[c_cod].astype(str).str.strip(),
        "nome": df[c_nome] if c_nome else "",
        "lon": num(df[c_lon]),
        "lat": num(df[c_lat])})
    if c_tipo:
        t = df[c_tipo].astype(str).str.lower()
        out["tipo"] = np.where(t.str.contains("pluv"), "pluviometrica",
                       np.where(t.str.contains("fluv"), "fluviometrica", t))
    else:
        out["tipo"] = ""
    if c_area:
        out["area_km2"] = num(df[c_area])
    return out.dropna(subset=["lon", "lat"]).reset_index(drop=True)



def parse_inventario_access(mdb_path: str, apenas_validas: bool = True
                            ) -> pd.DataFrame:
    """Lê o inventário nacional da ANA direto do arquivo Access (.mdb/.accdb).

    Requer a ferramenta `mdbtools` instalada no sistema (mdb-export). Extrai a
    tabela `Estacao` (40k+ estações) e normaliza para o mesmo formato de
    parse_inventario_csv: cod, nome, lon, lat, tipo, area_km2.

    O tipo é derivado das flags da ANA:
      TipoEstacaoPluviometro=1  -> 'pluviometrica'
      TipoEstacaoDescLiquida=1  -> 'fluviometrica' (mede vazão)
    Uma estação pode ter as duas flags; nesse caso marca 'pluvio_fluvio'.

    apenas_validas: descarta linhas marcadas como Removido/Temporario.
    """
    import subprocess, io as _io
    try:
        out = subprocess.run(["mdb-export", mdb_path, "Estacao"],
                             capture_output=True, text=True, check=True).stdout
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        raise RuntimeError(
            "Falha ao ler o Access. Instale mdbtools (ex.: 'apt install "
            f"mdbtools' ou 'conda install -c conda-forge mdbtools'). Erro: {e}")
    df = pd.read_csv(_io.StringIO(out), low_memory=False)

    if apenas_validas:
        for flag in ("Removido", "Temporario"):
            if flag in df.columns:
                df = df[df[flag].fillna(0) == 0]

    plu = df.get("TipoEstacaoPluviometro", 0).fillna(0).astype(int) == 1
    flu = df.get("TipoEstacaoDescLiquida", 0).fillna(0).astype(int) == 1
    tipo = np.where(plu & flu, "pluvio_fluvio",
            np.where(plu, "pluviometrica",
             np.where(flu, "fluviometrica", "outra")))

    out_df = pd.DataFrame({
        "cod": df["Codigo"].astype("Int64").astype(str),
        "nome": df.get("Nome", "").astype(str).str.strip(),
        "lon": pd.to_numeric(df["Longitude"], errors="coerce"),
        "lat": pd.to_numeric(df["Latitude"], errors="coerce"),
        "tipo": tipo,
        "area_km2": pd.to_numeric(df.get("AreaDrenagem"), errors="coerce")})
    return out_df.dropna(subset=["lon", "lat"]).reset_index(drop=True)


def parse_serie_hidroweb(path_or_buf, tipo: int,
                         consistencia: str = "preferir_consistido",
                         sep: str = ";", encoding: str = "latin-1"
                         ) -> pd.Series:
    """Converte um arquivo de série do HidroWeb (formato "mensal em linhas,
    dias em colunas") numa série DIÁRIA canônica.

    O formato clássico do HidroWeb traz uma linha por mês com colunas
    Chuva01..Chuva31 (ou Vazao01..Vazao31), além de NivelConsistencia
    (1=bruto, 2=consistido). Esta função desempilha para série diária.

    consistencia: 'preferir_consistido' usa nível 2 quando existe, senão 1;
                  'apenas_consistido' descarta nível 1; 'apenas_bruto' o
                  contrário.
    """
    df = pd.read_csv(path_or_buf, sep=sep, encoding=encoding, dtype=str,
                     skip_blank_lines=True)
    cols = {c.lower().strip(): c for c in df.columns}
    prefixo = "chuva" if tipo == TIPO_CHUVA else "vazao"

    c_data = None
    for k in ("data", "datames", "data_mes", "mesano"):
        if k in cols:
            c_data = cols[k]; break
    if c_data is None:
        raise ValueError(f"Coluna de data não encontrada: {list(df.columns)}")
    c_cons = cols.get("nivelconsistencia") or cols.get("consistencia")

    dia_cols = []
    for d in range(1, 32):
        for cand in (f"{prefixo}{d:02d}", f"{prefixo}{d}",
                     f"{prefixo}_{d:02d}"):
            if cand.lower() in cols:
                dia_cols.append((d, cols[cand.lower()])); break
    if not dia_cols:
        raise ValueError(f"Colunas diárias '{prefixo}NN' não encontradas.")

    def to_num(s):
        return pd.to_numeric(s.astype(str).str.replace(",", ".", regex=False),
                             errors="coerce")

    registros = {}            # data -> (valor, nivel)
    for _, row in df.iterrows():
        try:
            base = pd.to_datetime(row[c_data], dayfirst=True)
        except Exception:
            continue
        nivel = 1
        if c_cons and pd.notna(row[c_cons]):
            try:
                nivel = int(float(row[c_cons]))
            except Exception:
                nivel = 1
        if consistencia == "apenas_consistido" and nivel != 2:
            continue
        if consistencia == "apenas_bruto" and nivel != 1:
            continue
        for d, col in dia_cols:
            try:
                data = base.replace(day=d)
            except ValueError:
                continue                       # dia inexistente no mês
            val = to_num(pd.Series([row[col]])).iloc[0]
            if pd.isna(val):
                continue
            # preferir consistido: nível maior vence
            if data not in registros or nivel > registros[data][1]:
                registros[data] = (val, nivel)

    if not registros:
        return pd.Series(dtype=float, name="valor")
    s = pd.Series({d: v for d, (v, _) in registros.items()}, name="valor")
    return s.sort_index().asfreq("D")


# ============================================================================
# 2. PROVEDORES (estrutura pronta; download não testável aqui)
# ============================================================================

@dataclass
class ResultadoAquisicao:
    serie: pd.Series
    fonte: str                      # 'api_oficial' | 'rest' | 'manual'
    cod: str
    tipo: int
    ok: bool
    mensagem: str = ""


class Provider(Protocol):
    def baixar(self, cod: str, tipo: int, ini: Optional[str],
               fim: Optional[str]) -> ResultadoAquisicao: ...


@dataclass
class ApiOficialProvider:
    """Nova API HidroWebService (requer token de 60 min).

    Fluxo: autentica (identificador/senha) → recebe token → consulta série.
    Os endpoints exatos vêm do manual da ANA; deixados parametrizáveis
    porque a ANA pode ajustá-los. Validar localmente com a credencial.
    """
    identificador: str = ""
    senha: str = ""
    base: str = API_BASE
    _token: Optional[str] = field(default=None, repr=False)
    _token_ts: float = 0.0

    def _autenticar(self):
        import requests
        url = f"{self.base}/EstacoesTelemetricas/OAUth/v1"
        r = requests.get(url, headers={"Identificador": self.identificador,
                                       "Senha": self.senha}, timeout=30)
        r.raise_for_status()
        self._token = r.json().get("items", {}).get("tokenautenticacao")
        self._token_ts = time.time()
        if not self._token:
            raise RuntimeError("Token não retornado pela API.")

    def _token_valido(self) -> bool:
        return self._token is not None and (time.time() - self._token_ts) < 3300

    def baixar(self, cod, tipo, ini=None, fim=None) -> ResultadoAquisicao:
        import requests
        try:
            if not self._token_valido():
                self._autenticar()
            # rota de série histórica adotada (chuva/vazão)
            rota = ("HidroSerieCotas" if tipo == 1 else
                    "HidroSerieChuva" if tipo == TIPO_CHUVA else
                    "HidroSerieVazao")
            url = f"{self.base}/EstacoesTelemetricas/{rota}/v1"
            params = {"Código da Estação": cod}
            if ini: params["Data Inicial (yyyy-MM-dd)"] = ini
            if fim: params["Data Final (yyyy-MM-dd)"] = fim
            r = requests.get(url, params=params,
                             headers={"Authorization": f"Bearer {self._token}"},
                             timeout=60)
            r.raise_for_status()
            serie = _json_ana_para_serie(r.json(), tipo)
            return ResultadoAquisicao(serie, "api_oficial", cod, tipo,
                                      ok=not serie.empty)
        except Exception as e:
            return ResultadoAquisicao(pd.Series(dtype=float), "api_oficial",
                                      cod, tipo, ok=False, mensagem=str(e))


@dataclass
class RestHidrowebProvider:
    """Rota REST do HidroWeb sem token (gera arquivo e baixa). Mais frágil.

    ATENÇÃO: o serviço de telemetria antigo da ANA tem descontinuação
    anunciada para 30/06/2026 e já opera sobre base secundária (menor
    desempenho/defasagem). Tratar como ponte até a API oficial com token."""
    base: str = REST_BASE

    def baixar(self, cod, tipo, ini=None, fim=None) -> ResultadoAquisicao:
        import requests
        try:
            url = f"{self.base}/documento/gerar"
            params = {"codigosEstacoes": cod, "tipoArquivo": tipo}
            if ini: params["periodoInicial"] = ini
            if fim: params["periodoFinal"] = fim
            r = requests.get(url, params=params, timeout=120)
            r.raise_for_status()
            serie = parse_serie_hidroweb(io.BytesIO(r.content), tipo)
            return ResultadoAquisicao(serie, "rest", cod, tipo,
                                      ok=not serie.empty)
        except Exception as e:
            return ResultadoAquisicao(pd.Series(dtype=float), "rest", cod,
                                      tipo, ok=False, mensagem=str(e))


@dataclass
class UploadManualProvider:
    """Fallback final: arquivo já baixado pelo usuário. Nunca falha por rede."""
    arquivos: dict = field(default_factory=dict)   # cod -> path/buffer

    def baixar(self, cod, tipo, ini=None, fim=None) -> ResultadoAquisicao:
        if cod not in self.arquivos:
            return ResultadoAquisicao(pd.Series(dtype=float), "manual", cod,
                                      tipo, ok=False,
                                      mensagem="Arquivo não fornecido.")
        try:
            fonte = self.arquivos[cod]
            if hasattr(fonte, "seek"):     # buffer reutilizável: rebobina
                fonte.seek(0)
            serie = parse_serie_hidroweb(fonte, tipo)
            if ini or fim:
                serie = serie.loc[ini:fim]
            return ResultadoAquisicao(serie, "manual", cod, tipo,
                                      ok=not serie.empty)
        except Exception as e:
            return ResultadoAquisicao(pd.Series(dtype=float), "manual", cod,
                                      tipo, ok=False, mensagem=str(e))


def _json_ana_para_serie(payload: dict, tipo: int) -> pd.Series:
    """Converte o JSON da nova API ANA em série diária. Estrutura conforme
    o manual; ajustar conforme o retorno real ao validar com token."""
    items = payload.get("items") or payload.get("dados") or []
    if isinstance(items, dict):
        items = [items]
    regs = {}
    campo_val = "chuva" if tipo == TIPO_CHUVA else "vazao"
    for it in items:
        data = it.get("data") or it.get("dataHora") or it.get("Data_Hora_Dado")
        val = it.get(campo_val) or it.get("valor") or it.get("Valor")
        if data is None or val is None:
            continue
        try:
            regs[pd.to_datetime(data)] = float(str(val).replace(",", "."))
        except Exception:
            continue
    if not regs:
        return pd.Series(dtype=float)
    return pd.Series(regs).sort_index().asfreq("D")


# ============================================================================
# 3. ORQUESTRADOR — tenta provedores em ordem, com fallback manual
# ============================================================================

@dataclass
class AquisicaoANA:
    api: Optional[ApiOficialProvider] = None
    rest: Optional[RestHidrowebProvider] = field(
        default_factory=RestHidrowebProvider)
    manual: UploadManualProvider = field(default_factory=UploadManualProvider)
    tentar_rest: bool = True

    def obter(self, cod: str, tipo: int, ini: Optional[str] = None,
              fim: Optional[str] = None) -> ResultadoAquisicao:
        """Tenta API oficial → REST → manual, parando no primeiro sucesso."""
        tentativas = []
        if self.api is not None:
            tentativas.append(self.api)
        if self.tentar_rest and self.rest is not None:
            tentativas.append(self.rest)
        tentativas.append(self.manual)

        ultimo = None
        for prov in tentativas:
            res = prov.baixar(cod, tipo, ini, fim)
            ultimo = res
            if res.ok:
                return res
        return ultimo

    def obter_varias(self, codigos: list, tipo: int, ini=None, fim=None
                     ) -> dict:
        """Baixa várias estações; retorna {cod: ResultadoAquisicao}."""
        return {c: self.obter(c, tipo, ini, fim) for c in codigos}
