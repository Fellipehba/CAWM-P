"""Centralized, schema-safe internationalization for the CAWM web apps.

Only human-facing text belongs here. Machine-readable column names, parameter
names, file schemas and scientific identifiers remain unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

LANGUAGES = {"English": "en", "Português (Brasil)": "pt-BR"}
DEFAULT_LANGUAGE = "en"

_EN = {
    "language": "Language / Idioma",
    "sim.title": "CAWM-S — daily rainfall–runoff simulation and calibration",
    "sim.caption": ("CAWM hydrological model (Campus Agreste Watershed Model, UFPE) · inputs: rainfall, "
                    "streamflow, basin area and monthly or daily PET. Signature-based Tlag, "
                    "automatic warm-up, calibration/validation/test split and benchmarks."),
    "sim.input": "1 · Input data", "sim.basin": "2 · Basin", "sim.advanced": "3 · Advanced",
    "sim.rain": "Daily rainfall (.txt) — `dd/mm/yyyy<TAB>mm/day`",
    "sim.flow": "Observed streamflow (.txt) — `dd/mm/yyyy<TAB>m³/s` (gaps allowed)",
    "sim.pet": "PET (.txt) — monthly `jan..dec<TAB>mm/day` OR daily `dd/mm/yyyy<TAB>mm/day`",
    "sim.name": "Basin name", "sim.default_name": "My basin", "sim.regime": "Hydrological regime",
    "sim.zero": "Policy for Q = 0", "sim.alpha": "alpha_et — supplementary evapotranspiration",
    "sim.area": "Area [km²]", "sim.kp": "kp (PET factor)", "sim.cal_kp": "Calibrate kp (0.60–0.90)",
    "sim.tlag": "Manual Tlag [days] (0 = signature estimate)", "sim.warmup": "Warm-up [days]",
    "sim.frac_cal": "Calibration fraction", "sim.frac_val": "Validation fraction",
    "sim.quick": "Fast calibration (1 seed)", "sim.iter": "DE iterations",
    "sim.run": "▶ Run calibration", "sim.upload_prompt": "⬅ Upload the three files in the sidebar and provide the basin area.",
    "sim.parameters": "Parameters", "sim.performance": "Performance by period (after warm-up)",
    "sim.hydrograph": "Hydrograph", "sim.fdc": "Flow-duration curve", "sim.log": "Log scale",
    "sim.observed": "Observed", "sim.simulated": "Simulated", "sim.flow_axis": "Streamflow [m³/s]",
    "sim.exceedance": "Exceedance probability [%]", "sim.cal_end": "end calibration", "sim.val_end": "end validation",
    "sim.mean_hydrograph": "Mean annual hydrograph (regime diagnostic)",
    "sim.peak_lag": "Peak lag (days)", "sim.inactive": "inactive/omitted", "sim.fixed": "fixed", "sim.calibrated": "calibrated",
    "sim.download_csv": "⬇ Download simulated series (CSV)",
    "sim.download_zip": "⬇ Download complete CAWM-S results (.zip)",
    "sim.bundle_help": "Built in memory from the current run; no calibration or simulation is repeated.",
    "sim.read_error": "Could not read the input files: {error}",
    "sim.day_rain": "Rainfall days", "sim.day_flow": "Valid streamflow days",
    "sim.pet_metric": "PET", "sim.daily": "daily series", "sim.monthly": "monthly climatology",
    "sim.period": "Rainfall period: {start} → {end}",
    "sim.zero_notice": "The series contains {count} day(s) with Q ≤ 0. Policy `{policy}` is applied independently of regime.",
    "sim.running": "Running…", "sim.preparing": "Preparing data and estimating signature-based Tlag…",
    "sim.sim_period": "Simulated period: **{start} – {end}** ({days} continuous days).",
    "sim.tlag_summary": "Tlag = **{tlag} d** · warm-up = {warmup} d",
    "sim.estimated": "Estimated time: **~{low}–{high} min** ({seconds:.1f} s per generation on this machine).",
    "sim.starting": "Starting calibration…", "sim.progress": "Calibration progress: {percent:.0f}%",
    "sim.calibration_complete": "Calibration completed",
    "sim.elapsed": "Completed in {seconds:.0f} s.", "sim.done": "Calibration completed ✅", "sim.failed": "Failed",
    "sim.bound_warning": "kp converged to the lower bound; this is a kp↔KL degeneracy signature. Consider fixing kp.",
    "sim.benchmark_note": "The fair simulation benchmark is seasonal climatology: **SS_vs_climatol > 0** indicates skill beyond the mean cycle. The final test must be considered once, after decisions are frozen.",
    "sim.seed_dispersion": "Between-seed dispersion (equifinality)",
    "sim.regime_help": "Independent of alpha_et and zero policy. 'unknown' blocks simulation until an explicit choice is made.",
    "sim.zero_help": "Controls observation eligibility only; it does not select the hydrological regime.",
    "sim.alpha_help": "0.0 is the primary candidate; 1.4 is the fixed sensitivity option.",
    "sim.kp_help": "PET used = input PET × kp. Use 1.0 for final PET; ignored when kp calibration is active.",
    "sim.cal_kp_help": "Adds kp as a fifth DE parameter. A boundary solution can indicate kp↔KL degeneracy.",
    "sim.quick_help": "Disable to use three seeds for an equifinality diagnostic.",
    "sim.iter_help": "Maximum differential-evolution generations; early stopping is automatic when NSE stabilizes.",
    "sim.upload_format": "**Format:** TAB-separated text, `dd/mm/yyyy`, dot decimal. Streamflow gaps are allowed; rainfall must be continuous.",
    "prep.title": "CAWM-P — basin data preparation",
    "prep.caption": "BHAE delineation · default ANA inventory · separate PLU/FLU stations · IDW mean rainfall · embedded monthly PET.",
    "prep.inventory": "Default ANA inventory", "prep.outlet": "Selected outlet", "prep.map": "Station map",
    "prep.idw": "Mean rainfall (IDW)", "prep.step1": "Step 1 · Station basin (prepared BHAE)",
    "prep.step2": "Step 2 · Rain-gauge selection", "prep.step3": "Step 3 · ANA series (rain gauges + outlet flow)",
    "prep.step4": "Step 4 · Basin mean rainfall", "prep.step5": "Step 5 · Monthly basin PET",
    "prep.step6": "Step 6 · Consolidated package", "prep.search": "Search FLU station (code or name)",
    "prep.result": "Result", "prep.load": "Load prepared basin", "prep.select": "Select PLU stations",
    "prep.download_rain": "⬇ Acquire rainfall for {count} PLU stations (ANA)",
    "prep.acquiring": "station {code} ({index}/{total})", "prep.completed": "Completed",
    "prep.completed_warn": "Completed with warnings", "prep.acq_summary": "{done}/{total} attempted · {ok} with data · {no_data} no data · {failed} failed · {not_attempted} not attempted.",
    "prep.report_download": "Download station_acquisition_report.csv",
    "prep.report_caption": "Every requested station is reported; no-data and failures are never silently dropped.",
    "prep.no_selection": "Select PLU stations in Step 2 to enable automatic acquisition.",
    "prep.outlet_flow": "**Observed outlet streamflow**", "prep.mean_button": "Calculate mean rainfall (IDW)",
    "prep.package": "Download minimum basin package (.zip)", "prep.no_basin": "Load a basin to continue.",
}

_PT = {
    "language": "Idioma / Language",
    "sim.title": "CAWM-S — simulação e calibração chuva–vazão diária",
    "sim.caption": "Modelo hidrológico CAWM (Campus Agreste Watershed Model, UFPE) · entradas: chuva, vazão, área e ETP mensal ou diária. Tlag por assinatura, warm-up automático, divisão calibração/validação/teste e benchmarks.",
    "sim.input": "1 · Dados de entrada", "sim.basin": "2 · Bacia", "sim.advanced": "3 · Avançado",
    "sim.rain": "Chuva diária (.txt) — `dd/mm/aaaa<TAB>mm/dia`", "sim.flow": "Vazão observada (.txt) — `dd/mm/aaaa<TAB>m³/s` (lacunas permitidas)",
    "sim.pet": "ETP (.txt) — mensal `jan..dez<TAB>mm/dia` OU diária `dd/mm/aaaa<TAB>mm/dia`",
    "sim.name": "Nome da bacia", "sim.default_name": "Minha bacia", "sim.regime": "Regime hidrológico",
    "sim.zero": "Política para Q = 0", "sim.alpha": "alpha_et — evapotranspiração suplementar",
    "sim.area": "Área [km²]", "sim.kp": "kp (fator da ETP)", "sim.cal_kp": "Calibrar kp (0,60–0,90)",
    "sim.tlag": "Tlag manual [dias] (0 = estimar por assinatura)", "sim.warmup": "Warm-up [dias]",
    "sim.frac_cal": "Fração de calibração", "sim.frac_val": "Fração de validação", "sim.quick": "Calibração rápida (1 semente)",
    "sim.iter": "Iterações do DE", "sim.run": "▶ Executar calibração", "sim.upload_prompt": "⬅ Carregue os três arquivos na barra lateral e informe a área.",
    "sim.parameters": "Parâmetros", "sim.performance": "Desempenho por período (após warm-up)",
    "sim.hydrograph": "Hidrograma", "sim.fdc": "Curva de permanência", "sim.log": "Escala log",
    "sim.observed": "Observada", "sim.simulated": "Simulada", "sim.flow_axis": "Vazão [m³/s]",
    "sim.exceedance": "Permanência [%]", "sim.cal_end": "fim calibração", "sim.val_end": "fim validação",
    "sim.mean_hydrograph": "Hidrograma do ano médio (diagnóstico de regime)",
    "sim.peak_lag": "Defasagem do pico (dias)", "sim.inactive": "inativo/omitido", "sim.fixed": "fixo", "sim.calibrated": "calibrado",
    "sim.download_csv": "⬇ Baixar série simulada (CSV)", "sim.download_zip": "⬇ Baixar resultados completos do CAWM-S (.zip)",
    "sim.bundle_help": "Gerado em memória com a execução atual; não repete calibração nem simulação.",
    "sim.read_error": "Erro ao ler os arquivos: {error}", "sim.day_rain": "Dias de chuva", "sim.day_flow": "Dias de vazão válida",
    "sim.pet_metric": "ETP", "sim.daily": "série diária", "sim.monthly": "climatologia mensal", "sim.period": "Período da chuva: {start} → {end}",
    "sim.zero_notice": "A série contém {count} dia(s) com Q ≤ 0. A política `{policy}` será aplicada independentemente do regime.",
    "sim.running": "Executando…", "sim.preparing": "Preparando dados e estimando Tlag por assinatura…",
    "sim.sim_period": "Período simulado: **{start} – {end}** ({days} dias contínuos).",
    "sim.tlag_summary": "Tlag = **{tlag} d** · warm-up = {warmup} d",
    "sim.estimated": "Tempo estimado: **~{low}–{high} min** ({seconds:.1f} s por geração nesta máquina).",
    "sim.starting": "Iniciando calibração…", "sim.progress": "Progresso da calibração: {percent:.0f}%",
    "sim.calibration_complete": "Calibração concluída",
    "sim.elapsed": "Concluído em {seconds:.0f} s.", "sim.done": "Calibração concluída ✅", "sim.failed": "Falhou",
    "sim.bound_warning": "kp convergiu para o limite inferior; assinatura da degenerescência kp↔KL. Considere fixar kp.",
    "sim.benchmark_note": "O benchmark justo para simulação é a climatologia sazonal: **SS_vs_climatol > 0** indica habilidade além do ciclo médio. O teste final deve ser considerado uma única vez após o congelamento das decisões.",
    "sim.seed_dispersion": "Dispersão entre sementes (equifinalidade)",
    "sim.regime_help": "Independe de alpha_et e da política de zeros. 'unknown' bloqueia a simulação até uma escolha explícita.",
    "sim.zero_help": "Controla apenas a elegibilidade da observação; não seleciona o regime hidrológico.",
    "sim.alpha_help": "0,0 é a configuração primária candidata; 1,4 é a sensibilidade fixa.",
    "sim.kp_help": "ETP usada = ETP de entrada × kp. Use 1,0 para ETP final; ignorado quando a calibração de kp está ativa.",
    "sim.cal_kp_help": "Inclui kp como quinto parâmetro do DE. Solução no limite pode indicar degenerescência kp↔KL.",
    "sim.quick_help": "Desligue para usar três sementes no diagnóstico de equifinalidade.",
    "sim.iter_help": "Máximo de gerações da evolução diferencial; a parada antecipada é automática quando o NSE estabiliza.",
    "sim.upload_format": "**Formato:** texto separado por TAB, `dd/mm/aaaa`, decimal com ponto. Vazão pode ter lacunas; chuva deve ser contínua.",
    "prep.title": "CAWM-P — preparação de dados da bacia", "prep.caption": "Delineamento BHAE · inventário ANA default · PLU/FLU separados · chuva média IDW · ETP mensal embarcada.",
    "prep.inventory": "Inventário ANA default", "prep.outlet": "Exutório selecionado", "prep.map": "Mapa de estações", "prep.idw": "Chuva média (IDW)",
    "prep.step1": "Passo 1 · Bacia da estação (BHAE pronta)", "prep.step2": "Passo 2 · Seleção de postos pluviométricos", "prep.step3": "Passo 3 · Séries da ANA (chuva dos postos + vazão do exutório)",
    "prep.step4": "Passo 4 · Chuva média da bacia", "prep.step5": "Passo 5 · ETP mensal da bacia", "prep.step6": "Passo 6 · Pacote consolidado",
    "prep.search": "Buscar posto FLU (código ou nome)", "prep.result": "Resultado", "prep.load": "Carregar bacia pronta", "prep.select": "Selecionar postos PLU",
    "prep.download_rain": "⬇ Baixar chuva dos {count} postos PLU (ANA)", "prep.acquiring": "posto {code} ({index}/{total})", "prep.completed": "Concluído",
    "prep.completed_warn": "Concluído com avisos", "prep.acq_summary": "{done}/{total} tentados · {ok} com dados · {no_data} sem dados · {failed} falharam · {not_attempted} não tentados.",
    "prep.report_download": "Baixar station_acquisition_report.csv", "prep.report_caption": "Todo posto solicitado aparece no relatório; ausência de dados e falhas nunca são descartadas silenciosamente.",
    "prep.no_selection": "Selecione os postos PLU no Passo 2 para habilitar a aquisição automática.",
    "prep.outlet_flow": "**Vazão observada do exutório**", "prep.mean_button": "Calcular chuva média (IDW)",
    "prep.package": "Baixar pacote mínimo da bacia (.zip)", "prep.no_basin": "Carregue uma bacia para continuar.",
}

CATALOGS = {"en": _EN, "pt-BR": _PT}


@dataclass(frozen=True)
class Translator:
    language: str = DEFAULT_LANGUAGE

    def __post_init__(self) -> None:
        if self.language not in CATALOGS:
            object.__setattr__(self, "language", DEFAULT_LANGUAGE)

    def __call__(self, key: str, **values: Any) -> str:
        template = CATALOGS[self.language].get(key, _EN.get(key, key))
        return template.format(**values) if values else template

    def choose(self, english: str, portuguese: str, **values: Any) -> str:
        """Localize one-off operational text while preserving one language switch."""
        template = portuguese if self.language == "pt-BR" else english
        return template.format(**values) if values else template


def select_language(st: Any, key: str) -> Translator:
    """Render the common selector. English is deliberately the default."""
    labels = list(LANGUAGES)
    chosen = st.sidebar.selectbox(_EN["language"], labels, index=0, key=key)
    return Translator(LANGUAGES[chosen])
