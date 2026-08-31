from pathlib import Path

import pandas as pd

import bhae_online as bo
import inventario as inv
import rainfall_qc as qc
import selecao_postos as sp
from combined_acquisition import acquire_rainfall_and_streamflow
from station_acquisition import AcquisitionPolicy

ROOT = Path(__file__).parents[1]


def test_obidos_selection_and_official_river_fixture():
    basin = bo.bacia_pronta(str(ROOT / "dados" / "bhae_bacias.parquet"), "17050001")
    assert basin.rio == "Rio Amazonas"
    inventory = sp.inventario_to_gdf(inv.carregar_inventario())
    plu = inventory[inventory.tipo.astype(str).str.lower().str.startswith("pluvio")]
    selected = sp.selecionar(basin.polygon, plu, buffer_km=10, idw_power=1,
                             tipos=("pluviometrica",))
    assert (len(selected.postos), selected.n_dentro, selected.n_buffer) == (915, 876, 39)


def test_medium_qc_and_acquisition_fixtures_are_deterministic():
    dates = pd.date_range("2020-01-01", periods=62)
    raw = {}
    for i in range(120):
        values = pd.Series(0.0, index=dates)
        values.iloc[30] = 100.0
        raw[str(i)] = values
    flags = qc.detect_flags(raw)
    decisions = qc.build_decisions(flags)
    assert len(flags) == 120
    assert decisions.selected_action.eq(qc.EXCLUDE_STATION_MONTH).all()

    calls = []
    def fetch(code, timeout_seconds):
        calls.append(code)
        return pd.Series([1.0], index=[dates[0]])
    result = acquire_rainfall_and_streamflow(
        [f"P{i}" for i in range(20)], "F1", fetch, fetch,
        policy=AcquisitionPolicy(max_attempts=1, backoff_seconds=0,
                                 jitter_seconds=0, max_workers=4),
    )
    assert len(result.rainfall) == 20
    assert len(calls) == 21
