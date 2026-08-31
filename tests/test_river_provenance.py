from pathlib import Path

import pandas as pd

from river_provenance import station_river_table


def test_enrichment_uses_outlet_only_and_never_station_name():
    raw = pd.DataFrame({"cod_posto": ["1", "2"], "exutorio": [10, 20],
                        "nome": ["RIO INVENTADO", "Amazonas"]})
    mapping = pd.DataFrame({"exutorio": [10], "rio_bhae": ["Rio Oficial"],
                            "bhae_versao": ["2024"]})
    out = station_river_table(raw, mapping)
    assert out.loc[out.cod_posto.eq("1"), "rio_bhae"].iloc[0] == "Rio Oficial"
    assert pd.isna(out.loc[out.cod_posto.eq("2"), "rio_bhae"].iloc[0])


def test_public_parquet_has_ten_fixed_authoritative_fixtures():
    path = Path(__file__).parents[1] / "dados" / "bhae_indice.parquet"
    actual = pd.read_parquet(path).set_index("cod_posto")["rio_bhae"]
    expected = {
        "17050001": "Rio Amazonas", "18460000": "Rio Xingu",
        "18480000": "Rio Fresco", "19090000": "Rio Jari",
        "20100000": "Rio das Almas", "20980000": "Rio Preto",
        "10011200": "Rio Cangaime", "10011300": "Rio Pastaza",
        "10014000": "Rio Napo", "10200000": "Rio Javari",
    }
    assert {code: actual.loc[code] for code in expected} == expected
