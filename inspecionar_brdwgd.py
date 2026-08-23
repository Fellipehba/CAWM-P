"""
CAWM Simplex-D — Inspetor dos arquivos NetCDF do BR-DWGD v3.2.4.

Uso (na máquina local, no ambiente do CAWM-Simplex):
    python inspecionar_brdwgd.py pr_19610101_19801231_BR-DWGD_UFES_UTEXAS_v_3.2.4.nc [outros.nc ...]

Ou, para inspecionar os três de uma vez:
    python inspecionar_brdwgd.py pr_*.nc

O que ele verifica (tudo em leitura preguiçosa — NÃO carrega os 2 GB):
  1. Dimensões, coordenadas e variáveis (nomes reais, que podem variar).
  2. Empacotamento: dtype em disco, scale_factor, add_offset, _FillValue,
     missing_value — crítico para ler os valores físicos corretos.
  3. Grade: extensão lat/lon, resolução, ordem (crescente/decrescente).
  4. Tempo: início, fim, calendário, passo, continuidade entre arquivos.
  5. Sanidade física: estatísticas de um recorte pequeno (média anual em
     mm deve cair na faixa plausível do Brasil, ~300–3500 mm/ano).

Requisitos: xarray + netCDF4 (já no ambiente do projeto).
"""
from __future__ import annotations

import sys
import glob

import numpy as np
import xarray as xr


def _achar_var_precipitacao(ds: xr.Dataset) -> str:
    """Detecta a variável de precipitação sem assumir o nome exato."""
    candidatas = [v for v in ds.data_vars if v.lower() in ("pr", "prec", "precipitation", "precip")]
    if candidatas:
        return candidatas[0]
    # fallback: a variável com mais dimensões
    return max(ds.data_vars, key=lambda v: ds[v].ndim)


def _achar_dim(ds: xr.Dataset, opcoes) -> str | None:
    for o in opcoes:
        if o in ds.dims or o in ds.coords:
            return o
    return None


def inspecionar(caminho: str) -> dict:
    print("=" * 78)
    print(f"ARQUIVO: {caminho}")
    print("=" * 78)

    # mask_and_scale=False primeiro, para ver o dado CRU em disco
    ds_cru = xr.open_dataset(caminho, mask_and_scale=False, decode_times=False)
    var = _achar_var_precipitacao(ds_cru)
    v = ds_cru[var]

    print(f"\n[1] Estrutura")
    print(f"    dimensões : {dict(ds_cru.sizes)}")
    print(f"    variáveis : {list(ds_cru.data_vars)}  ->  precipitação = '{var}'")
    print(f"    coords    : {list(ds_cru.coords)}")

    print(f"\n[2] Empacotamento da variável '{var}'")
    print(f"    dtype em disco : {v.dtype}")
    for attr in ("scale_factor", "add_offset", "_FillValue", "missing_value", "units", "long_name"):
        if attr in v.attrs or attr in v.encoding:
            print(f"    {attr:14s} : {v.attrs.get(attr, v.encoding.get(attr))}")
    if not any(a in v.attrs for a in ("scale_factor", "add_offset")):
        print("    (sem scale/offset nos attrs — verificar encoding após decode)")

    ds_cru.close()

    # agora com decodificação completa (valores físicos, tempo real)
    ds = xr.open_dataset(caminho)  # lazy
    v = ds[_achar_var_precipitacao(ds)]

    dlat = _achar_dim(ds, ["latitude", "lat", "y"])
    dlon = _achar_dim(ds, ["longitude", "lon", "x"])
    dtem = _achar_dim(ds, ["time", "tempo", "date"])
    if not all([dlat, dlon, dtem]):
        print(f"\nATENÇÃO: dims não identificadas automaticamente "
              f"(lat={dlat}, lon={dlon}, time={dtem}). Reportar a estrutura acima.")
        return {}

    lat = ds[dlat].values
    lon = ds[dlon].values
    tempo = ds[dtem].values

    print(f"\n[3] Grade")
    print(f"    lat : {lat.min():.3f} a {lat.max():.3f}  "
          f"(n={lat.size}, passo={np.diff(lat)[0]:+.3f}, "
          f"{'crescente' if lat[1] > lat[0] else 'DECRESCENTE'})")
    print(f"    lon : {lon.min():.3f} a {lon.max():.3f}  "
          f"(n={lon.size}, passo={np.diff(lon)[0]:+.3f})")

    print(f"\n[4] Tempo")
    print(f"    início : {tempo[0]}")
    print(f"    fim    : {tempo[-1]}")
    print(f"    n      : {tempo.size} passos")
    passos = np.unique(np.diff(tempo).astype("timedelta64[D]"))
    print(f"    passo  : {passos} {'(diário contínuo, OK)' if passos.size == 1 else 'ATENÇÃO: passo irregular!'}")

    # [5] sanidade física: recorte pequeno na Zona da Mata de PE (região do Sirinhaém)
    print(f"\n[5] Sanidade física (recorte 0.5° x 0.5° em torno de -8.6, -35.5)")
    sel_lat = slice(-8.9, -8.3) if lat[1] > lat[0] else slice(-8.3, -8.9)
    rec = v.sel({dlat: sel_lat, dlon: slice(-35.8, -35.2)})
    if rec.sizes.get(dlat, 0) == 0 or rec.sizes.get(dlon, 0) == 0:
        print("    (recorte vazio — grade não cobre a área? reportar)")
    else:
        # carrega SÓ o recorte (pequeno)
        rec = rec.load()
        anual = float(rec.mean(dim=[dlat, dlon]).mean()) * 365.25
        print(f"    células no recorte : {rec.sizes[dlat]} x {rec.sizes[dlon]}")
        print(f"    fração NaN         : {float(np.isnan(rec.values).mean()):.3f}")
        print(f"    média anual        : {anual:.0f} mm/ano  "
              f"[{'plausível' if 300 <= anual <= 3500 else 'FORA DA FAIXA — investigar unidades/escala'}]")
        print(f"    máx diário         : {float(np.nanmax(rec.values)):.1f} mm")

    info = {"var": v.name, "dlat": dlat, "dlon": dlon, "dtem": dtem,
            "t0": str(tempo[0]), "t1": str(tempo[-1]), "n_t": int(tempo.size)}
    ds.close()
    return info


if __name__ == "__main__":
    caminhos = []
    for arg in sys.argv[1:]:
        caminhos.extend(sorted(glob.glob(arg)))
    if not caminhos:
        print("Uso: python inspecionar_brdwgd.py <arquivo.nc> [arquivo2.nc ...]")
        sys.exit(1)

    infos = [inspecionar(c) for c in caminhos]

    # continuidade entre arquivos consecutivos
    if len(infos) > 1 and all(infos):
        print("\n" + "=" * 78)
        print("CONTINUIDADE ENTRE ARQUIVOS")
        for a, b in zip(infos[:-1], infos[1:]):
            fim = np.datetime64(a["t1"][:10])
            ini = np.datetime64(b["t1"][:10]) if False else np.datetime64(b["t0"][:10])
            gap = int((ini - fim).astype("timedelta64[D]").astype(int))
            print(f"    {a['t1'][:10]} -> {b['t0'][:10]} : intervalo de {gap} dia(s) "
                  f"[{'OK, contíguos' if gap == 1 else 'ATENÇÃO: descontinuidade!'}]")
