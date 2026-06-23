# CAWM Simplex — Preparador (App geoespacial)

App Streamlit que parte do bruto: ponto do exutório → delineia a bacia (BHAE,
por atributo `nuareamont` + topologia `nutrjus`) → seleciona postos → carrega
séries da ANA → preenche falhas → calcula a chuva média IDW → exporta o "pacote
da bacia" para o Simulador. Tudo por upload.

## Rodar local (recomendado)
```bash
pip install -r requirements.txt
streamlit run app_preparador.py
```
`dados/inventario_ana_estacoes.csv` (inventário nacional) está incluído para o
Passo 2. Camadas BHAE (TRECHOS/ÁREAS) são enviadas por upload (.gpkg/.shp/.zip).

## Deploy no Streamlit Community Cloud
1. Suba **o conteúdo desta pasta como raiz de um repositório** (entrypoint
   `app_preparador.py` na raiz, `requirements.txt` ao lado).
2. `geopandas>=0.14` + `shapely>=2.0` instalam por *wheels* no Python 3.12 do
   Cloud — sem `packages.txt`. **Se** o log de build acusar GEOS/GDAL/
   spatialindex, crie um `packages.txt` **na raiz do repo** com:
   ```
   libgeos-dev
   libspatialindex-dev
   ```

## Atenção — limite de memória
O Community Cloud tem ~1 GB de RAM. Recortes de bacia (ex.: BHAE-SIRINHAEM)
cabem bem como demo. A **BHAE nacional** provavelmente estoura a memória — para
o pipeline nacional completo, rode local (conda + script), como na arquitetura
decidida. O parser de `.mdb` da ANA exige `mdbtools` (apt); o fluxo via CSV
incluído dispensa isso.

## Atualização — Preparador com inventário default e ETP embarcada

Esta versão incorpora o fluxo de menor custo computacional para o MVP:

1. `dados/etp_brasil.npz` é carregado como climatologia mensal default de ETP.
2. `dados/inventario_ana_estacoes.csv` é carregado automaticamente.
3. Estações pluviométricas e fluviométricas são separadas rigidamente:
   - FLU: seleção do exutório/vazão;
   - PLU: chuva média por IDW.
4. A barra lateral permite buscar estação fluviométrica por código ou nome.
5. O mapa exibe apenas estações filtradas espacialmente; nunca renderiza o inventário nacional inteiro.
6. O limite consolidado da bacia pode ser baixado em GeoJSON.
7. A drenagem a montante pode ser baixada como função secundária.
8. O app gera um pacote mínimo `.zip` com bacia, metadados, ETP e, quando disponível, chuva média.

O HydroBR/HidrowebService ainda não foi incorporado nesta versão. O upload manual das séries permanece como fallback até a definição do provedor de aquisição.
