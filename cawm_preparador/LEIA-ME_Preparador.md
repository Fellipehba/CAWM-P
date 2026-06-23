# App 1 — Preparador de dados (CAWM Simplex)

Fluxo: ponto do exutório → delineamento (BHAE) → seleção de postos →
séries de chuva → chuva média (IDW) → pacote para o Simulador.

## Instalação (recomenda-se conda pelas libs geoespaciais)
```
conda create -n cawm-prep python=3.11
conda activate cawm-prep
conda install -c conda-forge geopandas folium streamlit
pip install streamlit-folium
```
ou: `pip install -r requirements_preparador.txt`

## Executar
```
streamlit run app_preparador.py
```

## Uso
1. Barra lateral: envie as duas camadas da BHAE (trechos e áreas) e informe
   o exutório (lon/lat). Clique "Delinear bacia".
2. Envie o inventário de estações (CSV: cod, lon, lat, tipo). "Selecionar postos".
3. Enquanto o token da ANA não estiver configurado, suba os arquivos de série
   do HidroWeb (o nome do arquivo deve conter o código da estação).
4. "Calcular chuva média" → baixe o CSV para usar no Simulador (App 2).

## Token ANA (quando chegar)
O provedor da API oficial já está no módulo aquisicao_ana.py. Configure as
credenciais e o download automático substitui o upload manual (que continua
como fallback). O token de 60 min é renovado automaticamente.

## Estado atual da validação
- Delineamento (BHAE): validado, Sirinhaém 1.288,5 km² (−2,1% vs referência).
- Seleção de postos: validado.
- Chuva média IDW: validado em diferença zero vs planilha.
- Aquisição ANA: parsers validados; provedores de download a validar local.
