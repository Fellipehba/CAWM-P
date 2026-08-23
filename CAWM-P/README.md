# CAWM-P — basin data preparer

CAWM-P is the station-based data-preparation component of CAWM Web. Public app: <https://cawm-p.streamlit.app/>.

```bash
pip install -r requirements.txt
streamlit run app_preparador.py
```

The app loads a prepared BHAE basin, selects ANA rain gauges spatially, acquires or accepts uploaded station series, calculates IDW mean rainfall and samples embedded monthly PET. English is the default UI; Portuguese (Brazil) is selectable.

Every automatic station batch produces `station_acquisition_report.csv`. The report contract permits `success`, `no_data`, `failed_after_retries`, `user_uploaded` and `not_attempted`; automatic ANA requests use the four applicable network states, while `user_uploaded` is reserved for user-supplied station series. Failures never disappear silently. Partial data may continue through the existing IDW workflow, with weights renormalized by the existing implementation, but the UI reports **Completed with warnings**. No new percentage blocking threshold is introduced.

CAWM-P depends on external ANA/HidroWeb availability and does not claim automatic gap filling in the primary app flow. Review station coverage and quality-control outputs before modelling. The national article experiment uses a separate controlled CAMELS-BR path.

## Português (Brasil)

O CAWM-P prepara o pacote station-based da bacia. Cada aquisição automática gera relatório por posto, distinguindo ausência legítima de dados de falha após tentativas. A correção da Rodada 15 não altera seleção espacial nem a equação IDW.
