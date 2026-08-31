import ana_hydrobr as ana


def test_streamflow_parser_prefers_consisted_duplicate():
    xml = b"""<root>
      <SerieHistorica><NivelConsistencia>1</NivelConsistencia><DataHora>2020-01-01T00:00:00</DataHora><Vazao01>1.5</Vazao01></SerieHistorica>
      <SerieHistorica><NivelConsistencia>2</NivelConsistencia><DataHora>2020-01-01T00:00:00</DataHora><Vazao01>2.5</Vazao01></SerieHistorica>
    </root>"""
    series = ana._parse_streamflow_xml(xml)
    assert series.loc["2020-01-01"] == 2.5


def test_bounded_flow_request_uses_type_three():
    class Response:
        content = b"<root><SerieHistorica><NivelConsistencia>2</NivelConsistencia><DataHora>2020-01-01T00:00:00</DataHora><Vazao01>4</Vazao01></SerieHistorica></root>"
        def raise_for_status(self): pass
    class Session:
        def __init__(self): self.calls = []
        def get(self, url, params, timeout):
            self.calls.append((params, timeout)); return Response()
    session = Session()
    series = ana.fetch_station_streamflow("17050001", timeout_seconds=7, session=session)
    assert series.iloc[0] == 4
    assert session.calls == [({"codEstacao": "17050001", "dataInicio": "", "dataFim": "", "tipoDados": "3", "nivelConsistencia": ""}, 7.0)]
