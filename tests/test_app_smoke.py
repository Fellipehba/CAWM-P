from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_app_starts_in_english_without_exceptions():
    app = Path(__file__).parents[1] / "app_preparador.py"
    at = AppTest.from_file(str(app), default_timeout=30).run()
    assert not at.exception
    assert any("CAWM-P" in title.value for title in at.title)
    assert any("Step 1" in item.value for item in at.subheader)


def test_one_streamlit_click_invokes_one_combined_flow():
    harness = Path(__file__).parent / "app_combined_harness.py"
    at = AppTest.from_file(str(harness), default_timeout=10).run()
    at.button[0].click().run()
    assert list(at.session_state.calls) == ["P1", "P2", "F1"]
    assert at.session_state.report_rows == 3
