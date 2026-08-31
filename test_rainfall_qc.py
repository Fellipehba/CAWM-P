import pandas as pd

import rainfall_qc as qc


def _raw_series():
    idx = pd.date_range("2020-01-01", "2020-02-29", freq="D")
    values = pd.Series(0.0, index=idx)
    values.loc["2020-01-31"] = 100.0
    values.loc["2020-02-10"] = 300.0
    return {"A": values}


def test_recommended_policy_preserves_raw_and_excludes_station_month():
    raw = _raw_series()
    before = raw["A"].copy(deep=True)
    flags = qc.detect_flags(raw)
    decisions = qc.build_decisions(flags, "recommended")
    assert decisions.loc[decisions.test.eq("total_mensal"), "selected_action"].eq(qc.EXCLUDE_STATION_MONTH).all()
    assert decisions.loc[decisions.test.eq("limite_fisico"), "selected_action"].eq(qc.KEEP_AS_RECORDED).all()
    cleaned = qc.apply_decisions(raw, decisions)
    pd.testing.assert_series_equal(raw["A"], before)
    assert cleaned["A"].loc["2020-01"].isna().all()
    assert cleaned["A"].loc["2020-02-10"] == 300.0


def test_presets_are_explicit_and_editable():
    flags = qc.detect_flags(_raw_series())
    assert qc.build_decisions(flags, "keep_all").selected_action.eq(qc.KEEP_AS_RECORDED).all()
    assert not qc.build_decisions(flags, "exclude_all").selected_action.eq(qc.KEEP_AS_RECORDED).any()


def test_day_only_and_keep_overrides_reapply_from_raw_deterministically():
    raw = _raw_series()
    flags = qc.detect_flags(raw)
    decisions = qc.build_decisions(flags)
    monthly = decisions.test.eq("total_mensal")
    decisions.loc[monthly, "selected_action"] = qc.EXCLUDE_FLAGGED_DAY
    day_only = qc.apply_decisions(raw, decisions)
    assert pd.isna(day_only["A"].loc["2020-01-31"])
    assert day_only["A"].loc["2020-01-01"] == 0.0
    decisions.loc[monthly, "selected_action"] = qc.KEEP_AS_RECORDED
    restored = qc.apply_decisions(raw, decisions)
    pd.testing.assert_series_equal(restored["A"], raw["A"])
    repeated = qc.apply_decisions(raw, decisions)
    pd.testing.assert_series_equal(repeated["A"], restored["A"])
