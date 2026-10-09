"""Per-visit preparation overrides (`prep_overrides.csv`): schema, validation, Fitdays row and age overrides."""
from __future__ import annotations

import math
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from inventory.catalog import FileChoice
from inventory.sensors import CpetData, Loaded, TymewearData, VisitData, load_participant_fitdays
from pipeline.cache import cache_key
from pipeline.prepare.anthro import SmmVisit, candidate_smm, fitdays_smm_kg, participant_smm
from pipeline.prepare.grid import Grid
from pipeline.prepare.group import PrepInput
from pipeline.prepare.hr import HRR_COLUMN, TYMEWEAR, build_hr, tanaka_hr_peak
from pipeline.prepare.overrides import (AGE_FROM_VISIT, FITDAYS_ROW, NO_OVERRIDES, OVERRIDE_COLUMNS, OVERRIDES_CSV,
                                        PrepOverrides, check_overrides, read_overrides)
from pipeline.prepare.settings import SMM_PER_VISIT, PrepSettings
from pipeline.prepare.visit import candidate_refs
from tests.test_anthro import LUCAS_FITDAYS

NO_CHOICE = FileChoice(None, (), None)
INTERVAL = SmmVisit("L14_Interval", "Interval", date(2026, 4, 22))
PRBS = SmmVisit("L14_PRBS", "PRBS", date(2026, 4, 15))
VISIT_S = 600


def _write(tmp_path: Path, rows: list[list[str]], columns: list[str] = OVERRIDE_COLUMNS) -> Path:
    path = tmp_path / "prep_overrides.csv"
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)
    return path


# --- reading ----------------------------------------------------------------------------------------------------

def test_missing_file_means_no_overrides(tmp_path: Path) -> None:
    assert read_overrides(tmp_path / "absent.csv") == NO_OVERRIDES


def test_rows_are_read_per_field(tmp_path: Path) -> None:
    path = _write(tmp_path, [["L14_PRBS", FITDAYS_ROW, "08:28 09/04/2026", "why"],
                             ["L14_Interval", AGE_FROM_VISIT, "L14_PRBS", "why"]])
    overrides = read_overrides(path)
    assert overrides.fitdays_rows == {"L14_PRBS": "08:28 09/04/2026"}
    assert overrides.age_from == {"L14_Interval": "L14_PRBS"}


@pytest.mark.parametrize(("rows", "columns", "message"), [
    ([["L14_PRBS", "smm_kg", "37", "why"]], OVERRIDE_COLUMNS, "unknown fields"),
    ([["L14_PRBS", FITDAYS_ROW, "08:28 09/04/2026", ""]], OVERRIDE_COLUMNS, "empty cells"),
    ([["L14_PRBS", FITDAYS_ROW, "a", "why"], ["L14_PRBS", FITDAYS_ROW, "b", "why"]], OVERRIDE_COLUMNS, "repeated"),
    ([["L14_PRBS", FITDAYS_ROW, "a"]], ["visit_id", "field", "value"], "columns"),
])
def test_bad_files_raise(tmp_path: Path, rows: list[list[str]], columns: list[str], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        read_overrides(_write(tmp_path, rows, columns))


def test_committed_file_holds_the_l14_fixes() -> None:
    overrides = read_overrides(OVERRIDES_CSV)
    assert overrides.fitdays_rows == {"L14_PRBS": "08:28 09/04/2026", "L14_Interval": "10:11 22/04/2026"}
    assert overrides.age_from == {"L14_Interval": "L14_PRBS"}


def test_overrides_are_part_of_the_cache_key() -> None:
    changed = PrepOverrides({"L14_PRBS": "08:28 09/04/2026"}, {})
    assert cache_key(PrepSettings()) != cache_key(PrepSettings(), overrides=changed)


# --- validation against the visit catalogue ----------------------------------------------------------------------

def _ref(visit_id: str, participant_id: str) -> SimpleNamespace:
    return SimpleNamespace(visit_id=visit_id, participant_id=participant_id)


REFS = (_ref("L14_PRBS", "L14"), _ref("L14_Interval", "L14"), _ref("L13_Ramp", "L13"))


@pytest.mark.parametrize(("overrides", "message"), [
    (PrepOverrides({"L99_Ramp": "x"}, {}), "non-candidate"),
    (PrepOverrides({}, {"L14_Interval": "L13_Ramp"}), "different participant"),
    (PrepOverrides({}, {"L14_Interval": "L14_Interval"}), "not another visit"),
    (PrepOverrides({}, {"L14_Interval": "L14_Missing"}), "not another visit"),
])
def test_invalid_overrides_raise(overrides: PrepOverrides, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        check_overrides(overrides, REFS, REFS)


def test_valid_overrides_pass() -> None:
    check_overrides(PrepOverrides({"L14_PRBS": "x"}, {"L14_Interval": "L14_PRBS"}), REFS, REFS)


# --- Fitdays row override ----------------------------------------------------------------------------------------

def test_row_override_picks_the_named_reading() -> None:
    assert fitdays_smm_kg(LUCAS_FITDAYS, INTERVAL, "10:11 22/04/2026") == pytest.approx(0.46 * 93.00)


def test_row_override_can_name_another_day() -> None:
    assert fitdays_smm_kg(LUCAS_FITDAYS, PRBS, "09:07 24/02/2026") == pytest.approx(0.45 * 93.20)


def test_row_override_that_matches_nothing_raises() -> None:
    with pytest.raises(ValueError, match="matches 0 Fitdays rows"):
        fitdays_smm_kg(LUCAS_FITDAYS, PRBS, "08:28 09/04/2026")


def test_visits_without_an_override_still_match_by_date() -> None:
    l01 = SmmVisit("L01_Ramp", "Ramp", date(2026, 2, 17))
    smm = participant_smm(LUCAS_FITDAYS, (l01, INTERVAL), SMM_PER_VISIT, {"L14_Interval": "10:12 22/04/2026"})
    assert smm == pytest.approx({"L01_Ramp": 0.44 * 93.55, "L14_Interval": 0.47 * 93.10})


# --- age override ------------------------------------------------------------------------------------------------

def _hr_input(header_age: float, age_override: float) -> PrepInput:
    t = np.arange(VISIT_S, dtype=float)
    series = pd.DataFrame({"t": t, "HR": 60.0 + 0.2 * t, "VE": np.nan, "BR": np.nan})
    tymewear = Loaded(NO_CHOICE, TymewearData(series, pd.DataFrame(), 0.0, "test"), None)
    cpet = Loaded(NO_CHOICE, CpetData(pd.DataFrame({"t": [], "VO2": [], "HR": []}), {"age": header_age}), None)
    visit = VisitData(SimpleNamespace(visit_id="L14_Interval"), None, None, {}, cpet, tymewear, None, None, None,
                      None)
    grid = Grid(np.arange(VISIT_S), np.zeros(VISIT_S, dtype=np.int8), np.full(VISIT_S, -1, dtype=np.int8))
    return PrepInput(visit, grid, PrepSettings(), (), TYMEWEAR, math.nan, age_override)


def test_age_override_replaces_the_header_age_in_hrpeak() -> None:
    hr_rest, hr_end = 60.0 + 0.2 * 59.5, 60.0 + 0.2 * (VISIT_S - 1)
    group = build_hr(_hr_input(header_age=20.0, age_override=25.4))
    expected = (hr_end - hr_rest) / (tanaka_hr_peak(25.4) - hr_rest)
    assert group.values[HRR_COLUMN][-1] == pytest.approx(expected)


def test_without_an_override_the_header_age_is_used() -> None:
    hr_rest, hr_end = 60.0 + 0.2 * 59.5, 60.0 + 0.2 * (VISIT_S - 1)
    group = build_hr(_hr_input(header_age=20.0, age_override=math.nan))
    assert group.values[HRR_COLUMN][-1] == pytest.approx((hr_end - hr_rest) / (tanaka_hr_peak(20.0) - hr_rest))


# --- real data ---------------------------------------------------------------------------------------------------

@pytest.mark.real
def test_real_overrides_name_candidates_and_resolve() -> None:
    refs, candidates = candidate_refs()
    overrides = read_overrides()
    check_overrides(overrides, refs, candidates)
    l14 = [ref for ref in candidates if ref.participant_id == "L14"]
    fitdays = {l14[0].path.parent: load_participant_fitdays(l14[0].path.parent)}
    smm = candidate_smm(l14, fitdays, SMM_PER_VISIT, overrides.fitdays_rows)
    assert smm["L14_PRBS"] == pytest.approx(0.423 * 88.45)
    assert smm["L14_Interval"] == pytest.approx(0.422 * 86.80)
