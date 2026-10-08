"""ANTHRO domain: sex and SMM (date matching, participant mean, fold scaling, registry, no "gender", real checks)."""
from __future__ import annotations

from dataclasses import replace
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from inventory.catalog import FileChoice
from inventory.sensors import CpetData, Loaded, VisitData, load_participant_fitdays
from pipeline.cache import ProcessedCache, cache_key
from pipeline.domains import INPUTS, SCALING, input_sensor_groups, parse_configuration
from pipeline.folds import build_fold_data, run_folds
from pipeline.inclusion import shared_mask
from pipeline.prepare.anthro import (SEX_COLUMN, SMM_COLUMN, STATIC_GROUP, SmmVisit, build_anthro, candidate_smm,
                                     fitdays_number, fitdays_smm_kg, parse_fitdays_date, participant_smm,
                                     read_cpet_test_date)
from pipeline.prepare.grid import Grid
from pipeline.prepare.group import PrepInput, presence_column
from pipeline.prepare.settings import SMM_PARTICIPANT_MEAN, SMM_PER_VISIT, PrepSettings
from pipeline.prepare.visit import candidate_refs
from tests.synthetic import SYNTHETIC_RUN, Synthetic

NO_CHOICE = FileChoice(None, (), None)
GRID_S = 50
FORBIDDEN_WORD = "gender"
# Gies-style export: English columns, a `Visit` label, two visits on 24 Jul (as for G08 Easy and Medium2).
GIES_FITDAYS = pd.DataFrame({
    "Date": ["13:25 Jul.9,2024", "16:26 Jul.12,2024", "10:10 Jul.24,2024", "12:48 Jul.24,2024", "18:03 Aug.1,2024"],
    "Weight": ["70.00kg", "70.40kg", "69.80kg", "69.90kg", "70.10kg"],
    "Skeletal muscle": ["40.0%", "41.0%", "42.0%", "43.0%", "- -"],
    "Visit": ["Ramp", "Hard", "Easy", "Medium2", "Medium"],
})
# Lucas-style export: Dutch columns, no `Visit` label.
LUCAS_FITDAYS = pd.DataFrame({
    "Datum": ["12:13 17/02/2026", "09:07 24/02/2026", "10:11 22/04/2026", "10:12 22/04/2026"],
    "Gewicht": ["93.55kg", "93.20kg", "93.00kg", "93.10kg"],
    "Skeletspier": ["44.0%", "45.0%", "46.0%", "47.0%"],
})


def _gies(visit_id: str, source_visit: str, day: date) -> SmmVisit:
    return SmmVisit(visit_id, source_visit, day)


# --- parsing --------------------------------------------------------------------------------------------------

@pytest.mark.parametrize(("text", "expected"), [
    ("17:29 Jul.24,2024", date(2024, 7, 24)),
    ("09:03 Aug.1,2024", date(2024, 8, 1)),
    ("12:13 17/02/2026", date(2026, 2, 17)),
    ("16:32 02/03/2026", date(2026, 3, 2)),  # day first: 2 March, not 3 February
])
def test_both_fitdays_date_formats(text: str, expected: date) -> None:
    assert parse_fitdays_date(text) == expected


def test_unreadable_fitdays_date_raises() -> None:
    with pytest.raises(ValueError, match="Fitdays date"):
        parse_fitdays_date("2024-07-24")


@pytest.mark.parametrize(("cell", "expected"), [("81.90kg", 81.9), ("42.7%", 42.7), (55.0, 55.0), ("43,5%", 43.5)])
def test_fitdays_numbers_drop_their_units(cell: object, expected: float) -> None:
    assert fitdays_number(cell) == pytest.approx(expected)


def test_missing_fitdays_number_is_nan() -> None:
    assert np.isnan(fitdays_number("- -"))


def _write_cpet_header(path: Path, first_row: list[object]) -> Path:
    pd.DataFrame([first_row, ["Last Name", "Study"]]).to_excel(path, header=False, index=False)
    return path


def test_cpet_test_date_is_read_day_first(tmp_path: Path) -> None:
    path = _write_cpet_header(tmp_path / "cpet.xlsx", ["ID1", None, None, "Test date", "02/03/2026", "Pressure"])
    assert read_cpet_test_date(path) == date(2026, 3, 2)


def test_missing_cpet_test_date_raises(tmp_path: Path) -> None:
    path = _write_cpet_header(tmp_path / "cpet.xlsx", ["ID1", None, None, "Barometric Pressure", 770])
    with pytest.raises(ValueError, match="test date"):
        read_cpet_test_date(path)


# --- matching by date -----------------------------------------------------------------------------------------

def test_every_visit_gets_smm_by_date_including_same_day_visits() -> None:
    visits = (_gies("G08_Ramp", "Ramp", date(2024, 7, 9)), _gies("G08_Hard", "Hard", date(2024, 7, 12)),
              _gies("G08_Easy", "Easy", date(2024, 7, 24)), _gies("G08_Medium", "Medium2", date(2024, 7, 24)))
    smm = participant_smm(GIES_FITDAYS, visits, SMM_PER_VISIT)
    assert smm == pytest.approx({"G08_Ramp": 0.40 * 70.0, "G08_Hard": 0.41 * 70.4,
                                 "G08_Easy": 0.42 * 69.8, "G08_Medium": 0.43 * 69.9})


def test_lucas_columns_without_visit_labels() -> None:
    assert fitdays_smm_kg(LUCAS_FITDAYS, SmmVisit("L01_Ramp", "Ramp", date(2026, 2, 17))) == pytest.approx(
        0.44 * 93.55)


def test_visit_without_a_same_day_row_raises_naming_it() -> None:
    with pytest.raises(ValueError, match=r"G08_Step: no Fitdays row on the CPET test date 2024-07-25"):
        fitdays_smm_kg(GIES_FITDAYS, _gies("G08_Step", "Step", date(2024, 7, 25)))


def test_same_day_rows_without_labels_raise() -> None:
    with pytest.raises(ValueError, match=r"L14_Interval: 2 Fitdays rows"):
        fitdays_smm_kg(LUCAS_FITDAYS, SmmVisit("L14_Interval", "Interval", date(2026, 4, 22)))


def test_visit_label_cross_check_fails_on_a_mismatch() -> None:
    with pytest.raises(ValueError, match=r"G08_Medium: .* labelled 'Hard', not 'Medium'"):
        fitdays_smm_kg(GIES_FITDAYS, _gies("G08_Medium", "Medium", date(2024, 7, 12)))


def test_matched_row_without_smm_raises() -> None:
    with pytest.raises(ValueError, match="G08_Medium: no SMM"):
        fitdays_smm_kg(GIES_FITDAYS, _gies("G08_Medium", "Medium", date(2024, 8, 1)))


def test_participant_mean_gives_one_smm_per_participant() -> None:
    visits = (_gies("G08_Ramp", "Ramp", date(2024, 7, 9)), _gies("G08_Hard", "Hard", date(2024, 7, 12)))
    smm = participant_smm(GIES_FITDAYS, visits, SMM_PARTICIPANT_MEAN)
    assert set(smm) == {"G08_Ramp", "G08_Hard"}
    assert len(set(smm.values())) == 1
    assert smm["G08_Ramp"] == pytest.approx((0.40 * 70.0 + 0.41 * 70.4) / 2)


def test_fitdays_input_is_not_mutated() -> None:
    before = GIES_FITDAYS.copy()
    participant_smm(GIES_FITDAYS, (_gies("G08_Ramp", "Ramp", date(2024, 7, 9)),), SMM_PARTICIPANT_MEAN)
    pd.testing.assert_frame_equal(GIES_FITDAYS, before)


def test_unknown_smm_mode_raises() -> None:
    with pytest.raises(ValueError, match="smm"):
        PrepSettings(smm="per_day")


@pytest.fixture
def source_roots(tmp_path: Path) -> tuple[Path, Path]:
    roots = (tmp_path / "prepare", tmp_path / "inventory")
    for root in roots:
        root.mkdir()
        (root / "module.py").write_text("X = 1\n", encoding="utf-8")
    return roots


def test_smm_flag_is_part_of_the_cache_key(source_roots: tuple[Path, Path]) -> None:
    default = PrepSettings()
    assert default.smm == SMM_PER_VISIT
    assert cache_key(default, source_roots) != cache_key(replace(default, smm=SMM_PARTICIPANT_MEAN), source_roots)


# --- builder --------------------------------------------------------------------------------------------------

def _input(sex: object, smm_kg: float) -> PrepInput:
    cpet = Loaded(NO_CHOICE, CpetData(pd.DataFrame({"t": [0.0]}), {"sex": sex, "age": 30}), None)
    visit = VisitData(SimpleNamespace(visit_id="G99_Ramp"), None, None, {}, cpet, None, None, None, None, None)
    t = np.arange(GRID_S)
    grid = Grid(t, np.zeros(GRID_S, dtype=np.int8), np.full(GRID_S, -1, dtype=np.int8))
    return PrepInput(visit, grid, PrepSettings(), (), "", smm_kg)


@pytest.mark.parametrize(("sex", "code"), [("Male", 1.0), ("Female", 0.0), (" male ", 1.0)])
def test_sex_is_one_for_male_and_constant_within_the_visit(sex: str, code: float) -> None:
    group = build_anthro(_input(sex, 31.5))
    assert group.sensor_group == STATIC_GROUP
    np.testing.assert_array_equal(group.values[SEX_COLUMN], np.full(GRID_S, code))
    np.testing.assert_array_equal(group.values[SMM_COLUMN], np.full(GRID_S, 31.5))
    assert group.present.all()


def test_unknown_sex_raises() -> None:
    with pytest.raises(ValueError, match="G99_Ramp: unknown sex"):
        build_anthro(_input("X", 31.5))


def test_unresolved_smm_makes_static_absent() -> None:
    group = build_anthro(_input("Male", float("nan")))
    assert not group.present.any()


# --- registry, fold scalers, naming ---------------------------------------------------------------------------

def test_wr_anthro_configuration_and_anthro_never_alone() -> None:
    config = parse_configuration("WR+ANTHRO")
    assert config.columns == ("WR", SEX_COLUMN, SMM_COLUMN)
    assert config.sensor_groups == ("WR", STATIC_GROUP)
    assert input_sensor_groups()[-1] == STATIC_GROUP
    assert list(INPUTS)[-1] == "ANTHRO"
    with pytest.raises(ValueError, match="time-varying"):
        parse_configuration("ANTHRO")
    with pytest.raises(ValueError, match="canonical order"):
        parse_configuration("ANTHRO+WR")


def test_sex_passes_through_the_fold_scaler_unchanged_and_smm_is_z_scored(synthetic: Synthetic) -> None:
    mask = shared_mask(synthetic.cache, synthetic.inclusion)
    fold = run_folds(SYNTHETIC_RUN, synthetic.cache)[0]
    columns = ("WR", SEX_COLUMN, SMM_COLUMN)
    data = build_fold_data(synthetic.cache, mask, fold, columns)
    assert (data.scalers[SEX_COLUMN].kind, data.scalers[SMM_COLUMN].kind) == ("none", "zscore")
    sex = data.features[:, columns.index(SEX_COLUMN)]
    np.testing.assert_array_equal(sex, synthetic.cache.column(SEX_COLUMN))
    assert set(np.unique(sex)) == {0.0, 1.0}


def test_no_column_or_identifier_says_gender(synthetic: Synthetic) -> None:
    names = [*synthetic.cache.columns, *INPUTS, *SCALING, *(c for spec in INPUTS.values() for c in spec.columns)]
    assert not [n for n in names if FORBIDDEN_WORD in n.lower()]


# --- real data ------------------------------------------------------------------------------------------------

@pytest.mark.real
def test_real_every_candidate_visit_gets_smm_by_date() -> None:
    _, candidates = candidate_refs()
    fitdays = {d: load_participant_fitdays(d) for d in dict.fromkeys(ref.path.parent for ref in candidates)}
    smm = candidate_smm(candidates, fitdays, SMM_PER_VISIT)
    assert set(smm) == {ref.visit_id for ref in candidates}
    assert all(np.isfinite(v) and v > 0 for v in smm.values())


@pytest.mark.real
def test_real_static_group_and_no_gender_column(real_cache: ProcessedCache) -> None:
    assert not [c for c in real_cache.columns if FORBIDDEN_WORD in c.lower()]
    assert (real_cache.column(presence_column(STATIC_GROUP)) == 1).all()
    assert set(np.unique(real_cache.column(SEX_COLUMN))) <= {0.0, 1.0}
