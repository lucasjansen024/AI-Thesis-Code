"""Inclusion CSV schema, the frozen shared-mask unit list, and the target mask."""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
import pytest

from inventory.build_inclusion import INCLUSION_CSV, SENSOR_GROUPS
from pipeline.cache import ProcessedCache
from pipeline.inclusion import InclusionSchemaError, WINDOW_S, check_schema, shared_mask, valid_units
from pipeline.prepare.grid import EXERCISE, UNITS
from pipeline.prepare.group import presence_column
from tests.synthetic import PARTICIPANTS, VISIT_S, WARMUP_END_S, ColumnEdit, make_synthetic

# Shared mask on the committed CSV (MAPPING §8): Gies 47 of 55 visits, Lucas 52 of 52.
SHARED_MASK_EXCLUDED_UNITS = {
    ("G04_Hard", "rep2"), ("G06_Ramp", "visit"), ("G07_Hard", "rep1"), ("G07_Hard", "rep2"),
    ("G07_Medium", "rep1"), ("G07_Ramp", "visit"), ("G08_Easy", "rep1"), ("G08_Easy", "rep2"),
    ("G09_Ramp", "visit"), ("G11_Ramp", "visit"), ("G13_Easy", "rep1"), ("G13_Easy", "rep2"),
    ("G13_Ramp", "visit"),
}
INCLUDED_VISITS = {"Gies": (47, 55), "Lucas": (52, 52)}
MISSING = r"\bmissing\b"
SOURCE_CHECKED_GROUPS = ("VO2", "WR")


def raw_csv() -> pd.DataFrame:
    return pd.read_csv(INCLUSION_CSV, dtype=str, keep_default_na=False)


# --- schema ---------------------------------------------------------------------------------------------------

def test_committed_csv_passes_the_schema(committed_inclusion: pd.DataFrame) -> None:
    assert committed_inclusion["valid"].dtype == bool
    assert set(committed_inclusion["sensor_group"]) == set(SENSOR_GROUPS)


def _set_cell(df: pd.DataFrame, row: int, column: str, value: str) -> pd.DataFrame:
    values = df[column].to_numpy(copy=True)
    values[row] = value
    return df.assign(**{column: values})


@pytest.mark.parametrize("tamper", [
    lambda df: df[list(reversed(df.columns))],
    lambda df: _set_cell(df, 0, "valid", "yes"),
    lambda df: _set_cell(df, 0, "unit", "rep3"),
    lambda df: _set_cell(df, 0, "sensor_group", "EMG"),
    lambda df: _set_cell(df, 0, "decided_by", "guess"),
    lambda df: _set_cell(df, int(np.flatnonzero(df["sensor_group"] == "VO2")[0]), "hr_source", "cpet"),
    lambda df: df.drop(index=3),
    lambda df: pd.concat([df, df.iloc[[0]]], ignore_index=True),
], ids=["column order", "valid value", "unit", "sensor group", "decided_by", "hr_source on VO2", "missing row",
        "duplicate row"])
def test_schema_rejects_tampered_csv(tamper: Callable[[pd.DataFrame], pd.DataFrame]) -> None:
    with pytest.raises(InclusionSchemaError):
        check_schema(tamper(raw_csv()))


# --- frozen shared-mask units (committed CSV) -----------------------------------------------------------------

def test_shared_mask_units_match_the_frozen_list(committed_inclusion: pd.DataFrame) -> None:
    all_units = set(zip(committed_inclusion["visit_id"], committed_inclusion["unit"]))
    included = valid_units(committed_inclusion, SENSOR_GROUPS)
    assert all_units - included == SHARED_MASK_EXCLUDED_UNITS
    assert {u for u in included if u[0] == "G04_Hard"} == {("G04_Hard", "rep1")}
    assert {u for u in included if u[0] == "G07_Medium"} == {("G07_Medium", "rep2")}


@pytest.mark.parametrize("cohort", sorted(INCLUDED_VISITS))
def test_shared_mask_visit_counts(committed_inclusion: pd.DataFrame, cohort: str) -> None:
    rows = committed_inclusion[committed_inclusion["cohort"] == cohort]
    included_visits = {v for v, _ in valid_units(rows, SENSOR_GROUPS)}
    assert (len(included_visits), rows["visit_id"].nunique()) == INCLUDED_VISITS[cohort]


# --- target mask (synthetic) ----------------------------------------------------------------------------------

def _targets_by_visit(cache: ProcessedCache, mask: np.ndarray) -> dict[str, np.ndarray]:
    t = cache.column("t").astype(int)
    return {row["visit_id"]: t[row["start"]:row["stop"]][mask[row["start"]:row["stop"]]]
            for row in cache.visits.to_dict("records")}


def test_all_valid_gives_every_exercise_second() -> None:
    data = make_synthetic()
    targets = _targets_by_visit(data.cache, shared_mask(data.cache, data.inclusion))
    for pid, _, protocol in PARTICIPANTS:
        np.testing.assert_array_equal(targets[f"{pid}_{protocol}"], np.arange(WARMUP_END_S, VISIT_S))


def test_invalid_unit_drops_the_visit() -> None:
    data = make_synthetic(invalid=frozenset({("G03_Ramp", "NIRS")}))
    targets = _targets_by_visit(data.cache, shared_mask(data.cache, data.inclusion))
    assert len(targets["G03_Ramp"]) == 0
    assert len(targets["G01_Ramp"]) == VISIT_S - WARMUP_END_S


def _with_gap(visit_id: str, column: str, second: int) -> ColumnEdit:
    def edit(pid: str, columns: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        if not visit_id.startswith(pid):
            return columns
        flags = columns[column].copy()
        flags[second] = 0
        return {**columns, column: flags}
    return edit


def test_input_gap_in_context_drops_every_window_that_covers_it() -> None:
    gap = 400
    data = make_synthetic(edit=_with_gap("G01_Ramp", presence_column("WR"), gap))
    targets = _targets_by_visit(data.cache, shared_mask(data.cache, data.inclusion))["G01_Ramp"]
    covered = np.arange(gap + 1, gap + WINDOW_S + 1)
    expected = np.setdiff1d(np.arange(WARMUP_END_S, VISIT_S), covered)
    np.testing.assert_array_equal(targets, expected)


def test_missing_vo2_drops_only_that_target() -> None:
    gap = 450
    data = make_synthetic(edit=_with_gap("L01_Step", presence_column("VO2"), gap))
    targets = _targets_by_visit(data.cache, shared_mask(data.cache, data.inclusion))["L01_Step"]
    np.testing.assert_array_equal(targets, np.setdiff1d(np.arange(WARMUP_END_S, VISIT_S), [gap]))


def test_cached_visit_missing_from_csv_raises() -> None:
    data = make_synthetic()
    with pytest.raises(InclusionSchemaError, match="L01_Step"):
        shared_mask(data.cache, data.inclusion[data.inclusion["visit_id"] != "L01_Step"])


# --- real data ------------------------------------------------------------------------------------------------

def _visit_rows(cache: ProcessedCache, visit_id: str) -> slice:
    row = cache.visits.set_index("visit_id").loc[visit_id]
    return slice(int(row["start"]), int(row["stop"]))


@pytest.mark.real
def test_g07_medium_rep2_first_target_uses_rep1_context(real_cache: ProcessedCache,
                                                        committed_inclusion: pd.DataFrame) -> None:
    rows = _visit_rows(real_cache, "G07_Medium")
    mask = shared_mask(real_cache, committed_inclusion)[rows]
    t = real_cache.column("t")[rows].astype(int)
    unit = real_cache.column("unit")[rows].astype(int)
    first = int(t[mask].min())
    assert first == 1290
    assert not (mask & (unit == UNITS.index("rep1"))).any()
    context = np.arange(first - WINDOW_S, first)
    np.testing.assert_array_equal(context[[0, -1]], [1103, 1289])
    assert (unit[context] == UNITS.index("rep1")).all()
    assert (real_cache.column(presence_column("WR"))[rows][context] == 1).all()


@pytest.mark.real
def test_missing_source_verdicts_agree_with_presence(real_cache: ProcessedCache,
                                                     committed_inclusion: pd.DataFrame) -> None:
    phase = real_cache.column("phase").astype(int)
    unit = real_cache.column("unit").astype(int)
    rows = committed_inclusion[committed_inclusion["sensor_group"].isin(SOURCE_CHECKED_GROUPS)]
    disagreements = []
    for verdict in rows.to_dict("records"):
        span = _visit_rows(real_cache, verdict["visit_id"])
        in_unit = (phase[span] == EXERCISE) & (unit[span] == UNITS.index(verdict["unit"]))
        present = real_cache.column(presence_column(verdict["sensor_group"]))[span][in_unit] == 1
        csv_missing = bool(pd.Series([verdict["reason"]]).str.contains(MISSING).iloc[0])
        if csv_missing == bool(present.any()):
            disagreements.append(f"{verdict['visit_id']} {verdict['unit']} {verdict['sensor_group']}: "
                                 f"CSV missing={csv_missing}, present seconds={int(present.sum())}")
    assert not disagreements, "\n".join(disagreements)
