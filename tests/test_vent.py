"""VENT domain: Tymewear V̇E and BR (trailing §2.7 cleaning, presence, registry, real-data checks)."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from inventory.catalog import FileChoice, choose_tymewear_file, enumerate_visits
from inventory.sensors import Loaded, TymewearData, VisitData, load_tymewear
from pipeline.cache import ProcessedCache
from pipeline.domains import parse_configuration
from pipeline.folds import build_fold_data, run_folds
from pipeline.inclusion import shared_mask
from pipeline.prepare.breath import clean_breaths, interpolate_to_grid
from pipeline.prepare.grid import EXERCISE, UNITS, Grid
from pipeline.prepare.group import PrepInput, presence_column
from pipeline.prepare.settings import PrepSettings
from pipeline.prepare.vent import BR_COLUMN, VE_COLUMN, VENT_GROUP, build_vent, finite_breaths
from tests.synthetic import SYNTHETIC_RUN, Synthetic
from tests.test_breath import legacy_clean

VISIT_S = 600
BREATH_SPACING_S = 2.3
NO_CHOICE = FileChoice(None, (), None)
REAL_VISITS = ("G01_Ramp", "G04_Hard", "G07_Medium", "L04_PRBS", "L05_Ramp")
VENT_SIGNALS = ("tymewear_ve", "tymewear_br")


# --- synthetic visit ------------------------------------------------------------------------------------------

def _breaths(spikes: bool = True) -> pd.DataFrame:
    t = np.arange(1.0, VISIT_S - 1, BREATH_SPACING_S)
    rng = np.random.default_rng(11)
    ve = 12.0 + 0.05 * t + rng.normal(0, 1.0, len(t))
    br = 15.0 + 0.02 * t + rng.normal(0, 0.8, len(t))
    if spikes:
        ve[[40, 120]] *= 4
        br[[60, 61]] *= 3
    return pd.DataFrame({"t": t, "VE": ve, "BR": br})


def _input(breaths: pd.DataFrame | None) -> PrepInput:
    tymewear = Loaded(NO_CHOICE, None, None)
    if breaths is not None:
        tymewear = Loaded(NO_CHOICE, TymewearData(pd.DataFrame(), breaths, 0.0, "test"), None)
    visit = VisitData(SimpleNamespace(visit_id="G99_Ramp"), None, None, {}, None, tymewear, None, None, None, None)
    grid = Grid(np.arange(VISIT_S), np.zeros(VISIT_S, dtype=np.int8), np.full(VISIT_S, -1, dtype=np.int8))
    return PrepInput(visit, grid, PrepSettings(), ())


def _expected(breaths: pd.DataFrame, column: str) -> np.ndarray:
    t, values = breaths["t"].to_numpy(), breaths[column].to_numpy()
    return interpolate_to_grid(t, legacy_clean(values, center=False), np.arange(VISIT_S))


def test_ve_and_br_are_legacy_trailing_cleaning_then_interpolation() -> None:
    breaths = _breaths()
    group = build_vent(_input(breaths))
    assert group.sensor_group == VENT_GROUP
    for column in (VE_COLUMN, BR_COLUMN):
        np.testing.assert_array_equal(group.values[column], _expected(breaths, column))


def test_median_is_trailing() -> None:
    """Changing breath k leaves every cleaned value before breath k unchanged (no look-ahead)."""
    values = _breaths(spikes=False)["VE"].to_numpy()
    k = 100
    perturbed = values.copy()
    perturbed[k] += 5.0
    before, after = clean_breaths(values, center=False), clean_breaths(perturbed, center=False)
    np.testing.assert_array_equal(before[:k], after[:k])
    assert not np.allclose(before[k:k + 5], after[k:k + 5], equal_nan=True)


def test_first_breaths_before_a_full_median_window_are_not_present() -> None:
    breaths = _breaths(spikes=False)
    group = build_vent(_input(breaths))
    breath_t = breaths["t"].to_numpy()
    first_full_window_s = breath_t[4]  # the 5th breath
    grid_t = np.arange(VISIT_S)
    assert not group.present[grid_t < first_full_window_s].any()
    # A rejected breath near the end leaves the trailing median NaN there (legacy behaviour), and
    # interpolation doesn't extrapolate, so presence ends at the last breath with both values cleaned.
    finite = np.isfinite(clean_breaths(breaths["VE"].to_numpy(), center=False)) & np.isfinite(
        clean_breaths(breaths["BR"].to_numpy(), center=False))
    last_clean_s = breath_t[np.flatnonzero(finite)[-1]]
    assert group.present[int(np.ceil(first_full_window_s)):int(np.floor(last_clean_s)) + 1].all()


def test_unsorted_and_non_finite_breaths_are_dropped_and_ordered() -> None:
    breaths = _breaths()
    messy = pd.concat([breaths.iloc[::-1], pd.DataFrame({"t": [np.nan, 50.0], "VE": [10.0, np.nan],
                                                         "BR": [10.0, 99.0]})], ignore_index=True)
    t, ve = finite_breaths(messy, VE_COLUMN)
    np.testing.assert_array_equal(t, breaths["t"].to_numpy())
    np.testing.assert_array_equal(ve, breaths["VE"].to_numpy())


def test_br_gap_makes_vent_absent_even_with_ve() -> None:
    breaths = _breaths(spikes=False)
    gap = (breaths["t"] > 300) & (breaths["t"] < 330)
    with_gap = breaths.assign(BR=breaths["BR"].where(~gap))
    group = build_vent(_input(with_gap))
    last_before, first_after = breaths["t"][breaths["t"] <= 300].max(), breaths["t"][breaths["t"] >= 330].min()
    absent = (np.arange(VISIT_S) > last_before) & (np.arange(VISIT_S) < first_after)
    assert np.isfinite(group.values[VE_COLUMN][absent]).all()
    assert not group.present[absent].any()


def test_no_tymewear_means_vent_absent() -> None:
    group = build_vent(_input(None))
    assert not group.present.any()
    assert np.isnan(group.values[VE_COLUMN]).all() and np.isnan(group.values[BR_COLUMN]).all()


def test_input_breaths_are_not_mutated() -> None:
    breaths = _breaths()
    before = breaths.copy()
    build_vent(_input(breaths))
    pd.testing.assert_frame_equal(breaths, before)


# --- domain registry, fold scalers ----------------------------------------------------------------------------

def test_wr_vent_configuration() -> None:
    config = parse_configuration("WR+VENT")
    assert config.columns == ("WR", VE_COLUMN, BR_COLUMN)
    assert config.sensor_groups == ("WR", VENT_GROUP)
    assert parse_configuration("WR+HR+VENT").columns == ("WR", "HR", "HRR", VE_COLUMN, BR_COLUMN)


def test_ve_and_br_are_z_scored(synthetic: Synthetic) -> None:
    mask = shared_mask(synthetic.cache, synthetic.inclusion)
    fold = run_folds(SYNTHETIC_RUN, synthetic.cache)[0]
    scalers = build_fold_data(synthetic.cache, mask, fold, parse_configuration("WR+VENT").columns).scalers
    assert (scalers[VE_COLUMN].kind, scalers[BR_COLUMN].kind) == ("zscore", "zscore")


# --- real data ------------------------------------------------------------------------------------------------

@pytest.mark.real
@pytest.mark.parametrize("visit_id", REAL_VISITS)
@pytest.mark.parametrize("column", [VE_COLUMN, BR_COLUMN])
def test_vent_cleaning_matches_legacy_on_real_visits(visit_id: str, column: str) -> None:
    ref = next(r for r in enumerate_visits() if r.visit_id == visit_id)
    breaths = load_tymewear(choose_tymewear_file(ref.path).selected, 0.0, "test").breaths
    _, values = finite_breaths(breaths, column)  # as build_vent feeds the cleaning
    assert len(values) > 0
    np.testing.assert_array_equal(clean_breaths(values, center=False), legacy_clean(values, center=False))


def _rows(cache: ProcessedCache, visit_id: str) -> slice:
    row = cache.visits.set_index("visit_id").loc[visit_id]
    return slice(int(row["start"]), int(row["stop"]))


@pytest.mark.real
def test_vent_missing_source_verdicts_agree_with_presence(real_cache: ProcessedCache,
                                                          committed_inclusion: pd.DataFrame) -> None:
    """The CSV says VENT is missing in a unit iff the processed flag has no present second there."""
    phase, unit = real_cache.column("phase").astype(int), real_cache.column("unit").astype(int)
    present_flag = real_cache.column(presence_column(VENT_GROUP)) == 1
    rows = committed_inclusion[committed_inclusion["sensor_group"] == VENT_GROUP]
    disagreements = []
    for verdict in rows.to_dict("records"):
        csv_missing = any(f"{signal} missing" in verdict["reason"] for signal in VENT_SIGNALS)
        span = _rows(real_cache, verdict["visit_id"])
        in_unit = (phase[span] == EXERCISE) & (unit[span] == UNITS.index(verdict["unit"]))
        present = present_flag[span][in_unit]
        if csv_missing == bool(present.any()):
            disagreements.append(f"{verdict['visit_id']} {verdict['unit']}: CSV missing={csv_missing}, "
                                 f"present seconds={int(present.sum())}")
    assert not rows.empty
    assert not disagreements, "\n".join(disagreements)
