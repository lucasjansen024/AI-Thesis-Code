"""§2.7 cleaning: exact match with the legacy functions; presence and interpolation on the grid."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from inventory.catalog import choose_cpet_file, enumerate_visits
from inventory.sensors import load_cpet
from pipeline.prepare.breath import breath_presence, clean_breaths, interpolate_to_grid
from src2_training_ready.processing.helper_functions import HelperFunctions

LEGACY_THRESHOLD, LEGACY_WINDOW = 3, 5
REAL_VISITS = ("G01_Ramp", "G04_Hard", "G07_Medium", "L04_PRBS", "L05_Ramp")


def legacy_clean(values: np.ndarray, center: bool) -> np.ndarray:
    df = pd.DataFrame({"x": np.asarray(values, dtype=float)})
    df = HelperFunctions.reject_outliers_pre_local_mean(df, "x", LEGACY_THRESHOLD, LEGACY_WINDOW)
    df = HelperFunctions.median_filter(df, "x", LEGACY_WINDOW, center=center)
    return df["x"].to_numpy()


def synthetic_breaths() -> np.ndarray:
    rng = np.random.default_rng(7)
    values = 1500 + rng.normal(0, 80, 400)
    values[[20, 77, 150, 151, 300]] *= 3  # spikes, one pair
    values[200:207] = 1600.0  # constant run: std 0, then 0/0 (kept) and x/0 (rejected)
    values[208] = 1700.0
    values[250] = np.nan
    return values


@pytest.mark.parametrize("center", [True, False])
def test_cleaning_matches_legacy_on_synthetic_breaths(center: bool) -> None:
    values = synthetic_breaths()
    np.testing.assert_array_equal(clean_breaths(values, center=center), legacy_clean(values, center=center))


def test_cleaning_rejects_spikes() -> None:
    cleaned = clean_breaths(synthetic_breaths(), center=True)
    assert np.nanmax(cleaned) < 2500


def test_cleaning_does_not_mutate_input() -> None:
    values = synthetic_breaths()
    before = values.copy()
    clean_breaths(values, center=True)
    np.testing.assert_array_equal(values, before)


def test_breath_presence_respects_gap_and_span() -> None:
    times = np.array([0.0, 5.0, 10.0, 40.0, 45.0])
    grid = np.arange(51)
    present = breath_presence(times, grid, max_gap_s=20)
    expected = (grid <= 10) | ((grid >= 40) & (grid <= 45))
    np.testing.assert_array_equal(present, expected)


def test_breath_presence_gap_at_limit_is_present() -> None:
    present = breath_presence(np.array([0.0, 20.0]), np.arange(21), max_gap_s=20)
    assert present.all()


def test_interpolation_is_linear_and_nan_outside() -> None:
    out = interpolate_to_grid(np.array([1.5, 3.5]), np.array([10.0, 30.0]), np.arange(5))
    np.testing.assert_allclose(out[[2, 3]], [15.0, 25.0])
    assert np.isnan(out[[0, 1, 4]]).all()


@pytest.mark.real
@pytest.mark.parametrize("visit_id", REAL_VISITS)
def test_vo2_cleaning_matches_legacy_on_real_visits(visit_id: str) -> None:
    ref = next(r for r in enumerate_visits() if r.visit_id == visit_id)
    cpet = load_cpet(choose_cpet_file(ref.path).selected).df
    vo2 = cpet["VO2"].to_numpy(dtype=float)
    vo2 = vo2[np.isfinite(vo2)]  # as build_vo2 feeds the cleaning
    np.testing.assert_array_equal(clean_breaths(vo2, center=True), legacy_clean(vo2, center=True))
