"""HR domain: HR and HRR (synthetic formula checks, cache key, real-data plausibility and presence)."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from inventory.build_inclusion import HR_SOURCES as INVENTORY_HR_SOURCES
from inventory.build_inclusion import SENSOR_GROUPS
from inventory.catalog import FileChoice
from inventory.sensors import CpetData, Loaded, TymewearData, VisitData
from pipeline.cache import ProcessedCache, cache_key
from pipeline.domains import parse_configuration
from pipeline.folds import build_fold_data, run_folds
from pipeline.inclusion import shared_mask, valid_units
from pipeline.prepare.grid import EXERCISE, UNITS, Grid
from pipeline.prepare.group import PrepInput, presence_column
from pipeline.prepare.hr import (CPET, HR_COLUMN, HRR_COLUMN, NO_SOURCE, TYMEWEAR, build_hr, read_hr_sources,
                                 tanaka_hr_peak)
from pipeline.prepare.settings import PrepSettings
from tests.synthetic import SYNTHETIC_RUN, Synthetic

VISIT_S, REST_S = 600, 120
AGE_YEARS = 31.7  # non-integer on purpose: full precision must be kept
HRR_PLAUSIBLE = (-0.1, 1.1)
HRR_PERCENTILES = (1, 99)
NO_CHOICE = FileChoice(None, (), None)
SIGNAL_OF_SOURCE = dict(INVENTORY_HR_SOURCES)  # tymewear -> tymewear_hr, cpet -> cpet_hr


# --- synthetic visit ------------------------------------------------------------------------------------------

def _hr_curve(t: np.ndarray, offset: float) -> np.ndarray:
    """Rest wobble, then a rise: the rest mean is not a round number."""
    return offset + np.where(t < REST_S, 3.0 * np.sin(t / 7.0), 0.15 * (t - REST_S))


def _visit(tymewear_t: np.ndarray | None, cpet_t: np.ndarray | None, age: object = AGE_YEARS) -> VisitData:
    """A VisitData with only what `build_hr` reads; Tymewear and CPET carry different HR (CPET +25 bpm)."""
    tymewear = Loaded(NO_CHOICE, None, None)
    if tymewear_t is not None:
        series = pd.DataFrame({"t": tymewear_t, "HR": _hr_curve(tymewear_t, 60.0), "VE": np.nan, "BR": np.nan})
        tymewear = Loaded(NO_CHOICE, TymewearData(series, pd.DataFrame(), 0.0, "test"), None)
    # The CPET header (and so age) exists even when a test gives the CPET no HR breaths.
    breath_t = np.array([], dtype=float) if cpet_t is None else cpet_t
    df = pd.DataFrame({"t": breath_t, "VO2": 1000.0, "HR": _hr_curve(breath_t, 85.0)})
    cpet = Loaded(NO_CHOICE, CpetData(df, {"age": age}), None)
    ref = SimpleNamespace(visit_id="G99_Ramp")
    return VisitData(ref, None, None, {}, cpet, tymewear, None, None, None, None)


def _input(visit: VisitData, source: str, settings: PrepSettings = PrepSettings()) -> PrepInput:
    t = np.arange(VISIT_S)
    grid = Grid(t, np.zeros(VISIT_S, dtype=np.int8), np.full(VISIT_S, -1, dtype=np.int8))
    return PrepInput(visit, grid, settings, (), source)


def _expected_hrr(hr_on_grid: np.ndarray, rest_seconds: np.ndarray) -> np.ndarray:
    hr_rest = hr_on_grid[rest_seconds].mean()
    hr_peak = 208.0 - 0.7 * AGE_YEARS
    return (hr_on_grid - hr_rest) / (hr_peak - hr_rest)


def test_tanaka_uses_full_precision_age() -> None:
    assert tanaka_hr_peak(AGE_YEARS) == pytest.approx(208.0 - 0.7 * 31.7, abs=1e-12)


def test_hrr_formula_exact_with_tymewear_source() -> None:
    t = np.arange(VISIT_S, dtype=float)
    group = build_hr(_input(_visit(tymewear_t=t, cpet_t=np.arange(0.0, VISIT_S, 3.0)), TYMEWEAR))
    hr = _hr_curve(t, 60.0)
    np.testing.assert_allclose(group.values[HR_COLUMN], hr, rtol=0, atol=1e-12)
    np.testing.assert_allclose(group.values[HRR_COLUMN], _expected_hrr(hr, t < REST_S), rtol=0, atol=1e-12)
    assert group.present.all()


def test_hrr_with_cpet_source_uses_cpet_rest_and_interpolation() -> None:
    breath_t = np.arange(0.5, VISIT_S, 3.0)  # second 0 lies before the first breath
    group = build_hr(_input(_visit(tymewear_t=np.arange(VISIT_S, dtype=float), cpet_t=breath_t), CPET))
    grid_t = np.arange(VISIT_S)
    hr = np.interp(grid_t, breath_t, _hr_curve(breath_t, 85.0), left=np.nan, right=np.nan)
    rest = np.isfinite(hr) & (grid_t < REST_S)
    assert not group.present[0] and group.present[1:int(breath_t[-1]) + 1].all()
    np.testing.assert_allclose(group.values[HR_COLUMN], hr, rtol=0, atol=1e-12)
    np.testing.assert_allclose(group.values[HRR_COLUMN], _expected_hrr(hr, rest), rtol=0, atol=1e-12)


def test_hrrest_window_setting_moves_hrrest() -> None:
    t = np.arange(VISIT_S, dtype=float)
    visit = _visit(tymewear_t=t, cpet_t=None)
    window = 60
    group = build_hr(_input(visit, TYMEWEAR, PrepSettings(hrrest_window_s=window)))
    hr = _hr_curve(t, 60.0)
    np.testing.assert_allclose(group.values[HRR_COLUMN], _expected_hrr(hr, t < window), rtol=0, atol=1e-12)


def test_no_source_means_hr_absent() -> None:
    group = build_hr(_input(_visit(tymewear_t=np.arange(VISIT_S, dtype=float), cpet_t=None), NO_SOURCE))
    assert not group.present.any()
    assert np.isnan(group.values[HR_COLUMN]).all() and np.isnan(group.values[HRR_COLUMN]).all()


def test_gap_longer_than_max_interp_gap_is_not_present() -> None:
    t = np.arange(VISIT_S, dtype=float)
    gap = (t > 300) & (t < 330)  # samples 300 and 330 are 30 s apart
    group = build_hr(_input(_visit(tymewear_t=t[~gap], cpet_t=None), TYMEWEAR))
    assert not group.present[301:330].any()
    assert group.present[:301].all() and group.present[330:].all()


def test_missing_age_raises() -> None:
    visit = _visit(tymewear_t=np.arange(VISIT_S, dtype=float), cpet_t=np.arange(0.0, VISIT_S, 3.0), age=None)
    with pytest.raises(ValueError, match="age"):
        build_hr(_input(visit, TYMEWEAR))


# --- HR source from the inclusion CSV -------------------------------------------------------------------------

def test_committed_csv_hr_sources() -> None:
    sources = read_hr_sources()
    assert (sources["G01_Ramp"], sources["G06_Ramp"], sources["G07_Ramp"]) == (TYMEWEAR, CPET, NO_SOURCE)


def _write_hr_rows(path: Path, rows: list[tuple[str, str, str]]) -> Path:
    pd.DataFrame(rows, columns=["visit_id", "sensor_group", "hr_source"]).to_csv(path, index=False)
    return path


def test_units_naming_different_sources_raise(tmp_path: Path) -> None:
    path = _write_hr_rows(tmp_path / "inclusion.csv", [("G01_Easy", "HR", "tymewear"), ("G01_Easy", "HR", "cpet")])
    with pytest.raises(ValueError, match="G01_Easy"):
        read_hr_sources(path)


def test_unknown_source_raises(tmp_path: Path) -> None:
    path = _write_hr_rows(tmp_path / "inclusion.csv", [("G01_Easy", "HR", "polar")])
    with pytest.raises(ValueError, match="polar"):
        read_hr_sources(path)


# --- cache key, domain registry, fold scalers -----------------------------------------------------------------

@pytest.fixture
def source_roots(tmp_path: Path) -> tuple[Path, Path]:
    roots = (tmp_path / "prepare", tmp_path / "inventory")
    for root in roots:
        root.mkdir()
        (root / "module.py").write_text("X = 1\n", encoding="utf-8")
    return roots


def test_hrrest_window_is_part_of_the_cache_key(source_roots: tuple[Path, Path]) -> None:
    default = PrepSettings()
    assert default.hrrest_window_s == REST_S
    changed = replace(default, hrrest_window_s=default.hrrest_window_s - 30)
    assert cache_key(default, source_roots) != cache_key(changed, source_roots)


def test_hr_sources_are_part_of_the_cache_key(source_roots: tuple[Path, Path]) -> None:
    before = cache_key(PrepSettings(), source_roots, {"G01_Ramp": TYMEWEAR, "G06_Ramp": CPET})
    reordered = cache_key(PrepSettings(), source_roots, {"G06_Ramp": CPET, "G01_Ramp": TYMEWEAR})
    switched = cache_key(PrepSettings(), source_roots, {"G01_Ramp": CPET, "G06_Ramp": CPET})
    assert before == reordered != switched


def test_wr_hr_configuration() -> None:
    config = parse_configuration("WR+HR")
    assert config.columns == ("WR", HR_COLUMN, HRR_COLUMN)
    assert config.sensor_groups == ("WR", "HR")


def test_hr_and_hrr_are_z_scored(synthetic: Synthetic) -> None:
    mask = shared_mask(synthetic.cache, synthetic.inclusion)
    fold = run_folds(SYNTHETIC_RUN, synthetic.cache)[0]
    scalers = build_fold_data(synthetic.cache, mask, fold, parse_configuration("WR+HR").columns).scalers
    assert (scalers[HR_COLUMN].kind, scalers[HRR_COLUMN].kind) == ("zscore", "zscore")


# --- real data ------------------------------------------------------------------------------------------------

def _rows(cache: ProcessedCache, visit_id: str) -> slice:
    row = cache.visits.set_index("visit_id").loc[visit_id]
    return slice(int(row["start"]), int(row["stop"]))


@pytest.mark.real
def test_hrr_percentiles_are_plausible_per_included_visit(real_cache: ProcessedCache,
                                                          committed_inclusion: pd.DataFrame) -> None:
    included = valid_units(committed_inclusion, SENSOR_GROUPS)
    phase, unit = real_cache.column("phase").astype(int), real_cache.column("unit").astype(int)
    present = real_cache.column(presence_column("HR")) == 1
    hrr = real_cache.column(HRR_COLUMN)
    failures, checked = [], 0
    for visit_id in sorted({v for v, _ in included}):
        span = _rows(real_cache, visit_id)
        unit_codes = [UNITS.index(u) for v, u in included if v == visit_id]
        seconds = (phase[span] == EXERCISE) & np.isin(unit[span], unit_codes) & present[span]
        if not seconds.any():
            failures.append(f"{visit_id}: no present HR in its included exercise seconds")
            continue
        low, high = np.percentile(hrr[span][seconds], HRR_PERCENTILES)
        checked += 1
        if low < HRR_PLAUSIBLE[0] or high > HRR_PLAUSIBLE[1]:
            failures.append(f"{visit_id}: HRR p1 {low:.3f}, p99 {high:.3f}")
    assert checked > 0
    assert not failures, "HRR outside [-0.1, 1.1]:\n" + "\n".join(failures)


@pytest.mark.real
def test_hr_missing_source_verdicts_agree_with_presence(real_cache: ProcessedCache,
                                                        committed_inclusion: pd.DataFrame) -> None:
    """The CSV says HR is missing in a unit iff no source passed or the chosen source has no data there."""
    phase, unit = real_cache.column("phase").astype(int), real_cache.column("unit").astype(int)
    present_flag = real_cache.column(presence_column("HR")) == 1
    rows = committed_inclusion[committed_inclusion["sensor_group"] == "HR"]
    disagreements = []
    for verdict in rows.to_dict("records"):
        source = verdict["hr_source"]
        csv_missing = source == NO_SOURCE or f"{SIGNAL_OF_SOURCE[source]} missing" in verdict["reason"]
        span = _rows(real_cache, verdict["visit_id"])
        in_unit = (phase[span] == EXERCISE) & (unit[span] == UNITS.index(verdict["unit"]))
        present = present_flag[span][in_unit]
        if csv_missing == bool(present.any()):
            disagreements.append(f"{verdict['visit_id']} {verdict['unit']} (source {source!r}): "
                                 f"CSV missing={csv_missing}, present seconds={int(present.sum())}")
    assert not disagreements, "\n".join(disagreements)
