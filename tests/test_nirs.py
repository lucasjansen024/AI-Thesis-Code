"""NIRS domain: TSI and Hb-diff (per-second means, artefact rows, header rate, flags, registry, real checks)."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from inventory.catalog import FileChoice, choose_nirs_file, enumerate_visits
from inventory.sensors import FIT_FACTOR, NIRS_COLUMNS, TSI, Loaded, NirsData, VisitData, load_nirs
from pipeline.cache import ProcessedCache, cache_key
from pipeline.domains import input_sensor_groups, parse_configuration
from pipeline.folds import build_fold_data, run_folds
from pipeline.inclusion import shared_mask
from pipeline.prepare.grid import EXERCISE, UNITS, Grid
from pipeline.prepare.group import GroupColumns, PrepInput, presence_column
from pipeline.prepare.nirs import (HBDIFF_COLUMN, NIRS_GROUP, TSI_COLUMN, artefact_rows, build_nirs, nirs_group,
                                   read_export_sample_rate)
from pipeline.prepare.settings import HBDIFF_MEAN3, HBDIFF_REST_MEAN, PrepSettings
from tests.synthetic import SYNTHETIC_RUN, Synthetic

RATE_HZ = 10.0
EVENT_S, EVENT_ROW = 80, 300  # marker at row 300 = 80 s: a 10 Hz recording starts at 50 s
N_SAMPLES = 4000  # 10 Hz: 50 s .. 450 s
GRID_S = 460
FIRST_SECOND = 50
CHANNEL_LEVELS = {1: (10.0, 4.0), 2: (12.0, 3.0), 3: (8.0, 5.0)}  # O2Hb, HHb: Tx3 diff 3, mean diff 6
O2HB_SLOPE_UM = 0.001  # per sample, same in every channel
PHYSICAL_TSI_PCT = 60.0
NONPHYSICAL_ROW = {"O2Hb1": 220.0, "O2Hb2": 205.0, "O2Hb3": 240.0, TSI: 94.09, FIT_FACTOR: 97.28}
NO_CHOICE = FileChoice(None, (), None)
GRID_T = np.arange(GRID_S)
REAL_VISITS = ("G01_Ramp", "G09_Hard", "L04_Step", "L05_Ramp")
NIRS_SIGNALS = ("nirs_tsi", "nirs_hbdiff")


# --- synthetic export -----------------------------------------------------------------------------------------

def _frame(tsi: np.ndarray | None = None, n: int = N_SAMPLES) -> pd.DataFrame:
    """Columns as `NirsData.df` (t is set by `_nirs`); O2Hb rises linearly, HHb is flat per channel."""
    i = np.arange(n, dtype=float)
    columns = {}
    for ch, (o2hb, hhb) in CHANNEL_LEVELS.items():
        columns[f"O2Hb{ch}"] = o2hb + O2HB_SLOPE_UM * i
        columns[f"HHb{ch}"] = np.full(n, hhb)
        columns[f"tHb{ch}"] = columns[f"O2Hb{ch}"] + hhb
    tsi_values = np.full(n, PHYSICAL_TSI_PCT) if tsi is None else tsi
    return pd.DataFrame({"t": np.nan, **columns, TSI: tsi_values, FIT_FACTOR: 99.8})


def _nirs(frame: pd.DataFrame, inventory_rate_hz: float = RATE_HZ, event_s: int = EVENT_S,
          event_row: int = EVENT_ROW, sync: str | None = None) -> NirsData:
    """A NirsData as `inventory.sensors.load_nirs` builds it (same t formula, same sync note)."""
    t = np.arange(len(frame)) / inventory_rate_hz - event_row / inventory_rate_hz + event_s
    note = f"event 1 of 1 (latest parsable) = {event_s} s" if sync is None else sync
    return NirsData(frame.assign(t=t), inventory_rate_hz, RATE_HZ, (f"A1 @row{event_row}:1min20",), note, 0)


def _group(nirs: NirsData, rate_hz: float = RATE_HZ, settings: PrepSettings = PrepSettings()) -> GroupColumns:
    return nirs_group(nirs, rate_hz, GRID_T, settings, "G99_Ramp")


def _index_mean_10hz(second: np.ndarray) -> np.ndarray:
    """Mean row index of the 10 Hz samples in [s, s+1): rows 10 s - 500 ... 10 s - 491."""
    return 10.0 * second - 495.5


# --- per-second averaging, header rate, artefact rows ---------------------------------------------------------

def test_second_t_is_the_mean_of_samples_in_t_to_t_plus_1() -> None:
    group = _group(_nirs(_frame(tsi=np.arange(N_SAMPLES, dtype=float))))
    assert group.sensor_group == NIRS_GROUP
    covered = (GRID_T >= FIRST_SECOND) & (GRID_T < FIRST_SECOND + N_SAMPLES / RATE_HZ)
    np.testing.assert_allclose(group.values[TSI_COLUMN][covered], _index_mean_10hz(GRID_T[covered]))
    assert np.isnan(group.values[TSI_COLUMN][~covered]).all()
    np.testing.assert_array_equal(group.present, covered)


def test_header_export_rate_is_used_not_the_inferred_rate() -> None:
    """The inventory inferred 10 Hz; the header says 5 Hz: 5 samples per second, anchored on the marker."""
    group = _group(_nirs(_frame(tsi=np.arange(N_SAMPLES, dtype=float))), rate_hz=5.0)
    start_s = EVENT_S - EVENT_ROW / 5.0  # 20 s
    covered = GRID_T >= start_s
    np.testing.assert_allclose(group.values[TSI_COLUMN][covered], 5.0 * GRID_T[covered] - 98.0)
    assert np.isnan(group.values[TSI_COLUMN][~covered]).all()
    assert group.values[TSI_COLUMN][EVENT_S] == pytest.approx(EVENT_ROW + 2.0)  # the marker row opens second 80


def test_all_zero_rows_and_a_nonphysical_first_row_are_dropped() -> None:
    frame = _frame()
    last, middle = N_SAMPLES - 1, 2000
    for column, value in NONPHYSICAL_ROW.items():
        frame.loc[0, column] = value
    frame.loc[[middle, last], [*NIRS_COLUMNS, TSI, FIT_FACTOR]] = 0.0
    group = _group(_nirs(frame))
    present = group.present
    np.testing.assert_array_equal(group.values[TSI_COLUMN][present], PHYSICAL_TSI_PCT)
    assert present[FIRST_SECOND] and present[int((last - EVENT_ROW) / RATE_HZ + EVENT_S)]
    rows_kept = np.arange(1, 10)  # second 50 = rows 0..9 without row 0
    expected_first = 3.0 + O2HB_SLOPE_UM * rows_kept.mean()
    assert group.values[HBDIFF_COLUMN][FIRST_SECOND] == pytest.approx(expected_first)


def test_only_the_first_row_is_checked_for_nonphysical_values() -> None:
    frame = _frame(n=20)
    assert not artefact_rows(frame).any()
    frame.loc[[0, 5], "O2Hb3"] = 240.0
    np.testing.assert_array_equal(np.flatnonzero(artefact_rows(frame)), [0])


def test_unsynced_export_is_absent() -> None:
    group = _group(_nirs(_frame(), sync="unsynced: no event marker"))
    assert not group.present.any()
    assert np.isnan(group.values[TSI_COLUMN]).all() and np.isnan(group.values[HBDIFF_COLUMN]).all()


def test_unknown_sync_note_raises() -> None:
    with pytest.raises(ValueError, match="sync note"):
        _group(_nirs(_frame(), sync="synced somehow"))


def test_no_nirs_file_means_nirs_absent() -> None:
    visit = VisitData(SimpleNamespace(visit_id="G99_Ramp"), None, None, {}, None, None,
                      Loaded(NO_CHOICE, None, None), None, None, None)
    grid = Grid(GRID_T, np.zeros(GRID_S, dtype=np.int8), np.full(GRID_S, -1, dtype=np.int8))
    group = build_nirs(PrepInput(visit, grid, PrepSettings(), ()))
    assert not group.present.any()
    assert set(group.values) == {TSI_COLUMN, HBDIFF_COLUMN}


def test_input_export_is_not_mutated() -> None:
    nirs = _nirs(_frame())
    before = nirs.df.copy()
    _group(nirs, settings=PrepSettings(hbdiff_baseline=HBDIFF_REST_MEAN))
    pd.testing.assert_frame_equal(nirs.df, before)


# --- Hb-diff flags --------------------------------------------------------------------------------------------

def test_hbdiff_channel_tx3_and_mean3() -> None:
    nirs = _nirs(_frame())
    tx3, mean3 = _group(nirs), _group(nirs, settings=PrepSettings(hbdiff_channel=HBDIFF_MEAN3))
    present = tx3.present
    expected_tx3 = 3.0 + O2HB_SLOPE_UM * _index_mean_10hz(GRID_T[present])
    np.testing.assert_allclose(tx3.values[HBDIFF_COLUMN][present], expected_tx3)
    np.testing.assert_allclose(mean3.values[HBDIFF_COLUMN][present], expected_tx3 + 3.0)  # mean diff 6 vs 3
    np.testing.assert_array_equal(tx3.values[TSI_COLUMN], mean3.values[TSI_COLUMN])


def test_hbdiff_baseline_none_and_rest_mean() -> None:
    nirs = _nirs(_frame())
    raw = _group(nirs)
    rest = _group(nirs, settings=PrepSettings(hbdiff_baseline=HBDIFF_REST_MEAN))
    in_rest = raw.present & (GRID_T < 120)
    assert in_rest.sum() == 120 - FIRST_SECOND
    raw_hbdiff, rest_hbdiff = raw.values[HBDIFF_COLUMN], rest.values[HBDIFF_COLUMN]
    np.testing.assert_allclose(rest_hbdiff[raw.present], raw_hbdiff[raw.present] - raw_hbdiff[in_rest].mean())
    assert rest_hbdiff[in_rest].mean() == pytest.approx(0.0, abs=1e-12)
    assert raw_hbdiff[in_rest].mean() > 1.0  # raw keeps the exported level
    np.testing.assert_array_equal(rest.present, raw.present)


def test_rest_mean_without_rest_data_makes_nirs_absent() -> None:
    late = _nirs(_frame(), event_s=200, event_row=0)  # recording starts at 200 s
    assert _group(late).present.any()
    assert not _group(late, settings=PrepSettings(hbdiff_baseline=HBDIFF_REST_MEAN)).present.any()


def test_unknown_flag_values_raise() -> None:
    with pytest.raises(ValueError, match="hbdiff_channel"):
        PrepSettings(hbdiff_channel="tx1")
    with pytest.raises(ValueError, match="hbdiff_baseline"):
        PrepSettings(hbdiff_baseline="min_max")


@pytest.fixture
def source_roots(tmp_path: Path) -> tuple[Path, Path]:
    roots = (tmp_path / "prepare", tmp_path / "inventory")
    for root in roots:
        root.mkdir()
        (root / "module.py").write_text("X = 1\n", encoding="utf-8")
    return roots


@pytest.mark.parametrize("change", [{"hbdiff_channel": HBDIFF_MEAN3}, {"hbdiff_baseline": HBDIFF_REST_MEAN}])
def test_hbdiff_flags_are_part_of_the_cache_key(source_roots: tuple[Path, Path], change: dict) -> None:
    default = PrepSettings()
    assert (default.hbdiff_channel, default.hbdiff_baseline) == ("tx3", "none")
    assert cache_key(default, source_roots) != cache_key(replace(default, **change), source_roots)


# --- header export rate ---------------------------------------------------------------------------------------

def _write_header(path: Path, rate: object) -> Path:
    rows = [["OxySoft export of:", "P1RampOxy.oxy4"], ["Data file sample rate:", 100],
            ["Export sample rate:", rate], ["Data file duration:", "3232,3"], [None, None], ["Column", "Legend"]]
    pd.DataFrame(rows).to_excel(path, header=False, index=False)
    return path


@pytest.mark.parametrize("cell", [10, "10", "10,0", "10 Hz"])
def test_export_sample_rate_is_read_from_the_header(tmp_path: Path, cell: object) -> None:
    assert read_export_sample_rate(_write_header(tmp_path / "oxy.xlsx", cell)) == 10.0


def test_missing_export_sample_rate_raises(tmp_path: Path) -> None:
    path = tmp_path / "oxy.xlsx"
    pd.DataFrame([["Data file sample rate:", 100]]).to_excel(path, header=False, index=False)
    with pytest.raises(ValueError, match="Export sample rate"):
        read_export_sample_rate(path)


# --- domain registry, fold scalers ----------------------------------------------------------------------------

def test_wr_tsi_and_wr_hbdiff_configurations() -> None:
    tsi, hbdiff = parse_configuration("WR+TSI"), parse_configuration("WR+HBDIFF")
    assert tsi.columns == ("WR", TSI_COLUMN) and hbdiff.columns == ("WR", HBDIFF_COLUMN)
    assert tsi.sensor_groups == hbdiff.sensor_groups == ("WR", NIRS_GROUP)
    assert parse_configuration("WR+HR+VENT+TSI").columns == ("WR", "HR", "HRR", "VE", "BR", TSI_COLUMN)
    assert input_sensor_groups().count(NIRS_GROUP) == 1


def test_tsi_and_hbdiff_are_z_scored(synthetic: Synthetic) -> None:
    mask = shared_mask(synthetic.cache, synthetic.inclusion)
    fold = run_folds(SYNTHETIC_RUN, synthetic.cache)[0]
    columns = (TSI_COLUMN, HBDIFF_COLUMN)
    scalers = build_fold_data(synthetic.cache, mask, fold, columns).scalers
    assert (scalers[TSI_COLUMN].kind, scalers[HBDIFF_COLUMN].kind) == ("zscore", "zscore")


# --- real data ------------------------------------------------------------------------------------------------

@pytest.mark.real
@pytest.mark.parametrize("visit_id", REAL_VISITS)
def test_real_header_rate_matches_the_inferred_rate(visit_id: str) -> None:
    ref = next(r for r in enumerate_visits() if r.visit_id == visit_id)
    path = choose_nirs_file(ref.path).selected
    assert read_export_sample_rate(path) == load_nirs(path).frequency_hz


def _rows(cache: ProcessedCache, visit_id: str) -> slice:
    row = cache.visits.set_index("visit_id").loc[visit_id]
    return slice(int(row["start"]), int(row["stop"]))


@pytest.mark.real
def test_nirs_missing_source_verdicts_agree_with_presence(real_cache: ProcessedCache,
                                                          committed_inclusion: pd.DataFrame) -> None:
    """The CSV says NIRS is missing in a unit iff the processed flag has no present second there."""
    phase, unit = real_cache.column("phase").astype(int), real_cache.column("unit").astype(int)
    present_flag = real_cache.column(presence_column(NIRS_GROUP)) == 1
    rows = committed_inclusion[committed_inclusion["sensor_group"] == NIRS_GROUP]
    disagreements = []
    for verdict in rows.to_dict("records"):
        csv_missing = any(f"{signal} missing" in verdict["reason"] for signal in NIRS_SIGNALS)
        span = _rows(real_cache, verdict["visit_id"])
        in_unit = (phase[span] == EXERCISE) & (unit[span] == UNITS.index(verdict["unit"]))
        present = present_flag[span][in_unit]
        if csv_missing == bool(present.any()):
            disagreements.append(f"{verdict['visit_id']} {verdict['unit']}: CSV missing={csv_missing}, "
                                 f"present seconds={int(present.sum())}")
    assert not rows.empty
    assert not disagreements, "\n".join(disagreements)
