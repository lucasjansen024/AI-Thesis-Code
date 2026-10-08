"""Read the committed inclusion CSV (hand-written schema check) and build the target mask.

Verdicts come only from `inventory/inclusion/inclusion.csv`; the pipeline never recomputes validity.
Target mask: second t of a visit is a target iff (1) phase is exercise, (2) its unit is valid for the required
unit groups (shared mask: every sensor group), (3) V̇O₂ is present at t, and (4) every required input group is
present on all of t-187 ... t-1.
`per_config` mask (sensitivity runs): unit valid for VO₂ + the configuration's own groups (static is one of them
only with ANTHRO), and only the configuration's input groups must be present in the context.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from inventory.build_inclusion import INCLUSION_CSV, OUTPUT_COLUMNS, SENSOR_GROUPS, units_of
from inventory.catalog import COHORT_PREFIX, NOMINAL_TIMELINES
from pipeline.cache import ProcessedCache
from pipeline.domains import TARGET_GROUP, Configuration, input_sensor_groups
from pipeline.prepare.grid import EXERCISE, NO_UNIT, UNITS
from pipeline.prepare.group import presence_column

WINDOW_S = 187  # §2.4: the window for target t is t-187 ... t-1 (= the TCN receptive field)
BOOLEANS = {"True": True, "False": False}
DECIDED_BY = ("rule", "review")
HR_SOURCES = ("", "tymewear", "cpet")
HR_GROUP = "HR"
PARTICIPANT_ID = re.compile(r"[A-Z]\d{2}")
MAX_LISTED_ERRORS = 20
KEY = ["visit_id", "unit", "sensor_group"]


class InclusionSchemaError(ValueError):
    """The inclusion CSV does not match the expected schema or misses cached visits."""


def _raise_if(errors: list[str], source: str) -> None:
    if errors:
        listed = errors[:MAX_LISTED_ERRORS]
        more = f" (+{len(errors) - len(listed)} more)" if len(errors) > len(listed) else ""
        raise InclusionSchemaError(f"{source}: " + " | ".join(listed) + more)


def _row_errors(row: dict, line: int) -> list[str]:
    errors = []
    cohort, protocol = row["cohort"], row["protocol"]
    if cohort not in COHORT_PREFIX:
        return [f"line {line}: cohort {cohort!r}"]
    if (cohort, protocol) not in NOMINAL_TIMELINES:
        return [f"line {line}: protocol {protocol!r} for {cohort}"]
    if not PARTICIPANT_ID.fullmatch(row["participant_id"]) or row["participant_id"][0] != COHORT_PREFIX[cohort]:
        errors.append(f"line {line}: participant_id {row['participant_id']!r} for {cohort}")
    if row["visit_id"] != f"{row['participant_id']}_{protocol}":
        errors.append(f"line {line}: visit_id {row['visit_id']!r}")
    if row["unit"] not in units_of(cohort, protocol):
        errors.append(f"line {line}: unit {row['unit']!r} for {cohort} {protocol}")
    if row["sensor_group"] not in SENSOR_GROUPS:
        errors.append(f"line {line}: sensor_group {row['sensor_group']!r}")
    if row["valid"] not in BOOLEANS:
        errors.append(f"line {line}: valid {row['valid']!r}")
    if row["decided_by"] not in DECIDED_BY:
        errors.append(f"line {line}: decided_by {row['decided_by']!r}")
    if row["hr_source"] not in HR_SOURCES or (row["hr_source"] and row["sensor_group"] != HR_GROUP):
        errors.append(f"line {line}: hr_source {row['hr_source']!r} on a {row['sensor_group']} row")
    return errors


def _grid_errors(table: pd.DataFrame) -> list[str]:
    """Every visit in the table has each of its units x every sensor group exactly once."""
    errors = [f"duplicate row {tuple(k)}" for k in table.loc[table.duplicated(KEY), KEY].to_numpy()]
    for visit_id, rows in table.groupby("visit_id", sort=False):
        cohort, protocol = rows["cohort"].iloc[0], rows["protocol"].iloc[0]
        expected = {(u, g) for u in units_of(cohort, protocol) for g in SENSOR_GROUPS}
        missing = expected - set(zip(rows["unit"], rows["sensor_group"]))
        if missing:
            errors.append(f"{visit_id}: missing unit x group {sorted(missing)}")
        if rows[["participant_id", "cohort", "protocol"]].nunique().max() > 1:
            errors.append(f"{visit_id}: inconsistent participant/cohort/protocol")
    return errors


def check_schema(raw: pd.DataFrame, source: str = "inclusion.csv") -> pd.DataFrame:
    """Validate a string-typed table; return it with `valid` as bool."""
    if list(raw.columns) != OUTPUT_COLUMNS:
        raise InclusionSchemaError(f"{source}: columns {list(raw.columns)}, expected {OUTPUT_COLUMNS}")
    errors = [e for line, row in enumerate(raw.to_dict("records"), start=2) for e in _row_errors(row, line)]
    _raise_if(errors, source)
    table = raw.assign(valid=raw["valid"].map(BOOLEANS))
    _raise_if(_grid_errors(table), source)
    return table


def read_inclusion(path: Path = INCLUSION_CSV) -> pd.DataFrame:
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    return check_schema(raw, path.name)


def check_cached_visits(inclusion: pd.DataFrame, visits: pd.DataFrame) -> None:
    """Every cached visit has its complete unit x group grid in the CSV, with the same IDs."""
    known = inclusion.drop_duplicates("visit_id").set_index("visit_id")
    errors = []
    for visit in visits.to_dict("records"):
        if visit["visit_id"] not in known.index:
            errors.append(f"{visit['visit_id']}: cached but not in the CSV")
            continue
        listed = known.loc[visit["visit_id"]]
        for field in ("participant_id", "cohort", "protocol"):
            if listed[field] != visit[field]:
                errors.append(f"{visit['visit_id']}: {field} {listed[field]!r} in CSV, {visit[field]!r} cached")
    _raise_if(errors, "inclusion.csv vs cache")


def valid_units(inclusion: pd.DataFrame, groups: Iterable[str]) -> set[tuple[str, str]]:
    """(visit_id, unit) pairs valid for every listed sensor group."""
    groups = tuple(groups)
    unknown = set(groups) - set(SENSOR_GROUPS)
    if unknown:
        raise ValueError(f"unknown sensor groups {sorted(unknown)}")
    rows = inclusion[inclusion["sensor_group"].isin(groups)]
    unit_ok = rows.groupby(["visit_id", "unit"], sort=False)["valid"].all()
    return {key for key, ok in unit_ok.items() if ok}


def context_present(present: np.ndarray, starts: np.ndarray, window: int) -> np.ndarray:
    """For each row r: all of present[r-window : r] within the row's visit (False if the window leaves it)."""
    cumulative = np.concatenate([[0], np.cumsum(present, dtype=np.int64)])
    rows = np.arange(len(present))
    first = rows - window
    inside = first >= starts
    counts = cumulative[rows] - cumulative[np.clip(first, 0, None)]
    return inside & (counts == window)


def _unit_valid(cache: ProcessedCache, units: set[tuple[str, str]]) -> np.ndarray:
    """Per row: its unit is in `units`. Rows outside exercise have no unit and are never valid."""
    lookup = np.array([[(v, u) in units for u in UNITS] for v in cache.visits["visit_id"]], dtype=bool)
    unit_code = cache.column("unit").astype(np.int64)
    has_unit = unit_code != NO_UNIT
    valid = np.zeros(cache.n_rows, dtype=bool)
    valid[has_unit] = lookup[cache.row_visit()[has_unit], unit_code[has_unit]]
    return valid


def target_mask(cache: ProcessedCache, inclusion: pd.DataFrame, unit_groups: Iterable[str],
                input_groups: Iterable[str], window: int = WINDOW_S) -> np.ndarray:
    """Boolean per cache row: is this second a target? (rules 1-4 in the module docstring)."""
    check_cached_visits(inclusion, cache.visits)
    starts = cache.visits["start"].to_numpy()[cache.row_visit()]
    mask = cache.column("phase").astype(np.int64) == EXERCISE
    mask &= _unit_valid(cache, valid_units(inclusion, unit_groups))
    mask &= cache.column(presence_column(TARGET_GROUP)) == 1
    for group in input_groups:
        mask &= context_present(cache.column(presence_column(group)) == 1, starts, window)
    return mask


def shared_mask(cache: ProcessedCache, inclusion: pd.DataFrame, window: int = WINDOW_S) -> np.ndarray:
    """Shared mask: unit valid for every sensor group; every registered input group present in the context."""
    return target_mask(cache, inclusion, SENSOR_GROUPS, input_sensor_groups(), window)


def config_mask(cache: ProcessedCache, inclusion: pd.DataFrame, config: Configuration,
                window: int = WINDOW_S) -> np.ndarray:
    """`per_config` mask: unit valid for VO₂ + the configuration's own groups; context needs only its input groups."""
    groups = config.sensor_groups
    return target_mask(cache, inclusion, (TARGET_GROUP, *groups), groups, window)


def per_config_masks(cache: ProcessedCache, inclusion: pd.DataFrame, configurations: Iterable[Configuration],
                     window: int = WINDOW_S) -> dict[str, np.ndarray]:
    """`{config_id: mask}`; configurations with the same sensor groups share one computed (read-only) array."""
    by_groups: dict[tuple[str, ...], np.ndarray] = {}
    masks = {}
    for config in configurations:
        if config.sensor_groups not in by_groups:
            by_groups[config.sensor_groups] = config_mask(cache, inclusion, config, window)
        masks[config.config_id] = by_groups[config.sensor_groups]
    return masks
