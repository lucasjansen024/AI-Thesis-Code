"""The contract between `visit.py` and the per-sensor-group builders (`breath.py`, `wr.py`, later hr/nirs/anthro).

A builder takes a `PrepInput` and returns a `GroupColumns`: its feature columns on the grid (NaN where undefined)
plus one presence flag per second for its sensor group. Register new builders in `visit.GROUP_BUILDERS`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from inventory.catalog import VisitRef
from inventory.sensors import VisitData
from pipeline.prepare.grid import Grid
from pipeline.prepare.settings import PrepSettings

PRESENCE_PREFIX = "present_"


def presence_column(sensor_group: str) -> str:
    return f"{PRESENCE_PREFIX}{sensor_group}"


@dataclass(frozen=True)
class PrepInput:
    visit: VisitData
    grid: Grid
    settings: PrepSettings
    refs: tuple[VisitRef, ...]  # every visit folder (incl. superseded), for sibling-file lookups
    hr_source: str = ""  # "tymewear" | "cpet" | "" (none), from inclusion.csv's HR rows
    smm_kg: float = math.nan  # resolved by `anthro.candidate_smm` per `settings.smm`; NaN = static group absent
    age_years: float = math.nan  # HRpeak age override (`prep_overrides.csv` age_from_visit); NaN = own CPET header


@dataclass(frozen=True)
class GroupColumns:
    sensor_group: str  # as in inclusion.csv: VO2, WR, HR, VENT, NIRS, static
    values: dict[str, np.ndarray]  # feature column -> float array on the grid
    present: np.ndarray  # bool per grid second: raw data present, no gap > max_interp_gap_s, value finite

    def __post_init__(self) -> None:
        lengths = {len(v) for v in self.values.values()} | {len(self.present)}
        if len(lengths) != 1:
            raise ValueError(f"{self.sensor_group}: columns of unequal length {sorted(lengths)}")
        for name, values in self.values.items():
            present_but_nan = self.present & ~np.isfinite(values)
            if present_but_nan.any():
                raise ValueError(f"{self.sensor_group}: {name} is NaN at {int(present_but_nan.sum())} present seconds")
