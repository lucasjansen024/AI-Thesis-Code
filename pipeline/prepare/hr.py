"""HR domain: HR and HRR on the grid, from the visit's HR source (Tymewear or the CPET chest strap).

- Source per visit = `hr_source` on the HR rows of `inventory/inclusion/inclusion.csv` (Tymewear if it passed
  the HR check, else CPET; empty = no usable source, so the HR group is never present).
- HR: no smoothing; linear interpolation at integer seconds; present where the source has samples no more than
  `max_interp_gap_s` apart.
- HRR = (HR - HRrest) / (HRpeak - HRrest). HRrest = mean HR of the same source over the present grid seconds
  in [0, hrrest_window_s). HRpeak = 208 - 0.7 x age (Tanaka), age from this visit's CPET header, full precision
  (or from the visit named by a `prep_overrides.csv` age_from_visit override).
"""
from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd

from inventory.build_inclusion import INCLUSION_CSV
from inventory.sensors import VisitData
from pipeline.prepare.breath import breath_presence, interpolate_to_grid
from pipeline.prepare.group import GroupColumns, PrepInput

logger = logging.getLogger(__name__)

HR_GROUP = "HR"
HR_COLUMN, HRR_COLUMN = "HR", "HRR"
TYMEWEAR, CPET, NO_SOURCE = "tymewear", "cpet", ""
HR_SOURCES = (TYMEWEAR, CPET)
TANAKA_INTERCEPT_BPM = 208.0
TANAKA_SLOPE_BPM_PER_YEAR = 0.7


def tanaka_hr_peak(age_years: float) -> float:
    return TANAKA_INTERCEPT_BPM - TANAKA_SLOPE_BPM_PER_YEAR * age_years


def heart_rate_reserve(hr: np.ndarray, hr_rest: float, hr_peak: float) -> np.ndarray:
    if not hr_peak > hr_rest:
        raise ValueError(f"HRpeak {hr_peak:.1f} bpm is not above HRrest {hr_rest:.1f} bpm")
    return (np.asarray(hr, dtype=float) - hr_rest) / (hr_peak - hr_rest)


def read_hr_sources(path: Path = INCLUSION_CSV) -> dict[str, str]:
    """visit_id -> HR source from the HR rows of the inclusion CSV. One source per visit (all units agree)."""
    table = pd.read_csv(path, dtype=str, keep_default_na=False, usecols=["visit_id", "sensor_group", "hr_source"])
    rows = table[table["sensor_group"] == HR_GROUP]
    unknown = sorted(set(rows["hr_source"]) - {*HR_SOURCES, NO_SOURCE})
    if unknown:
        raise ValueError(f"{path.name}: unknown hr_source values {unknown}")
    per_visit = rows.groupby("visit_id", sort=True)["hr_source"].unique()
    conflicting = sorted(v for v, sources in per_visit.items() if len(sources) > 1)
    if conflicting:
        raise ValueError(f"{path.name}: units of one visit name different HR sources: {conflicting}")
    return {visit_id: str(sources[0]) for visit_id, sources in per_visit.items()}


def _source_samples(visit: VisitData, source: str) -> tuple[np.ndarray, np.ndarray] | None:
    """(t, HR) of the source's finite samples on the CPET clock, or None if the source did not load."""
    if source == TYMEWEAR:
        if not visit.tymewear.ok:
            return None
        frame = visit.tymewear.data.series
    else:
        if not visit.cpet.ok or "HR" not in visit.cpet.data.df.columns:
            return None
        frame = visit.cpet.data.df
    t, hr = frame["t"].to_numpy(dtype=float), frame["HR"].to_numpy(dtype=float)
    finite = np.isfinite(t) & np.isfinite(hr)
    return t[finite], hr[finite]


def visit_age(visit: VisitData) -> float:
    """Age in years from this visit's CPET header (needed for HRpeak; present in every visit)."""
    age = visit.cpet.data.header.get("age") if visit.cpet.ok else None
    try:
        years = float(age)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{visit.ref.visit_id}: no numeric age in the CPET header ({age!r})") from exc
    if not math.isfinite(years) or years <= 0:
        raise ValueError(f"{visit.ref.visit_id}: implausible CPET header age {years}")
    return years


def build_hr(inp: PrepInput) -> GroupColumns:
    n = len(inp.grid.t)
    absent = GroupColumns(HR_GROUP, {HR_COLUMN: np.full(n, np.nan), HRR_COLUMN: np.full(n, np.nan)},
                          np.zeros(n, dtype=bool))
    visit_id = inp.visit.ref.visit_id
    if inp.hr_source == NO_SOURCE:
        return absent
    if inp.hr_source not in HR_SOURCES:
        raise ValueError(f"{visit_id}: unknown HR source {inp.hr_source!r}")
    samples = _source_samples(inp.visit, inp.hr_source)
    if samples is None:
        logger.warning("%s: inclusion.csv names HR source %r but it did not load; HR absent", visit_id, inp.hr_source)
        return absent
    sample_t, sample_hr = samples
    hr = interpolate_to_grid(sample_t, sample_hr, inp.grid.t)
    present = breath_presence(sample_t, inp.grid.t, inp.settings.max_interp_gap_s) & np.isfinite(hr)
    in_rest = present & (inp.grid.t < inp.settings.hrrest_window_s)
    if not in_rest.any():
        logger.warning("%s: no %s HR in the first %d s; HRR undefined, HR absent",
                       visit_id, inp.hr_source, inp.settings.hrrest_window_s)
        return absent
    age = inp.age_years if math.isfinite(inp.age_years) else visit_age(inp.visit)
    hrr = heart_rate_reserve(hr, float(hr[in_rest].mean()), tanaka_hr_peak(age))
    return GroupColumns(HR_GROUP, {HR_COLUMN: hr, HRR_COLUMN: hrr}, present)
