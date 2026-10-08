"""VENT domain: Tymewear V̇E and BR on the grid, from the raw breath-by-breath columns.

- Source: `TymewearData.breaths` (`VE breath by breath`, `BR breath by breath`), never Tymewear's smoothed
  columns and never CPET V̇E/Rf (same flow signal as the V̇O₂ target).
- Sync is done by the inventory loader: Gies breath t = Tymewear t - `VE_lag`; Lucas is already on the CPET clock.
- Per feature: non-finite breaths dropped, §2.7 with a **trailing** 5-breath median, linear interpolation at
  integer seconds. VENT is present where both features have raw breaths no more than `max_interp_gap_s` apart
  and both interpolated values are finite.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.prepare.breath import breath_presence, clean_breaths, interpolate_to_grid
from pipeline.prepare.group import GroupColumns, PrepInput

VENT_GROUP = "VENT"
VE_COLUMN, BR_COLUMN = "VE", "BR"
VENT_COLUMNS = (VE_COLUMN, BR_COLUMN)


def finite_breaths(breaths: pd.DataFrame, column: str) -> tuple[np.ndarray, np.ndarray]:
    """(t, value) of the breaths where both are finite, in time order (stable: equal times keep file order)."""
    t, values = breaths["t"].to_numpy(dtype=float), breaths[column].to_numpy(dtype=float)
    finite = np.isfinite(t) & np.isfinite(values)
    order = np.argsort(t[finite], kind="stable")
    return t[finite][order], values[finite][order]


def _clean_on_grid(breaths: pd.DataFrame, column: str, inp: PrepInput) -> tuple[np.ndarray, np.ndarray]:
    """(feature on the grid, presence of its raw breaths) for one breath column."""
    breath_t, values = finite_breaths(breaths, column)
    cleaned = clean_breaths(values, center=False)
    on_grid = interpolate_to_grid(breath_t, cleaned, inp.grid.t)
    return on_grid, breath_presence(breath_t, inp.grid.t, inp.settings.max_interp_gap_s)


def build_vent(inp: PrepInput) -> GroupColumns:
    n = len(inp.grid.t)
    if not inp.visit.tymewear.ok:
        return GroupColumns(VENT_GROUP, {c: np.full(n, np.nan) for c in VENT_COLUMNS}, np.zeros(n, dtype=bool))
    breaths = inp.visit.tymewear.data.breaths
    ve, ve_present = _clean_on_grid(breaths, VE_COLUMN, inp)
    br, br_present = _clean_on_grid(breaths, BR_COLUMN, inp)
    present = ve_present & br_present & np.isfinite(ve) & np.isfinite(br)
    return GroupColumns(VENT_GROUP, {VE_COLUMN: ve, BR_COLUMN: br}, present)
