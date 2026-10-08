"""§2.7 breath-by-breath cleaning (exact vectorised port of legacy `HelperFunctions`) and the V̇O₂ target.

Legacy reference: `src2_training_ready/processing/helper_functions.py`
`reject_outliers_pre_local_mean` (z >= 3 vs the previous 5 breaths) + `median_filter` (5 breaths; centred for
V̇O₂, trailing for V̇E/BR). Then linear interpolation at integer seconds.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.prepare.group import GroupColumns, PrepInput

OUTLIER_Z = 3.0
OUTLIER_WINDOW = 5
MEDIAN_WINDOW = 5
VO2_GROUP = "VO2"
VO2_COLUMN = "VO2"


def reject_outliers_pre_local_mean(values: np.ndarray, threshold: float = OUTLIER_Z,
                                   window: int = OUTLIER_WINDOW) -> np.ndarray:
    """NaN where |x_i - mean(x_{i-5..i-1})| / std(x_{i-5..i-1}) >= threshold; the first `window` breaths are kept.

    Same pandas rolling statistics as legacy, so NaN/inf behave identically (0/0 keeps, x/0 rejects).
    """
    series = pd.Series(np.asarray(values, dtype=float))
    previous_mean = series.rolling(window=window).mean().shift(1)
    previous_std = series.rolling(window=window).std().shift(1)
    z = ((series - previous_mean) / previous_std).abs()
    z.iloc[:window] = 0.0
    return series.mask(z >= threshold).to_numpy()


def median_filter(values: np.ndarray, window: int = MEDIAN_WINDOW, center: bool = True) -> np.ndarray:
    return pd.Series(np.asarray(values, dtype=float)).rolling(window=window, center=center).median().to_numpy()


def clean_breaths(values: np.ndarray, center: bool) -> np.ndarray:
    return median_filter(reject_outliers_pre_local_mean(values), center=center)


def interpolate_to_grid(breath_t: np.ndarray, values: np.ndarray, grid_t: np.ndarray) -> np.ndarray:
    """Linear interpolation at integer seconds between finite breaths; NaN outside their span."""
    finite = np.isfinite(breath_t) & np.isfinite(values)
    if not finite.any():
        return np.full(len(grid_t), np.nan)
    return np.interp(grid_t, breath_t[finite], values[finite], left=np.nan, right=np.nan)


def breath_presence(breath_t: np.ndarray, grid_t: np.ndarray, max_gap_s: float) -> np.ndarray:
    """Second t is present if it lies between two raw breaths at most `max_gap_s` apart (or on a breath)."""
    times = np.sort(breath_t[np.isfinite(breath_t)])
    if len(times) == 0:
        return np.zeros(len(grid_t), dtype=bool)
    right = np.searchsorted(times, grid_t, side="left")  # first breath at or after t
    inside = (grid_t >= times[0]) & (grid_t <= times[-1])
    right = np.clip(right, 0, len(times) - 1)
    left = np.clip(right - 1, 0, len(times) - 1)
    on_breath = times[right] == grid_t
    gap = times[right] - times[left]
    return inside & (on_breath | (gap <= max_gap_s))


def build_vo2(inp: PrepInput) -> GroupColumns:
    """Target V̇O₂ (CPET, mL/min): non-finite breaths dropped, §2.7 with a centred median, then interpolated."""
    n = len(inp.grid.t)
    if not inp.visit.cpet.ok:
        return GroupColumns(VO2_GROUP, {VO2_COLUMN: np.full(n, np.nan)}, np.zeros(n, dtype=bool))
    cpet = inp.visit.cpet.data.df
    breaths = cpet.loc[np.isfinite(cpet["VO2"].to_numpy(dtype=float)), ["t", "VO2"]]
    breath_t = breaths["t"].to_numpy(dtype=float)
    cleaned = clean_breaths(breaths["VO2"].to_numpy(dtype=float), center=True)
    vo2 = interpolate_to_grid(breath_t, cleaned, inp.grid.t)
    present = breath_presence(breath_t, inp.grid.t, inp.settings.max_interp_gap_s) & np.isfinite(vo2)
    return GroupColumns(VO2_GROUP, {VO2_COLUMN: vo2}, present)
