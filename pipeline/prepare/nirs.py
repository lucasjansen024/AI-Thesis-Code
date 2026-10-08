"""NIRS domain: TSI and Hb-diff on the grid, from the PortaMon export loaded by `inventory.sensors.load_nirs`.

- TSI: raw TSI %. Hb-diff: O2Hb - HHb of channel Tx3 (`hbdiff_channel="tx3"`) or the mean over channels 1-3
  (`"mean3"`), raw as exported (`hbdiff_baseline="none"`) or minus the visit's 0-120 s mean (`"rest_mean"`).
  No occlusion, no within-visit min-max; the fit factor, Lucas's native `HbDiff` and the short-separation
  channel are never used.
- Sync: the inventory's latest parsable `XminYY` marker, no manual corrections. Sample times are recomputed from
  that marker with the header's "Export sample rate" (the inventory infers the rate from rows / duration).
- Loader fixes, applied here because `inventory/` stays untouched:
  - all-zero rows (every O2Hb/HHb/tHb channel exactly 0; Gies exports often end with one) are dropped;
  - a nonphysical first row is dropped. The legacy loader's `iloc[2:]` already removes the export's first data
    row, which is the nonphysical one in Gies, so this check normally finds nothing; it guards that coupling;
  - duplicate NIRS columns: `inventory.sensors._column` already keeps the fullest one.
- 10 Hz -> 1 Hz: grid second t = mean of the samples in [t, t+1). No interpolation: a second without a finite
  sample is NaN. NIRS (TSI and Hb-diff together) is present where both 1 Hz values are finite.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from inventory.sensors import NIRS_CHANNELS, NIRS_COLUMNS, TSI, NirsData
from pipeline.prepare.group import GroupColumns, PrepInput
from pipeline.prepare.settings import HBDIFF_REST_MEAN, HBDIFF_TX3, PrepSettings

logger = logging.getLogger(__name__)

NIRS_GROUP = "NIRS"
TSI_COLUMN, HBDIFF_COLUMN = "TSI", "HBDIFF"
NIRS_FEATURE_COLUMNS = (TSI_COLUMN, HBDIFF_COLUMN)
TX3_CHANNEL = 3
HBDIFF_REST_WINDOW_S = 120  # MAPPING §7: rest_mean subtracts the same visit's 0-120 s mean
DHB_COLUMNS = tuple(f"{kind}{ch}" for ch in NIRS_CHANNELS for kind in ("O2Hb", "HHb"))
# Gies's nonphysical first export row has O2Hb ~195-245 µM; real ΔHb stays within tens of µM of its zero.
NONPHYSICAL_DHB_UM = 150.0
EXPORT_RATE_LABEL = "export sample rate"
HEADER_SCAN_ROWS = 100  # the OxySoft header block is far shorter; the data table follows it
NUMBER = re.compile(r"\d+(?:[.,]\d+)?")
# Format of `inventory.sensors._nirs_time`'s sync note; a mismatch raises rather than guessing.
SYNC_NOTE = re.compile(r"event \d+ of \d+ \(latest parsable\) = (-?\d+) s")
UNSYNCED_PREFIX = "unsynced"
ROW_TOLERANCE = 1e-3  # rows; the inventory's t is exact up to float rounding


@dataclass(frozen=True)
class SyncAnchor:
    row: int  # row of `NirsData.df` holding the sync marker
    event_s: float  # its CPET time


def read_export_sample_rate(path: Path) -> float:
    """The header's "Export sample rate" (Hz): the rate of the exported table, not of the device."""
    header = pd.read_excel(path, header=None, nrows=HEADER_SCAN_ROWS, usecols=[0, 1])
    labels = header.iloc[:, 0].astype(str).str.strip().str.lower()
    hits = header.index[labels.str.contains(EXPORT_RATE_LABEL, regex=False)]
    if len(hits) == 0:
        raise ValueError(f"{path.name}: no 'Export sample rate' in the first {HEADER_SCAN_ROWS} header rows")
    value = header.iat[hits[0], 1]
    match = NUMBER.search(str(value))
    rate = float(match.group(0).replace(",", ".")) if match else float("nan")
    if not rate > 0:
        raise ValueError(f"{path.name}: unreadable export sample rate {value!r}")
    return rate


def sync_anchor(nirs: NirsData) -> SyncAnchor | None:
    """The marker the inventory synced on (row, CPET seconds), or None if the export is unsynced."""
    if nirs.sync.startswith(UNSYNCED_PREFIX) or nirs.df.empty:
        return None
    match = SYNC_NOTE.fullmatch(nirs.sync)
    if match is None:
        raise ValueError(f"unrecognised NIRS sync note {nirs.sync!r}")
    event_s = float(match.group(1))
    row_float = (event_s - float(nirs.df["t"].iloc[0])) * nirs.frequency_hz
    row = round(row_float)
    if abs(row_float - row) > ROW_TOLERANCE or not 0 <= row < len(nirs.df):
        raise ValueError(f"NIRS sync {nirs.sync!r} does not fall on a row (row {row_float:.4f} of {len(nirs.df)})")
    return SyncAnchor(row, event_s)


def sample_times(n_samples: int, rate_hz: float, anchor: SyncAnchor) -> np.ndarray:
    """CPET time of every row; integer arithmetic before the division keeps whole seconds exact."""
    return (np.arange(n_samples) - anchor.row) / rate_hz + anchor.event_s


def artefact_rows(df: pd.DataFrame) -> np.ndarray:
    """Bool per row: all-zero rows, and a first row whose ΔHb is nonphysical."""
    all_zero = (df[list(NIRS_COLUMNS)] == 0).all(axis=1).to_numpy()
    first = np.arange(len(df)) == 0
    nonphysical = (df[list(DHB_COLUMNS)].abs() > NONPHYSICAL_DHB_UM).any(axis=1).to_numpy()
    return all_zero | (first & nonphysical)


def hbdiff_samples(df: pd.DataFrame, channel: str) -> np.ndarray:
    """O2Hb - HHb per sample: channel Tx3, or the mean over channels 1-3 (NaN if any channel is NaN)."""
    channels = (TX3_CHANNEL,) if channel == HBDIFF_TX3 else NIRS_CHANNELS
    diffs = [df[f"O2Hb{ch}"].to_numpy(dtype=float) - df[f"HHb{ch}"].to_numpy(dtype=float) for ch in channels]
    return np.column_stack(diffs).mean(axis=1)


def per_second_mean(sample_t: np.ndarray, values: np.ndarray, grid_t: np.ndarray) -> np.ndarray:
    """Grid second t (contiguous integer grid) = mean of the finite samples in [t, t+1); NaN where none."""
    finite = np.isfinite(sample_t) & np.isfinite(values)
    index = np.floor(sample_t[finite]).astype(np.int64) - int(grid_t[0])
    inside = (index >= 0) & (index < len(grid_t))
    counts = np.bincount(index[inside], minlength=len(grid_t))
    sums = np.bincount(index[inside], weights=values[finite][inside], minlength=len(grid_t))
    return np.divide(sums, counts, out=np.full(len(grid_t), np.nan), where=counts > 0)


def _absent(n: int) -> GroupColumns:
    return GroupColumns(NIRS_GROUP, {c: np.full(n, np.nan) for c in NIRS_FEATURE_COLUMNS}, np.zeros(n, dtype=bool))


def nirs_group(nirs: NirsData, export_rate_hz: float, grid_t: np.ndarray, settings: PrepSettings,
               visit_id: str) -> GroupColumns:
    """TSI and Hb-diff on the grid from one loaded export (pure: the header rate is passed in)."""
    anchor = sync_anchor(nirs)
    if anchor is None:
        logger.warning("%s: NIRS not synced (%s, %d rows); NIRS absent", visit_id, nirs.sync, len(nirs.df))
        return _absent(len(grid_t))
    df = nirs.df
    keep = ~artefact_rows(df)
    if not keep.all():
        logger.info("%s: dropped %d NIRS artefact rows", visit_id, int((~keep).sum()))
    t = sample_times(len(df), export_rate_hz, anchor)[keep]
    tsi = per_second_mean(t, df[TSI].to_numpy(dtype=float)[keep], grid_t)
    hbdiff = per_second_mean(t, hbdiff_samples(df, settings.hbdiff_channel)[keep], grid_t)
    present = np.isfinite(tsi) & np.isfinite(hbdiff)
    if settings.hbdiff_baseline == HBDIFF_REST_MEAN:
        in_rest = present & (grid_t < HBDIFF_REST_WINDOW_S)
        if not in_rest.any():
            logger.warning("%s: no NIRS in the first %d s; rest-mean Hb-diff undefined, NIRS absent",
                           visit_id, HBDIFF_REST_WINDOW_S)
            return _absent(len(grid_t))
        hbdiff = hbdiff - hbdiff[in_rest].mean()
    return GroupColumns(NIRS_GROUP, {TSI_COLUMN: tsi, HBDIFF_COLUMN: hbdiff}, present)


def build_nirs(inp: PrepInput) -> GroupColumns:
    nirs = inp.visit.nirs
    if not nirs.ok:
        return _absent(len(inp.grid.t))
    visit_id = inp.visit.ref.visit_id
    rate = read_export_sample_rate(nirs.choice.selected)
    if rate != nirs.data.frequency_hz:
        logger.warning("%s: NIRS header export rate %g Hz differs from the inferred %g Hz; using the header",
                       visit_id, rate, nirs.data.frequency_hz)
    return nirs_group(nirs.data, rate, inp.grid.t, inp.settings, visit_id)
