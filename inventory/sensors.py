"""Load one visit's sensors through the existing loaders and put them on the CPET clock."""
from __future__ import annotations

import contextlib
import io
import logging
import math
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Iterator
from xml.etree import ElementTree

import numpy as np
import pandas as pd

from inventory.annotations import Annotation
from inventory.catalog import (
    NOMINAL_TIMELINES,
    FileChoice,
    Timeline,
    VisitRef,
    choose_cpet_file,
    choose_fitdays_file,
    choose_nirs_file,
    choose_protocol_xml,
    choose_tymewear_file,
    reference_prepared_file,
)
from src2_training_ready.loaders.NIRS_loader import NIRSLoader
from src2_training_ready.loaders.VE_loader import VELoader
from src2_training_ready.loaders.VO2_loader import VO2Loader

logger = logging.getLogger(__name__)

CPET_COLUMNS = ("t", "VO2", "VE", "Rf", "HR", "Power", "RPM", "Phase")
# Cells of the CPET header, as read by Preparer._create_fit_files.
CPET_HEADER_CELLS = {"sex": (2, 1), "age": (3, 1), "height": (4, 1), "dob": (6, 1), "protocol": (11, 4)}
NIRS_HEADER_FREQUENCY_CELL = (3, 1)
NIRS_CHANNELS = (1, 2, 3)
NIRS_COLUMNS = tuple(f"{kind}{ch}" for ch in NIRS_CHANNELS for kind in ("O2Hb", "HHb", "tHb"))
TSI, FIT_FACTOR = "TSI%", "TSI Fit Factor"
RAMP_MIN_CADENCE_RPM = 60
WARMUP_PHASES = {"WARMUP", "ERWÄRMUNG", "UNLOADED PEDALLING"}
EXERCISE_PHASES = {"EXERCISE", "BELASTUNG"}
FITDAYS_MARKER_COLUMNS = {"Date", "Datum", "Weight", "Gewicht"}
SECONDS_PER_MINUTE = 60
RAMP_WARMUP_W = 30.0  # both cohorts (inventory: measured warm-up 30 W in every ramp)
# Ergometer step size per ramp rate (W/min), read off measured power; interval = step / rate.
RAMP_STEP_W = {20: 1.0, 25: 5.0, 30: 1.0, 35: 7.0}
# Protocols whose step differs from the default for their rate (dev: L04 ran 4 W every 12 s).
RAMP_STEP_W_BY_PROTOCOL = {"VO2Ramp20W": 4.0}
RAMP_RATE_IN_NAME = re.compile(r"(?<!\d)(\d{2})\s*_?w", re.IGNORECASE)  # "30WVO2EstimationRamp", "VO2max_25Watt"


@dataclass(frozen=True)
class Loaded:
    """One sensor's load attempt: the file choice, the data or the error."""

    choice: FileChoice
    data: object | None
    error: str | None

    @property
    def ok(self) -> bool:
        return self.data is not None


@dataclass(frozen=True)
class CpetData:
    df: pd.DataFrame  # CPET_COLUMNS that exist, one row per breath
    header: dict  # sex, age, height, dob; dob stays in memory, never written out


@dataclass(frozen=True)
class TymewearData:
    series: pd.DataFrame  # t, HR, VE, BR at ~1 Hz, CPET clock
    breaths: pd.DataFrame  # t, VE, BR per breath, CPET clock
    lag_s: float
    lag_note: str


@dataclass(frozen=True)
class NirsData:
    df: pd.DataFrame  # t (CPET clock) + NIRS_COLUMNS + TSI + FIT_FACTOR
    frequency_hz: float
    header_frequency: object
    events: tuple[str, ...]
    sync: str
    n_all_zero_rows: int  # Gies exports often end with one; it looks like data


@dataclass(frozen=True)
class VisitData:
    ref: VisitRef
    annotation: Annotation
    timeline: Timeline
    cpet_phase_starts: dict
    cpet: Loaded
    tymewear: Loaded
    nirs: Loaded
    protocol: Loaded
    reference_wr: Loaded
    fitdays: Loaded


@contextlib.contextmanager
def _quiet() -> Iterator[None]:
    """The legacy loaders print every column; keep the console readable. Exceptions still propagate."""
    with contextlib.redirect_stdout(io.StringIO()):
        yield


def _attempt(label: str, choice: FileChoice, load: Callable[[Path], object]) -> Loaded:
    if choice.selected is None:
        return Loaded(choice, None, None)
    try:
        return Loaded(choice, load(choice.selected), None)
    except Exception as exc:  # safety net: one bad file must not stop the inventory; logged and recorded
        logger.exception("%s: could not load %s", label, choice.selected)
        return Loaded(choice, None, f"{type(exc).__name__}: {exc}")


def _cell(df: pd.DataFrame, row: int, col: int) -> object:
    if row >= df.shape[0] or col >= df.shape[1]:
        return None
    value = df.iat[row, col]
    return None if pd.isna(value) else value


def _column(df: pd.DataFrame, name: str) -> pd.Series:
    if name not in df.columns:
        return pd.Series(np.nan, index=df.index)
    selected = df[name]
    if isinstance(selected, pd.DataFrame):  # duplicate column names in the export (e.g. L04 Step NIRS)
        candidates = [pd.to_numeric(selected.iloc[:, i], errors="coerce") for i in range(selected.shape[1])]
        counts = [int(c.notna().sum()) for c in candidates]
        logger.warning("column %r appears %d times (numeric values: %s); using the fullest",
                       name, len(candidates), counts)
        return candidates[counts.index(max(counts))]
    return pd.to_numeric(selected, errors="coerce")


def load_cpet(path: Path) -> CpetData:
    with _quiet():
        loader = VO2Loader(path)
    df = loader.df.loc[:, ~loader.df.columns.duplicated()]
    kept = df[[c for c in CPET_COLUMNS if c in df.columns]]
    kept = kept.dropna(subset=["t"]).sort_values("t").reset_index(drop=True)
    header = {key: _cell(loader.complete_df, r, c) for key, (r, c) in CPET_HEADER_CELLS.items()}
    return CpetData(kept, header)


def load_tymewear(path: Path, lag_s: float, lag_note: str) -> TymewearData:
    with _quiet():
        loader = VELoader(path)
    ts, bxb = loader.ve_prepared_df, loader.bxb_df
    series = pd.DataFrame({
        "t": _column(ts, "Time") - lag_s,
        "HR": _column(ts, "HR"),
        "VE": _column(ts, "VE"),
        "BR": _column(ts, "BR"),
    })
    breaths = pd.DataFrame({
        "t": _column(bxb, "t") - lag_s,
        "VE": _column(bxb, "VE breath by breath"),
        "BR": _column(bxb, "BR breath by breath"),
    })
    return TymewearData(series, breaths, lag_s, lag_note)


def _event_time_text(df: pd.DataFrame, row: int) -> str:
    cell = df.loc[row, "Event Time"] if "Event Time" in df.columns else None
    if isinstance(cell, pd.Series):  # duplicate "Event Time" columns: take the first filled one
        cell = cell.dropna().iloc[0] if cell.notna().any() else None
    return "" if cell is None or pd.isna(cell) else str(cell)


def _parse_event_seconds(text: str) -> int:
    """Same rule as Preparer._NIRS_event_to_sec: '<m>min<s>' -> seconds."""
    minutes, _, rest = text.partition("min")
    seconds = re.search(r"\d+", rest)
    return int(minutes) * 60 + (int(seconds.group(0)) if seconds else 0)


def _nirs_time(df: pd.DataFrame, frequency_hz: float) -> tuple[np.ndarray, tuple[str, ...], str]:
    samples = np.arange(len(df)) / frequency_hz
    if "Event" not in df.columns or not df["Event"].notna().any():
        return samples, (), "unsynced: no event marker"
    rows = df.index[df["Event"].notna()]
    events = tuple(f"{df.at[r, 'Event']}@row{r}:{_event_time_text(df, r)}" for r in rows)
    # Dev rule (7 Oct): the latest '<m>min<s>' marker is the sync point; earlier ones can be aborted attempts.
    for number, row in reversed(list(enumerate(rows, start=1))):
        text = _event_time_text(df, row)
        try:
            event_s = _parse_event_seconds(text)
        except ValueError:
            logger.warning("NIRS event %d time %r not in '<m>min<s>' form; trying an earlier one", number, text)
            continue
        note = f"event {number} of {len(rows)} (latest parsable) = {event_s} s"
        return samples - row / frequency_hz + event_s, events, note
    return samples, events, "unsynced: no parsable event time"


def load_nirs(path: Path) -> NirsData:
    with _quiet():
        loader = NIRSLoader(path)
    df = loader.df.reset_index(drop=True)
    frequency_hz = float(loader.details.frequency)
    t, events, sync = _nirs_time(df, frequency_hz)
    signals = {name: _column(df, name) for name in (*NIRS_COLUMNS, TSI, FIT_FACTOR)}
    out = pd.DataFrame({"t": t, **signals})
    header_frequency = _cell(loader.complete_df, *NIRS_HEADER_FREQUENCY_CELL)
    n_all_zero_rows = int((out[list(NIRS_COLUMNS)] == 0).all(axis=1).sum())
    return NirsData(out, frequency_hz, header_frequency, events, sync, n_all_zero_rows)


def load_protocol_xml(path: Path) -> pd.DataFrame:
    """Ergometer protocol steps (Time, Load1) as a 1 Hz ideal work rate."""
    root = ElementTree.parse(path).getroot()
    steps = sorted(
        (float(step.findtext("Time")), float(step.findtext("Load1")))
        for step in root.iter("ErgoProtocolStep")
    )
    if not steps:
        raise ValueError(f"no ErgoProtocolStep elements in {path.name}")
    times = np.array([s[0] for s in steps])
    loads = np.array([s[1] for s in steps])
    t = np.arange(0, int(times[-1]))
    return pd.DataFrame({"t": t, "W": loads[np.searchsorted(times, t, side="right") - 1]})


def load_reference_wr(path: Path) -> pd.DataFrame:
    wr = pd.read_csv(path)["WR"].to_numpy(dtype=float)
    return pd.DataFrame({"t": np.arange(len(wr)), "W": wr})


def _fitdays_readers() -> tuple[Callable[[Path], pd.DataFrame], ...]:
    # Same order as Preparer._create_fit_files: the ".csv" files are often old-format Excel.
    return (
        lambda p: pd.read_excel(p, engine="xlrd"),
        pd.read_excel,
        lambda p: pd.read_csv(p, sep=None, engine="python", encoding="utf-8-sig"),
        lambda p: pd.read_csv(p, sep=";", encoding="latin1"),
    )


def load_fitdays(path: Path) -> pd.DataFrame:
    errors = []
    for read in _fitdays_readers():
        try:
            df = read(path)
        except Exception as exc:  # format sniffing: each failure is kept and reported if all fail
            errors.append(f"{type(exc).__name__}: {exc}")
            continue
        df = df.rename(columns=lambda c: str(c).strip().replace("﻿", ""))
        if FITDAYS_MARKER_COLUMNS & set(df.columns):
            return df
        errors.append(f"no known columns in {list(df.columns)[:5]}")
    raise ValueError("no reader understood the file: " + " | ".join(errors))


def ramp_end_s(cpet: pd.DataFrame) -> int:
    """Task failure: the last breath with cadence >= 60 rpm (as Preparer._truncate_ramp_df)."""
    if "RPM" not in cpet.columns:
        return int(cpet["t"].max())
    pedalling = cpet.loc[cpet["RPM"] >= RAMP_MIN_CADENCE_RPM, "t"]
    return int(pedalling.max()) if len(pedalling) else int(cpet["t"].max())


def cpet_phase_starts(cpet: pd.DataFrame) -> dict:
    if "Phase" not in cpet.columns:
        return {"warmup": math.nan, "exercise": math.nan}
    phase = cpet["Phase"].astype(str).str.strip().str.upper()

    def first_t(names: set[str]) -> float:
        hits = cpet.loc[phase.isin(names), "t"]
        return float(hits.iloc[0]) if len(hits) else math.nan

    return {"warmup": first_t(WARMUP_PHASES), "exercise": first_t(EXERCISE_PHASES)}


def _tymewear_lag(ref: VisitRef, annotation: Annotation) -> tuple[float, str]:
    if ref.cohort == "Lucas":
        return 0.0, "Lucas: Tymewear on CPET clock"
    if math.isnan(annotation.tymewear_lag_s):
        return 0.0, "no VE_lag annotated; unshifted"
    return annotation.tymewear_lag_s, f"shifted by VE_lag {annotation.tymewear_lag_s:g} s"


def ramp_protocol_wr(protocol_name: str, timeline: Timeline) -> pd.DataFrame:
    """Ideal ramp WR at 1 Hz from the CPET protocol name: rest 0 W, warm-up, then the ergometer's staircase.

    The first increment comes one step interval after the warm-up ends (assumed; not yet verified).
    """
    match = RAMP_RATE_IN_NAME.search(protocol_name)
    if match is None or int(match.group(1)) not in RAMP_STEP_W:
        raise ValueError(f"no known ramp rate in CPET protocol name {protocol_name!r}")
    if timeline.exercise_end_s is None:
        raise ValueError("ramp end unknown")
    rate_w_per_min = int(match.group(1))
    step_w = RAMP_STEP_W_BY_PROTOCOL.get(protocol_name.strip(), RAMP_STEP_W[rate_w_per_min])
    interval_s = step_w * SECONDS_PER_MINUTE / rate_w_per_min
    t = np.arange(0, int(timeline.exercise_end_s) + 1)
    steps = np.floor(np.maximum(t - timeline.warmup_end_s, 0) / interval_s)
    w = np.where(t < timeline.rest_end_s, 0.0, RAMP_WARMUP_W + step_w * steps)
    return pd.DataFrame({"t": t, "W": w})


def _load_protocol_wr(ref: VisitRef, cpet: Loaded, timeline: Timeline) -> Loaded:
    """Submax: the ergometer protocol xml. Ramp: no xml exists, so the CPET protocol name defines it."""
    if ref.protocol != "Ramp":
        return _attempt(f"{ref.visit_id} xml", choose_protocol_xml(ref.path, ref.protocol), load_protocol_xml)
    if not cpet.ok:
        return Loaded(FileChoice(None, (), "no CPET file to read the protocol name from"), None, None)
    name = str(cpet.data.header.get("protocol") or "")
    return _attempt(f"{ref.visit_id} ramp protocol", cpet.choice, lambda _: ramp_protocol_wr(name, timeline))


def load_visit(ref: VisitRef, annotation: Annotation, fitdays: Loaded) -> VisitData:
    cpet = _attempt(f"{ref.visit_id} cpet", choose_cpet_file(ref.path), load_cpet)
    lag_s, lag_note = _tymewear_lag(ref, annotation)
    tymewear = _attempt(
        f"{ref.visit_id} tymewear",
        choose_tymewear_file(ref.path),
        lambda p: load_tymewear(p, lag_s, lag_note),
    )
    nirs = _attempt(f"{ref.visit_id} nirs", choose_nirs_file(ref.path), load_nirs)
    reference_path = reference_prepared_file(ref, "WR")
    reference_choice = FileChoice(reference_path if reference_path.exists() else None, (), None)
    reference_wr = _attempt(f"{ref.visit_id} reference WR", reference_choice, load_reference_wr)

    timeline = NOMINAL_TIMELINES[(ref.cohort, ref.protocol)]
    phase_starts = {"warmup": math.nan, "exercise": math.nan}
    if cpet.ok:
        phase_starts = cpet_phase_starts(cpet.data.df)
        if timeline.exercise_end_s is None:
            timeline = replace(timeline, exercise_end_s=ramp_end_s(cpet.data.df))
    protocol = _load_protocol_wr(ref, cpet, timeline)
    return VisitData(ref, annotation, timeline, phase_starts, cpet, tymewear, nirs, protocol, reference_wr, fitdays)


def load_participant_fitdays(participant_dir: Path) -> Loaded:
    return _attempt(f"{participant_dir.name} fitdays", choose_fitdays_file(participant_dir), load_fitdays)
